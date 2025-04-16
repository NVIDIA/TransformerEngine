# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Attention."""
from contextlib import nullcontext
from importlib.metadata import version as get_pkg_version
from importlib.metadata import PackageNotFoundError
import math
import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import warnings
import logging

import numpy as np
from packaging.version import Version as PkgVersion

import torch

import transformer_engine_torch as tex
from transformer_engine.pytorch.utils import get_cudnn_version
from transformer_engine.pytorch.cpp_extensions.fused_attn import (
    fused_attn_fwd,
    fused_attn_bwd,
    FusedAttnBackend,
    META_QKV,
    META_O,
)
from transformer_engine.pytorch.fp8 import (
    get_fp8_te_dtype,
    get_fp8_torch_dtype,
)
from transformer_engine.pytorch.float8_tensor import Float8Tensor
from transformer_engine.pytorch.tensor._internal.float8_tensor_base import Float8TensorBase
from transformer_engine.pytorch.module.base import TransformerEngineBaseModule
from transformer_engine.pytorch.utils import (
    attention_mask_func,
    split_tensor_along_dim,
    get_device_compute_capability,
)
from transformer_engine.pytorch.constants import (
    AttnMaskTypes,
    AttnTypes,
    QKVLayouts,
    dist_group_type,
    TE_DType,
)
from transformer_engine.pytorch.softmax import FusedScaleMaskSoftmax
from transformer_engine.pytorch.distributed import (
    get_distributed_world_size,
    checkpoint,
    set_all_rng_states,
    CudaRNGStatesTracker,
    graph_safe_rng_available,
)
from transformer_engine.pytorch.jit import no_torch_dynamo
from transformer_engine.pytorch.graph import is_graph_capturing
from transformer_engine.pytorch.dot_product_attention.inference import InferenceParams
from transformer_engine.pytorch.tensor.quantized_tensor import (
    QuantizedTensor,
    prepare_for_saving,
    restore_from_saved,
)

# Import attention utils
import transformer_engine.pytorch.dot_product_attention.utils as dpa_utils
from transformer_engine.pytorch.dot_product_attention.utils import FlashAttentionUtils as fa_utils
from transformer_engine.pytorch.dot_product_attention.utils import AttentionLogging as attn_log
from transformer_engine.pytorch.dot_product_attention.rope import apply_rotary_pos_emb
from .cpu_offload import mark_activation_offload
from transformer_engine.pytorch.dot_product_attention.context_parallel import attn_forward_func_with_cp


# Setup Attention Logging
attn_log.setup_logging()

# Global vars for flash attn v2 and v3 imports
flash_attn_cuda_bwd = None
flash_attn_func = None
flash_attn_varlen_func = None
_flash_attn_fwd = None
_flash_attn_bwd = None
_flash_attn_varlen_fwd = None
_flash_attn_varlen_bwd = None
try:
    fa_utils.version = PkgVersion(get_pkg_version("flash-attn"))
except PackageNotFoundError:
    pass  # only print warning if use_flash_attention_2 = True in get_attention_backend
else:
    if torch.cuda.is_available() and get_device_compute_capability() >= (10, 0):
        if fa_utils.version_required_blackwell <= fa_utils.version <= fa_utils.max_version:
            fa_utils.is_installed = True
    elif fa_utils.version_required <= fa_utils.version <= fa_utils.max_version:
        fa_utils.is_installed = True

    if fa_utils.is_installed:
        from flash_attn_2_cuda import varlen_bwd as flash_attn_cuda_bwd
        from flash_attn.flash_attn_interface import flash_attn_func, flash_attn_varlen_func
        from flash_attn.flash_attn_interface import _flash_attn_forward as _flash_attn_fwd
        from flash_attn.flash_attn_interface import _flash_attn_backward as _flash_attn_bwd
        from flash_attn.flash_attn_interface import (
            _flash_attn_varlen_forward as _flash_attn_varlen_fwd,
        )
        from flash_attn.flash_attn_interface import (
            _flash_attn_varlen_backward as _flash_attn_varlen_bwd,
        )

        # Setup Flash attention utils
        fa_utils.set_flash_attention_version()
    elif (
        torch.cuda.is_available()
        and get_device_compute_capability() >= (8, 0)
        and dpa_utils._NVTE_FLASH_ATTN
    ):
        attn_log.fa_logger.warning(
            "Supported flash-attn versions are %s. Found flash-attn %s.",
            dpa_utils._get_supported_versions(
                (
                    fa_utils.version_required
                    if get_device_compute_capability() < (10, 0)
                    else fa_utils.version_required_blackwell
                ),
                fa_utils.max_version,
            ),
            fa_utils.version,
        )
try:
    fa_utils.fa3_version = PkgVersion(get_pkg_version("flash-attn-3"))
except PackageNotFoundError:
    pass  # only print warning if use_flash_attention_3 = True in get_attention_backend
else:
    from flash_attn_3.flash_attn_interface import flash_attn_func as flash_attn_func_v3
    from flash_attn_3.flash_attn_interface import (
        flash_attn_varlen_func as flash_attn_varlen_func_v3,
    )
    from flash_attn_3.flash_attn_interface import (
        flash_attn_with_kvcache as flash_attn_with_kvcache_v3,
    )
    from flash_attn_3.flash_attn_interface import _flash_attn_forward as _flash_attn_fwd_v3
    from flash_attn_3.flash_attn_interface import _flash_attn_backward as _flash_attn_bwd_v3

    fa_utils.set_flash_attention_3_params()

# Global vars for available attention backends and ALiBi cache
_attention_backends = {
    "attention_params": None,
    "use_flash_attention": None,
    "flash_attention_backend": None,
    "use_fused_attention": None,
    "fused_attention_backend": None,
    "use_unfused_attention": None,
    "backend_selection_requires_update": False,
}

_alibi_cache = {
    "_num_heads": None,
    "_alibi_slopes": None,
    "_max_seqlen_q": None,
    "_max_seqlen_kv": None,
    "_bottom_right_alignment": True,
    "_alibi_bias": None,
    "_alibi_slopes_require_update": False,
    "_alibi_bias_require_update": False,
}

__all__ = ["DotProductAttention"]


def maybe_contiguous(tensor: torch.Tensor) -> torch.Tensor:
    """Make tensor contiguous if final stride is not 1."""
    return tensor.contiguous() if tensor.stride(-1) != 1 else tensor


class _SplitAlongDim(torch.autograd.Function):
    """"""

    @staticmethod
    def forward(
        ctx,
        mixed_x_layer: torch.Tensor,
        split_dim: int,
        split_size_or_sections: Union[int, List[int], Tuple[int]],
        squeeze=False,
    ) -> Tuple[torch.Tensor, ...]:
        # pylint: disable=missing-function-docstring
        ctx.split_dim = split_dim
        ctx.split_size_or_sections = split_size_or_sections
        if isinstance(mixed_x_layer, Float8TensorBase) and not isinstance(
            mixed_x_layer, Float8Tensor
        ):
            return tuple(
                Float8TensorBase(
                    fp8_scale_inv=mixed_x_layer._scale_inv,
                    fp8_dtype=mixed_x_layer._fp8_dtype,
                    data=x.squeeze(split_dim) if squeeze else x,
                    shape=x.squeeze(split_dim).shape if squeeze else x.shape,
                    quantizer=mixed_x_layer._quantizer,
                )
                for x in torch.split(
                    mixed_x_layer._data,
                    split_size_or_sections=split_size_or_sections,
                    dim=split_dim,
                )
            )
        if isinstance(mixed_x_layer, Float8Tensor):
            return tuple(
                Float8Tensor.make_like(
                    mixed_x_layer,
                    data=x.squeeze(split_dim) if squeeze else x,
                    shape=x.squeeze(split_dim).shape if squeeze else x.shape,
                )
                for x in torch.split(
                    mixed_x_layer._data,
                    split_size_or_sections=split_size_or_sections,
                    dim=split_dim,
                )
            )
        out_list = torch.split(mixed_x_layer, split_size_or_sections, dim=split_dim)
        if squeeze:
            out_list = [x.squeeze(split_dim) for x in out_list]
        return out_list

    @staticmethod
    def backward(ctx, *grad_outputs):
        # pylint: disable=missing-function-docstring
        assert len(grad_outputs) > 0, "No gradients received for backprop!"

        if isinstance(ctx.split_size_or_sections, (list, tuple)):
            split_sizes = ctx.split_size_or_sections
            assert len(grad_outputs) == len(
                split_sizes
            ), "Unequal number of gradients vs split sections for backprop!"
        if isinstance(ctx.split_size_or_sections, int):
            split_sizes = [ctx.split_size_or_sections] * len(grad_outputs)
        dims = len(grad_outputs[0].shape)
        split_dim = (ctx.split_dim + dims) % dims

        if isinstance(grad_outputs[0], Float8Tensor):
            noop_ok = True
            strides = grad_outputs[0].stride()
            data_ptr = grad_outputs[0]._data.untyped_storage().data_ptr()
            shape = list(grad_outputs[0].shape)
            for i, tensor in enumerate(grad_outputs):
                shape_i = shape
                shape_i[split_dim] = split_sizes[i]
                offset_size = sum(split_sizes[:i]) * np.prod(shape[split_dim + 1 :])
                if (
                    tensor.stride() != strides
                    or list(tensor.shape) != shape_i
                    or tensor._data.untyped_storage().data_ptr() != data_ptr
                    or tensor.storage_offset() != offset_size
                ):
                    noop_ok = False
                    break
            if noop_ok:
                ret = torch.Tensor().to(
                    device=grad_outputs[0].device, dtype=grad_outputs[0]._data.dtype
                )
                new_shape = list(shape)
                new_shape[split_dim] = sum(split_sizes)
                ret.set_(
                    grad_outputs[0]._data.untyped_storage(),
                    grad_outputs[0]._data.storage_offset(),
                    new_shape,
                    strides,
                )
                return (
                    Float8Tensor.make_like(grad_outputs[0], data=ret, shape=ret.shape),
                    None,
                    None,
                )

            grad_outputs_data = [x._data for x in grad_outputs]
            data = torch.cat(grad_outputs_data, dim=split_dim)
            return (
                Float8Tensor.make_like(grad_outputs[0], data=data, shape=data.shape),
                None,
                None,
                None,
            )
        noop_ok = True
        strides = grad_outputs[0].stride()
        data_ptr = grad_outputs[0].untyped_storage().data_ptr()
        shape = list(grad_outputs[0].shape)
        for i, tensor in enumerate(grad_outputs):
            shape_i = shape
            shape_i[split_dim] = split_sizes[i]
            offset_size = sum(split_sizes[:i]) * np.prod(shape[split_dim + 1 :])
            if (
                tensor.stride() != strides
                or list(tensor.shape) != shape_i
                or tensor.untyped_storage().data_ptr() != data_ptr
                or tensor.storage_offset() != offset_size
            ):
                noop_ok = False
                break
        if noop_ok:
            ret = torch.Tensor().to(device=grad_outputs[0].device, dtype=grad_outputs[0].dtype)
            new_shape = list(shape)
            new_shape[split_dim] = sum(split_sizes)
            ret.set_(
                grad_outputs[0].untyped_storage(),
                grad_outputs[0].storage_offset(),
                new_shape,
                strides,
            )
            return ret, None, None

        return torch.cat(grad_outputs, dim=split_dim), None, None


class UnfusedDotProductAttention(torch.nn.Module):
    """Parallel attention w/o QKV and Proj Gemms
    BMM1 -> softmax + dropout -> BMM2
    """

    def __init__(
        self,
        softmax_scale: float,
        attention_type: str = "self",
        attention_dropout: float = 0.0,
        attention_dropout_ctx: Optional[Callable] = nullcontext,
        layer_number: Optional[int] = None,
    ) -> None:
        super().__init__()

        self.softmax_scale = softmax_scale
        self.attention_type = attention_type
        self.attention_dropout_ctx = attention_dropout_ctx
        self.layer_number = layer_number

        self.scale_mask_softmax = FusedScaleMaskSoftmax(attention_mask_func)

        # Dropout. Note that for a single iteration, this layer will generate
        # different outputs on different number of parallel partitions but
        # on average it should not be partition dependent.
        self.attention_dropout = torch.nn.Dropout(attention_dropout)

        # An FP16 training trick required for certain GPT-like models.
        self.apply_qk_layer_scaling = (
            bool(int(os.getenv("NVTE_APPLY_QK_LAYER_SCALING", "0"))) and layer_number is not None
        )

    def forward(
        self,
        query_layer: torch.Tensor,
        key_layer: torch.Tensor,
        value_layer: torch.Tensor,
        qkv_layout: str = "sbh3d",
        cu_seqlens_q: Optional[torch.Tensor] = None,  # pylint: disable=unused-argument
        cu_seqlens_kv: Optional[torch.Tensor] = None,  # pylint: disable=unused-argument
        attn_mask_type: str = "causal",
        attention_mask: Optional[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]] = None,
        window_size: Optional[Tuple[int, int]] = None,
        core_attention_bias_type: str = "no_bias",
        core_attention_bias: Optional[torch.Tensor] = None,
        alibi_slopes: Optional[torch.Tensor] = None,
        inference_params: Optional[InferenceParams] = None,
    ) -> torch.Tensor:
        """Unfused attention fprop"""
        assert (
            qkv_layout in QKVLayouts
        ), f"UnfusedDotProductAttention does not support qkv_layout = {qkv_layout}!"

        # get q_format and kv_format for training and inference
        qkv_format, q_format, _ = dpa_utils.get_qkv_format(qkv_layout, inference_params)
        if inference_params is not None and inference_params.is_paged:
            key_layer, value_layer = inference_params.convert_paged_to_nonpaged(self.layer_number)

        if qkv_format == "bshd":
            # convert to sbhd and use sbhd implementation for now
            query_layer, key_layer, value_layer = [
                x.transpose(0, 1) for x in [query_layer, key_layer, value_layer]
            ]
        if qkv_format == "sbhd_2bshd":
            key_layer, value_layer = [x.transpose(0, 1) for x in [key_layer, value_layer]]

        total_tokens, batch_size = None, None
        if qkv_format == "thd_2bshd":
            total_tokens, batch_size = query_layer.shape[0], key_layer.shape[0]
            query_layer = tex.convert_thd_to_bshd(
                query_layer,
                cu_seqlens_q,
                batch_size,
                inference_params.max_ctx_len,
            )
            query_layer, key_layer, value_layer = [
                x.transpose(0, 1) for x in [query_layer, key_layer, value_layer]
            ]
        batch_size, max_seqlen_q, max_seqlen_kv = (
            query_layer.shape[1],
            query_layer.shape[0],
            key_layer.shape[0],
        )

        if "padding" in attn_mask_type and attention_mask is None:
            attention_mask = dpa_utils.get_padding_mask(
                batch_size, cu_seqlens_q, cu_seqlens_kv, max_seqlen_q, max_seqlen_kv
            )
        attn_mask_type, attention_mask, actual_seqlens_q, actual_seqlens_kv = (
            dpa_utils.get_full_mask(
                max_seqlen_q,
                max_seqlen_kv,
                attn_mask_type=attn_mask_type,
                attention_mask=attention_mask,
                window_size=window_size,
                attention_type=self.attention_type,
            )
        )

        batch_size, seqlen = query_layer.shape[1], query_layer.shape[0]
        apply_qk_layer_scaling = self.apply_qk_layer_scaling and key_layer.dtype == torch.float16

        # [b, np, sq, sk]
        output_size = (
            query_layer.size(1),
            query_layer.size(2),
            query_layer.size(0),
            key_layer.size(0),
        )

        if key_layer.shape[2] != query_layer.shape[2]:
            assert (
                query_layer.shape[2] % key_layer.shape[2] == 0
            ), "The number of attention heads must be divisible by the number of GQA groups!"
            key_layer = key_layer.repeat_interleave(
                int(query_layer.shape[2] / key_layer.shape[2]), dim=2
            )
            value_layer = value_layer.repeat_interleave(
                int(query_layer.shape[2] / value_layer.shape[2]), dim=2
            )

        # [sq, b, np, hn] -> [sq, b * np, hn]
        query_layer = query_layer.reshape(output_size[2], output_size[0] * output_size[1], -1)
        # [sk, b, np, hn] -> [sk, b * np, hn]
        key_layer = key_layer.reshape(output_size[3], output_size[0] * output_size[1], -1)

        # preallocting result tensor: [b * np, sq, sk]
        matmul_result = torch.empty(
            output_size[0] * output_size[1],
            output_size[2],
            output_size[3],
            dtype=query_layer.dtype,
            device=torch.cuda.current_device(),
        )

        scale = self.softmax_scale
        if apply_qk_layer_scaling:
            scale /= self.layer_number

        # Raw attention scores. [b * np, sq, sk]
        if core_attention_bias_type == "no_bias":
            matmul_result = torch.baddbmm(
                matmul_result,
                query_layer.transpose(0, 1),  # [b * np, sq, hn]
                key_layer.transpose(0, 1).transpose(1, 2),  # [b * np, hn, sk]
                beta=0.0,
                alpha=scale,
            ).view(*output_size)

        elif core_attention_bias_type == "pre_scale_bias":
            assert core_attention_bias is not None, "core_attention_bias should not be None!"
            matmul_result = torch.bmm(
                query_layer.transpose(0, 1),  # [b * np, sq, hn]
                key_layer.transpose(0, 1).transpose(1, 2),  # [b * np, hn, sk]
            )
            matmul_result = matmul_result.view(*output_size) + core_attention_bias
            matmul_result *= scale

        elif core_attention_bias_type in ["post_scale_bias", "alibi"]:
            if core_attention_bias_type == "post_scale_bias":
                assert core_attention_bias is not None, "core_attention_bias should not be None!"
            if core_attention_bias_type == "alibi":
                _, core_attention_bias = dpa_utils.get_alibi(
                    _alibi_cache,
                    output_size[1],
                    output_size[2],
                    output_size[3],
                    actual_seqlens_q=actual_seqlens_q if "padding" in attn_mask_type else None,
                    actual_seqlens_kv=actual_seqlens_kv if "padding" in attn_mask_type else None,
                    alibi_slopes=alibi_slopes,
                    bottom_right_alignment=attn_mask_type not in ["causal", "padding_causal"],
                )
            matmul_result = torch.baddbmm(
                matmul_result,
                query_layer.transpose(0, 1),  # [b * np, sq, hn]
                key_layer.transpose(0, 1).transpose(1, 2),  # [b * np, hn, sk]
                beta=0.0,
                alpha=scale,
            )
            matmul_result = (matmul_result.view(*output_size) + core_attention_bias).to(
                dtype=query_layer.dtype
            )

        # attention scores and attention mask [b, np, sq, sk]
        softmax_scale = self.layer_number if apply_qk_layer_scaling else None
        attention_probs = self.scale_mask_softmax(
            matmul_result, attention_mask, attn_mask_type, softmax_scale
        )

        # mask out the pad positions in softmax results, mostly for the rows (pad tokens from q)
        # the columns (pad tokens from k) are already zeroed out during softmax
        if "padding" in attn_mask_type:
            attention_probs = attention_probs.masked_fill(attention_mask, 0)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        with self.attention_dropout_ctx():
            attention_probs = self.attention_dropout(attention_probs)

        # value_layer -> context layer.
        # [sk, b, np, hn] --> [b, np, sq, hn]
        output_size = (
            value_layer.size(1),
            value_layer.size(2),
            query_layer.size(0),
            value_layer.size(3),
        )

        # change view [sk, b * np, hn]
        value_layer = value_layer.reshape(value_layer.size(0), output_size[0] * output_size[1], -1)

        # change view [b * np, sq, sk]
        attention_probs = attention_probs.view(output_size[0] * output_size[1], output_size[2], -1)

        # matmul: [b * np, sq, hn]
        context_layer = torch.bmm(attention_probs, value_layer.transpose(0, 1))

        # change view [b, np, sq, hn]
        context_layer = context_layer.view(*output_size)

        if q_format == "sbhd":
            # [b, np, sq, hn] --> [sq, b, np, hn]
            context_layer = context_layer.permute(2, 0, 1, 3).contiguous()

            # [sq, b, np, hn] --> [sq, b, hp]
            context_layer = context_layer.view(seqlen, batch_size, -1)

        if q_format == "bshd":
            # [b, np, sq, hn] --> [b, sq, np, hn]
            context_layer = context_layer.permute(0, 2, 1, 3).contiguous()

            # [b, sq, np, hn] --> [b, sq, hp]
            context_layer = context_layer.view(batch_size, seqlen, -1)

        if q_format == "thd":
            # [b, np, sq, hn] --> [b, sq, np, hn]
            context_layer = context_layer.permute(0, 2, 1, 3).contiguous()

            # [b, sq, np, hn] --> [tq, np, hn]
            context_layer = tex.convert_bshd_to_thd(
                context_layer,
                cu_seqlens_q,
                total_tokens,
            )

            # [tq, np, hn] --> [tq, hp]
            context_layer = context_layer.view(total_tokens, -1)

        return context_layer


class _PrepareQKVForFA(torch.autograd.Function):
    """This class converts QKV from interleaved (s, b, ...) layout
    to separate contiguous q, k, v tensors in (b, s, ...) layout."""

    @staticmethod
    def forward(
        _ctx: torch.autograd.function.FunctionCtx,  # unused
        query_layer: torch.Tensor,
        key_layer: torch.Tensor,
        value_layer: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # pylint: disable=missing-function-docstring
        # All inputs received are non-contiguous tensors.
        # The `query_layer` tensor is used to access the
        # full memory region of the QKV tensor.
        qkv = tex.fa_prepare_fwd(query_layer)
        q, k, v = split_tensor_along_dim(qkv, 0, 3)
        query_layer = torch.squeeze(q, 0)
        key_layer = torch.squeeze(k, 0)
        value_layer = torch.squeeze(v, 0)
        return query_layer, key_layer, value_layer

    @staticmethod
    def backward(
        _ctx: torch.autograd.function.FunctionCtx,  # unused
        dq: torch.Tensor,
        dk: torch.Tensor,
        dv: torch.Tensor,
    ) -> Tuple[Union[torch.Tensor, None], ...]:
        # pylint: disable=missing-function-docstring
        dqkv = tex.fa_prepare_bwd(dq, dk, dv)
        dq, dk, dv = split_tensor_along_dim(dqkv, -1, 3)
        return dq, dk, dv


class FlashAttention(torch.nn.Module):
    """Dot product attention, using HazyResearch flash-attn package:
    https://github.com/Dao-AILab/flash-attention
    """

    def __init__(
        self,
        softmax_scale: float,
        attention_dropout: float = 0.0,
        attention_dropout_ctx: Optional[Callable] = nullcontext,
        attention_type: str = "self",
        layer_number: Optional[int] = None,
        deterministic: bool = False,
    ) -> None:
        super().__init__()

        if fa_utils.is_installed:
            assert (
                fa_utils.version >= fa_utils.version_required
            ), f"FlashAttention minimum version {fa_utils.version_required} is required."
            assert (
                fa_utils.version <= fa_utils.max_version
            ), f"FlashAttention maximum version {fa_utils.max_version} is supported."

        self.softmax_scale = softmax_scale
        self.attention_dropout_ctx = attention_dropout_ctx
        self.attention_dropout = attention_dropout
        self.attention_type = attention_type
        self.layer_number = 1 if layer_number is None else layer_number
        self.deterministic = deterministic
        self.logger = logging.getLogger("FlashAttention")
        self.logger.setLevel(attn_log._log_level)
        if not self.logger.hasHandlers():
            self.logger.addHandler(attn_log._stream_handler)

    def forward(
        self,
        query_layer: torch.Tensor,
        key_layer: torch.Tensor,
        value_layer: torch.Tensor,
        attention_mask: Optional[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]] = None,
        qkv_layout: str = "sbh3d",
        cu_seqlens_q: Optional[torch.Tensor] = None,
        cu_seqlens_kv: Optional[torch.Tensor] = None,
        max_seqlen_q: Optional[int] = None,
        max_seqlen_kv: Optional[int] = None,
        attn_mask_type: str = "causal",
        window_size: Optional[Tuple[int, int]] = None,
        alibi_slopes: Optional[torch.Tensor] = None,
        cp_group: Optional[Union[dist_group_type, List[dist_group_type]]] = None,
        cp_global_ranks: List[int] = None,
        cp_stream: torch.cuda.Stream = None,
        cp_comm_type: str = "p2p",
        fp8: bool = False,
        fp8_meta: Optional[Dict[str, Any]] = None,
        quantizers=None,
        inference_params: Optional[InferenceParams] = None,
        flash_attention_backend: Optional[PkgVersion] = PkgVersion("0"),
    ) -> torch.Tensor:
        """flash-attn fprop"""

        assert all(
            x.dtype in [torch.float16, torch.bfloat16] or isinstance(x, Float8Tensor)
            for x in [query_layer, key_layer, value_layer]
        ), "FlashAttention only supports FP16 and BF16 data types, or Float8Tensors."
        assert (
            query_layer.is_cuda and key_layer.is_cuda and value_layer.is_cuda
        ), "FlashAttention currently only supports CUDA tensors."
        assert (
            qkv_layout in QKVLayouts
        ), f"FlashAttention does not support qkv_layout = {qkv_layout}!"

        cp_size = 1
        if isinstance(cp_group, dist_group_type):
            cp_size = get_distributed_world_size(cp_group)
        elif isinstance(cp_group, list):
            for group in cp_group:
                cp_size *= get_distributed_world_size(group)
        context_parallel = cp_size > 1

        # get q_format and kv_format for training and inference
        qkv_format, q_format, kv_format = dpa_utils.get_qkv_format(qkv_layout, inference_params)

        # convert q, k, v to bshd if they are in sbhd; qkv_format doesn't change
        if all(not isinstance(x, Float8Tensor) for x in [query_layer, key_layer, value_layer]):
            if qkv_format == "sbhd":
                # For now just 128, will make it more general in the future
                if (
                    query_layer.shape[-1] == 128
                    and query_layer.shape[0] * query_layer.shape[1] >= 512
                    and qkv_layout == "sbh3d"
                ):
                    query_layer, key_layer, value_layer = _PrepareQKVForFA.apply(
                        query_layer, key_layer, value_layer
                    )
                else:
                    query_layer, key_layer, value_layer = [
                        x.transpose(0, 1).contiguous()
                        for x in (query_layer, key_layer, value_layer)
                    ]
            elif q_format == "sbhd" and kv_format == "bshd":
                query_layer = query_layer.transpose(0, 1).contiguous()
            if context_parallel:
                query_layer, key_layer, value_layer = [
                    x.contiguous() for x in (query_layer, key_layer, value_layer)
                ]
        else:
            if qkv_format == "sbhd":
                query_layer._data, key_layer._data, value_layer._data = [
                    x.transpose(0, 1).contiguous()
                    for x in (query_layer._data, key_layer._data, value_layer._data)
                ]
                query_layer, key_layer, value_layer = [
                    Float8Tensor.make_like(x, data=x._data, shape=x._data.shape)
                    for x in (query_layer, key_layer, value_layer)
                ]
            elif q_format == "sbhd" and kv_format == "bshd":
                query_layer._data = query_layer._data.transpose(0, 1).contiguous()
                query_layer = Float8Tensor.make_like(
                    query_layer, data=query_layer._data, shape=query_layer._data.shape
                )
            if context_parallel:
                query_layer._data, key_layer._data, value_layer._data = [
                    x.contiguous() for x in (query_layer._data, key_layer._data, value_layer._data)
                ]

        # get batch_size, max_seqlen and cu_seqlens
        batch_size, context_len = None, None
        if inference_params is None:
            if qkv_format in ["sbhd", "bshd"]:
                batch_size = query_layer.shape[0]
                max_seqlen_q, max_seqlen_kv = query_layer.shape[1], key_layer.shape[1]
                max_seqlen_q *= cp_size
                max_seqlen_kv *= cp_size

                if "padding" in attn_mask_type:
                    assert (
                        not context_parallel
                    ), "Padding mask not supported with context parallelism!"

                    # [b * s, h, d]
                    query_layer, key_layer, value_layer = [
                        x.reshape(x.shape[0] * x.shape[1], *x.shape[2:])
                        for x in [query_layer, key_layer, value_layer]
                    ]

                    if self.attention_type == "self":
                        assert (
                            max_seqlen_q == max_seqlen_kv
                        ), "Maximum sequence length for Q and KV should be the same."
                        if cu_seqlens_q is None:
                            assert (
                                attention_mask is not None
                            ), "Please provide attention_mask for padding!"
                            cu_seqlens_q, indices_q = dpa_utils.get_cu_seqlens_and_indices(
                                attention_mask
                            )
                        else:
                            indices_q = dpa_utils.get_indices(max_seqlen_q, cu_seqlens_q)
                        cu_seqlens_kv = cu_seqlens_q
                        query_layer, key_layer, value_layer = dpa_utils.PackTensors.apply(
                            indices_q, query_layer, key_layer, value_layer
                        )
                    else:
                        if cu_seqlens_q is None or cu_seqlens_kv is None:
                            assert (
                                attention_mask is not None
                            ), "Please provide attention_mask for padding!"
                            cu_seqlens_q, indices_q = dpa_utils.get_cu_seqlens_and_indices(
                                attention_mask[0]
                            )
                            cu_seqlens_kv, indices_kv = dpa_utils.get_cu_seqlens_and_indices(
                                attention_mask[1]
                            )
                        else:
                            indices_q = dpa_utils.get_indices(max_seqlen_q, cu_seqlens_q)
                            indices_kv = dpa_utils.get_indices(max_seqlen_kv, cu_seqlens_kv)
                        query_layer = dpa_utils.PackTensors.apply(indices_q, query_layer)
                        key_layer, value_layer = dpa_utils.PackTensors.apply(
                            indices_kv, key_layer, value_layer
                        )
                else:
                    # Cumulative sequence lengths for unpadded data
                    if cu_seqlens_q is None:
                        cu_seqlens_q = dpa_utils.get_full_cu_seqlens(
                            batch_size,
                            max_seqlen_q,
                            query_layer.device,
                        )
                    if cu_seqlens_kv is None:
                        cu_seqlens_kv = dpa_utils.get_full_cu_seqlens(
                            batch_size,
                            max_seqlen_kv,
                            key_layer.device,
                        )
            elif qkv_format == "thd":
                assert (
                    cu_seqlens_q is not None and cu_seqlens_kv is not None
                ), "cu_seqlens_q and cu_seqlens_kv can not be None when qkv_format = thd!"
                if max_seqlen_q is None:
                    seqlens_q = cu_seqlens_q[1:] - cu_seqlens_q[:-1]
                    max_seqlen_q = seqlens_q.max().item()
                if max_seqlen_kv is None:
                    seqlens_kv = cu_seqlens_kv[1:] - cu_seqlens_kv[:-1]
                    max_seqlen_kv = seqlens_kv.max().item()
        else:
            if qkv_format in ["sbhd_2bshd", "bshd"]:
                # q is in bshd in both cases from conversion above or the original input
                batch_size, context_len = query_layer.shape[:2]
                cu_seqlens_q = cu_seqlens_q[: batch_size + 1]
                cu_seqlens_kv = cu_seqlens_kv[: batch_size + 1]
                # convert from bshd to thd_2bshd for flash_attn_varlen_func/_with_kvcache;
                # kernel assumes tensor is contiguous
                if isinstance(query_layer, Float8Tensor):
                    query_layer._data = tex.convert_bshd_to_thd(
                        query_layer._data,
                        cu_seqlens_q,
                        batch_size * context_len,
                    )
                    query_layer = Float8Tensor.make_like(
                        query_layer, data=query_layer._data, shape=query_layer._data.shape
                    )
                else:
                    query_layer = tex.convert_bshd_to_thd(
                        query_layer,
                        cu_seqlens_q,
                        batch_size * context_len,
                    )

        use_flash_attn_3 = False
        if flash_attention_backend is not None and flash_attention_backend > PkgVersion("3.0.0b"):
            use_flash_attn_3 = True
        if context_parallel and all(
            not isinstance(x, Float8Tensor) for x in [query_layer, key_layer, value_layer]
        ):
            assert (
                alibi_slopes is None
            ), "Alibi slope bias addition is not supported with context parallelism."
            with self.attention_dropout_ctx():
                output = attn_forward_func_with_cp(
                    self.training,
                    query_layer,
                    key_layer,
                    value_layer,
                    cu_seqlens_q,
                    cu_seqlens_kv,
                    max_seqlen_q,
                    max_seqlen_kv,
                    cu_seqlens_q if qkv_format == "thd" else None,
                    cu_seqlens_kv if qkv_format == "thd" else None,
                    self.attention_dropout if self.training else 0.0,
                    cp_group,
                    cp_global_ranks,
                    cp_stream,
                    cp_comm_type,
                    softmax_scale=self.softmax_scale,
                    qkv_format="bshd" if qkv_format == "sbhd" else qkv_format,
                    attn_mask_type=attn_mask_type,
                    deterministic=self.deterministic,
                    window_size=window_size,
                    quantizers=quantizers,
                    pad_between_seqs=False,
                    use_flash_attn_3=use_flash_attn_3,
                )
        else:

            from .cpu_offload import CPUOffloadEnabled

            if CPUOffloadEnabled:
                mark_activation_offload(
                    query_layer, key_layer, value_layer, cu_seqlens_q, cu_seqlens_kv
                )

            with self.attention_dropout_ctx():
                #       | API                     | use cases
                # ----------------------------------------------------------------------
                # FA v2 | flash_attn_func         | bshd/sbhd + not padding
                #       | flash_attn_varlen_func  | bshd/sbhd + padding
                #       |                         | thd + padding
                #       |                         | KV cache (not-paged/paged), i.e.
                #       |                         |     bshd/sbhd/thd + padding
                # FA v3 | flash_attn_func         | bshd/sbhd + not padding
                #       | flash_attn_varlen_func  | bshd/sbhd + padding
                #       |                         | thd + padding
                #       | flash_attn_with_kvcache | KV cache (not-paged/paged), i.e.
                #       |                         |     bshd/sbhd/thd + padding
                fa_optional_forward_args_thd = []
                if qkv_format in ["bshd", "sbhd"] and "padding" not in attn_mask_type:
                    func = (
                        flash_attn_func if not use_flash_attn_3 else flash_attn_func_v3
                    )  # pylint: disable=possibly-used-before-assignment
                else:
                    if not use_flash_attn_3:
                        func = flash_attn_varlen_func
                    elif inference_params is None:
                        func = flash_attn_varlen_func_v3  # pylint: disable=possibly-used-before-assignment
                    else:
                        func = flash_attn_with_kvcache_v3  # pylint: disable=possibly-used-before-assignment
                    if not use_flash_attn_3 or inference_params is None:
                        fa_optional_forward_args_thd.append(cu_seqlens_q)
                        fa_optional_forward_args_thd.append(cu_seqlens_kv)
                        fa_optional_forward_args_thd.append(max_seqlen_q)
                        fa_optional_forward_args_thd.append(max_seqlen_kv)
                if not use_flash_attn_3:
                    fa_optional_forward_kwargs = {}
                    if fa_utils.v2_3_plus:
                        fa_optional_forward_kwargs["window_size"] = window_size
                    if fa_utils.v2_4_plus:
                        fa_optional_forward_kwargs["alibi_slopes"] = alibi_slopes
                    if fa_utils.v2_4_1_plus:
                        fa_optional_forward_kwargs["deterministic"] = self.deterministic
                    if inference_params is not None:
                        # use block_table kwarg to support thd_2bshd for non-paged
                        fa_optional_forward_kwargs["block_table"] = (
                            inference_params.cache_manager.page_table[:batch_size]
                            if inference_params.is_paged
                            else inference_params.cache_manager.batch_indices_post_step.unsqueeze(
                                1
                            )[:batch_size]
                        )
                    output = func(
                        query_layer,
                        key_layer,
                        value_layer,
                        *fa_optional_forward_args_thd,
                        self.attention_dropout if self.training else 0.0,
                        softmax_scale=self.softmax_scale,
                        causal="causal" in attn_mask_type,
                        **fa_optional_forward_kwargs,
                    )
                else:
                    fa_3_optional_forward_kwargs = {}
                    fa_3_optional_forward_kwargs["window_size"] = window_size
                    if inference_params is None:
                        fa_3_optional_forward_kwargs["deterministic"] = self.deterministic
                    else:
                        fa_3_optional_forward_kwargs["cu_seqlens_q"] = cu_seqlens_q
                        fa_3_optional_forward_kwargs["max_seqlen_q"] = max_seqlen_q
                        cache_seqlens = cu_seqlens_kv[1:] - cu_seqlens_kv[:-1]
                        fa_3_optional_forward_kwargs["cache_seqlens"] = cache_seqlens
                        # flash_attn_with_kvcache accepts thd_2bshd for non-paged
                        if inference_params.is_paged:
                            fa_3_optional_forward_kwargs["page_table"] = (
                                inference_params.cache_manager.page_table[:batch_size]
                            )
                    if fp8:
                        QKV_quantizer = quantizers["scaling_fwd"][META_QKV]
                        torch_dtype = get_fp8_torch_dtype(fp8_meta["recipe"], fprop_tensor=True)
                        torch_orig_dtype = query_layer.dtype

                        def convert_to_torch_float8(tensor, dtype):
                            out = torch.Tensor().to(device=tensor.device, dtype=dtype)
                            out.set_(
                                tensor._data.untyped_storage(),
                                tensor._data.storage_offset(),
                                tensor._data.shape,
                                tensor._data.stride(),
                            )
                            return out

                        # "fp8_mha" decides outputs in fp8, while inputs are inferred from
                        # the real dtype
                        assert isinstance(key_layer, query_layer.__class__) and isinstance(
                            value_layer, query_layer.__class__
                        ), "q, k, and v must have the same type."
                        if not isinstance(query_layer, Float8Tensor):
                            query_layer, key_layer, value_layer = (
                                QKV_quantizer(x) for x in [query_layer, key_layer, value_layer]
                            )
                        batch_size = cu_seqlens_q.shape[0] - 1
                        num_heads_k = key_layer.shape[-2]
                        fa_3_optional_forward_kwargs["q_descale"] = (
                            query_layer._scale_inv.unsqueeze(0).repeat(batch_size, num_heads_k)
                        )
                        fa_3_optional_forward_kwargs["k_descale"] = key_layer._scale_inv.unsqueeze(
                            0
                        ).repeat(batch_size, num_heads_k)
                        fa_3_optional_forward_kwargs["v_descale"] = (
                            value_layer._scale_inv.unsqueeze(0).repeat(batch_size, num_heads_k)
                        )
                        query_layer, key_layer, value_layer = (
                            convert_to_torch_float8(x, torch_dtype)
                            for x in [query_layer, key_layer, value_layer]
                        )
                    try:
                        output = func(
                            query_layer,
                            key_layer,
                            value_layer,
                            *fa_optional_forward_args_thd,
                            softmax_scale=self.softmax_scale,
                            causal="causal" in attn_mask_type,
                            **fa_3_optional_forward_kwargs,
                        )
                        if isinstance(output, (List, Tuple)):
                            output = output[0]
                    except TypeError as e:
                        if fa_utils.v3_0_0_beta:
                            e.args = (
                                e.args[0]
                                + ". Please update your flash-attn v3 (beta) installation as it "
                                + "may have added more supported arguments to its API. \n"
                                + fa_utils.v3_installation_steps,
                            ) + e.args[1:]
                        raise

                    if fp8:
                        output = output.to(dtype=torch_orig_dtype)
                    if fp8 and fp8_meta["recipe"].fp8_mha:
                        O_quantizer = quantizers["scaling_fwd"][META_O]
                        output = O_quantizer(output)

        if inference_params is None:
            if qkv_format in ["sbhd", "bshd"] and "padding" in attn_mask_type:
                output = dpa_utils.UnpackTensor.apply(indices_q, batch_size * max_seqlen_q, output)
        elif qkv_format in ["bshd", "sbhd_2bshd"]:
            # all KV caching cases use thd_2bshd for calculation
            # convert results back to bshd from thd_2bshd
            if isinstance(query_layer, Float8Tensor):
                output._data = tex.convert_thd_to_bshd(
                    output._data,
                    cu_seqlens_q,
                    batch_size,
                    context_len,
                )
                output = Float8Tensor.make_like(output, data=output._data, shape=output._data.shape)
            else:
                output = tex.convert_thd_to_bshd(
                    output,
                    cu_seqlens_q,
                    batch_size,
                    context_len,
                )

        if q_format == "sbhd":
            # (bs)hd -> bs(hd) -> sb(hd)
            if fp8 and fp8_meta["recipe"].fp8_mha:
                output_data = (
                    output._data.reshape(batch_size, max_seqlen_q // cp_size, -1)
                    .transpose(0, 1)
                    .contiguous()
                )
                output = Float8Tensor.make_like(
                    output,
                    data=output_data,
                    shape=output_data.shape,
                )
            else:
                output = output.view(batch_size, max_seqlen_q // cp_size, -1).transpose(0, 1)
        elif q_format == "bshd":
            # (bs)hd -> bs(hd)
            output = output.reshape(batch_size, max_seqlen_q // cp_size, -1)
        elif q_format == "thd":
            # thd -> t(hd)
            output = output.reshape(output.shape[0], -1)

        return output.contiguous()


def _combine_tensors(
    tensors: List[torch.Tensor],
    dim: int,
) -> torch.Tensor:
    """Combine tensors along a particular dimension"""

    num_tensors = len(tensors)
    new_shape = list(tensors[0].shape)
    new_shape.insert(dim, num_tensors)
    if isinstance(tensors[0], Float8Tensor):
        new_stride = list(tensors[0]._data.stride())
        new_stride.insert(dim, int(new_stride[dim - 1] / num_tensors))
        combined_tensor = torch.Tensor().to(device=tensors[0].device, dtype=tensors[0]._data.dtype)
        combined_tensor.set_(
            tensors[0]._data.untyped_storage(),
            tensors[0]._data.storage_offset(),
            new_shape,
            new_stride,
        )
        combined_tensor = Float8Tensor.make_like(tensors[0], data=combined_tensor, shape=new_shape)
    else:
        new_stride = list(tensors[0].stride())
        new_stride.insert(dim, int(new_stride[dim - 1] / num_tensors))
        combined_tensor = torch.Tensor().to(device=tensors[0].device, dtype=tensors[0].dtype)
        combined_tensor.set_(
            tensors[0].untyped_storage(), tensors[0].storage_offset(), new_shape, new_stride
        )

    return combined_tensor


class FusedAttnFunc(torch.autograd.Function):
    """Function for FusedAttention with separate Q, K, V tensors"""

    @staticmethod
    def forward(
        ctx,
        is_training,
        max_seqlen_q,
        max_seqlen_kv,
        cu_seqlens_q,
        cu_seqlens_kv,
        cu_seqlens_q_padded,
        cu_seqlens_kv_padded,
        page_table_k,
        page_table_v,
        q,
        k,
        v,
        attn_bias,
        attn_scale,
        dropout_p,
        fast_zero_fill,
        qkv_layout,
        attn_bias_type,
        attn_mask_type,
        window_size,
        rng_gen,
        fused_attention_backend,
        use_FAv2_bwd,
        fp8,
        fp8_meta,
        quantizers,
        deterministic,
    ):
        # pylint: disable=missing-function-docstring
        # "fp8_mha" decides outputs in fp8, while inputs are inferred from the real dtype
        is_input_fp8 = False
        is_output_fp8 = fp8_meta["recipe"].fp8_mha if "recipe" in fp8_meta else False

        # FP16/BF16 attn:                  fake_dtype = torch.float16 or torch.bfloat16
        # FP8 attn, is_output_fp8 = False: fake_dtype = torch.float16 or torch.bfloat16
        # FP8 attn, is_output_fp8 = True:  fake_dtype = torch.float8_e4m3fn
        fake_dtype = q.dtype

        QKV_quantizer, O_quantizer, S_quantizer, dQKV_quantizer, dO_quantizer, dP_quantizer = (
            dpa_utils.get_attention_quantizers(fp8, quantizers, cp_specific_quantizers=False)
        )
        if fp8:
            fused_attention_backend = FusedAttnBackend["FP8"]
            assert isinstance(k, q.__class__) and isinstance(
                v, q.__class__
            ), "q, k, and v must have the same type."

            is_input_fp8 = isinstance(q, Float8Tensor)
            q_fp8, k_fp8, v_fp8 = None, None, None
            if is_input_fp8:
                q_fp8, k_fp8, v_fp8 = q, k, v
            else:
                # 1: qkv packed, 2: kv packed, 3: qkv separate
                qkv_group = len(qkv_layout.replace("paged_kv_", "").split("_"))
                match qkv_group:
                    case 1:
                        dim = qkv_layout.find("3")
                        qkv = _combine_tensors([q, k, v], dim)
                        qkv_c = qkv.view(-1, qkv.shape[-3] * qkv.shape[-2] * qkv.shape[-1])
                        qkv_fp8 = QKV_quantizer(qkv)
                        q_fp8, k_fp8, v_fp8 = _SplitAlongDim.apply(qkv_fp8, dim, [1, 1, 1], True)
                    case 2:
                        q_fp8 = QKV_quantizer(q)
                        dim = qkv_layout.split("_")[1].find("2")
                        kv = _combine_tensors([k, v], dim)
                        kv_c = kv.view(-1, kv.shape[-3] * kv.shape[-2] * kv.shape[-1])
                        kv_fp8 = QKV_quantizer(kv_c)
                        k_fp8, v_fp8 = _SplitAlongDim.apply(kv_fp8, dim, [1, 1], True)
                    case 3:
                        q_fp8 = QKV_quantizer(q)
                        k_fp8 = QKV_quantizer(k)
                        v_fp8 = QKV_quantizer(v)
                    case _:
                        raise "Invalid qkv_layout " + qkv_layout
            # q_fp8, k_fp8, v_fp8, out_fp8: torch.float8_e4m3fn
            out_fp8, aux_ctx_tensors = fused_attn_fwd(
                is_training,
                max_seqlen_q,
                max_seqlen_kv,
                cu_seqlens_q,
                cu_seqlens_kv,
                q_fp8,
                k_fp8,
                v_fp8,
                fake_dtype,
                fused_attention_backend,
                attn_bias,
                cu_seqlens_q_padded,
                cu_seqlens_kv_padded,
                None,
                None,
                S_quantizer,
                O_quantizer,
                attn_scale,
                dropout_p,
                fast_zero_fill,
                qkv_layout,
                attn_bias_type,
                attn_mask_type,
                window_size,
                rng_gen,
            )
            if is_output_fp8:
                out_ret = out_fp8
            else:
                out_ret = out_fp8.dequantize().view(out_fp8.shape)
            # is_output_fp8 = False: out_save.dtype = torch.float16 or torch.bfloat16
            # is_output_fp8 = True:  out_save.dtype = torch.float8_e4m3fn
            out_save = out_ret

            if not int(os.getenv("NVTE_FP8_DPA_BWD", "1")):
                # 1: qkv packed, 2: kv packed, 3: qkv separate
                if is_input_fp8:
                    qkv_group = len(qkv_layout.replace("paged_kv_", "").split("_"))
                    if qkv_group == 1:
                        dim = qkv_layout.find("3")
                        qkv = _combine_tensors([q, k, v], dim)
                        qkv_c = qkv.view(-1, qkv.shape[-3] * qkv.shape[-2] * qkv.shape[-1])
                        qkv_no_fp8 = qkv_c.dequantize().view(qkv.shape)
                        q, k, v = _SplitAlongDim.apply(qkv_no_fp8, dim, [1, 1, 1], True)
                    if qkv_group == 2:
                        q = q.dequantize()
                        dim = qkv_layout.replace("paged_kv_", "").split("_")[1].find("2")
                        kv = _combine_tensors([k, v], dim)
                        kv_c = kv.view(-1, kv.shape[-3] * kv.shape[-2] * kv.shape[-1])
                        kv_no_fp8 = kv.dequantize()
                        k, v = _SplitAlongDim.apply(kv_no_fp8, dim, [1, 1], True)
                    if qkv_group == 3:
                        q = q.dequantize()
                        k = k.dequantize()
                        v = v.dequantize()
                if is_output_fp8:
                    out_save = out_fp8.dequantize()

            fp8_tensors = (q_fp8, k_fp8, v_fp8, out_fp8)
        else:
            # q, k, v, out_ret: torch.float16 or torch.bfloat16
            out_ret, aux_ctx_tensors = fused_attn_fwd(
                is_training,
                max_seqlen_q,
                max_seqlen_kv,
                cu_seqlens_q,
                cu_seqlens_kv,
                q,
                k,
                v,
                fake_dtype,
                fused_attention_backend,
                attn_bias,
                cu_seqlens_q_padded,
                cu_seqlens_kv_padded,
                page_table_k,
                page_table_v,
                None,  # s_quantizer
                None,  # o_quantizer
                attn_scale,
                dropout_p,
                fast_zero_fill,
                qkv_layout,
                attn_bias_type,
                attn_mask_type,
                window_size,
                rng_gen,
            )
            out_save = out_ret
            fp8_tensors = (None, None, None, None)

        ctx.fp8 = fp8 and int(os.getenv("NVTE_FP8_DPA_BWD", "1"))

        from .cpu_offload import CPUOffloadEnabled

        if CPUOffloadEnabled:
            if ctx.fp8:
                tensor_list = fp8_tensors
            else:
                tensor_list = [q, k, v, out_save]

            qkv_layout = "sbhd_sbhd_sbhd"
            mark_activation_offload(*tensor_list)
            mark_activation_offload(*aux_ctx_tensors)

        ctx.is_input_fp8 = is_input_fp8
        ctx.is_output_fp8 = is_output_fp8
        qkvo_tensors = (q, k, v, out_save) if not ctx.fp8 else (None, None, None, None)
        tensors_to_save, tensor_objects = prepare_for_saving(
            *fp8_tensors,
            *qkvo_tensors,
            cu_seqlens_q,
            cu_seqlens_kv,
            cu_seqlens_q_padded,
            cu_seqlens_kv_padded,
            *aux_ctx_tensors,
        )
        ctx.save_for_backward(*tensors_to_save)
        ctx.tensor_objects = tensor_objects
        ctx.fp8_meta = fp8_meta

        ctx.dQKV_quantizer = dQKV_quantizer
        ctx.dO_quantizer = dO_quantizer
        ctx.dP_quantizer = dP_quantizer
        ctx.S_quantizer = S_quantizer
        if ctx.fp8:
            ctx.S_quantizer = S_quantizer.copy()
            ctx.S_quantizer.scale = S_quantizer.scale.clone()

        ctx.max_seqlen_q = max_seqlen_q
        ctx.max_seqlen_kv = max_seqlen_kv
        ctx.attn_scale = attn_scale
        ctx.dropout_p = dropout_p
        ctx.fast_zero_fill = fast_zero_fill
        ctx.qkv_layout = qkv_layout
        ctx.attn_bias_type = attn_bias_type
        ctx.attn_mask_type = attn_mask_type
        ctx.window_size = window_size
        ctx.fused_attention_backend = (
            fused_attention_backend if ctx.fp8 else FusedAttnBackend["F16_arbitrary_seqlen"]
        )
        ctx.use_FAv2_bwd = use_FAv2_bwd
        ctx.deterministic = deterministic

        return out_ret

    @staticmethod
    def backward(ctx, d_out):
        # pylint: disable=missing-function-docstring
        if ctx.is_output_fp8:
            assert isinstance(
                d_out, Float8Tensor
            ), "Gradient of the DPA output must be in Float8Tensor type for FP8 MHA."

        # FP16/BF16 attn:                  fake_dtype = torch.float16 or torch.bfloat16
        # FP8 attn, is_output_fp8 = False: fake_dtype = torch.float16 or torch.bfloat16
        # FP8 attn, is_output_fp8 = True:  fake_dtype = torch.float8_e5m2
        fake_dtype = d_out.dtype

        d_out = d_out.contiguous()
        (
            q_fp8,
            k_fp8,
            v_fp8,
            out_fp8,
            q,
            k,
            v,
            out,
            cu_seqlens_q,
            cu_seqlens_kv,
            cu_seqlens_q_padded,
            cu_seqlens_kv_padded,
            *other_tensors,
        ) = restore_from_saved(ctx.tensor_objects, ctx.saved_tensors)

        aux_ctx_tensors = other_tensors

        if not aux_ctx_tensors[0].is_contiguous():
            aux_ctx_tensors[0] = aux_ctx_tensors[0].contiguous()
        rest = [None]
        if ctx.use_FAv2_bwd:
            softmax_lse, rng_state = aux_ctx_tensors
            dq = torch.empty_like(q)
            dk = torch.empty_like(k)
            dv = torch.empty_like(v)
            d_out, q, k, v, out = [maybe_contiguous(x) for x in (d_out, q, k, v, out)]
            flash_attn_cuda_bwd(
                d_out,
                q,
                k,
                v,
                out,
                softmax_lse,
                dq,
                dk,
                dv,
                cu_seqlens_q,
                cu_seqlens_kv,
                ctx.max_seqlen_q,
                ctx.max_seqlen_kv,
                ctx.dropout_p,
                ctx.attn_scale,
                False,
                "causal" in ctx.attn_mask_type,
                None,
                rng_state,
            )
            dq = dq[..., : d_out.shape[-1]]
            dk = dk[..., : d_out.shape[-1]]
            dv = dv[..., : d_out.shape[-1]]
        else:
            with torch.cuda.nvtx.range("_FusedAttn"):
                if ctx.fp8:
                    if ctx.is_output_fp8:
                        d_out_fp8 = d_out
                    else:
                        d_out_fp8 = ctx.dO_quantizer(d_out)
                    dqkv_dtype = TE_DType[d_out_fp8._data.dtype]
                    # q_fp8, k_fp8, v_fp8, out_fp8:      torch.float8_e4m3fn
                    # d_out_fp8, dq_fp8, dk_fp8, dv_fp8: torch.float8_e5m2
                    dq_fp8, dk_fp8, dv_fp8, *rest = fused_attn_bwd(
                        ctx.max_seqlen_q,
                        ctx.max_seqlen_kv,
                        cu_seqlens_q,
                        cu_seqlens_kv,
                        q_fp8,
                        k_fp8,
                        v_fp8,
                        out_fp8,
                        d_out_fp8,
                        fake_dtype,
                        dqkv_dtype,
                        aux_ctx_tensors,
                        ctx.fused_attention_backend,
                        cu_seqlens_q_padded,
                        cu_seqlens_kv_padded,
                        ctx.S_quantizer,
                        ctx.dP_quantizer,
                        ctx.dQKV_quantizer,
                        ctx.attn_scale,
                        ctx.dropout_p,
                        ctx.fast_zero_fill,
                        ctx.qkv_layout,
                        ctx.attn_bias_type,
                        ctx.attn_mask_type,
                        ctx.window_size,
                        ctx.deterministic,
                    )

                    # is_input_fp8 = False: dq, dk, dv: torch.float16 or torch.bfloat16
                    # is_input_fp8 = True:  dq, dk, dv: torch.float8_e5m2
                    if not ctx.is_input_fp8:
                        qkv_group = len(ctx.qkv_layout.replace("paged_kv_", "").split("_"))
                        if qkv_group == 1:
                            dim = ctx.qkv_layout.find("3")
                            dqkv_fp8_data = _combine_tensors(
                                [dq_fp8._data, dk_fp8._data, dv_fp8._data], dim
                            )
                            dqkv_fp8 = dq_fp8.make_like(
                                tensor=dq_fp8, data=dqkv_fp8_data, shape=dqkv_fp8_data.shape
                            )
                            dqkv = dqkv_fp8.dequantize()
                            dq, dk, dv = _SplitAlongDim.apply(dqkv, dim, [1, 1, 1], True)
                        if qkv_group == 2:
                            dq = dq_fp8.dequantize()
                            dim = ctx.qkv_layout.split("_")[1].find("2")
                            dkv_fp8 = _combine_tensors([dk_fp8, dv_fp8], dim)
                            dkv_c_fp8 = dkv_fp8.view(
                                -1, dkv_fp8.shape[-3] * dkv_fp8.shape[-2] * dkv_fp8.shape[-1]
                            )
                            dkv = dkv_c_fp8.dequantize()
                            dk, dv = _SplitAlongDim.apply(dkv, dim, [1, 1], True)
                        if qkv_group == 3:
                            dq = dq_fp8.dequantize()
                            dk = dk_fp8.dequantize()
                            dv = dv_fp8.dequantize()
                    else:
                        dq, dk, dv = dq_fp8, dk_fp8, dv_fp8
                else:
                    if isinstance(d_out, QuantizedTensor):
                        d_out = d_out.dequantize()
                    dqkv_dtype = TE_DType[d_out.dtype]
                    # q, k, v, out, d_out, dq, dk, dv: torch.float16 or torch.bfloat16
                    dq, dk, dv, *rest = fused_attn_bwd(
                        ctx.max_seqlen_q,
                        ctx.max_seqlen_kv,
                        cu_seqlens_q,
                        cu_seqlens_kv,
                        q,
                        k,
                        v,
                        out,
                        d_out,
                        fake_dtype,
                        dqkv_dtype,
                        aux_ctx_tensors,
                        ctx.fused_attention_backend,
                        cu_seqlens_q_padded,
                        cu_seqlens_kv_padded,
                        None,
                        None,
                        None,
                        ctx.attn_scale,
                        ctx.dropout_p,
                        ctx.fast_zero_fill,
                        ctx.qkv_layout,
                        ctx.attn_bias_type,
                        ctx.attn_mask_type,
                        ctx.window_size,
                        ctx.deterministic,
                    )

        # if no_bias or alibi, return dqkv
        if ctx.attn_bias_type in ["no_bias", "alibi"]:
            return (
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                dq,
                dk,
                dv,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            )
        # else, return (dqkv, dbias)
        return (
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            dq,
            dk,
            dv,
            rest[0],
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


class FusedAttention(torch.nn.Module):
    """Dot product attention, with multiple backends:

    1. FusedAttnBackend["F16_max512_seqlen"]
       cuDNN based fused attention for FP16/BF16 and <=512 sequence length.
    2. FusedAttnBackend["F16_arbitrary_seqlen"]
       cuDNN based fused attention for FP16/BF16 and any sequence length.

    Support matrix:

    | backend       | 1                       | 2                              |
    | flash based   | no                      | yes                            |
    | cuDNN based   | yes                     | yes                            |
    | qkv dtype     | fp16/bf16               | fp16/bf16                      |
    | attn_type     | self/cross              | self/cross                     |
    | qkv_layout    |                         |                                |
    |  - (q,k,v)    | sb3hd, bs3hd            | sb3hd, bs3hd, sbh3d, bsh3d     |
    |               | sbhd_sb2hd, bshd_bs2hd  | sbhd_sb2hd, bshd_bs2hd         |
    |               | bshd_bshd_bshd          | sbhd_sbh2d, bshd_bsh2d         |
    |               |                         | sbhd_sbhd_sbhd, bshd_bshd_bshd |
    | mask_type     | causal/padding/no_mask  | causal/padding/no_mask         |
    | bias_type     | post_scale_bias/no_bias | post_scale_bias/alibi/no_bias  |
    | dropout       | yes                     | yes                            |
    | max_seqlen    | <=512, multiple of 64   | any, multiple of 64            |
    | head_dim      | 64                      | <=128, multiple of 8           |
    | output dtype  | fp16/bf16               | fp16/bf16                      |
    """

    def __init__(
        self,
        softmax_scale: float,
        attention_dropout: float = 0.0,
        attention_dropout_ctx: Optional[Callable] = nullcontext,
        attention_type: str = "self",
        layer_number: Optional[int] = None,
        deterministic: bool = False,
    ) -> None:
        super().__init__()

        self.softmax_scale = softmax_scale
        self.attention_dropout = attention_dropout
        self.attention_dropout_ctx = attention_dropout_ctx
        self.attention_type = attention_type
        self.use_FAv2_bwd = os.getenv(
            "NVTE_FUSED_ATTN_USE_FAv2_BWD", "0"
        ) == "1" and get_device_compute_capability() == (9, 0)
        self.layer_number = 1 if layer_number is None else layer_number
        self.deterministic = deterministic

        def remove_extra_states_check(self, incompatible_keys):  # pylint: disable=unused-argument
            """
            Temporarily remove fused_attention._extra_state as a missing key
            or an unexpected key when loading Transformer Engine checkpoints.
            Please store FP8 metadata as DotProductAttention's _extra_state,
            rather than FusedAttention's _extra_state. This hook will be
            phased out in Transformer Engine 2.0.
            """
            for key in incompatible_keys.missing_keys:
                if "fused_attention._extra_state" in key:
                    incompatible_keys.missing_keys.remove(key)
            for key in incompatible_keys.unexpected_keys:
                if "fused_attention._extra_state" in key:
                    incompatible_keys.unexpected_keys.remove(key)
                    warnings.warn(
                        "fused_attention._extra_state is not loaded from checkpoint. Please map "
                        "FusedAttention's _extra_state to DotProductAttention's _extra_state."
                    )

        self.register_load_state_dict_post_hook(remove_extra_states_check)

    @no_torch_dynamo()
    def forward(
        self,
        query_layer: torch.Tensor,
        key_layer: torch.Tensor,
        value_layer: torch.Tensor,
        qkv_layout: str = "sbh3d",
        cu_seqlens_q: Optional[torch.Tensor] = None,
        cu_seqlens_kv: Optional[torch.Tensor] = None,
        cu_seqlens_q_padded: Optional[torch.Tensor] = None,
        cu_seqlens_kv_padded: Optional[torch.Tensor] = None,
        max_seqlen_q: Optional[int] = None,
        max_seqlen_kv: Optional[int] = None,
        attn_mask_type: str = "causal",
        attention_mask: Optional[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]] = None,
        window_size: Optional[Tuple[int, int]] = None,
        fused_attention_backend: tex.NVTE_Fused_Attn_Backend = tex.NVTE_Fused_Attn_Backend.NVTE_No_Backend,
        core_attention_bias_type: str = "no_bias",
        core_attention_bias: Optional[torch.Tensor] = None,
        fast_zero_fill: bool = True,
        cp_group: Optional[Union[dist_group_type, List[dist_group_type]]] = None,
        cp_global_ranks: List[int] = None,
        cp_stream: torch.cuda.Stream = None,
        cp_comm_type: str = "p2p",
        fp8: bool = False,
        fp8_meta: Optional[Dict[str, Any]] = None,
        quantizers=None,
        pad_between_seqs: bool = False,
        inference_params: Optional[InferenceParams] = None,
    ) -> torch.Tensor:
        """fused attention fprop"""
        assert (
            fused_attention_backend != tex.NVTE_Fused_Attn_Backend.NVTE_No_Backend
        ), "No fused attention backend supports this input combination!"
        assert all(
            x.dtype in [torch.float16, torch.bfloat16] or isinstance(x, Float8Tensor)
            for x in [query_layer, key_layer, value_layer]
        ), "FusedAttention only supports FP16 and BF16 data types, or Float8Tensors."
        assert (
            query_layer.is_cuda and key_layer.is_cuda and value_layer.is_cuda
        ), "FusedAttention only supports CUDA tensors."
        assert (
            qkv_layout in QKVLayouts
        ), f"FusedAttention does not support qkv_layout = {qkv_layout}!"

        cp_size = 1
        if isinstance(cp_group, dist_group_type):
            cp_size = get_distributed_world_size(cp_group)
        elif isinstance(cp_group, list):
            for group in cp_group:
                cp_size *= get_distributed_world_size(group)
        context_parallel = cp_size > 1

        # get q_format and kv_format for training and inference
        qkv_format, q_format, kv_format = dpa_utils.get_qkv_format(qkv_layout, inference_params)

        # cuDNN can work with 0-length sequences in the batch for both bshd/sbhd and thd formats
        # however, for bshd/sbhd, q/k/v tensors need to have the same batch size as indicated by
        # cu_seqlens, whereas thd does not have this requirement
        # e.g. if q_format = bshd, and q.shape = [3, 1, 16, 64], we should have k.shape[0] =
        # v.shape[0] = q.shape[0], and cu_seqlens_q.shape = cu_seqlens_kv.shape = [4]
        if q_format in ["bshd", "sbhd"] or kv_format in ["bshd", "sbhd"]:
            batch_size = query_layer.shape[0] if q_format == "bshd" else query_layer.shape[1]
            cu_seqlens_q = cu_seqlens_q[: batch_size + 1]
            cu_seqlens_kv = cu_seqlens_kv[: batch_size + 1]

        page_table = None
        if inference_params is None:
            if qkv_format in ["sbhd", "bshd"]:
                if qkv_format == "sbhd":
                    batch_size = query_layer.shape[1]
                    max_seqlen_q = query_layer.shape[0]
                    max_seqlen_kv = key_layer.shape[0]
                if qkv_format == "bshd":
                    batch_size = query_layer.shape[0]
                    max_seqlen_q = query_layer.shape[1]
                    max_seqlen_kv = key_layer.shape[1]
                max_seqlen_q *= cp_size
                max_seqlen_kv *= cp_size
                if "padding" in attn_mask_type:
                    assert (
                        not context_parallel
                    ), "Padding mask not supported with context parallelism!"
                    if cu_seqlens_q is None or cu_seqlens_kv is None:
                        if attention_mask is None:
                            raise RuntimeError(
                                "Please provide attention_mask or cu_seqlens for padding!"
                            )
                        if self.attention_type == "self":
                            cu_seqlens_q = dpa_utils.get_cu_seqlens(attention_mask)
                            cu_seqlens_kv = cu_seqlens_q
                        else:
                            cu_seqlens_q = dpa_utils.get_cu_seqlens(attention_mask[0])
                            cu_seqlens_kv = dpa_utils.get_cu_seqlens(attention_mask[1])
                else:
                    if cu_seqlens_q is None:
                        cu_seqlens_q = dpa_utils.get_full_cu_seqlens(
                            batch_size,
                            max_seqlen_q,
                            query_layer.device,
                        )
                    if cu_seqlens_kv is None:
                        cu_seqlens_kv = dpa_utils.get_full_cu_seqlens(
                            batch_size,
                            max_seqlen_kv,
                            key_layer.device,
                        )
            if qkv_format == "thd":
                assert (
                    max_seqlen_q is not None
                    and max_seqlen_kv is not None
                    and cu_seqlens_q is not None
                    and cu_seqlens_kv is not None
                ), "max_seqlen_q/kv and cu_seqlens_q/kv can not be None when qkv_format is thd!"
        elif inference_params.is_paged:
            page_table = inference_params.cache_manager.page_table

        if (q_format == "thd" or "padding" in attn_mask_type) and cu_seqlens_q_padded is None:
            cu_seqlens_q_padded = cu_seqlens_q
        if (kv_format == "thd" or "padding" in attn_mask_type) and cu_seqlens_kv_padded is None:
            cu_seqlens_kv_padded = cu_seqlens_kv

        use_FAv2_bwd = (
            self.use_FAv2_bwd
            and (core_attention_bias_type == "no_bias")
            and (fused_attention_backend == tex.NVTE_Fused_Attn_Backend.NVTE_F16_arbitrary_seqlen)
        )

        if fp8:
            assert fused_attention_backend == tex.NVTE_Fused_Attn_Backend.NVTE_FP8, (
                f"cuDNN attention sub-backend {int(tex.NVTE_Fused_Attn_Backend.NVTE_FP8)}"
                " is required for FP8 attention!"
            )
            assert fp8_meta is not None, "FP8 metadata fp8_meta is required for FP8 attention!"
            assert not context_parallel or fp8_meta["recipe"].reduce_amax, (
                "Amax reduction across TP+CP group is necessary when using context parallelism with"
                " FP8!"
            )

        if context_parallel:
            assert (
                fp8
                or fused_attention_backend == tex.NVTE_Fused_Attn_Backend.NVTE_F16_arbitrary_seqlen
            ), f"{fused_attention_backend} does not work with context parallelism!"
            assert core_attention_bias_type not in [
                "alibi"
            ], f"{core_attention_bias_type} is not supported with context parallelism!"
            query_layer, key_layer, value_layer = [
                x.contiguous() for x in (query_layer, key_layer, value_layer)
            ]
            with self.attention_dropout_ctx():
                output = attn_forward_func_with_cp(
                    self.training,
                    query_layer,
                    key_layer,
                    value_layer,
                    cu_seqlens_q,
                    cu_seqlens_kv,
                    max_seqlen_q,
                    max_seqlen_kv,
                    cu_seqlens_q_padded,
                    cu_seqlens_kv_padded,
                    self.attention_dropout if self.training else 0.0,
                    cp_group,
                    cp_global_ranks,
                    cp_stream,
                    cp_comm_type,
                    softmax_scale=self.softmax_scale,
                    qkv_format=qkv_format,
                    attn_mask_type=attn_mask_type,
                    attn_bias_type=core_attention_bias_type,
                    attn_bias=core_attention_bias,
                    deterministic=self.deterministic,
                    use_fused_attention=True,
                    window_size=window_size,
                    fp8=fp8,
                    fp8_meta=fp8_meta,
                    quantizers=quantizers,
                    pad_between_seqs=pad_between_seqs,
                )
        else:
            with self.attention_dropout_ctx():
                output = FusedAttnFunc.apply(
                    self.training,
                    max_seqlen_q,
                    max_seqlen_kv,
                    cu_seqlens_q,
                    cu_seqlens_kv,
                    cu_seqlens_q_padded,
                    cu_seqlens_kv_padded,
                    page_table,
                    page_table,
                    query_layer,
                    key_layer,
                    value_layer,
                    core_attention_bias,
                    self.softmax_scale,
                    self.attention_dropout if self.training else 0.0,
                    fast_zero_fill,
                    qkv_layout,
                    core_attention_bias_type,
                    attn_mask_type,
                    window_size,
                    None,  # rng_gen
                    fused_attention_backend,
                    use_FAv2_bwd,
                    fp8,
                    fp8_meta,
                    quantizers,
                    self.deterministic,
                )

        # ...hd -> ...(hd)
        return output.view(*output.shape[:-2], -1)


class DotProductAttention(TransformerEngineBaseModule):
    """Allows the model to jointly attend to information from different
    representation subspaces as described in the paper:
    `Attention Is All You Need <https://arxiv.org/abs/1706.03762>`_.

    .. note::

        Argument :attr:`attention_mask` in the `forward` call is only used when
        :attr:`attn_mask_type` includes '"padding"' or `"arbitrary"`.

    .. warning::

        FlashAttention uses a non-deterministic algorithm for optimal performance. To observe
        deterministic behavior at the cost of performance, use FlashAttention version >= `2.4.1`
        and set the environment variable :attr:`NVTE_ALLOW_NONDETERMINISTIC_ALGO=0`. In order
        to disable`flash-attn` entirely, set :attr:`NVTE_FLASH_ATTN=0`.

    .. note::

        Transformer Engine stores the FP8 metadata under a `._extra_state` key when checkpointing.
        As the FP8 attention support expands from one backend to multiple backends, the location
        of that key has also shifted (see `FP8 checkpoint compatibility <https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/faq.html#fp8-checkpoint-compatibility>`_).


    Parameters
    ----------
    num_attention_heads : int
                         number of attention heads in the transformer layer.
    kv_channels : Union[int, Tuple[int, int]]
                the head size in key and value tensors. If the same, :attr:`kv_channels` can be
                an integer; if not, :attr:`kv_channels` should be a tuple of two integers.
    num_gqa_groups : Optional[int] = None
                    number of GQA groups in the transformer layer.
                    Grouped Query Attention is described in
                    `this paper <https://arxiv.org/pdf/2305.13245.pdf>`_.
                    This only affects the keys and values, not the queries.
                    GQA-1 is equivalent to Multi-Query Attention
                    (`MQA <https://arxiv.org/pdf/1911.02150.pdf>`_), while GQA-H
                    is equivalent to MHA, i.e. `num_gqa_groups = num_attention_heads`.
    attention_dropout: float, default = 0.0
                      dropout probability for the dropout op during multi-head attention.
    attn_mask_type: str, default = `causal`
                   type of attention mask passed into softmax operation, options are "`no_mask`",
                   "`padding`", "`causal`", "`padding,causal`", "`causal,padding`",
                   "`padding_causal`", "`causal_bottom_right`", "`padding_causal_bottom_right`", and
                   "`arbitrary`", where "`padding,causal`", "`causal,padding`" and "`padding_causal`"
                   are equivalent. This arg can be overridden by :attr:`attn_mask_type` in the
                   `forward` method. It is useful for cases involving compilation/tracing, e.g.
                   ONNX export, and the forward arg is useful for dynamically changing mask types,
                   e.g. a different mask for training and inference.
                   1. For "`no_mask`", no attention mask is applied.
                   2. For "`causal`", "`causal_bottom_right`", or the causal mask in
                   "`padding_causal`" and "`padding_causal_bottom_right`", Transformer Engine
                   calculates and applies an upper triangular mask to the softmax input.
                   No user input is needed. Causal masks without the "`bottom_right`" appendix align
                   the diagonal line to the top left corner of the softmax matrix. With
                   "`bottom_right`", the causal mask is aligned to the bottom right corner, which is
                   often used in inference/KV caching.
                   3. For "`padding`", or the padding mask in "`padding_causal`" and
                   "`padding_causal_bottom_right`", users need to provide the locations of padded
                   tokens, either via :attr:`cu_seqlens_q` and :attr:`cu_seqlens_kv` (both in shape
                   [batch_size + 1]), or via :attr:`attention_mask` (one tensor for self-attention
                   in shape [batch_size, 1, 1, max_seqlen_q], or two tensors in a tuple for
                   cross-attention in shapes [batch_size, 1, 1, max_seqlen_q] and
                   [batch_size, 1, 1, max_seqlen_kv]).
                   4. For "`arbitrary`", users need to provide a mask that is broadcastable to
                   the shape of softmax input [batch_size, num_heads, max_seqlen_q, max_seqlen_kv].
    window_size: Optional[Tuple[int, int]], default = `None`
                sliding window size for local attention, where query at position i attends to keys
                in [i + seqlen_k - seqlen_q - window_size[0], i + seqlen_k - seqlen_q
                + window_size[1]] inclusive. Special cases (-1, -1) and (-1, 0) mean no sliding
                window and causal mask specifically. Both `causal` and `causal_bottom_right` masks
                map to `window_size = (-1, 0)` and Transformer Engine distinguishes them based on
                `attn_mask_type`. Similar to :attr:`attn_mask_type`, `window_size` can
                be overridden by :attr:`window_size` in `forward` as well.
    attention_type: str, default = `self`
                   type of attention, either "`self`" and "`cross`".
    layer_number: int, default = `None`
                 layer number of the current `DotProductAttention` when multiple such modules
                 are concatenated, for instance in consecutive transformer blocks.
    qkv_format: str, default = `sbhd`
               dimension format for `query_layer`, `key_layer` and `value_layer`,
               {`sbhd`, `bshd`, `thd`}. `s` stands for the sequence length, `b` batch size,
               `h` the number of heads, `d` head size, and `t` the total number of tokens
               in a batch, with `t = sum(s_i), for i = 0...b-1`. `sbhd` and `bshd` formats
               are used for when sequences in a batch are of equal length or padded to
               equal length, and the `thd` format is used for when sequences in a batch
               have different lengths. Please note that these formats do not reflect how
               tensors `query_layer`, `key_layer`, `value_layer` are laid out in memory.
               For that, please use `get_qkv_layout` to gain the layout information.
    softmax_scale: Optional[float], default = `None`
                softmax scale for the attention scores. If `None`, defaults to
                `1.0/math.sqrt(kv_channels if isinstance(kv_channels, int) else kv_channels[0])`.

    Parallelism parameters
    ----------------------
    sequence_parallel : bool, default = `False`
                       if set to `True`, uses sequence parallelism.
    tp_size : int, default = 1
             tensor parallel world size.
    tp_group : ProcessGroup, default = `None`
              tensor parallel process group.
    cp_group : Union[ProcessGroup, List[ProcessGroup]], default = `None`
              context parallel process group.
              ProcessGroup is for cp_comm_type of "p2p", "all_gather", and "a2a".
              List[ProcessGroup] is for cp_comm_type of "a2a+p2p", where cp_group[0]
              and cp_group[1] are for a2a and p2p communications respectively.
    cp_global_ranks : list of global rank IDs, default = `None`
                     global rank IDs of GPUs that are in cp_group.
    cp_stream : CUDA stream, default = `None`
               context parallelism splits flash attention into multiple steps for
               compute and communication overlapping. To address the wave quantization
               issue of each split step, we add an additional CUDA stream so that we
               can overlap two flash attention kernels.
    cp_comm_type : str, default = `p2p`
                  inter-gpu communication type for context parallelism.
                  Can be "p2p" or "all_gather" or "a2a" or "a2a+p2p".
                  "p2p": Exchange KV chunks with P2P communications in ring topology.
                         P2P is async and can be overlapped with attention compute.
                  "all_gather": All-gather to get full sequence of KV before attention.
                                The all-gather is not async, and cannot be overlapped.
                  "a2a": Like DeepSpeed Ulysses, scatter attention heads across the CP
                         group, and gather to get full sequence of QKV.
                  "a2a+p2p": hierarchical CP implementation. First applying a2a to QKV
                  across each CP sub-group (e.g., via NVLink), then exchanging KV with
                  p2p between sub-groups (e.g., via IBLink).
    """

    def __init__(
        self,
        num_attention_heads: int,
        kv_channels: Union[int, Tuple[int, int]],
        num_gqa_groups: Optional[int] = None,
        attention_dropout: float = 0.0,
        qkv_format: str = "sbhd",
        attn_mask_type: str = "causal",
        window_size: Optional[Tuple[int, int]] = None,
        sequence_parallel: bool = False,
        tp_size: int = 1,
        get_rng_state_tracker: Optional[Callable] = None,
        tp_group: Optional[dist_group_type] = None,
        layer_number: Optional[int] = None,
        attention_type: str = "self",
        cp_group: Optional[Union[dist_group_type, List[dist_group_type]]] = None,
        cp_global_ranks: List[int] = None,
        cp_stream: torch.cuda.Stream = None,
        cp_comm_type: str = "p2p",
        softmax_scale: Optional[float] = None,
    ) -> None:
        super().__init__()

        self.logger = logging.getLogger("DotProductAttention")
        self.logger.setLevel(attn_log._log_level)
        if not self.logger.hasHandlers():
            self.logger.addHandler(attn_log._stream_handler)
        self.qkv_format = qkv_format
        attn_mask_type = attn_mask_type.replace(",", "_")
        if attn_mask_type == "causal_padding":
            attn_mask_type = "padding_causal"
        self.attn_mask_type = attn_mask_type
        self.window_size = dpa_utils.check_set_window_size(attn_mask_type, window_size)
        if tp_group is None:
            self.tp_size = tp_size
            if tp_size == 1:
                self.set_tensor_parallel_group(tp_group)
        else:
            self.tp_size = get_distributed_world_size(tp_group)
            self.set_tensor_parallel_group(tp_group)
        self.get_rng_state_tracker = get_rng_state_tracker
        self.num_attention_heads = num_attention_heads
        self.layer_number = 1 if layer_number is None else layer_number
        self.cp_group = cp_group
        self.cp_global_ranks = cp_global_ranks
        self.cp_stream = cp_stream
        self.cp_comm_type = cp_comm_type

        self.hidden_size_per_attention_head_k = (
            kv_channels if isinstance(kv_channels, int) else kv_channels[0]
        )
        self.hidden_size_per_attention_head_v = (
            kv_channels if isinstance(kv_channels, int) else kv_channels[1]
        )

        self.num_gqa_groups = num_attention_heads if num_gqa_groups is None else num_gqa_groups
        self.num_gqa_groups_per_partition = int(self.num_gqa_groups // self.tp_size)

        assert (
            num_attention_heads % self.num_gqa_groups == 0
        ), "The number of attention heads must be divisible by the number of GQA groups!"

        self.rng_states_tracker = None
        if sequence_parallel or get_rng_state_tracker is None:
            attention_dropout_ctx = nullcontext
        else:
            self.rng_states_tracker = get_rng_state_tracker()
            set_all_rng_states(self.rng_states_tracker.get_states())
            attention_dropout_ctx = self.rng_states_tracker.fork

        if softmax_scale is None:
            softmax_scale = 1.0 / math.sqrt(
                kv_channels if isinstance(kv_channels, int) else kv_channels[0]
            )

        self.deterministic = (
            not bool(int(os.getenv("NVTE_ALLOW_NONDETERMINISTIC_ALGO", "1")))
            or torch.are_deterministic_algorithms_enabled()
        )
        # To use the workspace optimization path for determinism, please
        # set NVTE_FUSED_ATTN_FORCE_WORKSPACE_OPT=1 for cuDNN >=8.9.5 and <9.0.0,
        # and set NVTE_ALLOW_NONDETERMINISTIC_ALGO=0 for cuDNN >=9.0.0.
        cudnn_version = get_cudnn_version()
        if (8, 9, 5) <= cudnn_version < (9, 0, 0):
            if self.deterministic:
                os.environ["NVTE_FUSED_ATTN_FORCE_WORKSPACE_OPT"] = "1"

            # CUDNN_FRONTEND_ATTN_DP_WORKSPACE_LIMIT
            # - unset:       enables workspace optimization when required workspace is <= 256MB
            #                or when bias gradient needs to be computed
            # - n:           enables workspace optimization when required workspace is <= n bytes
            # - -1:          enables workspace optimization always
            # - 0:           disables workspace optimization always
            if "NVTE_FUSED_ATTN_FORCE_WORKSPACE_OPT" in os.environ:
                if os.environ["NVTE_FUSED_ATTN_FORCE_WORKSPACE_OPT"] == "0":
                    os.environ["CUDNN_FRONTEND_ATTN_DP_WORKSPACE_LIMIT"] = "0"
                if os.environ["NVTE_FUSED_ATTN_FORCE_WORKSPACE_OPT"] == "1":
                    os.environ["CUDNN_FRONTEND_ATTN_DP_WORKSPACE_LIMIT"] = "-1"

        assert attention_type in AttnTypes, f"attention_type {attention_type} not supported"

        self.attention_type = attention_type
        self.attention_dropout = attention_dropout

        attn_kwargs = {
            "attention_dropout": attention_dropout,
            "attention_dropout_ctx": attention_dropout_ctx,
        }

        self.flash_attention = FlashAttention(
            softmax_scale,
            attention_type=attention_type,
            layer_number=layer_number,
            deterministic=self.deterministic,
            **attn_kwargs,
        )

        # Instantiating three types since use of flash-attn and FusedAttention
        # might be ruled out due to forward inputs.
        self.fused_attention = FusedAttention(
            softmax_scale,
            attention_type=attention_type,
            layer_number=layer_number,
            deterministic=self.deterministic,
            **attn_kwargs,
        )

        self.unfused_attention = UnfusedDotProductAttention(
            softmax_scale,
            attention_type=attention_type,
            **attn_kwargs,
            layer_number=layer_number,
        )

        def remove_extra_states_check(self, incompatible_keys):  # pylint: disable=unused-argument
            """
            Temporarily remove core_attention._extra_state as a missing key
            when loading older Transformer Engine checkpoints. Will phase out
            this hook in Transformer Engine 2.0.
            """
            for key in incompatible_keys.missing_keys:
                if "core_attention._extra_state" in key:
                    incompatible_keys.missing_keys.remove(key)

        self.register_load_state_dict_post_hook(remove_extra_states_check)

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        """
        This function helps to load Transformer Engine 1.6 and 1.7 checkpoints, where FP8 attention
        metadata is stored under the `core_attention.fused_attention._extra_state` key and not the
        `core_attention._extra_state` key. Please see `FP8 checkpoint compatibility
        <https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/faq.html#fp8-checkpoint-compatibility>`_ for more details.
        """
        fused_attn_key = False
        dot_product_attn_key = False
        for k in state_dict.keys():
            if "core_attention.fused_attention._extra_state" in k:
                fused_attn_key = True
            if "core_attention._extra_state" in k:
                dot_product_attn_key = True
        if fused_attn_key and not dot_product_attn_key:
            prefix = prefix + "fused_attention."
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )

    def _checkpointed_attention_forward(
        self,
        attention_func: Callable,
        *forward_args: Tuple[torch.Tensor, ...],
        **forward_kwargs: Dict[str, Any],
    ) -> torch.Tensor:
        """Forward method with activation checkpointing."""

        def custom_forward(*input_args, **input_kwargs):
            return attention_func(*input_args, **input_kwargs)

        hidden_states = checkpoint(
            custom_forward,
            distribute_saved_activations=False,
            get_rng_state_tracker=self.get_rng_state_tracker,
            tp_group=self.tp_group,
            *forward_args,
            **forward_kwargs,
        )

        return hidden_states

    def set_context_parallel_group(
        self,
        cp_group: Union[dist_group_type, List[dist_group_type], None],
        cp_global_ranks: List[int],
        cp_stream: torch.cuda.Stream,
        cp_comm_type: str = "p2p",
    ) -> None:
        """
        Set the context parallel attributes for the given
        module before executing the forward pass.

        Parameters
        ----------
        cp_group : Union[ProcessGroup, List[ProcessGroup]]
                  context parallel process group.
                  ProcessGroup is for cp_comm_type of "p2p", "all_gather", and "a2a".
                  List[ProcessGroup] is for cp_comm_type of "a2a+p2p", where cp_group[0]
                  and cp_group[1] are for a2a and p2p communications respectively.
        cp_global_ranks : List[int]
                         list of global ranks in the context group.
        cp_stream : torch.cuda.Stream
                   cuda stream for context parallel execution.
        cp_comm_type : str, default = `p2p`
                      inter-gpu communication type for context parallelism.
                      Can be "p2p" or "all_gather" or "a2a" or "a2a+p2p".
                      "p2p": Exchange KV chunks with P2P communications in ring topology.
                             P2P is async and can be overlapped with attention compute.
                      "all_gather": All-gather to get full sequence of KV before attention.
                                    The all-gather is not async, and cannot be overlapped.
                      "a2a": Like DeepSpeed Ulysses, scatter attention heads across the CP
                             group, and gather to get full sequence of QKV.
                      "a2a+p2p": hierarchical CP implementation. First applying a2a to QKV
                      across each CP sub-group (e.g., via NVLink), then exchanging KV with
                      p2p between sub-groups (e.g., via IBLink).
        """
        self.cp_group = cp_group
        self.cp_global_ranks = cp_global_ranks
        self.cp_stream = cp_stream
        self.cp_comm_type = cp_comm_type

    @no_torch_dynamo(recursive=False)
    def forward(
        self,
        query_layer: torch.Tensor,
        key_layer: torch.Tensor,
        value_layer: torch.Tensor,
        attention_mask: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]] = None,
        qkv_format: str = None,
        cu_seqlens_q: torch.Tensor = None,
        cu_seqlens_kv: torch.Tensor = None,
        cu_seqlens_q_padded: torch.Tensor = None,
        cu_seqlens_kv_padded: torch.Tensor = None,
        max_seqlen_q: int = None,
        max_seqlen_kv: int = None,
        attn_mask_type: Optional[str] = None,
        window_size: Optional[Tuple[int, int]] = None,
        checkpoint_core_attention: bool = False,
        core_attention_bias_type: str = "no_bias",
        core_attention_bias: Optional[torch.Tensor] = None,
        alibi_slopes: Optional[torch.Tensor] = None,
        fast_zero_fill: bool = True,
        inference_params: Optional[InferenceParams] = None,
        pad_between_seqs: Optional[bool] = None,
    ) -> torch.Tensor:
        """
        Dot Product Attention Layer.

        .. note::

            Argument :attr:`attention_mask` is only used when :attr:`attn_mask_type`
            includes '"padding"' or `"arbitrary"`.

        .. note::

            DotProductAttention supports three backends: 1) FlashAttention which calls
            HazyResearch/Dao-AILab's `flash-attn <https://arxiv.org/pdf/2305.13245.pdf>`_
            PyTorch API, 2) FusedAttention which has multiple fused attention implementations
            based on `cuDNN Graph API
            <https://docs.nvidia.com/deeplearning/cudnn/developer-guide/index.html#op-fusion>`_
            (see :attr:`FusedAttention` for more details on FusedAttention backends), and 3)
            UnfusedDotProductAttention which is the native PyTorch implementation
            with fused scaled masked softmax.

        .. note::

            Users can use environment variables :attr:`NVTE_FLASH_ATTN`, :attr:`NVTE_FUSED_ATTN`,
            and :attr:`NVTE_FUSED_ATTN_BACKEND` to control which DotProductAttention backend,
            and FusedAttention backend if applicable, to use. Transformer Engine prioritizes
            FlashAttention over FusedAttention and over UnfusedDotProductAttention.
            If FusedAttention is being used, users can also choose to switch to flash-attn's
            implementation for backward by setting :attr:`NVTE_FUSED_ATTN_USE_FAv2_BWD=1`
            (default: 0), because of the performance differences between various versions of
            flash-attn and FusedAttention. Further, :attr:`NVTE_FUSED_ATTN_FORCE_WORKSPACE_OPT`
            can be used to enable (:attr:`1`) or disable (:attr:`0`) the workspace related
            optimizations in FusedAttention. When unset, Transformer Engine determines the code path
            based on its internal logic. These optimizations trade memory for performance
            and should be used with care.

        .. note::
            .. _cu_seqlens note:

            When training data has variable sequence lengths, users have two options.

            1. Manipulate the data and pad all sequences to the same length. Use
               :attr:`qkv_format` = {"bshd", "sbhd"} and
               :attr:`attn_mask_type` = {"padding", "padding_causal", "padding_causal_bottom_right"}.
               Pass in :attr:`cu_seqlens_q` and :attr:`cu_seqlens_kv`, or :attr:`attention_mask`
               (which will be converted to :attr:`cu_seqlens_q` and :attr:`cu_seqlens_kv`), to provide
               the real sequence length information. For example, a batch of 3 sequences
               [a a a b b c c c c] can be padded to [a a a PAD b b PAD PAD c c c c], and the cumulative
               sequence length tensors would be
               :attr:`cu_seqlens_q` = :attr:`cu_seqlens_kv` = [0, 3, 5, 9] for self-attention.

            2. Do not perform padding on training data. Use :attr:`qkv_format` = "thd" and
               :attr:`attn_mask_type` = {"padding", "padding_causal", "padding_causal_bottom_right"}.
               Pass in :attr:`cu_seqlens_q` and :attr:`cu_seqlens_kv`, or :attr:`attention_mask`,
               as in option 1. For example, a batch of 3 sequences [a a a b b c c c c] can be processed
               without any padding, and the sequence length tensors would be
               :attr:`cu_seqlens_q` = :attr:`cu_seqlens_kv` = [0, 3, 5, 9] for self-attention.

               In certain use cases, a varying number of identifier tokens are inserted between
               sequences. These tokens do not participate in the attention calculation.
               :attr:`cu_seqlens_q_padded` and :attr:`cu_seqlens_kv_padded` must be specified
               in such cases to correctly identify the start and end of each sequence in a batch.
               For example, a batch of 3 sequences [a a a 1 b b 2 2 c c c c 3] would have
               :attr:`cu_seqlens_q` = :attr:`cu_seqlens_kv` = [0, 3, 5, 9], and
               :attr:`cu_seqlens_q_padded` = :attr:`cu_seqlens_kv_padded` = [0, 4, 8, 13]
               for self-attention.

        .. note::
            .. _max_seqlen note:

            When :attr:`qkv_format` = {"bshd", "sbhd"}, sequences are of equal length in a batch.
            :attr:`max_seqlen_q` and :attr:`max_seqlen_kv` should be the same as the "s" dimension of
            :attr:`query_layer` and :attr:`key_layer` tensors. When unset, Transformer Engine will
            infer them as such.

            When :attr:`qkv_format` = "thd", sequences have varying lengths. :attr:`max_seqlen_q` and
            :attr:`max_seqlen_kv` should be the maximum query and key/value sequence length in a batch.
            When unset, Transformer Engine deduces them from :attr:`cu_seqlens_q` and :attr:`cu_seqlens_kv`.
            This deduction costs a small kernel and some CPU-GPU synchronization, and to avoid this
            overhead, users are recommended to obtain the maximum sequence lengths from the data loaders
            and pass them in.

            - As the maximum sequence lengths, batch size, and number of tokens change from batch to batch,
              dynamic shapes need to be supported for tensor construction. FlashAttention and
              UnfusedDotProductAttention naturally do so, while FusedAttention requires parameters to be static
              to create graphs before performance heuristics analysis. To reduce the number of graphs created
              per run, Transformer Engine 1.13+ quantizes relevant parameters: for cuDNN < 9.6, {batch size,
              :attr:`max_seqlen_q`, :attr:`max_seqlen_kv`}, and for cuDNN >= 9.6, {"t" dimension of
              :attr:`query_layer`, "t" dimension of :attr:`key_layer`}.

        Parameters
        ----------
        query_layer : torch.Tensor
                     Query tensor.
        key_layer : torch.Tensor
                   Key tensor.
        value_layer : torch.Tensor
                     Value tensor.
        attention_mask: Optional[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]],
             default = `None`. Boolean tensor(s) used to mask out attention softmax input.
             It should be `None` for causal masks and "`no_mask`". For padding masks, it should be
             a single tensor of [batch_size, 1, 1, seqlen_q] for self-attention, and a tuple of
             two tensors in shapes [batch_size, 1, 1, seqlen_q] and [batch_size, 1, 1, seqlen_kv]
             for cross-attention. For "`arbitrary`" mask, it should be in a shape broadcastable
             to [batch_size, num_heads, max_seqlen_q, max_seqlen_kv]. A `True` value means
             the corresponding position is masked out and a `False` means that position
             is allowed to participate in attention.
        qkv_format: str, default = `None`
                   If provided, overrides :attr:`qkv_format` from initialization.
        cu_seqlens_q: Optional[torch.Tensor], default = `None`
                   Cumulative sum of sequence lengths (without offset) in a batch for `query_layer`,
                   with shape [batch_size + 1] and dtype torch.int32.
                   See :ref:`note<cu_seqlens note>` for more details.
        cu_seqlens_kv: Optional[torch.Tensor], default = `None`
                   Cumulative sum of sequence lengths (without offset) in a batch for `key_layer`
                   and `value_layer`, with shape [batch_size + 1] and dtype torch.int32.
                   See :ref:`note<cu_seqlens note>` for more details.
        cu_seqlens_q_padded: Optional[torch.Tensor], default = `None`
                   Cumulative sum of sequence lengths (with offset) in a batch for
                   `query_layer`, with shape [batch_size + 1] and dtype torch.int32.
                   When there is no padding between sequences in a batch,
                   `cu_seqlens_q_padded = cu_seqlens_q`.
                   See :ref:`note<cu_seqlens note>` for more details.
        cu_seqlens_kv_padded: Optional[torch.Tensor], default = `None`
                   Cumulative sum of sequence lengths (with offset) in a batch for `key_layer`
                   and `value_layer`, with shape [batch_size + 1] and dtype torch.int32.
                   When there is no padding between sequences in a batch,
                   `cu_seqlens_kv_padded = cu_seqlens_kv`.
                   See :ref:`note<cu_seqlens note>` for more details.
        max_seqlen_q: Optional[int], default = `None`
                      Maximum sequence length in `query_layer`.
                      See :ref:`note<max_seqlen note>` for more details.
        max_seqlen_kv: Optional[int], default = `None`
                       Maximum sequence length in `key_layer` and `value_layer`.
                       See :ref:`note<max_seqlen note>` for more details.
        attn_mask_type: {'no_mask', 'padding', 'causal', 'padding,causal', 'causal,padding',
                       'padding_causal', 'causal_bottom_right', 'padding_causal_bottom_right',
                       'arbitrary'}, default = `None`. Type of attention mask passed into
                       softmax operation. 'padding,causal', 'causal,padding' and 'padding_causal'
                       are equivalent. By default, causal masks are aligned to the top left corner
                       of the softmax matrix. When "`bottom_right`" is specified in the mask type,
                       causal masks are aligned to the bottom right corner.
        window_size: Optional[Tuple[int, int]], default = `None`
                    Sliding window size for local attention.
        checkpoint_core_attention : bool, default = `False`
                                   If true, forward activations for attention are recomputed
                                   during the backward pass in order to save memory that would
                                   otherwise be occupied to store the forward activations until
                                   backprop.
        core_attention_bias_type: str, default = `no_bias`
                    Bias type, {`no_bias`, `pre_scale_bias`, `post_scale_bias`, `alibi`}
        core_attention_bias: Optional[torch.Tensor], default = `None`
                    Bias tensor for Q * K.T, shape [1, num_head, max_seqlen_q, max_seqlen_kv].
                    It should be 'None' for 'no_bias' and 'alibi' bias types.
        alibi_slopes: Optional[torch.Tensor], default = `None`
                     ALiBi slopes in FP32 and shape [nheads] or [batch_size, nheads].
                     It adds a bias of (-alibi_slope * (i + seqlen_k - seqlen_q - j))
                     to the attention score of query i and key j.
        fast_zero_fill: bool, default = `True`
                    Whether to use the fast path to set output tensors to 0 or not.
        inference_params: Optional[InferenceParams], default = `None`
            Optimizes execution performance during inference by caching Keys and Values of the
            current decoding iteration. These cached values are appended to the K and V values
            computed in previous iterations, eliminating the need to recalculate them for the
            entire sequence.
            Initialization of `inference_params` is required prior to use to ensure sufficient
            memory allocation.
            Adjustments of the sequence_len_offset should be done after a complete forward pass.
            If rotary positional embeddings (RoPE) are utilized, they must be prepared beforehand.
            Supports "sbhd" and "bshd" layouts, with the "sbhd" layout being more efficient.
        pad_between_seqs: Optional[bool], default = `None`
            If None, inferred from qkv_format, cu_seqlens and cu_seqlens_padded.
            If true, there are padding tokens between individual sequences in a packed batch.
        """

        with self.prepare_forward(
            query_layer,
            num_gemms=3,
            allow_non_contiguous=True,
        ) as query_layer:
            # checks for RNG
            if self.rng_states_tracker is not None and is_graph_capturing():
                assert isinstance(
                    self.rng_states_tracker, CudaRNGStatesTracker
                ), "Unsupported RNG states tracker."
                assert (
                    graph_safe_rng_available()
                ), "Upgrade PyTorch version to get RNG manipulation support for cuda graph capture."

            # checks for FP8
            if self.fp8:
                if self.fp8_meta["recipe"].fp8_mha:
                    if not self.fp8_meta["recipe"].fp8_dpa:
                        self.fp8_meta["recipe"].fp8_dpa = True
                        self.logger.warning(
                            """Forcing fp8_meta["recipe"].fp8_dpa=True due to """
                            """fp8_meta["recipe"].fp8_mha=True"""
                        )
            if self.fp8 and self.fp8_meta["recipe"].fp8_dpa:
                forward_dtype = get_fp8_te_dtype(self.fp8_meta["recipe"], fprop_tensor=True)
                backward_dtype = get_fp8_te_dtype(self.fp8_meta["recipe"], fprop_tensor=False)
                assert forward_dtype in [
                    tex.DType.kFloat8E4M3,
                    tex.DType.kFloat8E5M2,
                ] and backward_dtype in [
                    tex.DType.kFloat8E4M3,
                    tex.DType.kFloat8E5M2,
                ], """DotProductAttention only supports "E4M3" and "E5M2" FP8 data types."""

            # checks for q/k/v shapes
            assert (
                query_layer.is_cuda and key_layer.is_cuda and value_layer.is_cuda
            ), "DotProductAttention only supports CUDA tensors."
            assert (
                query_layer.dtype == key_layer.dtype and query_layer.dtype == value_layer.dtype
            ), "Queries, keys and values must have the same data type!"
            assert (
                key_layer.shape[:-1] == value_layer.shape[:-1]
            ), "Keys and values must have the same batch size, sequence length and number of heads!"
            num_attention_heads = query_layer.shape[-2]
            num_gqa_groups = key_layer.shape[-2]
            assert (
                query_layer.shape[-1] == key_layer.shape[-1]
            ), "Queries and keys must have the same head dimension!"
            head_dim_qk, head_dim_v = query_layer.shape[-1], value_layer.shape[-1]
            assert (
                head_dim_qk == self.hidden_size_per_attention_head_k
            ), f"Keys have head_dim = {head_dim_qk}, "
            "but expected head_dim = {self.hidden_size_per_attention_head_k}!"
            assert (
                head_dim_v == self.hidden_size_per_attention_head_v
            ), f"Values have head_dim = {head_dim_v}, "
            "but expected head_dim = {self.hidden_size_per_attention_head_v}!"
            assert num_gqa_groups == self.num_gqa_groups_per_partition, (
                "Keys and values must have num_gqa_group ="
                f" {self.num_gqa_groups_per_partition} heads! Found {num_gqa_groups}."
            )

            # checks for attention mask
            if attn_mask_type is None:
                attn_mask_type = self.attn_mask_type
            else:
                attn_mask_type = attn_mask_type.replace(",", "_")
                if attn_mask_type == "causal_padding":
                    attn_mask_type = "padding_causal"
            assert (
                attn_mask_type in AttnMaskTypes
            ), f"Attention mask type {attn_mask_type} is not supported!"

            # checks for sliding window
            if window_size is None:
                window_size = self.window_size
            window_size = dpa_utils.check_set_window_size(attn_mask_type, window_size)

            # checks for qkv_format
            if qkv_format is None:
                qkv_format = self.qkv_format
            assert qkv_format in [
                "sbhd",
                "bshd",
                "thd",
            ], "DotProductAttention only supports qkv_format = {'sbhd', 'bshd', 'thd'}!"
            batch_size = None
            if qkv_format in ["sbhd", "bshd"]:
                assert all(
                    len(x.shape) == 4 for x in (query_layer, key_layer, value_layer)
                ), f"Queries, keys and values must be 4D tensors when {qkv_format=}!"
                if qkv_format == "sbhd":
                    batch_size = query_layer.shape[1]
                    max_seqlen_q = query_layer.shape[0] if max_seqlen_q is None else max_seqlen_q
                    max_seqlen_kv = key_layer.shape[0] if max_seqlen_kv is None else max_seqlen_kv
                else:
                    batch_size = query_layer.shape[0]
                    max_seqlen_q = query_layer.shape[1] if max_seqlen_q is None else max_seqlen_q
                    max_seqlen_kv = key_layer.shape[1] if max_seqlen_kv is None else max_seqlen_kv
            if qkv_format == "thd":
                assert all(
                    len(x.shape) == 3 for x in (query_layer, key_layer, value_layer)
                ), "Queries, keys and values must be 3D tensors when qkv_format = thd!"
                assert (
                    "padding" in attn_mask_type
                ), "Attention mask type must be padding or padding_causal for qkv_format=thd!"
                assert (
                    cu_seqlens_q is not None and cu_seqlens_kv is not None
                ), "cu_seqlens_q and cu_seqlens_kv can not be None when qkv_format = thd!"
                assert (
                    cu_seqlens_q.shape == cu_seqlens_kv.shape
                    and len(cu_seqlens_q.shape) == 1
                    and len(cu_seqlens_kv.shape) == 1
                ), "cu_seqlens_q and cu_seqlens_q must both have shape [batch_size + 1]!"
                assert (
                    cu_seqlens_q.dtype == torch.int32 and cu_seqlens_kv.dtype == torch.int32
                ), "cu_seqlens_q and cu_seqlens_q must both be in dtype torch.int32!"
                batch_size = len(cu_seqlens_q) - 1
                if max_seqlen_q is None:
                    if cu_seqlens_q_padded is not None:
                        seqlens_q = cu_seqlens_q_padded[1:] - cu_seqlens_q_padded[:-1]
                    else:
                        seqlens_q = cu_seqlens_q[1:] - cu_seqlens_q[:-1]
                    max_seqlen_q = int((seqlens_q.max().item() + 63) // 64 * 64)
                if max_seqlen_kv is None:
                    if cu_seqlens_kv_padded is not None:
                        seqlens_kv = cu_seqlens_kv_padded[1:] - cu_seqlens_kv_padded[:-1]
                    else:
                        seqlens_kv = cu_seqlens_kv[1:] - cu_seqlens_kv[:-1]
                    max_seqlen_kv = int((seqlens_kv.max().item() + 63) // 64 * 64)

            # update KV cache and retrieve saved tokens from cache for inference
            if inference_params is not None:
                assert self.layer_number is not None, "Layer number must be set!"

                # convert top-left causal to bottom-right causal due to KV caching
                # users can still use the same attention mask for inference as for training
                assert "padding" in attn_mask_type, "KV caching requires padding mask!"
                if attn_mask_type == "padding_causal":
                    attn_mask_type = attn_mask_type + "_bottom_right"

                self.attention_type = "cross"
                self.flash_attention.attention_type = self.attention_type
                self.fused_attention.attention_type = self.attention_type
                self.unfused_attention.attention_type = self.attention_type

                query_layer, key_layer, value_layer = [
                    x.contiguous() if not x.is_contiguous() else x
                    for x in [query_layer, key_layer, value_layer]
                ]

                # get full K/V tensors from cache and adjust cu_seqlens, qkv_format based on the cache
                (
                    key_layer,
                    value_layer,
                    cu_seqlens_q,
                    cu_seqlens_kv,
                    max_seqlen_kv,
                    qkv_format,
                ) = inference_params.step(
                    self.layer_number,
                    key_layer,
                    value_layer,
                    qkv_format,
                )
                cu_seqlens_q_padded = None
                cu_seqlens_kv_padded = None

            # get qkv's memory layout
            if all(isinstance(x, Float8Tensor) for x in [query_layer, key_layer, value_layer]):
                (
                    qkv_layout,
                    query_layer._data,
                    key_layer._data,
                    value_layer._data,
                    q_format,
                    kv_format,
                ) = dpa_utils.get_qkv_layout(
                    query_layer._data,
                    key_layer._data,
                    value_layer._data,
                    qkv_format=qkv_format,
                    inference_params=inference_params,
                )
            else:
                (
                    qkv_layout,
                    query_layer,
                    key_layer,
                    value_layer,
                    q_format,
                    kv_format,
                ) = dpa_utils.get_qkv_layout(
                    query_layer,
                    key_layer,
                    value_layer,
                    qkv_format=qkv_format,
                    inference_params=inference_params,
                )

            # adjust max_seqlen and cu_seqlens for CP
            cp_size = 1
            if isinstance(self.cp_group, dist_group_type):
                cp_size = get_distributed_world_size(self.cp_group)
            elif isinstance(self.cp_group, list):
                for group in self.cp_group:
                    cp_size *= get_distributed_world_size(group)
            context_parallel = cp_size > 1
            if q_format in ["sbhd", "bshd"]:
                max_seqlen_q *= cp_size
                if cu_seqlens_q is None:
                    if "padding" in attn_mask_type:
                        assert (
                            attention_mask is not None
                        ), "Please provide attention_mask for padding!"
                        if self.attention_type == "self":
                            cu_seqlens_q = dpa_utils.get_cu_seqlens(attention_mask)
                        else:
                            cu_seqlens_q = dpa_utils.get_cu_seqlens(attention_mask[0])
                    else:
                        cu_seqlens_q = dpa_utils.get_full_cu_seqlens(
                            batch_size,
                            max_seqlen_q,
                            query_layer.device,
                        )
            if kv_format in ["sbhd", "bshd"]:
                max_seqlen_kv *= cp_size
                if cu_seqlens_kv is None:
                    if "padding" in attn_mask_type:
                        assert (
                            attention_mask is not None
                        ), "Please provide attention_mask for padding!"
                        if self.attention_type == "self":
                            cu_seqlens_kv = dpa_utils.get_cu_seqlens(attention_mask)
                        else:
                            cu_seqlens_kv = dpa_utils.get_cu_seqlens(attention_mask[1])
                    else:
                        cu_seqlens_kv = dpa_utils.get_full_cu_seqlens(
                            batch_size,
                            max_seqlen_kv,
                            key_layer.device,
                        )

            # set ALiBi attributes
            global _alibi_cache
            if alibi_slopes is not None:
                assert (
                    core_attention_bias_type == "alibi"
                ), "core_attention_bias_type must be alibi in order to use alibi_slopes!"
                if self.layer_number == 1:
                    _alibi_cache["_alibi_slopes_require_update"] = True
                    _alibi_cache["_alibi_bias_require_update"] = True
            bottom_right_alignment = (attn_mask_type not in ["causal", "padding_causal"],)
            if core_attention_bias_type == "alibi":
                assert (
                    core_attention_bias is None
                ), "core_attention_bias must be None when core_attention_bias_type is alibi!"
                if (
                    _alibi_cache["_num_heads"] != query_layer.shape[-2]
                    or _alibi_cache["_max_seqlen_q"] != max_seqlen_q
                    or _alibi_cache["_max_seqlen_kv"] != max_seqlen_kv
                    or _alibi_cache["_bottom_right_alignment"] != bottom_right_alignment
                    or _alibi_cache["_alibi_slopes"] is None
                ):
                    _alibi_cache["_alibi_slopes_require_update"] = True
                    _alibi_cache["_alibi_bias_require_update"] = True

            # detect bias shape
            core_attention_bias_shape = None
            if core_attention_bias is not None:
                if (
                    core_attention_bias.shape[0] == batch_size
                    and core_attention_bias.shape[1] == query_layer.shape[-2]
                ):
                    core_attention_bias_shape = "bhss"
                elif (
                    core_attention_bias.shape[0] == 1
                    and core_attention_bias.shape[1] == query_layer.shape[-2]
                ):
                    core_attention_bias_shape = "1hss"
                elif (
                    core_attention_bias.shape[0] == batch_size and core_attention_bias.shape[1] == 1
                ):
                    core_attention_bias_shape = "b1ss"
                elif core_attention_bias.shape[0] == 1 and core_attention_bias.shape[1] == 1:
                    core_attention_bias_shape = "11ss"
                else:
                    assert (
                        False
                    ), "core_attention_bias must be in one of {bhss, 1hss, b1ss, 11ss} shapes"

            if pad_between_seqs is None:
                if qkv_format == "thd":
                    pad_between_seqs = (
                        cu_seqlens_q_padded is not None
                        and not torch.equal(cu_seqlens_q_padded[:-1], cu_seqlens_q[:-1])
                    ) or (
                        cu_seqlens_kv_padded is not None
                        and not torch.equal(cu_seqlens_kv_padded[:-1], cu_seqlens_kv[:-1])
                    )
                else:
                    pad_between_seqs = False

            # gather attention params for get_attention_backend
            attention_params = dpa_utils.AttentionParams(
                qkv_type=type(query_layer),
                qkv_dtype=query_layer.dtype,
                qkv_layout=qkv_layout,
                batch_size=batch_size,
                num_heads=num_attention_heads,
                num_gqa_groups=num_gqa_groups,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_kv=max_seqlen_kv,
                head_dim_qk=head_dim_qk,
                head_dim_v=head_dim_v,
                attn_mask_type=attn_mask_type,
                window_size=window_size,
                alibi_slopes_shape=alibi_slopes.shape if alibi_slopes is not None else None,
                core_attention_bias_type=core_attention_bias_type,
                core_attention_bias_shape=core_attention_bias_shape,
                core_attention_bias_requires_grad=(
                    core_attention_bias.requires_grad if core_attention_bias is not None else False
                ),
                pad_between_seqs=pad_between_seqs,
                attention_dropout=self.attention_dropout,
                context_parallel=context_parallel,
                deterministic=self.deterministic,
                is_training=self.training,
                fp8=self.fp8,
                fp8_meta=self.fp8_meta,
                inference_params=inference_params,
            )
            global _attention_backends
            if (
                _attention_backends["attention_params"] is None
                or attention_params != _attention_backends["attention_params"]
            ):
                _attention_backends["attention_params"] = attention_params
                _attention_backends["backend_selection_requires_update"] = True
            if _attention_backends["backend_selection_requires_update"]:
                (
                    use_flash_attention,
                    flash_attention_backend,
                    use_fused_attention,
                    fused_attention_backend,
                    use_unfused_attention,
                    _,
                ) = dpa_utils.get_attention_backend(attention_params)
                # Set global _attention_backends var using return value
                # from get_attention_backend()
                _attention_backends["use_flash_attention"] = use_flash_attention
                _attention_backends["flash_attention_backend"] = flash_attention_backend
                _attention_backends["use_fused_attention"] = use_fused_attention
                _attention_backends["fused_attention_backend"] = fused_attention_backend
                _attention_backends["use_unfused_attention"] = use_unfused_attention
                _attention_backends["backend_selection_requires_update"] = False
                if use_flash_attention:
                    self.logger.info(
                        "Running with FlashAttention backend (version %s)",
                        flash_attention_backend,
                    )
                elif use_fused_attention:
                    self.logger.info(
                        "Running with FusedAttention backend (sub-backend %s)",
                        int(fused_attention_backend),
                    )
                elif use_unfused_attention:
                    self.logger.info("Running with UnfusedDotProductAttention backend")
            else:
                use_flash_attention = _attention_backends["use_flash_attention"]
                flash_attention_backend = _attention_backends["flash_attention_backend"]
                use_fused_attention = _attention_backends["use_fused_attention"]
                fused_attention_backend = _attention_backends["fused_attention_backend"]
                use_unfused_attention = _attention_backends["use_unfused_attention"]

            # raise exception if no backend is available
            if sum([use_flash_attention, use_fused_attention, use_unfused_attention]) == 0:
                raise ValueError(
                    "No dot product attention backend is available for the provided inputs. Please"
                    " run with NVTE_DEBUG=1 NVTE_DEBUG_LEVEL=2 to find out the reasons for"
                    " disabling all backends."
                )

            # run attention
            if use_flash_attention:
                if core_attention_bias_type == "alibi":
                    alibi_slopes, _ = dpa_utils.get_alibi(
                        _alibi_cache,
                        query_layer.shape[-2],
                        max_seqlen_q,
                        max_seqlen_kv,
                        alibi_slopes=alibi_slopes,
                    )
                return self.flash_attention(
                    query_layer,
                    key_layer,
                    value_layer,
                    attention_mask=attention_mask,
                    qkv_layout=qkv_layout,
                    cu_seqlens_q=cu_seqlens_q,
                    cu_seqlens_kv=cu_seqlens_kv,
                    attn_mask_type=attn_mask_type,
                    window_size=window_size,
                    alibi_slopes=alibi_slopes,
                    cp_group=self.cp_group,
                    cp_global_ranks=self.cp_global_ranks,
                    cp_stream=self.cp_stream,
                    cp_comm_type=self.cp_comm_type,
                    max_seqlen_q=max_seqlen_q,
                    max_seqlen_kv=max_seqlen_kv,
                    fp8=self.fp8 and self.fp8_meta["recipe"].fp8_dpa,
                    fp8_meta=self.fp8_meta,
                    quantizers=self.quantizers,
                    inference_params=inference_params,
                    flash_attention_backend=flash_attention_backend,
                )

            if use_fused_attention:
                fu_core_attention_bias_type = core_attention_bias_type
                fu_core_attention_bias = core_attention_bias
                if core_attention_bias_type == "alibi" and (
                    alibi_slopes is not None or max_seqlen_q != max_seqlen_kv
                ):
                    fu_core_attention_bias_type = "post_scale_bias"
                    _, fu_core_attention_bias = dpa_utils.get_alibi(
                        _alibi_cache,
                        query_layer.shape[-2],
                        max_seqlen_q,
                        max_seqlen_kv,
                        alibi_slopes=alibi_slopes,
                        bias_dtype=query_layer.dtype,
                        bottom_right_alignment=attn_mask_type not in ["causal", "padding_causal"],
                    )
                # checkpoint_core_attention=False
                if checkpoint_core_attention:
                    return self._checkpointed_attention_forward(
                        self.fused_attention,
                        query_layer,
                        key_layer,
                        value_layer,
                        qkv_layout=qkv_layout,
                        cu_seqlens_q=cu_seqlens_q,
                        cu_seqlens_kv=cu_seqlens_kv,
                        cu_seqlens_q_padded=cu_seqlens_q_padded,
                        cu_seqlens_kv_padded=cu_seqlens_kv_padded,
                        max_seqlen_q=max_seqlen_q,
                        max_seqlen_kv=max_seqlen_kv,
                        attn_mask_type=attn_mask_type,
                        attention_mask=attention_mask,
                        window_size=window_size,
                        fused_attention_backend=fused_attention_backend,
                        core_attention_bias_type=fu_core_attention_bias_type,
                        core_attention_bias=fu_core_attention_bias,
                        fast_zero_fill=fast_zero_fill,
                        cp_group=self.cp_group,
                        cp_global_ranks=self.cp_global_ranks,
                        cp_stream=self.cp_stream,
                        cp_comm_type=self.cp_comm_type,
                        fp8=self.fp8 and self.fp8_meta["recipe"].fp8_dpa,
                        fp8_meta=self.fp8_meta,
                        quantizers=self.quantizers,
                        pad_between_seqs=pad_between_seqs,
                        inference_params=inference_params,
                    )
                return self.fused_attention(
                    query_layer,
                    key_layer,
                    value_layer,
                    qkv_layout=qkv_layout,
                    cu_seqlens_q=cu_seqlens_q,
                    cu_seqlens_kv=cu_seqlens_kv,
                    cu_seqlens_q_padded=cu_seqlens_q_padded,
                    cu_seqlens_kv_padded=cu_seqlens_kv_padded,
                    max_seqlen_q=max_seqlen_q,
                    max_seqlen_kv=max_seqlen_kv,
                    attn_mask_type=attn_mask_type,
                    attention_mask=attention_mask,
                    window_size=window_size,
                    fused_attention_backend=fused_attention_backend,
                    core_attention_bias_type=fu_core_attention_bias_type,
                    core_attention_bias=fu_core_attention_bias,
                    fast_zero_fill=fast_zero_fill,
                    cp_group=self.cp_group,
                    cp_global_ranks=self.cp_global_ranks,
                    cp_stream=self.cp_stream,
                    cp_comm_type=self.cp_comm_type,
                    fp8=self.fp8 and self.fp8_meta["recipe"].fp8_dpa,
                    fp8_meta=self.fp8_meta,
                    quantizers=self.quantizers,
                    pad_between_seqs=pad_between_seqs,
                    inference_params=inference_params,
                )

            from .cpu_offload import CPUOffloadEnabled

            if CPUOffloadEnabled:
                warnings.warn(
                    "Attention activation Offloading is only implemented"
                    "with Flash Attention and Fused Attention!"
                )

            if use_unfused_attention:
                if checkpoint_core_attention:
                    return self._checkpointed_attention_forward(
                        self.unfused_attention,
                        query_layer,
                        key_layer,
                        value_layer,
                        qkv_layout=qkv_layout,
                        cu_seqlens_q=cu_seqlens_q,
                        cu_seqlens_kv=cu_seqlens_kv,
                        attn_mask_type=attn_mask_type,
                        attention_mask=attention_mask,
                        window_size=window_size,
                        core_attention_bias_type=core_attention_bias_type,
                        core_attention_bias=core_attention_bias,
                        alibi_slopes=alibi_slopes,
                        inference_params=inference_params,
                    )
                return self.unfused_attention(
                    query_layer,
                    key_layer,
                    value_layer,
                    qkv_layout=qkv_layout,
                    cu_seqlens_q=cu_seqlens_q,
                    cu_seqlens_kv=cu_seqlens_kv,
                    attn_mask_type=attn_mask_type,
                    attention_mask=attention_mask,
                    window_size=window_size,
                    core_attention_bias_type=core_attention_bias_type,
                    core_attention_bias=core_attention_bias,
                    alibi_slopes=alibi_slopes,
                    inference_params=inference_params,
                )
            return None
