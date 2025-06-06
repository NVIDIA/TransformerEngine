# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Attention Backends."""
from contextlib import nullcontext
from importlib.metadata import version as get_pkg_version
from importlib.metadata import PackageNotFoundError
import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import warnings
import logging
from packaging.version import Version as PkgVersion

import torch
import transformer_engine_torch as tex
from transformer_engine.pytorch.utils import (
    SplitAlongDim,
    get_device_compute_capability,
    combine_tensors,
    split_tensor_along_dim,
)
from transformer_engine.pytorch.utils import attention_mask_func
from transformer_engine.pytorch.tensor.quantized_tensor import (
    QuantizedTensor,
    prepare_for_saving,
    restore_from_saved,
)
from transformer_engine.pytorch.float8_tensor import Float8Tensor
from transformer_engine.pytorch.constants import (
    TE_DType,
    QKVLayouts,
    dist_group_type,
)
from transformer_engine.pytorch.cpp_extensions.fused_attn import (
    fused_attn_fwd,
    fused_attn_bwd,
    FusedAttnBackend,
    META_O,
    META_QKV,
)
from transformer_engine.pytorch.fp8 import get_fp8_torch_dtype
from transformer_engine.pytorch.distributed import get_distributed_world_size
from transformer_engine.pytorch.jit import no_torch_dynamo
from transformer_engine.pytorch.attention.dot_product_attention.context_parallel import (
    attn_forward_func_with_cp,
)
from transformer_engine.pytorch.attention.dot_product_attention.softmax import FusedScaleMaskSoftmax
from transformer_engine.pytorch.attention.inference import InferenceParams

# Import attention utils
import transformer_engine.pytorch.attention.dot_product_attention.utils as dpa_utils
from transformer_engine.pytorch.attention.dot_product_attention.utils import (
    FlashAttentionUtils as fa_utils,
)
from transformer_engine.pytorch.attention.dot_product_attention.utils import (
    AttentionLogging as attn_log,
)

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
    flash_attn_func_v3 = None
    flash_attn_varlen_func_v3 = None
    flash_attn_with_kvcache_v3 = None
    # pass  # only print warning if use_flash_attention_3 = True in get_attention_backend
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
        _alibi_cache: Dict[str, Any],
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
                batch_size,
                cu_seqlens_q,
                cu_seqlens_kv,
                max_seqlen_q,
                max_seqlen_kv,
                self.attention_type,
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
        if attn_log._is_logging_setup is False:
            attn_log.setup_logging()
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
            from transformer_engine.pytorch.cpu_offload import (
                CPUOffloadEnabled,
                mark_activation_offload,
            )

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
                        qkv = combine_tensors([q, k, v], dim)
                        qkv_c = qkv.view(-1, qkv.shape[-3] * qkv.shape[-2] * qkv.shape[-1])
                        qkv_fp8 = QKV_quantizer(qkv)
                        q_fp8, k_fp8, v_fp8 = SplitAlongDim.apply(qkv_fp8, dim, [1, 1, 1], True)
                    case 2:
                        q_fp8 = QKV_quantizer(q)
                        dim = qkv_layout.split("_")[1].find("2")
                        kv = combine_tensors([k, v], dim)
                        kv_c = kv.view(-1, kv.shape[-3] * kv.shape[-2] * kv.shape[-1])
                        kv_fp8 = QKV_quantizer(kv_c)
                        k_fp8, v_fp8 = SplitAlongDim.apply(kv_fp8, dim, [1, 1], True)
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
                        qkv = combine_tensors([q, k, v], dim)
                        qkv_c = qkv.view(-1, qkv.shape[-3] * qkv.shape[-2] * qkv.shape[-1])
                        qkv_no_fp8 = qkv_c.dequantize().view(qkv.shape)
                        q, k, v = SplitAlongDim.apply(qkv_no_fp8, dim, [1, 1, 1], True)
                    if qkv_group == 2:
                        q = q.dequantize()
                        dim = qkv_layout.replace("paged_kv_", "").split("_")[1].find("2")
                        kv = combine_tensors([k, v], dim)
                        kv_c = kv.view(-1, kv.shape[-3] * kv.shape[-2] * kv.shape[-1])
                        kv_no_fp8 = kv.dequantize()
                        k, v = SplitAlongDim.apply(kv_no_fp8, dim, [1, 1], True)
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

        from transformer_engine.pytorch.cpu_offload import (
            CPUOffloadEnabled,
            mark_activation_offload,
        )

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
            d_out, q, k, v, out = [dpa_utils.maybe_contiguous(x) for x in (d_out, q, k, v, out)]
            # from transformer_engine.pytorch.attention.dot_product_attention import flash_attn_cuda_bwd
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
                            dqkv_fp8_data = combine_tensors(
                                [dq_fp8._data, dk_fp8._data, dv_fp8._data], dim
                            )
                            dqkv_fp8 = dq_fp8.make_like(
                                tensor=dq_fp8, data=dqkv_fp8_data, shape=dqkv_fp8_data.shape
                            )
                            dqkv = dqkv_fp8.dequantize()
                            dq, dk, dv = SplitAlongDim.apply(dqkv, dim, [1, 1, 1], True)
                        if qkv_group == 2:
                            dq = dq_fp8.dequantize()
                            dim = ctx.qkv_layout.split("_")[1].find("2")
                            dkv_fp8 = combine_tensors([dk_fp8, dv_fp8], dim)
                            dkv_c_fp8 = dkv_fp8.view(
                                -1, dkv_fp8.shape[-3] * dkv_fp8.shape[-2] * dkv_fp8.shape[-1]
                            )
                            dkv = dkv_c_fp8.dequantize()
                            dk, dv = SplitAlongDim.apply(dkv, dim, [1, 1], True)
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
