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
from itertools import count

import torch
import transformer_engine_torch as tex
from transformer_engine.pytorch.utils import (
    get_device_compute_capability,
    split_tensor_along_dim,
)
from transformer_engine.pytorch.utils import attention_mask_func, nvtx_range_push, nvtx_range_pop
from transformer_engine.pytorch.tensor._internal.float8_tensor_base import Float8TensorBase
from transformer_engine.pytorch.tensor.float8_tensor import (
    Float8Quantizer,
    Float8CurrentScalingQuantizer,
)
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
from transformer_engine.pytorch.fp8 import get_fp8_torch_dtype, FP8GlobalStateManager
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
    combine_and_quantize,
    combine_and_dequantize,
)
from transformer_engine.pytorch.attention.dot_product_attention.utils import (
    AttentionLogging as attn_log,
)
from transformer_engine.pytorch import export
from transformer_engine.pytorch.export import is_in_onnx_export_mode

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


def _rmse(a, b):
    return torch.sqrt((torch.pow((a - b), 2) / a.numel()).sum())


global_fwd_count = count(0, 1)
global_bwd_count = count(0, 1)

import random


def set_random_seed(seed: int):
    """Set random seed for Python random and PyTorch."""
    # Set Python random seed
    random.seed(seed)
    # Set PyTorch seed
    torch.manual_seed(seed)
    # Set CUDA seed
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Deterministic CuDNN behavior
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def generate_random_diag(n: int, seed: int | None = None) -> torch.Tensor:
    """Generate a random diagonal matrix of size n. Values are -1 or 1."""
    if seed is not None:
        set_random_seed(seed)

    # Generate randomized diagonal matrix
    S = torch.zeros(n, n)
    diag_values = 2 * torch.randint(0, 2, (n,)) - 1  # Random -1 or 1
    S.diagonal().copy_(diag_values)

    assert torch.all(torch.abs(S.diagonal()) == 1)

    return S


def hadamard_matrix(n: int) -> torch.Tensor:
    """Generate NxN Hadamard matrix"""
    if n == 1:
        return torch.tensor([[1]])
    else:
        # Check that n is a power of 2
        assert (n & (n - 1)) == 0

        H_prev = hadamard_matrix(n // 2)
        H = torch.cat(
            [torch.cat([H_prev, H_prev], dim=1), torch.cat([H_prev, -H_prev], dim=1)],
            dim=0,
        )
        H = H / torch.sqrt(torch.tensor(2))
        return H


def generate_rht(n: int, seed: int | None = None) -> torch.Tensor:
    """Generate NxN Randomized Hadamard Transform tensor"""
    # Check that n is a power of 2
    assert (n & (n - 1)) == 0
    assert n > 1

    # Generate Hadamard matrix
    H = hadamard_matrix(n)

    # Generate randomized diagonal matrix
    S = generate_random_diag(n, seed)

    # Multiply Hadamard matrix by randomized diagonal matrix
    RHT = torch.matmul(S, H)

    return RHT


def apply_rht(x: torch.Tensor, rht: torch.Tensor) -> torch.Tensor:
    # Check that RHT is square
    assert rht.ndim == 2
    assert rht.shape[0] == rht.shape[1]

    orig_shape = x.shape
    x = torch.reshape(x, (-1, rht.shape[0]))
    x = torch.matmul(x, rht)
    x = torch.reshape(x, orig_shape)
    return x


def undo_rht(x: torch.Tensor, rht: torch.Tensor) -> torch.Tensor:
    # Check that RHT is square
    assert rht.ndim == 2
    assert rht.shape[0] == rht.shape[1]

    orig_shape = x.shape
    x = torch.reshape(x, (-1, rht.shape[1]))
    x = torch.matmul(x, rht.t())
    x = torch.reshape(x, orig_shape)
    return x


class _UnfusedDPAQuantizationEmulator(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor1, tensor2, tensor3, quantizer, quantizer_name, qkv_layout):
        if quantizer_name == "QKV_quantizer":
            if isinstance(quantizer, MXFP8Quantizer):
                tensors_q = [quantizer(x).dequantize() for x in [tensor1, tensor2, tensor3]]
            else:
                query_layer, key_layer, value_layer = tensor1, tensor2, tensor3
                query_layer, key_layer, value_layer = [
                    x.contiguous() for x in [query_layer, key_layer, value_layer]
                ]
                q_fp8, k_fp8, v_fp8 = combine_and_quantize(
                    qkv_layout, query_layer, key_layer, value_layer, quantizer
                )
                tensors_q = combine_and_dequantize(qkv_layout, q_fp8, k_fp8, v_fp8)
        elif quantizer_name == "S_quantizer":
            s_fp8 = quantizer(tensor1)
            tensors_q = (s_fp8.dequantize(), tensor2, tensor3)
        else:
            tensors_q = (tensor1, tensor2, tensor3)
        ctx.quantizer = quantizer
        ctx.quantizer_name = quantizer_name
        return tensors_q[0], tensors_q[1], tensors_q[2]

    @staticmethod
    def backward(ctx, grad1, grad2, grad3):
        if ctx.quantizer_name == "dP_quantizer":
            dp_fp8 = ctx.quantizer(grad1)
            tensors_q = dp_fp8.dequantize(), grad2, grad3
        elif ctx.quantizer_name == "dO_quantizer":
            do_fp8 = ctx.quantizer(grad1)
            tensors_q = do_fp8.dequantize(), grad2, grad3
        else:
            tensors_q = grad1, grad2, grad3
        return tensors_q[0], tensors_q[1], tensors_q[2], None, None, None


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

        def mask_func(x, y):
            return (
                export.onnx_attention_mask_func(x, y)
                if is_in_onnx_export_mode()
                else attention_mask_func(x, y)
            )

        self.scale_mask_softmax = FusedScaleMaskSoftmax(mask_func)

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
        fp8: bool = False,
        fp8_meta: Optional[Dict[str, Any]] = None,
        quantizers=None,
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

        # get quantizers from DPA; all Nones if not fp8
        QKV_quantizer, O_quantizer, S_quantizer, dQKV_quantizer, dO_quantizer, dP_quantizer = (
            dpa_utils.get_attention_quantizers(
                fp8, fp8_meta, quantizers, cp_specific_quantizers=False
            )
        )
        S_quantizer = Float8CurrentScalingQuantizer(fp8_dtype=S_quantizer.dtype, device="cuda")
        dP_quantizer = Float8CurrentScalingQuantizer(fp8_dtype=dP_quantizer.dtype, device="cuda")

        # quantize and dequantize q,k,v to simulate CS
        query_layer, key_layer, value_layer = _UnfusedDPAQuantizationEmulator.apply(
            query_layer, key_layer, value_layer, QKV_quantizer, "QKV_quantizer", qkv_layout
        )

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

        # quantize and dequantize dP to simulate CS
        matmul_result, *_ = _UnfusedDPAQuantizationEmulator.apply(
            matmul_result, None, None, dP_quantizer, "dP_quantizer", None
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

        # quantize and dequantize S to simulate CS
        attention_probs, *_ = _UnfusedDPAQuantizationEmulator.apply(
            attention_probs, None, None, S_quantizer, "S_quantizer", None
        )

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

        # quantize and dequantize dO to simulate CS
        attention_probs, *_ = _UnfusedDPAQuantizationEmulator.apply(
            attention_probs, None, None, dO_quantizer, "dO_quantizer", None
        )

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
        fp8_output: bool = False,
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
                    fp8_output=False,
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
                    if fp8 and fp8_output:
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
            if fp8 and fp8_output:
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
    """FusedAttention forward and backward implementation"""

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
        fp8_output,
        layer_number,
    ):
        # pylint: disable=missing-function-docstring

        # add NVTX range
        nvtx_label = "transformer_engine.FusedAttnFunc.forward"
        nvtx_range_push(f"{nvtx_label}")

        # input types are inferred from the real data while output types are controlled by fp8_output
        # fp8_output should be set upstream as (DPA.fp8 and DPA.fp8_meta["recipe"].fp8_mha)
        assert isinstance(k, q.__class__) and isinstance(
            v, q.__class__
        ), "q, k, v must be of the same class, e.g. torch.Tensor or Float8Tensor."
        is_input_fp8 = isinstance(q, Float8Tensor)
        is_output_fp8 = fp8_output

        # whether fwd kernel in FP8: fp8 = (DPA.fp8 and DPA.fp8_meta["recipe"].fp8_dpa)
        # whether bwd kernel in FP8:
        is_bwd_fp8 = fp8 and int(os.getenv("NVTE_FP8_DPA_BWD", "1"))

        # get quantizers from DPA; all Nones if not fp8
        QKV_quantizer, O_quantizer, S_quantizer, dQKV_quantizer, dO_quantizer, dP_quantizer = (
            dpa_utils.get_attention_quantizers(
                fp8, fp8_meta, quantizers, cp_specific_quantizers=False
            )
        )

        # get nominal data type for out
        # FP16/BF16 attention: torch.float16 or torch.bfloat16
        # FP8 attention:       torch.float16 or torch.bfloat16
        out_nominal_dtype = q.dtype

        aux_ctx_tensors_clone = []
        if fp8:
            fused_attention_backend = FusedAttnBackend["FP8"]

            rht_size = int(os.getenv("NVTE_RHT_BMM1", "0"))
            if rht_size > 0:
                rht = generate_rht(rht_size, seed=1234).cuda().to(torch.bfloat16)
                q, k = [apply_rht(x, rht) for x in [q, k]]

            # q, k, v:             torch.Tensor; dtype = torch.float16 or torch.bfloat16
            # q_fp8, k_fp8, v_fp8: Float8Tensor; dtype = torch.float16 or torch.bfloat16
            #                                    fp8_dtype = tex.DType.kFloat8E4M3
            if is_input_fp8:
                q_fp8, k_fp8, v_fp8 = q, k, v
            else:
                q_fp8, k_fp8, v_fp8 = combine_and_quantize(qkv_layout, q, k, v, QKV_quantizer)

            if bool(int(os.getenv("NVTE_Emulate_in_F16", "0"))):
                if bool(int(os.getenv("NVTE_Emulate_QDQ_QKV", "0"))):
                    q_deq16, k_deq16, v_deq16 = combine_and_dequantize(
                        qkv_layout, q_fp8, k_fp8, v_fp8, src_nominal_dtype=out_nominal_dtype
                    )
                else:
                    q_deq16, k_deq16, v_deq16 = q, k, v
                out_, aux_ctx_tensors = fused_attn_fwd(
                    is_training,
                    max_seqlen_q,
                    max_seqlen_kv,
                    cu_seqlens_q,
                    cu_seqlens_kv,
                    q_deq16,
                    k_deq16,
                    v_deq16,
                    out_nominal_dtype,
                    FusedAttnBackend["F16_arbitrary_seqlen"],
                    attn_bias,
                    cu_seqlens_q_padded,
                    cu_seqlens_kv_padded,
                    None,
                    None,
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
                if bool(int(os.getenv("NVTE_PRINT", "0"))):
                    layer = int(os.getenv("NVTE_LAYER_NUMBER", str(layer_number)))
                    procid = int(os.getenv("SLURM_PROCID", "0"))
                    if layer_number == layer and procid == 0:
                        print(f">>>>>>>>>>>>>>>>>> fwd emulate16")
                        torch.cuda.synchronize()
                        t_in = [q, k, v]
                        t_f8 = [out_, aux_ctx_tensors[0], aux_ctx_tensors[1]]
                        t_f16 = [q_deq16, k_deq16, v_deq16]
                        rmse = [f"{_rmse(x,y).item():.4e}" for x, y in zip(t_in, t_f16)]
                        tin_minmax = [(x.min().item(), x.max().item()) for x in t_in]
                        t8_minmax = [(x.min().item(), x.max().item()) for x in t_f8]
                        t16_minmax = [(x.min().item(), x.max().item()) for x in t_f16]
                        tin_minmax_strs = [f"({mi:.4e},{ma:.4e})" for (mi, ma) in tin_minmax]
                        t8_minmax_strs = [f"({mi:.4e},{ma:.4e})" for (mi, ma) in t8_minmax]
                        t16_minmax_strs = [f"({mi:.4e},{ma:.4e})" for (mi, ma) in t16_minmax]
                        names_minmax = ["qkv        ", "emu-qkv    ", "emu-o/stats"]
                        for nm, mm in zip(
                            names_minmax, [tin_minmax_strs, t16_minmax_strs, t8_minmax_strs]
                        ):
                            mm = ", ".join(mm)
                            print(f">>>> fwd p{procid} l{layer_number} {nm}: {mm}")
                        print(f">>>> fwd p{procid} l{layer_number} rmse_qkv   : {", ".join(rmse)}")
            else:
                # out_:
                # DelayedScaling:       Float8Tensor; dtype = torch.float16 or torch.bfloat16
                #                                     fp8_dtype = tex.DType.kFloat8E4M3
                # Float8CurrentScaling: torch.Tensor; dtype = torch.float16 or torch.bfloat16
                out_, aux_ctx_tensors = fused_attn_fwd(
                    is_training,
                    max_seqlen_q,
                    max_seqlen_kv,
                    cu_seqlens_q,
                    cu_seqlens_kv,
                    q_fp8,
                    k_fp8,
                    v_fp8,
                    out_nominal_dtype,
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
            # repeat FP8 in F16
            if bool(int(os.getenv("NVTE_REPEAT_in_F16", "0"))):
                q_clone, k_clone, v_clone = [x.detach().clone() for x in [q, k, v]]
                out_clone, aux_ctx_tensors_clone = fused_attn_fwd(
                    is_training,
                    max_seqlen_q,
                    max_seqlen_kv,
                    cu_seqlens_q,
                    cu_seqlens_kv,
                    q_clone,
                    k_clone,
                    v_clone,
                    out_nominal_dtype,
                    FusedAttnBackend["F16_arbitrary_seqlen"],
                    attn_bias,
                    cu_seqlens_q_padded,
                    cu_seqlens_kv_padded,
                    None,
                    None,
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
                if bool(int(os.getenv("NVTE_PRINT", "0"))):
                    layer = int(os.getenv("NVTE_LAYER_NUMBER", str(layer_number)))
                    procid = int(os.getenv("SLURM_PROCID", "0"))
                    atol = float(os.getenv("NVTE_ATOL", "1e-2"))
                    rtol = float(os.getenv("NVTE_ATOL", "1e-2"))
                    if layer_number == layer and procid == 0:
                        print(f">> aux[0], {aux_ctx_tensors[0].view(-1)[:10]}")
                        print(
                            f">> aux[0], {aux_ctx_tensors[1].nonzero().sum()},"
                            f" {aux_ctx_tensors[1].shape}"
                        )
                        print(f">> aux[1], {aux_ctx_tensors[1].view(-1)[:10]}")
                        print(f">> aux_clone[0], {aux_ctx_tensors_clone[0].view(-1)[:10]}")
                        out_close = torch.allclose(out_, out_clone, atol=atol, rtol=rtol)
                        lse_close = torch.allclose(
                            aux_ctx_tensors[0], aux_ctx_tensors_clone[0], atol=atol, rtol=rtol
                        )
                        rmse = [
                            _rmse(x, y)
                            for x, y in [
                                [out_, out_clone],
                                [aux_ctx_tensors[0], aux_ctx_tensors_clone[0]],
                            ]
                        ]
                        print(f">>>> fwd p{procid} l{layer_number} {out_close=} {lse_close=}")
                        print(f">>>> fwd p{procid} l{layer_number} {rmse=}")
                        t_in = [q, k, v]
                        t_f8 = [out_, aux_ctx_tensors[0], aux_ctx_tensors[1]]
                        t_f16 = [out_clone, aux_ctx_tensors_clone[0]]
                        tin_minmax = [(x.min().item(), x.max().item()) for x in t_in]
                        t8_minmax = [(x.min().item(), x.max().item()) for x in t_f8]
                        t16_minmax = [(x.min().item(), x.max().item()) for x in t_f16]
                        tin_minmax_strs = [f"({mi:.4e},{ma:.4e})" for (mi, ma) in tin_minmax]
                        t8_minmax_strs = [f"({mi:.4e},{ma:.4e})" for (mi, ma) in t8_minmax]
                        t16_minmax_strs = [f"({mi:.4e},{ma:.4e})" for (mi, ma) in t16_minmax]
                        names_minmax = ["qkv        ", "f8  o/stats", "f16 o/stats"]
                        for nm, mm in zip(
                            names_minmax, [tin_minmax_strs, t8_minmax_strs, t16_minmax_strs]
                        ):
                            mm = ", ".join(mm)
                            print(f">>>> fwd p{procid} l{layer_number} {nm}: {mm}")
                        torch.cuda.synchronize()
                        global_counter = next(global_fwd_count)
                        print(f">>> fwd {global_counter=}")
                        if global_counter % 400 == 0:
                            print(f">> saving fwd for {global_counter=}")
                            tensors_fp8 = [
                                (
                                    (x._data, x._scale_inv, x._quantizer.scale, x._quantizer.amax)
                                    if isinstance(x, Float8TensorBase)
                                    else x
                                )
                                for x in [q_fp8, k_fp8, v_fp8, out_, aux_ctx_tensors]
                            ]
                            tensors_fp8.extend([(S_quantizer.scale, S_quantizer.amax)])
                            tensors_f16 = [
                                q_clone,
                                k_clone,
                                v_clone,
                                out_clone,
                                aux_ctx_tensors_clone,
                            ]
                            save_path = "/results/"
                            torch.save(
                                tensors_fp8,
                                save_path + "fwd_tensors_fp8_" + str(global_counter) + ".pt",
                            )
                            torch.save(
                                tensors_f16,
                                save_path + "fwd_tensors_f16_" + str(global_counter) + ".pt",
                            )

            # out_fp8: Float8Tensor; dtype = torch.float16 or torch.bfloat16
            #                        fp8_dtype = tex.DType.kFloat8E4M3
            # out:     torch.Tensor; dtype = torch.float16 or torch.bfloat16
            out_fp8 = out_
            out = out_

            if isinstance(out_, Float8Tensor):
                if not is_output_fp8 or not is_bwd_fp8:
                    out = out_.dequantize().view(out_.shape)
            else:
                if is_output_fp8 or is_bwd_fp8:
                    out_fp8 = O_quantizer(out_)

            # return appropriate tensors
            out_ret = out_fp8 if is_output_fp8 else out

            # save appropriate tensors
            fp8_tensors = (q_fp8, k_fp8, v_fp8, out_fp8) if is_bwd_fp8 else (None, None, None, None)
            if not is_bwd_fp8 and is_input_fp8:
                q, k, v = combine_and_dequantize(qkv_layout, q_fp8, k_fp8, v_fp8)
            qkvo_tensors = (q, k, v, out)
            names = [
                "QKV_quantizer ",
                "S_quantizer   ",
                "O_quantizer   ",
                # "dQKV_quantizer",
                # "dO_quantizer  ",
                # "dP_quantizer  ",
            ]
            quantizers = [
                QKV_quantizer,
                S_quantizer,
                O_quantizer,
                # ctx.dQKV_quantizer,
                # d_out_fp8._quantizer,
                # ctx.dP_quantizer,
            ]
            if (
                int(os.getenv("SLURM_PROCID", "0")) == 0
                and bool(int(os.getenv("NVTE_PRINT", "0")))
                and layer_number == int(os.getenv("NVTE_LAYER_NUMBER", "1"))
            ):
                # torch.cuda.synchronize()
                print(f">>>>{layer_number}: {fp8_meta["recipe"]}")
                for i, x in enumerate(quantizers):
                    if x is None:
                        print(f">>>>{layer_number} {names[i]}: None")
                    else:
                        print(
                            f">>>>{layer_number} {names[i]}: {x}"
                            # f" {'CS' if x.__class__ == Float8CurrentScalingQuantizer else 'DS'},"
                            # f" scale={x.scale.item():.4e}, amax={x.amax.item():.4e}, (scale x"
                            # f" amax)={(x.scale * x.amax).item():.4e}"
                        )
                #        if x.amax.isnan():
                #            print(
                #                f">>>>{layer_number} dqkv.isnan:"
                #                f" {[(x.dtype, x.isnan().sum()) for x in [out_]]}"
                #            )
                print(
                    f">>>>{layer_number} out.minmax: {out_.__class__},"
                    f" {[(x.abs().min().item(), x.abs().max().item()) for x in [out_]]}"
                )
        else:
            # q, k, v, out_: torch.Tensor; dtype = torch.float16 or torch.bfloat16
            out_, aux_ctx_tensors = fused_attn_fwd(
                is_training,
                max_seqlen_q,
                max_seqlen_kv,
                cu_seqlens_q,
                cu_seqlens_kv,
                q,
                k,
                v,
                out_nominal_dtype,
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
            out = out_
            out_ret = out_
            fp8_tensors = (None, None, None, None)
            qkvo_tensors = q, k, v, out

        nvtx_range_pop(f"{nvtx_label}")

        ctx.fp8 = is_bwd_fp8

        from transformer_engine.pytorch.cpu_offload import (
            CPUOffloadEnabled,
            mark_activation_offload,
        )

        if CPUOffloadEnabled:
            if ctx.fp8:
                tensor_list = fp8_tensors
            else:
                tensor_list = [q, k, v, out]

            qkv_layout = "sbhd_sbhd_sbhd"
            mark_activation_offload(*tensor_list)
            mark_activation_offload(*aux_ctx_tensors)

        ctx.is_input_fp8 = is_input_fp8
        ctx.is_output_fp8 = is_output_fp8
        tensors_to_save, tensor_objects = prepare_for_saving(
            *fp8_tensors,
            *qkvo_tensors,
            cu_seqlens_q,
            cu_seqlens_kv,
            cu_seqlens_q_padded,
            cu_seqlens_kv_padded,
            *aux_ctx_tensors,
            *aux_ctx_tensors_clone,
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
            if fp8_meta["recipe"].delayed() or fp8_meta["recipe"].float8_current_scaling():
                ctx.S_quantizer.scale = S_quantizer.scale.clone()
        ctx.layer_number = layer_number

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
            assert isinstance(d_out, Float8Tensor), (
                "Gradient of the DPA output is expected to be in Float8Tensor type but found"
                " {d_out.__class__}."
            )

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

        aux_ctx_tensors = other_tensors[:3]
        aux_ctx_tensors_clone = other_tensors[3:]

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
            with torch.cuda.nvtx.range("FusedAttnFunc.backward"):
                # get nominal data type of dq, dk, dv
                # FP16/BF16 attention: torch.float16 or torch.bfloat16
                # FP8 attention:       torch.float16 or torch.bfloat16
                dqkv_nominal_dtype = d_out.dtype

                if ctx.fp8:
                    # d_out:     torch.Tensor; dtype = torch.float16 or torch.bfloat16
                    # d_out_fp8: Float8Tensor; dtype = torch.float16 or torch.bfloat16
                    #                          fp8_dtype = tex.DType.kFloat8E5M2
                    if ctx.is_output_fp8:
                        ctx.dO_quantizer = d_out._quantizer
                        d_out_fp8 = d_out
                    else:
                        d_out_fp8 = ctx.dO_quantizer(d_out)

                    # get tex.DType for dq, dk, dv data
                    dqkv_te_dtype = d_out_fp8._fp8_dtype

                    names = [
                        # "QKV_quantizer ",
                        # "O_quantizer   ",
                        # "S_quantizer   ",
                        "dO_quantizer  ",
                        "dP_quantizer  ",
                        "dQKV_quantizer",
                    ]
                    quantizers = [
                        # q_fp8._quantizer,
                        # out_fp8._quantizer,
                        # ctx.S_quantizer,
                        d_out_fp8._quantizer,
                        ctx.dP_quantizer,
                        ctx.dQKV_quantizer,
                    ]
                    # if (
                    #    int(os.getenv("SLURM_PROCID", "0")) == 0
                    #    and bool(int(os.getenv("NVTE_PRINT", "0")))
                    #    and ctx.layer_number == int(os.getenv("NVTE_LAYER_NUMBER", "1"))
                    # ):
                    #    for i, x in enumerate(quantizers):
                    #        if x is not None:
                    #            if x.amax.isnan():
                    #                print(
                    #                    f">>>>{layer_numer} before dqkv.isnan:"
                    #                    f" {[x.isnan().sum() for x in [dq_, dk_, dv_]]}"
                    #                )
                    if bool(int(os.getenv("NVTE_Emulate_in_F16", "0"))):
                        if bool(int(os.getenv("NVTE_Emulate_QDQ_QKV", "0"))):
                            q_deq16, k_deq16, v_deq16 = combine_and_dequantize(
                                ctx.qkv_layout,
                                q_fp8,
                                k_fp8,
                                v_fp8,
                                src_nominal_dtype=dqkv_nominal_dtype,
                            )
                        else:
                            q_deq16, k_deq16, v_deq16 = q, k, v
                        if bool(int(os.getenv("NVTE_Emulate_QDQ_O", "0"))):
                            out_deq16 = out_fp8.dequantize(dtype=dqkv_nominal_dtype)
                        else:
                            out_deq16 = out
                        if bool(int(os.getenv("NVTE_Emulate_QDQ_dO", "0"))):
                            d_out_deq16 = d_out_fp8.dequantize(dtype=dqkv_nominal_dtype)
                        else:
                            d_out_deq16 = d_out

                        dq_, dk_, dv_, *rest = fused_attn_bwd(
                            ctx.max_seqlen_q,
                            ctx.max_seqlen_kv,
                            cu_seqlens_q,
                            cu_seqlens_kv,
                            q_deq16,
                            k_deq16,
                            v_deq16,
                            out_deq16,
                            d_out_deq16,
                            dqkv_nominal_dtype,
                            TE_DType[dqkv_nominal_dtype],  # dqkv_te_dtype,
                            aux_ctx_tensors,
                            FusedAttnBackend["F16_arbitrary_seqlen"],
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
                        if bool(int(os.getenv("NVTE_PRINT", "0"))):
                            layer = int(os.getenv("NVTE_LAYER_NUMBER", str(ctx.layer_number)))
                            procid = int(os.getenv("SLURM_PROCID", "0"))
                            if ctx.layer_number == layer and procid == 0:
                                print(f">>>>>>>>>>>>>>>>>> bwd emulate16")
                                torch.cuda.synchronize()
                                t_in = [q, k, v, out, d_out]
                                t_f8 = [dq_, dk_, dv_]
                                t_f16 = [q_deq16, k_deq16, v_deq16, out_deq16, d_out_deq16]
                                rmse = [f"{_rmse(x,y).item():.4e}" for x, y in zip(t_in, t_f16)]
                                tin_minmax = [(x.min().item(), x.max().item()) for x in t_in]
                                t8_minmax = [(x.min().item(), x.max().item()) for x in t_f8]
                                t16_minmax = [(x.min().item(), x.max().item()) for x in t_f16]
                                tin_minmax_strs = [
                                    f"({mi:.4e},{ma:.4e})" for (mi, ma) in tin_minmax
                                ]
                                t8_minmax_strs = [f"({mi:.4e},{ma:.4e})" for (mi, ma) in t8_minmax]
                                t16_minmax_strs = [
                                    f"({mi:.4e},{ma:.4e})" for (mi, ma) in t16_minmax
                                ]
                                names_minmax = ["qkvodo     ", "emu-qkvodo ", "emu-dqkv   "]
                                for nm, mm in zip(
                                    names_minmax, [tin_minmax_strs, t16_minmax_strs, t8_minmax_strs]
                                ):
                                    mm = ", ".join(mm)
                                    print(f">>>> bwd p{procid} l{ctx.layer_number} {nm}: {mm}")
                                print(
                                    f">>>> bwd p{procid} l{ctx.layer_number} rmse_qkvodo:"
                                    f" {', '.join(rmse)}"
                                )

                    else:
                        # q_fp8, k_fp8, v_fp8, out_fp8: Float8Tensor; dtype = torch.float16 or torch.bfloat16,
                        #                               fp8_dtype = tex.DType.kFloat8E4M3
                        # d_out_fp8:                    Float8Tensor; dtype = torch.float16 or torch.bfloat16
                        #                               fp8_dtype = tex.DType.kFloat8E5M2
                        # dq_, dk_, dv_:
                        # DelayedScaling:               Float8Tensor; dtype = torch.float16 or torch.bfloat16
                        #                               fp8_dtype = tex.DType.kFloat8E5M2
                        # Float8CurrentScaling:         torch.Tensor; dtype = torch.float16 or torch.bfloat16
                        out_bwd = out_fp8
                        if bool(int(os.getenv("NVTE_F16_O", "0"))):
                            out_bwd = out
                            if bool(int(os.getenv("NVTE_PRINT", "0"))):
                                layer = int(os.getenv("NVTE_LAYER_NUMBER", str(ctx.layer_number)))
                                procid = int(os.getenv("SLURM_PROCID", "0"))
                                if ctx.layer_number == layer and procid == 0:
                                    print(
                                        f">>>>>>>>>>>>>>>>>> bwd f16 O",
                                        out_bwd.dtype,
                                        dqkv_nominal_dtype,
                                        dqkv_te_dtype,
                                    )
                        dq_, dk_, dv_, *rest = fused_attn_bwd(
                            ctx.max_seqlen_q,
                            ctx.max_seqlen_kv,
                            cu_seqlens_q,
                            cu_seqlens_kv,
                            q_fp8,
                            k_fp8,
                            v_fp8,
                            out_bwd,
                            d_out_fp8,
                            dqkv_nominal_dtype,
                            dqkv_te_dtype,
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

                    # dq_fp8, dk_fp8, dv_fp8: Float8Tensor; dtype = torch.float16 or torch.bfloat16
                    #                                       fp8_dtype = tex.DType.kFloat8E4M3
                    # dq, dk, dv:             torch.Tensor; dtype = torch.float16 or torch.bfloat16
                    dq_fp8, dk_fp8, dv_fp8 = dq_, dk_, dv_
                    dq, dk, dv = dq_, dk_, dv_
                    if (
                        all(isinstance(x, Float8Tensor) for x in [dq_, dk_, dv_])
                        and not ctx.is_input_fp8
                    ):
                        dq, dk, dv = combine_and_dequantize(
                            ctx.qkv_layout,
                            dq_fp8,
                            dk_fp8,
                            dv_fp8,
                            src_nominal_dtype=dq_fp8.dtype,
                        )
                    if (
                        not all(isinstance(x, Float8Tensor) for x in [dq_, dk_, dv_])
                        and ctx.is_input_fp8
                    ):
                        # return dq_fp8, dk_fp8, dv_fp8
                        dq, dk, dv = combine_and_quantize(
                            ctx.qkv_layout, dq, dk, dv, ctx.dQKV_quantizer
                        )

                    if (
                        int(os.getenv("SLURM_PROCID", "0")) == 0
                        and bool(int(os.getenv("NVTE_PRINT", "0")))
                        and ctx.layer_number == int(os.getenv("NVTE_LAYER_NUMBER", "1"))
                    ):
                        # torch.cuda.synchronize()
                        for i, x in enumerate(quantizers):
                            if x is None:
                                print(f">>>>{ctx.layer_number} {names[i]}: None")
                            else:
                                print(
                                    f">>>>{ctx.layer_number} {names[i]}: {x}"
                                    # f" scale={x.scale.item():.4e}, amax={x.amax.item():.4e}, (scale"
                                    # f" x amax)={(x.scale * x.amax).item():.4e}"
                                )
                        #        if x.amax.isnan():
                        #            print(
                        #                f">>>>{ctx.layer_number} dqkv.isnan:"
                        #                f" {[(x.dtype, x.isnan().sum()) for x in [dq_, dk_, dv_]]}"
                        #            )
                        print(
                            f">>>>{ctx.layer_number} dqkv.minmax:"
                            f" {dq_.__class__} {[(x.abs().min().item(), x.abs().max().item()) for x in [dq_, dk_, dv_]]}"
                        )
                    # repeat FP8 in F16
                    if bool(int(os.getenv("NVTE_REPEAT_in_F16", "0"))):
                        assert all(
                            isinstance(x, torch.Tensor) for x in [q, k, v, out, d_out]
                        ), "BWD: qkv must be F16"
                        q_clone, k_clone, v_clone, out_clone, d_out_clone = [
                            x.detach().clone() for x in [q, k, v, out, d_out]
                        ]
                        dq_clone, dk_clone, dv_clone, *rest_clone = fused_attn_bwd(
                            ctx.max_seqlen_q,
                            ctx.max_seqlen_kv,
                            cu_seqlens_q,
                            cu_seqlens_kv,
                            q_clone,
                            k_clone,
                            v_clone,
                            out_clone,
                            d_out_clone,
                            dqkv_nominal_dtype,
                            TE_DType[dqkv_nominal_dtype],  # dqkv_te_dtype,
                            aux_ctx_tensors_clone,
                            FusedAttnBackend["F16_arbitrary_seqlen"],
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
                        if bool(int(os.getenv("NVTE_PRINT", "0"))):
                            layer = int(os.getenv("NVTE_LAYER_NUMBER", str(ctx.layer_number)))
                            procid = int(os.getenv("SLURM_PROCID", "0"))
                            atol = float(os.getenv("NVTE_ATOL", "1e-2"))
                            rtol = float(os.getenv("NVTE_ATOL", "1e-2"))
                            if ctx.layer_number == layer and procid == 0:
                                dqkv_close = [
                                    torch.allclose(x, y, atol=atol, rtol=rtol)
                                    for x, y in zip([dq_, dk_, dv_], [dq_clone, dk_clone, dv_clone])
                                ]
                                rmse = [
                                    _rmse(x, y)
                                    for x, y in zip([dq_, dk_, dv_], [dq_clone, dk_clone, dv_clone])
                                ]
                                print(f">>>> bwd p{procid} l{ctx.layer_number} {dqkv_close=}")
                                print(f">>>> bwd p{procid} l{ctx.layer_number} {rmse=}")
                                t_in = [q, k, v, out, d_out]
                                t_f8 = [dq_, dk_, dv_]
                                t_f16 = [dq_clone, dk_clone, dv_clone]
                                tin_minmax = [(x.min().item(), x.max().item()) for x in t_in]
                                t8_minmax = [(x.min().item(), x.max().item()) for x in t_f8]
                                t16_minmax = [(x.min().item(), x.max().item()) for x in t_f16]
                                tin_minmax_strs = [
                                    f"({mi:.4e},{ma:.4e})" for (mi, ma) in tin_minmax
                                ]
                                t8_minmax_strs = [f"({mi:.4e},{ma:.4e})" for (mi, ma) in t8_minmax]
                                t16_minmax_strs = [
                                    f"({mi:.4e},{ma:.4e})" for (mi, ma) in t16_minmax
                                ]
                                names_minmax = ["qkvodo  ", "f8  dqkv", "f16 dqkv"]
                                for nm, mm in zip(
                                    names_minmax, [tin_minmax_strs, t8_minmax_strs, t16_minmax_strs]
                                ):
                                    mm = ", ".join(mm)
                                    print(f">>>> bwd p{procid} l{ctx.layer_number} {nm}: {mm}")
                                torch.cuda.synchronize()
                                global_counter = next(global_bwd_count)
                                print(f">>> bwd {global_counter=}")
                                if global_counter % 400 == 0:
                                    print(f">> saving bwd for {global_counter=}")
                                    tensors_fp8 = [
                                        (
                                            (
                                                x._data,
                                                x._scale_inv,
                                                x._quantizer.scale,
                                                x._quantizer.amax,
                                            )
                                            if isinstance(x, Float8TensorBase)
                                            else x
                                        )
                                        for x in [
                                            q_fp8,
                                            k_fp8,
                                            v_fp8,
                                            out_fp8,
                                            d_out_fp8,
                                            aux_ctx_tensors,
                                            dq_,
                                            dk_,
                                            dv_,
                                        ]
                                    ]
                                    tensors_fp8.extend(
                                        [
                                            (x.scale, x.amax)
                                            for x in [ctx.S_quantizer, ctx.dP_quantizer]
                                        ]
                                    )
                                    tensors_f16 = [
                                        q_clone,
                                        k_clone,
                                        v_clone,
                                        out_clone,
                                        d_out_clone,
                                        aux_ctx_tensors_clone,
                                        dq_clone,
                                        dk_clone,
                                        dv_clone,
                                    ]
                                    save_path = "/results/"
                                    torch.save(
                                        tensors_fp8,
                                        save_path
                                        + "bwd_tensors_fp8_"
                                        + str(global_counter)
                                        + ".pt",
                                    )
                                    torch.save(
                                        tensors_f16,
                                        save_path
                                        + "bwd_tensors_f16_"
                                        + str(global_counter)
                                        + ".pt",
                                    )

                else:
                    if isinstance(d_out, QuantizedTensor):
                        d_out = d_out.dequantize()
                    dqkv_te_dtype = TE_DType[d_out.dtype]
                    # q, k, v, out, d_out, dq, dk, dv: torch.Tensor; torch.float16 or torch.bfloat16
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
                        dqkv_nominal_dtype,
                        dqkv_te_dtype,
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

        dbias = None
        if ctx.attn_bias_type == "post_scale_bias":
            dbias = rest[0]
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
            dbias,
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
        fp8_output: bool = False,
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
            cp_size = get_distributed_world_size(cp_group[0])
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
            primary_recipe = FP8GlobalStateManager.get_fp8_recipe()
            assert fused_attention_backend == tex.NVTE_Fused_Attn_Backend.NVTE_FP8, (
                f"cuDNN attention sub-backend {int(tex.NVTE_Fused_Attn_Backend.NVTE_FP8)}"
                " is required for FP8 attention!"
            )
            assert fp8_meta is not None, "FP8 metadata fp8_meta is required for FP8 attention!"
            if primary_recipe.delayed():
                assert not context_parallel or primary_recipe.reduce_amax, (
                    "Amax reduction across TP+CP group is necessary when using context parallelism"
                    " with FP8!"
                )
            if primary_recipe.float8_current_scaling() and context_parallel:
                all_quantizers = dpa_utils.get_attention_quantizers(
                    fp8, fp8_meta, quantizers, cp_specific_quantizers=True
                )
                for q in all_quantizers:
                    if isinstance(q, Float8CurrentScalingQuantizer):
                        q.with_amax_reduction = True
                        q.amax_reduction_group = (
                            cp_group[0] if cp_comm_type == "a2a+p2p" else cp_group
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
                    fp8_output=fp8_output,
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
                    fp8_output,
                    self.layer_number,
                )

        # ...hd -> ...(hd)
        return output.view(*output.shape[:-2], -1)
