# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Pure Python replacement for the pybind11 `transformer_engine_torch` module.

This module provides the same API as `transformer_engine_torch` but routes
all calls through the stable ABI extension (`te_stable_abi`), eliminating
the dependency on unstable PyTorch C++ internals (ATen, c10, pybind11).

The stable extension is loaded once via `torch.ops.load_library()`.
All ops are accessed as `torch.ops.transformer_engine_stable.<op_name>`.
"""

# pylint: disable=missing-class-docstring,missing-function-docstring

import ctypes as _ctypes
import glob
import importlib.util

from enum import IntEnum
from pathlib import Path

import torch

# ============================================================================
# Load the stable ABI shared library
# ============================================================================

_loaded = False


def _load_stable_lib():
    global _loaded
    if _loaded:
        return
    te_spec = importlib.util.find_spec("transformer_engine")
    if te_spec is not None and te_spec.origin is not None:
        te_dir = Path(te_spec.origin).parent
        candidates = glob.glob(str(te_dir / "te_stable_abi*"))
        if candidates:
            torch.ops.load_library(candidates[0])
            _loaded = True
            return
    raise FileNotFoundError("Could not find shared object file: te_stable_abi")


_load_stable_lib()
_ops = torch.ops.transformer_engine_stable


def _not_implemented(name):
    """Create a stub function that raises NotImplementedError."""

    def fn(*args, **kwargs):
        raise NotImplementedError(
            f"{name} is not yet implemented in the stable ABI module. "
            "This function needs a native stable implementation."
        )

    fn.__name__ = name
    return fn


def _fill_fp8_transpose_if_needed(tensor):
    """Fill the FP8 transpose buffer for Float8Tensor (delayed/current scaling) if pre-allocated.

    The pybind11 fused LayerNorm+FP8 kernel fills both rowwise (_data) and columnwise
    (_transpose) buffers in one shot. This helper mirrors that behavior for the stable ABI
    path, where layernorm_fwd/rmsnorm_fwd calls quantize_new (which fills only _data).
    Without this, update_usage(rowwise=False) sees _transpose_invalid=True and deletes both
    _data and _transpose, leaving nothing for the backward wgrad GEMM.
    """
    if not hasattr(tensor, "_data") or tensor._data is None:
        return
    if not hasattr(tensor, "_transpose") or tensor._transpose is None:
        return
    if not getattr(tensor, "_transpose_invalid", True):
        return  # already valid
    fp8_dtype_attr = getattr(tensor, "_fp8_dtype", None)
    if fp8_dtype_attr is None:
        return
    from transformer_engine.pytorch.tensor._extract import _FP8_DTYPE_TO_TE

    fp8_te_dtype = _FP8_DTYPE_TO_TE.get(str(fp8_dtype_attr), 7)
    tensor._transpose = _ops.fp8_transpose(tensor._data, fp8_te_dtype, tensor._transpose)
    tensor._transpose_invalid = False


def _try_fused_norm_quantize_layernorm(
    inp_data, w_data, bias, eps, quantizer, sm_margin, zero_centered_gamma
):
    """Attempt fused layernorm+quantize via _noalloc op for delayed scaling.

    Returns (q_out, mu, rsigma) if fused, or None to fall back to unfused path.
    """
    q_type = type(quantizer).__name__
    if q_type != "Float8Quantizer":
        return None
    scale = getattr(quantizer, "scale", None)
    if scale is None or not isinstance(scale, torch.Tensor) or scale.numel() == 0:
        return None

    from transformer_engine.pytorch.tensor._extract import _FP8_DTYPE_TO_TE

    shape = list(inp_data.shape)
    device = inp_data.device
    q_out = quantizer.make_empty(shape, dtype=inp_data.dtype, device=device)

    out_data = q_out._data
    fp8_dtype = getattr(q_out, "_fp8_dtype", None)
    te_dtype = _FP8_DTYPE_TO_TE.get(str(fp8_dtype), 7) if fp8_dtype else 7
    out_scale_inv = getattr(q_out, "_scale_inv", None)
    out_amax = getattr(quantizer, "amax", None)
    if out_amax is not None and (not isinstance(out_amax, torch.Tensor) or out_amax.numel() == 0):
        out_amax = None

    # Allocate mu/rsigma (1D, product of all dims except last)
    outer_size = 1
    for d in shape[:-1]:
        outer_size *= d
    mu = torch.empty(outer_size, dtype=torch.float32, device=device)
    rsigma = torch.empty(outer_size, dtype=torch.float32, device=device)

    _ops.layernorm_fwd_noalloc(
        inp_data,
        w_data,
        bias,
        eps,
        out_data,
        te_dtype,
        out_amax,
        scale,
        out_scale_inv,
        0,
        mu,
        rsigma,
        sm_margin,
        zero_centered_gamma,
    )

    q_out._transpose_invalid = True
    _fill_fp8_transpose_if_needed(q_out)
    return q_out, mu, rsigma


def _try_fused_norm_quantize_rmsnorm(
    inp_data, w_data, eps, quantizer, sm_margin, zero_centered_gamma
):
    """Attempt fused rmsnorm+quantize via _noalloc op for delayed scaling.

    Returns (q_out, rsigma) if fused, or None to fall back to unfused path.
    """
    q_type = type(quantizer).__name__
    if q_type != "Float8Quantizer":
        return None
    scale = getattr(quantizer, "scale", None)
    if scale is None or not isinstance(scale, torch.Tensor) or scale.numel() == 0:
        return None

    from transformer_engine.pytorch.tensor._extract import _FP8_DTYPE_TO_TE

    shape = list(inp_data.shape)
    device = inp_data.device
    q_out = quantizer.make_empty(shape, dtype=inp_data.dtype, device=device)

    out_data = q_out._data
    fp8_dtype = getattr(q_out, "_fp8_dtype", None)
    te_dtype = _FP8_DTYPE_TO_TE.get(str(fp8_dtype), 7) if fp8_dtype else 7
    out_scale_inv = getattr(q_out, "_scale_inv", None)
    out_amax = getattr(quantizer, "amax", None)
    if out_amax is not None and (not isinstance(out_amax, torch.Tensor) or out_amax.numel() == 0):
        out_amax = None

    outer_size = 1
    for d in shape[:-1]:
        outer_size *= d
    rsigma = torch.empty(outer_size, dtype=torch.float32, device=device)

    _ops.rmsnorm_fwd_noalloc(
        inp_data,
        w_data,
        eps,
        out_data,
        te_dtype,
        out_amax,
        scale,
        out_scale_inv,
        0,
        rsigma,
        sm_margin,
        zero_centered_gamma,
    )

    q_out._transpose_invalid = True
    _fill_fp8_transpose_if_needed(q_out)
    return q_out, rsigma


def _extract_gemm_operand(tensor, use_rowwise):  # pylint: disable=unused-argument
    """Extract rowwise + optional columnwise buffers and metadata for GEMM.

    Always returns rowwise_data as the primary buffer (logical shape).
    When the tensor has a separate columnwise buffer (FP8 block-scaling or
    MXFP8), that buffer is also returned so the stable C++ GEMM can set
    both on the TensorWrapper and let CanonicalizeGemmInput choose at
    runtime based on transa/transb.

    Returns:
        (data, te_dtype, scale_inv, scaling_mode,
         with_gemm_swizzled_scales, colwise_data, colwise_scale_inv,
         amax, colwise_amax)
    """
    from transformer_engine.pytorch.tensor._extract import extract_tensor_data
    from transformer_engine.pytorch.tensor._extract import _detect_scaling_mode

    data, te_dtype, scale_inv, scaling_mode = extract_tensor_data(tensor)
    # For GEMM, use the real scaling mode (MXFP8/NVFP4/block-scaling).
    # extract_tensor_data returns DELAYED for MXFP8/NVFP4 because they lack
    # _is_2D_scaled, but the GEMM C++ code needs the correct mode.
    scaling_mode = _detect_scaling_mode(tensor)
    with_gemm_swizzled_scales = bool(getattr(tensor, "_with_gemm_swizzled_scales", False))
    colwise_data = None
    colwise_scale_inv = None

    if hasattr(tensor, "_rowwise_data") and getattr(tensor, "_rowwise_data", None) is not None:
        # Primary data: always rowwise (logical shape)
        data = tensor._rowwise_data
        scale_inv = getattr(tensor, "_rowwise_scale_inv", None)
        # Columnwise data (optional): C++ TensorWrapper will pick the right
        # one based on transa/transb via CanonicalizeGemmInput.
        cw = getattr(tensor, "_columnwise_data", None)
        if cw is not None:
            colwise_data = cw
            colwise_scale_inv = getattr(tensor, "_columnwise_scale_inv", None)
    elif (
        hasattr(tensor, "_rowwise_data")
        and getattr(tensor, "_rowwise_data", None) is None
        and getattr(tensor, "_columnwise_data", None) is not None
        and not hasattr(tensor, "_data")
    ):
        # Columnwise-only tensor (e.g. Float8BlockwiseQTensor or MXFP8Tensor with rowwise=False).
        # Pass an empty placeholder for rowwise_data so C++ skips set_rowwise_data.
        # Set the correct scaling_mode and FP8 dtype from tensor attributes.
        fp8_dtype_attr = getattr(tensor, "_fp8_dtype", None)
        if fp8_dtype_attr is not None:
            from transformer_engine.pytorch.tensor._extract import _FP8_DTYPE_TO_TE

            te_dtype = _FP8_DTYPE_TO_TE.get(str(fp8_dtype_attr), 7)
        # scaling_mode already set by _detect_scaling_mode above
        cw = tensor._columnwise_data
        csi = getattr(tensor, "_columnwise_scale_inv", None)
        # Pass empty rowwise placeholder so C++ skips set_rowwise_data
        data = cw.new_empty(0)
        scale_inv = None
        colwise_data = cw
        colwise_scale_inv = csi
    elif hasattr(tensor, "_data"):
        # Float8Tensor (delayed scaling).
        fp8_data = getattr(tensor, "_data", None)
        fp8_trans = (
            None
            if getattr(tensor, "_transpose_invalid", True)
            else getattr(tensor, "_transpose", None)
        )
        fp8_dtype_attr = getattr(tensor, "_fp8_dtype", None)
        if fp8_dtype_attr is not None:
            # Resolve FP8 te_dtype from the tensor's actual FP8 dtype
            from transformer_engine.pytorch.tensor._extract import _FP8_DTYPE_TO_TE

            te_dtype = _FP8_DTYPE_TO_TE.get(str(fp8_dtype_attr), 7)
        si = getattr(tensor, "_scale_inv", None)
        if fp8_data is not None:
            # Normal case: rowwise data available
            data = fp8_data
            scale_inv = si
            if fp8_trans is not None:
                # Columnwise (transpose) buffer also available
                colwise_data = fp8_trans
                colwise_scale_inv = si
        elif fp8_trans is not None:
            # Columnwise-only: _data is None, only transpose buffer exists.
            # Pass an empty placeholder for rowwise_data so C++ buildInputTensorWrapper
            # skips set_rowwise_data (numel==0). NVTE's CanonicalizeGemmInput will use
            # the columnwise buffer (the transpose) with a flipped transa/transb flag.
            data = fp8_trans.new_empty(0)
            scale_inv = si
            colwise_data = fp8_trans
            colwise_scale_inv = si

    # NVFP4 tensors carry per-tensor amax needed by the GEMM formula:
    #   output = fp4_value * scale_e4m3 * amax / (6 * 448)
    amax = getattr(tensor, "_amax_rowwise", None)
    colwise_amax = getattr(tensor, "_amax_columnwise", None)

    return (
        data,
        te_dtype,
        scale_inv,
        scaling_mode,
        with_gemm_swizzled_scales,
        colwise_data,
        colwise_scale_inv,
        amax,
        colwise_amax,
    )


# ============================================================================
# Enums (replace pybind11 enum bindings)
# ============================================================================


class DType(IntEnum):
    kByte = 0
    kInt16 = 1
    kInt32 = 2
    kInt64 = 3
    kFloat32 = 4
    kFloat16 = 5
    kBFloat16 = 6
    kFloat8E4M3 = 7
    kFloat8E5M2 = 8
    kFloat8E8M0 = 9
    kFloat4E2M1 = 10


class FP8FwdTensors(IntEnum):
    GEMM1_INPUT = 0
    GEMM1_WEIGHT = 1
    GEMM1_OUTPUT = 2
    GEMM2_INPUT = 3
    GEMM2_WEIGHT = 4
    GEMM2_OUTPUT = 5
    GEMM3_INPUT = 6
    GEMM3_WEIGHT = 7
    GEMM3_OUTPUT = 8


class FP8BwdTensors(IntEnum):
    GRAD_OUTPUT1 = 0
    GRAD_INPUT1 = 1
    GRAD_OUTPUT2 = 2
    GRAD_INPUT2 = 3
    GRAD_OUTPUT3 = 4
    GRAD_INPUT3 = 5


# ============================================================================
# FP8TensorMeta (replace pybind11 class binding)
# ============================================================================


class FP8TensorMeta:
    def __init__(self):
        self.scale = torch.tensor([], dtype=torch.float32)
        self.scale_inv = torch.tensor([], dtype=torch.float32)
        self.amax_history = torch.tensor([], dtype=torch.float32)


# ============================================================================
# Version / info queries
# ============================================================================


def get_cublasLt_version():
    """Return the cublasLt library version."""
    return _ops.get_cublasLt_version()


def get_cudnn_version():
    """Return the cuDNN library version."""
    return _ops.get_cudnn_version()


def get_num_cublas_streams():
    """Return the number of cuBLAS compute streams."""
    return _ops.get_num_cublas_streams()


# ============================================================================
# Softmax ops (direct passthrough)
# ============================================================================

scaled_softmax_forward = _ops.scaled_softmax_forward
scaled_softmax_backward = _ops.scaled_softmax_backward
scaled_masked_softmax_forward = _ops.scaled_masked_softmax_forward
scaled_masked_softmax_backward = _ops.scaled_masked_softmax_backward
scaled_upper_triang_masked_softmax_forward = _ops.scaled_upper_triang_masked_softmax_forward
scaled_upper_triang_masked_softmax_backward = _ops.scaled_upper_triang_masked_softmax_backward
scaled_aligned_causal_masked_softmax_forward = _ops.scaled_aligned_causal_masked_softmax_forward
scaled_aligned_causal_masked_softmax_backward = _ops.scaled_aligned_causal_masked_softmax_backward

# ============================================================================
# Padding
# ============================================================================

fused_multi_row_padding = _ops.fused_multi_row_padding
fused_multi_row_unpadding = _ops.fused_multi_row_unpadding

# ============================================================================
# Misc
# ============================================================================

splits_to_offsets = _ops.splits_to_offsets

# ============================================================================
# RoPE
# ============================================================================


def fused_rope_forward(  # pylint: disable=redefined-builtin
    input, freqs, start_positions, qkv_format, interleaved, cu_seqlens, cp_size, cp_rank
):
    return _ops.fused_rope_forward(
        input, freqs, start_positions, int(qkv_format), interleaved, cu_seqlens, cp_size, cp_rank
    )


def fused_rope_backward(
    output_grads, freqs, start_positions, qkv_format, interleaved, cu_seqlens, cp_size, cp_rank
):
    return _ops.fused_rope_backward(
        output_grads,
        freqs,
        start_positions,
        int(qkv_format),
        interleaved,
        cu_seqlens,
        cp_size,
        cp_rank,
    )


def fused_qkv_rope_forward(
    qkv_input,
    q_freqs,
    k_freqs,
    start_positions,
    qkv_split_arg_list,
    qkv_format,
    interleaved,
    cp_size,
    cp_rank,
):
    return _ops.fused_qkv_rope_forward(
        qkv_input,
        q_freqs,
        k_freqs,
        start_positions,
        list(qkv_split_arg_list),
        int(qkv_format),
        interleaved,
        cp_size,
        cp_rank,
    )


def fused_qkv_rope_backward(
    q_grad_out,
    k_grad_out,
    v_grad_out,
    q_freqs,
    k_freqs,
    qkv_split_arg_list,
    qkv_format,
    interleaved,
    cp_size,
    cp_rank,
):
    return _ops.fused_qkv_rope_backward(
        q_grad_out,
        k_grad_out,
        v_grad_out,
        q_freqs,
        k_freqs,
        list(qkv_split_arg_list),
        int(qkv_format),
        interleaved,
        cp_size,
        cp_rank,
    )


# ============================================================================
# Router
# ============================================================================


def fused_topk_with_score_function_fwd(
    logits,
    topk,
    use_pre_softmax,
    num_groups=None,
    group_topk=None,
    scaling_factor=None,
    score_function="softmax",
    expert_bias=None,
):
    return _ops.fused_topk_with_score_function_fwd(
        logits,
        topk,
        use_pre_softmax,
        num_groups if num_groups is not None else -1,
        group_topk if group_topk is not None else -1,
        scaling_factor if scaling_factor is not None else 1.0,
        score_function,
        expert_bias,
    )


def fused_topk_with_score_function_bwd(
    num_tokens,
    num_experts,
    routing_map,
    intermediate_output,
    grad_probs,
    grad_logits,
    topk,
    use_pre_softmax,
    scaling_factor=None,
    score_function="softmax",
):
    _ops.fused_topk_with_score_function_bwd(
        num_tokens,
        num_experts,
        routing_map,
        intermediate_output,
        grad_probs,
        grad_logits,
        topk,
        use_pre_softmax,
        scaling_factor if scaling_factor is not None else 1.0,
        score_function,
    )


fused_score_for_moe_aux_loss_fwd = _ops.fused_score_for_moe_aux_loss_fwd
fused_score_for_moe_aux_loss_bwd = _ops.fused_score_for_moe_aux_loss_bwd
fused_moe_aux_loss_fwd = _ops.fused_moe_aux_loss_fwd
fused_moe_aux_loss_bwd = _ops.fused_moe_aux_loss_bwd

# ============================================================================
# Dropout
# ============================================================================


def dropout_fwd(
    input, dropout_probability, out=None
):  # pylint: disable=redefined-builtin,unused-argument
    """Dropout forward. RNG state extracted from default CUDA generator."""
    device = input.device if hasattr(input, "device") else torch.device("cuda")
    # Extract from torch tensor if input is a py handle-like
    if hasattr(input, "_data"):
        inp_tensor = input._data
    elif isinstance(input, torch.Tensor):
        inp_tensor = input
    else:
        inp_tensor = input

    gen = torch.cuda.default_generators[device.index or 0]
    rng_state = torch.empty(2, dtype=torch.int64, device=device)
    seed = gen.initial_seed()
    offset = gen.get_offset()
    # Advance generator state to avoid overlap with subsequent random ops.
    # Matches old pybind path: gen->philox_cuda_state(rng_elts_per_thread=4)
    gen.set_offset(offset + 4)
    rng_state[0] = seed
    rng_state[1] = offset

    output, mask = _ops.dropout_fwd(inp_tensor, rng_state, dropout_probability)
    return [output, mask]


def dropout_bwd(grad_output, mask, dropout_probability, grad_input=None):
    return _ops.dropout_bwd(grad_output, mask, dropout_probability, grad_input)


# ============================================================================
# Transpose ops
# ============================================================================


def fp8_transpose(input, otype, *, out=None):  # pylint: disable=redefined-builtin
    return _ops.fp8_transpose(input, int(otype), out)


def nvfp4_data_transpose(inp, output=None, *, out=None):  # pylint: disable=redefined-builtin
    """Transpose NVFP4 packed data."""
    if output is None:
        output = out
    return _ops.nvfp4_data_transpose(inp, output)


nvfp4_2d_scale_transpose = _ops.nvfp4_2d_scale_transpose
nvfp4_expand_scale_to_fp8 = _ops.nvfp4_expand_scale_to_fp8
nvfp4_compute_per_block_scale = _ops.nvfp4_compute_per_block_scale
nvfp4_fused_scale = _ops.nvfp4_fused_scale
nvfp4_compute_global_scale = _ops.nvfp4_compute_global_scale
swap_first_dims = _ops.swap_first_dims

# ============================================================================
# Attention helpers
# ============================================================================

fa_prepare_fwd = _ops.fa_prepare_fwd
fa_prepare_bwd = _ops.fa_prepare_bwd
thd_read_half_tensor = _ops.thd_read_half_tensor
thd_second_half_lse_correction = _ops.thd_second_half_lse_correction
thd_read_second_half_lse = _ops.thd_read_second_half_lse
thd_out_correction = _ops.thd_out_correction
thd_grad_correction = _ops.thd_grad_correction
thd_get_partitioned_indices = _ops.thd_get_partitioned_indices
convert_thd_to_bshd = _ops.convert_thd_to_bshd
convert_bshd_to_thd = _ops.convert_bshd_to_thd


def copy_to_kv_cache(
    new_k,
    new_v,
    k_cache,
    v_cache,
    page_table,
    cu_new_lens,
    cu_cached_lens,
    qkv_format,
    b,
    max_ctx_len,
    max_seq_len,
    max_pages_per_seq,
    is_non_paged,
):
    _ops.copy_to_kv_cache(
        new_k,
        new_v,
        k_cache,
        v_cache,
        page_table,
        cu_new_lens,
        cu_cached_lens,
        int(qkv_format),
        b,
        max_ctx_len,
        max_seq_len,
        max_pages_per_seq,
        is_non_paged,
    )


# ============================================================================
# Recipe / amax / scale
# ============================================================================

compute_amax = _ops.compute_amax


def get_fused_attn_backend(
    is_training,
    q_dtype,
    kv_dtype,
    qkv_layout,
    bias_type,
    attn_mask_type,
    softmax_type,
    p_dropout,
    num_attn_heads,
    num_gqa_groups,
    max_seqlen_q,
    max_seqlen_kv,
    head_dim_qk,
    head_dim_v,
    window_size_left,
    window_size_right,
    return_max_logit,
    cuda_graph,
    deterministic,
):
    """Get fused attention backend via stable ABI op."""
    return _ops.get_fused_attn_backend(
        bool(is_training),
        int(q_dtype),
        int(kv_dtype),
        int(qkv_layout),
        int(bias_type),
        int(attn_mask_type),
        int(softmax_type),
        float(p_dropout),
        int(num_attn_heads),
        int(num_gqa_groups),
        int(max_seqlen_q),
        int(max_seqlen_kv),
        int(head_dim_qk),
        int(head_dim_v),
        int(window_size_left),
        int(window_size_right),
        bool(return_max_logit),
        bool(cuda_graph),
        bool(deterministic),
    )


def fused_amax_and_scale_update_after_reduction(
    amax_reduction_buffer, amax_histories, scales, amax_compute_algo, fp8_dtype, margin
):
    num = len(amax_histories)
    ah_ptrs = torch.tensor([t.data_ptr() for t in amax_histories], dtype=torch.int64)
    # Shape format: [ndim, dim0, dim1] per tensor (dim1=0 for 1D)
    ah_shapes = torch.tensor(
        [[t.dim(), t.shape[0], t.shape[1] if t.dim() >= 2 else 0] for t in amax_histories],
        dtype=torch.int64,
    ).flatten()
    sc_ptrs = torch.tensor([t.data_ptr() for t in scales], dtype=torch.int64)
    sc_shapes = torch.tensor(
        [[t.dim(), t.shape[0], t.shape[1] if t.dim() >= 2 else 0] for t in scales],
        dtype=torch.int64,
    ).flatten()
    _ops.fused_amax_and_scale_update(
        amax_reduction_buffer,
        ah_ptrs,
        ah_shapes,
        sc_ptrs,
        sc_shapes,
        num,
        amax_compute_algo,
        int(fp8_dtype),
        margin,
    )


# ============================================================================
# Partial cast
# ============================================================================

fp8_block_scaling_compute_partial_amax = _ops.fp8_block_scaling_compute_partial_amax


def fp8_block_scaling_partial_cast(inp, out, scale, h, w, start_offset, block_len, out_dtype):
    _ops.fp8_block_scaling_partial_cast(
        inp, out, scale, h, w, start_offset, block_len, int(out_dtype)
    )


mxfp8_scaling_compute_partial_amax = _ops.mxfp8_scaling_compute_partial_amax
mxfp8_scaling_partial_cast = _ops.mxfp8_scaling_partial_cast

nvfp4_2d_compute_partial_amax = _ops.nvfp4_2d_compute_partial_amax


def nvfp4_2d_partial_cast(inp, out, scale, global_scale, h, w, start_offset, block_len=16):
    """Match pybind signature — out may be quantized tensor."""
    from transformer_engine.pytorch.tensor._extract import extract_tensor_data

    out_data, _out_dtype, out_si, out_sm = extract_tensor_data(out)
    # The C++ kernel expects kByte (0) for the raw uint8 data buffer,
    # not the logical FP4 type (kFloat4E2M1=10).
    _ops.nvfp4_2d_partial_cast_noalloc(
        inp, out_data, 0, out_si, out_sm, scale, global_scale, h, w, start_offset, block_len
    )


# ============================================================================
# Permutation
# ============================================================================


def moe_permute_fwd(
    input, dtype, indices, num_out_tokens, workspace, max_expanded_token_num
):  # pylint: disable=redefined-builtin
    num_tokens = input.size(0)
    topK = indices.size(1)

    # Workspace management: workspace is a list of [sorted_indices, row_id, sorted_row_id, temp_storage]
    # On first call (empty workspace), allocate. Reuse on subsequent calls.
    if not workspace:
        options = {"dtype": torch.int32, "device": input.device, "requires_grad": False}
        sorted_indices = torch.empty(max_expanded_token_num, **options)
        row_id = torch.arange(0, max_expanded_token_num, dtype=torch.int32, device=input.device)
        sorted_row_id = torch.empty(max_expanded_token_num, **options)
        # temp_storage placeholder (not used in Python sort path)
        temp_storage = torch.empty(0, dtype=torch.int8, device=input.device)
        workspace.extend([sorted_indices, row_id, sorted_row_id, temp_storage])

    # Radix sort: sort indices to get sorted_indices and sorted_row_id
    flat_indices = indices.reshape(-1)[: num_tokens * topK]
    sorted_indices_out, sort_perm = torch.sort(flat_indices, stable=True)
    workspace[0][: num_tokens * topK] = sorted_indices_out
    workspace[2][: num_tokens * topK] = workspace[1][: num_tokens * topK][sort_perm]
    sorted_row_id = workspace[2][: num_tokens * topK]

    # Pre-allocate row_id_map
    row_id_map = torch.empty(num_tokens * topK, dtype=torch.int32, device=input.device)

    permuted_output, row_id_map = _ops.moe_permute_fwd(
        input, int(dtype), sorted_row_id, row_id_map, num_tokens, topK, num_out_tokens
    )
    return permuted_output, row_id_map, workspace


def moe_permute_bwd(
    input, dtype, row_id_map, prob, num_tokens, topK
):  # pylint: disable=redefined-builtin
    return _ops.moe_unpermute_fwd(input, int(dtype), row_id_map, prob, num_tokens, topK)


def moe_unpermute_fwd(
    input, dtype, row_id_map, prob, num_tokens, topK
):  # pylint: disable=redefined-builtin
    return _ops.moe_unpermute_fwd(input, int(dtype), row_id_map, prob, num_tokens, topK)


def moe_unpermute_bwd(
    input_bwd, input_fwd, dtype, row_id_map, prob
):  # pylint: disable=redefined-builtin
    act_grad, prob_grad = _ops.moe_unpermute_bwd(input_bwd, input_fwd, int(dtype), row_id_map, prob)
    # Reshape prob_grad from [num_tokens * topK] to match probs shape
    if prob.numel() > 0 and prob_grad.numel() > 0:
        prob_grad = prob_grad.view(prob.shape)
    elif prob.numel() > 0:
        prob_grad = torch.zeros_like(prob)
    return act_grad, prob_grad


# ============================================================================
# Normalization
# ============================================================================


def layernorm_bwd(dz, x, mu, rsigma, gamma, sm_margin, zero_centered_gamma):
    dx, dgamma, dbeta = _ops.layernorm_bwd(dz, x, mu, rsigma, gamma, sm_margin, zero_centered_gamma)
    return [dx, dgamma, dbeta]


def rmsnorm_bwd(dz, x, rsigma, gamma, sm_margin, zero_centered_gamma):
    dx, dgamma = _ops.rmsnorm_bwd(dz, x, rsigma, gamma, sm_margin, zero_centered_gamma)
    return [dx, dgamma]


def rmsnorm_bwd_add(dz, x, add, rsigma, gamma, sm_margin, zero_centered_gamma):
    dx, dgamma = _ops.rmsnorm_bwd_add(dz, x, add, rsigma, gamma, sm_margin, zero_centered_gamma)
    return [dx, dgamma]


def layernorm_fwd(  # pylint: disable=redefined-builtin,unused-argument
    input, weight, bias, eps, out, quantizer, out_dtype, sm_margin, zero_centered_gamma
):
    """LayerNorm forward with optional quantization via stable ABI."""
    # Get raw input tensor (may be a quantized type)
    from transformer_engine.pytorch.tensor._extract import extract_tensor_data

    inp_data = input if isinstance(input, torch.Tensor) else extract_tensor_data(input)[0]
    w_data = weight if isinstance(weight, torch.Tensor) else extract_tensor_data(weight)[0]

    if quantizer is None or out is not None:
        # Unquantized path or pre-allocated output
        result_out, mu, rsigma = _ops.layernorm_fwd(
            inp_data, w_data, bias, eps, sm_margin, zero_centered_gamma
        )
        if quantizer is not None and out is not None:
            # Quantize the output in-place
            from transformer_engine.pytorch.tensor._quantize_stable import quantize_into

            quantize_into(result_out, quantizer, out)
            return [out, mu, rsigma]
        return [result_out, mu, rsigma]

    # Quantized path: try fused norm+quantize first (delayed scaling only)
    fused = _try_fused_norm_quantize_layernorm(
        inp_data, w_data, bias, eps, quantizer, sm_margin, zero_centered_gamma
    )
    if fused is not None:
        q_out, mu, rsigma = fused
        return [q_out, mu, rsigma]

    # Fallback: unfused norm then quantize
    result_out, mu, rsigma = _ops.layernorm_fwd(
        inp_data, w_data, bias, eps, sm_margin, zero_centered_gamma
    )
    from transformer_engine.pytorch.tensor._quantize_stable import quantize_new

    q_out = quantize_new(result_out, quantizer)
    _fill_fp8_transpose_if_needed(q_out)
    return [q_out, mu, rsigma]


def rmsnorm_fwd(
    input, weight, eps, out, quantizer, out_dtype, sm_margin, zero_centered_gamma
):  # pylint: disable=redefined-builtin,unused-argument
    """RMSNorm forward with optional quantization via stable ABI."""
    from transformer_engine.pytorch.tensor._extract import extract_tensor_data

    inp_data = input if isinstance(input, torch.Tensor) else extract_tensor_data(input)[0]
    w_data = weight if isinstance(weight, torch.Tensor) else extract_tensor_data(weight)[0]

    if quantizer is None or out is not None:
        result_out, rsigma = _ops.rmsnorm_fwd(inp_data, w_data, eps, sm_margin, zero_centered_gamma)
        if quantizer is not None and out is not None:
            from transformer_engine.pytorch.tensor._quantize_stable import quantize_into

            quantize_into(result_out, quantizer, out)
            return [out, None, rsigma]
        return [result_out, None, rsigma]

    # Quantized path: try fused norm+quantize first (delayed scaling only)
    fused = _try_fused_norm_quantize_rmsnorm(
        inp_data, w_data, eps, quantizer, sm_margin, zero_centered_gamma
    )
    if fused is not None:
        q_out, rsigma = fused
        return [q_out, None, rsigma]

    # Fallback: unfused norm then quantize
    result_out, rsigma = _ops.rmsnorm_fwd(inp_data, w_data, eps, sm_margin, zero_centered_gamma)
    from transformer_engine.pytorch.tensor._quantize_stable import quantize_new

    q_out = quantize_new(result_out, quantizer)
    _fill_fp8_transpose_if_needed(q_out)
    return [q_out, None, rsigma]


# ============================================================================
# NVSHMEM — implemented via stable ops + ctypes
# ============================================================================

_nvshmem_lib = None


def _get_nvshmem_lib():
    """Load libnvshmem_host.so via ctypes (cached)."""
    global _nvshmem_lib
    if _nvshmem_lib is not None:
        return _nvshmem_lib
    try:
        _nvshmem_lib = _ctypes.CDLL("libnvshmem_host.so")
        return _nvshmem_lib
    except OSError as exc:
        raise RuntimeError(
            "NVSHMEM not available. Ensure libnvshmem_host.so is installed and "
            "TE was built with NVTE_ENABLE_NVSHMEM=1."
        ) from exc


def init_nvshmem_backend(process_group):
    """Initialize NVSHMEM backend using a PyTorch distributed process group.

    Uses torch.distributed for the broadcast (replacing the old pybind c10d path)
    and ctypes for NVSHMEM init calls.
    """
    import torch.distributed as dist

    lib = _get_nvshmem_lib()
    my_rank = dist.get_rank(process_group)
    num_ranks = dist.get_world_size(process_group)

    # nvshmemx_uniqueid_t is 128 bytes
    UNIQUEID_SIZE = 128
    id_tensor = torch.zeros(UNIQUEID_SIZE, dtype=torch.uint8)

    if my_rank == 0:
        # nvshmemx_get_uniqueid(nvshmemx_uniqueid_t *id)
        lib.nvshmemx_get_uniqueid(_ctypes.c_void_p(id_tensor.data_ptr()))

    # Broadcast the unique ID from rank 0
    id_gpu = id_tensor.cuda()
    dist.broadcast(id_gpu, src=0, group=process_group)
    id_tensor.copy_(id_gpu.cpu())

    # nvshmemx_init_attr_t + nvshmemx_set_attr_uniqueid_args + nvshmemx_init_attr
    INIT_ATTR_SIZE = 256  # nvshmemx_init_attr_t struct size (generous)
    attr_buf = ((_ctypes.c_char) * INIT_ATTR_SIZE)()
    _ctypes.memset(attr_buf, 0, INIT_ATTR_SIZE)

    lib.nvshmemx_set_attr_uniqueid_args(
        _ctypes.c_int(my_rank),
        _ctypes.c_int(num_ranks),
        _ctypes.c_void_p(id_tensor.data_ptr()),
        _ctypes.byref(attr_buf),
    )

    NVSHMEMX_INIT_WITH_UNIQUEID = 1
    lib.nvshmemx_init_attr(
        _ctypes.c_int(NVSHMEMX_INIT_WITH_UNIQUEID),
        _ctypes.byref(attr_buf),
    )

    # Validate
    lib.nvshmem_my_pe.restype = _ctypes.c_int
    lib.nvshmem_n_pes.restype = _ctypes.c_int
    assert (
        my_rank == lib.nvshmem_my_pe()
    ), f"my_rank {my_rank} != nvshmem_my_pe {lib.nvshmem_my_pe()}"
    assert (
        num_ranks == lib.nvshmem_n_pes()
    ), f"num_ranks {num_ranks} != nvshmem_n_pes {lib.nvshmem_n_pes()}"


def create_nvshmem_tensor(shape, dtype):
    """Allocate a tensor in NVSHMEM shared memory.

    Uses the stable ABI from_blob with nvshmem_free deleter to create a tensor
    whose memory is allocated via nvshmem_malloc and accessible by remote PEs.
    """
    # Map torch dtype to ScalarType int for the stable op
    _DTYPE_TO_SCALAR = {
        torch.float32: 6,  # ScalarType::Float
        torch.float64: 7,  # ScalarType::Double
        torch.float16: 5,  # ScalarType::Half
        torch.bfloat16: 15,  # ScalarType::BFloat16
        torch.uint8: 0,  # ScalarType::Byte
        torch.int32: 3,  # ScalarType::Int
        torch.int64: 4,  # ScalarType::Long
    }
    scalar_type = _DTYPE_TO_SCALAR.get(dtype)
    if scalar_type is None:
        raise ValueError(f"Unsupported dtype {dtype} for nvshmem_create_tensor")

    numel = 1
    for s in shape:
        numel *= s
    device_idx = torch.cuda.current_device()

    # Stable op allocates via nvshmem_malloc and wraps with from_blob + deleter
    flat_tensor = _ops.nvshmem_create_tensor(numel, scalar_type, device_idx)
    return flat_tensor.view(shape)


def nvshmem_send_on_current_stream(src, dst, peer, signal):
    """Send tensor data to a remote PE with signal using NVSHMEM."""
    _ops.nvshmem_send_on_current_stream(src, dst, int(peer), signal)


def nvshmem_wait_on_current_stream(signal, wait_kind="stream"):
    """Wait for a signal from a remote PE using NVSHMEM."""
    wait_kind_map = {"kernel": 0, "nvshmem": 1, "stream": 2}
    wait_kind_int = wait_kind_map.get(wait_kind, 2)
    _ops.nvshmem_wait_on_current_stream(signal, wait_kind_int)


def nvshmem_finalize():
    """Finalize NVSHMEM and free resources."""
    lib = _get_nvshmem_lib()
    lib.nvshmem_finalize()


# ============================================================================
# Check if userbuffers uses MPI
# ============================================================================

# ============================================================================
# GEMM
# ============================================================================


def generic_gemm(  # pylint: disable=unused-argument,redefined-outer-name
    A,
    transa,
    B,
    transb,
    D,
    quantizer,
    out_dtype,
    bias,
    bias_type,
    gelu,
    gelu_in,
    grad,
    workspace,
    workspaceSize,
    accumulate,
    use_split_accumulator,
    comm_overlap=None,
    comm_type=None,
    extra_output=None,
    bulk_overlap=False,
    alpha=1.0,
    beta=None,
):
    """GEMM via stable ABI ops with Python-side tensor metadata extraction."""
    from transformer_engine.pytorch.tensor._extract import extract_tensor_data

    # Ensure workspace is large enough for cuBLAS. The pybind path dynamically
    # resized the workspace; the stable path receives a fixed tensor.
    _MIN_WORKSPACE = 33554432  # 32 MiB, matches pybind default
    if isinstance(workspace, torch.Tensor) and workspace.numel() < _MIN_WORKSPACE:
        workspace = torch.empty(_MIN_WORKSPACE, dtype=torch.uint8, device=workspace.device)

    A_data, A_dtype, A_scale_inv, A_sm, A_swizzled, A_cw_data, A_cw_scale_inv, A_amax, A_cw_amax = (
        _extract_gemm_operand(A, transa)
    )
    B_data, B_dtype, B_scale_inv, B_sm, B_swizzled, B_cw_data, B_cw_scale_inv, B_amax, B_cw_amax = (
        _extract_gemm_operand(B, not transb)
    )
    _TORCH_DT = {torch.float32: 4, torch.float16: 5, torch.bfloat16: 6, torch.uint8: 0}
    _TE_TO_TORCH_DT = {
        4: torch.float32,
        5: torch.float16,
        6: torch.bfloat16,
        0: torch.uint8,
        7: torch.uint8,  # kFloat8E4M3 stored as uint8
        8: torch.uint8,  # kFloat8E5M2 stored as uint8
    }

    # A tensor may be columnwise-only (rowwise data is an empty placeholder, numel=0,
    # but colwise data exists). Only skip_gemm when NO data is available at all.
    def _operand_has_data(data, cw_data):
        return data.numel() > 0 or (
            cw_data is not None and isinstance(cw_data, torch.Tensor) and cw_data.numel() > 0
        )

    skip_gemm = not _operand_has_data(A_data, A_cw_data) or not _operand_has_data(B_data, B_cw_data)

    _fused_output_quant = (
        False  # Will be set True only for fused delayed-scaling output quantization
    )
    if D is not None:
        D_data, D_dtype, D_scale_inv, D_sm = extract_tensor_data(D)
        D_amax = getattr(D, "_amax", None) or (
            getattr(quantizer, "amax", None) if quantizer else None
        )
        D_scale = getattr(quantizer, "scale", None) if quantizer else None
        if isinstance(D_amax, torch.Tensor) and D_amax.numel() == 0:
            D_amax = None
        if isinstance(D_scale, torch.Tensor) and D_scale.numel() == 0:
            D_scale = None
        # Zero stale amax/scale for CurrentScaling (matches quantize_into behavior)
        if quantizer is not None and "CurrentScaling" in type(quantizer).__name__:
            if isinstance(D_amax, torch.Tensor):
                D_amax.zero_()
            if isinstance(D_scale, torch.Tensor):
                D_scale.zero_()
    else:
        # NVTE GEMM column-major convention:
        #   A1 = last dim of A, A0 = product of all other dims
        #   B1 = last dim of B, B0 = product of all other dims
        #   k = (transa ? A1 : A0),  M = (transa ? A0 : A1)
        # Output shape mirrors pybind getGemmOutputShape:
        #   transb=True  → (B1, M)
        #   transb=False → (*B_shape[:-1], M)   — preserves multi-dim batch dims
        #
        # When A (or B) is columnwise-only, A_data is an empty placeholder (shape (0,)).
        # Derive logical dims from the columnwise buffer.
        # MXFP8Tensor: columnwise has same logical shape as rowwise → use shape[-1].
        # Float8Tensor, Float8BlockwiseQTensor: columnwise = physical transpose → use shape[0].
        def _cw_is_same_shape(t):
            return "MXFP8" in type(t).__name__

        _A_cw_is_transpose = not _cw_is_same_shape(A)
        _B_cw_is_transpose = not _cw_is_same_shape(B)
        # FP4 data is packed (2 elements per byte), so physical shape has K/2
        # in the last dim. Double it to get the logical dimension, matching
        # the C++ gemm.cpp buildInputTensorWrapper logic.
        _kFloat4E2M1 = 10
        _A_fp4 = A_dtype == _kFloat4E2M1
        _B_fp4 = B_dtype == _kFloat4E2M1
        if A_data.numel() == 0 and A_cw_data is not None:
            if _A_cw_is_transpose:
                A1 = A_cw_data.shape[0]  # physical transpose: shape[0] = last dim of logical
                A0 = A_cw_data.numel() // max(A1, 1)
            else:
                A1 = A_cw_data.shape[-1]  # same logical shape: shape[-1] = last dim
                A0 = A_cw_data.numel() // max(A1, 1)
            if _A_fp4:
                # Columnwise FP4: physical transpose [K, M/2] → logical [K, M]
                if _A_cw_is_transpose:
                    A0 *= 2
                else:
                    A1 *= 2
        else:
            A1 = A_data.shape[-1]
            A0 = A_data.numel() // max(A1, 1)
            if _A_fp4:
                # Rowwise FP4: [M, K/2] → logical [M, K]
                A1 *= 2
        if B_data.numel() == 0 and B_cw_data is not None:
            if _B_cw_is_transpose:
                B1 = B_cw_data.shape[0]
            else:
                B1 = B_cw_data.shape[-1]
            if _B_fp4:
                if _B_cw_is_transpose:
                    pass  # B1 is from shape[0], which is the last logical dim (correct for non-packed dim)
                else:
                    B1 *= 2
        else:
            B1 = B_data.shape[-1]
            if _B_fp4:
                B1 *= 2
        M = A0 if transa else A1
        if transb:
            out_shape = [B1, M]
        else:
            out_shape = list(B_data.shape[:-1]) + [M]
        if quantizer is not None:
            # Determine the logical output dtype (pybind: output_dtype = out_dtype ? *out_dtype : A_tensor.dtype())
            if isinstance(out_dtype, torch.dtype):
                _q_fake_dtype = out_dtype
            elif out_dtype is not None:
                _q_fake_dtype = _TE_TO_TORCH_DT.get(int(out_dtype), torch.bfloat16)
            else:
                _q_fake_dtype = A.dtype if isinstance(A, torch.Tensor) else torch.bfloat16

            # Decide fused vs unfused quantization.
            # The pybind path only fuses output quantization when:
            #   - Inputs are low-precision (FP8)
            #   - Output quantizer is Float8Quantizer (delayed scaling)
            #   - Inputs are per-tensor scaling (Float8Tensor, not blockwise/MXFP8/NVFP4)
            # All other cases use unfused quantization: GEMM → HP output → quantize.
            _low_precision = A_dtype in (7, 8) or B_dtype in (7, 8)  # kFloat8E4M3 or kFloat8E5M2
            _is_delayed_scaling_quantizer = type(quantizer).__name__ == "Float8Quantizer"
            _is_per_tensor_input = A_sm == 0 and B_sm == 0  # both DELAYED_TENSOR_SCALING
            _fused_output_quant = (
                _low_precision and _is_delayed_scaling_quantizer and _is_per_tensor_input
            )
            if _fused_output_quant:
                D = quantizer.make_empty(
                    out_shape,
                    dtype=_q_fake_dtype,
                    device=A_data.device,
                )
                D_data, D_dtype, D_scale_inv, D_sm = extract_tensor_data(D)
                D_amax = getattr(quantizer, "amax", None)
                D_scale = getattr(quantizer, "scale", None)
                if "CurrentScaling" in type(quantizer).__name__:
                    if isinstance(D_amax, torch.Tensor):
                        D_amax.zero_()
                    if isinstance(D_scale, torch.Tensor):
                        D_scale.zero_()
            else:
                # Unfused quantization: GEMM produces HP output, quantize separately after.
                # Use _q_fake_dtype for the intermediate HP output.
                out_dt = _q_fake_dtype
                out_te_dtype = _TORCH_DT.get(out_dt, 6)
                D = torch.empty(*out_shape, dtype=out_dt, device=A_data.device)
                D_data, D_dtype, D_scale_inv, D_sm = D, out_te_dtype, None, 0
                D_amax, D_scale = None, None
        else:
            if isinstance(out_dtype, torch.dtype):
                out_dt = out_dtype
                out_te_dtype = _TORCH_DT.get(out_dt, 6)
            elif out_dtype is not None:
                out_te_dtype = int(out_dtype)
                out_dt = _TE_TO_TORCH_DT.get(out_te_dtype, torch.bfloat16)
            else:
                out_dt = A.dtype if isinstance(A, torch.Tensor) else torch.bfloat16
                out_te_dtype = _TORCH_DT.get(out_dt, 6)
            D = torch.empty(*out_shape, dtype=out_dt, device=A_data.device)
            D_data, D_dtype, D_scale_inv, D_sm = D, out_te_dtype, None, 0
            D_amax, D_scale = None, None

    # Skip GEMM when any operand is empty (e.g., zero-token inputs in backward pass).
    # Still allocate D above so callers always get a tensor (possibly zero-element).
    if skip_gemm:
        D_tensor = extract_tensor_data(D)[0] if D is not None else D_data
        if isinstance(D_tensor, torch.Tensor) and D_tensor.numel() > 0 and not accumulate:
            D_tensor.zero_()
        # When grad=True and bias is provided, the pybind path allocates a zeroed
        # dbias tensor (at::empty → zero_()) and returns it.  Returning None here
        # would break callers that store the dbias reference for delayed wgrad
        # (e.g. GroupedLinear.backward_dw assigns bias_params[i].grad = grad_biases_[i]).
        skip_dbias = None
        if bias is not None and isinstance(bias, torch.Tensor) and bias.numel() > 0 and grad:
            bias.zero_()
            dbias_dt = D_data.dtype if isinstance(D_data, torch.Tensor) else torch.bfloat16
            skip_dbias = torch.zeros(B_data.shape[-1], dtype=dbias_dt, device=A_data.device)
        if gelu_in is not None and isinstance(gelu_in, torch.Tensor) and gelu_in.numel() > 0:
            gelu_in.zero_()
        return [D, skip_dbias, gelu_in, extra_output]

    # For FP8 delayed-tensor-scaling GEMMs on Hopper (non-Blackwell), cuBLAS only
    # supports TN layout. When A is not transposed (NN/NT), NVTE's CanonicalizeGemmInput
    # requires A to have columnwise data (the physical transpose). Similarly, when B is
    # transposed (TT/NT), B needs columnwise data.
    # Create the FP8 transpose on-the-fly when missing.
    _NVTE_DELAYED = 0  # NVTE_DELAYED_TENSOR_SCALING
    _NVTE_MXFP8 = 1  # NVTE_MXFP8_1D_SCALING
    if not transa and A_cw_data is None and A_sm == _NVTE_DELAYED and A_dtype in (7, 8):
        A_cw_data = _ops.fp8_transpose(A_data, A_dtype, None)
        A_cw_scale_inv = A_scale_inv
    if transb and B_cw_data is None and B_sm == _NVTE_DELAYED and B_dtype in (7, 8):
        B_cw_data = _ops.fp8_transpose(B_data, B_dtype, None)
        B_cw_scale_inv = B_scale_inv

    # For MXFP8: when the GEMM needs columnwise data but it's missing or uninitialized
    # (e.g., tensors produced by GEMM+GELU fusion that only have rowwise data), create
    # columnwise data on-the-fly by dequantizing rowwise and re-quantizing bidirectionally.
    def _ensure_mxfp8_columnwise(data, dtype, scale_inv, cw_data, cw_si, sm):
        """Create MXFP8 columnwise data from rowwise if missing."""
        if sm != _NVTE_MXFP8 or data is None or data.numel() == 0:
            return cw_data, cw_si
        # Dequantize rowwise data to get the high-precision source
        src = _ops.dequantize(data, dtype, scale_inv, None, sm, 6)  # 6 = kBFloat16
        # Allocate columnwise buffers if needed
        if cw_data is None:
            cw_data = torch.empty_like(data)
        if cw_si is None:
            cw_si = torch.empty_like(scale_inv)
        # Save rowwise data/scale before bidirectional quantization overwrites them
        rw_data_backup = data.clone()
        rw_si_backup = scale_inv.clone()
        # Re-quantize with both rowwise+columnwise (bidirectional)
        _ops.quantize_bidirectional(
            src,
            data,
            dtype,
            None,
            None,
            scale_inv,
            cw_data,
            cw_si,
            sm,
            False,
            0.0,
            None,
            False,  # nvfp4_2d_quantization
        )
        # Restore original rowwise data (only columnwise was needed)
        data.copy_(rw_data_backup)
        scale_inv.copy_(rw_si_backup)
        return cw_data, cw_si

    if not transa and A_sm == _NVTE_MXFP8 and A_dtype in (7, 8):
        A_cw_data, A_cw_scale_inv = _ensure_mxfp8_columnwise(
            A_data, A_dtype, A_scale_inv, A_cw_data, A_cw_scale_inv, A_sm
        )
    if transb and B_sm == _NVTE_MXFP8 and B_dtype in (7, 8):
        B_cw_data, B_cw_scale_inv = _ensure_mxfp8_columnwise(
            B_data, B_dtype, B_scale_inv, B_cw_data, B_cw_scale_inv, B_sm
        )

    # When grad=True with bias, allocate a fresh dbias tensor for the GEMM kernel to write into.
    # The pybind path does the same: at::empty({B_shape[-1]}, dtype=out_tensor.dtype).
    dbias = None
    bias_arg = bias
    if bias is not None and grad:
        dbias_dt = D_data.dtype if isinstance(D_data, torch.Tensor) else torch.bfloat16
        dbias = torch.empty(B_data.shape[-1], dtype=dbias_dt, device=A_data.device)
        bias_arg = dbias

    # GELU epilogue: when gelu=True the pybind version allocates a pre_gelu_out tensor
    # (forward) or uses the caller-provided gelu_in (backward).  The C++ GEMM only
    # enables the GELU epilogue when pre_gelu_out has a value, so we must create it here.
    if gelu and gelu_in is None and not grad:
        # Forward: allocate pre-GELU output (same shape/dtype as D output).
        # The pybind version uses gelu_type = low_precision ? bias_type : out_tensor.dtype()
        # For high-precision (non-FP8) this is the output dtype.
        _low_precision = A_dtype in (7, 8) or B_dtype in (7, 8)
        if _low_precision:
            gelu_dt = (
                bias.dtype
                if bias is not None and isinstance(bias, torch.Tensor)
                else torch.bfloat16
            )
        else:
            gelu_dt = D_data.dtype if isinstance(D_data, torch.Tensor) else torch.bfloat16
        out_shape_for_gelu = (
            list(D_data.shape) if isinstance(D_data, torch.Tensor) else list(D.shape)
        )
        gelu_in = torch.empty(out_shape_for_gelu, dtype=gelu_dt, device=A_data.device)

    if comm_overlap is not None:
        _ops.gemm_with_comm_overlap(
            A_data,
            A_dtype,
            A_scale_inv,
            A_cw_data,
            A_cw_scale_inv,
            A_sm,
            A_swizzled,
            transa,
            B_data,
            B_dtype,
            B_scale_inv,
            B_cw_data,
            B_cw_scale_inv,
            B_sm,
            B_swizzled,
            transb,
            D_data,
            D_dtype,
            D_amax,
            D_scale,
            D_scale_inv,
            D_sm,
            bias_arg,
            int(bias_type) if bias_type is not None else 0,
            gelu_in,
            workspace,
            grad,
            accumulate,
            use_split_accumulator,
            comm_overlap._handle,
            int(comm_type),
            bulk_overlap,
            extra_output,
        )
    else:
        _ops.gemm(
            A_data,
            A_dtype,
            A_scale_inv,
            A_cw_data,
            A_cw_scale_inv,
            A_sm,
            A_swizzled,
            transa,
            B_data,
            B_dtype,
            B_scale_inv,
            B_cw_data,
            B_cw_scale_inv,
            B_sm,
            B_swizzled,
            transb,
            D_data,
            D_dtype,
            D_amax,
            D_scale,
            D_scale_inv,
            D_sm,
            bias_arg,
            int(bias_type) if bias_type is not None else 0,
            gelu_in,
            workspace,
            grad,
            accumulate,
            use_split_accumulator,
            alpha,
            A_amax,
            A_cw_amax,
            B_amax,
            B_cw_amax,
        )

    # Unfused output quantization: if the GEMM produced HP output and the caller
    # requested a quantized output, quantize now (matches pybind unfused_quantization_needed).
    if quantizer is not None and not _fused_output_quant and not skip_gemm:
        D = quantizer(D)

    return [D, dbias, gelu_in, extra_output]


# ============================================================================
# Quantize (match pybind11 signatures)
# ============================================================================


def quantize(tensor, quantizer, output=None, noop=None):
    """Quantize using stable ABI ops, bypassing pybind11."""
    from transformer_engine.pytorch.tensor._quantize_stable import quantize_into, quantize_new

    if quantizer is None:
        return tensor
    if output is not None:
        quantize_into(tensor, quantizer, output, noop)
        return output
    return quantize_new(tensor, quantizer)


def dequantize(input, otype):  # pylint: disable=redefined-builtin
    """Dequantize using stable ABI ops."""
    from transformer_engine.pytorch.tensor._extract import extract_tensor_data

    in_data, in_dtype, in_scale_inv, in_sm = extract_tensor_data(input)
    _TORCH_TO_TE = {torch.float32: 4, torch.float16: 5, torch.bfloat16: 6}
    out_te_dtype = _TORCH_TO_TE.get(otype, 4) if isinstance(otype, torch.dtype) else int(otype)
    # NVFP4 dequantize kernel needs per-tensor amax for reconstruction
    in_amax = None
    for attr in ("_amax_rowwise", "_amax", "amax"):
        a = getattr(input, attr, None)
        if isinstance(a, torch.Tensor) and a.numel() > 0:
            in_amax = a
            break
    return _ops.dequantize(in_data, in_dtype, in_scale_inv, in_amax, in_sm, out_te_dtype)


multi_tensor_quantize = _not_implemented("multi_tensor_quantize")


def group_quantize(tensor, quantizer, num_tensors, first_dims):
    """Quantize a grouped tensor (multiple tensors concatenated along dim 0).

    Pure Python implementation: splits by first_dims, quantizes each chunk,
    then constructs a GroupedTensor with concatenated data/scale buffers.
    """
    from transformer_engine.pytorch.tensor._quantize_stable import quantize_new
    from transformer_engine.pytorch.tensor.grouped_tensor import GroupedTensor

    tensor = tensor.contiguous()
    N = tensor.shape[-1]
    device = tensor.device

    # Get split sizes
    if first_dims is not None:
        splits = first_dims.tolist() if isinstance(first_dims, torch.Tensor) else list(first_dims)
    else:
        M_each = tensor.shape[0] // num_tensors
        splits = [M_each] * num_tensors

    # Quantize each chunk
    chunks = []
    offset = 0
    for i in range(num_tensors):
        n = int(splits[i])
        chunk = tensor[offset : offset + n]
        chunks.append(quantize_new(chunk, quantizer))
        offset += n

    # Concatenate rowwise data and scale_inv into flat 1D buffers
    all_data = []
    all_si = []
    all_cw_data = []
    all_cw_si = []
    offsets_list = [0]
    si_offsets_list = [0]
    cw_si_offsets_list = [0]
    shapes_list = []

    for qt in chunks:
        rd = getattr(qt, "_rowwise_data", getattr(qt, "_data", None))
        if rd is None:
            rd = qt if isinstance(qt, torch.Tensor) else None
        rsi = getattr(qt, "_rowwise_scale_inv", getattr(qt, "_scale_inv", None))
        cwd = getattr(qt, "_columnwise_data", None)
        cwsi = getattr(qt, "_columnwise_scale_inv", None)
        if rd is not None:
            all_data.append(rd.reshape(-1))
        if rsi is not None:
            all_si.append(rsi.reshape(-1))
        if cwd is not None:
            all_cw_data.append(cwd.reshape(-1))
        if cwsi is not None:
            all_cw_si.append(cwsi.reshape(-1))
        # Track shapes (flattened to 2D) - use whichever data is available
        ref_data = rd if rd is not None else cwd
        M_i = ref_data.shape[0] if ref_data is not None and ref_data.ndim >= 1 else 1
        N_i = (
            ref_data.shape[-1]
            if ref_data is not None and ref_data.ndim >= 2
            else (ref_data.numel() if ref_data is not None else 0)
        )
        shapes_list.append((M_i, N_i))
        offsets_list.append(offsets_list[-1] + (rd.numel() if rd is not None else 0))
        if rsi is not None:
            si_offsets_list.append(si_offsets_list[-1] + rsi.numel())
        if cwsi is not None:
            cw_si_offsets_list.append(cw_si_offsets_list[-1] + cwsi.numel())

    flat_data = (
        torch.cat(all_data) if all_data else torch.empty(0, dtype=torch.uint8, device=device)
    )
    flat_si = torch.cat(all_si) if all_si else None
    flat_cw_data = torch.cat(all_cw_data) if all_cw_data else None
    flat_cw_si = torch.cat(all_cw_si) if all_cw_si else None

    total_M = sum(int(s) for s in splits)
    logical_shape = (total_M, N)

    # Build first_dims tensor on device
    first_dims_tensor = (
        first_dims
        if isinstance(first_dims, torch.Tensor)
        else torch.tensor(splits, dtype=torch.int64, device=device)
    )

    # Compute tensor_offsets
    tensor_offsets = torch.tensor(offsets_list[:-1], dtype=torch.int64, device=device)

    gt = GroupedTensor(
        shape=logical_shape,
        dtype=tensor.dtype,
        num_tensors=num_tensors,
        shapes=shapes_list,
        quantizer=quantizer,
        data=flat_data,
        columnwise_data=flat_cw_data,
        scale_inv=flat_si,
        columnwise_scale_inv=flat_cw_si,
        first_dims=first_dims_tensor,
        tensor_offsets=tensor_offsets,
        offsets=offsets_list,
        scale_inv_offsets=si_offsets_list if flat_si is not None else None,
        columnwise_scale_inv_offsets=cw_si_offsets_list if flat_cw_si is not None else None,
    )
    return gt


def split_quantize(
    tensor, split_sections, quantizer_list, disable_bulk_allocation=False
):  # pylint: disable=unused-argument
    """Split tensor along dim 0 and quantize each split independently.

    Python implementation of pybind split_quantize. Uses per-split quantize_new,
    matching the "UNFUSED" allocation/quantization path in the C++ version.
    The bulk-allocation optimizations (for Float8Block/MXFP8/NVFP4) are not
    implemented; correctness is preserved for all quantizer types via the unfused path.
    """
    from transformer_engine.pytorch.tensor._quantize_stable import quantize_new

    num_splits = len(split_sections)
    if num_splits == 0:
        return []
    tensor = tensor.contiguous()
    results = []
    offset = 0
    for i in range(num_splits):
        n = split_sections[i]
        split = tensor[offset : offset + n]
        quantizer = quantizer_list[i]
        if quantizer is None:
            results.append(split)
        else:
            results.append(quantize_new(split, quantizer))
        offset += n
    return results


# ============================================================================
# Swizzle (match pybind11 signature)
# ============================================================================


def swizzle_scales_for_gemm_(tensor):
    """Swizzle MXFP8/NVFP4 scales in-place for later GEMM use."""
    if getattr(tensor, "_with_gemm_swizzled_scales", False):
        return

    _, te_dtype, _, scaling_mode, _, _, _, _, _ = _extract_gemm_operand(tensor, True)

    if not hasattr(_ops, "swizzle_scale_for_gemm"):
        return

    if hasattr(tensor, "_rowwise_data") and getattr(tensor, "_rowwise_scale_inv", None) is not None:
        tensor._rowwise_scale_inv = _ops.swizzle_scale_for_gemm(
            tensor._rowwise_data, tensor._rowwise_scale_inv, te_dtype, scaling_mode
        )

    if (
        hasattr(tensor, "_columnwise_data")
        and getattr(tensor, "_columnwise_scale_inv", None) is not None
    ):
        tensor._columnwise_scale_inv = _ops.swizzle_scale_for_gemm(
            tensor._columnwise_data, tensor._columnwise_scale_inv, te_dtype, scaling_mode
        )

    if hasattr(tensor, "_scale_inv") and getattr(tensor, "_scale_inv", None) is not None:
        tensor._scale_inv = _ops.swizzle_scale_for_gemm(
            tensor._data, tensor._scale_inv, te_dtype, scaling_mode
        )

    tensor._with_gemm_swizzled_scales = True


# ============================================================================
# Activation ops (match pybind11 individual function names)
# ============================================================================


def _make_activation_fwd(act_type, shape_divisor=1):
    _TE_DTYPE = {torch.float32: 4, torch.float16: 5, torch.bfloat16: 6}
    DELAYED = 0

    def fn(input, quantizer):  # pylint: disable=redefined-builtin
        from transformer_engine.pytorch.tensor._extract import extract_tensor_data

        inp = input if isinstance(input, torch.Tensor) else extract_tensor_data(input)[0]
        out_shape = list(inp.shape)
        if shape_divisor > 1:
            out_shape[-1] //= shape_divisor
        te_dt = _TE_DTYPE.get(inp.dtype, 6)
        device = inp.device

        if quantizer is None:
            # Path: no quantization
            out = torch.empty(out_shape, dtype=inp.dtype, device=device)
            _ops.activation_fwd_noalloc(inp, out, te_dt, None, None, None, DELAYED, act_type)
            return out

        # Determine implementation path (matches C++ activation_helper dispatch)
        q_type = type(quantizer).__name__
        is_delayed = (
            "Float8Quantizer" in q_type and "Current" not in q_type and "Block" not in q_type
        )
        is_mxfp8 = "MXFP8" in q_type
        is_current_scaling = "CurrentScaling" in q_type

        if quantizer is None or is_delayed or is_mxfp8:
            # FULLY_FUSED: kernel writes directly to quantized output
            out_py = quantizer.make_empty(out_shape, dtype=inp.dtype, device=device)
            out_data, out_dtype, out_scale_inv, out_sm = extract_tensor_data(out_py)

            if is_mxfp8:
                out_sm = 1  # MXFP8_1D_SCALING
            elif is_delayed:
                out_sm = 0  # DELAYED_TENSOR_SCALING

            out_amax = getattr(quantizer, "amax", None)
            out_scale = getattr(quantizer, "scale", None)
            if isinstance(out_amax, torch.Tensor) and out_amax.numel() == 0:
                out_amax = None
            if isinstance(out_scale, torch.Tensor) and out_scale.numel() == 0:
                out_scale = None

            _ops.activation_fwd_noalloc(
                inp, out_data, out_dtype, out_amax, out_scale, out_scale_inv, out_sm, act_type
            )

            if hasattr(out_py, "_fp8_dtype") and hasattr(quantizer, "dtype"):
                out_py._fp8_dtype = quantizer.dtype

            # activation_fwd_noalloc only fills _data (rowwise). If _transpose was
            # pre-allocated by make_empty, it is still uninitialized. Mark it invalid
            # and recompute from _data so downstream GEMMs see valid columnwise data.
            if hasattr(out_py, "_transpose") and out_py._transpose is not None:
                out_py._transpose_invalid = True
                _fill_fp8_transpose_if_needed(out_py)
            return out_py

        if is_current_scaling:
            # FUSED_ACTIVATION_AMAX_FP8: activation→hp+amax, then quantize_from_amax
            amax = getattr(quantizer, "amax", torch.zeros(1, dtype=torch.float32, device=device))
            # The activation kernel uses atomicMaxFloat to accumulate the amax,
            # so it must be zeroed before each call (matches pybind path's
            # create_unquantized_tensor_with_amax which calls amax.zero_()).
            amax.zero_()
            # Compute activation to hp output WITH amax
            hp_out = torch.empty(out_shape, dtype=inp.dtype, device=device)
            _ops.activation_fwd_noalloc(inp, hp_out, te_dt, amax, None, None, DELAYED, act_type)
            # Quantize using pre-computed amax
            out_py = quantizer.make_empty(out_shape, dtype=inp.dtype, device=device)
            from transformer_engine.pytorch.tensor._quantize_stable import quantize_into

            # Set use_existing_amax so quantize_into uses quantize_from_amax
            orig = getattr(quantizer, "use_existing_amax", False)
            quantizer.use_existing_amax = True
            quantize_into(hp_out, quantizer, out_py)
            quantizer.use_existing_amax = orig
            return out_py

        # UNFUSED (block scaling, NVFP4 with post-RHT amax):
        # activation→hp, then full quantize
        hp_out = torch.empty(out_shape, dtype=inp.dtype, device=device)
        _ops.activation_fwd_noalloc(inp, hp_out, te_dt, None, None, None, DELAYED, act_type)
        from transformer_engine.pytorch.tensor._quantize_stable import quantize_new

        return quantize_new(hp_out, quantizer)

    return fn


def _make_activation_bwd(act_type):
    _TE_DTYPE = {torch.float32: 4, torch.float16: 5, torch.bfloat16: 6}
    DELAYED = 0

    def fn(grad, input, quantizer):  # pylint: disable=redefined-builtin
        from transformer_engine.pytorch.tensor._extract import extract_tensor_data

        inp = input if isinstance(input, torch.Tensor) else extract_tensor_data(input)[0]
        grad_t = grad if isinstance(grad, torch.Tensor) else extract_tensor_data(grad)[0]

        if quantizer is None:
            te_dt = _TE_DTYPE.get(inp.dtype, 6)
            out = torch.empty_like(inp)
            _ops.dactivation_noalloc(grad_t, inp, out, te_dt, None, None, None, 0, act_type)
            return out

        q_type = type(quantizer).__name__
        is_current_scaling = "CurrentScaling" in q_type

        if is_current_scaling:
            # Current scaling: compute backward activation to hp output first,
            # then quantize. This mirrors the forward path for current scaling.
            # The fused dactivation kernel for delayed scaling uses a pre-existing
            # scale, but current scaling needs to compute amax+scale on-the-fly.
            te_dt = _TE_DTYPE.get(inp.dtype, 6)
            amax = getattr(
                quantizer, "amax", torch.zeros(1, dtype=torch.float32, device=inp.device)
            )
            # The dactivation kernel uses atomicMaxFloat to accumulate the amax,
            # so it must be zeroed before each call.
            amax.zero_()
            hp_out = torch.empty_like(inp)
            _ops.dactivation_noalloc(
                grad_t, inp, hp_out, te_dt, amax, None, None, DELAYED, act_type
            )
            # Quantize using pre-computed amax
            out_py = quantizer.make_empty(list(inp.shape), dtype=inp.dtype, device=inp.device)
            from transformer_engine.pytorch.tensor._quantize_stable import quantize_into

            orig = getattr(quantizer, "use_existing_amax", False)
            quantizer.use_existing_amax = True
            quantize_into(hp_out, quantizer, out_py)
            quantizer.use_existing_amax = orig

            # Fill transpose if pre-allocated (needed for wgrad GEMM)
            _fill_fp8_transpose_if_needed(out_py)
            return out_py

        # Quantized output (delayed scaling, block scaling, MXFP8, NVFP4)
        out_py = quantizer.make_empty(list(inp.shape), dtype=inp.dtype, device=inp.device)
        out_data, out_dtype, out_scale_inv, out_sm = extract_tensor_data(out_py)

        if "Block" in q_type:
            out_sm = 3 if getattr(quantizer, "block_scaling_dim", 2) == 2 else 2  # BLOCK_2D=3, 1D=2
        elif "MXFP8" in q_type:
            out_sm = 1  # MXFP8_1D_SCALING
        elif "NVFP4" in q_type:
            out_sm = 4  # NVFP4_1D_SCALING

        out_amax = getattr(quantizer, "amax", None)
        out_scale = getattr(quantizer, "scale", None)
        if isinstance(out_amax, torch.Tensor) and out_amax.numel() == 0:
            out_amax = None
        if isinstance(out_scale, torch.Tensor) and out_scale.numel() == 0:
            out_scale = None

        _ops.dactivation_noalloc(
            grad_t, inp, out_data, out_dtype, out_amax, out_scale, out_scale_inv, out_sm, act_type
        )

        if hasattr(out_py, "_fp8_dtype") and hasattr(quantizer, "dtype"):
            out_py._fp8_dtype = quantizer.dtype

        # dactivation_noalloc only fills _data (rowwise). If _transpose was
        # pre-allocated by make_empty, it is still uninitialized. Mark it invalid
        # and recompute from _data so downstream GEMMs see valid columnwise data.
        if hasattr(out_py, "_transpose") and out_py._transpose is not None:
            out_py._transpose_invalid = True
            _fill_fp8_transpose_if_needed(out_py)
        return out_py

    return fn


# 0=gelu, 1=glu, 2=geglu, 3=qgelu, 4=qgeglu, 5=relu, 6=reglu, 7=srelu, 8=sreglu, 9=silu, 10=swiglu
gelu = _make_activation_fwd(0)
glu = _make_activation_fwd(1, 2)
geglu = _make_activation_fwd(2, 2)
qgelu = _make_activation_fwd(3)
qgeglu = _make_activation_fwd(4, 2)
relu = _make_activation_fwd(5)
reglu = _make_activation_fwd(6, 2)
srelu = _make_activation_fwd(7)
sreglu = _make_activation_fwd(8, 2)
silu = _make_activation_fwd(9)
swiglu = _make_activation_fwd(10, 2)

dgelu = _make_activation_bwd(0)
dglu = _make_activation_bwd(1)
dgeglu = _make_activation_bwd(2)
dqgelu = _make_activation_bwd(3)
dqgeglu = _make_activation_bwd(4)
drelu = _make_activation_bwd(5)
dreglu = _make_activation_bwd(6)
dsrelu = _make_activation_bwd(7)
dsreglu = _make_activation_bwd(8)
dsilu = _make_activation_bwd(9)
dswiglu = _make_activation_bwd(10)


def clamped_swiglu(input, quantizer, limit, alpha):  # pylint: disable=redefined-builtin
    inp = input if isinstance(input, torch.Tensor) else input
    out = torch.empty(*inp.shape[:-1], inp.shape[-1] // 2, dtype=inp.dtype, device=inp.device)
    _TORCH_TO_TE_DT = {
        torch.float32: int(DType.kFloat32),
        torch.float16: int(DType.kFloat16),
        torch.bfloat16: int(DType.kBFloat16),
    }
    _ops.clamped_activation_fwd_noalloc(
        inp,
        out,
        _TORCH_TO_TE_DT.get(inp.dtype, int(DType.kBFloat16)),
        None,
        None,
        None,
        0,
        limit,
        alpha,
        0,
    )
    if quantizer is not None:
        from transformer_engine.pytorch.tensor._quantize_stable import quantize_new

        out = quantize_new(out, quantizer)
    return out


def clamped_dswiglu(grad, input, quantizer, limit, alpha):  # pylint: disable=redefined-builtin
    inp = input if isinstance(input, torch.Tensor) else input
    out = torch.empty_like(inp)
    _TORCH_TO_TE_DT = {
        torch.float32: int(DType.kFloat32),
        torch.float16: int(DType.kFloat16),
        torch.bfloat16: int(DType.kBFloat16),
    }
    _ops.clamped_dactivation_noalloc(
        grad,
        inp,
        out,
        _TORCH_TO_TE_DT.get(inp.dtype, int(DType.kBFloat16)),
        None,
        None,
        None,
        0,
        limit,
        alpha,
        0,
    )
    if quantizer is not None:
        from transformer_engine.pytorch.tensor._quantize_stable import quantize_new

        out = quantize_new(out, quantizer)
    return out


# ============================================================================
# Bias ops (match pybind11 individual function names)
# ============================================================================


def bgrad_quantize(grad_output, quantizer):
    """Compute bias gradient and optionally quantize grad_output.

    Mirrors pybind bgrad_quantize: compute grad_bias via sum, quantize grad_output
    into an FP8 tensor (Float8Quantizer / MXFP8), or return unchanged otherwise.
    """
    bias_size = grad_output.shape[-1]
    grad_bias = grad_output.reshape(-1, bias_size).sum(dim=0)
    if quantizer is None:
        return [grad_bias, grad_output]
    # Quantize grad_output into the appropriate FP8/quantized format
    from transformer_engine.pytorch.tensor._quantize_stable import quantize_new

    grad_input = quantize_new(grad_output.contiguous(), quantizer)
    return [grad_bias, grad_input]


def _make_dbias_dact(act_type, fused_act_type):
    """Create a dbias+dactivation backward function.

    act_type: index into the full dact kernel table (0-10), used by dactivation_noalloc.
    fused_act_type: index into the compact fused dact+dbias table in bias.cpp (0-4),
        used by dact_dbias_noalloc. Mapping: 0=dgelu, 1=dsilu, 2=drelu, 3=dqgelu, 4=dsrelu.
    """

    def fn(grad_output, act_input, quantizer):
        from transformer_engine.pytorch.tensor._extract import extract_tensor_data

        bias_size = act_input.shape[-1]
        _TORCH_TO_TE_DT = {
            torch.float32: int(DType.kFloat32),
            torch.float16: int(DType.kFloat16),
            torch.bfloat16: int(DType.kBFloat16),
        }
        in_te_dt = _TORCH_TO_TE_DT.get(act_input.dtype, int(DType.kBFloat16))
        device = act_input.device

        q_name = type(quantizer).__name__ if quantizer is not None else ""
        is_mxfp8 = "MXFP8" in q_name

        # Float8Quantizer (delayed scaling) fused dact+dbias+quantize kernel requires
        # output TensorWrapper with BOTH rowwise and columnwise buffers to work on Hopper
        # (SM < 10.0). Our stable path only provides rowwise output, so use the unfused
        # path for delayed FP8 and let update_usage() create columnwise lazily later.
        if quantizer is None or not is_mxfp8:
            # Unfused: compute dact in bf16, then sum for bias, then quantize separately
            temp = torch.empty_like(act_input)
            _ops.dactivation_noalloc(
                grad_output, act_input, temp, in_te_dt, None, None, None, 0, act_type
            )
            grad_bias = temp.view(-1, bias_size).sum(dim=0)
            if quantizer is not None:
                from transformer_engine.pytorch.tensor._quantize_stable import quantize_new

                grad_input = quantize_new(temp, quantizer)
            else:
                grad_input = temp
        else:
            # Fused path (MXFP8 only): use dact_dbias_noalloc with MXFP8 output.
            # Float8Quantizer (delayed) is handled in the unfused branch above.
            out_sm = 1  # MXFP8_1D=1
            out_te_dt = int(getattr(quantizer, "dtype", DType.kFloat8E4M3))
            out = quantizer.make_empty(list(act_input.shape), dtype=act_input.dtype, device=device)
            out_data, _out_dtype, out_scale_inv, _ = extract_tensor_data(out)
            out_amax = getattr(quantizer, "amax", None)
            out_scale = getattr(quantizer, "scale", None)
            if isinstance(out_amax, torch.Tensor) and out_amax.numel() == 0:
                out_amax = None
            if isinstance(out_scale, torch.Tensor) and out_scale.numel() == 0:
                out_scale = None
            grad_bias = torch.empty(bias_size, dtype=act_input.dtype, device=device)
            _ops.dact_dbias_noalloc(
                grad_output,
                act_input,
                grad_bias,
                out_data,
                out_te_dt,
                out_amax,
                out_scale,
                out_scale_inv,
                out_sm,
                fused_act_type,
            )
            grad_input = out
        return [grad_bias, grad_input]

    return fn


# C++ dact_table order (full, for dactivation_noalloc):
#   0=dgelu, 1=dglu, 2=dgeglu, 3=dqgelu, 4=dqgeglu,
#   5=drelu, 6=dreglu, 7=dsrelu, 8=dsreglu, 9=dsilu, 10=dswiglu
# C++ fused_table order (compact, for dact_dbias_noalloc in bias.cpp):
#   0=dgelu, 1=dsilu, 2=drelu, 3=dqgelu, 4=dsrelu
dbias_dgelu = _make_dbias_dact(act_type=0, fused_act_type=0)
dbias_dsilu = _make_dbias_dact(act_type=9, fused_act_type=1)
dbias_drelu = _make_dbias_dact(act_type=5, fused_act_type=2)
dbias_dqgelu = _make_dbias_dact(act_type=3, fused_act_type=3)
dbias_dsrelu = _make_dbias_dact(act_type=7, fused_act_type=4)

# ============================================================================
# Grouped GEMM
# ============================================================================

_TORCH_DT = {torch.float32: 4, torch.float16: 5, torch.bfloat16: 6, torch.uint8: 0}
_TE_TO_TORCH_DT = {
    4: torch.float32,
    5: torch.float16,
    6: torch.bfloat16,
    0: torch.uint8,
    7: torch.uint8,  # kFloat8E4M3 stored as uint8
    8: torch.uint8,  # kFloat8E5M2 stored as uint8
}


def _quantizer_to_te_dtype(quantizer):
    """Return TE DType int for a quantizer's output dtype (or kBFloat16 if unknown)."""
    if quantizer is None:
        return int(DType.kBFloat16)
    dt = getattr(quantizer, "dtype", None)
    if dt is not None:
        return int(dt)
    return int(DType.kBFloat16)


def _quantizer_to_scaling_mode(quantizer):
    """Return NVTEScalingMode int for a quantizer."""
    if quantizer is None:
        return 0  # DELAYED_TENSOR_SCALING
    qname = type(quantizer).__name__
    if "MXFP8" in qname:
        return 1
    if "NVFP4" in qname:
        return 4
    if "Block" in qname:
        block_dim = getattr(quantizer, "block_scaling_dim", 2)
        return 3 if block_dim == 2 else 2
    return 0  # DELAYED_TENSOR_SCALING


def _grouped_tensor_to_stable_args(gt):
    """Extract flat buffer args from a Python GroupedTensor for stable C++ grouped GEMM ops.

    Returns a tuple of 13 values matching the grouped_gemm C++ op parameter order:
      (rowwise_data, columnwise_data, scale_inv, columnwise_scale_inv,
       first_dims, last_dims, tensor_offsets,
       te_dtype, scaling_mode, logical_0, logical_1, num_tensors, swizzled)
    """
    quantizer = getattr(gt, "quantizer", None)
    logical_shape = gt.logical_shape
    return (
        gt.rowwise_data,
        gt.columnwise_data,
        gt.scale_inv,
        gt.columnwise_scale_inv,
        gt.first_dims,
        gt.last_dims,
        gt.tensor_offsets,
        _quantizer_to_te_dtype(quantizer),
        _quantizer_to_scaling_mode(quantizer),
        logical_shape[0],
        logical_shape[1],
        gt.num_tensors,
        bool(getattr(gt, "_with_gemm_swizzled_scales", False)),
    )


def te_general_grouped_gemm(  # pylint: disable=unused-argument
    A,
    transa,
    B,
    transb,
    D,
    out_dtype,
    m_splits,
    bias,
    bias_type,
    single_output,
    pre_gelu_out,
    grad,
    workspace,
    workspace_size,
    accumulate,
    use_split_accumulator,
    math_sm_count,
):
    """Grouped GEMM via stable ABI: iterate and call _ops.gemm() for each pair.

    Replaces pybind11 te_general_grouped_gemm which calls nvte_multi_tensor_gemm.
    Multi-stream parallelism is not preserved but correctness is maintained.
    """
    from transformer_engine.pytorch.tensor._extract import extract_tensor_data

    num_gemms = len(A)

    # Workspace: multi-stream returns a list; single-stream is a Tensor
    ws = workspace[0] if isinstance(workspace, (list, tuple)) else workspace

    # Handle single_output: D[0] is one flat tensor; slice into per-gemm sub-views
    if single_output and D is not None:
        assert m_splits is not None, "single_output requires m_splits"
        flat_D = D[0]
        D_list = []
        offset = 0
        for m in m_splits:
            D_list.append(flat_D[offset : offset + m])
            offset += m
    else:
        D_list = list(D) if D is not None else [None] * num_gemms

    bias_type_int = int(bias_type) if bias_type is not None else int(DType.kBFloat16)

    for i in range(num_gemms):
        Ai, Bi = A[i], B[i]
        Di = D_list[i]

        # Handle empty pair (matches pybind zero-and-continue behaviour)
        def _numel(t):
            if isinstance(t, torch.Tensor):
                return t.numel()
            for attr in ("_data", "_rowwise_data", "_columnwise_data"):
                d = getattr(t, attr, None)
                if isinstance(d, torch.Tensor):
                    return d.numel()
            return 1  # unknown type, assume non-empty

        if _numel(Ai) == 0 or _numel(Bi) == 0:
            if Di is not None and Di.numel() > 0 and not accumulate:
                Di.zero_()
            if bias[i].numel() > 0 and grad:
                bias[i].zero_()
            if pre_gelu_out[i].numel() > 0:
                pre_gelu_out[i].zero_()
            continue

        A_data, A_te_dtype, A_si, A_sm, A_swizzled, A_cw, A_cw_si, A_amax_i, A_cw_amax_i = (
            _extract_gemm_operand(Ai, transa)
        )
        B_data, B_te_dtype, B_si, B_sm, B_swizzled, B_cw, B_cw_si, B_amax_i, B_cw_amax_i = (
            _extract_gemm_operand(Bi, not transb)
        )

        # Mirror generic_gemm: compute on-the-fly transpose when delayed-scaling FP8
        # tensor is missing its columnwise buffer (e.g. _transpose_invalid=True).
        _NVTE_DELAYED = 0
        if not transa and A_cw is None and A_sm == _NVTE_DELAYED and A_te_dtype in (7, 8):
            A_cw = _ops.fp8_transpose(A_data, A_te_dtype, None)
            A_cw_si = A_si
        if transb and B_cw is None and B_sm == _NVTE_DELAYED and B_te_dtype in (7, 8):
            B_cw = _ops.fp8_transpose(B_data, B_te_dtype, None)
            B_cw_si = B_si

        if Di is None:
            # Allocate output: column-major convention → shape (N, M)
            _kFloat4E2M1 = 10
            A1 = A_data.shape[-1]
            A0 = A_data.numel() // max(A1, 1)
            B1 = B_data.shape[-1]
            # FP4 packed: double last dim for logical shape
            if A_te_dtype == _kFloat4E2M1:
                A1 *= 2
            if B_te_dtype == _kFloat4E2M1:
                B1 *= 2
            M = A0 if transa else A1
            N = B1 if transb else B_data.shape[-2] if B_data.ndim > 1 else B1
            out_te_int = int(out_dtype) if out_dtype is not None else int(DType.kBFloat16)
            out_dt = _TE_TO_TORCH_DT.get(out_te_int, torch.bfloat16)
            Di = torch.empty(N, M, dtype=out_dt, device=A_data.device)

        D_data, D_te_dtype, D_si, D_sm = extract_tensor_data(Di)

        bias_i = bias[i] if bias[i].numel() > 0 else None
        gelu_i = pre_gelu_out[i] if pre_gelu_out[i].numel() > 0 else None

        # For grad=True with bias: the kernel writes dbias into bias_i in-place,
        # which is already the pre-allocated grad_bias tensor passed by the caller.
        _ops.gemm(
            A_data,
            A_te_dtype,
            A_si,
            A_cw,
            A_cw_si,
            A_sm,
            A_swizzled,
            transa,
            B_data,
            B_te_dtype,
            B_si,
            B_cw,
            B_cw_si,
            B_sm,
            B_swizzled,
            transb,
            D_data,
            D_te_dtype,
            None,
            None,
            D_si,
            D_sm,
            bias_i,
            bias_type_int,
            gelu_i,
            ws,
            grad,
            accumulate,
            use_split_accumulator,
            1.0,
            A_amax_i,
            A_cw_amax_i,
            B_amax_i,
            B_cw_amax_i,
        )

        if single_output and D is not None:
            # The D_list slice is already a view into D[0]; no copy needed
            pass

    return bias


def te_general_grouped_gemm_for_grouped_tensor(
    A,
    transa,
    B,
    transb,
    D,
    bias,
    alpha,
    beta,
    workspace_setup,
    workspace_cublas,
    use_split_accumulator,
    math_sm_count,
):
    """Grouped GEMM for GroupedTensor inputs (Blackwell+ nvte_grouped_gemm)."""
    A_args = _grouped_tensor_to_stable_args(A)
    B_args = _grouped_tensor_to_stable_args(B)
    D_args = _grouped_tensor_to_stable_args(D)

    if bias is not None:
        bias_args = _grouped_tensor_to_stable_args(bias)
        has_bias = True
    else:
        bias_args = (
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            int(DType.kBFloat16),
            0,
            1,
            1,
            1,
            False,
        )
        has_bias = False

    _ops.grouped_gemm_for_grouped_tensor(
        *A_args,
        transa,
        *B_args,
        transb,
        *D_args,
        alpha,
        beta,
        workspace_setup,
        workspace_cublas,
        use_split_accumulator,
        math_sm_count,
        has_bias,
        *bias_args,
    )
    return D


def te_general_grouped_gemm_for_discrete_in(
    A,
    transa,
    B,
    transb,
    D,
    bias,
    alpha,
    beta,
    workspace_setup,
    workspace_cublas,
    use_split_accumulator,
    math_sm_count,
):
    """Grouped GEMM with discrete A list, GroupedTensor B/D (Blackwell+)."""
    B_args = _grouped_tensor_to_stable_args(B)
    D_args = _grouped_tensor_to_stable_args(D)

    if bias is not None:
        bias_args = _grouped_tensor_to_stable_args(bias)
        has_bias = True
    else:
        bias_args = (
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            int(DType.kBFloat16),
            0,
            1,
            1,
            1,
            False,
        )
        has_bias = False

    # Pack A tensors: each element of A is an individual tensor (weight per expert).
    # We pass per-tensor fields as flat packed int64 pointer tensors.
    num_a = len(A)
    device = B_args[0].device if B_args[0] is not None else alpha.device
    A_rowwise_ptrs = torch.zeros(num_a, dtype=torch.int64, device=device)
    A_colwise_ptrs = torch.zeros(num_a, dtype=torch.int64, device=device)
    A_si_ptrs = torch.zeros(num_a, dtype=torch.int64, device=device)
    A_csi_ptrs = torch.zeros(num_a, dtype=torch.int64, device=device)
    A_shapes = torch.zeros(num_a, 2, dtype=torch.int64, device="cpu")
    A_te_dtypes = torch.zeros(num_a, dtype=torch.int32, device="cpu")
    A_scaling_modes = torch.zeros(num_a, dtype=torch.int32, device="cpu")
    for i, Ai in enumerate(A):
        ai_data, ai_dtype, ai_si, ai_sm, _, ai_cw, ai_cw_si, _, _ = _extract_gemm_operand(
            Ai, transa
        )
        if ai_data is not None and ai_data.numel() > 0:
            A_rowwise_ptrs[i] = ai_data.data_ptr()
            A_shapes[i, 0] = ai_data.shape[0]
            A_shapes[i, 1] = ai_data.shape[1] if ai_data.ndim > 1 else 1
        if ai_cw is not None and ai_cw.numel() > 0:
            A_colwise_ptrs[i] = ai_cw.data_ptr()
        if ai_si is not None and ai_si.numel() > 0:
            A_si_ptrs[i] = ai_si.data_ptr()
        if ai_cw_si is not None and ai_cw_si.numel() > 0:
            A_csi_ptrs[i] = ai_cw_si.data_ptr()
        A_te_dtypes[i] = ai_dtype
        A_scaling_modes[i] = ai_sm

    _ops.grouped_gemm_for_discrete_in(
        A_rowwise_ptrs,
        A_colwise_ptrs,
        A_si_ptrs,
        A_csi_ptrs,
        A_shapes.to(device),
        A_te_dtypes.to(device),
        A_scaling_modes.to(device),
        num_a,
        *B_args,
        transb,
        *D_args,
        alpha,
        beta,
        workspace_setup,
        workspace_cublas,
        use_split_accumulator,
        math_sm_count,
        has_bias,
        *bias_args,
    )
    return D


def te_general_grouped_gemm_for_discrete_out(  # pylint: disable=unused-argument
    A,
    transa,
    B,
    transb,
    D,
    bias,
    alpha,
    beta,
    workspace_setup,
    workspace_cublas,
    use_split_accumulator,
    math_sm_count,
):
    """Grouped GEMM with GroupedTensor A/B, discrete D list (Blackwell+)."""
    A_args = _grouped_tensor_to_stable_args(A)
    B_args = _grouped_tensor_to_stable_args(B)

    num_d = len(D)
    device = A_args[0].device if A_args[0] is not None else alpha.device
    D_rowwise_ptrs = torch.zeros(num_d, dtype=torch.int64, device=device)
    D_si_ptrs = torch.zeros(num_d, dtype=torch.int64, device=device)
    D_shapes = torch.zeros(num_d, 2, dtype=torch.int64, device="cpu")
    D_te_dtypes = torch.zeros(num_d, dtype=torch.int32, device="cpu")
    D_scaling_modes = torch.zeros(num_d, dtype=torch.int32, device="cpu")
    from transformer_engine.pytorch.tensor._extract import extract_tensor_data

    for i, Di in enumerate(D):
        d_data, d_dtype, d_si, d_sm = extract_tensor_data(Di)
        if d_data is not None and d_data.numel() > 0:
            D_rowwise_ptrs[i] = d_data.data_ptr()
            D_shapes[i, 0] = d_data.shape[0]
            D_shapes[i, 1] = d_data.shape[1] if d_data.ndim > 1 else 1
        if d_si is not None and d_si.numel() > 0:
            D_si_ptrs[i] = d_si.data_ptr()
        D_te_dtypes[i] = d_dtype
        D_scaling_modes[i] = d_sm

    _ops.grouped_gemm_for_discrete_out(
        *A_args,
        transa,
        *B_args,
        transb,
        D_rowwise_ptrs,
        D_si_ptrs,
        D_shapes.to(device),
        D_te_dtypes.to(device),
        D_scaling_modes.to(device),
        num_d,
        alpha,
        beta,
        workspace_setup,
        workspace_cublas,
        use_split_accumulator,
        math_sm_count,
    )
    return D


# ============================================================================
# NVFP4 multi-tensor ops (iterate using single-tensor stable ops)
# ============================================================================


def nvfp4_multi_tensor_fused_scale(
    block_amax_list,
    global_amax_list,
    per_block_scale_list,
    target_scale_list,
    target_amax_list,
    tile_rows_list,
    tile_cols_list,
    rows_padded_list,
    block_len,
):
    for i, block_amax in enumerate(block_amax_list):
        _ops.nvfp4_fused_scale(
            block_amax,
            global_amax_list[i],
            per_block_scale_list[i],
            target_scale_list[i],
            target_amax_list[i],
            tile_rows_list[i],
            tile_cols_list[i],
            rows_padded_list[i],
            block_len,
        )


def nvfp4_2d_multi_tensor_transpose(
    rowwise_data_list,
    columnwise_data_list,
    rowwise_scale_inv_list,
    columnwise_scale_inv_list,
    M_list,
    K_list,
):
    for i, _rowwise_data in enumerate(rowwise_data_list):
        _ops.nvfp4_data_transpose(_rowwise_data, columnwise_data_list[i])
        M = M_list[i]
        K = K_list[i]
        M_tiles = (M + 15) // 16
        K_tiles = (K + 15) // 16
        _ops.nvfp4_2d_scale_transpose(
            rowwise_scale_inv_list[i], columnwise_scale_inv_list[i], M_tiles, K_tiles
        )


def nvfp4_multi_tensor_2d_partial_cast(
    inp_list,
    out_list,
    scale_list,
    global_scale_list,
    h_list,
    w_list,
    start_offset_list,
    block_len=16,
):
    for i, _inp in enumerate(inp_list):
        # out_list[i] may be a quantized tensor — extract raw data
        out = out_list[i]
        if isinstance(out, torch.Tensor):
            _ops.nvfp4_2d_partial_cast_noalloc(
                inp_list[i],
                out,
                0,  # kByte — raw uint8 data buffer type
                None,
                4,
                scale_list[i],
                global_scale_list[i],
                h_list[i],
                w_list[i],
                start_offset_list[i],
                block_len,
            )


def nvfp4_multi_tensor_compute_partial_amax(
    master_weight_list,
    partial_amax_list,
    global_amax_list,
    h_list,
    w_list,
    start_offset_list,
    block_len=16,
):
    for i, _master_weight in enumerate(master_weight_list):
        _ops.nvfp4_2d_compute_partial_amax(
            _master_weight,
            partial_amax_list[i],
            h_list[i],
            w_list[i],
            start_offset_list[i],
            block_len,
        )
        _ops.compute_amax(partial_amax_list[i], global_amax_list[i])


# ============================================================================
# Multi-tensor ops (match pybind11 signatures with pointer packing)
# ============================================================================


def _pack_tensor_lists(tensor_lists):
    """Pack tensor lists into flat int64 tensors for the pointer-pack pattern."""
    num_lists = len(tensor_lists)
    num_tensors = len(tensor_lists[0])
    ptrs = torch.tensor([t.data_ptr() for lst in tensor_lists for t in lst], dtype=torch.int64)
    shapes = torch.tensor(
        [[t.numel(), t.element_size()] for lst in tensor_lists for t in lst], dtype=torch.int64
    ).flatten()
    _TORCH_DT = {
        torch.float32: 4,
        torch.float16: 5,
        torch.bfloat16: 6,
        torch.uint8: 0,
        torch.int16: 1,
        torch.int32: 2,
        torch.int64: 3,
        torch.bool: 0,
    }
    dtypes = torch.tensor(
        [_TORCH_DT.get(t.dtype, 4) for lst in tensor_lists for t in lst], dtype=torch.int64
    )
    return ptrs, shapes, dtypes, num_lists, num_tensors


def multi_tensor_scale(chunk_size, is_infinite, tensor_lists, scale):
    ptrs, shapes, dtypes, nl, nt = _pack_tensor_lists(tensor_lists)
    _ops.multi_tensor_scale(chunk_size, is_infinite, ptrs, shapes, dtypes, nl, nt, scale)


def multi_tensor_scale_tensor(chunk_size, is_infinite, tensor_lists, scale):
    ptrs, shapes, dtypes, nl, nt = _pack_tensor_lists(tensor_lists)
    _ops.multi_tensor_scale_tensor(chunk_size, is_infinite, ptrs, shapes, dtypes, nl, nt, scale)


def multi_tensor_l2norm(chunk_size, noop_flag, tensor_lists, per_tensor=False):
    ptrs, shapes, dtypes, nl, nt = _pack_tensor_lists(tensor_lists)
    return _ops.multi_tensor_l2norm(chunk_size, noop_flag, ptrs, shapes, dtypes, nl, nt, per_tensor)


def multi_tensor_unscale_l2norm(chunk_size, noop_flag, tensor_lists, inv_scale, per_tensor=False):
    ptrs, shapes, dtypes, nl, nt = _pack_tensor_lists(tensor_lists)
    return _ops.multi_tensor_unscale_l2norm(
        chunk_size, noop_flag, ptrs, shapes, dtypes, nl, nt, inv_scale, per_tensor
    )


def multi_tensor_adam(
    chunk_size,
    noop_flag,
    tensor_lists,
    lr,
    beta1,
    beta2,
    epsilon,
    step,
    mode,
    bias_correction,
    weight_decay,
):
    ptrs, shapes, dtypes, nl, nt = _pack_tensor_lists(tensor_lists)
    _ops.multi_tensor_adam(
        chunk_size,
        noop_flag,
        ptrs,
        shapes,
        dtypes,
        nl,
        nt,
        lr,
        beta1,
        beta2,
        epsilon,
        step,
        mode,
        bias_correction,
        weight_decay,
    )


def multi_tensor_adam_capturable(
    chunk_size,
    noop_flag,
    tensor_lists,
    lr,
    beta1,
    beta2,
    epsilon,
    step,
    mode,
    bias_correction,
    weight_decay,
    inv_scale,
):
    ptrs, shapes, dtypes, nl, nt = _pack_tensor_lists(tensor_lists)
    _ops.multi_tensor_adam_capturable(
        chunk_size,
        noop_flag,
        ptrs,
        shapes,
        dtypes,
        nl,
        nt,
        lr,
        beta1,
        beta2,
        epsilon,
        step,
        mode,
        bias_correction,
        weight_decay,
        inv_scale,
    )


def multi_tensor_adam_capturable_master(
    chunk_size,
    noop_flag,
    tensor_lists,
    lr,
    beta1,
    beta2,
    epsilon,
    step,
    mode,
    bias_correction,
    weight_decay,
    inv_scale,
):
    ptrs, shapes, dtypes, nl, nt = _pack_tensor_lists(tensor_lists)
    _ops.multi_tensor_adam_capturable_master(
        chunk_size,
        noop_flag,
        ptrs,
        shapes,
        dtypes,
        nl,
        nt,
        lr,
        beta1,
        beta2,
        epsilon,
        step,
        mode,
        bias_correction,
        weight_decay,
        inv_scale,
    )


def multi_tensor_adam_param_remainder(
    chunk_size,
    noop_flag,
    tensor_lists,
    lr,
    beta1,
    beta2,
    epsilon,
    step,
    mode,
    bias_correction,
    weight_decay,
):
    ptrs, shapes, dtypes, nl, nt = _pack_tensor_lists(tensor_lists)
    _ops.multi_tensor_adam_param_remainder(
        chunk_size,
        noop_flag,
        ptrs,
        shapes,
        dtypes,
        nl,
        nt,
        lr,
        beta1,
        beta2,
        epsilon,
        step,
        mode,
        bias_correction,
        weight_decay,
    )


def multi_tensor_adam_fp8(
    chunk_size,
    noop_flag,
    tensor_lists,
    lr,
    beta1,
    beta2,
    epsilon,
    step,
    mode,
    bias_correction,
    weight_decay,
    fp8_dtype,
):
    ptrs, shapes, dtypes, nl, nt = _pack_tensor_lists(tensor_lists)
    _ops.multi_tensor_adam_fp8(
        chunk_size,
        noop_flag,
        ptrs,
        shapes,
        dtypes,
        nl,
        nt,
        lr,
        beta1,
        beta2,
        epsilon,
        step,
        mode,
        bias_correction,
        weight_decay,
        int(fp8_dtype),
    )


def multi_tensor_sgd(
    chunk_size,
    noop_flag,
    tensor_lists,
    wd,
    momentum,
    dampening,
    lr,
    nesterov,
    first_run,
    wd_after_momentum,
    scale,
):
    ptrs, shapes, dtypes, nl, nt = _pack_tensor_lists(tensor_lists)
    _ops.multi_tensor_sgd(
        chunk_size,
        noop_flag,
        ptrs,
        shapes,
        dtypes,
        nl,
        nt,
        wd,
        momentum,
        dampening,
        lr,
        nesterov,
        first_run,
        wd_after_momentum,
        scale,
    )


def multi_tensor_compute_scale_and_scale_inv(
    chunk_size, noop_flag, tensor_lists, max_fp8, force_pow_2_scales=False, epsilon=0.0
):
    ptrs, shapes, dtypes, nl, nt = _pack_tensor_lists(tensor_lists)
    _ops.multi_tensor_compute_scale_and_scale_inv(
        chunk_size, noop_flag, ptrs, shapes, dtypes, nl, nt, max_fp8, force_pow_2_scales, epsilon
    )


def multi_tensor_compute_scale_inv_e8m0(
    chunk_size, dummy, tensor_lists
):  # pylint: disable=unused-argument
    ptrs, shapes, dtypes, nl, nt = _pack_tensor_lists(tensor_lists)
    # Pass a dummy CUDA tensor to drive dispatch to CUDA backend
    dummy_cuda = torch.empty(1, device="cuda", dtype=torch.int64)
    _ops.multi_tensor_compute_scale_inv_e8m0(chunk_size, dummy_cuda, ptrs, shapes, dtypes, nl, nt)


# ============================================================================
# CommOverlap types and classes
# ============================================================================


class CommOverlapType(IntEnum):
    RS = 0
    AG = 1


class CommOverlapAlgo(IntEnum):
    BULK_OVERLAP_AG = 0
    BULK_OVERLAP_RS = 1
    SPLIT_PIPELINED_AG_P2P = 2
    SPLIT_PIPELINED_RS = 3
    SPLIT_PIPELINED_RS_P2P = 4
    ATOMIC_GEMM_RS = 5
    ATOMIC_GEMM_AG_P2P = 6
    ATOMIC_GEMM_RS_P2P = 7
    EXTERNAL_BULK_OVERLAP_AG = 8


class Float8BlockScaleTensorFormat(IntEnum):
    GEMM_READY = 0
    COMPACT = 1
    INVALID = 2


class CommOverlapCore:
    def __init__(self):
        pass

    def is_atomic_gemm(self):
        return False

    def is_p2p_overlap(self):
        return False

    def is_fp8_ubuf(self):
        return False


class CommOverlapBase(CommOverlapCore):
    pass


class CommOverlapP2PBase(CommOverlapCore):
    pass


_AllgatherCB = _ctypes.CFUNCTYPE(
    None,
    _ctypes.c_void_p,
    _ctypes.c_size_t,
    _ctypes.c_void_p,
    _ctypes.c_size_t,
    _ctypes.c_char_p,
)
_BarrierCB = _ctypes.CFUNCTYPE(None, _ctypes.c_char_p)

_TORCH_TO_TE_DTYPE = {
    torch.float32: 4,
    torch.float16: 5,
    torch.bfloat16: 6,
    torch.uint8: 0,
    torch.int8: 0,
}


class CommOverlapHelper:
    """Python replacement for pybind11 CommOverlapHelper."""

    def __init__(self, world_group=None, intra_domain_group=None):
        if world_group is None:
            raise RuntimeError(
                "CommOverlapHelper requires a process group (MPI-only builds "
                "are not supported in the stable ABI path)"
            )
        self.myrank = torch.distributed.get_rank(world_group)
        self.numranks = torch.distributed.get_world_size(world_group)
        backend = torch.distributed.get_backend(world_group)
        self.backend_is_nccl = backend == "nccl"
        if intra_domain_group is not None:
            self.mylocal = torch.distributed.get_rank(intra_domain_group)
            self.numlocal = torch.distributed.get_world_size(intra_domain_group)
            if self.numlocal == self.numranks:
                self.mynode, self.numnodes = 0, 1
            else:
                self.mynode = self.myrank // self.numlocal
                self.numnodes = self.numranks // self.numlocal
        else:
            self.mylocal = self.myrank
            self.numlocal = self.numranks
            self.mynode, self.numnodes = 0, 1
        self._groups = {
            "world": world_group,
            "intra": intra_domain_group if intra_domain_group is not None else world_group,
        }
        self.initialized = True

    def ub_allgather(self, globaldata_ptr, globalbytes, localdata_ptr, localbytes, group_name):
        group = self._groups.get(group_name, self._groups["world"])
        num_ranks = torch.distributed.get_world_size(group)
        local_buf = (_ctypes.c_uint8 * localbytes).from_address(localdata_ptr)
        local_tensor = torch.frombuffer(local_buf, dtype=torch.uint8).clone()
        if self.backend_is_nccl:
            local_tensor = local_tensor.cuda()
        chunks = [torch.empty_like(local_tensor) for _ in range(num_ranks)]
        torch.distributed.all_gather(chunks, local_tensor, group=group)
        global_tensor = torch.cat(chunks)
        if self.backend_is_nccl:
            global_tensor = global_tensor.cpu()
        _ctypes.memmove(globaldata_ptr, global_tensor.data_ptr(), globalbytes)

    def ub_barrier(self, group_name):
        group = self._groups.get(group_name, self._groups["world"])
        torch.distributed.barrier(group=group)


def _make_comm_callbacks(helper):
    """Create ctypes callback objects bound to a CommOverlapHelper."""
    h = helper

    @_AllgatherCB
    def _ag_cb(gptr, gb, lptr, lb, grp):
        h.ub_allgather(gptr, gb, lptr, lb, grp.decode() if grp else "world")

    @_BarrierCB
    def _bar_cb(grp):
        h.ub_barrier(grp.decode() if grp else "world")

    return _ag_cb, _bar_cb


class CommOverlap:
    """Python replacement for pybind11 CommOverlap (handle-based stable ABI)."""

    def __init__(
        self,
        shape,
        dtype,
        helper,
        tp_size,
        num_splits=3,
        num_max_streams=3,
        comm_cga_size=2,
        gemm_priority=0,
        comm_priority=0,
        num_comm_sm=16,
        set_sm_margin=True,
        atomic_gemm=False,
        rs_overlap_first_gemm=False,
    ):
        self._ag_cb, self._bar_cb = _make_comm_callbacks(helper)
        ag_ptr = _ctypes.cast(self._ag_cb, _ctypes.c_void_p).value
        bar_ptr = _ctypes.cast(self._bar_cb, _ctypes.c_void_p).value
        _ops.register_comm_callbacks(ag_ptr, bar_ptr)

        buf_dtype = _TORCH_TO_TE_DTYPE.get(dtype, 6)
        self._handle = _ops.create_comm_overlap(
            list(shape),
            buf_dtype,
            helper.myrank,
            helper.numranks,
            helper.mylocal,
            helper.numlocal,
            helper.mynode,
            helper.numnodes,
            tp_size,
            num_splits,
            num_max_streams,
            comm_cga_size,
            gemm_priority,
            comm_priority,
            num_comm_sm,
            set_sm_margin,
            atomic_gemm,
            rs_overlap_first_gemm,
        )

    def copy_into_buffer(self, input, local_chunk=False):  # pylint: disable=redefined-builtin
        _ops.comm_overlap_copy_into_buffer(input, self._handle, local_chunk)

    def get_buffer(self, local_chunk=False, shape=None):
        if shape is not None and len(shape) >= 2:
            dim0, dim1 = shape[0], shape[1]
        else:
            dim0, dim1 = -1, -1
        return _ops.comm_overlap_get_buffer(self._handle, local_chunk, dim0, dim1)

    def get_communication_stream(self):
        raw = _ops.comm_overlap_get_stream(self._handle)
        s = torch.cuda.ExternalStream(raw)
        return s, s  # send == recv for non-P2P

    def is_fp8_ubuf(self):
        return _ops.comm_overlap_is_fp8_ubuf(self._handle)

    def is_atomic_gemm(self):
        return _ops.comm_overlap_is_atomic_gemm(self._handle)

    def is_p2p_overlap(self):
        return _ops.comm_overlap_is_p2p(self._handle)

    def __del__(self):
        if hasattr(self, "_handle"):
            _ops.destroy_comm_overlap(self._handle)


class CommOverlapP2P:
    """Python replacement for pybind11 CommOverlapP2P (handle-based stable ABI)."""

    def __init__(
        self,
        shape,
        dtype,
        helper,
        tp_size,
        comm_type,
        num_max_streams=3,
        comm_cga_size=1,
        gemm_priority=0,
        comm_priority=0,
        num_comm_sm=1,
        set_sm_margin=False,
        use_ce=True,
        atomic_gemm=False,
        aggregate=False,
    ):
        self._ag_cb, self._bar_cb = _make_comm_callbacks(helper)
        ag_ptr = _ctypes.cast(self._ag_cb, _ctypes.c_void_p).value
        bar_ptr = _ctypes.cast(self._bar_cb, _ctypes.c_void_p).value
        _ops.register_comm_callbacks(ag_ptr, bar_ptr)

        buf_dtype = _TORCH_TO_TE_DTYPE.get(dtype, 6)
        self._handle = _ops.create_comm_overlap_p2p(
            list(shape),
            buf_dtype,
            helper.myrank,
            helper.numranks,
            helper.mylocal,
            helper.numlocal,
            helper.mynode,
            helper.numnodes,
            tp_size,
            int(comm_type),
            num_max_streams,
            comm_cga_size,
            gemm_priority,
            comm_priority,
            num_comm_sm,
            set_sm_margin,
            use_ce,
            atomic_gemm,
            aggregate,
        )

    def copy_into_buffer(self, input, local_chunk=False):  # pylint: disable=redefined-builtin
        _ops.comm_overlap_copy_into_buffer(input, self._handle, local_chunk)

    def get_buffer(self, local_chunk=False, shape=None):
        if shape is not None and len(shape) >= 2:
            dim0, dim1 = shape[0], shape[1]
        else:
            dim0, dim1 = -1, -1
        return _ops.comm_overlap_get_buffer(self._handle, local_chunk, dim0, dim1)

    def get_communication_stream(self):
        send_raw, recv_raw = _ops.comm_overlap_p2p_get_streams(self._handle)
        return torch.cuda.ExternalStream(send_raw), torch.cuda.ExternalStream(recv_raw)

    def is_fp8_ubuf(self):
        return _ops.comm_overlap_is_fp8_ubuf(self._handle)

    def is_atomic_gemm(self):
        return _ops.comm_overlap_is_atomic_gemm(self._handle)

    def is_p2p_overlap(self):
        return _ops.comm_overlap_is_p2p(self._handle)

    def __del__(self):
        if hasattr(self, "_handle"):
            _ops.destroy_comm_overlap_p2p(self._handle)


# ============================================================================
# Fused attention (match pybind11 signatures)
# ============================================================================


def fused_attn_fwd(  # pylint: disable=unused-argument
    max_seqlen_q,
    max_seqlen_kv,
    is_training,
    attn_scale,
    p_dropout,
    set_zero,
    qkv_layout,
    bias_type,
    attn_mask_type,
    softmax_type,
    window_size,
    bottom_right_diagonal,
    cu_seqlens_q,
    cu_seqlens_kv,
    Q,
    K,
    V,
    fake_dtype,
    cu_seqlens_q_padded=None,
    cu_seqlens_kv_padded=None,
    page_table_k=None,
    page_table_v=None,
    s_quantizer=None,
    o_quantizer=None,
    Bias=None,
    SoftmaxOffset=None,
    rng_gen=None,
    rng_elts_per_thread=0,
    return_max_logit=False,
    cuda_graph=False,
):
    """Fused attention forward via stable ABI fused_attn_fwd_noalloc."""
    from transformer_engine.pytorch.tensor._extract import extract_tensor_data

    # Extract Q/K/V raw buffers
    Q_data, Q_dtype_int, Q_si, Q_sm = extract_tensor_data(Q)
    K_data, K_dtype_int, K_si, K_sm = extract_tensor_data(K)
    V_data, V_dtype_int, V_si, V_sm = extract_tensor_data(V)

    device = Q_data.device
    _TORCH_DT = {torch.float32: 4, torch.float16: 5, torch.bfloat16: 6, torch.uint8: 0}

    # Determine O dtype from fake_dtype
    if isinstance(fake_dtype, torch.dtype):
        O_torch_dtype = fake_dtype
        O_dtype_int = _TORCH_DT.get(fake_dtype, Q_dtype_int)
    else:
        O_dtype_int = Q_dtype_int
        O_torch_dtype = Q_data.dtype

    # Allocate O: Q shape with V's last dim
    O_shape = list(Q_data.shape)
    O_shape[-1] = V_data.shape[-1]
    if o_quantizer is not None:
        O_tensor = o_quantizer.make_empty(O_shape, dtype=O_torch_dtype, device=device)
        O_data, O_dtype_int, O_si, O_sm = extract_tensor_data(O_tensor)
        O_amax = getattr(o_quantizer, "amax", None)
        O_scale = getattr(o_quantizer, "scale", None)
        # Initialize scale_inv = 1/scale (pybind does this in Float8Quantizer::create_tensor).
        # make_empty leaves _scale_inv uninitialized; the NVTE kernel does NOT write it.
        if O_scale is not None and O_si is not None:
            O_si.copy_(O_scale.float().reciprocal())
    else:
        O_tensor = torch.empty(O_shape, dtype=O_torch_dtype, device=device)
        O_data, _, O_si, O_sm = O_tensor, O_dtype_int, None, 0
        O_amax, O_scale = None, None

    # Allocate S (softmax placeholder — shape determined by kernel on first pass)
    if s_quantizer is not None:
        S_tensor = s_quantizer.make_empty([0], dtype=torch.float32, device=device)
        S_data, S_dtype_int, S_si, S_sm = extract_tensor_data(S_tensor)
        S_amax = getattr(s_quantizer, "amax", None)
        S_scale = getattr(s_quantizer, "scale", None)
        # Initialize scale_inv = 1/scale (same as O above)
        if S_scale is not None and S_si is not None:
            S_si.copy_(S_scale.float().reciprocal())
    else:
        S_tensor = torch.empty([0], dtype=torch.float32, device=device)
        S_data, S_dtype_int, S_si, S_sm = S_tensor, 4, None, 0
        S_amax, S_scale = None, None

    # rng_state [seed, offset] — zeros for p_dropout=0; for training with dropout
    # the kernel writes the actual used state into the aux tensor pack.
    rng_state = torch.zeros([2], dtype=torch.int64, device=device)

    result = _ops.fused_attn_fwd_noalloc(
        int(max_seqlen_q),
        int(max_seqlen_kv),
        bool(is_training),
        float(attn_scale),
        float(p_dropout),
        bool(set_zero),
        int(qkv_layout),
        int(bias_type),
        int(attn_mask_type),
        int(softmax_type),
        list(window_size),
        bool(bottom_right_diagonal),
        cu_seqlens_q,
        cu_seqlens_kv,
        Q_data,
        Q_dtype_int,
        Q_si,
        Q_sm,
        K_data,
        K_dtype_int,
        K_si,
        K_sm,
        V_data,
        V_dtype_int,
        V_si,
        V_sm,
        S_data,
        S_dtype_int,
        S_amax,
        S_scale,
        S_si,
        S_sm,
        O_data,
        O_dtype_int,
        O_amax,
        O_scale,
        O_si,
        O_sm,
        cu_seqlens_q_padded,
        cu_seqlens_kv_padded,
        page_table_k,
        page_table_v,
        Bias,
        SoftmaxOffset,
        rng_state,
        bool(return_max_logit),
        bool(cuda_graph),
    )
    # result = (aux0, ..., aux9, num_aux)
    *aux_tensors, num_aux = result
    # Return format matches pybind: [O, stats, (max if applicable), rng_state, ...]
    return [O_tensor] + list(aux_tensors[:num_aux])


_te_lib_handle = None


def _get_te_lib():
    """Get cached ctypes handle to libtransformer_engine.so."""
    global _te_lib_handle
    if _te_lib_handle is not None:
        return _te_lib_handle
    te_spec = importlib.util.find_spec("transformer_engine")
    if te_spec is not None and te_spec.origin is not None:
        te_dir = Path(te_spec.origin).parent.parent
        candidates = glob.glob(str(te_dir / "libtransformer_engine*.so"))
        if candidates:
            _te_lib_handle = _ctypes.CDLL(candidates[0])
            return _te_lib_handle
    raise RuntimeError("Could not find libtransformer_engine.so")


def _get_qkv_layout_group(qkv_layout_int):
    """Call nvte_get_qkv_layout_group via ctypes. Returns int layout group."""
    try:
        lib = _get_te_lib()
        fn = lib.nvte_get_qkv_layout_group
        fn.restype = _ctypes.c_int
        fn.argtypes = [_ctypes.c_int]
        return fn(qkv_layout_int)
    except RuntimeError:
        return 4  # fallback: NVTE_HD_HD_HD


def fused_attn_bwd(
    max_seqlen_q,
    max_seqlen_kv,
    attn_scale,
    p_dropout,
    set_zero,
    qkv_layout,
    bias_type,
    attn_mask_type,
    softmax_type,
    window_size,
    bottom_right_diagonal,
    deterministic,
    cu_seqlens_q,
    cu_seqlens_kv,
    Q,
    K,
    V,
    O,
    dO,
    fake_dtype,
    dqkv_dtype,
    aux_ctx_tensors,
    cu_seqlens_q_padded=None,
    cu_seqlens_kv_padded=None,
    s_quantizer=None,
    dp_quantizer=None,
    dqkv_quantizer=None,
    cuda_graph=False,
):
    """Fused attention backward via stable ABI fused_attn_bwd_packed."""
    from transformer_engine.pytorch.tensor._extract import extract_tensor_data

    _TORCH_DT = {torch.float32: 4, torch.float16: 5, torch.bfloat16: 6, torch.uint8: 0}
    _TE_TO_TORCH_DT = {
        4: torch.float32,
        5: torch.float16,
        6: torch.bfloat16,
        0: torch.uint8,
        7: torch.uint8,  # kFloat8E4M3 stored as uint8
        8: torch.uint8,  # kFloat8E5M2 stored as uint8
    }

    # Extract input tensor data
    Q_data, Q_dtype, Q_si, Q_sm = extract_tensor_data(Q)
    K_data, K_dtype, K_si, K_sm = extract_tensor_data(K)
    V_data, V_dtype, V_si, V_sm = extract_tensor_data(V)
    O_data, O_dtype, O_si, O_sm = extract_tensor_data(O)
    dO_data, dO_dtype, dO_si, dO_sm = extract_tensor_data(dO)

    device = Q_data.device

    # S and dP: empty placeholder tensors for backward.
    # For FP8, the pybind version creates these via quantizer_helper(s_quantizer, {0}, ...)
    # which sets up FP8 dtype and scale_inv. Replicate by detecting quantizer type.
    S_data = torch.empty([0], dtype=torch.float32, device=device)
    S_dtype = 4  # kFloat32
    S_amax, S_scale, S_si, S_sm = None, None, None, 0
    dP_data = torch.empty([0], dtype=torch.float32, device=device)
    dP_dtype = 4
    dP_amax, dP_scale, dP_si, dP_sm = None, None, None, 0

    def _fp8_quantizer_metadata(quantizer):
        """Extract FP8 metadata from a quantizer for S/dP tensors."""
        q_dtype = getattr(quantizer, "dtype", None)
        if q_dtype is None:
            return None
        # dtype may be torch.dtype or integer TE enum
        _FP8_DTYPES = {torch.float8_e4m3fn, torch.float8_e5m2, 7, 8}
        if q_dtype not in _FP8_DTYPES:
            return None
        te_dt = 7 if q_dtype in (torch.float8_e4m3fn, 7) else 8
        q_scale = getattr(quantizer, "scale", None)
        # Use the quantizer's amax tensor so the kernel's amax output flows back to
        # FP8GlobalStateManager's amax_history buffer (matching the pybind path where
        # Float8Quantizer::set_quantization_params wires the TensorWrapper's amax).
        q_amax = getattr(quantizer, "amax", None)
        amax = (
            q_amax if q_amax is not None else torch.zeros([1], dtype=torch.float32, device=device)
        )
        if q_scale is not None:
            scale = q_scale.clone().detach().to(dtype=torch.float32, device=device).reshape(1)
            si = (1.0 / q_scale).to(dtype=torch.float32, device=device).reshape(1)
        else:
            scale = torch.ones([1], dtype=torch.float32, device=device)
            si = torch.ones([1], dtype=torch.float32, device=device)
        return te_dt, amax, scale, si

    if s_quantizer is not None:
        meta = _fp8_quantizer_metadata(s_quantizer)
        if meta is not None:
            S_data = torch.empty([0], dtype=torch.uint8, device=device)
            S_dtype, S_amax, S_scale, S_si = meta
            S_sm = 0  # NVTE_DELAYED_TENSOR_SCALING
    if dp_quantizer is not None:
        meta = _fp8_quantizer_metadata(dp_quantizer)
        if meta is not None:
            dP_data = torch.empty([0], dtype=torch.uint8, device=device)
            dP_dtype, dP_amax, dP_scale, dP_si = meta
            dP_sm = 0  # NVTE_DELAYED_TENSOR_SCALING

    # Determine output grad dtype.
    # For Float8CurrentScalingQuantizer, the pybind version allocates dQ/dK/dV as
    # fake_dtype (BF16), NOT uint8. Replicate that here.
    _is_current_scaling = (
        dqkv_quantizer is not None and "CurrentScaling" in type(dqkv_quantizer).__name__
    )
    if dqkv_dtype is not None:
        dqkv_te_dtype = int(dqkv_dtype)
        if _is_current_scaling:
            # Current scaling: output is high-precision with amax tracking
            dqkv_torch_dtype = fake_dtype if isinstance(fake_dtype, torch.dtype) else Q_data.dtype
        else:
            dqkv_torch_dtype = _TE_TO_TORCH_DT.get(dqkv_te_dtype, Q_data.dtype)
    elif isinstance(fake_dtype, torch.dtype):
        dqkv_torch_dtype = fake_dtype
        dqkv_te_dtype = _TORCH_DT.get(fake_dtype, Q_dtype)
    else:
        dqkv_torch_dtype = Q_data.dtype
        dqkv_te_dtype = Q_dtype

    Q_shape = list(Q_data.shape)
    K_shape = list(K_data.shape)
    V_shape = list(V_data.shape)

    # Allocate dQ/dK/dV based on layout group.
    # IMPORTANT: for packed layouts, dQ/dK/dV must be NON-CONTIGUOUS VIEWS of the packed
    # tensor (dQKV / dKV), NOT separate .contiguous() copies.  cuDNN computes the output
    # gradient stride from the qkv_layout flag (e.g. NVTE_3HD → stride [3*B*H*D, 3*H*D, D, 1])
    # and writes using that stride starting from dQ.data_ptr().  If dQ.data_ptr() points to
    # a small contiguous tensor (only S*B*H*D elements) the write overflows → illegal address.
    # The pybind11 backend does the same (extensions/attention.cpp:367–378).
    layout_group = _get_qkv_layout_group(int(qkv_layout))
    if layout_group == 0:  # NVTE_3HD: packed dQKV with 3 in third-to-last dim
        dQKV_shape = Q_shape[:-2] + [3] + Q_shape[-2:]
        dQKV = torch.empty(dQKV_shape, dtype=dqkv_torch_dtype, device=device)
        dQ = dQKV[..., 0, :, :]  # non-contiguous view; data_ptr = dQKV.data_ptr()
        dK = dQKV[..., 1, :, :]  # non-contiguous view; data_ptr = dQKV.data_ptr() + H*D*sizeof
        dV = dQKV[..., 2, :, :]  # non-contiguous view; data_ptr = dQKV.data_ptr() + 2*H*D*sizeof
    elif layout_group == 1:  # NVTE_H3D: packed dQKV with 3 in second-to-last
        dQKV_shape = Q_shape[:-1] + [3, Q_shape[-1]]
        dQKV = torch.empty(dQKV_shape, dtype=dqkv_torch_dtype, device=device)
        dQ = dQKV[..., 0, :]  # non-contiguous view
        dK = dQKV[..., 1, :]
        dV = dQKV[..., 2, :]
    elif layout_group == 2:  # NVTE_HD_2HD
        dQ = torch.empty(Q_shape, dtype=dqkv_torch_dtype, device=device)
        dKV_shape = K_shape[:-2] + [2] + K_shape[-2:]
        dKV = torch.empty(dKV_shape, dtype=dqkv_torch_dtype, device=device)
        dK = dKV[..., 0, :, :]  # non-contiguous view
        dV = dKV[..., 1, :, :]
    elif layout_group == 3:  # NVTE_HD_H2D
        dQ = torch.empty(Q_shape, dtype=dqkv_torch_dtype, device=device)
        dKV_shape = K_shape[:-1] + [2, K_shape[-1]]
        dKV = torch.empty(dKV_shape, dtype=dqkv_torch_dtype, device=device)
        dK = dKV[..., 0, :]  # non-contiguous view
        dV = dKV[..., 1, :]
    else:  # NVTE_HD_HD_HD (4) and Paged_KV (5)
        dQ = torch.empty(Q_shape, dtype=dqkv_torch_dtype, device=device)
        dK = torch.empty(K_shape, dtype=dqkv_torch_dtype, device=device)
        dV = torch.empty(V_shape, dtype=dqkv_torch_dtype, device=device)

    dQ_data, dQ_te_dtype, dQ_si, dQ_sm = extract_tensor_data(dQ)
    dK_data, dK_te_dtype, dK_si, dK_sm = extract_tensor_data(dK)
    dV_data, dV_te_dtype, dV_si, dV_sm = extract_tensor_data(dV)
    dQ_amax = dQ_scale = dK_amax = dK_scale = dV_amax = dV_scale = None
    if dqkv_te_dtype in (7, 8) and not _is_current_scaling:
        # Delayed scaling: dQ/dK/dV are FP8 (uint8), need scale/scale_inv/amax.
        # The pybind version uses Float8Quantizer::create_tensor which computes
        # scale_inv = 1/scale from the quantizer's scale. Replicate that here.
        dQ_te_dtype = dqkv_te_dtype
        dK_te_dtype = dqkv_te_dtype
        dV_te_dtype = dqkv_te_dtype
        dqkv_q_scale = getattr(dqkv_quantizer, "scale", None) if dqkv_quantizer else None
        if dqkv_q_scale is not None:
            dqkv_scale_f32 = dqkv_q_scale.detach().to(dtype=torch.float32, device=device).reshape(1)
            dqkv_si_f32 = (1.0 / dqkv_q_scale).to(dtype=torch.float32, device=device).reshape(1)
        else:
            dqkv_scale_f32 = torch.ones([1], dtype=torch.float32, device=device)
            dqkv_si_f32 = torch.ones([1], dtype=torch.float32, device=device)
        # Use the quantizer's amax tensor so the kernel's amax output flows back
        # to FP8GlobalStateManager's amax_history buffer. The pybind path achieves
        # this via Float8Quantizer::create_tensor → set_quantization_params, which
        # wires the TensorWrapper's amax to the quantizer's amax. Without this, the
        # backward amax never updates, causing stale scales in batch test runs.
        dqkv_q_amax = getattr(dqkv_quantizer, "amax", None)
        dQ_amax = (
            dqkv_q_amax
            if dqkv_q_amax is not None
            else torch.zeros([1], dtype=torch.float32, device=device)
        )
        dQ_scale = dqkv_scale_f32
        dQ_si = dqkv_si_f32.clone()
        dK_amax = (
            dqkv_q_amax
            if dqkv_q_amax is not None
            else torch.zeros([1], dtype=torch.float32, device=device)
        )
        dK_scale = dqkv_scale_f32.clone()
        dK_si = dqkv_si_f32.clone()
        dV_amax = (
            dqkv_q_amax
            if dqkv_q_amax is not None
            else torch.zeros([1], dtype=torch.float32, device=device)
        )
        dV_scale = dqkv_scale_f32.clone()
        dV_si = dqkv_si_f32.clone()
    elif _is_current_scaling:
        # Current scaling: dQ/dK/dV are high-precision (BF16/FP16) with amax tracking.
        # The pybind version uses create_unquantized_tensor_with_amax → NoneQuantizer tensor.
        # dQ/dK/dV are already allocated as fake_dtype, extract_tensor_data returns BF16 dtype.
        # Just need amax for cuDNN.
        dQ_amax = torch.zeros([1], dtype=torch.float32, device=device)
        dK_amax = torch.zeros([1], dtype=torch.float32, device=device)
        dV_amax = torch.zeros([1], dtype=torch.float32, device=device)

    # dBias: allocate when bias_type not in {NO_BIAS=0, ALIBI=3}
    # The pybind version derives dBias shape from the last aux_ctx_tensor (the saved bias
    # from forward) when aux_ctx_tensors has >= 2 entries.  This is critical because the
    # bias may have batch dimension > 1, e.g. [b, h, s, s] instead of [1, h, s, s].
    # Hardcoding [1, h, s, s] causes cuDNN dimension-mismatch errors on backward.
    num_heads_q = Q_shape[-2] if len(Q_shape) >= 2 else 1
    num_aux = len(aux_ctx_tensors) if aux_ctx_tensors else 0
    dBias = None
    if int(bias_type) not in (0, 3):  # 0=NVTE_NO_BIAS, 3=NVTE_ALIBI
        if num_aux >= 2:
            bias_shape = list(aux_ctx_tensors[-1].shape)
            dBias = torch.empty(bias_shape, dtype=dqkv_torch_dtype, device=device)
        else:
            dBias = torch.empty(
                [1, num_heads_q, max_seqlen_q, max_seqlen_kv],
                dtype=dqkv_torch_dtype,
                device=device,
            )
        # For THD format, cuDNN accumulates into dBias, so it must be zero-initialized.
        # Use zeros for safety (the pybind version only zero-fills for THD, but zeros
        # is always correct and the perf difference is negligible).
        dBias.zero_()

    # dSoftmaxOffset: allocate when softmax_type != VANILLA (0)
    dSoftmaxOffset = None
    if int(softmax_type) != 0:
        dSoftmaxOffset = torch.zeros([1, num_heads_q, 1, 1], dtype=torch.float32, device=device)

    # Pack dtype/scaling_mode info into a CPU int64 tensor
    dtype_info = torch.tensor(
        [
            Q_dtype,
            Q_sm,
            K_dtype,
            K_sm,
            V_dtype,
            V_sm,
            O_dtype,
            O_sm,
            dO_dtype,
            dO_sm,
            S_dtype,
            S_sm,
            dP_dtype,
            dP_sm,
            dQ_te_dtype,
            dQ_sm,
            dK_te_dtype,
            dK_sm,
            dV_te_dtype,
            dV_sm,
        ],
        dtype=torch.int64,
        device="cpu",
    )

    # Flatten aux_ctx_tensors, padding to 10 slots
    num_aux = len(aux_ctx_tensors) if aux_ctx_tensors else 0
    aux_list = (list(aux_ctx_tensors) if aux_ctx_tensors else []) + [None] * (10 - num_aux)

    _ops.fused_attn_bwd_packed(
        int(max_seqlen_q),
        int(max_seqlen_kv),
        float(attn_scale),
        float(p_dropout),
        bool(set_zero),
        int(qkv_layout),
        int(bias_type),
        int(attn_mask_type),
        int(softmax_type),
        list(window_size),
        bool(bottom_right_diagonal),
        bool(deterministic),
        bool(cuda_graph),
        cu_seqlens_q,
        cu_seqlens_kv,
        cu_seqlens_q_padded,
        cu_seqlens_kv_padded,
        Q_data,
        Q_si,
        K_data,
        K_si,
        V_data,
        V_si,
        O_data,
        O_si,
        dO_data,
        dO_si,
        S_data,
        S_amax,
        S_scale,
        S_si,
        dP_data,
        dP_amax,
        dP_scale,
        dP_si,
        dQ_data,
        dQ_amax,
        dQ_scale,
        dQ_si,
        dK_data,
        dK_amax,
        dK_scale,
        dK_si,
        dV_data,
        dV_amax,
        dV_scale,
        dV_si,
        dBias,
        dSoftmaxOffset,
        dtype_info,
        num_aux,
        *aux_list,
    )

    # For delayed-scaling FP8 output, wrap raw uint8 tensors in Float8Tensor with
    # scale_inv so downstream operations correctly dequantize. For current scaling,
    # the output is already high-precision (BF16) and doesn't need wrapping.
    if dqkv_te_dtype in (7, 8) and dqkv_quantizer is not None and not _is_current_scaling:
        from transformer_engine.pytorch.tensor.float8_tensor import Float8Tensor

        fp8_dtype_val = getattr(dqkv_quantizer, "dtype", dqkv_te_dtype)
        fake_dt = fake_dtype if isinstance(fake_dtype, torch.dtype) else torch.bfloat16

        def _wrap_fp8(raw_tensor, si_tensor):
            return Float8Tensor(
                raw_tensor.shape,
                fake_dt,
                data=raw_tensor,
                fp8_scale_inv=si_tensor,
                fp8_dtype=fp8_dtype_val,
                quantizer=dqkv_quantizer,
            )

        dQ = _wrap_fp8(dQ, dQ_si)
        dK = _wrap_fp8(dK, dK_si)
        dV = _wrap_fp8(dV, dV_si)

    return [dQ, dK, dV, dBias, dSoftmaxOffset]


def bulk_overlap_ag_with_external_gemm(allgather_communicator, send_stream, recv_stream):
    _ops.bulk_overlap_ag_with_external_gemm(
        allgather_communicator._handle, send_stream.cuda_stream, recv_stream.cuda_stream
    )


# ============================================================================
# NVTE enums exposed via pybind
# ============================================================================


class NVTE_QKV_Layout(IntEnum):
    NVTE_SB3HD = 0
    NVTE_SBH3D = 1
    NVTE_SBHD_SB2HD = 2
    NVTE_SBHD_SBH2D = 3
    NVTE_SBHD_SBHD_SBHD = 4
    NVTE_BS3HD = 5
    NVTE_BSH3D = 6
    NVTE_BSHD_BS2HD = 7
    NVTE_BSHD_BSH2D = 8
    NVTE_BSHD_BSHD_BSHD = 9
    NVTE_T3HD = 10
    NVTE_TH3D = 11
    NVTE_THD_T2HD = 12
    NVTE_THD_TH2D = 13
    NVTE_THD_THD_THD = 14
    NVTE_SBHD_BSHD_BSHD = 15
    NVTE_BSHD_SBHD_SBHD = 16
    NVTE_THD_BSHD_BSHD = 17
    NVTE_THD_SBHD_SBHD = 18
    NVTE_Paged_KV_BSHD_BSHD_BSHD = 19
    NVTE_Paged_KV_BSHD_SBHD_SBHD = 20
    NVTE_Paged_KV_SBHD_BSHD_BSHD = 21
    NVTE_Paged_KV_SBHD_SBHD_SBHD = 22
    NVTE_Paged_KV_THD_BSHD_BSHD = 23
    NVTE_Paged_KV_THD_SBHD_SBHD = 24


class NVTE_QKV_Format(IntEnum):
    NVTE_SBHD = 0
    NVTE_BSHD = 1
    NVTE_THD = 2
    NVTE_BSHD_2SBHD = 3
    NVTE_SBHD_2BSHD = 4
    NVTE_THD_2BSHD = 5
    NVTE_THD_2SBHD = 6


class NVTE_Bias_Type(IntEnum):
    NVTE_NO_BIAS = 0
    NVTE_PRE_SCALE_BIAS = 1
    NVTE_POST_SCALE_BIAS = 2
    NVTE_ALIBI = 3


class NVTE_Mask_Type(IntEnum):
    NVTE_NO_MASK = 0
    NVTE_PADDING_MASK = 1
    NVTE_CAUSAL_MASK = 2
    NVTE_PADDING_CAUSAL_MASK = 3
    NVTE_CAUSAL_BOTTOM_RIGHT_MASK = 4
    NVTE_PADDING_CAUSAL_BOTTOM_RIGHT_MASK = 5
    NVTE_ARBITRARY_MASK = 6


class NVTE_Softmax_Type(IntEnum):
    NVTE_VANILLA_SOFTMAX = 0
    NVTE_OFF_BY_ONE_SOFTMAX = 1
    NVTE_LEARNABLE_SOFTMAX = 2


class NVTE_Fused_Attn_Backend(IntEnum):
    NVTE_No_Backend = -1
    NVTE_F16_max512_seqlen = 0
    NVTE_F16_arbitrary_seqlen = 1
    NVTE_FP8 = 2


def device_supports_multicast():
    """Check if current device supports multicast."""
    return bool(_ops.device_supports_multicast(-1))


def get_stream_priority_range():
    """Get CUDA stream priority range."""
    result = _ops.get_stream_priority_range(-1)
    return int(result[0]), int(result[1])


def ubuf_built_with_mpi():
    """Check if TE was built with NVTE_UB_WITH_MPI=1."""
    return bool(_ops.ubuf_built_with_mpi())


# Register stable GEMM op as a passthrough in QuantizedTensor.__torch_dispatch__
# so that FP8/quantized tensors are NOT dequantized before entering the GEMM op.
# This mirrors how te_moe ops are registered in permutation.py.
def _register_passthrough_ops():
    import sys

    # Only run if quantized_tensor is already in sys.modules (avoid circular import).
    # If it's not imported yet, quantized_tensor.py will call this on its own after
    # importing _stable_torch_module.
    if "transformer_engine.pytorch.quantized_tensor" not in sys.modules:
        return
    try:
        from transformer_engine.pytorch.quantized_tensor import (
            _quantized_tensor_passthrough_ops,
        )

        _quantized_tensor_passthrough_ops.add(torch.ops.transformer_engine_stable.gemm.default)
    except (ImportError, AttributeError):
        pass


_register_passthrough_ops()
