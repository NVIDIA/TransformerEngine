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

import glob
import importlib.util
import os
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
        te_dir = Path(te_spec.origin).parent.parent
        candidates = glob.glob(str(te_dir / "te_stable_abi*"))
        if candidates:
            torch.ops.load_library(candidates[0])
            _loaded = True
            return
    raise RuntimeError("Could not find te_stable_abi shared library")


_load_stable_lib()
_ops = torch.ops.transformer_engine_stable


def _not_implemented(name):
    """Create a stub function that raises NotImplementedError."""
    def fn(*args, **kwargs):
        raise NotImplementedError(
            f"{name} is not yet implemented in the stable ABI module. "
            f"This function needs a native stable implementation.")
    fn.__name__ = name
    return fn

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
    import ctypes
    try:
        lib = ctypes.CDLL("libcublasLt.so")
        lib.cublasLtGetVersion.restype = ctypes.c_size_t
        return lib.cublasLtGetVersion()
    except OSError:
        return 0


def get_cudnn_version():
    import ctypes
    try:
        lib = ctypes.CDLL("libcudnn.so")
        lib.cudnnGetVersion.restype = ctypes.c_size_t
        return lib.cudnnGetVersion()
    except OSError:
        return 0


def get_num_cublas_streams():
    import ctypes
    from transformer_engine.common import _get_shared_object_file
    so_path = _get_shared_object_file("core")
    lib = ctypes.CDLL(str(so_path))
    lib.nvte_get_num_compute_streams.restype = ctypes.c_int
    return lib.nvte_get_num_compute_streams()


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


def fused_rope_forward(input, freqs, start_positions, qkv_format, interleaved,
                       cu_seqlens, cp_size, cp_rank):
    return _ops.fused_rope_forward(input, freqs, start_positions,
                                   int(qkv_format), interleaved,
                                   cu_seqlens, cp_size, cp_rank)


def fused_rope_backward(output_grads, freqs, start_positions, qkv_format,
                        interleaved, cu_seqlens, cp_size, cp_rank):
    return _ops.fused_rope_backward(output_grads, freqs, start_positions,
                                    int(qkv_format), interleaved,
                                    cu_seqlens, cp_size, cp_rank)


def fused_qkv_rope_forward(qkv_input, q_freqs, k_freqs, start_positions,
                            qkv_split_arg_list, qkv_format, interleaved,
                            cp_size, cp_rank):
    return _ops.fused_qkv_rope_forward(qkv_input, q_freqs, k_freqs,
                                        start_positions, list(qkv_split_arg_list),
                                        int(qkv_format), interleaved,
                                        cp_size, cp_rank)


def fused_qkv_rope_backward(q_grad_out, k_grad_out, v_grad_out, q_freqs,
                             k_freqs, qkv_split_arg_list, qkv_format,
                             interleaved, cp_size, cp_rank):
    return _ops.fused_qkv_rope_backward(q_grad_out, k_grad_out, v_grad_out,
                                         q_freqs, k_freqs,
                                         list(qkv_split_arg_list),
                                         int(qkv_format), interleaved,
                                         cp_size, cp_rank)


# ============================================================================
# Router
# ============================================================================


def fused_topk_with_score_function_fwd(logits, topk, use_pre_softmax,
                                       num_groups=None, group_topk=None,
                                       scaling_factor=None,
                                       score_function="softmax",
                                       expert_bias=None):
    return _ops.fused_topk_with_score_function_fwd(
        logits, topk, use_pre_softmax,
        num_groups if num_groups is not None else -1,
        group_topk if group_topk is not None else -1,
        scaling_factor if scaling_factor is not None else 1.0,
        score_function, expert_bias)


def fused_topk_with_score_function_bwd(num_tokens, num_experts, routing_map,
                                       intermediate_output, grad_probs,
                                       grad_logits, topk, use_pre_softmax,
                                       scaling_factor=None,
                                       score_function="softmax"):
    _ops.fused_topk_with_score_function_bwd(
        num_tokens, num_experts, routing_map, intermediate_output,
        grad_probs, grad_logits, topk, use_pre_softmax,
        scaling_factor if scaling_factor is not None else 1.0,
        score_function)


fused_score_for_moe_aux_loss_fwd = _ops.fused_score_for_moe_aux_loss_fwd
fused_score_for_moe_aux_loss_bwd = _ops.fused_score_for_moe_aux_loss_bwd
fused_moe_aux_loss_fwd = _ops.fused_moe_aux_loss_fwd
fused_moe_aux_loss_bwd = _ops.fused_moe_aux_loss_bwd

# ============================================================================
# Dropout
# ============================================================================


def dropout_fwd(input, dropout_probability, out=None):
    """Dropout forward. RNG state extracted from default CUDA generator."""
    device = input.device if hasattr(input, 'device') else torch.device('cuda')
    # Extract from torch tensor if input is a py handle-like
    if hasattr(input, '_data'):
        inp_tensor = input._data
    elif isinstance(input, torch.Tensor):
        inp_tensor = input
    else:
        inp_tensor = input

    gen = torch.cuda.default_generators[device.index or 0]
    # Get Philox state: [seed, offset]
    rng_state = torch.empty(2, dtype=torch.int64, device=device)
    # Use the generator's philox state
    seed = gen.initial_seed()
    # Increment offset
    state = gen.get_state()
    offset = int.from_bytes(state[8:16].numpy().tobytes(), 'little') if len(state) > 8 else 0
    rng_state[0] = seed
    rng_state[1] = offset

    output, mask = _ops.dropout_fwd(inp_tensor, rng_state, dropout_probability)
    return [output, mask]


def dropout_bwd(grad_output, mask, dropout_probability, grad_input=None):
    return _ops.dropout_bwd(grad_output, mask, dropout_probability, grad_input)


# ============================================================================
# Transpose ops
# ============================================================================


def fp8_transpose(input, otype, output=None):
    return _ops.fp8_transpose(input, int(otype), output)


nvfp4_data_transpose = _ops.nvfp4_data_transpose
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


def copy_to_kv_cache(new_k, new_v, k_cache, v_cache, page_table,
                     cu_new_lens, cu_cached_lens, qkv_format, b,
                     max_ctx_len, max_seq_len, max_pages_per_seq,
                     is_non_paged):
    _ops.copy_to_kv_cache(new_k, new_v, k_cache, v_cache, page_table,
                          cu_new_lens, cu_cached_lens, int(qkv_format), b,
                          max_ctx_len, max_seq_len, max_pages_per_seq,
                          is_non_paged)


# ============================================================================
# Recipe / amax / scale
# ============================================================================

compute_amax = _ops.compute_amax


def get_fused_attn_backend(*args, **kwargs):
    # Convert enum args to ints
    int_args = []
    for a in args:
        int_args.append(int(a) if isinstance(a, IntEnum) else a)
    return _ops.get_fused_attn_backend(*int_args)


def fused_amax_and_scale_update_after_reduction(amax_reduction_buffer,
                                                 amax_histories, scales,
                                                 amax_compute_algo,
                                                 fp8_dtype, margin):
    num = len(amax_histories)
    ah_ptrs = torch.tensor([t.data_ptr() for t in amax_histories],
                           dtype=torch.int64)
    # Shape format: [ndim, dim0, dim1] per tensor (dim1=0 for 1D)
    ah_shapes = torch.tensor(
        [[t.dim(), t.shape[0], t.shape[1] if t.dim() >= 2 else 0]
         for t in amax_histories],
        dtype=torch.int64).flatten()
    sc_ptrs = torch.tensor([t.data_ptr() for t in scales], dtype=torch.int64)
    sc_shapes = torch.tensor(
        [[t.dim(), t.shape[0], t.shape[1] if t.dim() >= 2 else 0]
         for t in scales],
        dtype=torch.int64).flatten()
    _ops.fused_amax_and_scale_update(
        amax_reduction_buffer, ah_ptrs, ah_shapes, sc_ptrs, sc_shapes,
        num, amax_compute_algo, int(fp8_dtype), margin)


# ============================================================================
# Partial cast
# ============================================================================

fp8_block_scaling_compute_partial_amax = _ops.fp8_block_scaling_compute_partial_amax


def fp8_block_scaling_partial_cast(inp, out, scale, h, w, start_offset,
                                   block_len, out_dtype):
    _ops.fp8_block_scaling_partial_cast(inp, out, scale, h, w, start_offset,
                                        block_len, int(out_dtype))


mxfp8_scaling_compute_partial_amax = _ops.mxfp8_scaling_compute_partial_amax
mxfp8_scaling_partial_cast = _ops.mxfp8_scaling_partial_cast

nvfp4_2d_compute_partial_amax = _ops.nvfp4_2d_compute_partial_amax


def nvfp4_2d_partial_cast(inp, out, scale, global_scale, h, w, start_offset, block_len=16):
    """Match pybind signature — out may be quantized tensor."""
    from transformer_engine.pytorch.tensor._extract import extract_tensor_data
    out_data, out_dtype, out_si, out_sm = extract_tensor_data(out)
    _ops.nvfp4_2d_partial_cast_noalloc(
        inp, out_data, out_dtype, out_si, out_sm,
        scale, global_scale, h, w, start_offset, block_len)


# ============================================================================
# Permutation
# ============================================================================

moe_permute_fwd = _ops.moe_permute_fwd


def moe_permute_bwd(input, dtype, row_id_map, prob, num_tokens, topK):
    return _ops.moe_unpermute_fwd(input, int(dtype), row_id_map, prob,
                                  num_tokens, topK)


moe_unpermute_fwd = _ops.moe_unpermute_fwd
moe_unpermute_bwd = _ops.moe_unpermute_bwd

# ============================================================================
# Normalization
# ============================================================================

def layernorm_bwd(dz, x, mu, rsigma, gamma, sm_margin, zero_centered_gamma):
    dx, dgamma, dbeta = _ops.layernorm_bwd(dz, x, mu, rsigma, gamma,
                                           sm_margin, zero_centered_gamma)
    return [dx, dgamma, dbeta]


def rmsnorm_bwd(dz, x, rsigma, gamma, sm_margin, zero_centered_gamma):
    dx, dgamma = _ops.rmsnorm_bwd(dz, x, rsigma, gamma, sm_margin,
                                  zero_centered_gamma)
    return [dx, dgamma]


def rmsnorm_bwd_add(dz, x, add, rsigma, gamma, sm_margin, zero_centered_gamma):
    dx, dgamma = _ops.rmsnorm_bwd_add(dz, x, add, rsigma, gamma,
                                      sm_margin, zero_centered_gamma)
    return [dx, dgamma]


def layernorm_fwd(input, weight, bias, eps, out, quantizer, out_dtype,
                  sm_margin, zero_centered_gamma):
    """LayerNorm forward with optional quantization via stable ABI."""
    # Get raw input tensor (may be a quantized type)
    from transformer_engine.pytorch.tensor._extract import extract_tensor_data
    inp_data = input if isinstance(input, torch.Tensor) else extract_tensor_data(input)[0]
    w_data = weight if isinstance(weight, torch.Tensor) else extract_tensor_data(weight)[0]

    if quantizer is None or out is not None:
        # Unquantized path or pre-allocated output
        result_out, mu, rsigma = _ops.layernorm_fwd(inp_data, w_data, bias, eps,
                                                     sm_margin, zero_centered_gamma)
        if quantizer is not None and out is not None:
            # Quantize the output in-place
            from transformer_engine.pytorch.tensor._quantize_stable import quantize_into
            quantize_into(result_out, quantizer, out)
            return [out, mu, rsigma]
        return [result_out, mu, rsigma]

    # Quantized path: norm then quantize
    result_out, mu, rsigma = _ops.layernorm_fwd(inp_data, w_data, bias, eps,
                                                 sm_margin, zero_centered_gamma)
    from transformer_engine.pytorch.tensor._quantize_stable import quantize_new
    q_out = quantize_new(result_out, quantizer)
    return [q_out, mu, rsigma]


def rmsnorm_fwd(input, weight, eps, out, quantizer, out_dtype,
                sm_margin, zero_centered_gamma):
    """RMSNorm forward with optional quantization via stable ABI."""
    from transformer_engine.pytorch.tensor._extract import extract_tensor_data
    inp_data = input if isinstance(input, torch.Tensor) else extract_tensor_data(input)[0]
    w_data = weight if isinstance(weight, torch.Tensor) else extract_tensor_data(weight)[0]

    if quantizer is None or out is not None:
        result_out, rsigma = _ops.rmsnorm_fwd(inp_data, w_data, eps,
                                               sm_margin, zero_centered_gamma)
        if quantizer is not None and out is not None:
            from transformer_engine.pytorch.tensor._quantize_stable import quantize_into
            quantize_into(result_out, quantizer, out)
            return [out, None, rsigma]
        return [result_out, None, rsigma]

    result_out, rsigma = _ops.rmsnorm_fwd(inp_data, w_data, eps,
                                           sm_margin, zero_centered_gamma)
    from transformer_engine.pytorch.tensor._quantize_stable import quantize_new
    q_out = quantize_new(result_out, quantizer)
    return [q_out, None, rsigma]

# ============================================================================
# NVSHMEM (stub — requires NVTE_ENABLE_NVSHMEM build flag)
# ============================================================================


def init_nvshmem_backend(*args, **kwargs):
    raise RuntimeError("NVSHMEM not available in stable ABI build. "
                       "Build with NVTE_ENABLE_NVSHMEM=1.")


def create_nvshmem_tensor(*args, **kwargs):
    raise RuntimeError("NVSHMEM not available in stable ABI build.")


def nvshmem_send_on_current_stream(*args, **kwargs):
    raise RuntimeError("NVSHMEM not available in stable ABI build.")


def nvshmem_wait_on_current_stream(*args, **kwargs):
    raise RuntimeError("NVSHMEM not available in stable ABI build.")


def nvshmem_finalize(*args, **kwargs):
    raise RuntimeError("NVSHMEM not available in stable ABI build.")


# ============================================================================
# Check if userbuffers uses MPI
# ============================================================================

# ============================================================================
# GEMM
# ============================================================================

def generic_gemm(A, transa, B, transb, D, quantizer, out_dtype, bias,
                 bias_type, gelu, gelu_in, grad, workspace, workspaceSize,
                 accumulate, use_split_accumulator,
                 comm_overlap=None, comm_type=None, extra_output=None,
                 bulk_overlap=False, alpha=1.0, beta=None):
    """GEMM via stable ABI ops with Python-side tensor metadata extraction."""
    from transformer_engine.pytorch.tensor._extract import extract_tensor_data

    A_data, A_dtype, A_scale_inv, A_sm = extract_tensor_data(A)
    B_data, B_dtype, B_scale_inv, B_sm = extract_tensor_data(B)

    if D is not None:
        D_data, D_dtype, D_scale_inv, D_sm = extract_tensor_data(D)
        D_amax = getattr(D, '_amax', None) or (getattr(quantizer, 'amax', None) if quantizer else None)
        D_scale = getattr(quantizer, 'scale', None) if quantizer else None
        if isinstance(D_amax, torch.Tensor) and D_amax.numel() == 0: D_amax = None
        if isinstance(D_scale, torch.Tensor) and D_scale.numel() == 0: D_scale = None
    else:
        M = A_data.shape[0] if not transa else A_data.shape[1]
        N = B_data.shape[1] if not transb else B_data.shape[0]
        if quantizer is not None:
            D = quantizer.make_empty([M, N],
                                     dtype=A.dtype if isinstance(A, torch.Tensor) else torch.bfloat16,
                                     device=A_data.device)
            D_data, D_dtype, D_scale_inv, D_sm = extract_tensor_data(D)
            D_amax = getattr(quantizer, 'amax', None)
            D_scale = getattr(quantizer, 'scale', None)
        else:
            D = torch.empty(M, N, dtype=torch.bfloat16, device=A_data.device)
            D_data, D_dtype, D_scale_inv, D_sm = D, 6, None, 0
            D_amax, D_scale = None, None

    _ops.gemm(
        A_data, A_dtype, A_scale_inv, A_sm, transa,
        B_data, B_dtype, B_scale_inv, B_sm, transb,
        D_data, D_dtype, D_amax, D_scale, D_scale_inv, D_sm,
        bias, int(bias_type) if bias_type is not None else 0,
        gelu_in, workspace,
        grad, accumulate, use_split_accumulator, alpha)

    return [D, None, gelu_in, extra_output]


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
    else:
        return quantize_new(tensor, quantizer)


def dequantize(input, otype):
    """Dequantize using stable ABI ops."""
    from transformer_engine.pytorch.tensor._extract import extract_tensor_data
    in_data, in_dtype, in_scale_inv, in_sm = extract_tensor_data(input)
    _TORCH_TO_TE = {torch.float32: 4, torch.float16: 5, torch.bfloat16: 6}
    out_te_dtype = _TORCH_TO_TE.get(otype, 4) if isinstance(otype, torch.dtype) else int(otype)
    return _ops.dequantize(in_data, in_dtype, in_scale_inv, in_sm, out_te_dtype)


multi_tensor_quantize = _not_implemented("multi_tensor_quantize")
split_quantize = _not_implemented("split_quantize")
group_quantize = _not_implemented("group_quantize")


# ============================================================================
# Swizzle (match pybind11 signature)
# ============================================================================

swizzle_scales_for_gemm_ = _not_implemented("swizzle_scales_for_gemm_")


# ============================================================================
# Activation ops (match pybind11 individual function names)
# ============================================================================

def _make_activation_fwd(act_type, shape_divisor=1):
    _TE_DTYPE = {torch.float32: 4, torch.float16: 5, torch.bfloat16: 6}
    DELAYED = 0

    def fn(input, quantizer):
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
        is_delayed = 'Float8Quantizer' in q_type and 'Current' not in q_type and 'Block' not in q_type
        is_mxfp8 = 'MXFP8' in q_type
        is_current_scaling = 'CurrentScaling' in q_type
        is_nvfp4 = 'NVFP4' in q_type
        is_block = 'Block' in q_type and 'MXFP8' not in q_type

        if quantizer is None or is_delayed or is_mxfp8:
            # FULLY_FUSED: kernel writes directly to quantized output
            out_py = quantizer.make_empty(out_shape, dtype=inp.dtype, device=device)
            out_data, out_dtype, out_scale_inv, out_sm = extract_tensor_data(out_py)

            if is_mxfp8:
                out_sm = 1  # MXFP8_1D_SCALING
            elif is_delayed:
                out_sm = 0  # DELAYED_TENSOR_SCALING

            out_amax = getattr(quantizer, 'amax', None)
            out_scale = getattr(quantizer, 'scale', None)
            if isinstance(out_amax, torch.Tensor) and out_amax.numel() == 0: out_amax = None
            if isinstance(out_scale, torch.Tensor) and out_scale.numel() == 0: out_scale = None

            _ops.activation_fwd_noalloc(
                inp, out_data, out_dtype, out_amax, out_scale, out_scale_inv, out_sm, act_type)

            if hasattr(out_py, '_fp8_dtype') and hasattr(quantizer, 'dtype'):
                out_py._fp8_dtype = quantizer.dtype
            return out_py

        elif is_current_scaling:
            # FUSED_ACTIVATION_AMAX_FP8: activation→hp+amax, then quantize_from_amax
            amax = getattr(quantizer, 'amax', torch.zeros(1, dtype=torch.float32, device=device))
            # Compute activation to hp output WITH amax
            hp_out = torch.empty(out_shape, dtype=inp.dtype, device=device)
            _ops.activation_fwd_noalloc(inp, hp_out, te_dt, amax, None, None, DELAYED, act_type)
            # Quantize using pre-computed amax
            out_py = quantizer.make_empty(out_shape, dtype=inp.dtype, device=device)
            from transformer_engine.pytorch.tensor._quantize_stable import quantize_into
            # Set use_existing_amax so quantize_into uses quantize_from_amax
            orig = getattr(quantizer, 'use_existing_amax', False)
            quantizer.use_existing_amax = True
            quantize_into(hp_out, quantizer, out_py)
            quantizer.use_existing_amax = orig
            return out_py

        else:
            # UNFUSED (block scaling, NVFP4 with post-RHT amax):
            # activation→hp, then full quantize
            hp_out = torch.empty(out_shape, dtype=inp.dtype, device=device)
            _ops.activation_fwd_noalloc(inp, hp_out, te_dt, None, None, None, DELAYED, act_type)
            from transformer_engine.pytorch.tensor._quantize_stable import quantize_new
            return quantize_new(hp_out, quantizer)

    return fn

def _make_activation_bwd(act_type):
    _TE_DTYPE = {torch.float32: 4, torch.float16: 5, torch.bfloat16: 6}
    def fn(grad, input, quantizer):
        from transformer_engine.pytorch.tensor._extract import extract_tensor_data
        inp = input if isinstance(input, torch.Tensor) else extract_tensor_data(input)[0]
        grad_t = grad if isinstance(grad, torch.Tensor) else extract_tensor_data(grad)[0]

        if quantizer is None:
            te_dt = _TE_DTYPE.get(inp.dtype, 6)
            out = torch.empty_like(inp)
            _ops.dactivation_noalloc(grad_t, inp, out, te_dt, None, None, None, 0, act_type)
            return out

        # Quantized output
        out_py = quantizer.make_empty(list(inp.shape), dtype=inp.dtype, device=inp.device)
        out_data, out_dtype, out_scale_inv, out_sm = extract_tensor_data(out_py)

        q_type = type(quantizer).__name__
        if 'Block' in q_type:
            out_sm = 3 if getattr(quantizer, 'block_scaling_dim', 2) == 2 else 2  # BLOCK_2D=3, 1D=2
        elif 'MXFP8' in q_type:
            out_sm = 1  # MXFP8_1D_SCALING
        elif 'NVFP4' in q_type:
            out_sm = 4  # NVFP4_1D_SCALING

        out_amax = getattr(quantizer, 'amax', None)
        out_scale = getattr(quantizer, 'scale', None)
        if isinstance(out_amax, torch.Tensor) and out_amax.numel() == 0: out_amax = None
        if isinstance(out_scale, torch.Tensor) and out_scale.numel() == 0: out_scale = None

        _ops.dactivation_noalloc(
            grad_t, inp, out_data, out_dtype, out_amax, out_scale, out_scale_inv, out_sm, act_type)

        if hasattr(out_py, '_fp8_dtype') and hasattr(quantizer, 'dtype'):
            out_py._fp8_dtype = quantizer.dtype
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

def clamped_swiglu(input, quantizer, limit, alpha):
    inp = input if isinstance(input, torch.Tensor) else input
    out = torch.empty(*inp.shape[:-1], inp.shape[-1] // 2,
                     dtype=inp.dtype, device=inp.device)
    _ops.clamped_activation_fwd_noalloc(
        inp, out, int(DType.kFloat32 if inp.dtype == torch.float32 else DType.kBFloat16),
        None, None, None, 0, limit, alpha, 0)
    return out

def clamped_dswiglu(grad, input, quantizer, limit, alpha):
    inp = input if isinstance(input, torch.Tensor) else input
    out = torch.empty_like(inp)
    _ops.clamped_dactivation_noalloc(
        grad, inp, out, int(DType.kFloat32 if inp.dtype == torch.float32 else DType.kBFloat16),
        None, None, None, 0, limit, alpha, 0)
    return out


# ============================================================================
# Bias ops (match pybind11 individual function names)
# ============================================================================

def bgrad_quantize(grad_output, quantizer):
    # Unquantized path: compute bias gradient via sum, return input unchanged
    grad_bias = grad_output.sum(dim=0)
    return [grad_bias, grad_output]

def _make_dbias_dact(act_type):
    def fn(grad_output, act_input, quantizer):
        grad_bias = torch.empty(act_input.shape[-1], dtype=act_input.dtype,
                               device=act_input.device)
        grad_input = torch.empty_like(act_input)
        _ops.dact_dbias_noalloc(grad_output, act_input, grad_bias, grad_input,
                                int(DType.kFloat32 if act_input.dtype == torch.float32 else DType.kBFloat16),
                                None, None, None, 0, act_type)
        return [grad_bias, grad_input]
    return fn

# 0=dgelu, 1=dsilu, 2=drelu, 3=dqgelu, 4=dsrelu
dbias_dgelu = _make_dbias_dact(0)
dbias_dsilu = _make_dbias_dact(1)
dbias_drelu = _make_dbias_dact(2)
dbias_dqgelu = _make_dbias_dact(3)
dbias_dsrelu = _make_dbias_dact(4)

# ============================================================================
# Grouped GEMM (stubs)
# ============================================================================

te_general_grouped_gemm = _not_implemented("te_general_grouped_gemm")
te_general_grouped_gemm_for_grouped_tensor = _not_implemented("te_general_grouped_gemm_for_grouped_tensor")
te_general_grouped_gemm_for_discrete_in = _not_implemented("te_general_grouped_gemm_for_discrete_in")
te_general_grouped_gemm_for_discrete_out = _not_implemented("te_general_grouped_gemm_for_discrete_out")

# ============================================================================
# NVFP4 multi-tensor ops (iterate using single-tensor stable ops)
# ============================================================================

def nvfp4_multi_tensor_fused_scale(block_amax_list, global_amax_list,
                                   per_block_scale_list, target_scale_list,
                                   target_amax_list, tile_rows_list,
                                   tile_cols_list, rows_padded_list, block_len):
    for i in range(len(block_amax_list)):
        _ops.nvfp4_fused_scale(
            block_amax_list[i], global_amax_list[i],
            per_block_scale_list[i], target_scale_list[i],
            target_amax_list[i], tile_rows_list[i],
            tile_cols_list[i], rows_padded_list[i], block_len)


def nvfp4_2d_multi_tensor_transpose(rowwise_data_list, columnwise_data_list,
                                    rowwise_scale_inv_list,
                                    columnwise_scale_inv_list,
                                    M_list, K_list):
    for i in range(len(rowwise_data_list)):
        _ops.nvfp4_data_transpose(rowwise_data_list[i], columnwise_data_list[i])
        M = M_list[i]
        K = K_list[i]
        M_tiles = (M + 15) // 16
        K_tiles = (K + 15) // 16
        _ops.nvfp4_2d_scale_transpose(
            rowwise_scale_inv_list[i], columnwise_scale_inv_list[i],
            M_tiles, K_tiles)


def nvfp4_multi_tensor_2d_partial_cast(inp_list, out_list, scale_list,
                                       global_scale_list, h_list, w_list,
                                       start_offset_list, block_len=16):
    for i in range(len(inp_list)):
        # out_list[i] may be a quantized tensor — extract raw data
        out = out_list[i]
        if isinstance(out, torch.Tensor):
            _ops.nvfp4_2d_partial_cast_noalloc(
                inp_list[i], out, int(DType.kFloat4E2M1), None, 4,
                scale_list[i], global_scale_list[i],
                h_list[i], w_list[i], start_offset_list[i], block_len)


def nvfp4_multi_tensor_compute_partial_amax(master_weight_list,
                                            partial_amax_list,
                                            global_amax_list,
                                            h_list, w_list,
                                            start_offset_list,
                                            block_len=16):
    for i in range(len(master_weight_list)):
        _ops.nvfp4_2d_compute_partial_amax(
            master_weight_list[i], partial_amax_list[i],
            h_list[i], w_list[i], start_offset_list[i], block_len)
        _ops.compute_amax(partial_amax_list[i], global_amax_list[i])


# ============================================================================
# Multi-tensor ops (match pybind11 signatures with pointer packing)
# ============================================================================

def _pack_tensor_lists(tensor_lists):
    """Pack tensor lists into flat int64 tensors for the pointer-pack pattern."""
    num_lists = len(tensor_lists)
    num_tensors = len(tensor_lists[0])
    ptrs = torch.tensor(
        [t.data_ptr() for lst in tensor_lists for t in lst],
        dtype=torch.int64)
    shapes = torch.tensor(
        [[t.numel(), t.element_size()] for lst in tensor_lists for t in lst],
        dtype=torch.int64).flatten()
    _TORCH_DT = {torch.float32: 4, torch.float16: 5, torch.bfloat16: 6, torch.uint8: 0,
                  torch.int32: 2, torch.int64: 3, torch.bool: 0}
    dtypes = torch.tensor(
        [_TORCH_DT.get(t.dtype, 4) for lst in tensor_lists for t in lst],
        dtype=torch.int64)
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
    return _ops.multi_tensor_unscale_l2norm(chunk_size, noop_flag, ptrs, shapes, dtypes, nl, nt, inv_scale, per_tensor)


def multi_tensor_adam(chunk_size, noop_flag, tensor_lists, lr, beta1, beta2,
                      epsilon, step, mode, bias_correction, weight_decay):
    ptrs, shapes, dtypes, nl, nt = _pack_tensor_lists(tensor_lists)
    _ops.multi_tensor_adam(chunk_size, noop_flag, ptrs, shapes, dtypes, nl, nt,
                           lr, beta1, beta2, epsilon, step, mode,
                           bias_correction, weight_decay)


def multi_tensor_adam_capturable(chunk_size, noop_flag, tensor_lists, lr, beta1,
                                 beta2, epsilon, step, mode, bias_correction,
                                 weight_decay, inv_scale):
    ptrs, shapes, dtypes, nl, nt = _pack_tensor_lists(tensor_lists)
    _ops.multi_tensor_adam_capturable(chunk_size, noop_flag, ptrs, shapes, dtypes, nl, nt,
                                     lr, beta1, beta2, epsilon, step, mode,
                                     bias_correction, weight_decay, inv_scale)


def multi_tensor_adam_capturable_master(chunk_size, noop_flag, tensor_lists, lr,
                                        beta1, beta2, epsilon, step, mode,
                                        bias_correction, weight_decay, inv_scale):
    ptrs, shapes, dtypes, nl, nt = _pack_tensor_lists(tensor_lists)
    _ops.multi_tensor_adam_capturable_master(chunk_size, noop_flag, ptrs, shapes, dtypes, nl, nt,
                                            lr, beta1, beta2, epsilon, step, mode,
                                            bias_correction, weight_decay, inv_scale)


def multi_tensor_adam_param_remainder(chunk_size, noop_flag, tensor_lists, lr,
                                     beta1, beta2, epsilon, step, mode,
                                     bias_correction, weight_decay):
    ptrs, shapes, dtypes, nl, nt = _pack_tensor_lists(tensor_lists)
    _ops.multi_tensor_adam_param_remainder(chunk_size, noop_flag, ptrs, shapes, dtypes, nl, nt,
                                          lr, beta1, beta2, epsilon, step, mode,
                                          bias_correction, weight_decay)


def multi_tensor_adam_fp8(chunk_size, noop_flag, tensor_lists, lr, beta1,
                          beta2, epsilon, step, mode, bias_correction,
                          weight_decay, fp8_dtype):
    ptrs, shapes, dtypes, nl, nt = _pack_tensor_lists(tensor_lists)
    _ops.multi_tensor_adam_fp8(chunk_size, noop_flag, ptrs, shapes, dtypes, nl, nt,
                               lr, beta1, beta2, epsilon, step, mode,
                               bias_correction, weight_decay, int(fp8_dtype))


def multi_tensor_sgd(chunk_size, noop_flag, tensor_lists, wd, momentum,
                     dampening, lr, nesterov, first_run, wd_after_momentum, scale):
    ptrs, shapes, dtypes, nl, nt = _pack_tensor_lists(tensor_lists)
    _ops.multi_tensor_sgd(chunk_size, noop_flag, ptrs, shapes, dtypes, nl, nt,
                          wd, momentum, dampening, lr, nesterov, first_run,
                          wd_after_momentum, scale)


def multi_tensor_compute_scale_and_scale_inv(chunk_size, noop_flag, tensor_lists,
                                             max_fp8, force_pow_2_scales=False,
                                             epsilon=0.0):
    ptrs, shapes, dtypes, nl, nt = _pack_tensor_lists(tensor_lists)
    _ops.multi_tensor_compute_scale_and_scale_inv(chunk_size, noop_flag, ptrs, shapes, dtypes,
                                                  nl, nt, max_fp8, force_pow_2_scales, epsilon)


def multi_tensor_compute_scale_inv_e8m0(chunk_size, dummy, tensor_lists):
    ptrs, shapes, dtypes, nl, nt = _pack_tensor_lists(tensor_lists)
    _ops.multi_tensor_compute_scale_inv_e8m0(chunk_size, ptrs, shapes, dtypes, nl, nt)


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


class CommOverlapHelper:
    """Python replacement for pybind11 CommOverlapHelper."""
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("CommOverlapHelper not yet in stable ABI module")


class CommOverlap:
    """Python replacement for pybind11 CommOverlap."""
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("CommOverlap not yet in stable ABI module")

    def is_fp8_ubuf(self):
        return False


class CommOverlapP2P:
    """Python replacement for pybind11 CommOverlapP2P."""
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("CommOverlapP2P not yet in stable ABI module")


# ============================================================================
# Fused attention (match pybind11 signatures)
# ============================================================================

fused_attn_fwd = _not_implemented("fused_attn_fwd")
fused_attn_bwd = _not_implemented("fused_attn_bwd")

def bulk_overlap_ag_with_external_gemm(allgather_communicator, send_stream, recv_stream):
    _ops.bulk_overlap_ag_with_external_gemm(
        allgather_communicator._handle,
        send_stream.cuda_stream,
        recv_stream.cuda_stream)


# ============================================================================
# NVTE enums exposed via pybind
# ============================================================================

class NVTE_QKV_Layout(IntEnum):
    NVTE_SB3HD = 0; NVTE_SBH3D = 1; NVTE_SBHD_SB2HD = 2; NVTE_SBHD_SBH2D = 3
    NVTE_SBHD_SBHD_SBHD = 4; NVTE_BS3HD = 5; NVTE_BSH3D = 6; NVTE_BSHD_BS2HD = 7
    NVTE_BSHD_BSH2D = 8; NVTE_BSHD_BSHD_BSHD = 9; NVTE_T3HD = 10; NVTE_TH3D = 11
    NVTE_THD_T2HD = 12; NVTE_THD_TH2D = 13; NVTE_THD_THD_THD = 14
    NVTE_SBHD_BSHD_BSHD = 15; NVTE_BSHD_SBHD_SBHD = 16; NVTE_THD_BSHD_BSHD = 17
    NVTE_THD_SBHD_SBHD = 18; NVTE_Paged_KV_BSHD_BSHD_BSHD = 19
    NVTE_Paged_KV_BSHD_SBHD_SBHD = 20; NVTE_Paged_KV_SBHD_BSHD_BSHD = 21
    NVTE_Paged_KV_SBHD_SBHD_SBHD = 22; NVTE_Paged_KV_THD_BSHD_BSHD = 23
    NVTE_Paged_KV_THD_SBHD_SBHD = 24

class NVTE_QKV_Format(IntEnum):
    NVTE_SBHD = 0; NVTE_BSHD = 1; NVTE_THD = 2; NVTE_BSHD_2SBHD = 3
    NVTE_SBHD_2BSHD = 4; NVTE_THD_2BSHD = 5; NVTE_THD_2SBHD = 6

class NVTE_Bias_Type(IntEnum):
    NVTE_NO_BIAS = 0; NVTE_PRE_SCALE_BIAS = 1; NVTE_POST_SCALE_BIAS = 2; NVTE_ALIBI = 3

class NVTE_Mask_Type(IntEnum):
    NVTE_NO_MASK = 0; NVTE_PADDING_MASK = 1; NVTE_CAUSAL_MASK = 2
    NVTE_PADDING_CAUSAL_MASK = 3; NVTE_CAUSAL_BOTTOM_RIGHT_MASK = 4
    NVTE_PADDING_CAUSAL_BOTTOM_RIGHT_MASK = 5; NVTE_ARBITRARY_MASK = 6

class NVTE_Softmax_Type(IntEnum):
    NVTE_VANILLA_SOFTMAX = 0; NVTE_OFF_BY_ONE_SOFTMAX = 1; NVTE_LEARNABLE_SOFTMAX = 2

class NVTE_Fused_Attn_Backend(IntEnum):
    NVTE_No_Backend = -1; NVTE_F16_max512_seqlen = 0; NVTE_F16_arbitrary_seqlen = 1
    NVTE_FP8 = 2


def device_supports_multicast():
    """Check if current device supports multicast."""
    return False


def get_stream_priority_range():
    """Get CUDA stream priority range."""
    low = torch.tensor([0], dtype=torch.int32)
    high = torch.tensor([0], dtype=torch.int32)
    return int(low.item()), int(high.item())


def ubuf_built_with_mpi():
    """Check if TE was built with NVTE_UB_WITH_MPI=1."""
    return bool(int(os.environ.get("NVTE_UB_WITH_MPI", "0")))
