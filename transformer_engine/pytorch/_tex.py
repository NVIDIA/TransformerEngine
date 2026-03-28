# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Routing module for transformer_engine_torch.

Base: pybind11 .so. Overrides are added incrementally as validated.
"""

from transformer_engine_torch import *  # noqa: F401,F403

# Batch 1: Normalization backward
from transformer_engine.pytorch._stable_torch_module import (
    layernorm_bwd, rmsnorm_bwd, rmsnorm_bwd_add,
    # Batch 2: Softmax
    scaled_softmax_forward, scaled_softmax_backward,
    scaled_masked_softmax_forward, scaled_masked_softmax_backward,
    scaled_upper_triang_masked_softmax_forward,
    scaled_upper_triang_masked_softmax_backward,
    scaled_aligned_causal_masked_softmax_forward,
    scaled_aligned_causal_masked_softmax_backward,
    # Batch 3: Padding + misc
    fused_multi_row_padding, fused_multi_row_unpadding, splits_to_offsets,
    # Batch 4: Router
    fused_topk_with_score_function_fwd, fused_topk_with_score_function_bwd,
    fused_score_for_moe_aux_loss_fwd, fused_score_for_moe_aux_loss_bwd,
    fused_moe_aux_loss_fwd, fused_moe_aux_loss_bwd,
    # Batch 5: Attention helpers
    fa_prepare_fwd, fa_prepare_bwd,
    thd_read_half_tensor, thd_second_half_lse_correction,
    thd_read_second_half_lse, thd_out_correction, thd_grad_correction,
    thd_get_partitioned_indices, convert_thd_to_bshd, convert_bshd_to_thd,
    # Batch 6: RoPE + KV cache
    fused_rope_forward, fused_rope_backward,
    fused_qkv_rope_forward, fused_qkv_rope_backward,
    copy_to_kv_cache,
    # Batch 7: Transpose (no DType args)
    nvfp4_data_transpose, nvfp4_2d_scale_transpose, nvfp4_expand_scale_to_fp8,
    nvfp4_compute_per_block_scale, nvfp4_fused_scale, nvfp4_compute_global_scale,
    swap_first_dims,
    # Batch 8: NVFP4 multi-tensor + partial cast (no DType args)
    nvfp4_multi_tensor_fused_scale, nvfp4_2d_multi_tensor_transpose,
    nvfp4_multi_tensor_compute_partial_amax,
    fp8_block_scaling_compute_partial_amax,
    mxfp8_scaling_compute_partial_amax, mxfp8_scaling_partial_cast,
    nvfp4_2d_compute_partial_amax,
    # Batch 9a: version queries only (compute_amax/fused_amax stay pybind)
    # Batch 10: Multi-tensor ops (pointer-pack pattern)
    multi_tensor_scale, multi_tensor_scale_tensor,
    multi_tensor_l2norm, multi_tensor_unscale_l2norm,
    multi_tensor_adam, multi_tensor_adam_capturable,
    multi_tensor_adam_capturable_master, multi_tensor_adam_param_remainder,
    multi_tensor_sgd,
    multi_tensor_compute_scale_and_scale_inv, multi_tensor_compute_scale_inv_e8m0,
    multi_tensor_quantize, split_quantize, group_quantize,
    # Batch 11: dequantize, dropout, misc stubs
    dequantize,
    dropout_fwd, dropout_bwd,
    # Batch 12: Misc stubs and utilities
    swizzle_scales_for_gemm_,
    nvfp4_2d_partial_cast, nvfp4_multi_tensor_2d_partial_cast,
    init_nvshmem_backend, create_nvshmem_tensor,
    nvshmem_send_on_current_stream, nvshmem_wait_on_current_stream, nvshmem_finalize,
    bulk_overlap_ag_with_external_gemm,
    ubuf_built_with_mpi, device_supports_multicast, get_stream_priority_range,
    # quantize, layernorm_fwd, rmsnorm_fwd: kept from pybind (gradient accum issues)
    # Batch 13: Activation + bias ops (4-path quantizer dispatch)
    gelu, dgelu, glu, dglu, geglu, dgeglu,
    qgelu, dqgelu, qgeglu, dqgeglu,
    relu, drelu, reglu, dreglu,
    srelu, dsrelu, sreglu, dsreglu,
    silu, dsilu, swiglu, dswiglu,
    # clamped_swiglu, clamped_dswiglu: kept from pybind (UNFUSED path tensor metadata)
    # bgrad_quantize, dbias_*: kept from pybind for now
    # GEMM, attention, grouped GEMM: kept from pybind (complex tensor metadata dispatch)
)

# ---------------------------------------------------------------------------
# DType-taking functions: wrapped with int() conversion
# These accept both pybind DType enums AND Python IntEnums
# ---------------------------------------------------------------------------

import transformer_engine.pytorch._stable_torch_module as _sm

def fp8_transpose(input, otype, *, out=None):
    return _sm._ops.fp8_transpose(input, int(otype), out)

def fp8_block_scaling_partial_cast(inp, out, scale, h, w, start_offset, block_len, out_dtype):
    _sm._ops.fp8_block_scaling_partial_cast(inp, out, scale, h, w, start_offset, block_len, int(out_dtype))

# get_fused_attn_backend: kept from pybind (no-tensor-arg dispatch issue)

def moe_permute_fwd(input, dtype, indices, num_out_tokens, workspace, max_expanded_token_num):
    return _sm._ops.moe_permute_fwd(input, int(dtype), indices, workspace, num_out_tokens, max_expanded_token_num)

def moe_permute_bwd(input, dtype, row_id_map, prob, num_tokens, topK):
    return _sm.moe_permute_bwd(input, int(dtype), row_id_map, prob, num_tokens, topK)

def moe_unpermute_fwd(input, dtype, row_id_map, prob, num_tokens, topK):
    return _sm._ops.moe_unpermute_fwd(input, int(dtype), row_id_map, prob, num_tokens, topK)

def moe_unpermute_bwd(input_bwd, input_fwd, dtype, row_id_map, prob):
    return _sm._ops.moe_unpermute_bwd(input_bwd, input_fwd, int(dtype), row_id_map, prob)

def multi_tensor_adam_fp8(chunk_size, noop_flag, tensor_lists, lr, beta1, beta2,
                          epsilon, step, mode, bias_correction, weight_decay, fp8_dtype):
    _sm.multi_tensor_adam_fp8(chunk_size, noop_flag, tensor_lists, lr, beta1, beta2,
                              epsilon, step, mode, bias_correction, weight_decay, int(fp8_dtype))

# generic_gemm, fused_attn: from pybind base import

# Enums and classes: kept from pybind .so (from base import)
# DType, NVTE_*, FP8*, CommOverlap*, Float8BlockScaleTensorFormat
# These are still the pybind versions, which is fine because:
# 1. All function overrides accept int(DType) via wrappers
# 2. Code that does isinstance(x, DType) still works
# 3. Code that does DType.kFloat8E4M3 still returns the pybind enum
