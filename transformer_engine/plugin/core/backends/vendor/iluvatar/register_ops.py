# Copyright (c) 2025, BAAI. All rights reserved.
#
# See LICENSE for license information.

"""
Iluvatar vendor backend operator registrations.

This module registers all VENDOR (Iluvatar) implementations from transformer_engine_torch.
"""

from __future__ import annotations

import functools

from ....types import OpImpl, BackendImplKind


def _bind_is_available(fn, is_available_fn):
    """Wrap a function and bind _is_available attribute for OpImpl.is_available() check."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        return fn(*args, **kwargs)
    wrapper._is_available = is_available_fn
    return wrapper


def register_builtins(registry) -> None:
    """
    Register all Iluvatar (VENDOR) operator implementations.

    Args:
        registry: Registry to register into
    """
    # Import Iluvatar backend to get all the wrapped tex functions
    from .iluvatar import IluvatarBackend

    # Create a backend instance to access the methods
    backend = IluvatarBackend()

    # Check if Iluvatar is available before registering
    if not backend.is_available():
        return

    # Bind is_available to all methods
    is_avail = backend.is_available

    impls = [
        # Normalization
        OpImpl(op_name="rmsnorm_fwd", impl_id="vendor.iluvatar", kind=BackendImplKind.VENDOR, fn=_bind_is_available(backend.rmsnorm_fwd, is_avail), vendor="Iluvatar", priority=100),
        OpImpl(op_name="rmsnorm_bwd", impl_id="vendor.iluvatar", kind=BackendImplKind.VENDOR, fn=_bind_is_available(backend.rmsnorm_bwd, is_avail), vendor="Iluvatar", priority=100),
        OpImpl(op_name="rmsnorm_bwd_add", impl_id="vendor.iluvatar", kind=BackendImplKind.VENDOR, fn=_bind_is_available(backend.rmsnorm_bwd_add, is_avail), vendor="Iluvatar", priority=100),
        OpImpl(op_name="layernorm_fwd", impl_id="vendor.iluvatar", kind=BackendImplKind.VENDOR, fn=_bind_is_available(backend.layernorm_fwd, is_avail), vendor="Iluvatar", priority=100),
        OpImpl(op_name="layernorm_bwd", impl_id="vendor.iluvatar", kind=BackendImplKind.VENDOR, fn=_bind_is_available(backend.layernorm_bwd, is_avail), vendor="Iluvatar", priority=100),

        # GEMM
        OpImpl(op_name="generic_gemm", impl_id="vendor.iluvatar", kind=BackendImplKind.VENDOR, fn=_bind_is_available(backend.generic_gemm, is_avail), vendor="Iluvatar", priority=100),
        OpImpl(op_name="te_general_grouped_gemm", impl_id="vendor.iluvatar", kind=BackendImplKind.VENDOR, fn=_bind_is_available(backend.te_general_grouped_gemm, is_avail), vendor="Iluvatar", priority=100),

        # Quantization
        OpImpl(op_name="quantize", impl_id="vendor.iluvatar", kind=BackendImplKind.VENDOR, fn=_bind_is_available(backend.quantize, is_avail), vendor="Iluvatar", priority=100),
        OpImpl(op_name="dequantize", impl_id="vendor.iluvatar", kind=BackendImplKind.VENDOR, fn=_bind_is_available(backend.dequantize, is_avail), vendor="Iluvatar", priority=100),
        OpImpl(op_name="bgrad_quantize", impl_id="vendor.iluvatar", kind=BackendImplKind.VENDOR, fn=_bind_is_available(backend.bgrad_quantize, is_avail), vendor="Iluvatar", priority=100),
        OpImpl(op_name="split_quantize", impl_id="vendor.iluvatar", kind=BackendImplKind.VENDOR, fn=_bind_is_available(backend.split_quantize, is_avail), vendor="Iluvatar", priority=100),

        # Activations - Forward
        OpImpl(op_name="gelu", impl_id="vendor.iluvatar", kind=BackendImplKind.VENDOR, fn=_bind_is_available(backend.gelu, is_avail), vendor="Iluvatar", priority=100),
        OpImpl(op_name="geglu", impl_id="vendor.iluvatar", kind=BackendImplKind.VENDOR, fn=_bind_is_available(backend.geglu, is_avail), vendor="Iluvatar", priority=100),
        OpImpl(op_name="qgelu", impl_id="vendor.iluvatar", kind=BackendImplKind.VENDOR, fn=_bind_is_available(backend.qgelu, is_avail), vendor="Iluvatar", priority=100),
        OpImpl(op_name="qgeglu", impl_id="vendor.iluvatar", kind=BackendImplKind.VENDOR, fn=_bind_is_available(backend.qgeglu, is_avail), vendor="Iluvatar", priority=100),
        OpImpl(op_name="relu", impl_id="vendor.iluvatar", kind=BackendImplKind.VENDOR, fn=_bind_is_available(backend.relu, is_avail), vendor="Iluvatar", priority=100),
        OpImpl(op_name="reglu", impl_id="vendor.iluvatar", kind=BackendImplKind.VENDOR, fn=_bind_is_available(backend.reglu, is_avail), vendor="Iluvatar", priority=100),
        OpImpl(op_name="srelu", impl_id="vendor.iluvatar", kind=BackendImplKind.VENDOR, fn=_bind_is_available(backend.srelu, is_avail), vendor="Iluvatar", priority=100),
        OpImpl(op_name="sreglu", impl_id="vendor.iluvatar", kind=BackendImplKind.VENDOR, fn=_bind_is_available(backend.sreglu, is_avail), vendor="Iluvatar", priority=100),
        OpImpl(op_name="silu", impl_id="vendor.iluvatar", kind=BackendImplKind.VENDOR, fn=_bind_is_available(backend.silu, is_avail), vendor="Iluvatar", priority=100),
        OpImpl(op_name="swiglu", impl_id="vendor.iluvatar", kind=BackendImplKind.VENDOR, fn=_bind_is_available(backend.swiglu, is_avail), vendor="Iluvatar", priority=100),
        OpImpl(op_name="clamped_swiglu", impl_id="vendor.iluvatar", kind=BackendImplKind.VENDOR, fn=_bind_is_available(backend.clamped_swiglu, is_avail), vendor="Iluvatar", priority=100),

        # Activations - Backward
        OpImpl(op_name="dgelu", impl_id="vendor.iluvatar", kind=BackendImplKind.VENDOR, fn=_bind_is_available(backend.dgelu, is_avail), vendor="Iluvatar", priority=100),
        OpImpl(op_name="dgeglu", impl_id="vendor.iluvatar", kind=BackendImplKind.VENDOR, fn=_bind_is_available(backend.dgeglu, is_avail), vendor="Iluvatar", priority=100),
        OpImpl(op_name="dqgelu", impl_id="vendor.iluvatar", kind=BackendImplKind.VENDOR, fn=_bind_is_available(backend.dqgelu, is_avail), vendor="Iluvatar", priority=100),
        OpImpl(op_name="dqgeglu", impl_id="vendor.iluvatar", kind=BackendImplKind.VENDOR, fn=_bind_is_available(backend.dqgeglu, is_avail), vendor="Iluvatar", priority=100),
        OpImpl(op_name="drelu", impl_id="vendor.iluvatar", kind=BackendImplKind.VENDOR, fn=_bind_is_available(backend.drelu, is_avail), vendor="Iluvatar", priority=100),
        OpImpl(op_name="dreglu", impl_id="vendor.iluvatar", kind=BackendImplKind.VENDOR, fn=_bind_is_available(backend.dreglu, is_avail), vendor="Iluvatar", priority=100),
        OpImpl(op_name="dsrelu", impl_id="vendor.iluvatar", kind=BackendImplKind.VENDOR, fn=_bind_is_available(backend.dsrelu, is_avail), vendor="Iluvatar", priority=100),
        OpImpl(op_name="dsreglu", impl_id="vendor.iluvatar", kind=BackendImplKind.VENDOR, fn=_bind_is_available(backend.dsreglu, is_avail), vendor="Iluvatar", priority=100),
        OpImpl(op_name="dsilu", impl_id="vendor.iluvatar", kind=BackendImplKind.VENDOR, fn=_bind_is_available(backend.dsilu, is_avail), vendor="Iluvatar", priority=100),
        OpImpl(op_name="dswiglu", impl_id="vendor.iluvatar", kind=BackendImplKind.VENDOR, fn=_bind_is_available(backend.dswiglu, is_avail), vendor="Iluvatar", priority=100),
        OpImpl(op_name="clamped_dswiglu", impl_id="vendor.iluvatar", kind=BackendImplKind.VENDOR, fn=_bind_is_available(backend.clamped_dswiglu, is_avail), vendor="Iluvatar", priority=100),

        # Activations - Bias + Backward
        OpImpl(op_name="dbias_dgelu", impl_id="vendor.iluvatar", kind=BackendImplKind.VENDOR, fn=_bind_is_available(backend.dbias_dgelu, is_avail), vendor="Iluvatar", priority=100),
        OpImpl(op_name="dbias_dsilu", impl_id="vendor.iluvatar", kind=BackendImplKind.VENDOR, fn=_bind_is_available(backend.dbias_dsilu, is_avail), vendor="Iluvatar", priority=100),
        OpImpl(op_name="dbias_drelu", impl_id="vendor.iluvatar", kind=BackendImplKind.VENDOR, fn=_bind_is_available(backend.dbias_drelu, is_avail), vendor="Iluvatar", priority=100),
        OpImpl(op_name="dbias_dqgelu", impl_id="vendor.iluvatar", kind=BackendImplKind.VENDOR, fn=_bind_is_available(backend.dbias_dqgelu, is_avail), vendor="Iluvatar", priority=100),
        OpImpl(op_name="dbias_dsrelu", impl_id="vendor.iluvatar", kind=BackendImplKind.VENDOR, fn=_bind_is_available(backend.dbias_dsrelu, is_avail), vendor="Iluvatar", priority=100),

        # Softmax
        OpImpl(op_name="scaled_softmax_forward", impl_id="vendor.iluvatar", kind=BackendImplKind.VENDOR, fn=_bind_is_available(backend.scaled_softmax_forward, is_avail), vendor="Iluvatar", priority=100),
        OpImpl(op_name="scaled_softmax_backward", impl_id="vendor.iluvatar", kind=BackendImplKind.VENDOR, fn=_bind_is_available(backend.scaled_softmax_backward, is_avail), vendor="Iluvatar", priority=100),
        OpImpl(op_name="scaled_masked_softmax_forward", impl_id="vendor.iluvatar", kind=BackendImplKind.VENDOR, fn=_bind_is_available(backend.scaled_masked_softmax_forward, is_avail), vendor="Iluvatar", priority=100),
        OpImpl(op_name="scaled_masked_softmax_backward", impl_id="vendor.iluvatar", kind=BackendImplKind.VENDOR, fn=_bind_is_available(backend.scaled_masked_softmax_backward, is_avail), vendor="Iluvatar", priority=100),
        OpImpl(op_name="scaled_upper_triang_masked_softmax_forward", impl_id="vendor.iluvatar", kind=BackendImplKind.VENDOR, fn=_bind_is_available(backend.scaled_upper_triang_masked_softmax_forward, is_avail), vendor="Iluvatar", priority=100),
        OpImpl(op_name="scaled_upper_triang_masked_softmax_backward", impl_id="vendor.iluvatar", kind=BackendImplKind.VENDOR, fn=_bind_is_available(backend.scaled_upper_triang_masked_softmax_backward, is_avail), vendor="Iluvatar", priority=100),
        OpImpl(op_name="scaled_aligned_causal_masked_softmax_forward", impl_id="vendor.iluvatar", kind=BackendImplKind.VENDOR, fn=_bind_is_available(backend.scaled_aligned_causal_masked_softmax_forward, is_avail), vendor="Iluvatar", priority=100),
        OpImpl(op_name="scaled_aligned_causal_masked_softmax_backward", impl_id="vendor.iluvatar", kind=BackendImplKind.VENDOR, fn=_bind_is_available(backend.scaled_aligned_causal_masked_softmax_backward, is_avail), vendor="Iluvatar", priority=100),

        # MOE operations
        OpImpl(op_name="moe_permute_fwd", impl_id="vendor.iluvatar", kind=BackendImplKind.VENDOR, fn=_bind_is_available(backend.moe_permute_fwd, is_avail), vendor="Iluvatar", priority=100),
        OpImpl(op_name="moe_permute_bwd", impl_id="vendor.iluvatar", kind=BackendImplKind.VENDOR, fn=_bind_is_available(backend.moe_permute_bwd, is_avail), vendor="Iluvatar", priority=100),
        OpImpl(op_name="moe_unpermute_fwd", impl_id="vendor.iluvatar", kind=BackendImplKind.VENDOR, fn=_bind_is_available(backend.moe_unpermute_fwd, is_avail), vendor="Iluvatar", priority=100),
        OpImpl(op_name="moe_unpermute_bwd", impl_id="vendor.iluvatar", kind=BackendImplKind.VENDOR, fn=_bind_is_available(backend.moe_unpermute_bwd, is_avail), vendor="Iluvatar", priority=100),

        # Fused attention
        OpImpl(op_name="get_fused_attn_backend", impl_id="vendor.iluvatar", kind=BackendImplKind.VENDOR, fn=_bind_is_available(backend.get_fused_attn_backend, is_avail), vendor="Iluvatar", priority=100),
        OpImpl(op_name="fused_attn_fwd", impl_id="vendor.iluvatar", kind=BackendImplKind.VENDOR, fn=_bind_is_available(backend.fused_attn_fwd, is_avail), vendor="Iluvatar", priority=100),
        OpImpl(op_name="fused_attn_bwd", impl_id="vendor.iluvatar", kind=BackendImplKind.VENDOR, fn=_bind_is_available(backend.fused_attn_bwd, is_avail), vendor="Iluvatar", priority=100),
        OpImpl(op_name="fa_prepare_fwd", impl_id="vendor.iluvatar", kind=BackendImplKind.VENDOR, fn=_bind_is_available(backend.fa_prepare_fwd, is_avail), vendor="Iluvatar", priority=100),
        OpImpl(op_name="fa_prepare_bwd", impl_id="vendor.iluvatar", kind=BackendImplKind.VENDOR, fn=_bind_is_available(backend.fa_prepare_bwd, is_avail), vendor="Iluvatar", priority=100),

        # KV cache
        OpImpl(op_name="copy_to_kv_cache", impl_id="vendor.iluvatar", kind=BackendImplKind.VENDOR, fn=_bind_is_available(backend.copy_to_kv_cache, is_avail), vendor="Iluvatar", priority=100),

        # Tensor format conversions
        OpImpl(op_name="convert_thd_to_bshd", impl_id="vendor.iluvatar", kind=BackendImplKind.VENDOR, fn=_bind_is_available(backend.convert_thd_to_bshd, is_avail), vendor="Iluvatar", priority=100),
        OpImpl(op_name="convert_bshd_to_thd", impl_id="vendor.iluvatar", kind=BackendImplKind.VENDOR, fn=_bind_is_available(backend.convert_bshd_to_thd, is_avail), vendor="Iluvatar", priority=100),

        # RoPE (Rotary Position Embedding)
        OpImpl(op_name="fused_rope_forward", impl_id="vendor.iluvatar", kind=BackendImplKind.VENDOR, fn=_bind_is_available(backend.fused_rope_forward, is_avail), vendor="Iluvatar", priority=100),
        OpImpl(op_name="fused_rope_backward", impl_id="vendor.iluvatar", kind=BackendImplKind.VENDOR, fn=_bind_is_available(backend.fused_rope_backward, is_avail), vendor="Iluvatar", priority=100),
        OpImpl(op_name="fused_qkv_rope_forward", impl_id="vendor.iluvatar", kind=BackendImplKind.VENDOR, fn=_bind_is_available(backend.fused_qkv_rope_forward, is_avail), vendor="Iluvatar", priority=100),
        OpImpl(op_name="fused_qkv_rope_backward", impl_id="vendor.iluvatar", kind=BackendImplKind.VENDOR, fn=_bind_is_available(backend.fused_qkv_rope_backward, is_avail), vendor="Iluvatar", priority=100),

        # TopK and MOE aux loss
        OpImpl(op_name="fused_topk_with_score_function_fwd", impl_id="vendor.iluvatar", kind=BackendImplKind.VENDOR, fn=_bind_is_available(backend.fused_topk_with_score_function_fwd, is_avail), vendor="Iluvatar", priority=100),
        OpImpl(op_name="fused_topk_with_score_function_bwd", impl_id="vendor.iluvatar", kind=BackendImplKind.VENDOR, fn=_bind_is_available(backend.fused_topk_with_score_function_bwd, is_avail), vendor="Iluvatar", priority=100),
        OpImpl(op_name="fused_score_for_moe_aux_loss_fwd", impl_id="vendor.iluvatar", kind=BackendImplKind.VENDOR, fn=_bind_is_available(backend.fused_score_for_moe_aux_loss_fwd, is_avail), vendor="Iluvatar", priority=100),
        OpImpl(op_name="fused_score_for_moe_aux_loss_bwd", impl_id="vendor.iluvatar", kind=BackendImplKind.VENDOR, fn=_bind_is_available(backend.fused_score_for_moe_aux_loss_bwd, is_avail), vendor="Iluvatar", priority=100),
        OpImpl(op_name="fused_moe_aux_loss_fwd", impl_id="vendor.iluvatar", kind=BackendImplKind.VENDOR, fn=_bind_is_available(backend.fused_moe_aux_loss_fwd, is_avail), vendor="Iluvatar", priority=100),
        OpImpl(op_name="fused_moe_aux_loss_bwd", impl_id="vendor.iluvatar", kind=BackendImplKind.VENDOR, fn=_bind_is_available(backend.fused_moe_aux_loss_bwd, is_avail), vendor="Iluvatar", priority=100),

        # Dropout
        OpImpl(op_name="dropout_fwd", impl_id="vendor.iluvatar", kind=BackendImplKind.VENDOR, fn=_bind_is_available(backend.dropout_fwd, is_avail), vendor="Iluvatar", priority=100),
        OpImpl(op_name="dropout_bwd", impl_id="vendor.iluvatar", kind=BackendImplKind.VENDOR, fn=_bind_is_available(backend.dropout_bwd, is_avail), vendor="Iluvatar", priority=100),

        # FP8 operations
        OpImpl(op_name="fp8_transpose", impl_id="vendor.iluvatar", kind=BackendImplKind.VENDOR, fn=_bind_is_available(backend.fp8_transpose, is_avail), vendor="Iluvatar", priority=100),
        OpImpl(op_name="swap_first_dims", impl_id="vendor.iluvatar", kind=BackendImplKind.VENDOR, fn=_bind_is_available(backend.swap_first_dims, is_avail), vendor="Iluvatar", priority=100),
        OpImpl(op_name="compute_amax", impl_id="vendor.iluvatar", kind=BackendImplKind.VENDOR, fn=_bind_is_available(backend.compute_amax, is_avail), vendor="Iluvatar", priority=100),
        OpImpl(op_name="fused_amax_and_scale_update_after_reduction", impl_id="vendor.iluvatar", kind=BackendImplKind.VENDOR, fn=_bind_is_available(backend.fused_amax_and_scale_update_after_reduction, is_avail), vendor="Iluvatar", priority=100),
        OpImpl(op_name="fp8_block_scaling_compute_partial_amax", impl_id="vendor.iluvatar", kind=BackendImplKind.VENDOR, fn=_bind_is_available(backend.fp8_block_scaling_compute_partial_amax, is_avail), vendor="Iluvatar", priority=100),
        OpImpl(op_name="fp8_block_scaling_partial_cast", impl_id="vendor.iluvatar", kind=BackendImplKind.VENDOR, fn=_bind_is_available(backend.fp8_block_scaling_partial_cast, is_avail), vendor="Iluvatar", priority=100),

        # Padding operations
        OpImpl(op_name="fused_multi_row_padding", impl_id="vendor.iluvatar", kind=BackendImplKind.VENDOR, fn=_bind_is_available(backend.fused_multi_row_padding, is_avail), vendor="Iluvatar", priority=100),
        OpImpl(op_name="fused_multi_row_unpadding", impl_id="vendor.iluvatar", kind=BackendImplKind.VENDOR, fn=_bind_is_available(backend.fused_multi_row_unpadding, is_avail), vendor="Iluvatar", priority=100),

        # Library version getters
        OpImpl(op_name="get_cublasLt_version", impl_id="vendor.iluvatar", kind=BackendImplKind.VENDOR, fn=_bind_is_available(backend.get_cublasLt_version, is_avail), vendor="Iluvatar", priority=100),
        OpImpl(op_name="get_cudnn_version", impl_id="vendor.iluvatar", kind=BackendImplKind.VENDOR, fn=_bind_is_available(backend.get_cudnn_version, is_avail), vendor="Iluvatar", priority=100),
        OpImpl(op_name="get_num_cublas_streams", impl_id="vendor.iluvatar", kind=BackendImplKind.VENDOR, fn=_bind_is_available(backend.get_num_cublas_streams, is_avail), vendor="Iluvatar", priority=100),

        # THD (Tensor, Hidden, Dimension) operations
        OpImpl(op_name="thd_read_half_tensor", impl_id="vendor.iluvatar", kind=BackendImplKind.VENDOR, fn=_bind_is_available(backend.thd_read_half_tensor, is_avail), vendor="Iluvatar", priority=100),
        OpImpl(op_name="thd_second_half_lse_correction", impl_id="vendor.iluvatar", kind=BackendImplKind.VENDOR, fn=_bind_is_available(backend.thd_second_half_lse_correction, is_avail), vendor="Iluvatar", priority=100),
        OpImpl(op_name="thd_read_second_half_lse", impl_id="vendor.iluvatar", kind=BackendImplKind.VENDOR, fn=_bind_is_available(backend.thd_read_second_half_lse, is_avail), vendor="Iluvatar", priority=100),
        OpImpl(op_name="thd_out_correction", impl_id="vendor.iluvatar", kind=BackendImplKind.VENDOR, fn=_bind_is_available(backend.thd_out_correction, is_avail), vendor="Iluvatar", priority=100),
        OpImpl(op_name="thd_grad_correction", impl_id="vendor.iluvatar", kind=BackendImplKind.VENDOR, fn=_bind_is_available(backend.thd_grad_correction, is_avail), vendor="Iluvatar", priority=100),
        OpImpl(op_name="thd_get_partitioned_indices", impl_id="vendor.iluvatar", kind=BackendImplKind.VENDOR, fn=_bind_is_available(backend.thd_get_partitioned_indices, is_avail), vendor="Iluvatar", priority=100),

        # NVSHMEM operations
        OpImpl(op_name="init_nvshmem_backend", impl_id="vendor.iluvatar", kind=BackendImplKind.VENDOR, fn=_bind_is_available(backend.init_nvshmem_backend, is_avail), vendor="Iluvatar", priority=100),
        OpImpl(op_name="create_nvshmem_tensor", impl_id="vendor.iluvatar", kind=BackendImplKind.VENDOR, fn=_bind_is_available(backend.create_nvshmem_tensor, is_avail), vendor="Iluvatar", priority=100),
        OpImpl(op_name="nvshmem_send_on_current_stream", impl_id="vendor.iluvatar", kind=BackendImplKind.VENDOR, fn=_bind_is_available(backend.nvshmem_send_on_current_stream, is_avail), vendor="Iluvatar", priority=100),
        OpImpl(op_name="nvshmem_wait_on_current_stream", impl_id="vendor.iluvatar", kind=BackendImplKind.VENDOR, fn=_bind_is_available(backend.nvshmem_wait_on_current_stream, is_avail), vendor="Iluvatar", priority=100),
        OpImpl(op_name="nvshmem_finalize", impl_id="vendor.iluvatar", kind=BackendImplKind.VENDOR, fn=_bind_is_available(backend.nvshmem_finalize, is_avail), vendor="Iluvatar", priority=100),

        # Multi-tensor operations
        OpImpl(op_name="multi_tensor_quantize", impl_id="vendor.iluvatar", kind=BackendImplKind.VENDOR, fn=_bind_is_available(backend.multi_tensor_quantize, is_avail), vendor="Iluvatar", priority=100),
        OpImpl(op_name="multi_tensor_scale", impl_id="vendor.iluvatar", kind=BackendImplKind.VENDOR, fn=_bind_is_available(backend.multi_tensor_scale, is_avail), vendor="Iluvatar", priority=100),
        OpImpl(op_name="multi_tensor_l2norm", impl_id="vendor.iluvatar", kind=BackendImplKind.VENDOR, fn=_bind_is_available(backend.multi_tensor_l2norm, is_avail), vendor="Iluvatar", priority=100),
        OpImpl(op_name="multi_tensor_unscale_l2norm", impl_id="vendor.iluvatar", kind=BackendImplKind.VENDOR, fn=_bind_is_available(backend.multi_tensor_unscale_l2norm, is_avail), vendor="Iluvatar", priority=100),
        OpImpl(op_name="multi_tensor_adam", impl_id="vendor.iluvatar", kind=BackendImplKind.VENDOR, fn=_bind_is_available(backend.multi_tensor_adam, is_avail), vendor="Iluvatar", priority=100),
        OpImpl(op_name="multi_tensor_adam_param_remainder", impl_id="vendor.iluvatar", kind=BackendImplKind.VENDOR, fn=_bind_is_available(backend.multi_tensor_adam_param_remainder, is_avail), vendor="Iluvatar", priority=100),
        OpImpl(op_name="multi_tensor_adam_fp8", impl_id="vendor.iluvatar", kind=BackendImplKind.VENDOR, fn=_bind_is_available(backend.multi_tensor_adam_fp8, is_avail), vendor="Iluvatar", priority=100),
        OpImpl(op_name="multi_tensor_adam_capturable", impl_id="vendor.iluvatar", kind=BackendImplKind.VENDOR, fn=_bind_is_available(backend.multi_tensor_adam_capturable, is_avail), vendor="Iluvatar", priority=100),
        OpImpl(op_name="multi_tensor_adam_capturable_master", impl_id="vendor.iluvatar", kind=BackendImplKind.VENDOR, fn=_bind_is_available(backend.multi_tensor_adam_capturable_master, is_avail), vendor="Iluvatar", priority=100),
        OpImpl(op_name="multi_tensor_sgd", impl_id="vendor.iluvatar", kind=BackendImplKind.VENDOR, fn=_bind_is_available(backend.multi_tensor_sgd, is_avail), vendor="Iluvatar", priority=100),
        OpImpl(op_name="multi_tensor_compute_scale_and_scale_inv", impl_id="vendor.iluvatar", kind=BackendImplKind.VENDOR, fn=_bind_is_available(backend.multi_tensor_compute_scale_and_scale_inv, is_avail), vendor="Iluvatar", priority=100),

        # Communication overlap operations
        OpImpl(op_name="bulk_overlap_ag_with_external_gemm", impl_id="vendor.iluvatar", kind=BackendImplKind.VENDOR, fn=_bind_is_available(backend.bulk_overlap_ag_with_external_gemm, is_avail), vendor="Iluvatar", priority=100),
        OpImpl(op_name="create_fp8_tensor_meta", impl_id="vendor.iluvatar", kind=BackendImplKind.VENDOR, fn=_bind_is_available(backend.create_fp8_tensor_meta, is_avail), vendor="Iluvatar", priority=100),
        OpImpl(op_name="create_comm_overlap_helper", impl_id="vendor.iluvatar", kind=BackendImplKind.VENDOR, fn=_bind_is_available(backend.create_comm_overlap_helper, is_avail), vendor="Iluvatar", priority=100),
        OpImpl(op_name="create_comm_overlap", impl_id="vendor.iluvatar", kind=BackendImplKind.VENDOR, fn=_bind_is_available(backend.create_comm_overlap, is_avail), vendor="Iluvatar", priority=100),
        OpImpl(op_name="create_comm_overlap_p2p", impl_id="vendor.iluvatar", kind=BackendImplKind.VENDOR, fn=_bind_is_available(backend.create_comm_overlap_p2p, is_avail), vendor="Iluvatar", priority=100),

        # FlashAttention class getter
        OpImpl(op_name="get_flash_attention_class", impl_id="vendor.iluvatar", kind=BackendImplKind.VENDOR, fn=_bind_is_available(backend.get_flash_attention_class, is_avail), vendor="Iluvatar", priority=100),

        # Attention backend selection
        OpImpl(op_name="get_attention_backend", impl_id="vendor.iluvatar", kind=BackendImplKind.VENDOR, fn=_bind_is_available(backend.get_attention_backend, is_avail), vendor="Iluvatar", priority=100),
    ]

    registry.register_many(impls)
