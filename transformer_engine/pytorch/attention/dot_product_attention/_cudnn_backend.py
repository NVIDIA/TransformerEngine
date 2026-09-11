# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""cuDNN SDPA capability selection for the PyTorch frontend."""

from __future__ import annotations

import warnings

from transformer_engine.common.attention.cudnn import (
    FusedAttentionConfig,
    check_f16_fused_attention_support,
    check_fp8_fused_attention_support,
    parse_attention_layout,
)
from transformer_engine.pytorch.constants import DType
from transformer_engine.pytorch.utils import (
    get_cudnn_version,
    get_device_compute_capability,
)


def _dtype_name(dtype: DType) -> str:
    if dtype == DType.kFloat16:
        return "float16"
    if dtype == DType.kBFloat16:
        return "bfloat16"
    if dtype == DType.kFloat8E4M3:
        return "float8_e4m3"
    if dtype == DType.kFloat8E5M2:
        return "float8_e5m2"
    return dtype.name


def get_fused_attn_backend(
    is_training,
    q_dtype,
    kv_dtype,
    qkv_layout,
    bias_type,
    attn_mask_type,
    softmax_type,
    dropout,
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
    """Return the Python cuDNN SDPA backend value for an attention configuration."""

    # Import lazily to avoid a circular import through dot_product_attention.utils.
    from .cudnn_attention import FusedAttnBackend

    q_dtype = DType.cast(q_dtype)
    kv_dtype = DType.cast(kv_dtype)
    if q_dtype != kv_dtype:
        raise ValueError("Q and KV must have the same data type")

    major, minor = get_device_compute_capability()
    sm_arch = major * 10 + minor
    cudnn_version_tuple = get_cudnn_version()
    layout = parse_attention_layout(qkv_layout)

    fp8_dtype = q_dtype in (DType.kFloat8E4M3, DType.kFloat8E5M2)
    if fp8_dtype:
        support = check_fp8_fused_attention_support(
            FusedAttentionConfig(
                is_training=bool(is_training),
                q_dtype=_dtype_name(q_dtype),
                kv_dtype=_dtype_name(kv_dtype),
                layout=layout,
                bias_type=bias_type,
                mask_type=attn_mask_type,
                softmax_type=softmax_type,
                dropout=float(dropout),
                num_attn_heads=int(num_attn_heads),
                num_gqa_groups=int(num_gqa_groups),
                max_seqlen_q=int(max_seqlen_q),
                max_seqlen_kv=int(max_seqlen_kv),
                head_dim_qk=int(head_dim_qk),
                head_dim_v=int(head_dim_v),
                window_size=(int(window_size_left), int(window_size_right)),
                return_max_logit=bool(return_max_logit),
                cuda_graph=bool(cuda_graph),
                deterministic=bool(deterministic),
                cudnn_version=cudnn_version_tuple,
                sm_arch=sm_arch,
            )
        )
        if support.supported:
            return FusedAttnBackend.FP8

    support = check_f16_fused_attention_support(
        FusedAttentionConfig(
            is_training=bool(is_training),
            q_dtype=_dtype_name(q_dtype),
            kv_dtype=_dtype_name(kv_dtype),
            layout=layout,
            bias_type=bias_type,
            mask_type=attn_mask_type,
            softmax_type=softmax_type,
            dropout=float(dropout),
            num_attn_heads=int(num_attn_heads),
            num_gqa_groups=int(num_gqa_groups),
            max_seqlen_q=int(max_seqlen_q),
            max_seqlen_kv=int(max_seqlen_kv),
            head_dim_qk=int(head_dim_qk),
            head_dim_v=int(head_dim_v),
            window_size=(int(window_size_left), int(window_size_right)),
            return_max_logit=bool(return_max_logit),
            cuda_graph=bool(cuda_graph),
            deterministic=bool(deterministic),
            cudnn_version=cudnn_version_tuple,
            sm_arch=sm_arch,
        )
    )
    if support.warning is not None:
        warnings.warn(support.warning)
    if support.supported:
        return FusedAttnBackend.F16_arbitrary_seqlen
    return FusedAttnBackend.No_Backend
