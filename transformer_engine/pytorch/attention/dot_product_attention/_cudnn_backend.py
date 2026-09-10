# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""cuDNN SDPA capability selection for the PyTorch frontend."""

from __future__ import annotations

import warnings

from transformer_engine.common.attention.cudnn import (
    AttentionLayout,
    FusedAttentionConfig,
    check_f16_fused_attention_support,
    encode_cudnn_version,
    requires_64bit_ragged_offset,
)
from transformer_engine.pytorch.constants import DType
from transformer_engine.pytorch.utils import (
    get_cudnn_version,
    get_device_compute_capability,
)


def _layout_info(qkv_layout: str):
    paged = qkv_layout.startswith("paged_kv_")
    layout = qkv_layout.removeprefix("paged_kv_")
    components = layout.split("_")

    def tensor_format(component: str) -> str:
        return "".join(char for char in component if char.isalpha())

    q_format = tensor_format(components[0])
    kv_format = tensor_format(components[-1]) if len(components) > 1 else q_format
    qkv_format = q_format if q_format == kv_format else f"{q_format}_2{kv_format}"
    if paged:
        layout_group = "paged_separate"
    elif len(components) == 1 and "3" in components[0]:
        layout_group = "h3d" if "h3d" in components[0] else "3hd"
    elif len(components) == 2:
        layout_group = "hd_h2d" if "h2d" in components[1] else "hd_2hd"
    elif q_format == "bhsd":
        layout_group = "sd_sd_sd"
    else:
        layout_group = "separate"
    return qkv_format, q_format, kv_format, layout_group


def _normalized_layout(qkv_layout: str) -> AttentionLayout:
    qkv_format, q_format, kv_format, layout_group = _layout_info(qkv_layout)
    return AttentionLayout(
        qkv_format=qkv_format,
        q_format=q_format,
        kv_format=kv_format,
        layout_group=layout_group,
        is_qkvpacked=layout_group in ("3hd", "h3d"),
    )


def _dtype_name(dtype: DType) -> str:
    if dtype == DType.kFloat16:
        return "float16"
    if dtype == DType.kBFloat16:
        return "bfloat16"
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
    cudnn_version = encode_cudnn_version(cudnn_version_tuple)
    layout = _normalized_layout(qkv_layout)
    requires_i64 = requires_64bit_ragged_offset(
        layout,
        num_attn_heads,
        num_gqa_groups,
        max_seqlen_q,
        max_seqlen_kv,
        head_dim_qk,
        head_dim_v,
    )

    # FP8 and MXFP8 remain PyTorch-specific. The shared policy below covers the
    # F16/BF16 subset implemented by both framework frontends.
    fp8_dtype = q_dtype in (DType.kFloat8E4M3, DType.kFloat8E5M2)
    fp8_shape_mask = (
        (
            cudnn_version >= 90201
            and sm_arch < 100
            and max_seqlen_q % 128 == 0
            and max_seqlen_kv % 128 == 0
            and head_dim_qk == 128
            and head_dim_v == 128
            and attn_mask_type in ("causal", "no_mask")
        )
        or (
            cudnn_version >= 90700
            and (
                (
                    sm_arch < 100
                    and not is_training
                    and head_dim_qk <= 256
                    and head_dim_v <= 256
                )
                or (
                    sm_arch < 100
                    and is_training
                    and head_dim_qk == 128
                    and head_dim_v == 128
                )
                or (sm_arch >= 100 and head_dim_qk <= 128 and head_dim_v <= 128)
            )
            and head_dim_qk % 16 == 0
            and head_dim_v % 16 == 0
            and attn_mask_type in ("no_mask", "causal", "padding", "padding_causal")
        )
        or (
            cudnn_version >= 92100
            and sm_arch >= 100
            and head_dim_qk <= 192
            and head_dim_v <= 128
            and head_dim_qk % 16 == 0
            and head_dim_v % 16 == 0
            and attn_mask_type in ("no_mask", "causal", "causal_bottom_right")
        )
    )
    fp8_format_softmax = (
        cudnn_version < 92100
        and layout.qkv_format in ("bshd", "sbhd")
        and softmax_type == "vanilla"
    ) or (cudnn_version >= 92100 and layout.qkv_format in ("bshd", "sbhd", "bhsd"))
    if (
        fp8_dtype
        and sm_arch >= 90
        and bias_type == "no_bias"
        and fp8_shape_mask
        and fp8_format_softmax
        and not requires_i64
        and cudnn_version != 91000
        and not return_max_logit
    ):
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
            allow_alibi=True,
        )
    )
    if support.warning is not None:
        warnings.warn(support.warning)
    if support.supported:
        return FusedAttnBackend.F16_arbitrary_seqlen
    return FusedAttnBackend.No_Backend
