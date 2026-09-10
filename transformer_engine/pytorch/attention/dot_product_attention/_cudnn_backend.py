# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""cuDNN SDPA capability selection for the PyTorch frontend.

This is intentionally kept in Python alongside the Python cuDNN graph builder.
The conditions preserve the compatibility policy of the removed TE-common
attention backend selector.
"""

from __future__ import annotations

import warnings

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


def _requires_64bit_ragged_offset(
    qkv_format: str,
    layout_group: str,
    num_attn_heads: int,
    num_gqa_groups: int,
    max_seqlen_q: int,
    max_seqlen_kv: int,
    head_dim_qk: int,
    head_dim_v: int,
) -> bool:
    if qkv_format != "thd":
        return False
    if layout_group in ("3hd", "h3d"):
        q = k = v = 3 * num_attn_heads * head_dim_qk * max_seqlen_q
    elif layout_group in ("hd_2hd", "hd_h2d"):
        q = num_attn_heads * head_dim_qk * max_seqlen_q
        k = v = 2 * num_gqa_groups * head_dim_qk * max_seqlen_kv
    else:
        q = num_attn_heads * head_dim_qk * max_seqlen_q
        k = num_gqa_groups * head_dim_qk * max_seqlen_kv
        v = num_gqa_groups * head_dim_v * max_seqlen_kv
    output = num_attn_heads * head_dim_qk * max_seqlen_q
    return max(q, k, v, output) > 2**31 - 1


def _version_number() -> int:
    major, minor, patch = get_cudnn_version()
    return major * 10000 + minor * 100 + patch


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

    # Import lazily to keep this module independent of graph construction and
    # avoid a circular import through dot_product_attention.utils.
    from .cudnn_attention import FusedAttnBackend

    q_dtype = DType.cast(q_dtype)
    kv_dtype = DType.cast(kv_dtype)
    if q_dtype != kv_dtype:
        raise ValueError("Q and KV must have the same data type")

    major, minor = get_device_compute_capability()
    sm_arch = major * 10 + minor
    cudnn_version = _version_number()
    qkv_format, q_format, kv_format, layout_group = _layout_info(qkv_layout)
    is_thd_layout = q_format == "thd" or kv_format == "thd"
    requires_i64 = _requires_64bit_ragged_offset(
        qkv_format,
        layout_group,
        num_attn_heads,
        num_gqa_groups,
        max_seqlen_q,
        max_seqlen_kv,
        head_dim_qk,
        head_dim_v,
    )
    supported_ragged_offset_size = not requires_i64 or cudnn_version >= 90500

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
        and qkv_format in ("bshd", "sbhd")
        and softmax_type == "vanilla"
    ) or (cudnn_version >= 92100 and qkv_format in ("bshd", "sbhd", "bhsd"))
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

    if q_dtype not in (DType.kFloat16, DType.kBFloat16):
        return FusedAttnBackend.No_Backend

    arch_supported = (
        (cudnn_version < 8903 and sm_arch in (80, 90))
        or (cudnn_version >= 8903 and 80 <= sm_arch < 100)
        or (cudnn_version >= 90700 and sm_arch >= 100)
    )
    seq_supported = cudnn_version >= 90000 or (
        max_seqlen_q % 64 == 0 and max_seqlen_kv % 64 == 0
    )
    heads_supported = cudnn_version >= 8907 or num_attn_heads == num_gqa_groups

    dim_supported = (
        head_dim_qk % 8 == 0
        and head_dim_v % 8 == 0
        and (
            (head_dim_qk <= 128 and head_dim_v <= 128)
            or (
                head_dim_qk <= 256
                and head_dim_v <= 256
                and (
                    (not is_training and sm_arch == 90 and cudnn_version >= 90100)
                    or (is_training and sm_arch == 90 and cudnn_version >= 90500)
                )
            )
            or (
                not is_training
                and sm_arch >= 100
                and cudnn_version >= 90900
                and max_seqlen_q > 1
                and layout_group != "paged_separate"
            )
            or (
                not is_training
                and cudnn_version >= 91002
                and (
                    layout_group == "paged_separate"
                    or max_seqlen_q > 1
                    or (
                        max_seqlen_q == 1
                        and attn_mask_type not in ("causal", "padding_causal")
                    )
                )
            )
            or (
                head_dim_qk == 192
                and head_dim_v == 128
                and is_training
                and sm_arch >= 100
                and cudnn_version >= 91100
            )
            or (
                head_dim_qk == 256
                and head_dim_v == 256
                and is_training
                and 100 <= sm_arch < 110
                and cudnn_version >= (92500 if is_thd_layout else 92300)
                and layout_group != "paged_separate"
                and bias_type == "no_bias"
                and dropout == 0.0
                and softmax_type == "vanilla"
                and (
                    (window_size_left == -1 and window_size_right == -1)
                    or (
                        attn_mask_type
                        in (
                            "causal",
                            "padding_causal",
                            "causal_bottom_right",
                            "padding_causal_bottom_right",
                        )
                        and window_size_right in (-1, 0)
                    )
                )
            )
        )
    )
    dim_supported = dim_supported and not (
        cudnn_version >= 91100
        and is_training
        and sm_arch == 90
        and head_dim_qk >= 128
        and head_dim_v >= 128
        and (head_dim_qk, head_dim_v) != (192, 128)
        and head_dim_qk != head_dim_v
    )

    bias_supported = (
        (cudnn_version < 8906 and bias_type == "no_bias")
        or (
            cudnn_version >= 8906
            and (
                bias_type == "no_bias"
                or (
                    bias_type == "alibi"
                    and attn_mask_type
                    not in (
                        "no_mask",
                        "padding",
                        "padding_causal",
                        "padding_causal_bottom_right",
                    )
                    and sm_arch >= 90
                )
                or (bias_type == "post_scale_bias" and sm_arch >= 90)
            )
        )
        or (cudnn_version >= 90000 and bias_type == "post_scale_bias" and sm_arch >= 80)
    )

    standard_format = qkv_format in ("sbhd", "bshd")
    mask_supported = (
        (cudnn_version < 8906 and attn_mask_type == "causal")
        or (
            cudnn_version >= 8906
            and standard_format
            and attn_mask_type in ("causal", "padding", "padding_causal", "no_mask")
        )
        or (
            cudnn_version >= 90100
            and qkv_format == "thd"
            and attn_mask_type in ("padding", "padding_causal")
        )
        or (
            cudnn_version >= 90300
            and standard_format
            and attn_mask_type == "causal_bottom_right"
            and max_seqlen_q % 64 == 0
            and max_seqlen_kv % 64 == 0
            and max_seqlen_q <= max_seqlen_kv
            and bias_type == "no_bias"
            and dropout == 0.0
        )
        or (
            cudnn_version >= 90500
            and layout_group == "paged_separate"
            and (
                attn_mask_type in ("padding", "padding_causal")
                or (
                    attn_mask_type == "padding_causal_bottom_right"
                    and max_seqlen_q % 64 == 0
                    and max_seqlen_kv % 64 == 0
                    and max_seqlen_q <= max_seqlen_kv
                )
            )
            and bias_type == "no_bias"
            and dropout == 0.0
        )
        or (
            cudnn_version >= 90600
            and attn_mask_type == "padding_causal_bottom_right"
            and max_seqlen_q % 64 == 0
            and max_seqlen_kv % 64 == 0
            and max_seqlen_q <= max_seqlen_kv
            and bias_type == "no_bias"
            and dropout == 0.0
        )
        or (
            cudnn_version >= 90700
            and (
                attn_mask_type in ("no_mask", "causal")
                or (
                    attn_mask_type
                    in ("padding", "padding_causal", "padding_causal_bottom_right")
                    and bias_type == "no_bias"
                    and dropout == 0.0
                )
                or (
                    attn_mask_type
                    in ("causal_bottom_right", "padding_causal_bottom_right")
                    and max_seqlen_q <= max_seqlen_kv
                )
            )
        )
    )

    bias_mask_supported = not (
        cudnn_version >= 8906
        and attn_mask_type in ("padding", "padding_causal")
        and bias_type == "post_scale_bias"
    )
    format_supported = (
        qkv_format in ("sbhd", "bshd", "bhsd")
        or (
            qkv_format == "thd"
            and sm_arch >= 90
            and (
                (cudnn_version >= 90100 and num_attn_heads == num_gqa_groups)
                or cudnn_version >= 90600
            )
        )
        or (
            q_format in ("sbhd", "bshd", "bhsd", "thd")
            and kv_format in ("sbhd", "bshd", "bhsd", "thd")
            and (q_format != "thd" or sm_arch >= 90)
            and (kv_format != "thd" or sm_arch >= 90)
            and cudnn_version >= 90700
        )
    )

    sliding_window_supported = (
        (
            cudnn_version < 90200
            and window_size_left == -1
            and window_size_right in (-1, 0)
        )
        or (
            cudnn_version >= 90200
            and (
                (
                    window_size_left == -1
                    and window_size_right == -1
                    and attn_mask_type == "no_mask"
                )
                or (
                    window_size_left >= -1
                    and window_size_right == 0
                    and (
                        attn_mask_type in ("no_mask", "causal")
                        or (
                            attn_mask_type == "causal_bottom_right"
                            and max_seqlen_q == max_seqlen_kv
                        )
                    )
                    and max_seqlen_q <= max_seqlen_kv
                    and dropout == 0.0
                    and bias_type == "no_bias"
                    and standard_format
                )
            )
        )
        or (
            cudnn_version >= 90600
            and (
                (window_size_left == -1 and window_size_right in (-1, 0))
                or (
                    window_size_left >= -1
                    and window_size_right >= -1
                    and (
                        (
                            attn_mask_type == "causal_bottom_right"
                            and (
                                sm_arch < 100
                                or (
                                    sm_arch >= 100
                                    and (
                                        (
                                            max_seqlen_q == max_seqlen_kv
                                            and cudnn_version <= 90700
                                        )
                                        or cudnn_version > 90700
                                    )
                                )
                            )
                        )
                        or attn_mask_type in ("no_mask", "padding", "padding_causal")
                        or (
                            attn_mask_type == "padding_causal_bottom_right"
                            and (
                                sm_arch < 100
                                or (
                                    sm_arch >= 100
                                    and (
                                        (
                                            max_seqlen_q == max_seqlen_kv
                                            and cudnn_version <= 90700
                                        )
                                        or cudnn_version > 90700
                                    )
                                )
                            )
                        )
                    )
                    and max_seqlen_q <= max_seqlen_kv
                    and bias_type == "no_bias"
                    and dropout == 0.0
                )
            )
        )
    )

    softmax_supported = cudnn_version >= 91301 or softmax_type == "vanilla"
    max_supported = not return_max_logit or cudnn_version >= 92100
    deterministic_supported = sm_arch < 100 or (
        not is_training
        or (
            is_training
            and not deterministic
            and (dropout == 0.0 or bias_type == "no_bias")
        )
        or (
            is_training
            and deterministic
            and cudnn_version >= 91801
            and dropout == 0.0
            and bias_type == "no_bias"
        )
    )

    supported = all(
        (
            arch_supported,
            seq_supported,
            heads_supported,
            dim_supported,
            bias_supported,
            mask_supported,
            bias_mask_supported,
            format_supported,
            sliding_window_supported,
            supported_ragged_offset_size,
            cudnn_version not in (91000, 91001),
            softmax_supported,
            max_supported,
            deterministic_supported,
        )
    )
    backend = (
        FusedAttnBackend.F16_arbitrary_seqlen
        if supported
        else FusedAttnBackend.No_Backend
    )

    if cudnn_version < 8900 and backend == FusedAttnBackend.F16_arbitrary_seqlen:
        backend = FusedAttnBackend.No_Backend
        warnings.warn("FP16/BF16 fused attention requires cuDNN 8.9.0 or newer")
    if (
        cudnn_version == 91400
        and max_seqlen_kv > 1024
        and window_size_left != -1
        and attn_mask_type not in ("causal", "causal_bottom_right")
    ):
        backend = FusedAttnBackend.No_Backend
        warnings.warn(
            "This non-causal sliding-window configuration requires cuDNN > 9.14.0"
        )
    if (
        cudnn_version <= 91500
        and is_training
        and standard_format
        and max_seqlen_kv % 128 != 0
        and cuda_graph
        and attn_mask_type
        not in ("padding", "padding_causal", "padding_causal_bottom_right")
    ):
        backend = FusedAttnBackend.No_Backend
        warnings.warn(
            "This backward CUDA-graph configuration requires cuDNN 9.15.1 or newer"
        )
    if backend == FusedAttnBackend.F16_arbitrary_seqlen and sm_arch == 120:
        if cudnn_version < 91801:
            backend = FusedAttnBackend.No_Backend
            warnings.warn("SM120 fused attention requires cuDNN 9.18.1 or newer")
        elif deterministic and is_training:
            backend = FusedAttnBackend.No_Backend
            warnings.warn(
                "Deterministic fused-attention backward is not supported on SM120"
            )
        elif qkv_layout in ("t3hd", "th3d"):
            backend = FusedAttnBackend.No_Backend
            warnings.warn("T3HD/TH3D fused attention is not supported on SM120")
    return backend
