# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Framework-independent cuDNN attention policy and graph-shape helpers."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AttentionLayout:
    """Normalized layout properties used by the cuDNN support policy."""

    qkv_format: str
    q_format: str
    kv_format: str
    layout_group: str
    is_qkvpacked: bool = False

    @property
    def is_thd(self) -> bool:
        """Return whether either Q or KV uses packed-token storage."""

        return self.q_format == "thd" or self.kv_format == "thd"


@dataclass(frozen=True)
class FusedAttentionConfig:
    """Normalized inputs to the shared FP16/BF16 cuDNN support policy."""

    is_training: bool
    q_dtype: str
    kv_dtype: str
    layout: AttentionLayout
    bias_type: str
    mask_type: str
    softmax_type: str
    dropout: float
    num_attn_heads: int
    num_gqa_groups: int
    max_seqlen_q: int
    max_seqlen_kv: int
    head_dim_qk: int
    head_dim_v: int
    window_size: tuple[int, int]
    return_max_logit: bool
    cuda_graph: bool
    deterministic: bool
    cudnn_version: tuple[int, int, int]
    sm_arch: int


@dataclass(frozen=True)
class FusedAttentionSupport:
    """Result of checking a normalized attention configuration."""

    supported: bool
    reason: str = ""
    warning: str | None = None


@dataclass(frozen=True)
class AttentionMask:
    """Framework-neutral interpretation of mask and sliding-window options."""

    causal: bool
    bottom_right: bool
    padding: bool
    bottom_right_diagonal: bool
    window_left: int
    window_right: int


def parse_attention_layout(qkv_layout: str) -> AttentionLayout:
    """Normalize a TE QKV layout string for framework-independent policy checks."""

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
    return AttentionLayout(
        qkv_format=qkv_format,
        q_format=q_format,
        kv_format=kv_format,
        layout_group=layout_group,
        is_qkvpacked=layout_group in ("3hd", "h3d"),
    )


def encode_cudnn_version(version: tuple[int, int, int]) -> int:
    """Encode a cuDNN backend version using its native integer convention."""

    major, minor, patch = (int(part) for part in version)
    magnitude = 1000 if major < 9 else 10000
    return major * magnitude + minor * 100 + patch


def round_up(value: int, multiple: int) -> int:
    """Round ``value`` up to a positive multiple."""

    return (int(value) + int(multiple) - 1) // int(multiple) * int(multiple)


def ragged_token_bucket(tokens: int) -> int:
    """Return the cuDNN graph bucket for a packed-token extent."""

    tokens = int(tokens)
    if tokens <= 1024:
        return 1024
    if tokens <= 32768:
        return 1 << (tokens - 1).bit_length()
    return round_up(tokens, 32768)


def ragged_batch_bucket(batch: int) -> int:
    """Return the cuDNN graph bucket for a ragged batch extent."""

    batch = int(batch)
    if batch <= 32:
        return 32
    if batch <= 512:
        return 1 << (batch - 1).bit_length()
    return round_up(batch, 512)


def normalize_attention_mask(
    *,
    causal: bool,
    bottom_right: bool,
    padding: bool,
    bottom_right_diagonal: bool,
    window_size: tuple[int, int],
    max_seqlen_q: int,
    max_seqlen_kv: int,
) -> AttentionMask:
    """Normalize equivalent causal and bottom-right mask configurations."""

    if bottom_right and max_seqlen_q == max_seqlen_kv and not padding:
        causal = True
        bottom_right = False
        bottom_right_diagonal = False
    return AttentionMask(
        causal=bool(causal),
        bottom_right=bool(bottom_right),
        padding=bool(padding),
        bottom_right_diagonal=bool(bottom_right_diagonal),
        window_left=int(window_size[0]),
        window_right=int(window_size[1]),
    )


def cudnn_mask_options(
    *,
    causal: bool,
    bottom_right: bool,
    padding: bool,
    bottom_right_diagonal: bool,
    window_size: tuple[int, int],
    max_seqlen_q: int,
    max_seqlen_kv: int,
    cudnn_version: tuple[int, int, int],
) -> dict[str, bool | int | str]:
    """Return canonical cuDNN SDPA mask options using framework-neutral values."""

    mask = normalize_attention_mask(
        causal=causal,
        bottom_right=bottom_right,
        padding=padding,
        bottom_right_diagonal=bottom_right_diagonal,
        window_size=window_size,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_kv=max_seqlen_kv,
    )
    version = encode_cudnn_version(cudnn_version)
    options: dict[str, bool | int | str] = {
        "diagonal_alignment": (
            "bottom_right" if mask.bottom_right_diagonal else "top_left"
        ),
        "is_padding": mask.padding,
    }
    if version < 90600:
        options["use_causal_mask"] = mask.causal
        options["use_causal_mask_bottom_right"] = mask.bottom_right
    if version >= 90200 and mask.window_left != -1:
        options["diagonal_band_left_bound"] = mask.window_left + 1
    if version >= 90600:
        if mask.window_right != -1:
            options["diagonal_band_right_bound"] = mask.window_right
        elif mask.causal or mask.bottom_right:
            options["diagonal_band_right_bound"] = 0
    return options


def requires_64bit_ragged_offset(
    layout: AttentionLayout,
    num_attn_heads: int,
    num_gqa_groups: int,
    max_seqlen_q: int,
    max_seqlen_kv: int,
    head_dim_qk: int,
    head_dim_v: int,
) -> bool:
    """Return whether legacy THD element offsets can overflow signed int32."""

    if layout.qkv_format != "thd":
        return False
    if layout.layout_group in ("3hd", "h3d", "qkv_packed"):
        q = k = v = 3 * num_attn_heads * head_dim_qk * max_seqlen_q
    elif layout.layout_group in ("hd_2hd", "hd_h2d", "kv_packed"):
        q = num_attn_heads * head_dim_qk * max_seqlen_q
        k = v = 2 * num_gqa_groups * head_dim_qk * max_seqlen_kv
    else:
        q = num_attn_heads * head_dim_qk * max_seqlen_q
        k = num_gqa_groups * head_dim_qk * max_seqlen_kv
        v = num_gqa_groups * head_dim_v * max_seqlen_kv
    output = num_attn_heads * head_dim_qk * max_seqlen_q
    return max(q, k, v, output) > 2**31 - 1


def _unsupported(reason: str, warning: str | None = None) -> FusedAttentionSupport:
    return FusedAttentionSupport(False, reason, warning)


def check_f16_fused_attention_support(
    config: FusedAttentionConfig,
) -> FusedAttentionSupport:
    """Check the shared FP16/BF16 cuDNN fused-attention compatibility policy."""

    if config.q_dtype != config.kv_dtype:
        return _unsupported("Q and KV must have the same data type")
    if config.q_dtype not in ("float16", "bfloat16"):
        return _unsupported("only FP16 and BF16 are supported")

    version = encode_cudnn_version(config.cudnn_version)
    arch = int(config.sm_arch)
    layout = config.layout
    is_thd = layout.is_thd
    is_training = bool(config.is_training)
    sq = int(config.max_seqlen_q)
    skv = int(config.max_seqlen_kv)
    h = int(config.num_attn_heads)
    hg = int(config.num_gqa_groups)
    dqk = int(config.head_dim_qk)
    dv = int(config.head_dim_v)
    dropout = float(config.dropout)
    bias = config.bias_type
    mask = config.mask_type
    softmax = config.softmax_type
    left, right = config.window_size

    if version < 8900:
        return _unsupported(
            "cuDNN is older than 8.9.0",
            "FP16/BF16 fused attention requires cuDNN 8.9.0 or newer",
        )

    architecture_ok = (
        (version < 8903 and arch in (80, 90))
        or (version >= 8903 and 80 <= arch < 100)
        or (version >= 90700 and arch >= 100)
    )
    if not architecture_ok:
        return _unsupported("device architecture is not supported")
    if version < 90000 and (sq % 64 or skv % 64):
        return _unsupported("sequence lengths must be multiples of 64")
    if version < 8907 and h != hg:
        return _unsupported("GQA requires cuDNN 8.9.7 or newer")
    if dqk % 8 or dv % 8:
        return _unsupported("head dimensions must be multiples of 8")

    standard_dim = dqk <= 128 and dv <= 128
    hopper_large_dim = (
        dqk <= 256
        and dv <= 256
        and (
            (not is_training and arch == 90 and version >= 90100)
            or (is_training and arch == 90 and version >= 90500)
        )
    )
    blackwell_fwd_any_dim = (
        not is_training
        and arch >= 100
        and version >= 90900
        and sq > 1
        and layout.layout_group != "paged_separate"
    )
    generic_fwd_any_dim = (
        not is_training
        and version >= 91002
        and (
            layout.layout_group == "paged_separate"
            or sq > 1
            or (sq == 1 and mask not in ("causal", "padding_causal"))
        )
    )
    blackwell_mla_bwd = (
        dqk == 192 and dv == 128 and is_training and arch >= 100 and version >= 91100
    )
    blackwell_d256_bwd = (
        dqk == 256
        and dv == 256
        and is_training
        and 100 <= arch < 110
        and version >= (92500 if is_thd else 92300)
        and layout.layout_group != "paged_separate"
        and bias == "no_bias"
        and dropout == 0.0
        and softmax == "vanilla"
        and (
            (left == -1 and right == -1)
            or (
                mask
                in (
                    "causal",
                    "padding_causal",
                    "causal_bottom_right",
                    "padding_causal_bottom_right",
                )
                and right in (-1, 0)
            )
        )
    )
    if not (
        standard_dim
        or hopper_large_dim
        or blackwell_fwd_any_dim
        or generic_fwd_any_dim
        or blackwell_mla_bwd
        or blackwell_d256_bwd
    ):
        return _unsupported("head dimensions are not supported")
    if (
        version >= 91100
        and is_training
        and arch == 90
        and dqk >= 128
        and dv >= 128
        and (dqk, dv) != (192, 128)
        and dqk != dv
    ):
        return _unsupported(
            "this Hopper backward head-dimension combination is unsupported"
        )

    alibi_supported = (
        bias == "alibi"
        and version >= 8906
        and arch >= 90
        and mask
        not in (
            "no_mask",
            "padding",
            "padding_causal",
            "padding_causal_bottom_right",
        )
    )
    post_scale_bias_supported = bias == "post_scale_bias" and (
        (version >= 8906 and arch >= 90) or (version >= 90000 and arch >= 80)
    )
    if bias != "no_bias" and not alibi_supported and not post_scale_bias_supported:
        return _unsupported("attention bias is not supported")

    standard_format = layout.qkv_format in ("sbhd", "bshd")
    basic_masks = mask in ("no_mask", "causal", "padding", "padding_causal")
    mask_ok = version < 8906 and mask == "causal"
    if version >= 8906 and standard_format and basic_masks:
        mask_ok = True
    if (
        version >= 90100
        and layout.qkv_format == "thd"
        and mask
        in (
            "padding",
            "padding_causal",
        )
    ):
        mask_ok = True
    if (
        version >= 90300
        and standard_format
        and mask == "causal_bottom_right"
        and sq % 64 == 0
        and skv % 64 == 0
        and sq <= skv
        and bias == "no_bias"
        and dropout == 0.0
    ):
        mask_ok = True
    if (
        version >= 90500
        and layout.layout_group == "paged_separate"
        and (
            mask in ("padding", "padding_causal")
            or (
                mask == "padding_causal_bottom_right"
                and sq % 64 == 0
                and skv % 64 == 0
                and sq <= skv
            )
        )
        and bias == "no_bias"
        and dropout == 0.0
    ):
        mask_ok = True
    if (
        version >= 90600
        and mask == "padding_causal_bottom_right"
        and sq % 64 == 0
        and skv % 64 == 0
        and sq <= skv
        and bias == "no_bias"
        and dropout == 0.0
    ):
        mask_ok = True
    if version >= 90700:
        modern_mask_ok = (
            mask in ("no_mask", "causal")
            or (
                mask in ("padding", "padding_causal", "padding_causal_bottom_right")
                and bias == "no_bias"
                and dropout == 0.0
            )
            or (
                mask in ("causal_bottom_right", "padding_causal_bottom_right")
                and sq <= skv
            )
        )
        mask_ok = mask_ok or modern_mask_ok
    if not mask_ok:
        return _unsupported("attention mask is not supported")
    if mask in ("padding", "padding_causal") and bias == "post_scale_bias":
        return _unsupported("post-scale bias cannot be combined with this padding mask")

    format_ok = (
        layout.qkv_format in ("sbhd", "bshd", "bhsd")
        or (
            layout.qkv_format == "thd"
            and arch >= 90
            and ((version >= 90100 and h == hg) or version >= 90600)
        )
        or (
            layout.q_format in ("sbhd", "bshd", "bhsd", "thd")
            and layout.kv_format in ("sbhd", "bshd", "bhsd", "thd")
            and (layout.q_format != "thd" or arch >= 90)
            and (layout.kv_format != "thd" or arch >= 90)
            and version >= 90700
        )
    )
    if not format_ok:
        return _unsupported("QKV format is not supported")

    pre_902_window = version < 90200 and left == -1 and right in (-1, 0)
    v902_window = version >= 90200 and (
        (left == -1 and right == -1 and mask == "no_mask")
        or (
            left >= -1
            and right == 0
            and (
                mask in ("no_mask", "causal")
                or (mask == "causal_bottom_right" and sq == skv)
            )
            and sq <= skv
            and dropout == 0.0
            and bias == "no_bias"
            and standard_format
        )
    )
    bottom_right_swa_supported = (
        mask not in ("causal_bottom_right", "padding_causal_bottom_right")
        or arch < 100
        or sq == skv
        or version > 90700
    )
    v906_window = version >= 90600 and (
        (left == -1 and right in (-1, 0))
        or (
            left >= -1
            and right >= -1
            and (
                mask
                in (
                    "no_mask",
                    "padding",
                    "padding_causal",
                    "causal_bottom_right",
                    "padding_causal_bottom_right",
                )
                or mask == "causal"
            )
        )
        and sq <= skv
        and bias == "no_bias"
        and dropout == 0.0
        and bottom_right_swa_supported
    )
    window_ok = pre_902_window or v902_window or v906_window
    if not window_ok:
        return _unsupported("sliding-window configuration is not supported")

    requires_i64 = requires_64bit_ragged_offset(
        layout,
        h,
        hg,
        sq,
        skv,
        dqk,
        dv,
    )
    if requires_i64 and version < 90500:
        return _unsupported("ragged offsets require int64 support")
    if version == 91000:
        return _unsupported("cuDNN 9.10.0 has known SDPA issues")
    if version < 91301 and softmax != "vanilla":
        return _unsupported("this softmax type requires cuDNN 9.13.1 or newer")
    if config.return_max_logit and version < 92100:
        return _unsupported("returning max logits requires cuDNN 9.21 or newer")
    if arch >= 100 and is_training:
        if config.deterministic:
            if version < 91801 or dropout != 0.0 or bias != "no_bias":
                return _unsupported("deterministic Blackwell backward is not supported")
        elif dropout != 0.0 and bias != "no_bias":
            return _unsupported("Blackwell backward does not support dropout with bias")

    if (
        version == 91400
        and skv > 1024
        and left != -1
        and mask not in ("causal", "causal_bottom_right")
    ):
        return _unsupported(
            "cuDNN 9.14.0 does not support this non-causal sliding window",
            "This non-causal sliding-window configuration requires cuDNN > 9.14.0",
        )
    if (
        version <= 91500
        and is_training
        and standard_format
        and skv % 128 != 0
        and config.cuda_graph
        and mask not in ("padding", "padding_causal", "padding_causal_bottom_right")
    ):
        return _unsupported(
            "this backward CUDA-graph configuration requires cuDNN 9.15.1",
            "This backward CUDA-graph configuration requires cuDNN 9.15.1 or newer",
        )
    if arch == 120:
        if version < 91801:
            return _unsupported(
                "SM120 requires cuDNN 9.18.1",
                "SM120 fused attention requires cuDNN 9.18.1 or newer",
            )
        if config.deterministic and is_training:
            return _unsupported(
                "deterministic backward is not supported on SM120",
                "Deterministic fused-attention backward is not supported on SM120",
            )
        if is_thd and layout.is_qkvpacked:
            return _unsupported(
                "QKV-packed THD attention is not supported on SM120",
                "T3HD/TH3D fused attention is not supported on SM120",
            )

    return FusedAttentionSupport(True)


def check_fp8_fused_attention_support(
    config: FusedAttentionConfig,
    *,
    scaling_mode: str | None = None,
) -> FusedAttentionSupport:
    """Check the shared cuDNN FP8/MXFP8 fused-attention compatibility policy."""

    if config.q_dtype != config.kv_dtype:
        return _unsupported("Q and KV must have the same data type")
    if config.q_dtype not in ("float8_e4m3", "float8_e5m2"):
        return _unsupported("only FP8 E4M3 and E5M2 are supported")

    version = encode_cudnn_version(config.cudnn_version)
    arch = int(config.sm_arch)
    layout = config.layout
    sq = int(config.max_seqlen_q)
    skv = int(config.max_seqlen_kv)
    dqk = int(config.head_dim_qk)
    dv = int(config.head_dim_v)
    mask = config.mask_type

    if scaling_mode not in (None, "delayed", "current", "mxfp8"):
        return _unsupported(f"unknown FP8 attention scaling mode {scaling_mode!r}")
    if arch < 90:
        return _unsupported("FP8 attention requires SM90 or newer")
    if arch >= 120:
        return _unsupported("FP8 attention is not supported on SM120 or newer")
    if config.is_training and config.deterministic and version < 91900:
        return _unsupported(
            "deterministic FP8 attention backward requires cuDNN 9.19 or newer"
        )
    if scaling_mode == "current":
        if arch < 100:
            return _unsupported("FP8 current-scaling attention requires SM100 or newer")
        if version < 91400:
            return _unsupported(
                "FP8 current-scaling attention requires cuDNN 9.14 or newer"
            )
    if scaling_mode == "mxfp8":
        if arch < 100:
            return _unsupported("MXFP8 attention requires SM100 or newer")
        if version < 92100:
            return _unsupported("MXFP8 attention requires cuDNN 9.21 or newer")
        if version in (92300, 92301):
            return _unsupported("cuDNN 9.23.0 and 9.23.1 have known MXFP8 SDPA issues")
    if config.bias_type != "no_bias":
        return _unsupported("FP8 attention does not support attention bias")
    if config.return_max_logit:
        return _unsupported("FP8 attention does not support returning max logits")
    if version == 91000:
        return _unsupported("cuDNN 9.10.0 has known SDPA issues")
    if (
        requires_64bit_ragged_offset(
            layout,
            config.num_attn_heads,
            config.num_gqa_groups,
            sq,
            skv,
            dqk,
            dv,
        )
        and version < 90500
    ):
        return _unsupported(
            "FP8 attention requires cuDNN 9.5 for 64-bit ragged offsets"
        )

    is_thd = layout.qkv_format == "thd"
    if is_thd:
        if version < 92300:
            return _unsupported("FP8 THD attention requires cuDNN 9.23 or newer")
        if mask not in ("padding", "padding_causal", "padding_causal_bottom_right"):
            return _unsupported("FP8 THD attention requires a padding mask")
        if config.is_training and arch < 100:
            return _unsupported("FP8 THD attention backward requires SM100 or newer")
        if config.is_training and config.softmax_type != "vanilla" and version < 92600:
            return _unsupported(
                "FP8 THD sink-token backward requires cuDNN 9.26 or newer"
            )
        if arch >= 100 and (dqk > 128 or dv > 128):
            return _unsupported(
                "FP8 THD attention supports head dimensions up to 128 on SM100+"
            )

    shape_mask_ok = (
        (
            version >= 90201
            and arch < 100
            and sq % 128 == 0
            and skv % 128 == 0
            and dqk == 128
            and dv == 128
            and mask in ("causal", "no_mask")
        )
        or (
            version >= 90700
            and (
                (arch < 100 and not config.is_training and dqk <= 256 and dv <= 256)
                or (arch < 100 and config.is_training and dqk == 128 and dv == 128)
                or (arch >= 100 and dqk <= 128 and dv <= 128)
            )
            and dqk % 16 == 0
            and dv % 16 == 0
            and (
                mask in ("no_mask", "causal", "padding", "padding_causal")
                or (arch >= 100 and mask == "padding_causal_bottom_right")
            )
        )
        or (
            version >= 92100
            and arch >= 100
            and dqk <= 192
            and dv <= 128
            and dqk % 16 == 0
            and dv % 16 == 0
            and mask in ("no_mask", "causal", "causal_bottom_right")
        )
    )
    if not shape_mask_ok:
        return _unsupported("FP8 attention shape or mask is not supported")

    format_softmax_ok = (
        (
            version < 92100
            and layout.qkv_format in ("bshd", "sbhd")
            and config.softmax_type == "vanilla"
        )
        or (version >= 92100 and layout.qkv_format in ("bshd", "sbhd", "bhsd"))
        or is_thd
    )
    if not format_softmax_ok:
        return _unsupported("FP8 attention layout or softmax type is not supported")
    return FusedAttentionSupport(True)
