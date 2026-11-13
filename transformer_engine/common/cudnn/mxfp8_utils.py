# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Framework-independent MXFP8 attention operand preparation rules."""

from .graph import round_up

# Tokens that share a columnwise MXFP8 scale; MXFP8 causal own-block attention computes each
# query's own aligned block of this many keys separately.
OWN_BLOCK = 32


def quantization_layouts(backward=False):
    """Return (rowwise, columnwise) requirements for Q, K, and V.

    Rowwise quantization groups 32 values along the head dimension; columnwise
    quantization groups 32 sequence positions. The two representations must be
    quantized independently from the higher-precision operand.
    """
    return ((True, backward), (True, backward), (backward, not backward))


def support_error(sm_arch, backend_version, qkv_format):
    """Return shared hardware, backend-defect, and layout exclusions."""
    if sm_arch < 100:
        return "MXFP8 attention requires SM100 or later."
    if 120 <= sm_arch < 130:
        return "cuDNN MXFP8 attention is not supported on SM12x."
    if backend_version < 92100:
        return "MXFP8 attention requires cuDNN 9.21.0 or later."
    if backend_version in (92300, 92301):
        return "MXFP8 attention is disabled for known cuDNN 9.23.0/9.23.1 defects."
    if qkv_format == "thd":
        return "MXFP8 attention does not support THD format."
    return ""


def causal_own_block_error(
    sm_arch,
    qkv_layout,
    attn_mask_type,
    window_size,
    head_dim_qk,
    head_dim_v,
    seqlen_q,
    seqlen_kv,
    dropout,
    bias_type,
    softmax_type,
):
    """Return why causal MXFP8 attention split at the 32-token scale block cannot serve a call.

    The completed-block half runs on cuDNN Frontend's SM100 MXFP8 forward engine
    (``sdpa_mxfp8(causal_exclude_own_block=32)``); the own-block half and the backward
    operand decode are Triton kernels for D = 128 and BSHD/SBHD payloads.
    """
    if sm_arch not in (100, 103):
        return "MXFP8 causal own-block attention requires SM100 or SM103."
    if attn_mask_type != "causal" or tuple(window_size or (-1, 0)) not in ((-1, 0), (-1, -1)):
        return "MXFP8 causal own-block attention requires a top-left causal mask without window."
    if qkv_layout not in ("bshd_bshd_bshd", "sbhd_sbhd_sbhd"):
        return "MXFP8 causal own-block attention requires separate BSHD or SBHD Q, K and V."
    if head_dim_qk != 128 or head_dim_v != 128:
        return "MXFP8 causal own-block attention requires head dimension 128."
    if seqlen_q != seqlen_kv:
        return "MXFP8 causal own-block attention requires equal query and key sequence lengths."
    if seqlen_q % 128:
        return "MXFP8 causal own-block attention requires sequence lengths divisible by 128."
    if dropout:
        return "MXFP8 causal own-block attention does not support dropout."
    if bias_type != "no_bias" or softmax_type != "vanilla":
        return "MXFP8 causal own-block attention does not support bias or a softmax offset."
    return ""


def shape_error(seqlen_q, seqlen_kv, head_dim_qk, head_dim_v):
    """Return the alignment requirements of the existing preparation kernels."""
    if seqlen_q % 128 or seqlen_kv % 128 or head_dim_qk % 32 or head_dim_v % 32:
        return (
            "MXFP8 attention requires sequence lengths divisible by 128 and head "
            "dimensions divisible by 32. "
            f"Received {seqlen_q=}, {seqlen_kv=}, {head_dim_qk=}, {head_dim_v=}."
        )
    return ""


def scale_reordering(shape, columnwise=False):
    """Describe padding and the F8_128x4 permutation of logical BHSD scales.

    Return the padded BHSD shape, the tiled reshape, and its permutation. The
    frontend applies these operations to device arrays. Each batch/head matrix
    is reordered independently, including columnwise scales.
    """
    batch, heads, rows, cols = shape
    if columnwise:
        rows, cols = cols, rows
    rows, cols = round_up(rows, 128), round_up(cols, 4)
    padded_shape = (batch, heads, cols, rows) if columnwise else (batch, heads, rows, cols)
    tiled_shape = (batch, heads, rows // 128, 4, 32, cols // 4, 4)
    return padded_shape, tiled_shape, (0, 1, 2, 5, 4, 3, 6)
