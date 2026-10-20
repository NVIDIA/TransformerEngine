# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Framework-independent MXFP8 attention operand preparation rules."""

from .graph import round_up


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
    if backend_version < 92100:
        return "MXFP8 attention requires cuDNN 9.21.0 or later."
    if backend_version in (92300, 92301):
        return "MXFP8 attention is disabled for known cuDNN 9.23.0/9.23.1 defects."
    if qkv_format == "thd":
        return "MXFP8 attention does not support THD format."
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
