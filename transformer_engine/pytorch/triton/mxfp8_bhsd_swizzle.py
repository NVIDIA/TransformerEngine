# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""PyTorch wrapper for the fused BHSD permute + GEMM swizzle of MXFP8 attention scales."""

from typing import List

import torch
import triton

from transformer_engine.common.triton.mxfp8_bhsd_swizzle import (
    MAX_TENSORS,
    columnwise_bhsd_swizzle_kernel,
    rowwise_bhsd_swizzle_kernel,
)

# Heads per rowwise program and 4-row groups per columnwise program.
_ROWWISE_HEADS = 16
_COLUMNWISE_GROUPS = 8


def _align(x: int, a: int) -> int:
    return (x + a - 1) // a * a


def _dims(scale: torch.Tensor, src_format: str):
    """(B, S, H, D, stride_s, stride_b, stride_h) of a 4D SBHD/BSHD scale tensor."""
    if src_format == "sbhd":
        S, B, H, D = scale.shape
        ss, sb, sh, _ = scale.stride()
    else:
        B, S, H, D = scale.shape
        sb, ss, sh, _ = scale.stride()
    return B, S, H, D, ss, sb, sh


def can_fuse_transpose_swizzle(fp8_tensors: List, src_format: str) -> bool:
    """Whether :func:`fused_transpose_swizzle` supports these tensors.

    Requires at most three tensors with 4D uint8 scales that are contiguous in the last
    dimension, and sequence lengths that are multiples of 128.
    """
    if src_format not in ("bshd", "sbhd") or not 0 < len(fp8_tensors) <= MAX_TENSORS:
        return False
    for t in fp8_tensors:
        for scale in (t._rowwise_scale_inv, t._columnwise_scale_inv):
            if scale is None:
                continue
            if scale.dim() != 4 or scale.dtype != torch.uint8 or scale.stride(-1) != 1:
                return False
        if t._rowwise_scale_inv is not None:
            if _dims(t._rowwise_scale_inv, src_format)[1] % 128 != 0:
                return False
        if t._columnwise_scale_inv is not None:
            B, SC, H, *_ = _dims(t._columnwise_scale_inv, src_format)
            if (B * H * SC) % 4 != 0:
                return False
    return any(
        t._rowwise_scale_inv is not None or t._columnwise_scale_inv is not None for t in fp8_tensors
    )


def fused_transpose_swizzle(fp8_tensors: List, src_format: str) -> None:
    """Permute MXFP8 scales from ``src_format`` to BHSD and swizzle them for the GEMM.

    Byte-identical to ``multi_tensor_transpose_to_bhsd`` followed by
    ``multi_tensor_swizzle_scales_for_gemm_unchecked_`` (see ``mxfp8_transpose_swizzle``), with
    one kernel per direction instead of two. The scales of all tensors go to one buffer.
    Modifies the tensors in place; check :func:`can_fuse_transpose_swizzle` first.
    """
    rows, cols, total = [], [], 0
    for t in fp8_tensors:
        rs = t._rowwise_scale_inv
        if rs is not None:
            B, S, H, DR, ss, sb, sh = _dims(rs, src_format)
            KP = _align(DR, 4)
            rows.append((t, rs, total, B, S, H, DR, KP, ss, sb, sh))
            total += B * H * S * KP
        cs = t._columnwise_scale_inv
        if cs is not None:
            B, SC, H, D, ss, sb, sh = _dims(cs, src_format)
            NP = _align(D, 128)
            MC = B * H * SC
            cols.append((t, cs, total, SC, H, D, NP, MC, ss, sb, sh))
            total += MC * NP
    device = (rows or cols)[0][1].device
    buf = torch.empty(total, dtype=torch.uint8, device=device)

    # Unused tensor slots get zero programs.
    row_args, row_programs, max_kp = [], 0, 4
    for t, rs, off, B, S, H, DR, KP, ss, sb, sh in rows:
        out = buf[off : off + B * H * S * KP]
        programs = B * triton.cdiv(H, _ROWWISE_HEADS) * (S // 128)
        row_args.append((rs, out, H, S, DR, KP, ss, sb, sh, programs))
        row_programs, max_kp = max(row_programs, programs), max(max_kp, KP)
        t._rowwise_scale_inv = out.view(B * H * S, KP)
    if rows:
        row_args += [(buf, buf, 1, 128, 0, 4, 0, 0, 0, 0)] * (MAX_TENSORS - len(row_args))
        rowwise_bhsd_swizzle_kernel[(row_programs, len(rows))](
            *[a for args in row_args for a in args],
            HB=_ROWWISE_HEADS,
            BLOCK_KT=triton.next_power_of_2(max_kp // 4),
        )

    col_args, col_programs = [], 0
    for t, cs, off, SC, H, D, NP, MC, ss, sb, sh in cols:
        out = buf[off : off + MC * NP]
        n_tiles = NP // 128
        programs = n_tiles * triton.cdiv(MC // 4, _COLUMNWISE_GROUPS)
        col_args.append((cs, out, H, SC, D, MC, ss, sb, sh, n_tiles, programs))
        col_programs = max(col_programs, programs)
        t._columnwise_scale_inv = out.view(MC, NP)
    if cols:
        col_args += [(buf, buf, 1, 1, 0, 4, 0, 0, 0, 1, 0)] * (MAX_TENSORS - len(col_args))
        columnwise_bhsd_swizzle_kernel[(col_programs, len(cols))](
            *[a for args in col_args for a in args], MB=_COLUMNWISE_GROUPS
        )

    for t in fp8_tensors:
        t._with_gemm_swizzled_scales = True
