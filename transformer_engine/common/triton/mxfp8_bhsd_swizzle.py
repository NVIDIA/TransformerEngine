# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Triton kernels that permute MXFP8 attention scales to BHSD and swizzle them for the GEMM.

Each kernel reads the SBHD/BSHD scales and writes the GEMM-swizzled BHSD layout directly, for up
to ``MAX_TENSORS`` tensors (query, key, value) in one launch. Grid axis 1 selects the tensor.
"""

import triton
import triton.language as tl

MAX_TENSORS = 3


@triton.jit
def _rowwise_tile(
    src, dst, H, S, DR, KP, ss, sb, sh, pid, HB: tl.constexpr, BLOCK_KT: tl.constexpr
):
    # Rowwise scales: matrix [B*H*S, KP] (row (b*H + h)*S + s), swizzled in 128x4 tiles.
    # A program covers 128 sequence positions x HB heads; lanes run along (head, k), which is
    # contiguous in the SBHD/BSHD source. Each int32 word packs scales k = 4*kt .. 4*kt+3 of
    # one row; row si of a tile lands at word (si % 32) * 4 + si // 32 of that tile.
    n_hb = tl.cdiv(H, HB)
    n_st = S // 128
    hb = pid % n_hb
    st = (pid // n_hb) % n_st
    b = pid // (n_hb * n_st)
    si = tl.arange(0, 128)
    j = tl.arange(0, HB * BLOCK_KT)
    h = hb * HB + j // BLOCK_KT
    kt = j % BLOCK_KT
    s = st * 128 + si
    base = s[:, None] * ss + b * sb + (h * sh)[None, :]
    ok = ((h < H) & (kt < KP // 4))[None, :]
    word = tl.zeros((128, HB * BLOCK_KT), dtype=tl.int32)
    for c in tl.static_range(4):
        k = kt * 4 + c
        v = tl.load(src + base + k[None, :], mask=ok & (k < DR)[None, :], other=0)
        word = word | (v.to(tl.int32) << (8 * c))
    tile = (b * H + h) * n_st + st
    dst_w = dst.to(tl.pointer_type(tl.int32), bitcast=True)
    off = (tile * (KP // 4) + kt)[None, :] * 128 + ((si % 32) * 4 + si // 32)[:, None]
    tl.store(dst_w + off, word, mask=ok)


@triton.jit
def _columnwise_tile(src, dst, H, SC, D, MC, ss, sb, sh, nt, mb, MB: tl.constexpr):
    # Columnwise scales: matrix [B*H*SC, NP] (row mc = (b*H + h)*SC + sc), swizzled as the
    # transposed [NP, MC] operand in 128x4 tiles. Word w of tile (nt, q) holds rows
    # mc = 4*q .. 4*q+3 at column n = nt*128 + (w % 4) * 32 + w // 4.
    w = tl.arange(0, 128)
    n = nt * 128 + (w % 4) * 32 + w // 4
    q = mb * MB + tl.arange(0, MB)
    word = tl.zeros((MB, 128), dtype=tl.int32)
    for c in tl.static_range(4):
        mc = q * 4 + c
        bh = mc // SC
        sc = mc % SC
        b = bh // H
        h = bh % H
        row = sc * ss + b * sb + h * sh
        m = (mc < MC)[:, None] & (n < D)[None, :]
        v = tl.load(src + row[:, None] + n[None, :], mask=m, other=0)
        word = word | (v.to(tl.int32) << (8 * c))
    dst_w = dst.to(tl.pointer_type(tl.int32), bitcast=True)
    off = (nt * (MC // 4) + q)[:, None] * 128 + w[None, :]
    tl.store(dst_w + off, word, mask=(q < MC // 4)[:, None])


@triton.jit
def rowwise_bhsd_swizzle_kernel(
    s0, d0, h0, sq0, dr0, kp0, ss0, sb0, sh0, t0,
    s1, d1, h1, sq1, dr1, kp1, ss1, sb1, sh1, t1,
    s2, d2, h2, sq2, dr2, kp2, ss2, sb2, sh2, t2,
    HB: tl.constexpr,
    BLOCK_KT: tl.constexpr,
):  # fmt: skip
    """Rowwise scales of up to three tensors: SBHD/BSHD -> swizzled BHSD.

    Per tensor: source, destination, H, S, D // 32, padded D // 32, source strides of S, B
    and H, and the number of programs.
    """
    pid = tl.program_id(0)
    tid = tl.program_id(1)
    if tid == 0:
        if pid < t0:
            _rowwise_tile(s0, d0, h0, sq0, dr0, kp0, ss0, sb0, sh0, pid, HB, BLOCK_KT)
    elif tid == 1:
        if pid < t1:
            _rowwise_tile(s1, d1, h1, sq1, dr1, kp1, ss1, sb1, sh1, pid, HB, BLOCK_KT)
    else:
        if pid < t2:
            _rowwise_tile(s2, d2, h2, sq2, dr2, kp2, ss2, sb2, sh2, pid, HB, BLOCK_KT)


@triton.jit
def columnwise_bhsd_swizzle_kernel(
    s0, d0, h0, sc0, dd0, mc0, ss0, sb0, sh0, nt0, t0,
    s1, d1, h1, sc1, dd1, mc1, ss1, sb1, sh1, nt1, t1,
    s2, d2, h2, sc2, dd2, mc2, ss2, sb2, sh2, nt2, t2,
    MB: tl.constexpr,
):  # fmt: skip
    """Columnwise scales of up to three tensors: SBHD/BSHD -> swizzled BHSD.

    Per tensor: source, destination, H, S // 32, D, B * H * S // 32, source strides of S, B
    and H, the number of 128-column tiles and the number of programs.
    """
    pid = tl.program_id(0)
    tid = tl.program_id(1)
    if tid == 0:
        if pid < t0:
            _columnwise_tile(s0, d0, h0, sc0, dd0, mc0, ss0, sb0, sh0, pid % nt0, pid // nt0, MB)
    elif tid == 1:
        if pid < t1:
            _columnwise_tile(s1, d1, h1, sc1, dd1, mc1, ss1, sb1, sh1, pid % nt1, pid // nt1, MB)
    else:
        if pid < t2:
            _columnwise_tile(s2, d2, h2, sc2, dd2, mc2, ss2, sb2, sh2, pid % nt2, pid // nt2, MB)
