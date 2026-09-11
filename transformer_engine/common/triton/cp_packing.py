# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Bit-preserving packing between balanced sequence and head partitions."""

import triton
import triton.language as tl


@triton.jit
def _pack_cp_tensors(
    Q,
    K,
    V,
    O,
    DO,
    LSE,
    WIRE,
    S: tl.constexpr,
    B: tl.constexpr,
    DQ: tl.constexpr,
    DV: tl.constexpr,
    H: tl.constexpr,
    CP: tl.constexpr,
    Q0: tl.constexpr,
    Q1: tl.constexpr,
    Q2: tl.constexpr,
    K0: tl.constexpr,
    K1: tl.constexpr,
    K2: tl.constexpr,
    V0: tl.constexpr,
    V1: tl.constexpr,
    V2: tl.constexpr,
    O0: tl.constexpr,
    O1: tl.constexpr,
    O2: tl.constexpr,
    D0: tl.constexpr,
    D1: tl.constexpr,
    D2: tl.constexpr,
    L0: tl.constexpr,
    L1: tl.constexpr,
    L2: tl.constexpr,
    FORWARD: tl.constexpr,
    W: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = i < CP * S * B * H * W
    f = i % W
    h = i // W % H
    b = i // (W * H) % B
    s = i // (W * H * B) % S
    peer = i // (W * H * B * S)
    if FORWARD:
        source_s = s
        source_h = peer * H + h
    else:
        source_s = tl.where(
            s < S // 2, peer * (S // 2) + s, (2 * CP - 1 - peer) * (S // 2) + s - S // 2
        )
        source_h = h
    bits = tl.load(Q + source_s * Q0 + b * Q1 + source_h * Q2 + f, valid & (f < DQ), other=0)
    bits += tl.load(
        K + source_s * K0 + b * K1 + source_h * K2 + f - DQ,
        valid & (f >= DQ) & (f < (2 * DQ)),
        other=0,
    )
    bits += tl.load(
        V + source_s * V0 + b * V1 + source_h * V2 + f - (2 * DQ),
        valid & (f >= (2 * DQ)) & (f < (2 * DQ + DV)),
        other=0,
    )
    if FORWARD:
        bits += tl.load(
            O + source_s * O0 + b * O1 + source_h * O2 + f - (2 * DQ + DV),
            valid & (f >= (2 * DQ + DV)) & (f < (2 * DQ + 2 * DV)),
            other=0,
        )
        bits += tl.load(
            DO + source_s * D0 + b * D1 + source_h * D2 + f - (2 * DQ + 2 * DV),
            valid & (f >= (2 * DQ + 2 * DV)) & (f < (2 * DQ + 3 * DV)),
            other=0,
        )
        lse = tl.load(
            LSE + b * L0 + source_h * L1 + source_s * L2,
            valid & (f >= (2 * DQ + 3 * DV)) & (f < (2 * DQ + 3 * DV + 2)),
            other=0,
        ).to(tl.uint32)
        half = tl.where(f == (2 * DQ + 3 * DV), lse & 65535, lse >> 16).to(tl.int16)
        bits += tl.where((f >= (2 * DQ + 3 * DV)) & (f < (2 * DQ + 3 * DV + 2)), half, 0)
    tl.store(WIRE + i, bits, valid)


@triton.jit
def _unpack_cp_tensors(
    WIRE,
    Q,
    K,
    V,
    O,
    DO,
    LSE,
    S: tl.constexpr,
    B: tl.constexpr,
    DQ: tl.constexpr,
    DV: tl.constexpr,
    H: tl.constexpr,
    CP: tl.constexpr,
    FORWARD: tl.constexpr,
    W: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = i < CP * S * B * H * W
    f = i % W
    h = i // W % H
    b = i // (W * H) % B
    s = i // (W * H * B) % S
    peer = i // (W * H * B * S)
    bits = tl.load(WIRE + i, valid, other=0)
    if FORWARD:
        target_s = tl.where(
            s < S // 2, peer * (S // 2) + s, (2 * CP - 1 - peer) * (S // 2) + s - S // 2
        )
        kv_owner = (peer + 1) % CP
        kv_s = tl.where(
            s < S // 2,
            kv_owner * (S // 2) + s,
            (2 * CP - 1 - kv_owner) * (S // 2) + s - S // 2,
        )
        row = (target_s * B + b) * H + h
        kv_row = (kv_s * B + b) * H + h
    else:
        row = (s * B + b) * (CP * H) + peer * H + h
        kv_row = row
    tl.store(Q + row * DQ + f, bits, valid & (f < DQ))
    tl.store(K + kv_row * DQ + f - DQ, bits, valid & (f >= DQ) & (f < (2 * DQ)))
    tl.store(V + kv_row * DV + f - (2 * DQ), bits, valid & (f >= (2 * DQ)) & (f < (2 * DQ + DV)))
    if FORWARD:
        tl.store(
            O + row * DV + f - (2 * DQ + DV),
            bits,
            valid & (f >= (2 * DQ + DV)) & (f < (2 * DQ + 2 * DV)),
        )
        tl.store(
            DO + row * DV + f - (2 * DQ + 2 * DV),
            bits,
            valid & (f >= (2 * DQ + 2 * DV)) & (f < (2 * DQ + 3 * DV)),
        )
        tl.store(
            LSE + 2 * ((b * H + h) * S * CP + target_s) + f - (2 * DQ + 3 * DV),
            bits,
            valid & (f >= (2 * DQ + 3 * DV)) & (f < (2 * DQ + 3 * DV + 2)),
        )
