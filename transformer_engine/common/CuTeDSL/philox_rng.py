# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Philox random-number generation for CuTeDSL kernels. Based on curanddx.hpp"""

import os

import cutlass
from cutlass import cute
from cutlass import Boolean, Uint32, Uint64

# per envvars.rst
NUM_PHILOX_ROUNDS = int(os.environ.get("NVTE_BUILD_NUM_PHILOX_ROUNDS", "10"))

# per curanddx.hpp
PHILOX_W32_0 = 0x9E3779B9
PHILOX_W32_1 = 0xBB67AE85
PHILOX_M4x32_0 = 0xD2511F53
PHILOX_M4x32_1 = 0xCD9E8D57


@cute.jit
def umulhi_u32(a: Uint32, b: Uint32) -> Uint32:
    """High 32 bits of the unsigned 32x32 product (`__umulhi`)."""
    return cute.arch.inline_ptx(
        "mul.hi.u32 {$w0}, {$r0}, {$r1};",
        write_only_types=[Uint32],
        read_only_args=[a, b],
    )


@cute.jit
def u64_lo32(v: Uint64) -> Uint32:
    """Low 32 bits of a u64."""
    return Uint32(v)


@cute.jit
def u64_hi32(v: Uint64) -> Uint32:
    """High 32 bits of a u64."""
    return Uint32(v >> Uint64(32))


@cute.jit
def bool_to_u64(b: Boolean) -> Uint64:
    """0 or 1 as a u64."""
    return Uint64(b)


class PhiloxRng:
    """Trace-time replica of the curanddx philox4x32 generator the CUDA kernel rounds with.

    The counter is kept as two u64 halves: the low half is the offset the state was initialized
    with and the high half the subsequence, exactly the state philox4x32_native_state::init
    reaches (both increments start from a zero counter, so their carry chains vanish), and
    incrementing the 128-bit counter is then one add-with-carry. Consumption is deterministic --
    every loop that draws random words is unrolled -- so the one-of-four word cycling of
    core::get_rbits is a Python-side list rather than a device-side index.
    """

    def __init__(self, seed: Uint64, subsequence: Uint64, offset: Uint64):
        self._key = (u64_lo32(seed), u64_hi32(seed))
        self._ctr_lo = offset
        self._ctr_hi = subsequence
        self._buf = []
        # Matches the CUDA kernel's `random_uint4 = rng.generate4()` at init.
        self._generate4()

    def _generate4(self):
        c = [
            u64_lo32(self._ctr_lo),
            u64_hi32(self._ctr_lo),
            u64_lo32(self._ctr_hi),
            u64_hi32(self._ctr_hi),
        ]
        k0, k1 = self._key
        for round_idx in range(NUM_PHILOX_ROUNDS):
            if round_idx > 0:
                k0 = k0 + Uint32(PHILOX_W32_0)
                k1 = k1 + Uint32(PHILOX_W32_1)
            lo0 = Uint32(PHILOX_M4x32_0) * c[0]
            hi0 = umulhi_u32(Uint32(PHILOX_M4x32_0), c[0])
            lo1 = Uint32(PHILOX_M4x32_1) * c[2]
            hi1 = umulhi_u32(Uint32(PHILOX_M4x32_1), c[2])
            c = [hi1 ^ c[1] ^ k0, lo1, hi0 ^ c[3] ^ k1, lo0]
        self._buf = c
        new_lo = self._ctr_lo + Uint64(1)
        self._ctr_hi = self._ctr_hi + bool_to_u64(new_lo == Uint64(0))
        self._ctr_lo = new_lo

    def get_rbits(self) -> Uint32:
        """The next 32 random bits, regenerating every four words like core::get_rbits."""
        if not self._buf:
            self._generate4()
        return self._buf.pop(0)


# --- Block draws: one Philox4x32 generate per output block, several streams at a time ---
#
# Each draw is exactly what curanddx's philox4x32_native_state<NUM_PHILOX_ROUNDS> returns from
# init(seed, subsequence, offset) followed by one generate4(): the counter is
# {offset_lo, offset_hi, subsequence_lo, subsequence_hi}. Only the cost differs from PhiloxRng:
#  - every round-0 term that reads only the launch-wide offset is evaluated once
#    (make_block_philox);
#  - when a thread draws for many blocks that share one subsequence word (its block column, say),
#    placing that word in the subsequence's LOW half makes both of round 0's products thread
#    constants, and the constant parts of rounds 0..2 fold away (make_block_philox_fold). The
#    other half carries the varying coordinate XORed with the launch constant `b`, a bijection that
#    cancels round 0's own XOR.

# make_block_philox_fold consumes rounds 0..2, so callers need a plain generator below 3 rounds.
BLOCK_PHILOX_FOLD_SUPPORTED = NUM_PHILOX_ROUNDS >= 3


# Subsequence of each output block, as (low word, logical high word). The low word is the
# coordinate a thread keeps fixed while it walks the other, so make_block_philox_fold can fold
# it. Its parity tells the directions apart, so the rowwise and columnwise draws of one element
# never share a subsequence; within a direction the pair is the block's coordinates.
#   rowwise block (row, block_col):                  (2 * block_col,       row)
#   columnwise block (row_block, col), col = 4q + i: (2 * (4q) + 1, 4 * row_block + i)
# The column's offset inside its group of four moves to the high word because a thread owns four
# adjacent columns and folds against their base.
@cute.jit
def rowwise_sub_lo(block_col: Uint32) -> Uint32:
    """Subsequence low word of a rowwise block."""
    return block_col << Uint32(1)


@cute.jit
def columnwise_sub_lo(quad_base_col: Uint32) -> Uint32:
    """Subsequence low word of the columnwise blocks of four adjacent columns."""
    return (quad_base_col << Uint32(1)) | Uint32(1)


@cute.jit
def columnwise_sub_hi(row_block: Uint32, col_in_quad: Uint32) -> Uint32:
    """Logical subsequence high word of a columnwise block."""
    return (row_block << Uint32(2)) | col_in_quad


@cute.jit
def mulhilo_u32(a: Uint32, b: Uint32):
    """Both halves of a 32 x 32 -> 64-bit product in one IMAD.WIDE.U32, as (lo, hi)."""
    p = cute.arch.inline_ptx(
        "mul.wide.u32 {$w0}, {$r0}, {$r1};",
        write_only_types=[Uint64],
        read_only_args=[a, b],
    )
    return Uint32(p), Uint32(p >> Uint64(32))


@cute.jit
def xor3_u32(a: Uint32, b: Uint32, c: Uint32) -> Uint32:
    """Three-way XOR in one LOP3."""
    return cute.arch.inline_ptx(
        "lop3.b32 {$w0}, {$r0}, {$r1}, {$r2}, 0x96;",
        write_only_types=[Uint32],
        read_only_args=[a, b, c],
    )


@cute.jit
def make_block_philox(seed: Uint64, offset: Uint64):
    """Round-0 terms that read only the key and the launch-wide offset, as
    (key0, key1, a, b, w): the key round 1 uses, a = offset_hi ^ k0 (round-0 c.x is
    hi(sub_lo * M1) ^ a), b = hi(offset_lo * M0) ^ k1 (round-0 c.z is b ^ sub_hi) and
    w = lo(offset_lo * M0) (round-0 c.w)."""
    k0 = u64_lo32(seed)
    k1 = u64_hi32(seed)
    lo0, hi0 = mulhilo_u32(u64_lo32(offset), Uint32(PHILOX_M4x32_0))
    return (
        k0 + Uint32(PHILOX_W32_0),
        k1 + Uint32(PHILOX_W32_1),
        u64_hi32(offset) ^ k0,
        hi0 ^ k1,
        lo0,
    )


@cute.jit
def make_block_philox_fold(state, sub_lo: Uint32):
    """Rounds 0..2 for a fixed subsequence low word, as (A, B, D, c1_2, k3x, k3y):
    c0_1 = hi1 ^ A, c0_2 = B ^ c1_1, c2_2 = hi0 ^ D, c1_2 the round-2 c1 and k3 the round-3 key."""
    kx, ky, a, _, w = state
    lo1, hi1 = mulhilo_u32(sub_lo, Uint32(PHILOX_M4x32_1))  # round 0: both products constant
    c0_0 = hi1 ^ a
    c1_0 = lo1
    lo0, hi0 = mulhilo_u32(c0_0, Uint32(PHILOX_M4x32_0))  # round 1: the .x product is constant
    fold_a = c1_0 ^ kx
    c2_1 = xor3_u32(hi0, w, ky)
    c3_1 = lo0
    kx = kx + Uint32(PHILOX_W32_0)
    ky = ky + Uint32(PHILOX_W32_1)  # round-2 key
    lo1, hi1 = mulhilo_u32(c2_1, Uint32(PHILOX_M4x32_1))  # round 2: the .z product is constant
    fold_b = hi1 ^ kx
    c1_2 = lo1
    fold_d = c3_1 ^ ky
    kx = kx + Uint32(PHILOX_W32_0)
    ky = ky + Uint32(PHILOX_W32_1)  # round-3 key
    return (fold_a, fold_b, fold_d, c1_2, kx, ky)


@cute.jit
def block_philox_draws(logical_hi, fold):
    """One draw per entry of `logical_hi` (the varying coordinate of a block sharing the fold's
    low word; the draw's subsequence high word is logical_hi ^ b), as a list of 4-word lists.
    The draws advance together, so the key schedule is paid once per batch."""
    fold_a, fold_b, fold_d, c1_2, kx, ky = fold
    c = []
    for hi in logical_hi:  # the varying remnant of rounds 0..2
        lo1, hi1 = mulhilo_u32(hi, Uint32(PHILOX_M4x32_1))
        lo0, hi0 = mulhilo_u32(hi1 ^ fold_a, Uint32(PHILOX_M4x32_0))
        c.append([fold_b ^ lo1, c1_2, hi0 ^ fold_d, lo0])
    for r in cutlass.range_constexpr(3, NUM_PHILOX_ROUNDS):
        for i in cutlass.range_constexpr(len(c)):
            lo0, hi0 = mulhilo_u32(c[i][0], Uint32(PHILOX_M4x32_0))
            lo1, hi1 = mulhilo_u32(c[i][2], Uint32(PHILOX_M4x32_1))
            c[i] = [xor3_u32(hi1, c[i][1], kx), lo1, xor3_u32(hi0, c[i][3], ky), lo0]
        if cutlass.const_expr(r < NUM_PHILOX_ROUNDS - 1):
            kx = kx + Uint32(PHILOX_W32_0)
            ky = ky + Uint32(PHILOX_W32_1)
    return c
