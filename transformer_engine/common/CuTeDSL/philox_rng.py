# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Philox random-number generation for CuTeDSL kernels. Based on curanddx.hpp"""

import os

from cutlass import Uint32, Uint64

from transformer_engine.common.CuTeDSL.utils import (
    bool_to_u64,
    u64_hi32,
    u64_lo32,
    umulhi_u32,
)


# per envvars.rst
NUM_PHILOX_ROUNDS = int(os.environ.get("NVTE_BUILD_NUM_PHILOX_ROUNDS", "10"))

# per curanddx.hpp
PHILOX_W32_0 = 0x9E3779B9
PHILOX_W32_1 = 0xBB67AE85
PHILOX_M4x32_0 = 0xD2511F53
PHILOX_M4x32_1 = 0xCD9E8D57

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
