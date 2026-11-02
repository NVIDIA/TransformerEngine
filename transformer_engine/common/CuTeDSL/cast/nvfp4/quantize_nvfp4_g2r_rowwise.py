# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Register-resident rowwise-only NVFP4 quantization: each 16-element block streams from one
32-byte load straight to its 8 output bytes, with no shared-memory staging of the data."""

from typing import Optional

import cutlass
from cutlass import cute
from cutlass import Boolean, Float32, Int32, Int64, Uint32, Uint64
from cuda.bindings.driver import CUstream  # pylint: disable=no-name-in-module

from transformer_engine.common.CuTeDSL.cast.nvfp4.utils import (
    NVFP4_BLOCK_SCALING_SIZE,
    block_scale_code,
    coefficient_of,
    compute_global_encode_sf,
    ld_global_nc_evict_first_v8,
    pack_row_block,
    prefetch_l2,
    rcp_rn_f32,
    row_offset,
    splat_f32x2,
    st_global_u8,
    st_global_v2,
)
from transformer_engine.common.CuTeDSL.philox_rng import (
    BLOCK_PHILOX_FOLD_SUPPORTED,
    block_philox_draws,
    make_block_philox,
    make_block_philox_fold,
    rowwise_sub_lo,
)
from .quantize_nvfp4_common import (
    CUTEDSL_DEBUG_LOGGING,
    NVFP4QuantizeConfig,
    NVFP4QuantizeKernelBase,
    device_sm_count,
    noop_flag_is_set,
)


class NVFP4QuantizeRowwiseG2RKernel(NVFP4QuantizeKernelBase):
    """Rowwise-only NVFP4 quantization from registers.

    A CTA owns _BLOCKS_PER_CTA consecutive blocks of ROWS rows, one block column per thread, and
    walks its rows _WAVE at a time: each thread holds _WAVE 32-byte loads in flight while the
    next wave is prefetched into L2. The per-block coefficient depends only on the block's scale
    byte, so its IEEE divide is evaluated once per CTA into a shared-memory table indexed by that
    byte. Stochastic rounding draws Philox once per block (philox_rng.make_block_philox_fold).
    """

    # A CTA covers 128 blocks = 2048 input bytes and 128 scale bytes per row.
    _BLOCKS_PER_CTA = 128
    _THREADS = 128
    # Rows per CTA. Round-to-nearest is memory-bound, and 4 rows (4 loads in flight per thread)
    # is where more CTAs stop buying latency hiding. Stochastic rounding is bound by the Philox
    # chain, so taller CTAs amortize the per-CTA prologue; it doubles _ROWS_SR while the grid
    # still fills the GPU.
    _ROWS_RNE = 4
    _ROWS_SR = 8
    # Rows held in registers at once (one 32-byte load each).
    _WAVE = 4
    # Minimum resident CTAs per SM (launch bounds): the occupancy each reaches anyway. Under SR,
    # 64 registers instead of 72 buys a quarter more warps to hide Philox with.
    _MIN_CTAS_PER_SM_RNE = 10
    _MIN_CTAS_PER_SM_SR = 8
    _MAX_GRID_Y = 65535

    @staticmethod
    def supports(cfg: NVFP4QuantizeConfig) -> bool:
        return (
            not cfg.RETURN_TRANSPOSE
            and not cfg.ROW_SCALED_NVFP4
            and not cfg.USE_FAST_MATH
            and (not cfg.USE_STOCHASTIC_ROUNDING or BLOCK_PHILOX_FOLD_SUPPORTED)
        )

    @staticmethod
    @cute.jit
    def eligible_to_dispatch(
        cfg: cutlass.Constexpr[NVFP4QuantizeConfig],
        mX: cute.Tensor,
        mO_row: cute.Tensor,
        mS_row: cute.Tensor,
        mO_col: Optional[cute.Tensor],
        mS_col: Optional[cute.Tensor],
    ) -> Boolean:
        M = mX.shape[0]
        # Rows fit grid.y at the shortest CTA, and the 32-byte loads and 8-byte stores are
        # aligned (rows and columns are multiples of 32, so only the bases matter).
        return (
            cute.ceil_div(M, NVFP4QuantizeRowwiseG2RKernel._ROWS_RNE)
            <= NVFP4QuantizeRowwiseG2RKernel._MAX_GRID_Y
            and mX.iterator.toint() % 32 == 0
            and mO_row.iterator.toint() % 8 == 0
        )

    def __init__(self, cfg: NVFP4QuantizeConfig):
        self.cfg = cfg

    @cute.jit
    def __call__(
        self,
        mX: cute.Tensor,
        mO_row: cute.Tensor,
        mS_row: cute.Tensor,
        mO_col: Optional[cute.Tensor],  # Unused, kept for API compatibility
        mS_col: Optional[cute.Tensor],  # Unused, kept for API compatibility
        mAmaxRow: cute.Tensor,
        mAmaxCol: Optional[cute.Tensor],  # Unused, kept for API compatibility
        mNoop: cute.Pointer,
        mRngState: Optional[cute.Tensor],
        stream: CUstream,
    ):
        if cutlass.const_expr(CUTEDSL_DEBUG_LOGGING):
            cute.printf(
                f"[CuTeDSL] NVFP4QuantizeRowwiseG2RKernel.__call__() with config: {self.cfg}\n"
            )
        M, N = mX.shape
        grid_x = cute.ceil_div(N // NVFP4_BLOCK_SCALING_SIZE, self._BLOCKS_PER_CTA)
        args = (mX, mO_row, mS_row, mAmaxRow, mNoop, mRngState)
        if cutlass.const_expr(self.cfg.USE_STOCHASTIC_ROUNDING):
            # Under SR the Philox chain, not DRAM, sets the pace, so the per-CTA prologue (amax
            # load, divides, coefficient table, barrier) is a real fraction of each CTA. Doubling
            # the stripe halves it, worth taking whenever the taller grid still fills the device.
            resident = self._MIN_CTAS_PER_SM_SR * device_sm_count()
            if Int64(grid_x) * cute.ceil_div(M, 2 * self._ROWS_SR) >= resident:
                self._launch(args, grid_x, M, stream, 2 * self._ROWS_SR)
            else:
                self._launch(args, grid_x, M, stream, self._ROWS_SR)
        else:
            self._launch(args, grid_x, M, stream, self._ROWS_RNE)

    @cute.jit
    def _launch(self, args, grid_x: Int32, M: Int32, stream: CUstream, ROWS: cutlass.Constexpr):
        min_ctas = (
            self._MIN_CTAS_PER_SM_SR
            if self.cfg.USE_STOCHASTIC_ROUNDING
            else self._MIN_CTAS_PER_SM_RNE
        )
        self.kernel(*args, ROWS).launch(
            grid=[grid_x, cute.ceil_div(M, ROWS), 1],
            block=[self._THREADS, 1, 1],
            stream=stream,
            min_blocks_per_mp=min_ctas,
            # Leave the shared-memory carveout to the driver. Derived from min_blocks_per_mp, it
            # ignores the driver's per-CTA reservation and caps residency below min_ctas.
            preferred_smem_carveout=-1,
        )

    @cute.kernel
    def kernel(
        self,
        mX: cute.Tensor,
        mO_row: cute.Tensor,
        mS_row: cute.Tensor,
        mAmaxRow: cute.Tensor,
        mNoop: cute.Pointer,
        mRngState: Optional[cute.Tensor],
        ROWS: cutlass.Constexpr,
    ):
        """Device entry for the rowwise-only register kernel."""
        STOCHASTIC = self.cfg.USE_STOCHASTIC_ROUNDING
        WAVE = self._WAVE
        tidx, _, _ = cute.arch.thread_idx()
        bidx, bidy, _ = cute.arch.block_idx()
        K = mX.shape[1]
        blocks_per_row = K // NVFP4_BLOCK_SCALING_SIZE
        s_pitch = cute.size(mS_row, mode=[1])

        smem = cutlass.memory.SmemAllocator()
        # Rowwise RNE consumes every coefficient as a packed pair, so it stores that form once;
        # SR keeps the narrower float.
        table = smem.allocate_tensor(
            Float32 if STOCHASTIC else Int64, cute.make_layout(self._THREADS), byte_alignment=16
        )

        # Launch-wide scalars are read here but consumed further down: each heads a dependency
        # chain off a global load, so reading them first overlaps those loads with the input's.
        skip = noop_flag_is_set(mNoop)
        amax_row = mAmaxRow[0]
        rng_seed = Uint64(0)
        rng_offset = Uint64(0)
        if cutlass.const_expr(STOCHASTIC):
            rng_seed = Uint64(mRngState[0])
            rng_offset = Uint64(mRngState[1])

        j0 = bidx * self._BLOCKS_PER_CTA + tidx
        j = j0
        if j0 >= blocks_per_row:
            j = Int32(0)  # clamp so the loads stay in bounds; the stores are predicated
        r0 = bidy * ROWS

        # The column term of every address is a thread constant, folded into the bases once.
        x_j = mX.iterator.toint() + Int64(j) * (NVFP4_BLOCK_SCALING_SIZE * 2)
        q_j = mO_row.iterator.toint() + Int64(j) * (NVFP4_BLOCK_SCALING_SIZE // 2)
        s_j = mS_row.iterator.toint() + Int64(j)
        x_pitch = K * 2  # bytes per input row
        q_pitch = K // 2  # bytes per output row

        v = [ld_global_nc_evict_first_v8(x_j + row_offset(r0 + i, x_pitch)) for i in range(WAVE)]

        if not skip:  # CTA-uniform, and no barrier has been reached yet
            # Resolve the RNG first: its offset-dependent multiply has the longer wait, and the
            # already-issued loads continue toward readiness behind it. The block column is a
            # thread constant, so placing it in the subsequence's low word folds rounds 0..2.
            fold = None
            if cutlass.const_expr(STOCHASTIC):
                fold = make_block_philox_fold(
                    make_block_philox(rng_seed, rng_offset), rowwise_sub_lo(Uint32(j))
                )

            S_enc = compute_global_encode_sf(amax_row)
            S_enc_div6 = S_enc * Float32(1.0 / 6.0)
            # One table entry per thread; its barrier is deferred into the first wave, past that
            # wave's amax trees, which depend only on the input already in flight.
            coeff = coefficient_of(Uint32(tidx), rcp_rn_f32(S_enc))
            if cutlass.const_expr(STOCHASTIC):
                table[tidx] = coeff
            else:
                table[tidx] = splat_f32x2(coeff)

            active = j0 < blocks_per_row
            for w in cutlass.range_constexpr(0, ROWS, WAVE):
                if cutlass.const_expr(w > 0):
                    v = [
                        ld_global_nc_evict_first_v8(x_j + row_offset(r0 + w + i, x_pitch))
                        for i in range(WAVE)
                    ]
                # Register-free L2 warm-up for the next wave. Those rows are inside this CTA's
                # stripe (ROWS divides 32 and M % 32 == 0), so no bounds test is needed.
                if cutlass.const_expr(w + WAVE < ROWS):
                    for i in cutlass.range_constexpr(WAVE):
                        prefetch_l2(x_j + row_offset(r0 + w + WAVE + i, x_pitch))
                # The wave's first random pair needs none of the loaded data, so draw it before
                # the amax trees that stall on the loads.
                rnd0 = None
                if cutlass.const_expr(STOCHASTIC):
                    rnd0 = block_philox_draws([Uint32(r0 + w), Uint32(r0 + w + 1)], fold)
                # Every block's amax tree and scale byte resolve before any of them packs.
                codes = [block_scale_code(v[i], S_enc_div6) for i in range(WAVE)]
                if cutlass.const_expr(w == 0):
                    cute.arch.sync_threads()  # the coefficient table is read only from here on

                for g in cutlass.range_constexpr(0, WAVE, 2):
                    rnd = [None, None]
                    if cutlass.const_expr(STOCHASTIC):
                        if cutlass.const_expr(g == 0):
                            rnd = rnd0
                        else:
                            rnd = block_philox_draws(
                                [Uint32(r0 + w + g), Uint32(r0 + w + g + 1)], fold
                            )
                    for i in cutlass.range_constexpr(g, g + 2):
                        if cutlass.const_expr(STOCHASTIC):
                            coeff2 = splat_f32x2(table[Int32(codes[i])])
                        else:
                            coeff2 = table[Int32(codes[i])]
                        w0, w1 = pack_row_block(v[i], coeff2, rnd[i - g], STOCHASTIC)
                        row = r0 + w + i
                        if active:
                            st_global_v2(q_j + row_offset(row, q_pitch), w0, w1)
                            st_global_u8(s_j + row_offset(row, s_pitch), codes[i])
