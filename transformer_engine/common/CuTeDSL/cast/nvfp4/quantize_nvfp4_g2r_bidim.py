# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Register-resident rowwise + columnwise NVFP4 quantization: every row is read twice, once per
direction, with the second read served by L2, and only the transposed output is staged in shared
memory."""

from typing import Optional

import cutlass
from cutlass import cute
from cutlass import Boolean, Float32, Int32, Int64, Uint32, Uint64
from cuda.bindings.driver import CUstream  # pylint: disable=no-name-in-module

from transformer_engine.common.CuTeDSL.utils import (
    abs_max_tree_bf16x2,
    ld_global_evict_last_v8,
    ld_global_v2,
    pack_f32x2,
    pack_u32x2,
    prefetch_l2,
    rcp_rn_f32,
    row_offset,
    st_global_u16,
    st_global_u32,
    st_global_u8,
    st_global_v2,
    st_global_v4,
    st_global_v8,
    unpack_i64_to_i32x2,
)
from transformer_engine.common.CuTeDSL.utils_fp8 import (
    cvt_f32_to_fp8e4m3_code,
)
from transformer_engine.common.CuTeDSL.utils_nvfp4 import (
    block_scale_code,
    coefficient_of,
    compute_global_encode_sf,
    pack_column_block,
    pack_row_block,
)
from transformer_engine.common.CuTeDSL.philox_rng import (
    BLOCK_PHILOX_FOLD_SUPPORTED,
    block_philox_draws,
    columnwise_sub_hi,
    columnwise_sub_lo,
    make_block_philox,
    make_block_philox_fold,
    rowwise_sub_lo,
)
from .quantize_nvfp4_common import (
    CUTEDSL_DEBUG_LOGGING,
    NVFP4_BLOCK_SCALING_SIZE,
    device_sm_count,
    noop_flag_is_set,
    NVFP4QuantizeConfig,
    NVFP4QuantizeKernelBase,
)


class NVFP4QuantizeBidimensionalG2RKernel(NVFP4QuantizeKernelBase):
    """Rowwise + columnwise NVFP4 quantization from registers.

    A CTA owns _TILE_K columns and walks up to _MAX_SLABS slabs of _SLAB_M rows, keeping the
    whole transposed FP4 result for its columns in shared memory:
      A. rowwise: one block per thread per quarter slab, from a GROUP-CTA wide strip (adjacent
         CTAs pool their rowwise work so a warp writes 256 contiguous payload bytes and whole
         32-byte scale sectors);
      B. columnwise: four adjacent columns per thread over one 16-row block, re-reading the rows
         phase A just brought into L2, into the transposed staging buffer;
      C. drain: four lanes per column, each writing one 32-byte sector of its transposed run.
    The slab count is chosen per launch by a cost model of streaming work against the per-CTA
    drain.
    """

    _THREADS = 128
    _TILE_K = 128
    _SLAB_M = 64
    _MAX_SLABS = 4
    _GROUP_CTAS = 4
    # Cost model for the slab count: waves * (_SLAB_WORK * slabs + _DRAIN_WORK), in memory
    # instructions per thread.
    _SLAB_WORK = 36
    _DRAIN_WORK = 51
    # Minimum resident CTAs per SM (launch bounds): the occupancy each reaches anyway.
    _MIN_CTAS_PER_SM_RNE = 7
    _MIN_CTAS_PER_SM_SR = 5
    _MAX_GRID_Y = 65535

    _BLOCKS_PER_SLAB = _SLAB_M // NVFP4_BLOCK_SCALING_SIZE
    # Transposed staging buffer: one row of _STAGING_WORDS words per column. A row stride of
    # 2 (mod 32) words puts the sixteen lanes of an 8-byte shared access on sixteen disjoint bank
    # pairs for both the staging store and the drain's gather, and an even stride keeps every
    # slot 8-byte aligned.
    _STAGING_WORDS = 34
    _STAGING_SCALE_BYTES = _BLOCKS_PER_SLAB * _MAX_SLABS + 4
    assert _STAGING_WORDS >= 2 * _BLOCKS_PER_SLAB * _MAX_SLABS
    assert _STAGING_WORDS % 32 == 2
    assert _STAGING_SCALE_BYTES % 4 == 0

    @staticmethod
    def supports(cfg: NVFP4QuantizeConfig) -> bool:
        return (
            cfg.RETURN_TRANSPOSE
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
        M, K = mX.shape
        # Row strides in elements stay 32-bit (16 rows of K per pass), and the 32-byte loads,
        # 8-byte rowwise stores, 32-byte transposed stores and 4-byte transposed scale stores are
        # aligned (rows and columns are multiples of 32, so only the bases matter).
        return (
            K <= (2**31 - 1) // 16
            and cute.ceil_div(M, NVFP4QuantizeBidimensionalG2RKernel._SLAB_M)
            <= NVFP4QuantizeBidimensionalG2RKernel._MAX_GRID_Y
            and mX.iterator.toint() % 32 == 0
            and mO_row.iterator.toint() % 8 == 0
            and mO_col.iterator.toint() % 32 == 0
            and mS_col.iterator.toint() % 4 == 0
        )

    def __init__(self, cfg: NVFP4QuantizeConfig):
        self.cfg = cfg

    @cute.jit
    def __call__(
        self,
        mX: cute.Tensor,
        mO_row: cute.Tensor,
        mS_row: cute.Tensor,
        mO_col: Optional[cute.Tensor],
        mS_col: Optional[cute.Tensor],
        mAmaxRow: cute.Tensor,
        mAmaxCol: Optional[cute.Tensor],
        mNoop: cute.Pointer,
        mRngState: Optional[cute.Tensor],
        stream: CUstream,
    ):
        if cutlass.const_expr(CUTEDSL_DEBUG_LOGGING):
            cute.printf(
                "[CuTeDSL] NVFP4QuantizeBidimensionalG2RKernel.__call__() with config:"
                f" {self.cfg}\n"
            )
        M, K = mX.shape
        args = (mX, mO_row, mS_row, mO_col, mS_col, mAmaxRow, mAmaxCol, mNoop, mRngState)
        # The strip remap needs every column of a GROUP-CTA strip to exist.
        if K % (self._GROUP_CTAS * self._TILE_K) == 0:
            self._launch(args, M, K, stream, self._GROUP_CTAS)
        else:
            self._launch(args, M, K, stream, 1)

    @cute.jit
    def _launch(self, args, M: Int32, K: Int32, stream: CUstream, GROUP: cutlass.Constexpr):
        min_ctas = (
            self._MIN_CTAS_PER_SM_SR
            if self.cfg.USE_STOCHASTIC_ROUNDING
            else self._MIN_CTAS_PER_SM_RNE
        )
        resident = Int64(min_ctas * device_sm_count())
        grid_x = cute.ceil_div(K, self._TILE_K)
        # Taller CTAs make each transposed run longer (the drain costs one request per column
        # whatever the run length) but shrink the grid, and a grid that no longer covers the
        # device several times wastes its tail. Pick the slab count by
        # waves * (slab streaming work + per-CTA drain).
        num_slabs = Int32(1)
        best = Int64(-1)
        for n in cutlass.range_constexpr(3):
            slabs = 1 << n
            if cutlass.const_expr(slabs <= self._MAX_SLABS):
                grid = Int64(grid_x) * cute.ceil_div(Int64(M), slabs * self._SLAB_M)
                waves = (grid + resident - 1) // resident
                filled = cutlass.min(grid, resident)
                cost = waves * (self._SLAB_WORK * slabs + self._DRAIN_WORK) * resident // filled
                if best < 0 or cost < best:
                    best = cost
                    num_slabs = Int32(slabs)
        self.kernel(*args, num_slabs, GROUP).launch(
            grid=[grid_x, cute.ceil_div(M, num_slabs * self._SLAB_M), 1],
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
        mO_col: cute.Tensor,
        mS_col: cute.Tensor,
        mAmaxRow: cute.Tensor,
        mAmaxCol: cute.Tensor,
        mNoop: cute.Pointer,
        mRngState: Optional[cute.Tensor],
        num_slabs: Int32,
        GROUP: cutlass.Constexpr,
    ):
        """Device entry for the rowwise + columnwise register kernel."""
        STOCHASTIC = self.cfg.USE_STOCHASTIC_ROUNDING
        THREADS = self._THREADS
        TILE_K = self._TILE_K
        SLAB_M = self._SLAB_M
        BLOCK = NVFP4_BLOCK_SCALING_SIZE
        tidx, _, _ = cute.arch.thread_idx()
        bidx, bidy, _ = cute.arch.block_idx()
        M, K = mX.shape
        blocks_per_row = K // BLOCK
        s_row_pitch = cute.size(mS_row, mode=[1])
        s_col_pitch = cute.size(mS_col, mode=[1])
        x_base = mX.iterator.toint()
        q_row_base = mO_row.iterator.toint()
        s_row_base = mS_row.iterator.toint()
        q_col_base = mO_col.iterator.toint()
        s_col_base = mS_col.iterator.toint()

        smem = cutlass.memory.SmemAllocator()
        # [0, 128) rowwise coefficients by scale byte, [128, 256) columnwise.
        table = smem.allocate_tensor(Float32, cute.make_layout(2 * THREADS), byte_alignment=16)
        # Transposed result, column-major: staging[column slot][word], as 8-byte pairs.
        staging = smem.allocate_tensor(
            Int64, cute.make_layout(TILE_K * self._STAGING_WORDS // 2), byte_alignment=16
        )
        staging_scales = smem.allocate_tensor(
            Uint32, cute.make_layout(TILE_K * self._STAGING_SCALE_BYTES // 4), byte_alignment=16
        )
        staging_scales_u8 = cute.make_tensor(
            cute.recast_ptr(staging_scales.iterator, dtype=cutlass.Uint8),
            cute.make_layout(TILE_K * self._STAGING_SCALE_BYTES),
        )

        # Launch-wide scalars are read here but consumed further down: each heads a dependency
        # chain off a global load, so reading them first overlaps those loads with the input's.
        skip = noop_flag_is_set(mNoop)
        amax_row = mAmaxRow[0]
        amax_col = mAmaxCol[0]
        rng_seed = Uint64(0)
        rng_offset = Uint64(0)
        if cutlass.const_expr(STOCHASTIC):
            rng_seed = Uint64(mRngState[0])
            rng_offset = Uint64(mRngState[1])

        c0 = bidx * TILE_K
        r00 = bidy * (num_slabs * SLAB_M)

        # Phase A lanes: JA block columns per row x RG rows per pass, from the GROUP-wide strip
        # this CTA belongs to (GROUP == 1 is the plain per-tile mapping).
        JA = 8 * GROUP
        RG = THREADS // JA
        ROWS_A = SLAB_M // GROUP
        qa = bidx % GROUP  # position inside the strip
        jbase = (bidx - qa) * (TILE_K // BLOCK)
        jl = tidx % JA
        rl = tidx // JA
        j0 = jbase + jl
        j = j0
        if j0 >= blocks_per_row:
            j = jbase
        arow = j0 < blocks_per_row
        # A band of at most 32 rows that starts inside the matrix also ends inside it
        # (M % 32 == 0), so clamping the band base is all the bounds work phase A's walk needs.
        BAND_SAFE = ROWS_A <= 32
        x_pitch = K * 2
        q_pitch = K // 2
        x_col = x_base + Int64(j) * (BLOCK * 2)
        q_col_off = q_row_base + Int64(j) * (BLOCK // 2)
        s_col_off = s_row_base + Int64(j)

        # Phase B lanes: four adjacent columns per thread (a warp spans 128 columns, one
        # 256-byte request per row) and one of the slab's four column blocks.
        cp = tidx % 32
        mb = tidx // 32
        c = c0 + 4 * cp
        cok = c < K

        # The first real load is inside the slab loop, so it cannot be hoisted above the
        # encode-scale chain; a prefetch can, and warms the first slab meanwhile.
        pb = r00 + ROWS_A * qa + rl
        for i in cutlass.range_constexpr(4):
            r = pb + RG * i
            if cutlass.const_expr(BAND_SAFE):
                if pb >= M:
                    r = r00 + RG * i
            else:
                if r >= M:
                    r = r00
            prefetch_l2(x_col + row_offset(r, x_pitch))

        if not skip:  # CTA-uniform, and no barrier has been reached yet
            S_enc_row = compute_global_encode_sf(amax_row)
            # The usual caller passes the same global amax in both directions; recognizing that
            # from the raw bits skips a second divide chain in the CTA prologue.
            S_enc_col = S_enc_row
            if amax_col.bitcast(Uint32) != amax_row.bitcast(Uint32):
                S_enc_col = compute_global_encode_sf(amax_col)
            S_enc_row_div6 = S_enc_row * Float32(1.0 / 6.0)
            S_enc_col_div6 = S_enc_col * Float32(1.0 / 6.0)
            # Both halves fold Philox rounds 0..2 against a low word that is constant for the
            # CTA: the rowwise half keys on its block column, the columnwise half on the base of
            # its four columns, with the column's offset in the quad folded into the high word.
            fold_row = None
            fold_col = None
            if cutlass.const_expr(STOCHASTIC):
                state = make_block_philox(rng_seed, rng_offset)
                fold_row = make_block_philox_fold(state, rowwise_sub_lo(Uint32(j)))
                fold_col = make_block_philox_fold(state, columnwise_sub_lo(Uint32(c)))

            # Equal encode scales share one set of divides.
            S_dec_row = rcp_rn_f32(S_enc_row)
            row_coeff = coefficient_of(Uint32(tidx), S_dec_row)
            col_coeff = row_coeff
            if S_enc_col.bitcast(Uint32) != S_enc_row.bitcast(Uint32):
                col_coeff = coefficient_of(Uint32(tidx), rcp_rn_f32(S_enc_col))
            table[tidx] = row_coeff
            table[THREADS + tidx] = col_coeff
            cute.arch.sync_threads()

            for sub in cutlass.range(num_slabs, unroll=1):
                r0 = r00 + sub * SLAB_M
                if r0 < M:
                    self._slab(
                        r0,
                        sub,
                        M,
                        x_base,
                        x_col,
                        q_col_off,
                        s_col_off,
                        x_pitch,
                        q_pitch,
                        s_row_pitch,
                        qa,
                        rl,
                        arow,
                        cp,
                        mb,
                        c,
                        cok,
                        num_slabs,
                        table,
                        staging,
                        staging_scales_u8,
                        S_enc_row_div6,
                        S_enc_col_div6,
                        fold_row,
                        fold_col,
                        GROUP,
                        BAND_SAFE,
                        STOCHASTIC,
                    )

            cute.arch.sync_threads()

            # Phase C: four lanes cooperate on one column, each writing a whole 32-byte sector,
            # so a column's transposed run leaves as one coalesced request.
            valid_blocks = cutlass.min(num_slabs * self._BLOCKS_PER_SLAB, (M - r00) // BLOCK)
            ll = tidx % 4
            cgrp = tidx // 4  # this lane group's column quad
            cbase = c0 + 4 * cgrp
            nblk = valid_blocks - 4 * ll  # payload blocks this lane owns (always even)
            q_col_pitch = M // 2
            wide = q_col_pitch % 32 == 0  # 32-byte alignment of a transposed row
            q_ptr = q_col_base + row_offset(cbase, q_col_pitch) + Int64(r00 // 2 + 32 * ll)
            s_ptr = s_col_base + row_offset(cbase, s_col_pitch) + Int64(r00 // BLOCK + 4 * ll)
            for q in cutlass.range_constexpr(4):
                slot = cgrp + 32 * q
                if cbase + q < K and nblk > 0:
                    d = []
                    for t in cutlass.range_constexpr(4):
                        lo, hi = unpack_i64_to_i32x2(
                            staging[slot * (self._STAGING_WORDS // 2) + 4 * ll + t]
                        )
                        d.append(Uint32(lo))
                        d.append(Uint32(hi))
                    if nblk >= 4:
                        if wide:
                            st_global_v8(q_ptr, d)  # one 32-byte sector
                        else:  # rows are only 16-byte aligned when M % 64 != 0
                            st_global_v4(q_ptr, d[0:4])
                            st_global_v4(q_ptr + Int64(16), d[4:8])
                    else:  # tail slab: this lane owns exactly two blocks
                        st_global_v4(q_ptr, d[0:4])
                    scale_word = staging_scales[slot * (self._STAGING_SCALE_BYTES // 4) + ll]
                    if nblk >= 4:
                        st_global_u32(s_ptr, scale_word)
                    else:
                        st_global_u16(s_ptr, scale_word)
                q_ptr = q_ptr + Int64(q_col_pitch)
                s_ptr = s_ptr + Int64(s_col_pitch)

    @cute.jit
    def _slab(
        self,
        r0: Int32,
        sub: Int32,
        M: Int32,
        x_base: Int64,
        x_col: Int64,
        q_col_off: Int64,
        s_col_off: Int64,
        x_pitch: Int32,
        q_pitch: Int32,
        s_row_pitch: Int32,
        qa: Int32,
        rl: Int32,
        arow: Boolean,
        cp: Int32,
        mb: Int32,
        c: Int32,
        cok: Boolean,
        num_slabs: Int32,
        table: cute.Tensor,
        staging: cute.Tensor,
        staging_scales_u8: cute.Tensor,
        S_enc_row_div6: Float32,
        S_enc_col_div6: Float32,
        fold_row,
        fold_col,
        GROUP: cutlass.Constexpr,
        BAND_SAFE: cutlass.Constexpr,
        STOCHASTIC: cutlass.Constexpr,
    ):
        """Phases A and B for the slab of rows [r0, r0 + _SLAB_M)."""
        THREADS = self._THREADS
        SLAB_M = self._SLAB_M
        BLOCK = NVFP4_BLOCK_SCALING_SIZE
        RG = THREADS // (8 * GROUP)
        ROWS_A = SLAB_M // GROUP

        # Phase A: one rowwise block per thread per quarter slab. A full slab needs no per-row
        # bounds test.
        rbase = r0 + ROWS_A * qa + rl
        afull = arow and r0 + SLAB_M <= M
        rows = [rbase + RG * i for i in range(4)]
        v = []
        for i in cutlass.range_constexpr(4):
            r = rows[i]
            if cutlass.const_expr(BAND_SAFE):
                if rbase >= M:
                    r = r0 + RG * i
            else:
                if r >= M:
                    r = r0
            v.append(ld_global_evict_last_v8(x_col + row_offset(r, x_pitch)))

        # Warm L2 for the next slab under SR only: RNE streams near its DRAM ceiling and a
        # lookahead competes with demand loads, while SR has the slack for it to land.
        if cutlass.const_expr(STOCHASTIC):
            if sub + 1 < num_slabs and r0 + 2 * SLAB_M <= M:
                for i in cutlass.range_constexpr(4):
                    prefetch_l2(x_col + row_offset(rows[i] + SLAB_M, x_pitch))

        # The Philox chain depends only on the row, never on the data, so it goes before the
        # amax trees that stall on the loads.
        rnd_row = [None] * 4
        if cutlass.const_expr(STOCHASTIC):
            for g in cutlass.range_constexpr(0, 4, 2):
                draws = block_philox_draws([Uint32(rows[g]), Uint32(rows[g + 1])], fold_row)
                rnd_row[g] = draws[0]
                rnd_row[g + 1] = draws[1]
        codes_a = [block_scale_code(v[i], S_enc_row_div6) for i in range(4)]
        coeffs_a = [table[Int32(codes_a[i])] for i in range(4)]
        coeff2_a = [pack_f32x2(c, c) for c in coeffs_a]
        for i in cutlass.range_constexpr(4):
            w0, w1 = pack_row_block(v[i], coeff2_a[i], rnd_row[i], STOCHASTIC)
            if afull or (rows[i] < M and arow):
                st_global_v2(q_col_off + row_offset(rows[i], q_pitch), w0, w1)
                st_global_u8(s_col_off + row_offset(rows[i], s_row_pitch), codes_a[i])

        # Phase B: the rows phase A just read are still in L2, so this column-major pass is a
        # cache re-read rather than a second DRAM stream.
        rb = r0 + mb * BLOCK
        bok = cok and rb + BLOCK <= M
        cc = c
        rr = rb
        if not bok:
            cc = Int32(0)
            rr = r0
        x_ptr = x_base + row_offset(rr, x_pitch) + Int64(cc) * 2
        raw = [[], []]
        for _ in cutlass.range_constexpr(BLOCK):
            lo, hi = ld_global_v2(x_ptr)
            raw[0].append(lo)
            raw[1].append(hi)
            x_ptr = x_ptr + Int64(x_pitch)

        word_offset = (sub * self._BLOCKS_PER_SLAB + mb) * 2
        byte_offset = sub * self._BLOCKS_PER_SLAB + mb
        # High word = (row block, column inside the quad); the quad base is in the folded low
        # word, so the four columns differ only in two bits.
        rnd = [None] * 4
        if cutlass.const_expr(STOCHASTIC):
            row_block = Uint32(rb) >> Uint32(4)
            for g in cutlass.range_constexpr(0, 4, 2):
                draws = block_philox_draws(
                    [
                        columnwise_sub_hi(row_block, Uint32(g)),
                        columnwise_sub_hi(row_block, Uint32(g + 1)),
                    ],
                    fold_col,
                )
                rnd[g] = draws[0]
                rnd[g + 1] = draws[1]
        # Both 16-row amax trees and all four coefficient lookups resolve before any column
        # packs, so the four shared loads are in flight together.
        codes_b = [None] * 4
        for p in cutlass.range_constexpr(2):  # one packed word holds a pair of adjacent columns
            pair_amax = abs_max_tree_bf16x2(raw[p]) & Uint32(0x7FFF7FFF)
            codes_b[2 * p] = cvt_f32_to_fp8e4m3_code(
                (pair_amax << Uint32(16)).bitcast(Float32) * S_enc_col_div6
            )
            codes_b[2 * p + 1] = cvt_f32_to_fp8e4m3_code(
                (pair_amax & Uint32(0xFFFF0000)).bitcast(Float32) * S_enc_col_div6
            )
        coeffs_b = [table[THREADS + Int32(codes_b[k])] for k in range(4)]
        coeff2_b = [pack_f32x2(c, c) for c in coeffs_b]
        for p in cutlass.range_constexpr(2):
            for q in cutlass.range_constexpr(2):
                cs = 2 * p + q
                w0, w1 = pack_column_block(raw[p], coeff2_b[cs], rnd[cs], q == 0, STOCHASTIC)
                slot = cs * 32 + cp  # one 8-byte staging store per column block
                staging[slot * (self._STAGING_WORDS // 2) + word_offset // 2] = pack_u32x2(w0, w1)
                staging_scales_u8[slot * self._STAGING_SCALE_BYTES + byte_offset] = cutlass.Uint8(
                    codes_b[cs]
                )
