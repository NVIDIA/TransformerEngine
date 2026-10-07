# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Register-resident CuTeDSL bidimensional MXFP8 cast, ported from cast_bidim.cuh."""

from typing import Optional

import cutlass
from cutlass import cute
from cutlass import Int32, Int64, Uint32, Uint8
from cuda.bindings.driver import CUstream  # pylint: disable=no-name-in-module

from transformer_engine.common.CuTeDSL.utils import (
    abs_max_x2_bf16,
    create_l2_policy,
)
from transformer_engine.common.CuTeDSL.utils_fp8 import mul_bf16x2x2_cvt_bf16x4_to_fp8x4
from .quantize_mxfp8_common import (
    MXFP8QuantizeConfig,
    MXFP8QuantizeKernelBase,
    _SmemAllocator,
    bf16_pair_magnitude,
    noop_flag_is_set,
)


@cute.jit
def bf16_mx_scale_reciprocal(magnitude: Uint32, MAX_EXPONENT: cutlass.Constexpr[int]) -> Uint32:
    """Exact BF16 reciprocal scale bits, including E8M0 zero, Inf and NaN cases."""
    rounded = (magnitude + Uint32(31)) & Uint32(0x7F80)
    if rounded < Uint32(MAX_EXPONENT << 7):  # pylint: disable=consider-using-max-builtin
        rounded = Uint32(MAX_EXPONENT << 7)
    reciprocal = Uint32((254 + MAX_EXPONENT) << 7) - rounded
    if magnitude >= Uint32(0x7F80):
        reciprocal = Uint32(0x0040)
        if magnitude != Uint32(0x7F80):
            reciprocal = Uint32(0x7FFF)
    return reciprocal


@cute.jit
def bf16_mx_scale_reciprocal_pair(pair: Int32, MAX_EXPONENT: cutlass.Constexpr[int]) -> Uint32:
    """Compute two reciprocal scales without borrowing across packed BF16 halves."""
    magnitudes = Uint32(pair) & Uint32(0x7FFF7FFF)
    rounded = (magnitudes + Uint32(0x001F001F)) & Uint32(0xFF80FF80)
    floor = (MAX_EXPONENT << 23) | (MAX_EXPONENT << 7)
    # Packed BF16 max accepts registers, so materialize the constant inside PTX.
    clamped = cute.arch.inline_ptx(
        "{ .reg.b32 floor_pair; "
        f"mov.b32 floor_pair, {floor}; "
        "max.xorsign.abs.bf16x2 {$w0}, {$r0}, floor_pair; }",
        write_only_types=[Int32],
        read_only_args=[rounded.bitcast(Int32)],
    )
    bias = ((254 + MAX_EXPONENT) << 23) | ((254 + MAX_EXPONENT) << 7)
    reciprocal = Uint32(bias) - Uint32(clamped)
    if ((magnitudes + Uint32(0x00800080)) & Uint32(0x80008000)) != Uint32(0):
        lo = bf16_mx_scale_reciprocal(magnitudes & Uint32(0x7FFF), MAX_EXPONENT)
        hi = bf16_mx_scale_reciprocal(magnitudes >> Uint32(16), MAX_EXPONENT)
        reciprocal = lo | (hi << Uint32(16))
    return reciprocal


@cute.jit
def bf16_mx_scale_bytes_pair(reciprocal: Uint32) -> Uint32:
    """Gather the E8M0 bytes of two packed BF16 reciprocals into the low half."""
    bytes_pair = (Uint32(0x01FE01FE) - ((reciprocal >> Uint32(7)) & Uint32(0x00FF00FF))) & Uint32(
        0x00FF00FF
    )
    return (bytes_pair & Uint32(0xFF)) | ((bytes_pair >> Uint32(8)) & Uint32(0xFF00))


@cute.jit
def quantize_bidimensional_mxfp8_g2r(
    gX_tile: cute.Tensor,
    gO_row_tile: cute.Tensor,
    gS_row_tile: cute.Tensor,
    gO_col_tile: cute.Tensor,
    gS_col_tile: cute.Tensor,
    cfg: cutlass.Constexpr[MXFP8QuantizeConfig],
    CTA_THREADS_Y: cutlass.Constexpr[int],
    CTA_THREADS_X: cutlass.Constexpr[int],
    ROWS_PER_THREAD: cutlass.Constexpr[int],
    ELEMENTS_PER_THREAD: cutlass.Constexpr[int],
    cache_policy: Int64,
):
    """Quantize one (TILE_ROWS, TILE_COLS) GMEM tile in both directions.

    TILE_ROWS = CTA_THREADS_Y * ROWS_PER_THREAD = 32, so every columnwise
    MXFP8 block is contained in the tile. Each warp owns ROWS_PER_THREAD
    consecutive rows, and each lane owns ELEMENTS_PER_THREAD adjacent columns.
    Load the tile once, then reuse its registers for both quantization passes.
    """
    TILE_ROWS = CTA_THREADS_Y * ROWS_PER_THREAD
    TILE_COLS = CTA_THREADS_X * ELEMENTS_PER_THREAD
    LANES_PER_MX_BLOCK = 32 // ELEMENTS_PER_THREAD
    COLUMN_PAIRS = TILE_COLS // 2
    COLUMN_PAIRS_PER_THREAD = ELEMENTS_PER_THREAD // 2
    PAIRS_PER_HALF = COLUMN_PAIRS_PER_THREAD // 2
    MAX_EXPONENT = 8 if cfg.FP8_DTYPE is cutlass.Float8E4M3FN else 15
    SCALE_DTYPE = cutlass.Float8E8M0FNU
    tidx, _, _ = cute.arch.thread_idx()
    warp, lane = tidx // CTA_THREADS_X, tidx % CTA_THREADS_X

    _, tv_payload = cute.make_layout_tv(
        thr_layout=cute.make_layout((CTA_THREADS_Y, CTA_THREADS_X), stride=(CTA_THREADS_X, 1)),
        val_layout=cute.make_layout(
            (ROWS_PER_THREAD, ELEMENTS_PER_THREAD), stride=(ELEMENTS_PER_THREAD, 1)
        ),
    )
    # Reshape each thread's values into (columns, rows), with contiguous columns.
    fragment_layout = cute.make_layout(
        (ELEMENTS_PER_THREAD, ROWS_PER_THREAD), stride=(1, ELEMENTS_PER_THREAD)
    )
    gX_thread = cute.composition(cute.composition(gX_tile, tv_payload)[tidx, None], fragment_layout)
    gO_row_thread = cute.composition(
        cute.composition(gO_row_tile, tv_payload)[tidx, None], fragment_layout
    )
    gO_col_thread = cute.composition(
        cute.composition(gO_col_tile, tv_payload)[tidx, None], fragment_layout
    )
    # Each group of lanes shares one rowwise scale for each of its rows.
    tv_row_scale = cute.make_layout(
        (
            (LANES_PER_MX_BLOCK, (CTA_THREADS_X // LANES_PER_MX_BLOCK, CTA_THREADS_Y)),
            ROWS_PER_THREAD,
        ),
        stride=((0, (TILE_ROWS, ROWS_PER_THREAD)), 1),
    )
    gS_row_thread = cute.composition(gS_row_tile, tv_row_scale)[tidx, None]
    # Warp 0 writes one scale per column, in each lane's original column order.
    gS_col_thread = cute.local_tile(gS_col_tile, (1, ELEMENTS_PER_THREAD), (0, lane))[0, None]

    # Keep the complete input fragment; reuse one row's output registers.
    rX = cute.make_rmem_tensor((ELEMENTS_PER_THREAD, ROWS_PER_THREAD), cfg.DTYPE)
    rO = cute.make_rmem_tensor((ELEMENTS_PER_THREAD,), cfg.FP8_DTYPE)
    rS_row = cute.make_rmem_tensor((1,), SCALE_DTYPE)
    rS_col = cute.make_rmem_tensor((ELEMENTS_PER_THREAD,), SCALE_DTYPE)
    load_atom = cute.make_copy_atom(
        cute.nvgpu.CopyG2ROp(),
        cfg.DTYPE,
        num_bits_per_copy=ELEMENTS_PER_THREAD * cfg.DTYPE.width,
        invariant=True,
    )
    store_atom = cute.make_copy_atom(
        cute.nvgpu.CopyR2GOp(),
        cfg.FP8_DTYPE,
        num_bits_per_copy=ELEMENTS_PER_THREAD * cfg.FP8_DTYPE.width,
    )
    row_scale_atom = cute.make_copy_atom(cute.nvgpu.CopyR2GOp(), SCALE_DTYPE, num_bits_per_copy=8)
    col_scale_atom = cute.make_copy_atom(
        cute.nvgpu.CopyR2GOp(), SCALE_DTYPE, num_bits_per_copy=ELEMENTS_PER_THREAD * 8
    )
    scale_and_pack = mul_bf16x2x2_cvt_bf16x4_to_fp8x4(cfg.FP8_DTYPE)
    rO_u32 = cute.recast_tensor(rO, Uint32)

    for row_idx in cutlass.range_constexpr(ROWS_PER_THREAD):
        cute.copy(load_atom, gX_thread[None, row_idx], rX[None, row_idx], cache_policy=cache_policy)

    @cute.jit
    def extract_row_amax(values_i32: cute.Tensor) -> Uint32:
        """Reduce a 32-column MXFP8 block across its two or four lane owners."""
        amax = values_i32[0]
        for pair_idx in cutlass.range_constexpr(1, cute.size(values_i32)):
            amax = abs_max_x2_bf16(amax, values_i32[pair_idx])
        for stage in cutlass.range_constexpr(LANES_PER_MX_BLOCK.bit_length() - 1):
            amax = abs_max_x2_bf16(amax, cute.arch.shuffle_sync_bfly(amax, 1 << stage))
        return bf16_pair_magnitude(amax)

    # Rowwise pass: lane shuffles close each 32-column reduction.
    for row_idx in cutlass.range_constexpr(ROWS_PER_THREAD):
        rX_row_i32 = cute.recast_tensor(rX[None, row_idx], Int32)
        reciprocal = bf16_mx_scale_reciprocal(extract_row_amax(rX_row_i32), MAX_EXPONENT)
        reciprocal_packed = reciprocal | (reciprocal << Uint32(16))
        for pack_idx in cutlass.range_constexpr(cute.size(rO_u32)):
            rO_u32[pack_idx] = scale_and_pack(
                rX_row_i32[2 * pack_idx],
                reciprocal_packed,
                rX_row_i32[2 * pack_idx + 1],
                reciprocal_packed,
            )
        cute.copy(store_atom, rO, gO_row_thread[None, row_idx], cache_policy=cache_policy)
        if lane % LANES_PER_MX_BLOCK == 0:
            rS_row[0] = Uint8(Uint32(254) - (reciprocal >> Uint32(7))).bitcast(SCALE_DTYPE)
            cute.copy(
                row_scale_atom,
                rS_row,
                cute.local_tile(gS_row_thread, (1,), (row_idx,)),
                cache_policy=cache_policy,
            )

    # Columnwise pass: first reduce each thread's rows into one maximum per column.
    rColumnAmax = cute.make_rmem_tensor((ELEMENTS_PER_THREAD,), cfg.DTYPE)
    rColumnScale = cute.make_rmem_tensor((ELEMENTS_PER_THREAD,), cfg.DTYPE)
    rColumnAmax_i32 = cute.recast_tensor(rColumnAmax, Int32)
    rColumnScale_u32 = cute.recast_tensor(rColumnScale, Uint32)
    rX_i32 = cute.recast_tensor(rX, Int32)
    for pair_idx in cutlass.range_constexpr(COLUMN_PAIRS_PER_THREAD):
        amax = rX_i32[pair_idx, 0]
        for row_idx in cutlass.range_constexpr(1, ROWS_PER_THREAD):
            amax = abs_max_x2_bf16(amax, rX_i32[pair_idx, row_idx])
        rColumnAmax_i32[pair_idx] = amax

    allocator = _SmemAllocator()
    sColumnAmax = allocator.allocate_tensor(
        cfg.DTYPE,
        cute.make_layout((CTA_THREADS_Y, TILE_COLS), stride=(TILE_COLS, 1)),
        byte_alignment=16,
    )
    sColumnScale = allocator.allocate_tensor(
        cfg.DTYPE, cute.make_layout((TILE_COLS,)), byte_alignment=16
    )
    sColumnAmax_i32 = cute.recast_tensor(sColumnAmax, Int32)
    sColumnScale_u32 = cute.recast_tensor(sColumnScale, Uint32)
    # Group each lane's column pairs into two halves. Placing the first halves
    # of all lanes before the second halves preserves CUDA's conflict-free
    # vectorized shared-memory accesses, without calculating pointer offsets.
    column_pair_layout = cute.make_layout(
        ((PAIRS_PER_HALF, 2), CTA_THREADS_X),
        stride=((1, COLUMN_PAIRS // 2), PAIRS_PER_HALF),
    )
    sColumnAmax_thread = cute.composition(sColumnAmax_i32, (None, column_pair_layout))[
        warp, (None, lane)
    ]
    sColumnScale_thread = cute.composition(sColumnScale_u32, column_pair_layout)[None, lane]
    for half_idx in cutlass.range_constexpr(2):
        cute.autovec_copy(
            cute.local_tile(rColumnAmax_i32, (PAIRS_PER_HALF,), (half_idx,)),
            cute.local_tile(sColumnAmax_thread, (PAIRS_PER_HALF,), (half_idx,)),
        )
    cute.arch.sync_threads()

    # One thread reduces each column pair across all warps (all 32 tile rows).
    if tidx < COLUMN_PAIRS:
        amax = sColumnAmax_i32[0, tidx]
        for warp_idx in cutlass.range_constexpr(1, CTA_THREADS_Y):
            amax = abs_max_x2_bf16(amax, sColumnAmax_i32[warp_idx, tidx])
        sColumnScale_u32[tidx] = bf16_mx_scale_reciprocal_pair(amax, MAX_EXPONENT)
    cute.arch.sync_threads()

    # The same layout returns the reciprocals to each lane's original columns.
    for half_idx in cutlass.range_constexpr(2):
        cute.autovec_copy(
            cute.local_tile(sColumnScale_thread, (PAIRS_PER_HALF,), (half_idx,)),
            cute.local_tile(rColumnScale_u32, (PAIRS_PER_HALF,), (half_idx,)),
        )
    if warp == 0:
        rS_col_u32 = cute.recast_tensor(rS_col, Uint32)
        for pack_idx in cutlass.range_constexpr(cute.size(rS_col_u32)):
            rS_col_u32[pack_idx] = bf16_mx_scale_bytes_pair(rColumnScale_u32[2 * pack_idx]) | (
                bf16_mx_scale_bytes_pair(rColumnScale_u32[2 * pack_idx + 1]) << Uint32(16)
            )
        cute.copy(col_scale_atom, rS_col, gS_col_thread, cache_policy=cache_policy)

    for row_idx in cutlass.range_constexpr(ROWS_PER_THREAD):
        rX_row_i32 = cute.recast_tensor(rX[None, row_idx], Int32)
        for pack_idx in cutlass.range_constexpr(cute.size(rO_u32)):
            rO_u32[pack_idx] = scale_and_pack(
                rX_row_i32[2 * pack_idx],
                rColumnScale_u32[2 * pack_idx],
                rX_row_i32[2 * pack_idx + 1],
                rColumnScale_u32[2 * pack_idx + 1],
            )
        cute.copy(store_atom, rO, gO_col_thread[None, row_idx], cache_policy=cache_policy)


class MXFP8QuantizeRegisterBidimensionalKernel(MXFP8QuantizeKernelBase):
    """Load one CTA tile into registers and quantize it along both axes.

    TILE_ROWS = _CTA_THREADS_Y * _ROWS_PER_THREAD = 32.
    TILE_COLS = _CTA_THREADS_X * ELEMENTS_PER_THREAD = 256 or 512.
    The 32-row tile contains complete columnwise MXFP8 blocks; shared memory
    combines the partial column maxima from the eight warps. Requires aligned
    BF16 input, M divisible by 32, N divisible by 256 and plain scale layouts.
    """

    _CTA_THREADS_Y = 8
    _CTA_THREADS_X = 32
    _ROWS_PER_THREAD = 4
    _ELEMENTS_PER_THREAD = 16

    def __init__(self, cfg: MXFP8QuantizeConfig):
        self.cfg = cfg

    @cute.jit
    def __call__(
        self,
        mX: cute.Tensor,
        mO_row: Optional[cute.Tensor],
        mS_row: Optional[cute.Tensor],
        mO_col: Optional[cute.Tensor],
        mS_col: Optional[cute.Tensor],
        mAmax: Optional[cute.Tensor],
        mNoop: cute.Pointer,
        mDActInput: Optional[cute.Tensor],
        mWorkspace: Optional[cute.Tensor],
        stream: CUstream,
    ):
        M, N = mX.shape
        TILE_ROWS = self._CTA_THREADS_Y * self._ROWS_PER_THREAD
        WIDE_TILE_COLS = self._CTA_THREADS_X * self._ELEMENTS_PER_THREAD
        if N % WIDE_TILE_COLS == 0:
            if Int64(M // TILE_ROWS) * (N // WIDE_TILE_COLS) >= 4096:
                self._launch(
                    mX,
                    mO_row,
                    mS_row,
                    mO_col,
                    mS_col,
                    mNoop,
                    stream,
                    CTA_THREADS_Y=self._CTA_THREADS_Y,
                    CTA_THREADS_X=self._CTA_THREADS_X,
                    ROWS_PER_THREAD=self._ROWS_PER_THREAD,
                    ELEMENTS_PER_THREAD=self._ELEMENTS_PER_THREAD,
                    CLUSTER_WIDTH=4,
                )
            else:
                self._launch(
                    mX,
                    mO_row,
                    mS_row,
                    mO_col,
                    mS_col,
                    mNoop,
                    stream,
                    CTA_THREADS_Y=self._CTA_THREADS_Y,
                    CTA_THREADS_X=self._CTA_THREADS_X,
                    ROWS_PER_THREAD=self._ROWS_PER_THREAD,
                    ELEMENTS_PER_THREAD=self._ELEMENTS_PER_THREAD,
                    CLUSTER_WIDTH=1,
                )
        else:
            self._launch(
                mX,
                mO_row,
                mS_row,
                mO_col,
                mS_col,
                mNoop,
                stream,
                CTA_THREADS_Y=self._CTA_THREADS_Y,
                CTA_THREADS_X=self._CTA_THREADS_X,
                ROWS_PER_THREAD=self._ROWS_PER_THREAD,
                ELEMENTS_PER_THREAD=self._ELEMENTS_PER_THREAD // 2,
                CLUSTER_WIDTH=8,
            )

    @cute.jit
    def _launch(
        self,
        mX: cute.Tensor,
        mO_row: cute.Tensor,
        mS_row: cute.Tensor,
        mO_col: cute.Tensor,
        mS_col: cute.Tensor,
        mNoop: cute.Pointer,
        stream: CUstream,
        CTA_THREADS_Y: cutlass.Constexpr[int],
        CTA_THREADS_X: cutlass.Constexpr[int],
        ROWS_PER_THREAD: cutlass.Constexpr[int],
        ELEMENTS_PER_THREAD: cutlass.Constexpr[int],
        CLUSTER_WIDTH: cutlass.Constexpr[int],
    ):
        assert CTA_THREADS_X == 32
        assert CTA_THREADS_Y * ROWS_PER_THREAD == 32
        assert ELEMENTS_PER_THREAD in (8, 16)
        M, N = mX.shape
        TILE_ROWS = CTA_THREADS_Y * ROWS_PER_THREAD
        TILE_COLS = CTA_THREADS_X * ELEMENTS_PER_THREAD
        THREADS_PER_CTA = CTA_THREADS_Y * CTA_THREADS_X
        grid_cols = N // TILE_COLS
        grid_rows = M // TILE_ROWS
        cluster_cols = Int32(CLUSTER_WIDTH)
        while cluster_cols > 1 and grid_cols % cluster_cols != 0:
            cluster_cols -= 1
        # Keep column CTAs on grid.x so clusters read adjacent column tiles.
        grid = [grid_cols, grid_rows]
        block = [THREADS_PER_CTA]
        cluster = [cluster_cols, 1, 1]
        # Preserve CUDA's compile-time strides for the two common column counts.
        if N == 2048:
            self.kernel(
                mX,
                mO_row,
                mS_row,
                mO_col,
                mS_col,
                mNoop,
                CTA_THREADS_Y=CTA_THREADS_Y,
                CTA_THREADS_X=CTA_THREADS_X,
                ROWS_PER_THREAD=ROWS_PER_THREAD,
                ELEMENTS_PER_THREAD=ELEMENTS_PER_THREAD,
                N_CONST=2048,
            ).launch(grid=grid, block=block, cluster=cluster, min_blocks_per_mp=4, stream=stream)
        elif N == 4096:
            self.kernel(
                mX,
                mO_row,
                mS_row,
                mO_col,
                mS_col,
                mNoop,
                CTA_THREADS_Y=CTA_THREADS_Y,
                CTA_THREADS_X=CTA_THREADS_X,
                ROWS_PER_THREAD=ROWS_PER_THREAD,
                ELEMENTS_PER_THREAD=ELEMENTS_PER_THREAD,
                N_CONST=4096,
            ).launch(grid=grid, block=block, cluster=cluster, min_blocks_per_mp=4, stream=stream)
        else:
            self.kernel(
                mX,
                mO_row,
                mS_row,
                mO_col,
                mS_col,
                mNoop,
                CTA_THREADS_Y=CTA_THREADS_Y,
                CTA_THREADS_X=CTA_THREADS_X,
                ROWS_PER_THREAD=ROWS_PER_THREAD,
                ELEMENTS_PER_THREAD=ELEMENTS_PER_THREAD,
                N_CONST=0,
            ).launch(grid=grid, block=block, cluster=cluster, min_blocks_per_mp=4, stream=stream)

    @cute.kernel
    def kernel(
        self,
        mX: cute.Tensor,
        mO_row: cute.Tensor,
        mS_row: cute.Tensor,
        mO_col: cute.Tensor,
        mS_col: cute.Tensor,
        mNoop: cute.Pointer,
        CTA_THREADS_Y: cutlass.Constexpr[int],
        CTA_THREADS_X: cutlass.Constexpr[int],
        ROWS_PER_THREAD: cutlass.Constexpr[int],
        ELEMENTS_PER_THREAD: cutlass.Constexpr[int],
        N_CONST: cutlass.Constexpr[int],
    ):
        """Check noop before constructing or accessing any CTA tile."""
        if not noop_flag_is_set(mNoop):
            M, N = mX.shape
            if cutlass.const_expr(N_CONST != 0):
                N = N_CONST
                payload_layout = cute.make_layout((M, N), stride=(N, 1))
                gX = cute.make_tensor(mX.iterator.align(32), payload_layout)
                gO_row = cute.make_tensor(mO_row.iterator, payload_layout)
                gO_col = cute.make_tensor(mO_col.iterator, payload_layout)
                gS_row = cute.make_tensor(
                    mS_row.iterator,
                    cute.make_layout((cute.size(mS_row, mode=[0]), N // 32), stride=(N // 32, 1)),
                )
                gS_col = cute.make_tensor(
                    mS_col.iterator.align(16),
                    cute.make_layout((cute.size(mS_col, mode=[0]), N), stride=(N, 1)),
                )
            else:
                # The dispatcher checks these alignments before choosing this kernel.
                gX = cute.make_tensor(mX.iterator.align(32), mX.layout)
                gO_row = mO_row
                gO_col = mO_col
                gS_row = mS_row
                gS_col = cute.make_tensor(mS_col.iterator.align(16), mS_col.layout)
            cta_col, cta_row, _ = cute.arch.block_idx()
            cta_coord = (cta_row, cta_col)
            TILE_ROWS = CTA_THREADS_Y * ROWS_PER_THREAD
            TILE_COLS = CTA_THREADS_X * ELEMENTS_PER_THREAD
            TILER = (TILE_ROWS, TILE_COLS)
            ROW_SCALE_TILER = (TILE_ROWS, TILE_COLS // 32)
            COL_SCALE_TILER = (TILE_ROWS // 32, TILE_COLS)
            gX_tile = cute.local_tile(gX, TILER, cta_coord)
            gO_row_tile = cute.local_tile(gO_row, TILER, cta_coord)
            gO_col_tile = cute.local_tile(gO_col, TILER, cta_coord)
            gS_row_tile = cute.local_tile(gS_row, ROW_SCALE_TILER, cta_coord)
            gS_col_tile = cute.local_tile(gS_col, COL_SCALE_TILER, cta_coord)
            cache_policy = create_l2_policy(True)
            quantize_bidimensional_mxfp8_g2r(
                gX_tile,
                gO_row_tile,
                gS_row_tile,
                gO_col_tile,
                gS_col_tile,
                self.cfg,
                CTA_THREADS_Y=CTA_THREADS_Y,
                CTA_THREADS_X=CTA_THREADS_X,
                ROWS_PER_THREAD=ROWS_PER_THREAD,
                ELEMENTS_PER_THREAD=ELEMENTS_PER_THREAD,
                cache_policy=cache_policy,
            )
