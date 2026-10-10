# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Specialized CuTeDSL bidimensional MXFP8 cast with staged TMA I/O."""

# Local @cute.struct classes are shared-memory storage descriptors.
# pylint: disable=missing-class-docstring

from typing import Optional, Type

import cutlass
from cutlass import cute
from cutlass import pipeline
from cutlass import Float32, Float8E8M0FNU, Int32, Int64, Uint32
from cuda.bindings.driver import CUstream  # pylint: disable=no-name-in-module

from transformer_engine.common.CuTeDSL.utils import (
    is_packed16,
    abs_max_x2_bf16,
    abs_max_x2_f16,
    x2_lo_to_f32_bf16,
    x2_lo_to_f32_f16,
    x2_hi_to_f32_bf16,
    x2_hi_to_f32_f16,
    fabs_f32,
    exp2f_rcp,
    pack_f32x2,
)
from transformer_engine.common.CuTeDSL.utils_fp8 import (
    cvt_f32_to_fp8e8m0fnu,
    mul_f32x2_cvt_f32x4_to_fp8x4,
    mul_f32x4_cvt_f32x4_to_fp8x4,
)
from .quantize_mxfp8_common import (
    CUTEDSL_DEBUG_LOGGING,
    MXFP8QuantizeConfig,
    MXFP8QuantizeKernelBase,
    MXFP8_BLOCK_SCALING_SIZE,
    THREADS_PER_WARP,
    derive_swizzled_scale_layout,
    noop_flag_is_set,
)


@cute.jit
def quantize_bidimensional_mxfp8_swizzled(
    sX_tile: cute.Tensor,  # 32x64 input tile (already sliced to this stage), Sw<3,4,3> swizzled
    sO_row_tile: cute.Tensor,  # 32x64 rowwise-output tile
    sO_col_tile: cute.Tensor,  # 32x64 colwise-output tile
    sS_row_tile: cute.Tensor,  # (32, 2) smem rowwise-scale staging tile (flushed in the epilogue)
    sS_col_tile: cute.Tensor,  # (1, 64) smem colwise-scale staging tile (flushed in the epilogue)
    sColReduce_warp: cute.Tensor,  # (32,) SMEM fp32 columnwise scale reduction buffer for this warp
    WARPS_PER_CTA: cutlass.Constexpr[int],
    MAX_NORM_RCP: cutlass.Constexpr[float],
    DTYPE: cutlass.Constexpr[Type[cutlass.Numeric]],
    FP8_DTYPE: cutlass.Constexpr[Type[cutlass.Numeric]],
):
    """Quantize a pre-sliced 32x64 tile -> both rowwise and colwise MXFP8. Elements are
    addressed through TV layouts: a warp = 32 lanes = 32 rows (lane == row), and each
    thread owns one 32-col row segment (its 32-col block) and one output column.

    No bounds check: the caller only loops over valid tiles (`num_tiles`), and
    M % 32 == 0 means a tile can only be partial along N. The partial tile's
    out-of-bounds warp still runs, but harmlessly -- it reads TMA zero-filled smem,
    its output columns are masked by the caller's TMA store, and its scale writes
    (rowwise and colwise alike) go to staging slots whose flush targets are past-N
    padding columns of the respective scale tensors.
    """
    mul_cvt4 = mul_f32x2_cvt_f32x4_to_fp8x4(FP8_DTYPE)
    mul_cvt4_elemwise = mul_f32x4_cvt_f32x4_to_fp8x4(FP8_DTYPE)

    _, tv_layout = cute.make_layout_tv(
        thr_layout=cute.make_layout(((MXFP8_BLOCK_SCALING_SIZE, 1), WARPS_PER_CTA)),  # ((32, 1) 2)
        val_layout=cute.make_layout((1, MXFP8_BLOCK_SCALING_SIZE)),
    )
    _, tv_layout_rowwise_scale = cute.make_layout_tv(
        thr_layout=cute.make_layout(((MXFP8_BLOCK_SCALING_SIZE, 1), WARPS_PER_CTA)),  # ((32, 1) 2)
        val_layout=cute.make_layout((1, 1)),
    )
    _, tv_layout_colwise_scale = cute.make_layout_tv(
        thr_layout=cute.make_layout((1, (MXFP8_BLOCK_SCALING_SIZE, WARPS_PER_CTA))),  # (1, (32, 2))
        val_layout=cute.make_layout((1, 1)),
    )

    tidx, _, _ = cute.arch.thread_idx()
    lane = tidx % MXFP8_BLOCK_SCALING_SIZE

    # Each composed [tidx, None] slice is 1-D (the value mode is flattened): the data
    # slices are size 32 (this thread's row segment), the scale slices size 1.
    tXsX = cute.composition(sX_tile, tv_layout)[tidx, None]  # (32,) input row segment
    tXsO_row = cute.composition(sO_row_tile, tv_layout)[tidx, None]  # (32,) rowwise out
    tXsO_col = cute.composition(sO_col_tile, tv_layout)[tidx, None]  # (32,) colwise out
    tSsS_row_tile = cute.composition(sS_row_tile, tv_layout_rowwise_scale)[tidx, None]  # (1,)
    tSsS_col_tile = cute.composition(sS_col_tile, tv_layout_colwise_scale)[tidx, None]  # (1,)

    rO_row = cute.make_rmem_tensor(MXFP8_BLOCK_SCALING_SIZE, FP8_DTYPE)
    rO_col = cute.make_rmem_tensor(MXFP8_BLOCK_SCALING_SIZE, FP8_DTYPE)
    rO_row_u32 = cute.make_tensor(
        cute.recast_ptr(rO_row.iterator, dtype=Uint32),
        cute.make_layout((MXFP8_BLOCK_SCALING_SIZE // 4,), stride=(1,)),
    )
    rO_col_u32 = cute.make_tensor(
        cute.recast_ptr(rO_col.iterator, dtype=Uint32),
        cute.make_layout((MXFP8_BLOCK_SCALING_SIZE // 4,), stride=(1,)),
    )
    sColReduce = cute.make_tensor(
        sColReduce_warp.iterator,
        cute.make_layout((MXFP8_BLOCK_SCALING_SIZE // 4, 4), stride=(4, 1)),
    )

    if cutlass.const_expr(is_packed16(DTYPE)):
        # If the input is bf16 / fp16, take this fast path and process 2 elements at a time in a packed i32
        abs_max_x2 = abs_max_x2_f16 if DTYPE is cutlass.Float16 else abs_max_x2_bf16
        x2_lo_to_f32 = x2_lo_to_f32_f16 if DTYPE is cutlass.Float16 else x2_lo_to_f32_bf16
        x2_hi_to_f32 = x2_hi_to_f32_f16 if DTYPE is cutlass.Float16 else x2_hi_to_f32_bf16
        rX = cute.make_rmem_tensor(MXFP8_BLOCK_SCALING_SIZE, DTYPE)
        # Do a vectorized load from SMEM to RMEM and unswizzle in the meantime.
        cute.autovec_copy(tXsX, rX)
        rX_2x = cute.make_tensor(
            cute.recast_ptr(rX.iterator, dtype=Int32),
            cute.make_layout((MXFP8_BLOCK_SCALING_SIZE // 2,), stride=(1,)),
        )
        sColReduce_2x = cute.make_tensor(
            cute.recast_ptr(sColReduce.iterator, dtype=Int64),
            cute.make_layout((MXFP8_BLOCK_SCALING_SIZE // 2,), stride=(1,)),
        )

        row_amax2 = rX_2x[0]
        for i in cutlass.range_constexpr(MXFP8_BLOCK_SCALING_SIZE // 2):
            pair = rX_2x[i]
            # Skip the first iteration because we assigned rX_2x[0] to row_amax2 already
            if cutlass.const_expr(i > 0):
                row_amax2 = abs_max_x2(row_amax2, pair)
            a_lo = fabs_f32(x2_lo_to_f32(pair))
            a_hi = fabs_f32(x2_hi_to_f32(pair))
            col_lo = cute.arch.warp_redux_sync(a_lo, kind="fmax")
            col_hi = cute.arch.warp_redux_sync(a_hi, kind="fmax")
            with cute.arch.elect_one():
                sColReduce_2x[i] = pack_f32x2(col_lo, col_hi)

        # Compute the rowwise scale factor
        row_amax = cute.arch.fmax(
            fabs_f32(x2_lo_to_f32(row_amax2)), fabs_f32(x2_hi_to_f32(row_amax2))
        )
        row_exp = cvt_f32_to_fp8e8m0fnu(row_amax * MAX_NORM_RCP)
        row_inv = exp2f_rcp(row_exp)
        tSsS_row_tile[0] = row_exp
        cute.arch.sync_warp()

        # Compute the colwise scale factor (only handle the one that belongs to this thread / lane)
        col_exp = cvt_f32_to_fp8e8m0fnu(sColReduce_warp[lane] * MAX_NORM_RCP)
        tSsS_col_tile[0] = col_exp
        sColReduce_warp[lane] = exp2f_rcp(col_exp)
        cute.arch.sync_warp()

        row_scale_2x = pack_f32x2(row_inv, row_inv)
        # Vectorized multiply-and-convert: 4 f32 → 4 fp8
        for j in cutlass.range_constexpr(MXFP8_BLOCK_SCALING_SIZE // 4):
            # Vectorized load for 4 columnwise scale factors from SMEM to RMEM
            col_inv4 = cute.make_rmem_tensor(4, Float32)
            cute.autovec_copy(sColReduce[j, None], col_inv4)  # LDS.128, warp-broadcast
            p01 = rX_2x[2 * j]
            p23 = rX_2x[2 * j + 1]
            f0 = x2_lo_to_f32(p01)
            f1 = x2_hi_to_f32(p01)
            f2 = x2_lo_to_f32(p23)
            f3 = x2_hi_to_f32(p23)
            # For rowwise quantized values, they use the same scale for all 4 elements,
            # so we can just pass two row_inv to mul_cvt4 to apply it to all 4 elements at once.
            rO_row_u32[j] = mul_cvt4(f0, f1, f2, f3, row_scale_2x)
            # For columnwise quantized values, each element has its own scale; the
            # elementwise variant fuses the per-element multiply into the cvt sequence.
            rO_col_u32[j] = mul_cvt4_elemwise(
                f0, f1, f2, f3, col_inv4[0], col_inv4[1], col_inv4[2], col_inv4[3]
            )
        cute.autovec_copy(rO_row, tXsO_row)
        cute.autovec_copy(rO_col, tXsO_col)
    else:
        # If input is fp32, take this slow path and process it normally without packing
        rX = cute.make_rmem_tensor(MXFP8_BLOCK_SCALING_SIZE, Float32)
        cute.autovec_copy(tXsX, rX)

        row_amax = Float32(0.0)
        for c in cutlass.range_constexpr(MXFP8_BLOCK_SCALING_SIZE):
            a = fabs_f32(rX[c])
            row_amax = cute.arch.fmax(row_amax, a)
            col_amax = cute.arch.warp_redux_sync(a, kind="fmax")
            with cute.arch.elect_one():
                sColReduce_warp[c] = col_amax

        # Compute the rowwise scale factor
        row_exp = cvt_f32_to_fp8e8m0fnu(row_amax * MAX_NORM_RCP)
        row_inv = exp2f_rcp(row_exp)
        tSsS_row_tile[0] = row_exp  # rowwise scale (this row-block, staged in smem)
        cute.arch.sync_warp()

        # Compute the colwise scale factor (only handle the one that belongs to this thread / lane)
        col_exp = cvt_f32_to_fp8e8m0fnu(sColReduce_warp[lane] * MAX_NORM_RCP)
        tSsS_col_tile[0] = col_exp  # colwise scale (this thread's column)
        sColReduce_warp[lane] = exp2f_rcp(col_exp)
        cute.arch.sync_warp()

        row_scale_2x = pack_f32x2(row_inv, row_inv)
        # Vectorized multiply-and-convert: 4 f32 → 4 fp8
        for j in cutlass.range_constexpr(MXFP8_BLOCK_SCALING_SIZE // 4):
            # Vectorized load for 4 columnwise scale factors from SMEM to RMEM
            col_inv4 = cute.make_rmem_tensor(4, Float32)
            cute.autovec_copy(sColReduce[j, None], col_inv4)
            offset = 4 * j
            # For rowwise quantized values, they use the same scale for all 4 elements,
            # so we can just pass two row_inv to mul_cvt4 to apply it to all 4 elements at once.
            rO_row_u32[j] = mul_cvt4(
                rX[offset], rX[offset + 1], rX[offset + 2], rX[offset + 3], row_scale_2x
            )
            # For columnwise quantized values, each element has its own scale; the
            # elementwise variant fuses the per-element multiply into the cvt sequence.
            rO_col_u32[j] = mul_cvt4_elemwise(
                rX[offset],
                rX[offset + 1],
                rX[offset + 2],
                rX[offset + 3],
                col_inv4[0],
                col_inv4[1],
                col_inv4[2],
                col_inv4[3],
            )
        cute.autovec_copy(rO_row, tXsO_row)
        cute.autovec_copy(rO_col, tXsO_col)


class MXFP8QuantizeSpecializedBidimensionalKernel(MXFP8QuantizeKernelBase):
    """Specialized cast-only rowwise+colwise MXFP8 kernel (swizzled TMA, one 32x32 tile per warp)."""

    _WARPS_PER_CTA = 2
    _THREADS_PER_CTA = _WARPS_PER_CTA * THREADS_PER_WARP
    # A warp handles a 32x32 tile
    _WARP_ROWS = MXFP8_BLOCK_SCALING_SIZE
    _WARP_COLS = MXFP8_BLOCK_SCALING_SIZE
    # A CTA handles a tile consisted two 32x32 warp subtiles side-by-side at a time, each handled by a warp
    _TILE_ROWS = _WARP_ROWS
    _TILE_COLS = _WARP_COLS * _WARPS_PER_CTA
    _NUM_STAGES = 2
    _NUM_TILES_X = 4
    _NUM_TILES_Y = 1
    _NUM_TILES = _NUM_TILES_X * _NUM_TILES_Y

    # Rows and columns for the rowwise / columnwise scale factor tensor
    _SCALE_ROWS = _TILE_ROWS // MXFP8_BLOCK_SCALING_SIZE
    _SCALE_COLS = _TILE_COLS // MXFP8_BLOCK_SCALING_SIZE

    def __init__(self, cfg: MXFP8QuantizeConfig):
        self.cfg = cfg
        if cfg.WITH_GEMM_SWIZZLED_SCALES:
            self._NUM_TILES_X = 2
            self._NUM_TILES_Y = 2
            self._NUM_TILES = self._NUM_TILES_X * self._NUM_TILES_Y

    @cute.jit
    def __call__(
        self,
        mX: cute.Tensor,
        mO_row: Optional[cute.Tensor],
        mS_row: Optional[cute.Tensor],
        mO_col: Optional[cute.Tensor],
        mS_col: Optional[cute.Tensor],
        mAmax: Optional[cute.Tensor],
        mNoop: cute.Pointer,  # f32 cast_noop flag; may be null, checked on device
        mDActInput: Optional[cute.Tensor],
        mWorkspace: Optional[cute.Tensor],
        stream: CUstream,
    ):
        if cutlass.const_expr(CUTEDSL_DEBUG_LOGGING):
            cute.printf(
                "[CuTeDSL] MXFP8QuantizeSpecializedBidimensionalKernel.__call__() with config:"
                f" {self.cfg}\n"
            )
        M = mX.shape[0]
        N = mX.shape[1]

        if cutlass.const_expr(self.cfg.WITH_GEMM_SWIZZLED_SCALES):
            mS_row, mS_col = derive_swizzled_scale_layout(
                M, N, self.cfg.ROWWISE, self.cfg.COLWISE, mS_row, mS_col
            )

        smem_tile_layout = cute.make_ordered_layout(
            (self._TILE_ROWS, self._TILE_COLS), order=(1, 0)
        )
        cta_tiler = (self._TILE_ROWS, self._TILE_COLS)
        # Apply 128B input Swizzle<3,4,3> to input tiles
        in_smem_layout = cute.make_composed_layout(cute.make_swizzle(3, 4, 3), 0, smem_tile_layout)
        # Apply 64B output swizzle<2,4,3> for output tiles
        out_smem_layout = cute.make_composed_layout(cute.make_swizzle(2, 4, 3), 0, smem_tile_layout)
        op_load = cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp()
        op_store = cute.nvgpu.cpasync.CopyBulkTensorTileS2GOp()

        # Input TMA atom
        tma_atom, tma_src = cute.nvgpu.cpasync.make_tiled_tma_atom(
            op_load,
            mX,
            in_smem_layout,
            cta_tiler,
            num_multicast=1,
        )
        # Rowwise output TMA atoms
        tma_atom_out_row, tma_dst_out_row = cute.nvgpu.cpasync.make_tiled_tma_atom(
            op_store,
            mO_row,
            out_smem_layout,
            cta_tiler,
            num_multicast=1,
        )
        # Colwise output TMA atoms
        tma_atom_out_col, tma_dst_out_col = cute.nvgpu.cpasync.make_tiled_tma_atom(
            op_store,
            mO_col,
            out_smem_layout,
            cta_tiler,
            num_multicast=1,
        )

        grid = [
            # Each CTA covers a (TILE_ROWS * NUM_TILES_Y, TILE_COLS * NUM_TILES_X) area.
            cute.ceil_div(Int32(N), self._TILE_COLS * self._NUM_TILES_X),
            cute.ceil_div(M, self._TILE_ROWS * self._NUM_TILES_Y),
        ]
        block = [self._THREADS_PER_CTA]
        self.kernel(
            mX,
            mS_row,
            mS_col,
            mNoop,
            mX.element_type,
            tma_atom,
            tma_src,
            tma_atom_out_row,
            tma_dst_out_row,
            tma_atom_out_col,
            tma_dst_out_col,
        ).launch(grid=grid, block=block, stream=stream)

    @cute.kernel
    def kernel(
        self,
        mX: cute.Tensor,
        mS_row: cute.Tensor,
        mS_col: cute.Tensor,
        mNoop: cute.Pointer,
        dtype: cutlass.Constexpr[Type[cutlass.Numeric]],
        tma_atom: cute.CopyAtom,
        tma_src: cute.Tensor,
        tma_atom_out_row: cute.CopyAtom,
        tma_dst_out_row: cute.Tensor,
        tma_atom_out_col: cute.CopyAtom,
        tma_dst_out_col: cute.Tensor,
    ):
        """Device entry: no-op the CTA when the noop flag is set, else run the quantize main loop."""
        if not noop_flag_is_set(mNoop):
            self._kernel_main(
                mX,
                mS_row,
                mS_col,
                dtype,
                tma_atom,
                tma_src,
                tma_atom_out_row,
                tma_dst_out_row,
                tma_atom_out_col,
                tma_dst_out_col,
            )

    @cute.jit
    def _kernel_main(
        self,
        mX: cute.Tensor,
        mS_row: cute.Tensor,
        mS_col: cute.Tensor,
        dtype: cutlass.Constexpr[Type[cutlass.Numeric]],
        tma_atom: cute.CopyAtom,
        tma_src: cute.Tensor,
        tma_atom_out_row: cute.CopyAtom,
        tma_dst_out_row: cute.Tensor,
        tma_atom_out_col: cute.CopyAtom,
        tma_dst_out_col: cute.Tensor,
    ):
        """Device entry for the specialized bidimensional (rowwise+colwise) cast kernel."""

        bidx, bidy, _ = cute.arch.block_idx()

        M = mX.shape[0]
        N = mX.shape[1]

        # The CTA owns a NUM_TILES_Y x NUM_TILES_X arrangement of tiles, starting here.
        tile_x_base = bidx * self._NUM_TILES_X
        tile_y_base = bidy * self._NUM_TILES_Y

        num_tiles_x = cutlass.min(
            self._NUM_TILES_X,
            # Valid 64-col tiles remaining from this CTA's start column (bidx * span).
            cute.ceil_div(Int32(N) - bidx * self._TILE_COLS * self._NUM_TILES_X, self._TILE_COLS),
        )
        num_tiles_y = cutlass.min(
            self._NUM_TILES_Y,
            # Valid 32-row tiles remaining from this CTA's start row (bidy * span).
            cute.ceil_div(Int32(M) - bidy * self._TILE_ROWS * self._NUM_TILES_Y, self._TILE_ROWS),
        )
        num_tiles = num_tiles_x * num_tiles_y

        FP8_DTYPE = self.cfg.FP8_DTYPE

        @cute.struct
        class SharedStorage:
            mbar_storage: cute.struct.MemRange[
                cute.Int64, 2 * self._NUM_STAGES
            ]  # (full,empty) per stage
            sX: cute.struct.Align[
                cute.struct.MemRange[dtype, self._TILE_ROWS * self._TILE_COLS * self._NUM_STAGES],
                128,
            ]
            sO_row: cute.struct.Align[
                cute.struct.MemRange[
                    FP8_DTYPE, self._TILE_ROWS * self._TILE_COLS * self._NUM_STAGES
                ],
                128,
            ]
            sO_col: cute.struct.Align[
                cute.struct.MemRange[
                    FP8_DTYPE, self._TILE_ROWS * self._TILE_COLS * self._NUM_STAGES
                ],
                128,
            ]
            # Per-warp colwise-reduce scratchpad
            sColReduce: cute.struct.Align[
                cute.struct.MemRange[Float32, THREADS_PER_WARP * self._WARPS_PER_CTA], 16
            ]
            # Staged rowwise scales for the whole CTA
            sScaleRow: cute.struct.Align[
                cute.struct.MemRange[
                    Float8E8M0FNU,
                    self._TILE_ROWS * self._NUM_TILES * self._TILE_COLS // MXFP8_BLOCK_SCALING_SIZE,
                ],
                16,
            ]
            # Staged colwise scales for the whole CTA
            sScaleCol: cute.struct.Align[
                cute.struct.MemRange[Float8E8M0FNU, self._NUM_TILES * self._TILE_COLS], 16
            ]

        smem = cutlass.memory.SmemAllocator()
        storage = smem.allocate(SharedStorage)

        tile_layout = cute.make_layout(
            ((self._TILE_ROWS, self._TILE_COLS), self._NUM_STAGES),
            stride=((self._TILE_COLS, 1), self._TILE_ROWS * self._TILE_COLS),
        )
        # SMEM input tile should have the same Swizzle<3,4,3> as the input TMA atom
        sX = storage.sX.get_tensor(tile_layout, swizzle=cute.make_swizzle(3, 4, 3))
        # SMEM output tiles should have the same Swizzle<2,4,3> as the output TMA atom
        sO_row = storage.sO_row.get_tensor(tile_layout, swizzle=cute.make_swizzle(2, 4, 3))
        sO_col = storage.sO_col.get_tensor(tile_layout, swizzle=cute.make_swizzle(2, 4, 3))
        # Reshape the per-warp colwise-reduce scratchpad to a (threads, warps) layout for easier indexing.
        sColReduce = storage.sColReduce.get_tensor(
            cute.make_layout((THREADS_PER_WARP, self._WARPS_PER_CTA), stride=(1, THREADS_PER_WARP))
        )
        # The whole rowwise scale factor SMEM tensor covered by a CTA
        sScaleRow2D = storage.sScaleRow.get_tensor(
            cute.make_ordered_layout(
                (
                    self._TILE_ROWS * self._NUM_TILES_Y,
                    self._SCALE_COLS * self._NUM_TILES_X,
                ),
                order=(1, 0),
            )
        )
        # Divide by a single tile's shape, so it becomes ((TILE_ROWS, SCALE_COLS), TILES) for easier indexing
        # where TILES = NUM_TILES_X * NUM_TILES_Y
        sScaleRow = cute.zipped_divide(sScaleRow2D, (self._TILE_ROWS, self._SCALE_COLS))
        # The whole colwise scale factor SMEM tensor covered by a CTA
        sScaleCol2D = storage.sScaleCol.get_tensor(
            cute.make_ordered_layout(
                (
                    self._SCALE_ROWS * self._NUM_TILES_Y,
                    self._TILE_COLS * self._NUM_TILES_X,
                ),
                order=(1, 0),
            )
        )
        # Divide by a single tile's shape, so it becomes ((SCALE_ROWS, TILE_COLS), TILES) for easier indexing
        # where TILES = NUM_TILES_X * NUM_TILES_Y
        sScaleCol = cute.zipped_divide(sScaleCol2D, (self._SCALE_ROWS, self._TILE_COLS))

        # Zero the scale SMEM buffers so partial tiles / padding columns flush as 0.
        tidx, _, _ = cute.arch.thread_idx()
        # View rowwise staging buffers as flat uint32 and stride the CTA over them.
        _ROW_SCALE_WORDS = self._TILE_ROWS * self._SCALE_COLS * self._NUM_TILES // 4
        sScaleRow_u32 = cute.make_tensor(
            cute.recast_ptr(sScaleRow2D.iterator, dtype=Uint32),
            cute.make_layout((_ROW_SCALE_WORDS,), stride=(1,)),
        )
        for i in cutlass.range_constexpr(cute.ceil_div(_ROW_SCALE_WORDS, self._THREADS_PER_CTA)):
            slot = i * self._THREADS_PER_CTA + tidx
            if slot < _ROW_SCALE_WORDS:
                sScaleRow_u32[slot] = Uint32(0)
        # View colwise staging buffers as flat uint32 and stride the CTA over them.
        _COL_SCALE_WORDS = self._SCALE_ROWS * self._TILE_COLS * self._NUM_TILES // 4
        sScaleCol_u32 = cute.make_tensor(
            cute.recast_ptr(sScaleCol2D.iterator, dtype=Uint32),
            cute.make_layout((_COL_SCALE_WORDS,), stride=(1,)),
        )
        for i in cutlass.range_constexpr(cute.ceil_div(_COL_SCALE_WORDS, self._THREADS_PER_CTA)):
            slot = i * self._THREADS_PER_CTA + tidx
            if slot < _COL_SCALE_WORDS:
                sScaleCol_u32[slot] = Uint32(0)

        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        if warp_idx == 0:
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom)

        # Only warp 0 is the producer (issues TMA)
        producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, 1)
        # Every warp is the consumer
        consumer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, self._WARPS_PER_CTA)
        # One TMA loads a tile
        tx_count = self._TILE_ROWS * self._TILE_COLS * dtype.width // 8

        mainloop_pipeline = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.mbar_storage.data_ptr(),
            num_stages=self._NUM_STAGES,
            producer_group=producer_group,
            consumer_group=consumer_group,
            tx_count=tx_count,
            cta_layout_vmnk=None,
        )

        prod_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self._NUM_STAGES
        )
        cons_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self._NUM_STAGES
        )

        gX_tiled = cute.zipped_divide(tma_src, (self._TILE_ROWS, self._TILE_COLS))
        tXsX, tXgX = cute.nvgpu.cpasync.tma_partition(
            tma_atom, 0, cute.make_layout(1), sX, gX_tiled
        )
        gO_row_tiled = cute.zipped_divide(tma_dst_out_row, (self._TILE_ROWS, self._TILE_COLS))
        tXsO_row, tXgO_row = cute.nvgpu.cpasync.tma_partition(
            tma_atom_out_row, 0, cute.make_layout(1), sO_row, gO_row_tiled
        )
        gO_col_tiled = cute.zipped_divide(tma_dst_out_col, (self._TILE_ROWS, self._TILE_COLS))
        tXsO_col, tXgO_col = cute.nvgpu.cpasync.tma_partition(
            tma_atom_out_col, 0, cute.make_layout(1), sO_col, gO_col_tiled
        )

        cute.arch.sync_threads()

        # Prologue: warp 0 prefetches up to NUM_STAGES tiles to fully fill the pipeline
        if warp_idx == 0:
            for s in cutlass.range_constexpr(self._NUM_STAGES):
                if s < num_tiles:
                    # This tile's position in the CTA's NUM_TILES_Y x NUM_TILES_X arrangement.
                    tile_y = s // num_tiles_x
                    tile_x = s % num_tiles_x
                    mainloop_pipeline.producer_acquire(prod_state)
                    cute.copy(
                        tma_atom,
                        tXgX[(None, (tile_y_base + tile_y, tile_x_base + tile_x))],
                        tXsX[(None, prod_state.index)],
                        tma_bar_ptr=mainloop_pipeline.producer_get_barrier(prod_state),
                    )
                    mainloop_pipeline.producer_commit(prod_state)
                    prod_state.advance()

        # Consumer: all warps fetch from the pipeline, process its tile, and issue a new load to the tile buffer it just consumed
        for tile_idx in cutlass.range(num_tiles, unroll=1):
            mainloop_pipeline.consumer_wait(cons_state)
            # Only allow at most _NUM_STAGES-1 stages to be in-flight, because this iteration will reuse the ring buffer
            # that is read _NUM_STAGES iterations ago so we must wait for whoever is reading that buffer to finish
            if warp_idx == 0:
                cute.arch.cp_async_bulk_wait_group(self._NUM_STAGES - 1, read=True)
            cute.arch.sync_threads()
            # The current pipeline stage index, which is the tile index modulo the number of stages.
            # This is used to index into the shared memory ring buffers that are wrapped around the number of stages.
            stage_idx = cons_state.index

            # This tile's position in the CTA's NUM_TILES_Y x NUM_TILES_X arrangement.
            tile_y = tile_idx // num_tiles_x
            tile_x = tile_idx % num_tiles_x

            # Process the 32x64 SMEM tile for this stage
            quantize_bidimensional_mxfp8_swizzled(
                sX[None, stage_idx],
                sO_row[None, stage_idx],
                sO_col[None, stage_idx],
                sScaleRow[(None, (tile_y, tile_x))],
                sScaleCol[(None, (tile_y, tile_x))],
                sColReduce[
                    None, warp_idx
                ],  # Pick the per-warp colwise-reduce scratchpad for this warp
                self._WARPS_PER_CTA,
                self.cfg.MAX_NORM_RCP,
                dtype,
                self.cfg.FP8_DTYPE,
            )

            # Make the smem output writes visible to the TMA async proxy, then store.
            cute.arch.fence_proxy("async.shared", space="cta")
            cute.arch.sync_threads()

            # We are done with this input pipeline SMEM buffer, signal the producer that it can write to this buffer
            mainloop_pipeline.consumer_release(cons_state)

            # Issue TMA and write the output to GMEM
            if warp_idx == 0:
                tile_coord = (tile_y_base + tile_y, tile_x_base + tile_x)
                cute.copy(
                    tma_atom_out_row, tXsO_row[(None, stage_idx)], tXgO_row[(None, tile_coord)]
                )
                cute.copy(
                    tma_atom_out_col, tXsO_col[(None, stage_idx)], tXgO_col[(None, tile_coord)]
                )
                cute.arch.cp_async_bulk_commit_group()

            cons_state.advance()

            # Producer: refill the buffer the consumer just freed with the tile
            # NUM_STAGES ahead.
            next_tile_idx = tile_idx + self._NUM_STAGES
            if next_tile_idx < num_tiles:
                if warp_idx == 0:
                    next_tile_y = next_tile_idx // num_tiles_x
                    next_tile_x = next_tile_idx % num_tiles_x
                    mainloop_pipeline.producer_acquire(prod_state)
                    cute.copy(
                        tma_atom,
                        tXgX[(None, (tile_y_base + next_tile_y, tile_x_base + next_tile_x))],
                        tXsX[(None, prod_state.index)],
                        tma_bar_ptr=mainloop_pipeline.producer_get_barrier(prod_state),
                    )
                    mainloop_pipeline.producer_commit(prod_state)
                    prod_state.advance()

        # TODO(kainingz): the rowwise WIDTH is hardcoded to 4. Consider using other width if we can convince
        # cute.autovec_copy that wider stores are safe (e.g. 16B) and the runtime row pitch allows it. This might
        # require us to change `scale_rowwise_shape`'s divisibility when we compile with fake tensors
        self._flush_scales_to_gmem(
            sScaleRow2D,
            mS_row,
            tidx,
            bidx,
            bidy,
            ROWS=self._TILE_ROWS * self._NUM_TILES_Y,
            COLS=self._SCALE_COLS * self._NUM_TILES_X,
            WIDTH=4,
        )
        self._flush_scales_to_gmem(
            sScaleCol2D,
            mS_col,
            tidx,
            bidx,
            bidy,
            ROWS=self._SCALE_ROWS * self._NUM_TILES_Y,
            COLS=self._TILE_COLS * self._NUM_TILES_X,
            WIDTH=1 if self.cfg.WITH_GEMM_SWIZZLED_SCALES else 16,
        )

        # Drain the final stores (their gmem writes must complete before the CTA exits).
        cute.arch.cp_async_bulk_wait_group(0, read=False)

    @cute.jit
    def _flush_scales_to_gmem(
        self,
        sScale2D: cute.Tensor,
        mS: cute.Tensor,
        tidx: Int32,
        bidx: Int32,
        bidy: Int32,
        ROWS: cutlass.Constexpr[int],
        COLS: cutlass.Constexpr[int],
        WIDTH: cutlass.Constexpr[int],
    ):
        """Flush a staged (ROWS, COLS) SMEM scale block (a plain row-major 2D tensor) to its (bidy, bidx) slice of the gmem scale tensor,
        where each GMEM slice has the same shape as this SMEM tile. Use `WIDTH` bytes per vectorized store.
        """
        # Use cute.size instead of cute.shape because under the swizzled layout each mode is a nested tuple ((32, 4, num_tiles_M) etc.)
        mS_M = cute.size(mS, mode=[0])
        mS_N = cute.size(mS, mode=[1])
        # Obtain the GMEM slice for the output scale factor block of this CTA
        mS_tile = cute.local_tile(mS, (ROWS, COLS), (bidy, bidx))

        ACTIVE_THREAD_COLS = COLS // WIDTH
        _, tv_flush_layout = cute.make_layout_tv(
            thr_layout=cute.make_layout((ROWS, ACTIVE_THREAD_COLS), stride=(ACTIVE_THREAD_COLS, 1)),
            val_layout=cute.make_layout((1, WIDTH), stride=(WIDTH, 1)),
        )
        TOTAL_ACTIVE = ROWS * ACTIVE_THREAD_COLS

        # We may have more total active threads than threads in a CTA, so we do multiple waves of stores to flush the whole buffer
        for wave in cutlass.range_constexpr(cute.ceil_div(TOTAL_ACTIVE, self._THREADS_PER_CTA)):
            thread_idx = wave * self._THREADS_PER_CTA + tidx
            if thread_idx < TOTAL_ACTIVE:
                # Find the position for this slot's vectorized store in the GMEM scale factor buffer
                thread_y = bidy * ROWS + thread_idx // ACTIVE_THREAD_COLS
                thread_x = bidx * COLS + (thread_idx % ACTIVE_THREAD_COLS) * WIDTH
                # For rowwise we have N divisible by 4 and WIDTH=4, and for colwise we have N divisible by 128 and WIDTH=16,
                # so `thread_x < mS_N` with vectorized store is safe here.
                # A thread only writes to a single row so `thread_y < mS_M` is also safe here.
                if thread_y < mS_M and thread_x < mS_N:
                    cute.autovec_copy(
                        cute.composition(sScale2D, tv_flush_layout)[thread_idx, None],
                        cute.composition(mS_tile, tv_flush_layout)[thread_idx, None],
                    )
