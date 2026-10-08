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
    bf16_mx_scale_reciprocal,
    bf16_mx_scale_reciprocal_pair,
    bf16_mx_scale_bytes_pair,
    create_l2_policy,
)
from transformer_engine.common.CuTeDSL.utils_fp8 import mul_bf16x2x2_cvt_bf16x4_to_fp8x4
from .quantize_mxfp8_common import (
    MXFP8_BLOCK_SCALING_SIZE,
    THREADS_PER_WARP,
    MXFP8QuantizeConfig,
    MXFP8QuantizeKernelBase,
    bf16_pair_magnitude,
    noop_flag_is_set,
)


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
    ELEMENTS_Y_PER_THREAD: cutlass.Constexpr[int],
    ELEMENTS_X_PER_THREAD: cutlass.Constexpr[int],
    cache_policy: Int64,
):
    """Quantize one (TILE_ROWS, TILE_COLS) GMEM tile in both directions.

    TILE_ROWS = CTA_THREADS_Y * ELEMENTS_Y_PER_THREAD = 32, so every columnwise
    MXFP8 block is contained in the tile. Each warp owns ELEMENTS_Y_PER_THREAD
    consecutive rows, and each lane owns ELEMENTS_X_PER_THREAD adjacent columns.
    Load the tile once, then reuse its registers for both quantization passes.
    """
    # Shape of the tile
    TILE_ROWS = CTA_THREADS_Y * ELEMENTS_Y_PER_THREAD
    TILE_COLS = CTA_THREADS_X * ELEMENTS_X_PER_THREAD

    # How many threads cooperate to process a single MXFP8 block (32 elements) for both quantized directions
    THREADS_X_PER_MX_BLOCK = MXFP8_BLOCK_SCALING_SIZE // ELEMENTS_X_PER_THREAD
    THREADS_Y_PER_MX_BLOCK = MXFP8_BLOCK_SCALING_SIZE // ELEMENTS_Y_PER_THREAD

    MAX_EXPONENT = 8 if cfg.FP8_DTYPE is cutlass.Float8E4M3FN else 15
    SCALE_DTYPE = cutlass.Float8E8M0FNU

    tidx, _, _ = cute.arch.thread_idx()

    thread_layout = cute.make_layout((CTA_THREADS_Y, CTA_THREADS_X), stride=(CTA_THREADS_X, 1))
    _, tv_layout = cute.make_layout_tv(
        thr_layout=thread_layout,
        val_layout=cute.make_layout(
            (ELEMENTS_Y_PER_THREAD, ELEMENTS_X_PER_THREAD), stride=(ELEMENTS_X_PER_THREAD, 1)
        ),
    )

    gX_thread = cute.composition(gX_tile, tv_layout)[tidx, None]
    gO_row_thread = cute.composition(gO_row_tile, tv_layout)[tidx, None]
    gO_col_thread = cute.composition(gO_col_tile, tv_layout)[tidx, None]

    # Group CTA threads by THREADS_X_PER_MX_BLOCK since they share the same rowwise scale for a MXFP8 block.
    # And assign stride 0 to that submode
    tv_row_scale = cute.make_layout(
        (
            (THREADS_X_PER_MX_BLOCK, (CTA_THREADS_X // THREADS_X_PER_MX_BLOCK, CTA_THREADS_Y)),
            ELEMENTS_Y_PER_THREAD,
        ),
        stride=(
            (0, (TILE_ROWS, ELEMENTS_Y_PER_THREAD)),
            1,
        ),
    )
    gS_row_thread = cute.composition(gS_row_tile, tv_row_scale)[tidx, None]

    # Group CTA threads by THREADS_Y_PER_MX_BLOCK since they share the same columnwise scale for a MXFP8 block.
    # And assign stride 0 to that submode
    tv_col_scale = cute.make_layout(
        (
            (CTA_THREADS_X, (THREADS_Y_PER_MX_BLOCK, CTA_THREADS_Y // THREADS_Y_PER_MX_BLOCK)),
            ELEMENTS_X_PER_THREAD,
        ),
        stride=(
            (ELEMENTS_X_PER_THREAD, (0, 1)),
            1,
        ),
    )
    gS_col_thread = cute.composition(gS_col_tile, tv_col_scale)[tidx, None]

    # Input fragment contains all rows it handles; output fragment contains only one row and we write it back row by row
    rX = cute.make_rmem_tensor((ELEMENTS_X_PER_THREAD, ELEMENTS_Y_PER_THREAD), cfg.DTYPE)
    rO = cute.make_rmem_tensor((ELEMENTS_X_PER_THREAD,), cfg.FP8_DTYPE)

    # For rowwise scales, we write one row at a time
    rS_row = cute.make_rmem_tensor((1,), SCALE_DTYPE)

    load_atom = cute.make_copy_atom(
        cute.nvgpu.CopyG2ROp(),
        cfg.DTYPE,
        num_bits_per_copy=ELEMENTS_X_PER_THREAD * cfg.DTYPE.width,
        invariant=True,
    )
    store_atom = cute.make_copy_atom(
        cute.nvgpu.CopyR2GOp(),
        cfg.FP8_DTYPE,
        num_bits_per_copy=ELEMENTS_X_PER_THREAD * cfg.FP8_DTYPE.width,
    )

    row_scale_atom = cute.make_copy_atom(cute.nvgpu.CopyR2GOp(), SCALE_DTYPE, num_bits_per_copy=8)
    col_scale_atom = cute.make_copy_atom(
        cute.nvgpu.CopyR2GOp(), SCALE_DTYPE, num_bits_per_copy=ELEMENTS_X_PER_THREAD * 8
    )

    scale_and_pack = mul_bf16x2x2_cvt_bf16x4_to_fp8x4(cfg.FP8_DTYPE)

    # Copy all rows a thread sweeps into the RMEM fragment at once
    for row_idx in cutlass.range_constexpr(ELEMENTS_Y_PER_THREAD):
        cute.copy(load_atom, gX_thread[None, row_idx], rX[None, row_idx], cache_policy=cache_policy)

    @cute.jit
    def extract_mx_block_amax(values_i32: cute.Tensor) -> Uint32:
        """Reduce a 32-column MXFP8 block across its two or four lane owners."""
        amax = values_i32[0]
        for bf16x2_idx in cutlass.range_constexpr(1, cute.size(values_i32)):
            amax = abs_max_x2_bf16(amax, values_i32[bf16x2_idx])
        for stage in cutlass.range_constexpr(THREADS_X_PER_MX_BLOCK.bit_length() - 1):
            amax = abs_max_x2_bf16(amax, cute.arch.shuffle_sync_bfly(amax, 1 << stage))
        return bf16_pair_magnitude(amax)

    # Rowwise pass: produce each row's rowwise scales and quantized outputs, and write them back to GMEM immediately.
    rO_u32 = cute.recast_tensor(rO, Uint32)
    # Only one thread from threads that share the MXFP8 block writes the rowwise scale
    is_writer_thread = tidx % THREADS_X_PER_MX_BLOCK == 0
    for row_idx in cutlass.range_constexpr(ELEMENTS_Y_PER_THREAD):
        rX_row_i32 = cute.recast_tensor(rX[None, row_idx], Int32)
        reciprocal = bf16_mx_scale_reciprocal(extract_mx_block_amax(rX_row_i32), MAX_EXPONENT)
        reciprocal_packed = reciprocal | (reciprocal << Uint32(16))
        for pack_idx in cutlass.range_constexpr(cute.size(rO_u32)):
            rO_u32[pack_idx] = scale_and_pack(
                rX_row_i32[2 * pack_idx],
                reciprocal_packed,
                rX_row_i32[2 * pack_idx + 1],
                reciprocal_packed,
            )
        # rO_u32 and rO share the same memory
        cute.copy(store_atom, rO, gO_row_thread[None, row_idx], cache_policy=cache_policy)
        if is_writer_thread:
            rS_row[0] = Uint8(bf16_mx_scale_bytes_pair(reciprocal_packed)).bitcast(SCALE_DTYPE)
            cute.copy(
                row_scale_atom,
                rS_row,
                cute.local_tile(gS_row_thread, (1,), (row_idx,)),
                cache_policy=cache_policy,
            )

    # Columnwise pass: only produce the colwise scale.
    # We first reduce each thread's rows into one maximum per column before reduce across different threads in the CTA
    rAmax_col = cute.make_rmem_tensor((ELEMENTS_X_PER_THREAD,), cfg.DTYPE)

    # Gather the amax for each column that this thread sweeps
    rX_i32 = cute.recast_tensor(rX, Int32)
    rAmax_col_i32 = cute.recast_tensor(rAmax_col, Int32)
    for bf16x2_idx in cutlass.range_constexpr(cute.size(rX_i32, mode=[0])):
        amax = rX_i32[bf16x2_idx, 0]
        for row_idx in cutlass.range_constexpr(1, ELEMENTS_Y_PER_THREAD):
            amax = abs_max_x2_bf16(amax, rX_i32[bf16x2_idx, row_idx])
        rAmax_col_i32[bf16x2_idx] = amax

    # Threads that process a columnwise MXFP8 block don't belong to the same warp so we have to use SMEM to reduce the scale
    allocator = cutlass.memory.SmemAllocator()
    sAmaxs_col_scratch_layout = cute.zipped_product(
        rAmax_col.layout,
        thread_layout,
    )
    sAmaxs_col_scratch = allocator.allocate_tensor(
        cfg.DTYPE, sAmaxs_col_scratch_layout, byte_alignment=16
    )
    sAmaxs_col_scratch_i32 = cute.recast_tensor(sAmaxs_col_scratch, Int32)
    sS_col_reduction_scratch = allocator.allocate_tensor(
        cfg.DTYPE, (CTA_THREADS_Y, CTA_THREADS_X), byte_alignment=16
    )
    sS_col_reduction_scratch_u32 = cute.recast_tensor(sS_col_reduction_scratch, Uint32)

    # Every thread copies the RMEM fragment to SMEM scratch space for reduction across warps later
    cute.autovec_copy(
        rAmax_col,
        sAmaxs_col_scratch[None, tidx],
    )

    # Reduce thread read amaxes from SMEM then calculate & write the columnwise scale to SMEM scratch space
    is_reduce_thread = tidx < CTA_THREADS_X
    if is_reduce_thread:
        amaxX2 = sAmaxs_col_scratch_i32[0, tidx]
        for scratch_row in cutlass.range_constexpr(CTA_THREADS_Y):
            amaxX2 = abs_max_x2_bf16(amaxX2, sAmaxs_col_scratch_i32[scratch_row, tidx])
        sS_col_reduction_scratch_u32[tidx] = bf16_mx_scale_reciprocal(amaxX2, MAX_EXPONENT)
    cute.arch.sync_threads()

    # Let the reduction threads write the RMEM columnwise scales back to GMEM
    if is_reduce_thread:
        rS_col_u32 = cute.recast_tensor(rS_col, Uint32)
        for packed_idx in cutlass.range_constexpr(cute.size(rS_col_u32)):
            rS_col_u32[packed_idx] = bf16_mx_scale_bytes_pair(rS_col_u32[2 * packed_idx]) | (
                bf16_mx_scale_bytes_pair(rS_col_u32[2 * packed_idx + 1]) << Uint32(16)
            )
        cute.copy(col_scale_atom, rS_col, gS_col_thread, cache_policy=cache_policy)

    # Every thread reads the reduced columnwise scales from SMEM to RMEM
    rS_col = cute.make_rmem_tensor((ELEMENTS_X_PER_THREAD,), cfg.DTYPE)
    cute.autovec_copy(
        sS_col_reduction_scratch[tidx],
        rS_col
    )

    # And then every thread uses the columnwise scales to calculate quantized outputs and writes it to GMEM
    
    


    # # One thread reduces each column pair across all warps (all 32 tile rows).
    # if tidx < cute.size(sColumnScales) // 2:
    #     sColumnAmaxs_i32 = cute.recast_tensor(sColumnAmaxs, Int32)
    #     sColumnScales_u32 = cute.recast_tensor(sColumnScales, Uint32)
    #     amax = sColumnAmaxs_i32[0, tidx]
    #     for warp_idx in cutlass.range_constexpr(1, CTA_THREADS_Y):
    #         amax = abs_max_x2_bf16(amax, sColumnAmaxs_i32[warp_idx, tidx])
    #     sColumnScales_u32[tidx] = bf16_mx_scale_reciprocal_pair(amax, MAX_EXPONENT)
    # cute.arch.sync_threads()

    # # The same layout returns the reciprocals to each lane's original columns.
    # for half_idx in cutlass.range_constexpr(2):
    #     cute.autovec_copy(
    #         cute.local_tile(sColumnScale_thread, (cute.size(rColumnScales) // 2,), (half_idx,)),
    #         cute.local_tile(rColumnScales, (cute.size(rColumnScales) // 2,), (half_idx,)),
    #     )

    # rColumnScales_u32 = cute.recast_tensor(rColumnScales, Uint32)
    # if warp == 0:
    #     rS_col_u32 = cute.recast_tensor(rS_col, Uint32)
    #     for pack_idx in cutlass.range_constexpr(cute.size(rS_col_u32)):
    #         rS_col_u32[pack_idx] = bf16_mx_scale_bytes_pair(rColumnScales_u32[2 * pack_idx]) | (
    #             bf16_mx_scale_bytes_pair(rColumnScales_u32[2 * pack_idx + 1]) << Uint32(16)
    #         )
    #     cute.copy(col_scale_atom, rS_col, gS_col_thread, cache_policy=cache_policy)

    # for row_idx in cutlass.range_constexpr(ELEMENTS_Y_PER_THREAD):
    #     rX_row_i32 = cute.recast_tensor(rX[None, row_idx], Int32)
    #     for pack_idx in cutlass.range_constexpr(cute.size(rO_u32)):
    #         rO_u32[pack_idx] = scale_and_pack(
    #             rX_row_i32[2 * pack_idx],
    #             rColumnScales_u32[2 * pack_idx],
    #             rX_row_i32[2 * pack_idx + 1],
    #             rColumnScales_u32[2 * pack_idx + 1],
    #         )
    #     cute.copy(store_atom, rO, gO_col_thread[None, row_idx], cache_policy=cache_policy)


class MXFP8QuantizeRegisterBidimensionalKernel(MXFP8QuantizeKernelBase):
    """Load one CTA tile into registers and quantize it along both axes.

    TILE_ROWS = _CTA_THREADS_Y * _ELEMENTS_Y_PER_THREAD = 32.
    TILE_COLS = _CTA_THREADS_X * ELEMENTS_X_PER_THREAD = 256 or 512.
    The 32-row tile contains complete columnwise MXFP8 blocks; shared memory
    combines the partial column maxima from the eight warps. Requires aligned
    BF16 input, M divisible by 32, N divisible by 256 and plain scale layouts.
    """

    _CTA_THREADS_Y = 8
    _CTA_THREADS_X = 32

    def __init__(self, cfg: MXFP8QuantizeConfig):
        self.cfg = cfg
        self._ELEMENTS_Y_PER_THREAD = MXFP8_BLOCK_SCALING_SIZE // self._CTA_THREADS_Y  # 4

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
        TILE_ROWS = self._CTA_THREADS_Y * self._ELEMENTS_Y_PER_THREAD
        WIDE_TILE_COLS = self._CTA_THREADS_X * 16
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
                    ELEMENTS_X_PER_THREAD=16,
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
                    ELEMENTS_X_PER_THREAD=16,
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
                ELEMENTS_X_PER_THREAD=8,
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
        ELEMENTS_X_PER_THREAD: cutlass.Constexpr[int],
        CLUSTER_WIDTH: cutlass.Constexpr[int],
    ):
        assert ELEMENTS_X_PER_THREAD in (8, 16)

        # Dispatch checks these alignments before selecting this kernel.
        mX = cute.make_tensor(mX.iterator.align(32), mX.layout)
        mS_col = cute.make_tensor(mS_col.iterator.align(16), mS_col.layout)
        M, N = mX.shape
        TILE_ROWS = self._CTA_THREADS_Y * self._ELEMENTS_Y_PER_THREAD
        TILE_COLS = self._CTA_THREADS_X * ELEMENTS_X_PER_THREAD
        THREADS_PER_CTA = self._CTA_THREADS_Y * self._CTA_THREADS_X

        grid_cols = N // TILE_COLS
        grid_rows = M // TILE_ROWS
        cluster_cols = Int32(CLUSTER_WIDTH)
        # Adjust cluster size so that the CTAs in the column direction can be evenly divided into clusters.
        while cluster_cols > 1 and grid_cols % cluster_cols != 0:
            cluster_cols -= 1
        # Keep column CTAs on grid.x so clusters read adjacent column tiles.
        grid = [grid_cols, grid_rows]
        block = [THREADS_PER_CTA]
        cluster = [cluster_cols, 1, 1]

        # Hardcode the layout using static N for some of the commonly used shapes
        if N == 2048:
            payload_layout = cute.make_layout((M, 2048), stride=(2048, 1))
            gX = cute.make_tensor(mX.iterator, payload_layout)
            gO_row = cute.make_tensor(mO_row.iterator, payload_layout)
            gO_col = cute.make_tensor(mO_col.iterator, payload_layout)
            gS_row = cute.make_tensor(
                mS_row.iterator,
                cute.make_layout((cute.size(mS_row, mode=[0]), 64), stride=(64, 1)),
            )
            gS_col = cute.make_tensor(
                mS_col.iterator,
                cute.make_layout((cute.size(mS_col, mode=[0]), 2048), stride=(2048, 1)),
            )
            self.kernel(
                gX,
                gO_row,
                gS_row,
                gO_col,
                gS_col,
                mNoop,
                ELEMENTS_X_PER_THREAD=ELEMENTS_X_PER_THREAD,
            ).launch(grid=grid, block=block, cluster=cluster, min_blocks_per_mp=4, stream=stream)
        elif N == 4096:
            payload_layout = cute.make_layout((M, 4096), stride=(4096, 1))
            gX = cute.make_tensor(mX.iterator, payload_layout)
            gO_row = cute.make_tensor(mO_row.iterator, payload_layout)
            gO_col = cute.make_tensor(mO_col.iterator, payload_layout)
            gS_row = cute.make_tensor(
                mS_row.iterator,
                cute.make_layout((cute.size(mS_row, mode=[0]), 128), stride=(128, 1)),
            )
            gS_col = cute.make_tensor(
                mS_col.iterator,
                cute.make_layout((cute.size(mS_col, mode=[0]), 4096), stride=(4096, 1)),
            )
            self.kernel(
                gX,
                gO_row,
                gS_row,
                gO_col,
                gS_col,
                mNoop,
                ELEMENTS_X_PER_THREAD=ELEMENTS_X_PER_THREAD,
            ).launch(grid=grid, block=block, cluster=cluster, min_blocks_per_mp=4, stream=stream)
        else:
            self.kernel(
                mX,
                mO_row,
                mS_row,
                mO_col,
                mS_col,
                mNoop,
                ELEMENTS_X_PER_THREAD=ELEMENTS_X_PER_THREAD,
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
        ELEMENTS_X_PER_THREAD: cutlass.Constexpr[int],
    ):
        """Skip the CTA when the noop flag is set; otherwise run the kernel body."""
        if not noop_flag_is_set(mNoop):
            self._kernel_main(
                mX,
                mO_row,
                mS_row,
                mO_col,
                mS_col,
                ELEMENTS_X_PER_THREAD=ELEMENTS_X_PER_THREAD,
            )

    @cute.jit
    def _kernel_main(
        self,
        mX: cute.Tensor,
        mO_row: cute.Tensor,
        mS_row: cute.Tensor,
        mO_col: cute.Tensor,
        mS_col: cute.Tensor,
        ELEMENTS_X_PER_THREAD: cutlass.Constexpr[int],
    ):
        """Construct CTA tile views and run bidimensional quantization."""
        cta_col, cta_row, _ = cute.arch.block_idx()
        cta_coord = (cta_row, cta_col)

        TILE_ROWS = self._CTA_THREADS_Y * self._ELEMENTS_Y_PER_THREAD
        TILE_COLS = self._CTA_THREADS_X * ELEMENTS_X_PER_THREAD

        TILER = (TILE_ROWS, TILE_COLS)
        ROW_SCALE_TILER = (TILE_ROWS, TILE_COLS // 32)
        COL_SCALE_TILER = (TILE_ROWS // 32, TILE_COLS)

        gX_tile = cute.local_tile(mX, TILER, cta_coord)
        gO_row_tile = cute.local_tile(mO_row, TILER, cta_coord)
        gO_col_tile = cute.local_tile(mO_col, TILER, cta_coord)
        gS_row_tile = cute.local_tile(mS_row, ROW_SCALE_TILER, cta_coord)
        gS_col_tile = cute.local_tile(mS_col, COL_SCALE_TILER, cta_coord)

        cache_policy = create_l2_policy(True)

        quantize_bidimensional_mxfp8_g2r(
            gX_tile,
            gO_row_tile,
            gS_row_tile,
            gO_col_tile,
            gS_col_tile,
            self.cfg,
            CTA_THREADS_Y=self._CTA_THREADS_Y,
            CTA_THREADS_X=self._CTA_THREADS_X,
            ELEMENTS_Y_PER_THREAD=self._ELEMENTS_Y_PER_THREAD,
            ELEMENTS_X_PER_THREAD=ELEMENTS_X_PER_THREAD,
            cache_policy=cache_policy,
        )
