# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Two-lane GMEM-to-RMEM rowwise MXFP8 cast, ported from cast_rowwise.cuh."""

from typing import Optional

import cutlass
from cutlass import cute
from cutlass import Boolean, Float32, Int32, Int64, Uint32, Uint8
from cuda.bindings.driver import CUstream

from transformer_engine.common.CuTeDSL.utils import (
    abs_max_x2_bf16,
    create_l2_policy,
    exp2_bf16x2_rcp,
)
from transformer_engine.common.CuTeDSL.utils_fp8 import (
    cvt_f32_to_fp8e8m0fnu,
    mul_bf16x2x2_cvt_bf16x4_to_fp8x4,
)
from .quantize_mxfp8_common import (
    MXFP8QuantizeConfig,
    MXFP8QuantizeKernelBase,
    bf16_pair_magnitude,
    derive_swizzled_scale_layout,
    noop_flag_is_set,
)


@cute.jit
def quantize_rowwise_mxfp8_g2r(
    gX_tiles: cute.Tensor,
    gO_tiles: cute.Tensor,
    mS_tiles: cute.Tensor,
    cfg: cutlass.Constexpr[MXFP8QuantizeConfig],
    M: Int32,
    N: Int32,
    CTA_THREADS_Y: cutlass.Constexpr[int],
    CTA_THREADS_X: cutlass.Constexpr[int],
    ELEMENTS_PER_THREAD: cutlass.Constexpr[int],
    NUM_TILES: cutlass.Constexpr[int],
    CHECK_BOUNDS: cutlass.Constexpr[bool],
    input_policy: Int64,
    output_policy: Int64,
):
    """Quantize a CTA's (TILE_ROWS, TILE_COLS, NUM_TILES) GMEM tiles.

    CTA_THREADS_X threads own each tile row. Each thread loads 16 BF16 values, and
    two adjacent lanes share the maximum and scale for a 32-value MXFP8 block.
    Load all tiles before reducing to preserve independent memory requests.
    """
    TILE_ROWS = CTA_THREADS_Y
    TILE_COLS = CTA_THREADS_X * ELEMENTS_PER_THREAD
    tidx, _, _ = cute.arch.thread_idx()

    _, tv_payload = cute.make_layout_tv(
        thr_layout=cute.make_layout((CTA_THREADS_Y, CTA_THREADS_X), stride=(CTA_THREADS_X, 1)),
        val_layout=cute.make_layout((1, ELEMENTS_PER_THREAD), stride=(ELEMENTS_PER_THREAD, 1)),
    )
    # Mode 0.0: two adjacent threads share one scale (stride 0).
    # Mode 0.1: thread pairs are arranged as (scale columns, tile rows).
    # Mode 1: each thread addresses one scale element.
    tv_scale = cute.make_layout(
        ((2, (CTA_THREADS_X // 2, CTA_THREADS_Y)), 1), stride=((0, (TILE_ROWS, 1)), 0)
    )

    # We load fragments from all GMEM tiles at once
    rX = cute.make_rmem_tensor((ELEMENTS_PER_THREAD, NUM_TILES), cfg.DTYPE)
    # We store a fragment of one tile at a time
    rO = cute.make_rmem_tensor((ELEMENTS_PER_THREAD,), cfg.FP8_DTYPE)
    # RMEM flags to indicate whether each tile is out of bounds or not
    # With CHECK_BOUNDS=False DCE will remove it so we don't pay extra cost when we don't use it
    tile_valid_flags = cute.make_rmem_tensor((NUM_TILES,), Boolean)

    # 16 * 2 bytes = 32 bytes which is already the widest
    LOAD_WIDTH = ELEMENTS_PER_THREAD * cfg.DTYPE.width
    load_atom = cute.make_copy_atom(
        cute.nvgpu.CopyG2ROp(),
        cfg.DTYPE,
        num_bits_per_copy=LOAD_WIDTH,
        invariant=True,  # hint that data being read stays unchanged during the kernel
    )
    store_atom = cute.make_copy_atom(
        cute.nvgpu.CopyR2GOp(),
        cfg.FP8_DTYPE,
        num_bits_per_copy=ELEMENTS_PER_THREAD * cfg.FP8_DTYPE.width,
    )

    scale_and_pack = mul_bf16x2x2_cvt_bf16x4_to_fp8x4(cfg.FP8_DTYPE)

    # Copy all tiles from GMEM to RMEM and potentially check their bounds
    for tile_idx in cutlass.range_constexpr(NUM_TILES):
        # gX_tiles is (TILE_ROWS, TILE_COLS, NUM_TILES)
        gX_tile = gX_tiles[None, None, tile_idx]
        gX_thread = cute.composition(gX_tile, tv_payload)[tidx, None]
        valid = Boolean(True)
        if cutlass.const_expr(CHECK_BOUNDS):
            cta_row, cta_col, _ = cute.arch.block_idx()
            row = cta_row * TILE_ROWS + tidx // CTA_THREADS_X
            col = (cta_col * NUM_TILES + tile_idx) * TILE_COLS + (
                tidx % CTA_THREADS_X
            ) * ELEMENTS_PER_THREAD
            valid = row < M and col < N
        tile_valid_flags[tile_idx] = valid
        rX[None, tile_idx].fill(0)
        if valid:
            cute.copy(load_atom, gX_thread, rX[None, tile_idx], cache_policy=input_policy)

    @cute.jit
    def extract_tile_amax(values_i32: cute.Tensor) -> Float32:
        """Return the 32-value MXFP8 block amax shared by two adjacent threads.

        Balanced eight-pair tree, followed by one butterfly between lane partners.
        """
        level = [
            abs_max_x2_bf16(values_i32[2 * i], values_i32[2 * i + 1])
            for i in range(cute.size(values_i32) // 2)
        ]
        amax_pair = abs_max_x2_bf16(
            abs_max_x2_bf16(level[0], level[2]), abs_max_x2_bf16(level[1], level[3])
        )
        # Compare with the neighbor thread's amax (2 threads cooperate to process a MXFP8 block).
        partner = cute.arch.shuffle_sync_bfly(amax_pair, 1)
        amax_pair = abs_max_x2_bf16(amax_pair, partner)
        return (bf16_pair_magnitude(amax_pair) << Uint32(16)).bitcast(Float32)

    for tile_idx in cutlass.range_constexpr(NUM_TILES):
        gO_tile = gO_tiles[None, None, tile_idx]
        mS_tile = mS_tiles[None, None, tile_idx]
        gO_thread = cute.composition(gO_tile, tv_payload)[tidx, None]
        mS_thread = cute.composition(mS_tile, tv_scale)[tidx, None]
        # Reinterpret the native tensors only for the packed BF16/FP8 instructions.
        rX_tile_i32 = cute.recast_tensor(rX[None, tile_idx], Int32)
        rO_u32 = cute.recast_tensor(rO, Uint32)
        amax = extract_tile_amax(rX_tile_i32)
        exponent = cvt_f32_to_fp8e8m0fnu(amax * cfg.MAX_NORM_RCP)

        # Collect 4 scale bytes and write them using one thread
        byte = Uint32(exponent.bitcast(Uint8))
        c1 = cute.arch.shuffle_sync_down(byte, 2)
        c2 = cute.arch.shuffle_sync_down(byte, 4)
        c3 = cute.arch.shuffle_sync_down(byte, 6)
        if tidx % 8 == 0 and tile_valid_flags[tile_idx]:
            scale_word = cute.make_tensor(
                cute.recast_ptr(mS_thread.iterator, dtype=Uint32), cute.make_layout(1)
            )
            scale_word[0] = byte | (c1 << Uint32(8)) | (c2 << Uint32(16)) | (c3 << Uint32(24))

        # Compute and write the quantized MXFP8 values to GMEM
        reciprocal_packed = exp2_bf16x2_rcp(exponent)
        for i in cutlass.range_constexpr(cute.size(rO_u32)):
            rO_u32[i] = scale_and_pack(
                rX_tile_i32[2 * i], reciprocal_packed, rX_tile_i32[2 * i + 1], reciprocal_packed
            )
        if tile_valid_flags[tile_idx]:
            cute.copy(store_atom, rO, gO_thread, cache_policy=output_policy)


class MXFP8QuantizeG2RRowwise2LaneKernel(MXFP8QuantizeKernelBase):
    """Two lanes per MXFP8 block, with a CTA sweeping NUM_TILES horizontally.

    One CTA tile has TILE_ROWS = CTA_THREADS_Y and
    TILE_COLS = CTA_THREADS_X * _ELEMENTS_PER_THREAD. The CTA
    processes NUM_TILES = 1 or 2 in the original input matrix. Both scale
    formats use the same payload tiling; a composed layout maps swizzled scales.
    _L2_CACHED_CTA_PERCENT tunes the share of CTAs whose input loads use normal
    L2 retention: 0 streams all input and 100 caches all input normally.
    """

    _CTA_THREADS_X = 32
    _ELEMENTS_PER_THREAD = 16
    _L2_CACHED_CTA_PERCENT = 40

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
        output_bytes = Int64(mX.shape[0]) * mX.shape[1]
        if output_bytes <= 12 << 20:
            self._launch(
                mX,
                mO_row,
                mS_row,
                mNoop,
                stream,
                CTA_THREADS_Y=8,
                CTA_THREADS_X=self._CTA_THREADS_X,
                NUM_TILES=1,
            )
        elif output_bytes <= 48 << 20:
            self._launch(
                mX,
                mO_row,
                mS_row,
                mNoop,
                stream,
                CTA_THREADS_Y=8,
                CTA_THREADS_X=self._CTA_THREADS_X,
                NUM_TILES=2,
            )
        elif output_bytes <= 96 << 20:
            self._launch(
                mX,
                mO_row,
                mS_row,
                mNoop,
                stream,
                CTA_THREADS_Y=4,
                CTA_THREADS_X=self._CTA_THREADS_X,
                NUM_TILES=2,
            )
        else:
            self._launch(
                mX,
                mO_row,
                mS_row,
                mNoop,
                stream,
                CTA_THREADS_Y=8,
                CTA_THREADS_X=self._CTA_THREADS_X,
                NUM_TILES=2,
            )

    @cute.jit
    def _launch(
        self,
        mX: cute.Tensor,
        mO: cute.Tensor,
        mS: cute.Tensor,
        mNoop: cute.Pointer,
        stream: CUstream,
        CTA_THREADS_Y: cutlass.Constexpr[int],
        CTA_THREADS_X: cutlass.Constexpr[int],
        NUM_TILES: cutlass.Constexpr[int],
    ):
        assert 0 <= self._L2_CACHED_CTA_PERCENT <= 100
        assert self._ELEMENTS_PER_THREAD == 16
        assert CTA_THREADS_X % 8 == 0
        assert CTA_THREADS_Y * CTA_THREADS_X % 32 == 0
        M, N = mX.shape
        # Each CTA processes NUM_TILES horizontal tiles of (TILE_ROWS, TILE_COLS) values.
        TILE_ROWS = CTA_THREADS_Y
        TILE_COLS = CTA_THREADS_X * self._ELEMENTS_PER_THREAD
        THREADS_PER_CTA = CTA_THREADS_Y * CTA_THREADS_X
        # Total columns covered by a CTA; its row count stays TILE_ROWS.
        CTA_ROWS = TILE_ROWS
        CTA_COLS = TILE_COLS * NUM_TILES

        grid_rows = cute.ceil_div(M, CTA_ROWS)
        grid_cols = cute.ceil_div(N, CTA_COLS)
        check_bounds = M % CTA_ROWS != 0 or N % CTA_COLS != 0
        grid = Int64(grid_rows) * grid_cols
        first_streaming_cta = grid * self._L2_CACHED_CTA_PERCENT // 100

        if grid > 0:
            if check_bounds:
                self.kernel(
                    mX,
                    mO,
                    mS,
                    mNoop,
                    first_streaming_cta,
                    CTA_THREADS_Y=CTA_THREADS_Y,
                    CTA_THREADS_X=CTA_THREADS_X,
                    NUM_TILES=NUM_TILES,
                    CHECK_BOUNDS=True,
                ).launch(grid=[grid_rows, grid_cols], block=[THREADS_PER_CTA], stream=stream)
            else:
                self.kernel(
                    mX,
                    mO,
                    mS,
                    mNoop,
                    first_streaming_cta,
                    CTA_THREADS_Y=CTA_THREADS_Y,
                    CTA_THREADS_X=CTA_THREADS_X,
                    NUM_TILES=NUM_TILES,
                    CHECK_BOUNDS=False,
                ).launch(grid=[grid_rows, grid_cols], block=[THREADS_PER_CTA], stream=stream)

    @cute.kernel
    def kernel(
        self,
        mX: cute.Tensor,
        mO: cute.Tensor,
        mS: cute.Tensor,
        mNoop: cute.Pointer,
        first_streaming_cta: Int64,
        CTA_THREADS_Y: cutlass.Constexpr[int],
        CTA_THREADS_X: cutlass.Constexpr[int],
        NUM_TILES: cutlass.Constexpr[int],
        CHECK_BOUNDS: cutlass.Constexpr[bool],
    ):
        """Check the noop flag before selecting or accessing any GMEM tile."""
        if not noop_flag_is_set(mNoop):
            # Dispatch checks 32-byte input alignment before selecting this kernel.
            gX = cute.make_tensor(mX.iterator.align(32), mX.layout)
            # CUDA's x/y axes hold row/column tiles in the same order as local_tile.
            cta_coord = cute.arch.block_idx()[:2]
            M, N = mX.shape
            TILE_ROWS = CTA_THREADS_Y
            TILE_COLS = CTA_THREADS_X * self._ELEMENTS_PER_THREAD
            CTA_ROWS = TILE_ROWS
            CTA_COLS = TILE_COLS * NUM_TILES

            gS = mS
            # Apply the swizzled scale layout if requested
            if cutlass.const_expr(self.cfg.WITH_GEMM_SWIZZLED_SCALES):
                mS_swizzled, _ = derive_swizzled_scale_layout(M, N, True, False, mS, None)
                gS = cute.composition(mS_swizzled, (cute.make_layout(M), cute.make_layout(N // 32)))

            # CTA_TILER is the block that a CTA processes in total, which consists of NUM_TILES horizontal tiles
            CTA_TILER = (CTA_ROWS, CTA_COLS)
            CTA_SCALE_TILER = (CTA_ROWS, CTA_COLS // 32)
            gX_cta = cute.local_tile(gX, CTA_TILER, cta_coord)
            gO_cta = cute.local_tile(mO, CTA_TILER, cta_coord)
            gS_cta = cute.local_tile(gS, CTA_SCALE_TILER, cta_coord)

            fraction = Float32(1.0)
            if cutlass.const_expr(self._L2_CACHED_CTA_PERCENT > 0):
                fraction = Float32(0.0)
                # A linear index is needed only for the L2 policy split.
                cta_id = cute.crd2idx(
                    (Int64(cta_coord[0]), Int64(cta_coord[1])), cute.arch.grid_dim()[:2]
                )
                if cta_id >= first_streaming_cta:
                    fraction = Float32(1.0)
            input_policy = create_l2_policy(False, fraction)
            output_policy = create_l2_policy(True)

            # Divide CTA_TILER to multiple TILER shape TILES
            TILER = (TILE_ROWS, TILE_COLS)
            SCALE_TILER = (TILE_ROWS, TILE_COLS // 32)
            # Mode 1's first index is 0 because CTA_ROWS == TILE_ROWS (there is only 1 tile on the row dimension).
            gX_tiles = cute.local_tile(gX_cta, TILER, (0, None))
            gO_tiles = cute.local_tile(gO_cta, TILER, (0, None))
            mS_tiles = cute.local_tile(gS_cta, SCALE_TILER, (0, None))

            quantize_rowwise_mxfp8_g2r(
                gX_tiles,
                gO_tiles,
                mS_tiles,
                self.cfg,
                M,
                N,
                CTA_THREADS_Y=CTA_THREADS_Y,
                CTA_THREADS_X=CTA_THREADS_X,
                ELEMENTS_PER_THREAD=self._ELEMENTS_PER_THREAD,
                NUM_TILES=NUM_TILES,
                CHECK_BOUNDS=CHECK_BOUNDS,
                input_policy=input_policy,
                output_policy=output_policy,
            )
