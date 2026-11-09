# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Two-lane GMEM-to-RMEM rowwise MXFP8 cast, ported from cast_rowwise.cuh."""

from typing import Optional, Type

import cutlass
from cutlass import cute
from cutlass import Boolean, Float32, Int32, Int64, Uint32
from cuda.bindings.driver import CUstream  # pylint: disable=no-name-in-module

from transformer_engine.common.CuTeDSL.utils import (
    abs_max_x2_bf16,
    bf16_pair_magnitude,
    create_l2_policy,
    exp2_bf16x2_rcp,
    max_num_bits_per_copy_ld_global,
    x2_hi_to_f32_bf16,
    x2_lo_to_f32_bf16,
    cvt_f32x2_to_bf16x2,
)
from transformer_engine.common.CuTeDSL.utils_fp8 import (
    cvt_f32_to_fp8e8m0fnu,
    mul_bf16x2x2_cvt_bf16x4_to_fp8x4,
)
from .quantize_mxfp8_common import (
    MXFP8QuantizeConfig,
    MXFP8QuantizeKernelBase,
    SUPPORTED_ACTIVATIONS,
    SUPPORTED_DACTIVATIONS,
    derive_swizzled_scale_layout,
)


@cute.jit
def _flush_scales_to_gmem(
    sScale: cute.Tensor,
    gS_tile: cute.Tensor,
    cta_row: Int32,
    cta_col: Int32,
    M: Int32,
    padded_cols: Int32,
    CTA_Y: cutlass.Constexpr[int],
    CTA_X: cutlass.Constexpr[int],
):
    """Flush each 4-row by 4-column scale block with a 16-byte store.

    sScale and gS_tile must have byte strides (4, 1) within each aligned (4, 4) row-column block.
    """
    tidx, _, _ = cute.arch.thread_idx()
    active_threads = CTA_X // 4
    _, tv_flush = cute.make_layout_tv(
        thr_layout=cute.make_layout((CTA_Y // 4, active_threads), stride=(active_threads, 1)),
        val_layout=cute.make_layout((4, 4), stride=(4, 1)),
    )
    if tidx < (CTA_Y // 4) * active_threads:
        thread_y = cta_row * CTA_Y + (tidx // active_threads) * 4
        thread_x = cta_col * CTA_X + (tidx % active_threads) * 4
        if thread_y < M and thread_x < padded_cols:
            cute.autovec_copy(
                cute.composition(sScale, tv_flush)[tidx, None],
                cute.composition(gS_tile, tv_flush)[tidx, None],
            )


@cute.jit
def quantize_mxfp8_gmem_tile_rowwise_2lane(
    gX_tiles: cute.Tensor,
    gA_tiles: Optional[cute.Tensor],
    gO_tiles: cute.Tensor,
    mS_tiles: cute.Tensor,
    N: Int32,
    ACTIVATION: cutlass.Constexpr[Optional[str]],
    WITH_ACT: cutlass.Constexpr[bool],
    WITH_DACT: cutlass.Constexpr[bool],
    DTYPE: cutlass.Constexpr[Type[cutlass.Numeric]],
    FP8_DTYPE: cutlass.Constexpr[Type[cutlass.Numeric]],
    MAX_NORM_RCP: cutlass.Constexpr[float],
    CTA_THREADS_Y: cutlass.Constexpr[int],
    CTA_THREADS_X: cutlass.Constexpr[int],
    ELEMENTS_PER_THREAD: cutlass.Constexpr[int],
    NUM_TILES: cutlass.Constexpr[int],
    CHECK_BOUNDS: cutlass.Constexpr[bool],
    input_policy: Int64,
    output_policy: Int64,
    cta_row: Int32,
    cta_col: Int32,
    num_elems: Int64,
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
    rX = cute.make_rmem_tensor((ELEMENTS_PER_THREAD, NUM_TILES), DTYPE)
    # Backward: the activation input, loaded alongside the incoming gradient in rX
    rA = cute.make_rmem_tensor((ELEMENTS_PER_THREAD, NUM_TILES), DTYPE)
    # We store a fragment of one tile at a time
    rO = cute.make_rmem_tensor((ELEMENTS_PER_THREAD,), FP8_DTYPE)
    # RMEM flags to indicate whether each tile is out of bounds or not
    # With CHECK_BOUNDS=False DCE will remove it so we don't pay extra cost when we don't use it
    tile_valid_flags = cute.make_rmem_tensor((NUM_TILES,), Boolean)

    # 16 * 2 bytes = 32 bytes, one 256-bit load where the toolchain supports it
    LOAD_WIDTH = min(
        ELEMENTS_PER_THREAD * DTYPE.width,
        max_num_bits_per_copy_ld_global(),
    )
    load_atom = cute.make_copy_atom(
        cute.nvgpu.CopyG2ROp(),
        DTYPE,
        num_bits_per_copy=LOAD_WIDTH,
        # TODO(kainingz): due to https://github.com/NVIDIA/cutlass/issues/3726 using
        # invariant=True here will drop the cache policy
        invariant=False,
    )
    act_load_atom = cute.make_copy_atom(
        cute.nvgpu.CopyG2ROp(),
        DTYPE,
        num_bits_per_copy=LOAD_WIDTH,
        invariant=False,
    )
    store_atom = cute.make_copy_atom(
        cute.nvgpu.CopyR2GOp(),
        FP8_DTYPE,
        num_bits_per_copy=ELEMENTS_PER_THREAD * FP8_DTYPE.width,
    )

    scale_and_pack = mul_bf16x2x2_cvt_bf16x4_to_fp8x4(FP8_DTYPE)

    # Copy all tiles from GMEM to RMEM and potentially check their bounds
    for tile_idx in cutlass.range_constexpr(NUM_TILES):
        # gX_tiles is (TILE_ROWS, TILE_COLS, NUM_TILES)
        gX_tile = gX_tiles[None, None, tile_idx]
        gX_thread = cute.composition(gX_tile, tv_payload)[tidx, None]
        valid = Boolean(True)
        if cutlass.const_expr(CHECK_BOUNDS):
            row = cta_row * TILE_ROWS + tidx // CTA_THREADS_X
            col = (cta_col * NUM_TILES + tile_idx) * TILE_COLS + (
                tidx % CTA_THREADS_X
            ) * ELEMENTS_PER_THREAD
            # When we apply the (-1, 1024) manually constructed layout to the input tensor,
            # the last row might not be a full row so we need to check OOB with element number
            # instead of just M and N
            valid = col < N and Int64(row) * N + col < num_elems
        tile_valid_flags[tile_idx] = valid
        rX[None, tile_idx].fill(0)
        if cutlass.const_expr(WITH_DACT):
            rA[None, tile_idx].fill(0)
        if valid:
            cute.copy(load_atom, gX_thread, rX[None, tile_idx], cache_policy=input_policy)
            if cutlass.const_expr(WITH_DACT):
                gA_thread = cute.composition(gA_tiles[None, None, tile_idx], tv_payload)[tidx, None]
                cute.copy(act_load_atom, gA_thread, rA[None, tile_idx], cache_policy=input_policy)

    # Fused activation: widen to FP32, apply it, and round back to BF16 as the general kernel
    # does (it truncates the activated value to the input type before the amax). The values are
    # then exactly BF16, so the packed BF16 amax and scaling below apply unchanged.
    if cutlass.const_expr(WITH_ACT or WITH_DACT):
        for tile_idx in cutlass.range_constexpr(NUM_TILES):
            rX_i32 = cute.recast_tensor(rX[None, tile_idx], Int32)
            rA_i32 = cute.recast_tensor(rA[None, tile_idx], Int32)
            for i in cutlass.range_constexpr(ELEMENTS_PER_THREAD // 2):
                lo = x2_lo_to_f32_bf16(rX_i32[i])
                hi = x2_hi_to_f32_bf16(rX_i32[i])
                if cutlass.const_expr(WITH_DACT):
                    dop = SUPPORTED_DACTIVATIONS[ACTIVATION]
                    lo = lo * dop(x2_lo_to_f32_bf16(rA_i32[i]))
                    hi = hi * dop(x2_hi_to_f32_bf16(rA_i32[i]))
                else:
                    op = SUPPORTED_ACTIVATIONS[ACTIVATION]
                    lo = op(lo)
                    hi = op(hi)
                rX_i32[i] = cvt_f32x2_to_bf16x2(lo, hi)

    @cute.jit
    def extract_mx_block_amax(values_i32: cute.Tensor) -> Float32:
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
        amax = extract_mx_block_amax(rX_tile_i32)
        exponent = cvt_f32_to_fp8e8m0fnu(amax * MAX_NORM_RCP)

        # The even lane of each pair stores its block's scale byte, as cast_rowwise.cuh does.
        # A warp's stores coalesce in either layout: one 16-byte run when packed, four
        # 4-byte runs when GEMM-swizzled. Gathering the bytes into 32-bit words first
        # touches the same sectors but adds shuffles and a branch (+12% instructions).
        # TODO(kainingz): not using vectorized word store seems to be faster in CuTeDSL.
        # Check if we should port this to CUDA
        if tidx % 2 == 0 and tile_valid_flags[tile_idx]:
            mS_thread[0] = exponent

        # Compute and write the quantized MXFP8 values to GMEM
        reciprocal_packed = exp2_bf16x2_rcp(exponent)
        for i in cutlass.range_constexpr(cute.size(rO_u32)):
            rO_u32[i] = scale_and_pack(
                rX_tile_i32[2 * i], reciprocal_packed, rX_tile_i32[2 * i + 1], reciprocal_packed
            )
        if tile_valid_flags[tile_idx]:
            cute.copy(store_atom, rO, gO_thread, cache_policy=output_policy)


class MXFP8QuantizeRowwise2LaneG2RKernel(MXFP8QuantizeKernelBase):
    """Two lanes per MXFP8 block, with a CTA sweeping NUM_TILES horizontally.

    One CTA tile has TILE_ROWS = CTA_THREADS_Y and
    TILE_COLS = CTA_THREADS_X * _ELEMENTS_PER_THREAD. The CTA
    processes NUM_TILES = 1 or 2 horizontally. With staged swizzled scales,
    tensor views group rows 32 apart to collect complete scale vectors.
    _L2_CACHED_CTA_PERCENT tunes the share of CTAs whose input loads use normal
    L2 retention: 0 streams all input and 100 caches all input normally.
    """

    _CTA_THREADS_X = 32
    _ELEMENTS_PER_THREAD = 16
    # cast_rowwise.cuh's tier table: the two smaller tiers stream all of their input.
    _L2_CACHED_CTA_PERCENT = 40
    _SMALL_TIER_CACHED_PERCENT = 0
    _MAX_GRID_Y = 65535

    @staticmethod
    def supports(cfg: MXFP8QuantizeConfig) -> bool:
        # BF16 packed math.
        return (
            not cfg.WITH_AMAX
            and not cfg.WITH_DBIAS
            and cfg.DTYPE is cutlass.BFloat16
            and cfg.ROWWISE
            and not cfg.COLWISE
        )

    @staticmethod
    @cute.jit
    def eligible_to_dispatch(
        cfg: cutlass.Constexpr[MXFP8QuantizeConfig],
        mX: cute.Tensor,
        mO_row: Optional[cute.Tensor],
        mS_row: Optional[cute.Tensor],
        mO_col: Optional[cute.Tensor],
        mS_col: Optional[cute.Tensor],
        mNoop: cute.Pointer,
        mDActInput: Optional[cute.Tensor],
    ) -> Boolean:
        N = mX.shape[1]
        act_input_aligned = Boolean(True)
        if cutlass.const_expr(mDActInput is not None):
            act_input_aligned = mDActInput.iterator.toint() % 32 == 0
        scale_stride_matches = Boolean(True)
        if cutlass.const_expr(not cfg.WITH_GEMM_SWIZZLED_SCALES):
            # The flat view needs the packed scales to be contiguous.
            scale_stride_matches = mS_row.shape[1] == N // 32
        # Column tiles use grid.x and row tiles spill from grid.y into grid.z, so no shape
        # exceeds the grid limits.
        return (
            N % 128 == 0
            and mX.iterator.toint() % 32 == 0
            and act_input_aligned
            and scale_stride_matches
            and mNoop.toint() == 0
        )

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
        mNoop: cute.Pointer,  # Unused: the dispatcher routes noop calls to the general kernel
        mDActInput: Optional[cute.Tensor],
        mWorkspace: Optional[cute.Tensor],
        stream: CUstream,
    ):
        M, N = mX.shape
        num_elems = Int64(M) * N
        # View the input tensor as a layout with last dimension = 1024 when not swizzled and the number of elements can be expressed using i32
        # However, M*N might not be divisible by 1024, so the last row of this layout might be a partial row with fewer than 1024 elements,
        # which is why we need to pass num_elems to the kernel so OOB checks will identify these regions as invalid offsets correctly
        if cutlass.const_expr(not self.cfg.WITH_GEMM_SWIZZLED_SCALES):
            if num_elems <= 2**31 - 1:
                rows = cute.ceil_div(num_elems, 1024)
                flat = cute.make_layout((rows, 1024), stride=(1024, 1))
                mA = None
                if cutlass.const_expr(self.cfg.WITH_DACT):
                    mA = cute.make_tensor(mDActInput.iterator, flat)
                self._launch_tiers(
                    cute.make_tensor(mX.iterator, flat),
                    mA,
                    cute.make_tensor(mO_row.iterator, flat),
                    cute.make_tensor(mS_row.iterator, cute.make_layout((rows, 32), stride=(32, 1))),
                    stream,
                    num_elems,
                )
            else:
                self._launch_tiers(mX, mDActInput, mO_row, mS_row, stream, num_elems)
        else:
            self._launch_tiers(mX, mDActInput, mO_row, mS_row, stream, num_elems)

    @cute.jit
    def _launch_tiers(
        self,
        mX: cute.Tensor,
        mA: Optional[cute.Tensor],
        mO_row: cute.Tensor,
        mS_row: cute.Tensor,
        stream: CUstream,
        num_elems: Int64,
    ):
        """Pick the launch configuration: the size tiers, except for swizzled scales with a fused
        activation when N does not fill the tiers' 1024-column CTAs."""
        if cutlass.const_expr(
            self.cfg.WITH_GEMM_SWIZZLED_SCALES and (self.cfg.WITH_ACT or self.cfg.WITH_DACT)
        ):
            if mX.shape[1] % 1024 != 0:
                # The swizzled path tiles the real (M, N) shape, so a 1024-column CTA overhangs
                # N and its idle lanes still pay for the activation math. 256-column CTAs divide
                # every eligible N (N % 128 == 0 and the tile is 2 x 128 columns).
                self._launch(
                    mX,
                    mA,
                    mO_row,
                    mS_row,
                    stream,
                    num_elems,
                    CTA_THREADS_Y=16,
                    CTA_THREADS_X=8,
                    NUM_TILES=2,
                    L2_CACHED_CTA_PERCENT=0,
                )
            else:
                self._launch_size_tiers(mX, mA, mO_row, mS_row, stream, num_elems)
        else:
            self._launch_size_tiers(mX, mA, mO_row, mS_row, stream, num_elems)

    @cute.jit
    def _launch_size_tiers(
        self,
        mX: cute.Tensor,
        mA: Optional[cute.Tensor],
        mO_row: cute.Tensor,
        mS_row: cute.Tensor,
        stream: CUstream,
        num_elems: Int64,
    ):
        """Pick the launch configuration for this many elements (cast_rowwise.cuh's tiers)."""
        output_bytes = num_elems
        if output_bytes <= 12 << 20:
            self._launch(
                mX,
                mA,
                mO_row,
                mS_row,
                stream,
                num_elems,
                CTA_THREADS_Y=8,
                CTA_THREADS_X=self._CTA_THREADS_X,
                NUM_TILES=1,
                L2_CACHED_CTA_PERCENT=self._SMALL_TIER_CACHED_PERCENT,
            )
        elif output_bytes <= 48 << 20:
            self._launch(
                mX,
                mA,
                mO_row,
                mS_row,
                stream,
                num_elems,
                CTA_THREADS_Y=8,
                CTA_THREADS_X=self._CTA_THREADS_X,
                NUM_TILES=2,
                L2_CACHED_CTA_PERCENT=self._SMALL_TIER_CACHED_PERCENT,
            )
        elif output_bytes <= 96 << 20:
            self._launch(
                mX,
                mA,
                mO_row,
                mS_row,
                stream,
                num_elems,
                CTA_THREADS_Y=4,
                CTA_THREADS_X=self._CTA_THREADS_X,
                NUM_TILES=2,
                L2_CACHED_CTA_PERCENT=self._L2_CACHED_CTA_PERCENT,
            )
        else:
            self._launch(
                mX,
                mA,
                mO_row,
                mS_row,
                stream,
                num_elems,
                CTA_THREADS_Y=8,
                CTA_THREADS_X=self._CTA_THREADS_X,
                NUM_TILES=2,
                L2_CACHED_CTA_PERCENT=self._L2_CACHED_CTA_PERCENT,
            )

    @cute.jit
    def _launch(
        self,
        mX: cute.Tensor,
        mA: Optional[cute.Tensor],
        mO: cute.Tensor,
        mS: cute.Tensor,
        stream: CUstream,
        num_elems: Int64,
        CTA_THREADS_Y: cutlass.Constexpr[int],
        CTA_THREADS_X: cutlass.Constexpr[int],
        NUM_TILES: cutlass.Constexpr[int],
        L2_CACHED_CTA_PERCENT: cutlass.Constexpr[int],
    ):
        assert 0 <= L2_CACHED_CTA_PERCENT <= 100
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
        check_bounds = M % CTA_ROWS != 0 or N % CTA_COLS != 0 or num_elems != Int64(M) * N
        grid = Int64(grid_rows) * grid_cols
        first_streaming_cta = grid * L2_CACHED_CTA_PERCENT // 100

        # Column tiles vary fastest, so consecutive CTAs read consecutive memory along a
        # row, as cast_rowwise.cuh's flat grid does. Putting row tiles on x instead
        # sends concurrent CTAs down a column strip, one 2 KiB piece per row, and costs
        # up to 9% at large shapes. Row tiles past grid.y's limit spill into grid.z.
        grid_y = cutlass.min(grid_rows, self._MAX_GRID_Y)
        grid_z = cute.ceil_div(grid_rows, self._MAX_GRID_Y)
        launch_grid = [grid_cols, grid_y, grid_z]
        kernel_args = (
            mX,
            mA,
            mO,
            mS,
            first_streaming_cta,
            num_elems,
            CTA_THREADS_Y,
            CTA_THREADS_X,
            NUM_TILES,
            L2_CACHED_CTA_PERCENT,
        )
        launch_args = {"grid": launch_grid, "block": [THREADS_PER_CTA], "stream": stream}
        stash_scale_to_smem = False
        if cutlass.const_expr(
            self.cfg.WITH_GEMM_SWIZZLED_SCALES and CTA_THREADS_Y % 4 == 0 and not self.cfg.WITH_DACT
        ):
            # Staged scale vectors need complete 128-row bands.
            stash_scale_to_smem = M % 128 == 0

        if grid > 0:
            if check_bounds:
                if stash_scale_to_smem:
                    self._launch_kernel(kernel_args, launch_args, True, True)
                else:
                    self._launch_kernel(kernel_args, launch_args, True, False)
            else:
                if stash_scale_to_smem:
                    self._launch_kernel(kernel_args, launch_args, False, True)
                else:
                    self._launch_kernel(kernel_args, launch_args, False, False)

    @cute.jit
    def _launch_kernel(
        self,
        kernel_args,
        launch_args,
        CHECK_BOUNDS: cutlass.Constexpr[bool],
        STASH_SCALE_TO_SMEM: cutlass.Constexpr[bool],
    ):
        self.kernel(
            *kernel_args, CHECK_BOUNDS=CHECK_BOUNDS, STASH_SCALE_TO_SMEM=STASH_SCALE_TO_SMEM
        ).launch(**launch_args)

    @cute.kernel
    def kernel(
        self,
        mX: cute.Tensor,
        mA: Optional[cute.Tensor],
        mO: cute.Tensor,
        mS: cute.Tensor,
        first_streaming_cta: Int64,
        num_elems: Int64,
        CTA_THREADS_Y: cutlass.Constexpr[int],
        CTA_THREADS_X: cutlass.Constexpr[int],
        NUM_TILES: cutlass.Constexpr[int],
        L2_CACHED_CTA_PERCENT: cutlass.Constexpr[int],
        CHECK_BOUNDS: cutlass.Constexpr[bool],
        STASH_SCALE_TO_SMEM: cutlass.Constexpr[bool],
    ):
        """Run the kernel body. Like cast_rowwise.cuh, there is no noop handling: the
        dispatcher only selects this kernel when no cast_noop flag is passed."""
        self._kernel_main(
            mX,
            mA,
            mO,
            mS,
            first_streaming_cta,
            num_elems,
            CTA_THREADS_Y=CTA_THREADS_Y,
            CTA_THREADS_X=CTA_THREADS_X,
            NUM_TILES=NUM_TILES,
            L2_CACHED_CTA_PERCENT=L2_CACHED_CTA_PERCENT,
            CHECK_BOUNDS=CHECK_BOUNDS,
            STASH_SCALE_TO_SMEM=STASH_SCALE_TO_SMEM,
        )

    @cute.jit
    def _kernel_main(
        self,
        mX: cute.Tensor,
        mA: Optional[cute.Tensor],
        mO: cute.Tensor,
        mS: cute.Tensor,
        first_streaming_cta: Int64,
        num_elems: Int64,
        CTA_THREADS_Y: cutlass.Constexpr[int],
        CTA_THREADS_X: cutlass.Constexpr[int],
        NUM_TILES: cutlass.Constexpr[int],
        L2_CACHED_CTA_PERCENT: cutlass.Constexpr[int],
        CHECK_BOUNDS: cutlass.Constexpr[bool],
        STASH_SCALE_TO_SMEM: cutlass.Constexpr[bool],
    ):
        """Construct CTA tile views and run the specialized quantization."""
        # Dispatch checks 32-byte input alignment before selecting this kernel.
        gX = cute.make_tensor(mX.iterator.align(32), mX.layout)
        if cutlass.const_expr(self.cfg.WITH_DACT):
            mA = cute.make_tensor(mA.iterator.align(32), mA.layout)
        # grid.x walks column tiles; grid.y and grid.z together walk row tiles.
        bidx, bidy, bidz = cute.arch.block_idx()
        cta_row = bidz * self._MAX_GRID_Y + bidy
        cta_coord = (cta_row, bidx)
        M, N = mX.shape
        TILE_ROWS = CTA_THREADS_Y
        TILE_COLS = CTA_THREADS_X * self._ELEMENTS_PER_THREAD
        CTA_ROWS = TILE_ROWS
        CTA_COLS = TILE_COLS * NUM_TILES

        gS = mS
        # Use the swizzled layout as is. Composing it with (M, N // 32) layouts to flatten
        # its shape made CuTe emit runtime integer div/mod on every scale address,
        # about 7x this kernel's instruction count.
        if cutlass.const_expr(self.cfg.WITH_GEMM_SWIZZLED_SCALES):
            gS, _ = derive_swizzled_scale_layout(M, N, True, False, mS, None)

        if cutlass.const_expr(STASH_SCALE_TO_SMEM):
            # Apply the same permutation to input, output and scale so it becomes transparent and
            # other code is unaware of this permutation pattern
            # However, this this permutation, the scale bytes emitted by a CTA are all in this CTA's scale
            # tile partition instead of scatter everywhere, so we can vectorize the flush to GMEM
            # For example, with a 128x128 input:
            #   gX & mO: (128,128):(128,1) -> ((4,32),128):((4096,128),1)
            #   gS: ((32,4,1),(4,1)):((16,4,512),(1,512)) -> ((4,32), 4):((4, 16),1)
            row_perm = cute.make_layout((4, 32, M // 128), stride=(32, 1, 128))
            gX = cute.composition(gX, (row_perm, None))
            mO = cute.composition(mO, (row_perm, None))
            gS = cute.composition(gS, (row_perm, None))

        CTA_TILER = (CTA_ROWS, CTA_COLS)
        CTA_SCALE_TILER = (CTA_ROWS, CTA_COLS // 32)
        gX_cta = cute.local_tile(gX, CTA_TILER, cta_coord)
        gO_cta = cute.local_tile(mO, CTA_TILER, cta_coord)
        gS_cta = cute.local_tile(gS, CTA_SCALE_TILER, cta_coord)

        fraction = Float32(1.0)
        if cutlass.const_expr(L2_CACHED_CTA_PERCENT > 0):
            fraction = Float32(0.0)
            # A linear index is needed only for the L2 policy split. It follows launch
            # order, so the cached CTAs are the first ones scheduled.
            gdx, _, _ = cute.arch.grid_dim()
            cta_id = Int64(cta_row) * gdx + bidx
            if cta_id >= first_streaming_cta:
                fraction = Float32(1.0)
        input_policy = create_l2_policy(False, fraction)
        output_policy = create_l2_policy(True)

        # Divide CTA_TILER to multiple TILER shape TILES
        TILER = (TILE_ROWS, TILE_COLS)
        SCALE_TILER = (TILE_ROWS, TILE_COLS // 32)
        # There is only one tile on the row dimension because CTA_ROWS == TILE_ROWS.
        gX_tiles = cute.local_tile(gX_cta, TILER, (0, None))
        gA_tiles = None
        if cutlass.const_expr(self.cfg.WITH_DACT):
            gA_tiles = cute.local_tile(cute.local_tile(mA, CTA_TILER, cta_coord), TILER, (0, None))
        gO_tiles = cute.local_tile(gO_cta, TILER, (0, None))

        if cutlass.const_expr(STASH_SCALE_TO_SMEM):
            # Preserve gS_cta's (4 rows, 4 columns):(4, 1) block ordering for the flush.
            sScale_scratch = cutlass.memory.SmemAllocator().allocate_tensor(
                mS.element_type, cute.make_layout_like(gS_cta), byte_alignment=16
            )
            # The tile quantizer writes scales to SMEM for the cooperative vectorized flush.
            mS_tiles = cute.local_tile(sScale_scratch, SCALE_TILER, (0, None))
        else:
            # The tile quantizer writes scales directly to GMEM.
            mS_tiles = cute.local_tile(gS_cta, SCALE_TILER, (0, None))

        # The last grid.z slice can overshoot the row tiles; those CTAs have no work.
        if cta_row * CTA_ROWS < M:
            quantize_mxfp8_gmem_tile_rowwise_2lane(
                gX_tiles,
                gA_tiles,
                gO_tiles,
                mS_tiles,
                N,
                ACTIVATION=self.cfg.ACTIVATION,
                WITH_ACT=self.cfg.WITH_ACT,
                WITH_DACT=self.cfg.WITH_DACT,
                DTYPE=self.cfg.DTYPE,
                FP8_DTYPE=self.cfg.FP8_DTYPE,
                MAX_NORM_RCP=self.cfg.MAX_NORM_RCP,
                CTA_THREADS_Y=CTA_THREADS_Y,
                CTA_THREADS_X=CTA_THREADS_X,
                ELEMENTS_PER_THREAD=self._ELEMENTS_PER_THREAD,
                NUM_TILES=NUM_TILES,
                CHECK_BOUNDS=CHECK_BOUNDS,
                input_policy=input_policy,
                output_policy=output_policy,
                cta_row=cta_coord[0],
                cta_col=cta_coord[1],
                num_elems=num_elems,
            )

            if cutlass.const_expr(STASH_SCALE_TO_SMEM):
                cute.arch.sync_threads()
                _flush_scales_to_gmem(
                    sScale_scratch,
                    cute.make_tensor(gS_cta.iterator.align(16), gS_cta.layout),
                    cta_row,
                    bidx,
                    M,
                    N // 32,
                    CTA_ROWS,
                    CTA_COLS // 32,
                )
