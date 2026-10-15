# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""One-lane GMEM-to-RMEM rowwise MXFP8 cast with vectorized global I/O."""

# Local @cute.struct classes are shared-memory storage descriptors.
# pylint: disable=missing-class-docstring

from typing import Optional, Type

import cutlass
from cutlass import cute
from cutlass import Boolean, Float8E8M0FNU, Int32, Uint32, Uint8
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
    mul_f32x2_cvt_packed16x4_to_fp8x4,
)
from .quantize_mxfp8_common import (
    CUTEDSL_DEBUG_LOGGING,
    MXFP8QuantizeConfig,
    MXFP8QuantizeKernelBase,
    MXFP8_BLOCK_SCALING_SIZE,
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
    width: cutlass.Constexpr[int],
):
    """Flush the staged (CTA_Y, CTA_X) scale tile to gmem with vectorized stores."""
    tidx, _, _ = cute.arch.thread_idx()
    # Previously each threads has 1 byte, but now we are doing vectorized store,
    # which means only a subset of threads will need to issue the store while other threads are not used.
    active_threads = CTA_X // width
    _, tv_flush = cute.make_layout_tv(
        thr_layout=cute.make_layout((CTA_Y, active_threads), stride=(active_threads, 1)),
        val_layout=cute.make_layout((1, width), stride=(width, 1)),
    )
    # We only need to use a subset of threads with shape (CTA_Y, active_threads) to write
    # so if the thread is outside of this subset, it will remain inactive
    if tidx < CTA_Y * active_threads:
        # Absolute position of the scale vector to write in the GMEM buffer
        thread_y = cta_row * CTA_Y + tidx // active_threads
        thread_x = cta_col * CTA_X + (tidx % active_threads) * width
        if thread_y < M and thread_x < padded_cols:
            cute.autovec_copy(
                cute.composition(sScale, tv_flush)[tidx, None],
                cute.composition(gS_tile, tv_flush)[tidx, None],
            )


@cute.jit
def quantize_mxfp8_gmem_tile_rowwise_1lane(
    gX_tile: cute.Tensor,
    gO_tile: cute.Tensor,
    gS_tile: cute.Tensor,
    M: Int32,
    N: Int32,
    padded_cols: Int32,
    cta_row: Int32,
    cta_col: Int32,
    DTYPE: cutlass.Constexpr[Type[cutlass.Numeric]],
    FP8_DTYPE: cutlass.Constexpr[Type[cutlass.Numeric]],
    MAX_NORM_RCP: cutlass.Constexpr[float],
    WITH_GEMM_SWIZZLED_SCALES: cutlass.Constexpr[bool],
    TILE_ROWS: cutlass.Constexpr[int],
    TILE_COLS: cutlass.Constexpr[int],
    STASH_SCALE_TO_SMEM: cutlass.Constexpr[bool],
):
    """Quantize one (TILE_ROWS, TILE_COLS) GMEM tile rowwise with TILE_ROWS * TILE_COLS / 32 threads.

    Each thread handles one 32-element MXFP8 chunk (= one scale block). (cta_row, cta_col) is the
    tile coordinate, used with M and N for bounds checks; padded_cols is the column count of the
    (padded) scale tensor that gS_tile is a tile of.
    """
    tidx, _, _ = cute.arch.thread_idx()

    # The CTA threads are grouped as (CTA_Y, CTA_X), so they cover a (TILE_ROWS, TILE_COLS)
    # input tile and the matching (CTA_Y, CTA_X) scale tile.
    CTA_Y = TILE_ROWS
    CTA_X = TILE_COLS // MXFP8_BLOCK_SCALING_SIZE
    _, tv_layout = cute.make_layout_tv(
        thr_layout=cute.make_layout((CTA_Y, CTA_X), stride=(CTA_X, 1)),
        val_layout=cute.make_layout(
            (1, MXFP8_BLOCK_SCALING_SIZE), stride=(MXFP8_BLOCK_SCALING_SIZE, 1)
        ),
    )
    _, tv_layout_scale = cute.make_layout_tv(
        thr_layout=cute.make_layout((CTA_Y, CTA_X), stride=(CTA_X, 1)),
        val_layout=cute.make_layout((1, 1), stride=(1, 1)),
    )

    gX_thread = cute.composition(gX_tile, tv_layout)[tidx, None]
    gO_thread = cute.composition(gO_tile, tv_layout)[tidx, None]
    gS_thread = cute.composition(gS_tile, tv_layout_scale)[tidx, None]

    rX_thread = cute.make_rmem_tensor(
        cute.make_layout((1, MXFP8_BLOCK_SCALING_SIZE), stride=(MXFP8_BLOCK_SCALING_SIZE, 1)),
        dtype=DTYPE,
    )
    abs_max_x2 = abs_max_x2_f16 if DTYPE is cutlass.Float16 else abs_max_x2_bf16
    x2_lo_to_f32 = x2_lo_to_f32_f16 if DTYPE is cutlass.Float16 else x2_lo_to_f32_bf16
    x2_hi_to_f32 = x2_hi_to_f32_f16 if DTYPE is cutlass.Float16 else x2_hi_to_f32_bf16
    mul_cvt4 = mul_f32x2_cvt_packed16x4_to_fp8x4(DTYPE, FP8_DTYPE)
    rX_i32 = cute.make_tensor(
        cute.recast_ptr(rX_thread.iterator, dtype=Int32),
        cute.make_layout((MXFP8_BLOCK_SCALING_SIZE // 2,), stride=(1,)),
    )
    rO_thread = cute.make_rmem_tensor(
        cute.make_layout((1, MXFP8_BLOCK_SCALING_SIZE), stride=(MXFP8_BLOCK_SCALING_SIZE, 1)),
        dtype=FP8_DTYPE,
    )
    rO_u32 = cute.make_tensor(
        cute.recast_ptr(rO_thread.iterator, dtype=Uint32),
        cute.make_layout(
            (MXFP8_BLOCK_SCALING_SIZE // 4,), stride=(1,)
        ),  # Unit is Uint32, divide by 4 here
    )

    sS_thread = None
    if cutlass.const_expr(STASH_SCALE_TO_SMEM):

        @cute.struct
        class SharedStorage:
            buf: cute.struct.Align[cute.struct.MemRange[Float8E8M0FNU, CTA_Y * CTA_X], 16]

        storage = cutlass.memory.SmemAllocator().allocate(SharedStorage)
        sScale = storage.buf.get_tensor(cute.make_layout((CTA_Y, CTA_X), stride=(CTA_X, 1)))
        # sScale is (CTA_Y, CTA_X):(CTA_X, 1), which is the same layout as tv_layout_scale
        # so sS_thread is really just a 1-element e8m0 buffer for this thread's scale.
        sS_thread = cute.composition(sScale, tv_layout_scale)[tidx, None]
        # Zero first so padding columns (cols past N/32 in the padded scale
        # matrix) flush as 0 and we never read uninitialized smem.
        # Raw 0x00: e8m0 has no zero value to convert from.
        sS_thread[0] = Uint8(0).bitcast(Float8E8M0FNU)
        cute.arch.sync_threads()

    row = cta_row * TILE_ROWS + tidx // CTA_X
    col = cta_col * TILE_COLS + (tidx % CTA_X) * MXFP8_BLOCK_SCALING_SIZE
    if row < M and col < N:
        cute.autovec_copy(gX_thread, rX_thread)
        amax_2x = rX_i32[0]
        # Skip the first iteration because we assigned rX_i32[0] to amax_2x already
        for i in cutlass.range_constexpr(1, MXFP8_BLOCK_SCALING_SIZE // 2):
            amax_2x = abs_max_x2(amax_2x, rX_i32[i])
        amax = cute.arch.fmax(fabs_f32(x2_lo_to_f32(amax_2x)), fabs_f32(x2_hi_to_f32(amax_2x)))

        biased_exp = cvt_f32_to_fp8e8m0fnu(amax * MAX_NORM_RCP)
        if cutlass.const_expr(STASH_SCALE_TO_SMEM):
            sS_thread[0] = biased_exp
        else:
            gS_thread[0] = biased_exp

        # Rescale + FP8 cast, 4 elements per fused mul_cvt (one uint32 out),
        # then a vectorized store. Mirrors CUDA's _use_cvt_4x path.
        inv_scale = exp2f_rcp(biased_exp)
        scale_2x = pack_f32x2(inv_scale, inv_scale)
        for i in cutlass.range_constexpr(MXFP8_BLOCK_SCALING_SIZE // 4):
            rO_u32[i] = mul_cvt4(rX_i32[2 * i], rX_i32[2 * i + 1], scale_2x)
        cute.autovec_copy(rO_thread, gO_thread)

    # Cooperative wide flush of the staged scales where padding columns flush as 0.
    if cutlass.const_expr(STASH_SCALE_TO_SMEM):
        cute.arch.sync_threads()
        if cutlass.const_expr(WITH_GEMM_SWIZZLED_SCALES):
            # Swizzled rowwise scale layout is ((32, 4, num_tiles_M), (4, num_tiles_SC)):((16, 4, num_tiles_SC * 512), (1, 512))
            # so we have at most 4 elements continuous which is our vectorized store width
            _flush_scales_to_gmem(
                sScale, gS_tile, cta_row, cta_col, M, padded_cols, CTA_Y, CTA_X, 4
            )
        elif padded_cols % 16 == 0:
            # If columns is divisible by 16, use 16 bytes as the vectorized store width
            _flush_scales_to_gmem(
                sScale, gS_tile, cta_row, cta_col, M, padded_cols, CTA_Y, CTA_X, 16
            )
        else:
            # Otherwise use 4 bytes as the vectorized store width.
            # Note our fake tensor requires 4 divisibility so this is enforced as long as you can get here
            _flush_scales_to_gmem(
                sScale, gS_tile, cta_row, cta_col, M, padded_cols, CTA_Y, CTA_X, 4
            )


class MXFP8QuantizeRowwise1LaneG2RKernel(MXFP8QuantizeKernelBase):
    """Specialized cast-only ROWWISE-only MXFP8 kernel.

    Requires N % 128 == 0 (full vectorizable column chunks).

    Plain rowwise-only quantize. Each thread owns one 32-element MXFP8 chunk and
    uses vectorized global loads/stores (no TMA used)."""

    _TILE_ROWS = 4
    _TILE_COLS = 1024
    _THREADS_PER_CTA = 128

    @staticmethod
    def supports(cfg: MXFP8QuantizeConfig) -> bool:
        return cfg.is_plain_cast() and is_packed16(cfg.DTYPE) and cfg.ROWWISE and not cfg.COLWISE

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
        M, N = mX.shape
        # N divisible by 128 for vectorized stores, and grid.y within limits.
        return (
            N % 128 == 0
            and cute.ceil_div(M, MXFP8QuantizeRowwise1LaneG2RKernel._TILE_ROWS) <= 65535
            and mNoop.toint() == 0
        )

    def __init__(self, cfg: MXFP8QuantizeConfig):
        self.cfg = cfg
        # If True, then this kernel will first write each thread's scale byte to a shared memory buffer,
        # then utilize vectorized store to flush the buffer to global memory.
        self._STASH_SCALE_TO_SMEM = True  # Hardcode to true for now

    @cute.jit
    def __call__(
        self,
        mX: cute.Tensor,
        mO_row: Optional[cute.Tensor],
        mS_row: Optional[cute.Tensor],
        mO_col: Optional[cute.Tensor],
        mS_col: Optional[cute.Tensor],  # Unused, kept for API compatibility
        mAmax: Optional[cute.Tensor],  # Unused, kept for API compatibility
        mNoop: cute.Pointer,  # Unused: the dispatcher routes noop calls to the general kernel
        mDActInput: Optional[cute.Tensor],  # Unused, kept for API compatibility
        mWorkspace: Optional[cute.Tensor],  # Unused, kept for API compatibility
        stream: CUstream,
    ):
        if cutlass.const_expr(CUTEDSL_DEBUG_LOGGING):
            cute.printf(
                f"[CuTeDSL] MXFP8QuantizeRowwise1LaneG2RKernel.__call__() with config: {self.cfg}\n"
            )

        M = mX.shape[0]
        N = mX.shape[1]

        if cutlass.const_expr(self.cfg.WITH_GEMM_SWIZZLED_SCALES):
            mS_row, _ = derive_swizzled_scale_layout(M, N, True, False, mS_row, None)

        grid = [
            cute.ceil_div(Int32(N), self._TILE_COLS),
            cute.ceil_div(M, self._TILE_ROWS),
        ]
        block = [self._THREADS_PER_CTA]

        self.kernel(
            mX,
            mO_row,
            mS_row,
        ).launch(grid=grid, block=block, stream=stream)

    @cute.kernel
    def kernel(
        self,
        mX: cute.Tensor,
        mO_row: cute.Tensor,
        mS_row: cute.Tensor,
    ):
        """Device entry. Like the CUDA specialized cast kernels it does not consume the
        cast_noop flag: the dispatcher routes calls with one to the general kernel."""
        self._kernel_main(mX, mO_row, mS_row)

    @cute.jit
    def _kernel_main(
        self,
        mX: cute.Tensor,
        mO_row: cute.Tensor,
        mS_row: cute.Tensor,
    ):
        """Device entry for the specialized rowwise-only cast kernel (vectorized global loads/stores, no TMA)."""
        bidx, bidy, _ = cute.arch.block_idx()
        M = mX.shape[0]
        N = mX.shape[1]

        TILER = (self._TILE_ROWS, self._TILE_COLS)
        SCALE_TILER = (self._TILE_ROWS, self._TILE_COLS // MXFP8_BLOCK_SCALING_SIZE)
        gX_tile = cute.local_tile(mX, TILER, (bidy, bidx))
        gO_tile = cute.local_tile(mO_row, TILER, (bidy, bidx))
        gS_tile = cute.local_tile(mS_row, SCALE_TILER, (bidy, bidx))

        quantize_mxfp8_gmem_tile_rowwise_1lane(
            gX_tile,
            gO_tile,
            gS_tile,
            M,
            N,
            # Use cute.size instead of .shape[1] because under the swizzled layout mode 1 is
            # a nested tuple ((4, num_tiles_SC)), not a plain int.
            cute.size(mS_row, mode=[1]),
            bidy,
            bidx,
            DTYPE=self.cfg.DTYPE,
            FP8_DTYPE=self.cfg.FP8_DTYPE,
            MAX_NORM_RCP=self.cfg.MAX_NORM_RCP,
            WITH_GEMM_SWIZZLED_SCALES=self.cfg.WITH_GEMM_SWIZZLED_SCALES,
            TILE_ROWS=self._TILE_ROWS,
            TILE_COLS=self._TILE_COLS,
            STASH_SCALE_TO_SMEM=self._STASH_SCALE_TO_SMEM,
        )
