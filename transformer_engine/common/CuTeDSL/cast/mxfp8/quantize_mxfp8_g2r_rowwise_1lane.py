# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""One-lane GMEM-to-RMEM rowwise MXFP8 cast with vectorized global I/O."""

# Local @cute.struct classes are shared-memory storage descriptors.
# pylint: disable=missing-class-docstring

from typing import Optional, Type

import cutlass
from cutlass import cute
from cutlass import Float8E8M0FNU, Int32, Uint32, Uint8
from cuda.bindings.driver import CUstream  # pylint: disable=no-name-in-module

from transformer_engine.common.CuTeDSL.utils import (
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
    noop_flag_is_set,
)


class MXFP8QuantizeG2RRowwise1LaneKernel(MXFP8QuantizeKernelBase):
    """Specialized cast-only ROWWISE-only MXFP8 kernel.

    Requires N % 128 == 0 (full vectorizable column chunks).

    Plain rowwise-only quantize. Each thread owns one 32-element MXFP8 chunk and
    uses vectorized global loads/stores (no TMA used)."""

    _TILE_ROWS = 4
    _TILE_COLS = 1024
    _THREADS_PER_CTA = 128

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
        mNoop: cute.Pointer,  # f32 cast_noop flag; may be null, checked on device
        mDActInput: Optional[cute.Tensor],  # Unused, kept for API compatibility
        mWorkspace: Optional[cute.Tensor],  # Unused, kept for API compatibility
        stream: CUstream,
    ):
        if cutlass.const_expr(CUTEDSL_DEBUG_LOGGING):
            cute.printf(
                f"[CuTeDSL] MXFP8QuantizeG2RRowwise1LaneKernel.__call__() with config: {self.cfg}\n"
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
            mNoop,
            mX.element_type,
        ).launch(grid=grid, block=block, stream=stream)

    @cute.kernel
    def kernel(
        self,
        mX: cute.Tensor,
        mO_row: cute.Tensor,
        mS_row: cute.Tensor,
        mNoop: cute.Pointer,
        DTYPE: cutlass.Constexpr[Type[cutlass.Numeric]],
    ):
        """Device entry: no-op the CTA when the noop flag is set, else run the quantize main loop."""
        if not noop_flag_is_set(mNoop):
            self._kernel_main(mX, mO_row, mS_row, DTYPE)

    @cute.jit
    def _kernel_main(
        self,
        mX: cute.Tensor,
        mO_row: cute.Tensor,
        mS_row: cute.Tensor,
        DTYPE: cutlass.Constexpr[Type[cutlass.Numeric]],
    ):
        """Device entry for the specialized rowwise-only cast kernel (vectorized global loads/stores, no TMA)."""
        tidx, _, _ = cute.arch.thread_idx()
        bidx, bidy, _ = cute.arch.block_idx()
        M = mX.shape[0]
        N = mX.shape[1]

        # Each thread handles one 32-element MXFP8 chunk (= one scale block).
        # The 128 threads in the CTA are grouped as (4, 32), so they cover a
        # (4, 1024) input tile and the matching (4, 32) scale tile.
        CTA_Y = self._TILE_ROWS
        CTA_X = self._TILE_COLS // MXFP8_BLOCK_SCALING_SIZE
        tiler, tv_layout = cute.make_layout_tv(
            thr_layout=cute.make_layout((CTA_Y, CTA_X), stride=(CTA_X, 1)),
            val_layout=cute.make_layout(
                (1, MXFP8_BLOCK_SCALING_SIZE), stride=(MXFP8_BLOCK_SCALING_SIZE, 1)
            ),
        )
        tiler_scale, tv_layout_scale = cute.make_layout_tv(
            thr_layout=cute.make_layout((CTA_Y, CTA_X), stride=(CTA_X, 1)),
            val_layout=cute.make_layout((1, 1), stride=(1, 1)),
        )

        # Select the tile that belongs to this CTA, then the fragment per thread.
        mX_tile = cute.local_tile(mX, tiler, (bidy, bidx))
        mO_tile = cute.local_tile(mO_row, tiler, (bidy, bidx))
        mS_tile = cute.local_tile(mS_row, tiler_scale, (bidy, bidx))
        mX_thread = cute.composition(mX_tile, tv_layout)[tidx, None]
        mO_thread = cute.composition(mO_tile, tv_layout)[tidx, None]
        mS_thread = cute.composition(mS_tile, tv_layout_scale)[tidx, None]

        rX_thread = cute.make_rmem_tensor(
            cute.make_layout((1, MXFP8_BLOCK_SCALING_SIZE), stride=(MXFP8_BLOCK_SCALING_SIZE, 1)),
            dtype=DTYPE,
        )
        abs_max_x2 = abs_max_x2_f16 if DTYPE is cutlass.Float16 else abs_max_x2_bf16
        x2_lo_to_f32 = x2_lo_to_f32_f16 if DTYPE is cutlass.Float16 else x2_lo_to_f32_bf16
        x2_hi_to_f32 = x2_hi_to_f32_f16 if DTYPE is cutlass.Float16 else x2_hi_to_f32_bf16
        mul_cvt4 = mul_f32x2_cvt_packed16x4_to_fp8x4(DTYPE, self.cfg.FP8_DTYPE)
        rX_i32 = cute.make_tensor(
            cute.recast_ptr(rX_thread.iterator, dtype=Int32),
            cute.make_layout((MXFP8_BLOCK_SCALING_SIZE // 2,), stride=(1,)),
        )
        rO_thread = cute.make_rmem_tensor(
            cute.make_layout((1, MXFP8_BLOCK_SCALING_SIZE), stride=(MXFP8_BLOCK_SCALING_SIZE, 1)),
            dtype=self.cfg.FP8_DTYPE,
        )
        rO_u32 = cute.make_tensor(
            cute.recast_ptr(rO_thread.iterator, dtype=Uint32),
            cute.make_layout(
                (MXFP8_BLOCK_SCALING_SIZE // 4,), stride=(1,)
            ),  # Unit is Uint32, divide by 4 here
        )

        sS_thread = None
        if cutlass.const_expr(self._STASH_SCALE_TO_SMEM):

            @cute.struct
            class SharedStorage:
                buf: cute.struct.Align[cute.struct.MemRange[Float8E8M0FNU, CTA_Y * CTA_X], 16]

            storage = cutlass.utils.SmemAllocator().allocate(SharedStorage)
            sScale = storage.buf.get_tensor(cute.make_layout((CTA_Y, CTA_X), stride=(CTA_X, 1)))
            # sScale is (CTA_Y, CTA_X):(CTA_X, 1), which is the same layout as tv_layout_scale
            # so sS_thread is really just a 1-element e8m0 buffer for this thread's scale.
            sS_thread = cute.composition(sScale, tv_layout_scale)[tidx, None]
            # Zero first so padding columns (cols past N/32 in the padded scale
            # matrix) flush as 0 and we never read uninitialized smem.
            # Raw 0x00: e8m0 has no zero value to convert from.
            sS_thread[0] = Uint8(0).bitcast(Float8E8M0FNU)
            cute.arch.sync_threads()

        row = bidy * self._TILE_ROWS + tidx // CTA_X
        col = bidx * self._TILE_COLS + (tidx % CTA_X) * MXFP8_BLOCK_SCALING_SIZE
        if row < M and col < N:
            cute.autovec_copy(mX_thread, rX_thread)
            amax_2x = rX_i32[0]
            # Skip the first iteration because we assigned rX_i32[0] to amax_2x already
            for i in cutlass.range_constexpr(1, MXFP8_BLOCK_SCALING_SIZE // 2):
                amax_2x = abs_max_x2(amax_2x, rX_i32[i])
            amax = cute.arch.fmax(fabs_f32(x2_lo_to_f32(amax_2x)), fabs_f32(x2_hi_to_f32(amax_2x)))

            biased_exp = cvt_f32_to_fp8e8m0fnu(amax * self.cfg.MAX_NORM_RCP)
            if cutlass.const_expr(self._STASH_SCALE_TO_SMEM):
                sS_thread[0] = biased_exp
            else:
                mS_thread[0] = biased_exp

            # Rescale + FP8 cast, 4 elements per fused mul_cvt (one uint32 out),
            # then a vectorized store. Mirrors CUDA's _use_cvt_4x path.
            inv_scale = exp2f_rcp(biased_exp)
            scale_2x = pack_f32x2(inv_scale, inv_scale)
            for i in cutlass.range_constexpr(MXFP8_BLOCK_SCALING_SIZE // 4):
                rO_u32[i] = mul_cvt4(rX_i32[2 * i], rX_i32[2 * i + 1], scale_2x)
            cute.autovec_copy(rO_thread, mO_thread)

        # Cooperative wide flush of the staged scales where padding columns flush as 0.
        if cutlass.const_expr(self._STASH_SCALE_TO_SMEM):
            cute.arch.sync_threads()
            # Use cute.size instead of .shape[1] because under the swizzled layout mode 1 is
            # a nested tuple ((4, num_tiles_SC)), not a plain int.
            padded_cols = cute.size(mS_row, mode=[1])
            if cutlass.const_expr(self.cfg.WITH_GEMM_SWIZZLED_SCALES):
                # Swizzled rowwise scale layout is ((32, 4, num_tiles_M), (4, num_tiles_SC)):((16, 4, num_tiles_SC * 512), (1, 512))
                # so we have at most 4 elements continuous which is our vectorized store width
                self._flush_scales_to_gmem(sScale, mS_tile, tidx, bidx, bidy, M, padded_cols, 4)
            elif padded_cols % 16 == 0:
                # If columns is divisible by 16, use 16 bytes as the vectorized store width
                self._flush_scales_to_gmem(sScale, mS_tile, tidx, bidx, bidy, M, padded_cols, 16)
            else:
                # Otherwise use 4 bytes as the vectorized store width.
                # Note our fake tensor requires 4 divisibility so this is enforced as long as you can get here
                self._flush_scales_to_gmem(sScale, mS_tile, tidx, bidx, bidy, M, padded_cols, 4)

    @cute.jit
    def _flush_scales_to_gmem(
        self,
        sScale: cute.Tensor,
        mS_tile: cute.Tensor,
        tidx: Int32,
        bidx: Int32,
        bidy: Int32,
        M: Int32,
        padded_cols: Int32,
        width: cutlass.Constexpr[int],
    ):
        """Flush the staged (CTA_Y, CTA_X) scale tile to gmem with vectorized stores."""
        CTA_Y = self._TILE_ROWS
        CTA_X = self._TILE_COLS // MXFP8_BLOCK_SCALING_SIZE
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
            thread_y = bidy * CTA_Y + tidx // active_threads
            thread_x = bidx * CTA_X + (tidx % active_threads) * width
            if thread_y < M and thread_x < padded_cols:
                cute.autovec_copy(
                    cute.composition(sScale, tv_flush)[tidx, None],
                    cute.composition(mS_tile, tv_flush)[tidx, None],
                )
