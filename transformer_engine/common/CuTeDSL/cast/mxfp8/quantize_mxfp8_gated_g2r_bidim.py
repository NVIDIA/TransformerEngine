# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Register-resident gated-activation MXFP8 cast in CuTeDSL.

A CTA covers 256 output columns of one 32-row band. Each thread loads its rows of the act and
gate halves (and of the incoming gradient, in backward) into registers, computes the gated op
there with the elementwise math of the shared-memory port (quantize_mxfp8_gated_g2s.py, which
mirrors gated_mxfp8.cuh), rounds it to BF16 as the CUDA kernels do, and quantizes the band in
both directions from registers with the bidimensional register-tile quantizer
(quantize_mxfp8_g2r_bidim.py). The backward quantizes its two output halves one after the other
from the same band. Both write the bytes gated_mxfp8.cuh writes.
"""

from typing import Optional

import cutlass
from cutlass import cute
from cutlass import Boolean, Float32, Int32, Int64
from cuda.bindings.driver import CUstream  # pylint: disable=no-name-in-module

from transformer_engine.common.CuTeDSL.utils import (
    create_l2_policy,
    cvt_f32x2_to_bf16x2,
    max_num_bits_per_copy_ld_global,
    x2_hi_to_f32_bf16,
    x2_lo_to_f32_bf16,
)
from .quantize_mxfp8_common import (
    CUTEDSL_DEBUG_LOGGING,
    MXFP8GatedQuantizeConfig,
    derive_swizzled_scale_layout,
)
from .quantize_mxfp8_g2r_bidim import (
    quantize_mxfp8_rmem_tile_bidimensional,
    thread_tile_bounds,
    thread_tile_layout,
)
from .quantize_mxfp8_gated_g2s import _split_gated_scales, gated_elementwise_op


@cute.jit
def _row_tile(
    mT: cute.Tensor,
    row_tile: Int32,
    cta_col: Int32,
    TILER: cutlass.Constexpr,
    ALIGNMENT: cutlass.Constexpr[int],
) -> cute.Tensor:
    """The (row_tile, cta_col) tile of a row-major mT, which may be a column slice of a wider
    tensor (row stride 2N): the tile's base takes the 64-bit product once and offsets within the
    tile are 32-bit. Every row stride is N or 2N, and the C++ dispatcher requires N % 32 == 0."""
    row_stride = cute.assume(Int32(mT.layout.stride[0]), divby=32)
    base = Int64(row_tile) * Int64(TILER[0]) * Int64(row_stride) + Int64(cta_col) * Int64(TILER[1])
    return cute.make_tensor(
        (mT.iterator + base).align(ALIGNMENT),
        cute.make_layout(TILER, stride=(row_stride, 1)),
    )


class MXFP8GatedQuantizeG2RKernel:
    """Register-resident gated MXFP8 cast: BF16 [M, 2N] -> E4M3 or E5M2, both directions or
    columnwise only, any M, N % 32 == 0 (the C++ dispatcher's requirement)."""

    _CTA_THREADS_Y = 8
    _CTA_THREADS_X = 32
    _ELEMENTS_Y_PER_THREAD = 4  # 8 warps x 4 rows = one 32-row band
    _ELEMENTS_X_PER_THREAD = 8  # 32 lanes x 8 columns = 256 columns
    _ROWS_IN_FLIGHT = 4  # input rows loaded at a time

    @staticmethod
    def supports(cfg: MXFP8GatedQuantizeConfig) -> bool:
        """BF16 with columnwise output (rowwise optional); GEMM-swizzled scales in the forward
        only, like the shared-memory port."""
        return (
            cfg.DTYPE is cutlass.BFloat16
            and cfg.COLWISE
            and not (cfg.IS_BWD and cfg.WITH_GEMM_SWIZZLED_SCALES)
        )

    @staticmethod
    @cute.jit
    def eligible_to_dispatch(
        mGatedInput: cute.Tensor,
        mS_col: cute.Tensor,
    ) -> Boolean:
        """Whether the request fits this kernel. The C++ dispatcher checks 16-byte alignment of
        the data pointers. Columnwise scales are 8-byte aligned, and grid.y stays within limits."""
        M = mGatedInput.shape[0]
        return (M + 31) // 32 <= 65535 and mS_col.iterator.toint() % 8 == 0

    def __init__(self, cfg: MXFP8GatedQuantizeConfig):
        assert self.supports(cfg), f"{cfg} is not supported by {type(self).__name__}"
        self.cfg = cfg
        # 4 resident CTAs per SM: 64 registers per thread, enough to load all 4 input rows at once.
        self._MIN_BLOCKS_PER_MP = 4

    @cute.jit
    def __call__(
        self,
        mGatedInput: cute.Tensor,  # [M, 2N] = [act | gate]
        mGrad: Optional[cute.Tensor],  # [M, N], backward only
        mO_row: Optional[cute.Tensor],  # [M, C], C = N (forward) or 2N (backward)
        mS_row: Optional[cute.Tensor],
        mO_col: cute.Tensor,  # [M, C], not transposed
        mS_col: cute.Tensor,
        limit: Optional[Float32],  # Clamped SwiGLU only
        alpha: Optional[Float32],  # Clamped SwiGLU only
        glu_linear_offset: Optional[Float32],  # Clamped SwiGLU only
        beta1: Optional[Float32],  # SiTUGLU only
        beta2: Optional[Float32],  # SiTUGLU only
        stream: CUstream,
    ):
        if cutlass.const_expr(CUTEDSL_DEBUG_LOGGING):
            cute.printf(
                f"[CuTeDSL] MXFP8GatedQuantizeG2RKernel.__call__() with config: {self.cfg}\n"
            )
        M = mGatedInput.shape[0]
        N = mO_col.shape[1] if not self.cfg.IS_BWD else mGrad.shape[1]
        TILE_ROWS = self._CTA_THREADS_Y * self._ELEMENTS_Y_PER_THREAD
        TILE_COLS = self._CTA_THREADS_X * self._ELEMENTS_X_PER_THREAD

        # eligible_to_dispatch checks this alignment.
        mS_col = cute.make_tensor(mS_col.iterator.align(8), mS_col.layout)
        mAct = cute.local_tile(mGatedInput, (M, N), (0, 0))
        mGate = cute.local_tile(mGatedInput, (M, N), (0, 1))

        # Output parts: the forward output, or the two halves [d act | d gate] of the backward one.
        mO_row1 = mS_row1 = mO_col1 = mS_col1 = None
        if cutlass.const_expr(self.cfg.IS_BWD):
            mS_row0, mS_row1, mS_col0, mS_col1 = _split_gated_scales(
                mS_row, mS_col, M, N, self.cfg.ROWWISE, True, False
            )
            mO_row0 = None
            if cutlass.const_expr(self.cfg.ROWWISE):
                mO_row0 = cute.local_tile(mO_row, (M, N), (0, 0))
                mO_row1 = cute.local_tile(mO_row, (M, N), (0, 1))
            mO_col0 = cute.local_tile(mO_col, (M, N), (0, 0))
            mO_col1 = cute.local_tile(mO_col, (M, N), (0, 1))
        else:
            mO_row0, mO_col0 = mO_row, mO_col
            mS_row0, mS_col0 = mS_row, mS_col
            if cutlass.const_expr(self.cfg.WITH_GEMM_SWIZZLED_SCALES):
                mS_row0, mS_col0 = derive_swizzled_scale_layout(
                    M, N, self.cfg.ROWWISE, True, mS_row, mS_col
                )

        # One band per CTA: the kernel has no per-CTA setup to amortize over a walk (no table,
        # unlike the bidimensional kernel).
        grid_cols = (N + TILE_COLS - 1) // TILE_COLS
        bands = (M + TILE_ROWS - 1) // TILE_ROWS

        # A tensor with a row or column tail takes the kernel whose tiles are predicated.
        if M % TILE_ROWS != 0 or N % TILE_COLS != 0:
            self.kernel(
                mAct,
                mGate,
                mGrad,
                mO_row0,
                mS_row0,
                mO_col0,
                mS_col0,
                mO_row1,
                mS_row1,
                mO_col1,
                mS_col1,
                limit,
                alpha,
                glu_linear_offset,
                beta1,
                beta2,
                TAILS=True,
            ).launch(
                grid=[grid_cols, bands],
                block=[self._CTA_THREADS_Y * self._CTA_THREADS_X],
                min_blocks_per_mp=self._MIN_BLOCKS_PER_MP,
                stream=stream,
            )
        else:
            self.kernel(
                mAct,
                mGate,
                mGrad,
                mO_row0,
                mS_row0,
                mO_col0,
                mS_col0,
                mO_row1,
                mS_row1,
                mO_col1,
                mS_col1,
                limit,
                alpha,
                glu_linear_offset,
                beta1,
                beta2,
                TAILS=False,
            ).launch(
                grid=[grid_cols, bands],
                block=[self._CTA_THREADS_Y * self._CTA_THREADS_X],
                min_blocks_per_mp=self._MIN_BLOCKS_PER_MP,
                stream=stream,
            )

    @cute.kernel
    def kernel(
        self,
        mAct: cute.Tensor,
        mGate: cute.Tensor,
        mGrad: Optional[cute.Tensor],
        mO_row0: Optional[cute.Tensor],
        mS_row0: Optional[cute.Tensor],
        mO_col0: cute.Tensor,
        mS_col0: cute.Tensor,
        mO_row1: Optional[cute.Tensor],
        mS_row1: Optional[cute.Tensor],
        mO_col1: Optional[cute.Tensor],
        mS_col1: Optional[cute.Tensor],
        limit: Optional[Float32],
        alpha: Optional[Float32],
        glu_linear_offset: Optional[Float32],
        beta1: Optional[Float32],
        beta2: Optional[Float32],
        TAILS: cutlass.Constexpr[bool],
    ):
        """Quantize the CTA's row band, blockIdx.y."""
        cta_col, row_tile, _ = cute.arch.block_idx()
        cache_policy = create_l2_policy(True)
        TILE_ROWS = self._CTA_THREADS_Y * self._ELEMENTS_Y_PER_THREAD
        TILE_COLS = self._CTA_THREADS_X * self._ELEMENTS_X_PER_THREAD
        # With TAILS, only the bands that cross the tensor's edge take the predicated body.
        if cutlass.const_expr(TAILS):
            if (row_tile + 1) * TILE_ROWS > mAct.shape[0] or (cta_col + 1) * TILE_COLS > mAct.shape[
                1
            ]:
                self._gated_row_tile(
                    mAct,
                    mGate,
                    mGrad,
                    mO_row0,
                    mS_row0,
                    mO_col0,
                    mS_col0,
                    mO_row1,
                    mS_row1,
                    mO_col1,
                    mS_col1,
                    limit,
                    alpha,
                    glu_linear_offset,
                    beta1,
                    beta2,
                    row_tile,
                    cta_col,
                    cache_policy,
                    TAILS=True,
                )
            else:
                self._gated_row_tile(
                    mAct,
                    mGate,
                    mGrad,
                    mO_row0,
                    mS_row0,
                    mO_col0,
                    mS_col0,
                    mO_row1,
                    mS_row1,
                    mO_col1,
                    mS_col1,
                    limit,
                    alpha,
                    glu_linear_offset,
                    beta1,
                    beta2,
                    row_tile,
                    cta_col,
                    cache_policy,
                    TAILS=False,
                )
        else:
            self._gated_row_tile(
                mAct,
                mGate,
                mGrad,
                mO_row0,
                mS_row0,
                mO_col0,
                mS_col0,
                mO_row1,
                mS_row1,
                mO_col1,
                mS_col1,
                limit,
                alpha,
                glu_linear_offset,
                beta1,
                beta2,
                row_tile,
                cta_col,
                cache_policy,
                TAILS=False,
            )

    @cute.jit
    def _quantize_part(
        self,
        rX: cute.Tensor,
        mO_row: Optional[cute.Tensor],
        mS_row: Optional[cute.Tensor],
        mO_col: cute.Tensor,
        mS_col: cute.Tensor,
        row_tile: Int32,
        cta_col: Int32,
        rows_left: Int32,
        cols_ok: Boolean,
        cache_policy: Int64,
        TAILS: cutlass.Constexpr[bool],
    ):
        """Quantize one output part of the band from its register tile."""
        TILE_ROWS = self._CTA_THREADS_Y * self._ELEMENTS_Y_PER_THREAD
        TILE_COLS = self._CTA_THREADS_X * self._ELEMENTS_X_PER_THREAD
        TILER = (TILE_ROWS, TILE_COLS)
        cta_coord = (row_tile, cta_col)
        gO_row_tile = gS_row_tile = None
        if cutlass.const_expr(self.cfg.ROWWISE):
            gO_row_tile = _row_tile(mO_row, row_tile, cta_col, TILER, 16)
            gS_row_tile = cute.local_tile(mS_row, (TILE_ROWS, TILE_COLS // 32), cta_coord)
        gO_col_tile = _row_tile(mO_col, row_tile, cta_col, TILER, 16)
        gS_col_tile = cute.local_tile(mS_col, (TILE_ROWS // 32, TILE_COLS), cta_coord)
        quantize_mxfp8_rmem_tile_bidimensional(
            rX,
            gO_row_tile,
            gS_row_tile,
            gO_col_tile,
            gS_col_tile,
            ROWWISE=self.cfg.ROWWISE,
            COLWISE=True,
            # The gated kernel's block amaxes ignore NaN like CUDA's gated kernels'.
            GATED=True,
            EDGE=TAILS,
            rows_left=rows_left,
            cols_ok=cols_ok,
            cols_valid=mO_col.shape[1] - cta_col * TILE_COLS,
            DTYPE=self.cfg.DTYPE,
            FP8_DTYPE=self.cfg.FP8_DTYPE,
            MAX_NORM_RCP=self.cfg.MAX_NORM_RCP,
            WITH_GEMM_SWIZZLED_SCALES=self.cfg.WITH_GEMM_SWIZZLED_SCALES,
            CTA_THREADS_Y=self._CTA_THREADS_Y,
            CTA_THREADS_X=self._CTA_THREADS_X,
            ELEMENTS_Y_PER_THREAD=self._ELEMENTS_Y_PER_THREAD,
            ELEMENTS_X_PER_THREAD=self._ELEMENTS_X_PER_THREAD,
            cache_policy=cache_policy,
        )

    @cute.jit
    def _thread_view(self, mT: cute.Tensor, row_tile: Int32, cta_col: Int32) -> cute.Tensor:
        """The calling thread's (ELEMENTS_X_PER_THREAD, ELEMENTS_Y_PER_THREAD) values of the
        (row_tile, cta_col) band of mT, in global memory."""
        ELEMENTS_Y = self._ELEMENTS_Y_PER_THREAD
        ELEMENTS_X = self._ELEMENTS_X_PER_THREAD
        TILER = (self._CTA_THREADS_Y * ELEMENTS_Y, self._CTA_THREADS_X * ELEMENTS_X)
        tidx, _, _ = cute.arch.thread_idx()
        tv_layout = thread_tile_layout(
            self._CTA_THREADS_Y, self._CTA_THREADS_X, ELEMENTS_Y, ELEMENTS_X
        )
        return cute.composition(_row_tile(mT, row_tile, cta_col, TILER, 16), tv_layout)[tidx, None]

    @cute.jit
    def _load_rows(
        self,
        gT_thread: cute.Tensor,
        first_row: cutlass.Constexpr[int],
        rows_left: Int32,
        cache_policy: Int64,
        TAILS: cutlass.Constexpr[bool],
    ) -> cute.Tensor:
        """Rows first_row .. first_row + _ROWS_IN_FLIGHT - 1 of the thread's values, in
        registers; with TAILS, rows from rows_left on load as zeros."""
        DTYPE = self.cfg.DTYPE
        ELEMENTS_X = self._ELEMENTS_X_PER_THREAD
        load_atom = cute.make_copy_atom(
            cute.nvgpu.CopyG2ROp(),
            DTYPE,
            num_bits_per_copy=min(ELEMENTS_X * DTYPE.width, max_num_bits_per_copy_ld_global()),
            invariant=False,
        )
        rT = cute.make_rmem_tensor((ELEMENTS_X, self._ROWS_IN_FLIGHT), DTYPE)
        for t in cutlass.range_constexpr(self._ROWS_IN_FLIGHT):
            if cutlass.const_expr(TAILS):
                rT[None, t].fill(0)
                row_pred = cute.make_rmem_tensor(
                    cute.make_layout((ELEMENTS_X, (1,)), stride=(0, (0,))), Boolean
                )
                row_pred[0] = first_row + t < rows_left
                cute.copy(
                    load_atom,
                    gT_thread[(None, first_row + t),],
                    rT[None, t],
                    pred=row_pred,
                    cache_policy=cache_policy,
                )
            else:
                cute.copy(
                    load_atom,
                    gT_thread[(None, first_row + t),],
                    rT[None, t],
                    cache_policy=cache_policy,
                )
        return rT

    @cute.jit
    def _gated_row(
        self,
        rAct_row: cute.Tensor,
        rGate_row: cute.Tensor,
        rGrad_row: Optional[cute.Tensor],
        rX0_row: cute.Tensor,
        rX1_row: Optional[cute.Tensor],
        limit: Optional[Float32],
        alpha: Optional[Float32],
        glu_linear_offset: Optional[Float32],
        beta1: Optional[Float32],
        beta2: Optional[Float32],
    ):
        """The gated op of one row of the thread's values, a BF16 pair at a time."""
        IS_BWD = self.cfg.IS_BWD
        rA_i32 = cute.recast_tensor(rAct_row, Int32)
        rG_i32 = cute.recast_tensor(rGate_row, Int32)
        rX0_i32 = cute.recast_tensor(rX0_row, Int32)
        if cutlass.const_expr(IS_BWD):
            rD_i32 = cute.recast_tensor(rGrad_row, Int32)
            rX1_i32 = cute.recast_tensor(rX1_row, Int32)
        for i in cutlass.range_constexpr(self._ELEMENTS_X_PER_THREAD // 2):
            grad_lo = grad_hi = None
            if cutlass.const_expr(IS_BWD):
                grad_lo = x2_lo_to_f32_bf16(rD_i32[i])
                grad_hi = x2_hi_to_f32_bf16(rD_i32[i])
            out_lo, gate_out_lo = gated_elementwise_op(
                x2_lo_to_f32_bf16(rA_i32[i]),
                x2_lo_to_f32_bf16(rG_i32[i]),
                grad_lo,
                limit,
                alpha,
                glu_linear_offset,
                beta1,
                beta2,
                ACTIVATION=self.cfg.GATED_ACTIVATION,
                IS_BWD=IS_BWD,
            )
            out_hi, gate_out_hi = gated_elementwise_op(
                x2_hi_to_f32_bf16(rA_i32[i]),
                x2_hi_to_f32_bf16(rG_i32[i]),
                grad_hi,
                limit,
                alpha,
                glu_linear_offset,
                beta1,
                beta2,
                ACTIVATION=self.cfg.GATED_ACTIVATION,
                IS_BWD=IS_BWD,
            )
            rX0_i32[i] = cvt_f32x2_to_bf16x2(out_lo, out_hi)
            if cutlass.const_expr(IS_BWD):
                rX1_i32[i] = cvt_f32x2_to_bf16x2(gate_out_lo, gate_out_hi)

    @cute.jit
    def _gated_row_tile(
        self,
        mAct: cute.Tensor,
        mGate: cute.Tensor,
        mGrad: Optional[cute.Tensor],
        mO_row0: Optional[cute.Tensor],
        mS_row0: Optional[cute.Tensor],
        mO_col0: cute.Tensor,
        mS_col0: cute.Tensor,
        mO_row1: Optional[cute.Tensor],
        mS_row1: Optional[cute.Tensor],
        mO_col1: Optional[cute.Tensor],
        mS_col1: Optional[cute.Tensor],
        limit: Optional[Float32],
        alpha: Optional[Float32],
        glu_linear_offset: Optional[Float32],
        beta1: Optional[Float32],
        beta2: Optional[Float32],
        row_tile: Int32,
        cta_col: Int32,
        cache_policy: Int64,
        TAILS: cutlass.Constexpr[bool],
    ):
        """Load one band's inputs, compute the gated op in registers and quantize its parts."""
        IS_BWD = self.cfg.IS_BWD
        DTYPE = self.cfg.DTYPE
        ELEMENTS_Y = self._ELEMENTS_Y_PER_THREAD
        ELEMENTS_X = self._ELEMENTS_X_PER_THREAD
        TILE_ROWS = self._CTA_THREADS_Y * ELEMENTS_Y
        TILE_COLS = self._CTA_THREADS_X * ELEMENTS_X
        TILER = (TILE_ROWS, TILE_COLS)
        tv_layout = thread_tile_layout(
            self._CTA_THREADS_Y, self._CTA_THREADS_X, ELEMENTS_Y, ELEMENTS_X
        )

        # With TAILS every band is predicated: rows and columns past the tensor load as zeros,
        # which every gated op maps to zero, and are not stored.
        rows_left = Int32(ELEMENTS_Y)
        cols_ok = Boolean(True)
        if cutlass.const_expr(TAILS):
            rows_left, cols_ok = thread_tile_bounds(
                TILER,
                tv_layout,
                mAct.shape[0] - row_tile * TILE_ROWS,
                mAct.shape[1] - cta_col * TILE_COLS,
            )

        gAct = self._thread_view(mAct, row_tile, cta_col)
        gGate = self._thread_view(mGate, row_tile, cta_col)
        gGrad = None
        if cutlass.const_expr(IS_BWD):
            gGrad = self._thread_view(mGrad, row_tile, cta_col)

        # The gated op in FP32, rounded to BF16 as gated_mxfp8.cuh does before taking amaxes:
        # rX0 is the forward output or the act half's gradient, rX1 the gate half's gradient.
        # Inputs are loaded _ROWS_IN_FLIGHT rows at a time.
        rX0 = cute.make_rmem_tensor((ELEMENTS_X, ELEMENTS_Y), DTYPE)
        rX1 = None
        if cutlass.const_expr(IS_BWD):
            rX1 = cute.make_rmem_tensor((ELEMENTS_X, ELEMENTS_Y), DTYPE)
        for first_row in cutlass.range_constexpr(0, ELEMENTS_Y, self._ROWS_IN_FLIGHT):
            rAct = self._load_rows(gAct, first_row, rows_left, cache_policy, TAILS=TAILS)
            rGate = self._load_rows(gGate, first_row, rows_left, cache_policy, TAILS=TAILS)
            rGrad = None
            if cutlass.const_expr(IS_BWD):
                rGrad = self._load_rows(gGrad, first_row, rows_left, cache_policy, TAILS=TAILS)
            for t in cutlass.range_constexpr(self._ROWS_IN_FLIGHT):
                rGrad_row = rX1_row = None
                if cutlass.const_expr(IS_BWD):
                    rGrad_row = rGrad[None, t]
                    rX1_row = rX1[None, first_row + t]
                self._gated_row(
                    rAct[None, t],
                    rGate[None, t],
                    rGrad_row,
                    rX0[None, first_row + t],
                    rX1_row,
                    limit,
                    alpha,
                    glu_linear_offset,
                    beta1,
                    beta2,
                )

        self._quantize_part(
            rX0,
            mO_row0,
            mS_row0,
            mO_col0,
            mS_col0,
            row_tile,
            cta_col,
            rows_left,
            cols_ok,
            cache_policy,
            TAILS=TAILS,
        )
        if cutlass.const_expr(IS_BWD):
            self._quantize_part(
                rX1,
                mO_row1,
                mS_row1,
                mO_col1,
                mS_col1,
                row_tile,
                cta_col,
                rows_left,
                cols_ok,
                cache_policy,
                TAILS=TAILS,
            )
