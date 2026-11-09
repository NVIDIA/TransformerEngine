# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Grouped MXFP8 quantization by a non-grouped kernel on a single-tensor view of the group.

When the group's data and scales are laid out exactly as one tensor's, a plain cast of the
group is byte-for-byte a single-tensor cast, so the non-grouped kernels apply unchanged:

* "stacked": SAME_BOTH_DIMS / VARYING_FIRST_DIM members start on 128-row boundaries and share
  the column count, so unless colwise scales are GEMM-swizzled (whose indices restart per
  member) the group is the stacked (sum(rows), cols) tensor.
* "flat": with packed (non-swizzled) rowwise scales only, scale i covers elements 32i..32i+31
  of the flat payload whatever the member shapes, as long as every member's cols are a
  multiple of 128 (which both grouped kernels require for VARYING_LAST_DIM /
  VARYING_BOTH_DIMS). Any group is then a cast of a (total / 128, 128) tensor.

The grouped kernels zero the scale padding and the non-grouped ones leave it untouched, so the
views are only taken when there is none.
"""

from typing import Optional, Type

import cutlass
from cutlass import cute
from cutlass import Boolean, Int32, Int64
from cuda.bindings.driver import CUstream  # pylint: disable=no-name-in-module

from .quantize_mxfp8_common import MXFP8QuantizeConfig, MXFP8QuantizeKernelBase
from .group_quantize_mxfp8_common import MXFP8GroupQuantizeConfig, MXFP8GroupQuantizeKernelBase

STACKED = "stacked"
FLAT = "flat"


class MXFP8GroupQuantizeSingleTensorViewKernel(MXFP8GroupQuantizeKernelBase):
    """Run one non-grouped MXFP8 quantize kernel on the stacked or flat view of the group."""

    def __init__(
        self, cfg: MXFP8GroupQuantizeConfig, kernel_cls: Type[MXFP8QuantizeKernelBase]
    ) -> None:
        assert self.supports(cfg, kernel_cls), f"{kernel_cls.__name__} cannot run {cfg}"
        self.cfg = cfg
        self.VIEW = self.view(cfg)
        self.single_cfg = self.single_tensor_cfg(cfg)
        self.kernel = kernel_cls(self.single_cfg)

    @staticmethod
    def view(cfg: MXFP8GroupQuantizeConfig) -> Optional[str]:
        """The single-tensor view the group's plain cast has, if any."""
        if cfg.WITH_DBIAS or cfg.WITH_ACT or cfg.WITH_DACT:
            return None
        if cfg.IS_SINGLE_TENSOR and not (cfg.COLWISE and cfg.WITH_GEMM_SWIZZLED_SCALES):
            return STACKED
        if cfg.ROWWISE and not cfg.COLWISE and not cfg.WITH_GEMM_SWIZZLED_SCALES:
            return FLAT
        return None

    @staticmethod
    def single_tensor_cfg(cfg: MXFP8GroupQuantizeConfig) -> MXFP8QuantizeConfig:
        """The non-grouped config of the same plain cast."""
        return MXFP8QuantizeConfig(
            dtype=cfg.DTYPE_STR,
            fp8_dtype=cfg.FP8_DTYPE_STR,
            rowwise=cfg.ROWWISE,
            colwise=cfg.COLWISE,
            with_gemm_swizzled_scales=cfg.WITH_GEMM_SWIZZLED_SCALES,
            with_amax=False,
            activation="none",
        )

    # pylint: disable-next=arguments-differ
    @staticmethod
    def supports(cfg: MXFP8GroupQuantizeConfig, kernel_cls: Type[MXFP8QuantizeKernelBase]) -> bool:
        view = MXFP8GroupQuantizeSingleTensorViewKernel.view(cfg)
        return view is not None and kernel_cls.supports(
            MXFP8GroupQuantizeSingleTensorViewKernel.single_tensor_cfg(cfg)
        )

    @cute.jit
    def _views(
        self,
        mX: cute.Tensor,
        mO_row: Optional[cute.Tensor],
        mO_col: Optional[cute.Tensor],
        mS_row: cute.Tensor,
        mS_col: cute.Tensor,
    ):
        """(x, o_row, s_row, o_col, s_col) of the single-tensor view, in the non-grouped
        kernels' argument order. They are compiled for Int32 extents; the scales arrive as flat
        buffers and are viewed at the 2D shapes TE allocates for a single (M, N) tensor."""
        if cutlass.const_expr(self.VIEW == STACKED):
            # Restate the divisibility the Int32 casts drop -- M by the compiled sym, N by the
            # eligibility check -- so the vectorized GMEM copies can prove their alignment.
            M = cute.assume(Int32(mX.shape[0]), divby=128)
            N = cute.assume(Int32(mX.shape[1]), divby=128)
        else:
            M = Int32(Int64(mX.shape[0]) * Int64(mX.shape[1]) // 128)
            N = 128
        data_layout = cute.make_layout((M, N), stride=(N, 1))
        x = cute.make_tensor(mX.iterator, data_layout)
        o_row = s_row = o_col = s_col = None
        if cutlass.const_expr(self.cfg.ROWWISE):
            o_row = cute.make_tensor(mO_row.iterator, data_layout)
            s_row = cute.make_tensor(
                mS_row.iterator, cute.make_layout((M, N // 32), stride=(N // 32, 1))
            )
        if cutlass.const_expr(self.cfg.COLWISE):
            o_col = cute.make_tensor(mO_col.iterator, data_layout)
            s_col = cute.make_tensor(mS_col.iterator, cute.make_layout((M // 32, N), stride=(N, 1)))
        return x, o_row, s_row, o_col, s_col

    @cute.jit
    def eligible_to_dispatch(
        self,
        mX: cute.Tensor,
        mO_row: Optional[cute.Tensor],
        mO_col: Optional[cute.Tensor],
        mS_row: cute.Tensor,
        mS_col: cute.Tensor,
        mOffsets: Optional[cute.Tensor],
        mFirstDims: Optional[cute.Tensor],
        mLastDims: Optional[cute.Tensor],
        num_tensors: Int32,
        mNoop: cute.Pointer,
    ) -> Boolean:
        # The non-grouped kernels index with Int32, so groups of 2^31 or more elements cannot
        # be viewed as one tensor.
        view_ok = Int64(mX.shape[0]) * Int64(mX.shape[1]) < Int64(2**31)
        if cutlass.const_expr(self.VIEW == STACKED):
            # No scale padding: rows are already 128-aligned, and N % 128 == 0 leaves no padded
            # rowwise (N / 32 % 4) or colwise (N % 128) columns.
            view_ok = view_ok and mX.shape[1] % 128 == 0
        x, o_row, s_row, o_col, s_col = self._views(mX, mO_row, mO_col, mS_row, mS_col)
        return view_ok and self.kernel.eligible_to_dispatch(
            self.single_cfg, x, o_row, s_row, o_col, s_col, mNoop, None
        )

    @cute.jit
    def __call__(
        self,
        mX: cute.Tensor,
        mO_row: Optional[cute.Tensor],
        mO_col: Optional[cute.Tensor],
        mS_row: cute.Tensor,
        mS_col: cute.Tensor,
        mOffsets: Optional[cute.Tensor],
        mFirstDims: Optional[cute.Tensor],
        mLastDims: Optional[cute.Tensor],
        mTensormaps: Optional[cute.Tensor],
        num_tensors: Int32,
        mNoop: cute.Pointer,
        mActInput: Optional[cute.Tensor],
        mWorkspace: Optional[cute.Tensor],
        stream: CUstream,
    ) -> None:
        x, o_row, s_row, o_col, s_col = self._views(mX, mO_row, mO_col, mS_row, mS_col)
        self.kernel(x, o_row, s_row, o_col, s_col, None, mNoop, None, None, stream)
