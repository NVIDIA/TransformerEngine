# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Grouped MXFP8 quantization by a non-grouped kernel on the stacked view of the group.

SAME_BOTH_DIMS members start on 128-row boundaries and share the column count, so unless the
colwise scales are GEMM-swizzled (their indices restart per member) a plain cast of the group
is byte-for-byte a cast of the stacked (rows, cols) tensor.

Only SAME_BOTH_DIMS, whose members always fill the buffer: varying members may fill only part
of it (CUDA graphs size it ahead of time), and a view would quantize the unused tail too.
"""

from typing import Optional, Type

import cutlass
from cutlass import cute
from cutlass import Boolean, Int32, Int64
from cuda.bindings.driver import CUstream  # pylint: disable=no-name-in-module

from .quantize_mxfp8_common import MXFP8QuantizeConfig, MXFP8QuantizeKernelBase
from .group_quantize_mxfp8_common import (
    SAME_BOTH_DIMS,
    MXFP8GroupQuantizeConfig,
    MXFP8GroupQuantizeKernelBase,
)


class MXFP8GroupQuantizeSingleTensorViewKernel(MXFP8GroupQuantizeKernelBase):
    """Run one non-grouped MXFP8 quantize kernel on the stacked view of the group."""

    def __init__(
        self, cfg: MXFP8GroupQuantizeConfig, kernel_cls: Type[MXFP8QuantizeKernelBase]
    ) -> None:
        assert self.supports(cfg, kernel_cls), f"{kernel_cls.__name__} cannot run {cfg}"
        self.cfg = cfg
        self.single_cfg = self.single_tensor_cfg(cfg)
        self.kernel = kernel_cls(self.single_cfg)

    @staticmethod
    def has_view(cfg: MXFP8GroupQuantizeConfig) -> bool:
        """Whether the group's cast is a cast of its stacked view."""
        plain_cast = not (cfg.WITH_DBIAS or cfg.WITH_ACT or cfg.WITH_DACT)
        swizzled_colwise = cfg.COLWISE and cfg.WITH_GEMM_SWIZZLED_SCALES
        return plain_cast and cfg.SHAPE_REP == SAME_BOTH_DIMS and not swizzled_colwise

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
        return MXFP8GroupQuantizeSingleTensorViewKernel.has_view(cfg) and kernel_cls.supports(
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
        """(x, o_row, s_row, o_col, s_col) of the stacked view, in the non-grouped kernels'
        argument order. They are compiled for Int32 extents; the scales arrive as flat buffers
        and are viewed at the 2D shapes TE allocates for a single (M, N) tensor."""
        # Restate the divisibility the Int32 casts drop -- M by the compiled sym, N by the
        # eligibility check -- so the vectorized GMEM copies can prove their alignment.
        M = cute.assume(Int32(mX.shape[0]), divby=128)
        N = cute.assume(Int32(mX.shape[1]), divby=128)
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
        # No scale padding: rows are already 128-aligned, and N % 128 == 0 leaves no padded
        # rowwise (N / 32 % 4) or colwise (N % 128) columns.
        view_ok = Int64(mX.shape[0]) * Int64(mX.shape[1]) < Int64(2**31) and mX.shape[1] % 128 == 0
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
