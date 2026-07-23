# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""NVFP4 quantization kernel implemented in CuTeDSL."""

import logging
import os
from typing import Optional, Type

import cutlass
from cutlass import cute
from cutlass import pipeline
from cutlass import Float32, Int16, Int32, Int64, Uint32, Uint8
from cutlass import Float4E2M1FN
from cuda.bindings.driver import CUstream  # pylint: disable=no-name-in-module
import tvm_ffi

from transformer_engine.common.CuTeDSL.utils import device_is_blackwell

CUTEDSL_DEBUG_LOGGING = os.environ.get("CUTEDSL_DEBUG_LOGGING", "0") == "1"

logger = logging.getLogger("transformer_engine.cutedsl.nvfp4")

# Number of elements per NVFP4 scale block (they share one E4M3 scale factor).
NVFP4_BLOCK_SCALING_SIZE = 16
# Input row/col divisibility the CUDA tuned-1D kernel requires (16B TMA alignment).
NVFP4_SHAPE_ALIGNMENT = 32


class NVFP4QuantizeConfig:
    """Instantiation parameters of the CuTE DSL kernel"""

    def __init__(
        self,
        use_stochastic_rounding: bool,
        use_fast_math: bool,
        row_scaled_nvfp4: bool,
        return_transpose: bool,
    ):
        self.USE_STOCHASTIC_ROUNDING = use_stochastic_rounding
        self.USE_FAST_MATH = use_fast_math
        self.ROW_SCALED_NVFP4 = row_scaled_nvfp4
        self.RETURN_TRANSPOSE = return_transpose

    def __str__(self):
        return (
            f"NVFP4QuantizeConfig(use_stochastic_rounding={self.USE_STOCHASTIC_ROUNDING}, "
            f"use_fast_math={self.USE_FAST_MATH}, "
            f"row_scaled_nvfp4={self.ROW_SCALED_NVFP4}, "
            f"return_transpose={self.RETURN_TRANSPOSE})"
        )

    __repr__ = __str__


class NVFP4QuantizeTransposeTuned1DKernel:
    """Tuned kernel to cast to NVFP4 and transpose"""

    def __init__(self, cfg):
        self.cfg = cfg
        # todo: derive tile / thread / stage constants from the CUDA tuned-1D kernel
        # (CHUNK 128x128, TMA box 64x64, 128 threads, block scale dim 16).

    @cute.jit
    def __call__(
        self,
        mX: cute.Tensor,
        mO_row: cute.Tensor,
        mS_row: cute.Tensor,
        mO_col: Optional[cute.Tensor],
        mS_col: Optional[cute.Tensor],
        mAmaxRow: cute.Tensor,
        mAmaxCol: Optional[cute.Tensor],
        mNoop: cute.Pointer,
        mRngState: Optional[cute.Tensor],
        stream: CUstream,
    ):
        if cutlass.const_expr(CUTEDSL_DEBUG_LOGGING):
            cute.printf(
                f"[CuTeDSL] NVFP4QuantizeTransposeTuned1DKernel.__call__() with config: {self.cfg}\n"
            )

        # todo: set up input/output TMA atoms, compute grid/block, and launch self.kernel(...).
        # The tensor plumbing (which args are present) is fixed by the config; see
        # compile_cutedsl_function_from_cfg for the matching fake-tensor signature.
        self.kernel(
            # todo
        ).launch(grid=grid, block=block, stream=stream)

    @cute.kernel
    def kernel(self, *todo):
        """Device entry for the NVFP4 tuned-1D quantize-transpose kernel."""
        # todo


def compile_cutedsl_function_from_cfg(cfg):
    """
    Return the compiled CuTeDSL function object for the given NVFP4 quantization config.
    """

    if not device_is_blackwell():
        raise RuntimeError("CuTeDSL NVFP4 backend requires compute capability >= 10.0 (Blackwell)")

    kernel_obj = NVFP4QuantizeTransposeTuned1DKernel(cfg)

    # M, N are the flattened 2D input dims; the CUDA tuned-1D kernel requires both to be
    # multiples of 32.
    sym_M = cute.sym_int32(divisibility=NVFP4_SHAPE_ALIGNMENT)
    sym_N = cute.sym_int32(divisibility=NVFP4_SHAPE_ALIGNMENT)

    # NVFP4 scale tensors, divisibility constraints per NVFP4Quantizer.get_scale_shape:
    scale_row_shape = (cute.sym_int32(divisibility=128), cute.sym_int32(divisibility=4))
    scale_col_shape = (cute.sym_int32(divisibility=128), cute.sym_int32(divisibility=4))

    def _gmem(dtype, shape, stride_order, align):
        return cute.runtime.make_fake_compact_tensor(
            dtype,
            shape,
            stride_order=stride_order,
            memspace=cute.AddressSpace.gmem,
            assumed_align=align,
        )


    in_fake = _gmem(cutlass.BFloat16, (sym_M, sym_N), stride_order=(1, 0), align=16)


    out_row_fake = _gmem(Float4E2M1FN, (sym_M, sym_N), stride_order=(1, 0), align=16)
    scale_row_fake = _gmem(Uint8, scale_row_shape, stride_order=(1, 0), align=4)

    # mO_col / mS_col / mAmaxCol: transpose (N, M) NVFP4 data + scales + global amax,
    # present iff a transpose is produced. Same fp4 logical-shape reasoning as above.
    out_col_fake = (
        _gmem(Float4E2M1FN, (sym_N, sym_M), stride_order=(1, 0), align=16)
        if cfg.RETURN_TRANSPOSE
        else None
    )
    scale_col_fake = (
        _gmem(Uint8, scale_col_shape, stride_order=(1, 0), align=4)
        if cfg.RETURN_TRANSPOSE
        else None
    )
    amax_col_fake = (
        _gmem(Float32, (1,), stride_order=(0,), align=4) if cfg.RETURN_TRANSPOSE else None
    )

    # mAmaxRow: per-row FP32 amax (M,) when row-scaled, else a single global amax (1,).
    amax_row_fake = (
        _gmem(Float32, (sym_M,), stride_order=(0,), align=4)
        if cfg.ROW_SCALED_NVFP4
        else _gmem(Float32, (1,), stride_order=(0,), align=4)
    )

    # mNoop: ALWAYS-present f32 device pointer to the single-element cast-noop flag.
    # Modeled as a raw pointer (not a tensor) so its runtime value can be null without
    # changing the compiled ABI -- the kernel does the null-check + noop[0] == 1.0f at
    # runtime. Address 0 is only a compile-time placeholder; the real (possibly-null)
    # pointer is supplied by the C++ dispatcher at launch time.
    # NOTE: verify against the installed cutlass -- if it exposes a dedicated
    # `cute.runtime.make_fake_ptr`, prefer that over make_ptr(..., 0, ...).
    noop_fake = cute.runtime.nullptr(
        Float32, mem_space=cute.AddressSpace.gmem, assumed_align=4
    )

    # mRngState: Philox {seed, offset}; present (and consumed) only for stochastic rounding.
    rng_state_fake = (
        _gmem(Int64, (2,), stride_order=(0,), align=8) if cfg.USE_STOCHASTIC_ROUNDING else None
    )

    compiled = cute.compile(
        kernel_obj,
        in_fake,  # mX
        out_row_fake,  # mO_row
        scale_row_fake,  # mS_row
        out_col_fake,  # mO_col
        scale_col_fake,  # mS_col
        amax_row_fake,  # mAmaxRow
        amax_col_fake,  # mAmaxCol
        noop_fake,  # mNoop
        rng_state_fake,  # mRngState
        cute.runtime.make_fake_stream(),  # stream
        options="--enable-tvm-ffi",
    )
    return compiled


def get_nvfp4_quantization_function(
    fn_name: str,
    use_stochastic_rounding: bool,
    use_fast_math: bool,
    row_scaled_nvfp4: bool,
    return_transpose: bool,
) -> bool:
    """Compile the NVFP4 quantize kernel for this config and register it in the TVM-FFI global registry"""

    if tvm_ffi.get_global_func(fn_name, allow_missing=True) is not None:
        return True

    cfg = NVFP4QuantizeConfig(
        use_stochastic_rounding=use_stochastic_rounding,
        use_fast_math=use_fast_math,
        row_scaled_nvfp4=row_scaled_nvfp4,
        return_transpose=return_transpose,
    )

    logger.debug("Compiling CuTeDSL NVFP4 quantization kernel for %s", cfg)
    try:
        compiled = compile_cutedsl_function_from_cfg(cfg)
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.warning(
            "CuTeDSL NVFP4 kernel compilation failed, falling back to the CUDA C++ kernel: %s",
            e,
        )
        return False
    tvm_ffi.register_global_func(fn_name, compiled, override=True)

    return True
