# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Shared definitions, kernel dispatch and TVM-FFI registration for CuTeDSL NVFP4."""

import abc
import functools
import logging
import os
from typing import Optional

import cutlass
from cutlass import cute
from cutlass import BFloat16, Boolean, Float4E2M1FN, Float8E4M3FN, Float32, Int64
from cuda.bindings.driver import CUstream  # pylint: disable=no-name-in-module
import tvm_ffi

from transformer_engine.common.CuTeDSL.cast.nvfp4.utils import (
    NVFP4_SCALE_PAD_INNER,
    NVFP4_SCALE_PAD_OUTER,
    NVFP4_SHAPE_ALIGNMENT,
)
from transformer_engine.common.CuTeDSL.utils import device_compute_capability

CUTEDSL_DEBUG_LOGGING = os.environ.get("CUTEDSL_DEBUG_LOGGING", "0") == "1"

logger = logging.getLogger("transformer_engine.cutedsl.nvfp4")


@functools.lru_cache(maxsize=None)
def device_sm_count() -> int:
    """SM count of the current device, which kernels size their grids against. Read when a
    kernel is compiled, so a compiled kernel assumes the SM count of the device it was compiled
    on; the C++ side keys kernels by SM architecture."""
    return cutlass.utils.HardwareInfo().get_device_multiprocessor_count()


@cute.jit
def noop_flag_is_set(mNoop: cute.Pointer) -> Boolean:
    """Whether the cast_noop flag says this quantization is a no-op and must be skipped.

    mNoop is a pointer rather than a tensor so that one compiled kernel serves both a present and
    an absent flag, hence the address is checked before it is dereferenced, exactly like the CUDA
    C++ kernel's `noop != nullptr && noop[0] == 1.0f`. The two checks cannot be joined with `and`,
    which the DSL lowers to a non-short-circuiting op that would load from the null pointer.
    """
    flag_is_set = Boolean(False)
    if mNoop.toint() != Int64(0):
        flag_is_set = cute.make_tensor(mNoop, cute.make_layout((1,)))[0] == Float32(1.0)
    return flag_is_set


class NVFP4QuantizeConfig:
    """Instantiation parameters of the CuTeDSL kernel. These are fixed once the kernel is
    compiled and behave as const expressions."""

    def __init__(
        self,
        use_stochastic_rounding: bool,
        use_fast_math: bool,
        row_scaled_nvfp4: bool,
        return_transpose: bool,
    ):
        self.USE_STOCHASTIC_ROUNDING = use_stochastic_rounding
        self.USE_FAST_MATH = use_fast_math
        if row_scaled_nvfp4 and return_transpose:
            raise ValueError("row-scaled NVFP4 quantization does not produce a transposed output")
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


class NVFP4QuantizeKernelBase(abc.ABC):
    """Base class for NVFP4 quantize kernels."""

    @staticmethod
    def supports(cfg: NVFP4QuantizeConfig) -> bool:  # pylint: disable=unused-argument
        """Compile time check: whether this specialized kernel handles cfg at all."""
        return False

    @staticmethod
    def eligible_to_dispatch(
        cfg: NVFP4QuantizeConfig,
        mX: cute.Tensor,
        mO_row: cute.Tensor,
        mS_row: cute.Tensor,
        mO_col: Optional[cute.Tensor],
        mS_col: Optional[cute.Tensor],
    ) -> Boolean:
        """Runtime check: whether we can dispatch to the instantiated kernel with this call's
        shapes and alignments."""
        raise NotImplementedError

    @abc.abstractmethod
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
        """
        Compiled kernel entrypoint (decorate with @cute.jit).
        All NVFP4 quantize kernels must implement this interface because it is the contract of
        the C++ call site in `quantize_transpose_nvfp4_cutedsl.cuh`, which passes the arguments
        in this exact order via tvm-ffi; a kernel must accept them even if it does not use them.
        """


class NVFP4QuantizeEntry(NVFP4QuantizeKernelBase):
    """Select the NVFP4 quantization kernel from the configuration and the runtime shapes."""

    def __init__(self, cfg: NVFP4QuantizeConfig):
        # Kernels import shared definitions from this module. Resolve their
        # implementations only when constructing the compile-time entry object.
        # pylint: disable=import-outside-toplevel
        from .quantize_nvfp4_g2r_bidim import NVFP4QuantizeBidimensionalG2RKernel
        from .quantize_nvfp4_g2r_rowwise import NVFP4QuantizeRowwiseG2RKernel
        from .quantize_nvfp4_g2s_general import NVFP4QuantizeGeneralG2SKernel

        # pylint: enable=import-outside-toplevel

        self.cfg = cfg
        # The general kernel handles every config; the specialized ones, in dispatch priority
        # order, take the calls they are eligible for.
        self.general_kernel = NVFP4QuantizeGeneralG2SKernel(cfg)
        self.specialized_kernels = [
            kernel_cls(cfg)
            for kernel_cls in (NVFP4QuantizeRowwiseG2RKernel, NVFP4QuantizeBidimensionalG2RKernel)
            if kernel_cls.supports(cfg)
        ]

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
        args = (mX, mO_row, mS_row, mO_col, mS_col, mAmaxRow, mAmaxCol, mNoop, mRngState, stream)
        dispatched_to_specialized = False
        for kernel in self.specialized_kernels:
            if not dispatched_to_specialized:
                if kernel.eligible_to_dispatch(self.cfg, mX, mO_row, mS_row, mO_col, mS_col):
                    dispatched_to_specialized = True
                    kernel(*args)
        if not dispatched_to_specialized:
            self.general_kernel(*args)


def compile_cutedsl_function_from_cfg(cfg):
    """
    Return the compiled CuTeDSL function object for the given NVFP4 quantization config.

    The fake tensors below are what fix the compiled ABI, so they must agree with `__call__`'s
    signature.
    """

    kernel_obj = NVFP4QuantizeEntry(cfg)

    def _gmem(dtype, shape, stride_order, align):
        """Fake row-major GMEM tensor; `align` is the assumed base alignment in bytes."""
        return cute.runtime.make_fake_compact_tensor(
            dtype,
            shape,
            stride_order=stride_order,
            memspace=cute.AddressSpace.gmem,
            assumed_align=align,
        )

    # Flattened 2D input dims. Their divisibility is also what keeps every row stride a multiple
    # of 16B as TMA requires: 2*N bytes for the bf16 input, N/2 and M/2 for the fp4 outputs.
    sym_M = cute.sym_int32(divisibility=NVFP4_SHAPE_ALIGNMENT)
    sym_N = cute.sym_int32(divisibility=NVFP4_SHAPE_ALIGNMENT)

    # Scale extents are TE's padded ones (NVFP4Quantizer::get_scale_shape):
    #   mS_row: (roundup(M, 128), roundup(ceil(N / 16), 4))
    #   mS_col: (roundup(N, 128), roundup(ceil(M / 16), 4))
    # Neither is expressible in sym_M / sym_N (SymInt has no `//` or `+`), so the scales get
    # fresh syms carrying only the divisibility the padding guarantees.
    scale_row_shape = (
        cute.sym_int32(divisibility=NVFP4_SCALE_PAD_OUTER),
        cute.sym_int32(divisibility=NVFP4_SCALE_PAD_INNER),
    )
    scale_col_shape = (
        cute.sym_int32(divisibility=NVFP4_SCALE_PAD_OUTER),
        cute.sym_int32(divisibility=NVFP4_SCALE_PAD_INNER),
    )

    # mX / mO_row / mS_row: (M, N) input, (M, N) rowwise fp4 output and its scales. The fp4
    # extents are logical element counts, so the row stride is N while only N/2 bytes are
    # stored. That is the kernel's view of it; the compiled ABI is not, because the DSL rewrites
    # every Float4E2M1FN argument into the packed float4_e2m1fnx2 form DLPack can express, with
    # the contiguous extent and the strides above it halved. What the caller hands over is
    # therefore the packed tensor, which is what tvm_ffi_bridge.h builds for an FP4 TE tensor.
    in_fake = _gmem(BFloat16, (sym_M, sym_N), stride_order=(1, 0), align=16)
    out_row_fake = _gmem(Float4E2M1FN, (sym_M, sym_N), stride_order=(1, 0), align=16)
    scale_row_fake = _gmem(Float8E4M3FN, scale_row_shape, stride_order=(1, 0), align=4)

    # mO_col / mS_col: (N, M) transposed fp4 output and its scales, present iff a transpose is
    # produced.
    out_col_fake = (
        _gmem(Float4E2M1FN, (sym_N, sym_M), stride_order=(1, 0), align=16)
        if cfg.RETURN_TRANSPOSE
        else None
    )
    scale_col_fake = (
        _gmem(Float8E4M3FN, scale_col_shape, stride_order=(1, 0), align=4)
        if cfg.RETURN_TRANSPOSE
        else None
    )

    # mAmaxRow: unpadded per-row amax (M,) when row-scaled, else a single global amax (1,).
    amax_row_fake = (
        _gmem(Float32, (sym_M,), stride_order=(0,), align=4)
        if cfg.ROW_SCALED_NVFP4
        else _gmem(Float32, (1,), stride_order=(0,), align=4)
    )
    # mAmaxCol: (1,) global amax of the transposed output, present iff a transpose is produced.
    amax_col_fake = (
        _gmem(Float32, (1,), stride_order=(0,), align=4) if cfg.RETURN_TRANSPOSE else None
    )

    # mNoop: always-present f32 pointer to the single-element cast-noop flag. A pointer rather
    # than a tensor so the runtime value can be null without changing the compiled ABI -- the
    # kernel does the null-check and the noop[0] == 1.0f test on device. The null address here
    # is only a compile-time placeholder for the pointer the dispatcher passes at launch.
    noop_fake = cute.runtime.nullptr(Float32, mem_space=cute.AddressSpace.gmem, assumed_align=4)

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
    """Compile the NVFP4 quantize kernel for this config and register it in the TVM-FFI global
    registry under EXACTLY `fn_name` (the key the C++ dispatcher built). Returns True if a kernel
    is registered under `fn_name`, False if the config is unsupported or compilation failed, in
    which case the caller caches the negative result and falls back to the CUDA C++ kernel.
    """

    # Already registered (e.g. by a prior call) -> supported.
    if tvm_ffi.get_global_func(fn_name, allow_missing=True) is not None:
        return True

    major, minor = device_compute_capability()
    if major < 10:
        logger.warning(
            "CuTeDSL NVFP4 backend requires compute capability >= 10.0 (Blackwell), "
            "but detected %d.%d; falling back to the CUDA C++ kernel.",
            major,
            minor,
        )
        return False

    try:
        cfg = NVFP4QuantizeConfig(
            use_stochastic_rounding=use_stochastic_rounding,
            use_fast_math=use_fast_math,
            row_scaled_nvfp4=row_scaled_nvfp4,
            return_transpose=return_transpose,
        )
    except ValueError as e:
        # The exception message states exactly why the config is unsupported. Surfacing it as a
        # warning lets the C++ dispatcher's CUDA fallback be recognized as expected.
        logger.warning(
            "CuTeDSL NVFP4 backend does not support this config, "
            "falling back to the CUDA C++ kernel: %s",
            e,
        )
        return False

    logger.debug("Compiling CuTeDSL NVFP4 quantization kernel for %s", cfg)
    try:
        compiled = compile_cutedsl_function_from_cfg(cfg)
    except NotImplementedError as e:
        # Raised while tracing a config this kernel does not cover yet, which is expected rather
        # than a failure, so it is logged apart from a genuine compilation error.
        logger.warning(
            "CuTeDSL NVFP4 backend does not implement this config yet, "
            "falling back to the CUDA C++ kernel: %s",
            e,
        )
        return False
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.warning(
            "CuTeDSL NVFP4 kernel compilation failed, falling back to the CUDA C++ kernel: %s",
            e,
        )
        return False
    # Register the native function rather than the Python wrapper cute.compile returns: the C++
    # side already passes the arguments positionally, so the wrapper is pure launch overhead.
    native = getattr(compiled, "__tvm_ffi_object__", lambda: None)()
    tvm_ffi.register_global_func(fn_name, native if native is not None else compiled, override=True)

    return True
