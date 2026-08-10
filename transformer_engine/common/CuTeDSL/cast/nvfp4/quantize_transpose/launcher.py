# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Compilation and TVM-FFI registration for NVFP4 quantize-transpose."""

import logging

from cutlass import cute
from cutlass import BFloat16, Float4E2M1FN, Float8E4M3FN, Float32, Int64
import tvm_ffi

from transformer_engine.common.CuTeDSL.cast.nvfp4.utils import (
    NVFP4_SCALE_PAD_INNER,
    NVFP4_SCALE_PAD_OUTER,
    NVFP4_SHAPE_ALIGNMENT,
)
from transformer_engine.common.CuTeDSL.utils import device_is_blackwell

from .config import NVFP4QuantizeConfig
from .kernel import NVFP4QuantizeTransposeTuned1DKernel


logger = logging.getLogger("transformer_engine.cutedsl.nvfp4")


def compile_cutedsl_function_from_cfg(cfg):
    """
    Return the compiled CuTeDSL function object for the given NVFP4 quantization config.

    The fake tensors below are what fix the compiled ABI, so they must agree with `__call__`'s
    signature.
    """

    if not device_is_blackwell():
        raise RuntimeError("CuTeDSL NVFP4 backend requires compute capability >= 10.0 (Blackwell)")

    kernel_obj = NVFP4QuantizeTransposeTuned1DKernel(cfg)

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
    tvm_ffi.register_global_func(fn_name, compiled, override=True)

    return True
