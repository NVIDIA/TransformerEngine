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
from cuda.bindings.driver import CUstream  # pylint: disable=no-name-in-module
import tvm_ffi

CUTEDSL_DEBUG_LOGGING = os.environ.get("CUTEDSL_DEBUG_LOGGING", "0") == "1"

logger = logging.getLogger("transformer_engine.cutedsl.mxfp8")


class NVFP4QuantizeConfig:
    """Instantiation parameters of the CuTE DSL kernel"""

    def __init__(
        self,
        # todo
    ):
        pass  # todo

    def __str__(self):
        return (
            # todo
        )

    __repr__ = __str__


class NVFP4QuantizeTransposeTuned1DKernel:
    """Tuned kernel to cast to NVFP4 and transpose"""

    def __init__(self, cfg):
        self.cfg = cfg
        # todo

    @cute.jit
    def __call__(
        self,
        *todo,
        stream: CUstream,
    ):
        # todo

        self.kernel(
            # todo
        ).launch(grid=grid, block=block, stream=stream)

    @cute.kernel
    def kernel(self, *todo):
        """Device entry for the specialized rowwise-only cast kernel (vectorized global loads/stores, no TMA)."""
        # todo


def get_kernel_class(cfg):
    """Currently, only the CuTE DSL counterpart of quantize_transpose_tuned_1D is supported"""
    # todo: checks
    return NVFP4QuantizeTransposeTuned1DKernel


def compile_cutedsl_function_from_cfg(cfg):
    """
    Return the compiled CuTeDSL function object for the given MXFP8 quantization config.
    """

    kernel_class = get_kernel_class(cfg)
    kernel_obj = kernel_class(cfg)

    # todo: determine kernel parameters
    # todo: create fake tensors

    compiled = cute.compile(
        kernel_obj,
        # todo
        options="--enable-tvm-ffi",
    )
    return compiled


def get_nvfp4_quantization_function(
    fn_name: str,
    # todo
) -> bool:
    """Compile the NVFP4 quantize kernel for this config and register it in the TVM-FFI global registry"""

    if tvm_ffi.get_global_func(fn_name, allow_missing=True) is not None:
        return True

    try:
        cfg = NVFP4QuantizeConfig(
            # todo
        )
    except ValueError as e:
        logger.warning(
            "CuTeDSL NVFP4 backend does not support this config, "
            "falling back to the CUDA C++ kernel: %s",
            e,
        )
        return False

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
