# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""CuTeDSL kernels for Transformer Engine.

To expose CuTeDSL kernels to C++, they should be registered via `register_cutedsl_backends()`.
They should provide a string function name which can be used to retrieve the corresponding CuTeDSL kernel function via TVM-FFI.
"""

import logging

import tvm_ffi

from transformer_engine.common.CuTeDSL.cast.mxfp8.dequantize_mxfp8 import (
    get_mxfp8_dequantization_function,
)
from transformer_engine.common.CuTeDSL.cast.mxfp8.quantize_mxfp8_common import (
    get_mxfp8_quantization_function,
    get_mxfp8_gated_quantization_function,
)
from transformer_engine.common.CuTeDSL.cast.mxfp8.requantize_mxfp8 import (
    get_mxfp8_requantization_function,
)

logger = logging.getLogger("transformer_engine.cutedsl")


def _register_ep_backends():
    """Register the EP grouped GEMM backends; a failure leaves the other backends registered."""
    try:
        from transformer_engine.common.CuTeDSL.ep_gemm import (  # pylint: disable=import-outside-toplevel
            dispatch_gemm,
            combine_gemm,
            requantize,
            wgrad_gemm,
        )

        for function in (
            dispatch_gemm.get_ep_dispatch_gemm_function,
            dispatch_gemm.get_ep_dispatch_gemm_workspace_layout,
            combine_gemm.get_ep_combine_gemm_function,
            combine_gemm.get_ep_combine_gemm_workspace_layout,
            wgrad_gemm.get_ep_wgrad_function,
            wgrad_gemm.get_ep_wgrad_workspace_layout,
            wgrad_gemm.get_ep_wgrad_reads_descriptors,
            requantize.get_ep_requant_function,
        ):
            tvm_ffi.register_global_func(function.__name__, function, override=True)
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.warning("EP grouped GEMM CuTeDSL backends are unavailable: %s", e)


def register_cutedsl_backends():
    """Register all available CuTeDSL backends for on-demand compilation via TVM-FFI.
    The C++ dispatcher retrieves them by the names defined here.
    """
    for entrypoint in (
        get_mxfp8_quantization_function,
        get_mxfp8_gated_quantization_function,
        get_mxfp8_dequantization_function,
        get_mxfp8_requantization_function,
    ):
        tvm_ffi.register_global_func(entrypoint.__name__, entrypoint, override=True)
    _register_ep_backends()
