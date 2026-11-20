# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""CuTeDSL kernels for Transformer Engine.

To expose CuTeDSL kernels to C++, they should be registered via `register_cutedsl_backends()`.
They should provide a string function name which can be used to retrieve the corresponding CuTeDSL kernel function via TVM-FFI.
"""

import tvm_ffi

from transformer_engine.common.CuTeDSL.cast.mxfp8.dequantize_mxfp8 import (
    get_mxfp8_dequantization_function,
)
from transformer_engine.common.CuTeDSL.cast.mxfp8.quantize_mxfp8_common import (
    get_mxfp8_quantization_function,
    get_mxfp8_gated_quantization_function,
)
from transformer_engine.common.CuTeDSL.cast.mxfp8.group_quantize_mxfp8_common import (
    get_mxfp8_group_quantization_function,
)
from transformer_engine.common.CuTeDSL.cast.mxfp8.requantize_mxfp8 import (
    get_mxfp8_requantization_function,
)
from transformer_engine.common.CuTeDSL.cast.mxfp8.group_scaled_swiglu_mxfp8 import (
    get_mxfp8_group_scaled_swiglu_function,
)


def register_cutedsl_backends():
    """Register all available CuTeDSL backends for on-demand compilation via TVM-FFI.
    The C++ dispatcher retrieves them by the names defined here.
    """
    for entrypoint in (
        get_mxfp8_quantization_function,
        get_mxfp8_gated_quantization_function,
        get_mxfp8_group_quantization_function,
        get_mxfp8_dequantization_function,
        get_mxfp8_requantization_function,
        get_mxfp8_group_scaled_swiglu_function,
    ):
        tvm_ffi.register_global_func(entrypoint.__name__, entrypoint, override=True)
