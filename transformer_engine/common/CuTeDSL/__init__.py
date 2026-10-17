# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""CuTeDSL kernels for Transformer Engine.

To expose CuTeDSL kernels to C++, they should be registered via `register_cutedsl_backends()`.
They should provide a string function name which can be used to retrieve the corresponding CuTeDSL kernel function via TVM-FFI.
"""

import tvm_ffi

from transformer_engine.common.CuTeDSL.cast.mxfp8.quantize_mxfp8 import (
    get_mxfp8_quantization_function,
)


def register_cutedsl_backends():
    """Register all available CuTeDSL backends for on-demand compilation via TVM-FFI.
    The C++ dispatcher retrieves them by the names defined here.
    """
    tvm_ffi.register_global_func(
        "get_mxfp8_quantization_function", get_mxfp8_quantization_function, override=True
    )

    from transformer_engine.common.CuTeDSL.frost_attn import get_frost_attention_backward

    tvm_ffi.register_global_func(
        "get_frost_attention_backward", get_frost_attention_backward, override=True
    )

    from transformer_engine.common.CuTeDSL.frost_attn_fwd import (
        get_frost_attention_forward,
        get_frost_attention_forward_constant,
    )

    tvm_ffi.register_global_func(
        "get_frost_attention_forward", get_frost_attention_forward, override=True
    )
    tvm_ffi.register_global_func(
        "get_frost_attention_forward_constant", get_frost_attention_forward_constant, override=True
    )
