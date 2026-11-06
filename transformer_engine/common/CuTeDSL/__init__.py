# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""CuTeDSL kernels for Transformer Engine.

Each kernel factory exported here takes a kernel config and returns the compiled TVM-FFI
function, or None if the config is unsupported. The C++ dispatcher calls it by name through
`compile_kernel.py` in a separate process and loads the exported library.
"""

from transformer_engine.common.CuTeDSL.cast.mxfp8.quantize_mxfp8 import (
    get_mxfp8_quantization_function,
)
