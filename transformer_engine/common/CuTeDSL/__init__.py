# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""CuTeDSL kernels for Transformer Engine.

Registration is explicit: call :func:`register_cutedsl_backends` to expose the
CuTeDSL kernel entrypoints (e.g. ``get_mxfp8_quantization_function``) as
TVM-FFI global functions. The C++ dispatcher (init_cutedsl_extension in the
PyTorch extension) imports this package and calls it once per process; it then
probes the names via ``tvm::ffi::Function::GetGlobal`` — finding one means the
CuTeDSL toolchain is available and the kernel may be compiled on demand, not
finding it means a plain C++ environment and the dispatcher falls back to the
CUDA C++ kernel.
"""

import tvm_ffi

from transformer_engine.common.CuTeDSL.cast.mxfp8.quantize_mxfp8 import (
    get_mxfp8_quantization_function,
)


def register_cutedsl_backends():
    # Register all available CuTeDSL backends for on-demand compilation via TVM-FFI.
    # The C++ dispatcher retrieves them by the names defined here.
    tvm_ffi.register_global_func(
        "get_mxfp8_quantization_function", get_mxfp8_quantization_function, override=True
    )
