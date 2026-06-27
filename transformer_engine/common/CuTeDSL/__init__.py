# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""CuTeDSL kernels for Transformer Engine.

Importing this package has a side effect: it registers the CuTeDSL kernel
entrypoints (e.g. ``get_mxfp8_quantization_function``) as TVM-FFI global
functions. The C++ dispatcher probes for those names via
``tvm::ffi::Function::GetGlobal`` — finding one means the process is running
inside a Python environment with the CuTeDSL toolchain available, so the kernel
may be compiled on demand; not finding it means a plain C++ environment, and
the dispatcher falls back to the CUDA C++ kernel.
"""

# Trigger the casting CuTeDSL entrypoints registration via TVM-FFI.
from . import cast  # noqa: F401
