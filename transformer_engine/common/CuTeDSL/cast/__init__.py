# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""CuTeDSL cast/quantization kernels. Importing pulls in each kernel module so
its TVM-FFI entrypoint is registered."""

from . import mxfp8  # noqa: F401  (import side effect: registers global funcs)
