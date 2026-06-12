# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""MXFP8 CuTeDSL kernels. Importing ``quantize_mxfp8`` runs its module body,
which registers the ``get_mxfp8_quantization_function`` TVM-FFI global func."""

from . import quantize_mxfp8  # noqa: F401  (import side effect: registers the global func)
