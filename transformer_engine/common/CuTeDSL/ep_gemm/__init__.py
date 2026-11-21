# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Grouped GEMMs overlapped with EP communication, loaded through TVM-FFI.

dispatch_gemm: reads row tiles as a dispatch delivers them (FC1 forward, FC2 data gradient).
combine_gemm: publishes column chunks for a combine to send (FC2 forward, FC1 data gradient).
wgrad_gemm: weight gradients chained behind the backward GEMMs.
requantize: columnwise MXFP8 operands of the weight gradients from received rowwise rows.
"""

from . import dispatch_gemm, combine_gemm, requantize, wgrad_gemm
