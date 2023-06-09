# Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""Transformer Engine bindings for Paddle"""

from .cpp_extensions import gemm, fp8_gemm, cast_to_fp8, cast_from_fp8
