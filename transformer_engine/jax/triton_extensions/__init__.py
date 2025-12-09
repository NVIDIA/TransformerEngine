# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""
Triton extensions for Transformer Engine JAX.

This module provides Triton kernel integration for TE primitives.

IMPORTANT: This module requires Triton to be installed. If you don't have Triton,
use transformer_engine.jax.cpp_extensions instead (CUDA/FFI based primitives).

Install Triton: pip install triton


Usage:
    # Import utilities
    from transformer_engine.jax.triton_extensions import triton_call_lowering

    # Use in your primitive's lowering
    @staticmethod
    def lowering(ctx, x, **kwargs):
        return triton_call_lowering(ctx, my_kernel, x, ...)

    # Use permutation functions
    from transformer_engine.jax.triton_extensions import make_row_id_map, permute_with_mask_map
"""

from .utils import *
from .permutation import *
