# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""
Triton extensions for Transformer Engine JAX.

This module provides Triton kernel integration for TE primitives.

IMPORTANT: This module requires Triton to be installed. If you don't have Triton,
use transformer_engine.jax.cpp_extensions instead (CUDA/FFI based primitives).


Triton Package Options:
-----------------------
There are two compatible Triton packages:

1. Standard 'triton' from OpenAI (recommended for JAX-only environments):
       pip install triton

2. 'pytorch-triton' from PyTorch's index (for mixed JAX+PyTorch environments):
       pip install torch --index-url https://download.pytorch.org/whl/cu121
       # pytorch-triton is automatically installed as a dependency

   Both packages work with JAX Triton kernels. The pytorch-triton package
   has version format "X.Y.Z+<commit_sha>" (e.g., "3.0.0+45fff310c8").

WARNING: Do NOT run 'pip install pytorch-triton' directly! The package on PyPI
is a placeholder that will fail with "RuntimeError: Should never be installed".
The real pytorch-triton only comes bundled with PyTorch from PyTorch's index.


Environment Variables:
    NVTE_USE_PYTORCH_TRITON: If set to "1", acknowledge using pytorch-triton
        for JAX Triton kernels (suppresses compatibility warnings). Set this
        when both JAX and PyTorch are installed in the same environment.

        Example:
            export NVTE_USE_PYTORCH_TRITON=1


Usage:
    # Import utilities
    from transformer_engine.jax.triton_extensions import triton_call_lowering

    # Use in your primitive's lowering
    @staticmethod
    def lowering(ctx, x, **kwargs):
        return triton_call_lowering(ctx, my_kernel, x, ...)

    # Use permutation functions
    from transformer_engine.jax.triton_extensions import make_row_id_map, permute_with_mask_map

    # Check Triton package info
    from transformer_engine.jax.triton_extensions import get_triton_info
    info = get_triton_info()
    print(f"Using Triton {info['version']} from {info['source']}")
"""

from .utils import *
from .permutation import *
