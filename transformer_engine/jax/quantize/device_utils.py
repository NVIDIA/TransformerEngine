# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""
Device utility functions for JAX quantization.

This module provides utility functions for checking device capabilities and compatibility
for quantization operations in JAX.
"""

import functools

import transformer_engine_jax

__all__ = [
    "get_device_compute_capability",
    "is_fp8_gemm_with_all_layouts_supported",
]


@functools.lru_cache(maxsize=None)
def get_device_compute_capability(gpu_id: int = 0) -> int:
    """
    Get the compute capability of the device.
    """
    return transformer_engine_jax.get_device_compute_capability(gpu_id)


@functools.lru_cache(maxsize=None)
def is_fp8_gemm_with_all_layouts_supported() -> bool:
    """Return True if using Blackwell architecture, False otherwise."""
    compute_capability = get_device_compute_capability()
    return 100 <= compute_capability < 120
