# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""Shared functions for the encoder tests"""
from functools import lru_cache

from transformer_engine.transformer_engine_jax import get_device_compute_capability


@lru_cache
def is_bf16_supported():
    """Return if BF16 has hardware supported"""
    gpu_arch = get_device_compute_capability(0)
    return gpu_arch >= 80
