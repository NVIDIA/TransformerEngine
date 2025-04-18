# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""Shared functions for the encoder tests"""
from functools import lru_cache

import transformer_engine
from transformer_engine_jax import get_device_compute_capability
from transformer_engine.common import recipe


@lru_cache
def is_bf16_supported():
    """Return if BF16 has hardware supported"""
    gpu_arch = get_device_compute_capability(0)
    return gpu_arch >= 80


@lru_cache
def is_fp8_supported():
    """Return if FP8 has hardware supported"""
    gpu_arch = get_device_compute_capability(0)
    return gpu_arch >= 90


@lru_cache
def is_mxfp8_supported():
    """Return if FP8 has hardware supported"""
    gpu_arch = get_device_compute_capability(0)
    return gpu_arch >= 100


def get_fp8_recipe_from_name_string(name: str):
    """Query recipe from a given name string"""
    match name:
        case "DelayedScaling":
            return recipe.DelayedScaling()
        case "MXFP8BlockScaling":
            return recipe.MXFP8BlockScaling()
        case "Float8CurrentScaling":
            return recipe.Float8CurrentScaling()
        case _:
            raise ValueError(f"Invalid fp8_recipe, got {name}")
