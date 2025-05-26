# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""Shared functions for the encoder tests"""
from functools import lru_cache

import jax
import jax.numpy
import transformer_engine
from transformer_engine_jax import get_device_compute_capability
from transformer_engine.common import recipe
import numpy as np


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


def assert_params_sufficiently_sharded(params, mesh, tolerance=0.01, print_info=False):
    """Checks whether most params are sharded across sharding axis.

    (Adapted from https://github.com/AI-Hypercomputer/maxtext/blob/315e551e5942b24656a4250dcfca986fb4135b72/MaxText/maxtext_utils.py#L348)

    This function determines whether the majority of parameters are distributed
    across a specified sharding axes with an acceptable tolerance. It compares the
    current distribution to a scenario where all parameters are fully sharded
    across the axes on which the params are sharded e.g. 'tensor' axis.

    Args:
        params: params of the model state
        mesh: mesh constructed from config
        tolerance: float between 0.0 and 1.0 representing the allowed percentage of
        non-sharded parameters.
    """

    def get_product_num_devices_for_weight_sharding(weight_sharding_axes):
        product_num_devices_for_weight_sharding = 1
        for axis in weight_sharding_axes:
            product_num_devices_for_weight_sharding *= mesh.shape.get(axis, 1)
        return product_num_devices_for_weight_sharding

    def assert_leaf_sharding(path, arr):

        # Is the weight sharded? Get the axes on which it is sharded.
        partition_spec = arr.sharding.spec
        weight_sharding_axes = set(partition_spec) - set([None])  # None is not a sharding axis

        # Total number of devices on the axes on which the weight is sharded.
        product_num_devices_for_weight_sharding = get_product_num_devices_for_weight_sharding(
            weight_sharding_axes
        )

        # Params present in one shard (on one device).
        shard = arr.addressable_shards[0]
        params_per_chip = np.prod(shard.data.shape)

        # Total number of params (across all devicess).
        total_params = jax.numpy.size(arr)

        # Percentage of params that are unsharded.
        unsharded_perc = (
            (params_per_chip / (total_params / product_num_devices_for_weight_sharding) - 1) * 100
            if params_per_chip < total_params
            else 100
        )

        if print_info:
            print(
                f"{path}: {unsharded_perc:.2f}% unsharded, unsharded param shape={arr.shape},"
                f" partition spec={partition_spec}"
            )

        # If the weight is sharded on any axis, then the percentage of
        # unsharded params should be less than the tolerance.
        assert (
            product_num_devices_for_weight_sharding == 1 or unsharded_perc < tolerance
        ), f"{path}: {unsharded_perc:.2f}% unsharded"

    jax.tree_util.tree_map_with_path(
        lambda p, x: assert_leaf_sharding("/".join(str(x) for x in p), x), params
    )


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
