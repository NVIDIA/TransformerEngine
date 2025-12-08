# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import jax

# Requires Ada (SM89) or newer for FP8 support
cc = jax.devices()[0].device_kind
assert (
    "RTX 40" in cc
    or "RTX 5" in cc
    or "Ada" in cc
    or "L40" in cc
    or "H100" in cc
    or "H200" in cc
    or "GH" in cc
    or "B100" in cc
    or "B200" in cc
    or "GB" in cc
), "This example requires SM89 (Ada) or newer"

# START_DELAYED_SCALING_EXAMPLE

import jax
import jax.numpy as jnp
import transformer_engine.jax as te
from transformer_engine.jax.flax import DenseGeneral
from transformer_engine.common.recipe import DelayedScaling

# Create FP8 Delayed Scaling recipe
recipe = DelayedScaling(
    margin=0,  # Margin for scaling factor computation (default: 0)
    amax_history_len=1024,  # Length of amax history window (default: 1024)
    amax_compute_algo="max",  # How to compute amax from history (default: "max")
)

with te.autocast(enabled=True, recipe=recipe):
    # Initialize layer and data
    layer = DenseGeneral(features=1024)
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (32, 128, 1024), dtype=jnp.bfloat16)
    params = layer.init(key, x)

    # Forward and backward pass
    def loss_fn(params):
        output = layer.apply(params, x)
        return output.sum()

    loss, grads = jax.value_and_grad(loss_fn)(params)

# END_DELAYED_SCALING_EXAMPLE
