# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

from transformer_engine.jax.quantize import get_device_compute_capability

# Requires Ada (SM89) or newer for FP8 support
assert get_device_compute_capability() >= 89, "This example requires SM89 (Ada) or newer"

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
    var_collect = layer.init(key, x)

    # Forward and backward pass
    def loss_fn(var_collect):
        output = layer.apply(var_collect, x)
        return output.sum()

    loss, grads = jax.value_and_grad(loss_fn)(var_collect)

# END_DELAYED_SCALING_EXAMPLE
