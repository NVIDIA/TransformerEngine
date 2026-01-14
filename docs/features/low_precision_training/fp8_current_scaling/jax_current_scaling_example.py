# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

# START_CURRENT_SCALING_EXAMPLE

import jax
import jax.numpy as jnp
import transformer_engine.jax as te
from transformer_engine.jax.flax import DenseGeneral
from transformer_engine.common.recipe import Float8CurrentScaling, Format

# Create FP8 Current Scaling recipe
# Available formats:
#   - Format.HYBRID (default) -- E4M3 for forward pass, E5M2 for backward pass
#   - Format.E4M3 -- E4M3 for both forward and backward pass
recipe = Float8CurrentScaling(fp8_format=Format.HYBRID)

with te.autocast(enabled=True, recipe=recipe):
    # Create and initialize layer
    layer = DenseGeneral(features=1024)
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (32, 128, 1024), dtype=jnp.bfloat16)
    var_collect = layer.init(key, x)

    # Forward and backward pass
    def loss_fn(var_collect):
        output = layer.apply(var_collect, x)
        return output.sum()

    loss, grads = jax.value_and_grad(loss_fn)(var_collect)

# END_CURRENT_SCALING_EXAMPLE
