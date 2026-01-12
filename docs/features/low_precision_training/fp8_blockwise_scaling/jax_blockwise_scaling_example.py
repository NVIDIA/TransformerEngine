# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

# Check for Hopper or newer GPU
from transformer_engine_jax import get_device_compute_capability

major_minor = get_device_compute_capability(0)
assert (
    major_minor >= 90
), f"FP8 Blockwise Scaling requires SM90 (Hopper) or later, got SM{major_minor}"

# START_BLOCKWISE_SCALING_EXAMPLE

import jax
import jax.numpy as jnp
import transformer_engine.jax as te
from transformer_engine.jax.flax import DenseGeneral
from transformer_engine.common.recipe import Float8BlockScaling

# Create FP8 Blockwise Scaling recipe
recipe = Float8BlockScaling(
    fp8_format=te.common.recipe.Format.E4M3,  # E4M3 or HYBRID (default: E4M3)
    x_block_scaling_dim=1,  # 1D scaling for activations (default: 1)
    w_block_scaling_dim=2,  # 2D scaling for weights (default: 2)
    grad_block_scaling_dim=1,  # 1D scaling for gradients (default: 1)
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

# END_BLOCKWISE_SCALING_EXAMPLE
