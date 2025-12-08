# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

# START_CURRENT_SCALING_EXAMPLE

import jax
import jax.numpy as jnp
import optax
import transformer_engine.jax as te
from transformer_engine.jax.flax import DenseGeneral
from transformer_engine.jax.sharding import MeshResource, global_shard_guard
from transformer_engine.jax.quantize import Float8CurrentScaling, Format

# Create FP8 Current Scaling recipe
# Available formats:
#   - Format.HYBRID (default) -- E4M3 for forward pass, E5M2 for backward pass
#   - Format.E4M3 -- E4M3 for both forward and backward pass
recipe = Float8CurrentScaling(fp8_format=Format.HYBRID)

with global_shard_guard(MeshResource()):
    with te.fp8_autocast(enabled=True, recipe=recipe, mesh_resource=MeshResource()):
        # Create and initialize layer
        layer = DenseGeneral(features=1024)
        key = jax.random.PRNGKey(0)
        x = jax.random.normal(key, (32, 128, 1024), dtype=jnp.bfloat16)
        params = layer.init(key, x)

        # Training with FP8 Current Scaling
        def loss_fn(params):
            output = layer.apply(params, x)
            return output.sum()

        loss, grads = jax.value_and_grad(loss_fn)(params)

        # Update parameters
        optimizer = optax.sgd(learning_rate=0.01)
        opt_state = optimizer.init(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)

# END_CURRENT_SCALING_EXAMPLE
