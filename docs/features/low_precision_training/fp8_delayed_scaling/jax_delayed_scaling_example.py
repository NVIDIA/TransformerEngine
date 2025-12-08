# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import jax

# Requires Ada (SM89) or newer for FP8 support
cc = jax.devices()[0].device_kind
assert (
    "RTX 40" in cc
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
import optax
import transformer_engine.jax as te
from transformer_engine.jax.flax import DenseGeneral
from transformer_engine.jax.sharding import MeshResource, global_shard_guard
from transformer_engine.common.recipe import DelayedScaling

# Create FP8 Delayed Scaling recipe
recipe = DelayedScaling(
    margin=0,  # Margin for scaling factor computation (default: 0)
    amax_history_len=1024,  # Length of amax history window (default: 1024)
    amax_compute_algo="max",  # How to compute amax from history (default: "max")
)

with global_shard_guard(MeshResource()):
    with te.autocast(enabled=True, recipe=recipe, mesh_resource=MeshResource()):
        # Initialize layer and data
        layer = DenseGeneral(features=1024)
        key = jax.random.PRNGKey(0)
        x = jax.random.normal(key, (32, 128, 1024), dtype=jnp.bfloat16)
        params = layer.init(key, x)

        # Training with FP8 Delayed Scaling
        def loss_fn(params):
            output = layer.apply(params, x)
            return output.sum()

        loss, grads = jax.value_and_grad(loss_fn)(params)

        # Update parameters
        optimizer = optax.adamw(learning_rate=1e-4)
        opt_state = optimizer.init(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)

# END_DELAYED_SCALING_EXAMPLE
