# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

# START_NVFP4_EXAMPLE

import jax
import jax.numpy as jnp
import optax
import transformer_engine.jax as te
from transformer_engine.jax.flax import DenseGeneral
from transformer_engine.jax.sharding import MeshResource, global_shard_guard
from transformer_engine.common.recipe import NVFP4Recipe, Format

# Define NVFP4 recipe
recipe = NVFP4Recipe(
    fp8_format=Format.E4M3,
    use_2d_weight_quantization=True,
    use_rht=True,
)

with global_shard_guard(MeshResource()):
    with te.fp8_autocast(enabled=True, recipe=recipe, mesh_resource=MeshResource()):
        # Initialize layer and data
        layer = DenseGeneral(features=1024)
        key = jax.random.PRNGKey(0)
        x = jax.random.normal(key, (32, 128, 1024), dtype=jnp.bfloat16)
        params = layer.init(key, x)

        # Training step
        def loss_fn(params):
            output = layer.apply(params, x)
            return output.sum()

        loss, grads = jax.value_and_grad(loss_fn)(params)

        # Update parameters
        optimizer = optax.adamw(learning_rate=1e-4)
        opt_state = optimizer.init(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)

# END_NVFP4_EXAMPLE
