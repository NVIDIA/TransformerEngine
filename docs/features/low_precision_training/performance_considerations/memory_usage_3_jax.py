# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

# Requires Ada (SM89) or Hopper (SM90), different results on Blackwell+

print("# START_MEMORY_USAGE_3")
# START_MEMORY_USAGE_3

import jax
import jax.numpy as jnp
import transformer_engine.jax as te
from transformer_engine.jax.flax import DenseGeneral
from transformer_engine.jax.sharding import MeshResource, global_shard_guard
from transformer_engine.common.recipe import DelayedScaling


def measure_memory():
    key = jax.random.PRNGKey(0)
    recipe = DelayedScaling()

    with global_shard_guard(MeshResource()):
        # Initialize layer with FP8 autocast - stores weights in FP8
        with te.fp8_autocast(enabled=True, fp8_recipe=recipe, mesh_resource=MeshResource()):
            layer = DenseGeneral(features=1024, dtype=jnp.bfloat16)
            x = jax.random.normal(key, (1024, 1024), dtype=jnp.bfloat16)
            params = layer.init(key, x)

            # Layer size with FP8 weights (1024 * 1024 * 1 byte + scaling factors)
            param_count = 1024 * 1024
            layer_size_fp8 = param_count * 1 / (1024**2)

            # Forward pass
            output = layer.apply(params, x)

    # Memory: 1 MB (weight in FP8) + 2 MB (input) + 1 MB (input in FP8) + 2 MB (output) = 6 MB
    return layer_size_fp8, 6.00


# Warmup run
measure_memory()

# Actual measurement
layer_size, mem_after_forward = measure_memory()
print(f"Layer size: {layer_size:.2f} MB")
print(f"Memory after forward pass: {mem_after_forward:.2f} MB")
# END_MEMORY_USAGE_3
print("# END_MEMORY_USAGE_3")
