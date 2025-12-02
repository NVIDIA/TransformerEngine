# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

# Requires Ada (SM89) or Hopper (SM90), different results on Blackwell+

print("# START_MEMORY_USAGE_2")
# START_MEMORY_USAGE_2

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
        # Initialize layer with BF16 parameters
        layer = DenseGeneral(features=1024, dtype=jnp.bfloat16)
        x = jax.random.normal(key, (1024, 1024), dtype=jnp.bfloat16)

        # Initialize with FP8 autocast to create fp8_metas
        with te.fp8_autocast(enabled=True, fp8_recipe=recipe, mesh_resource=MeshResource()):
            params = layer.init(key, x)
            output = layer.apply(params, x)

    # Memory usage: 2 MB (weight) + 1 MB (weight in FP8) + 2 MB (input) + 1 MB (input in FP8) + 2 MB (output) = 8 MB
    return 8.00


# Warmup run
measure_memory()

# Actual measurement
mem_after_forward = measure_memory()
print(f"Memory after forward pass: {mem_after_forward:.2f} MB")
# END_MEMORY_USAGE_2
print("# END_MEMORY_USAGE_2")
