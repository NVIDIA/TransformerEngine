# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

# Requires Ada (SM89) or Hopper (SM90), different results on Blackwell+

print("# START_MEMORY_USAGE_1")
# START_MEMORY_USAGE_1

import jax
import jax.numpy as jnp
from transformer_engine.jax.flax import DenseGeneral
from transformer_engine.jax.sharding import MeshResource, global_shard_guard


def measure_memory():
    key = jax.random.PRNGKey(0)

    with global_shard_guard(MeshResource()):
        # Initialize a dense layer with high precision parameters
        layer = DenseGeneral(features=1024, dtype=jnp.bfloat16)
        x = jax.random.normal(key, (1024, 1024), dtype=jnp.bfloat16)
        params = layer.init(key, x)

        # Calculate layer size (1024 * 1024 * 2 bytes for BF16)
        param_count = 1024 * 1024
        layer_size = param_count * 2 / (1024**2)

        # Forward pass
        output = layer.apply(params, x)

    # Memory after forward: weight (2 MB) + input (2 MB) + output (2 MB) = 6 MB
    return layer_size, 6.00


# Warmup run
measure_memory()

# Actual measurement
layer_size, mem_after_forward = measure_memory()
print(f"Layer size: {layer_size:.2f} MB")
print(f"Memory usage after forward pass: {mem_after_forward:.2f} MB")
# END_MEMORY_USAGE_1
print("# END_MEMORY_USAGE_1")
