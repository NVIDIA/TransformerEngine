# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

# Requires Ada (SM89) or Hopper (SM90), different results on Blackwell+

print("# START_MEMORY_USAGE_1")

import jax
import jax.numpy as jnp
from transformer_engine.jax.flax import DenseGeneral


def get_gpu_memory_mb():
    """Get current GPU memory usage in MB."""
    jax.effects_barrier()
    stats = jax.local_devices()[0].memory_stats()
    return stats["bytes_in_use"] / (1024**2) if stats else 0.0


def measure_memory():
    key = jax.random.PRNGKey(0)
    jax.clear_caches()

    init_memory = get_gpu_memory_mb()

    # Initialize layer with BF16 parameters
    layer = DenseGeneral(features=1024, dtype=jnp.bfloat16)
    x = jax.random.normal(key, (1024, 1024), dtype=jnp.bfloat16)
    params = layer.init(key, x)

    # Forward pass in high precision
    output = layer.apply(params, x)

    mem_after_forward = get_gpu_memory_mb() - init_memory
    return mem_after_forward


# Warmup run
measure_memory()

# Actual measurement
mem_after_forward = measure_memory()
print(f"Memory usage after forward pass: {mem_after_forward:.2f} MB")

print("# END_MEMORY_USAGE_1")
