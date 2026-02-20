# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

# Requires Ada (SM89) or Hopper (SM90), different results on Blackwell+

print("# START_MEMORY_USAGE_2")

import jax
import jax.numpy as jnp
import transformer_engine.jax as te
from transformer_engine.jax.flax import DenseGeneral
from transformer_engine.common.recipe import DelayedScaling


key = jax.random.PRNGKey(0)
recipe = DelayedScaling()
jax.clear_caches()


# Initialize layer with BF16 parameters (outside autocast)
layer = DenseGeneral(features=1024, dtype=jnp.bfloat16)
x = jax.random.normal(key, (1024, 1024), dtype=jnp.bfloat16)


# Forward and backward pass with FP8 compute
with te.autocast(enabled=True, recipe=recipe):
    var_collect = layer.init(key, x)

    @jax.jit
    def loss_fn(var_collect, x):
        output = layer.apply(var_collect, x)
        return output.sum()

    # Trace the backward pass - this allocates saved tensors
    _, backward_fn = jax.vjp(loss_fn, var_collect, x)

del x

print("Tensors in memory:")
total_bytes = 0
for arr in jax.live_arrays():
    total_bytes += arr.nbytes
    if arr.nbytes > 200000:  # do not count small tensors
        print(f"  Shape: {arr.shape}, Dtype: {arr.dtype}, Size: {arr.nbytes / 1024:.1f} KB")
print(f"  Total from all live arrays: {total_bytes / (1024**2):.2f} MB")

print("# END_MEMORY_USAGE_2")
