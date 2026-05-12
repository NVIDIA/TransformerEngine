# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

# Requires Ada (SM89) or Hopper (SM90), different results on Blackwell+

# START_FUSED_LAYERS

import jax
import jax.numpy as jnp
import transformer_engine.jax as te
from transformer_engine.jax.flax import LayerNorm, DenseGeneral, LayerNormDenseGeneral
from transformer_engine.common.recipe import DelayedScaling

key = jax.random.PRNGKey(0)
x = jax.random.normal(key, (32, 128, 1024), dtype=jnp.bfloat16)

# Example 1: Separate LayerNorm and DenseGeneral layers
layer_norm = LayerNorm()
dense = DenseGeneral(features=1024)

# Initialize parameters
ln_params = layer_norm.init(key, x)
dense_params = dense.init(key, x)

# Two separate operations
normalized = layer_norm.apply(ln_params, x)
output_separate = dense.apply(dense_params, normalized)

# Example 2: Fused LayerNormDenseGeneral layer
fused_layer = LayerNormDenseGeneral(features=1024)

# Initialize and apply with FP8 autocast
recipe = DelayedScaling()
with te.autocast(enabled=True, recipe=recipe):
    fused_params = fused_layer.init(key, x)
    output_fused, _ = fused_layer.apply(fused_params, x)  # Returns (output, ln_output)

# The fused layer is more efficient as it combines LayerNorm and quantization

# END_FUSED_LAYERS
