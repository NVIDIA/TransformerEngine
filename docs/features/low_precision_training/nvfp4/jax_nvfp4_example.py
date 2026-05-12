# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

# Check for Blackwell or newer GPU
from transformer_engine.jax.quantize import get_device_compute_capability

assert (
    get_device_compute_capability() >= 100
), f"NVFP4 requires SM100 (Blackwell) or later, got SM{get_device_compute_capability()}"

# START_NVFP4_EXAMPLE

import jax
import jax.numpy as jnp
import transformer_engine.jax as te
from transformer_engine.jax.flax import DenseGeneral
from transformer_engine.common.recipe import NVFP4BlockScaling

# Define NVFP4 recipe
# 2D weight quantization and RHT are enabled by default
recipe = NVFP4BlockScaling()
# To disable features, use:
#   recipe = NVFP4BlockScaling(disable_rht=True, disable_2d_quantization=True)

with te.autocast(enabled=True, recipe=recipe):
    # Initialize layer and data
    layer = DenseGeneral(features=1024)
    key, sr_key = jax.random.split(jax.random.PRNGKey(0))
    x = jax.random.normal(key, (32, 128, 1024), dtype=jnp.bfloat16)

    # NVFP4 requires sr_rng for stochastic rounding
    rngs = {"sr_rng": sr_key}
    var_collect = layer.init({"params": key, "sr_rng": sr_key}, x)

    # Forward and backward pass
    def loss_fn(var_collect):
        output = layer.apply(var_collect, x, rngs=rngs)
        return output.sum()

    loss, grads = jax.value_and_grad(loss_fn)(var_collect)

# END_NVFP4_EXAMPLE
