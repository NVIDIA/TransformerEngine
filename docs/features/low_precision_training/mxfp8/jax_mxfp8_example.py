# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

# Check for Blackwell or newer GPU
from transformer_engine.jax.quantize import get_device_compute_capability

assert (
    get_device_compute_capability() >= 100
), f"MXFP8 requires SM100 (Blackwell) or later, got SM{get_device_compute_capability()}"

# START_MXFP8_EXAMPLE

import jax
import jax.numpy as jnp
import transformer_engine.jax as te
from transformer_engine.jax.flax import DenseGeneral
from transformer_engine.common.recipe import MXFP8BlockScaling, Format

# Create MXFP8 recipe
recipe = MXFP8BlockScaling(
    fp8_format=Format.E4M3,  # FP8 format (default: E4M3, E5M2 not supported)
)

with te.autocast(enabled=True, recipe=recipe):
    # Initialize layer and data
    layer = DenseGeneral(features=1024)
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (32, 128, 1024), dtype=jnp.bfloat16)
    var_collect = layer.init(key, x)

    # Forward and backward pass
    def loss_fn(var_collect):
        output = layer.apply(var_collect, x)
        return output.sum()

    loss, grads = jax.value_and_grad(loss_fn)(var_collect)

# END_MXFP8_EXAMPLE
