# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import jax

# Check for Blackwell or newer GPU
gpu = jax.devices("gpu")[0]
major, minor = gpu.compute_capability.split(".")
assert int(major) >= 10, f"NVFP4 requires SM100 (Blackwell) or later, got SM{major}{minor}"

# START_NVFP4_EXAMPLE

import jax
import jax.numpy as jnp
import transformer_engine.jax as te
from transformer_engine.jax.flax import DenseGeneral
from transformer_engine.common.recipe import NVFP4BlockScaling, Format

# Define NVFP4 recipe
recipe = NVFP4BlockScaling(
    fp8_format=Format.E4M3,
    use_2d_weight_quantization=True,
    use_rht=True,
)

with te.autocast(enabled=True, recipe=recipe):
    # Initialize layer and data
    layer = DenseGeneral(features=1024)
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (32, 128, 1024), dtype=jnp.bfloat16)
    params = layer.init(key, x)

    # Forward and backward pass
    def loss_fn(params):
        output = layer.apply(params, x)
        return output.sum()

    loss, grads = jax.value_and_grad(loss_fn)(params)

# END_NVFP4_EXAMPLE
