# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

from transformer_engine.jax.quantize import get_device_compute_capability

# Requires Ada (SM89) or newer for FP8 support
assert get_device_compute_capability() >= 89, "This example requires SM89 (Ada) or newer"

# START_AUTOCAST_BASIC

import jax
import jax.numpy as jnp
import transformer_engine.jax as te
from transformer_engine.jax.flax import TransformerLayer
from transformer_engine.common.recipe import DelayedScaling, Format

# Set up recipe
recipe = DelayedScaling()

# Model initialization must happen inside autocast
with te.autocast(enabled=True, recipe=recipe):
    layer = TransformerLayer(
        hidden_size=1024,
        mlp_hidden_size=4096,
        num_attention_heads=16,
    )

    init_key, dropout_key = jax.random.split(jax.random.PRNGKey(0))
    x = jax.random.normal(init_key, (32, 128, 1024), dtype=jnp.bfloat16)
    var_collect = layer.init({"params": init_key, "dropout": dropout_key}, x)

    # Forward and backward pass (both inside autocast for JAX)
    def loss_fn(var_collect):
        output = layer.apply(var_collect, x, rngs={"dropout": dropout_key})
        return output.sum()

    loss, grads = jax.value_and_grad(loss_fn)(var_collect)

# END_AUTOCAST_BASIC


# START_AUTOCAST_SEQUENTIAL

encoder_recipe = DelayedScaling(fp8_format=Format.E4M3)
decoder_recipe = DelayedScaling(fp8_format=Format.HYBRID)

with te.autocast(enabled=True, recipe=encoder_recipe):
    encoder = TransformerLayer(hidden_size=1024, mlp_hidden_size=4096, num_attention_heads=16)
    encoder_var_collect = encoder.init({"params": init_key, "dropout": dropout_key}, x)
    hidden = encoder.apply(encoder_var_collect, x, rngs={"dropout": dropout_key})

with te.autocast(enabled=True, recipe=decoder_recipe):
    decoder = TransformerLayer(hidden_size=1024, mlp_hidden_size=4096, num_attention_heads=16)
    decoder_var_collect = decoder.init({"params": init_key, "dropout": dropout_key}, hidden)
    output = decoder.apply(decoder_var_collect, hidden, rngs={"dropout": dropout_key})

# END_AUTOCAST_SEQUENTIAL


# START_AUTOCAST_NESTED

outer_recipe = DelayedScaling(fp8_format=Format.E4M3)
inner_recipe = DelayedScaling(fp8_format=Format.HYBRID)

with te.autocast(enabled=True, recipe=outer_recipe):
    # layer1 uses outer_recipe
    layer1 = TransformerLayer(hidden_size=1024, mlp_hidden_size=4096, num_attention_heads=16)
    var_collect1 = layer1.init({"params": init_key, "dropout": dropout_key}, x)
    hidden = layer1.apply(var_collect1, x, rngs={"dropout": dropout_key})

    with te.autocast(enabled=True, recipe=inner_recipe):
        # layer2 uses inner_recipe (overrides outer)
        layer2 = TransformerLayer(hidden_size=1024, mlp_hidden_size=4096, num_attention_heads=16)
        var_collect2 = layer2.init({"params": init_key, "dropout": dropout_key}, hidden)
        hidden = layer2.apply(var_collect2, hidden, rngs={"dropout": dropout_key})

    # layer3 uses outer_recipe again
    layer3 = TransformerLayer(hidden_size=1024, mlp_hidden_size=4096, num_attention_heads=16)
    var_collect3 = layer3.init({"params": init_key, "dropout": dropout_key}, hidden)
    output = layer3.apply(var_collect3, hidden, rngs={"dropout": dropout_key})

# END_AUTOCAST_NESTED
