# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import jax

# Requires Ada (SM89) or newer for FP8 support
cc = jax.devices()[0].device_kind
assert (
    "RTX 40" in cc
    or "L40" in cc
    or "H100" in cc
    or "H200" in cc
    or "GH" in cc
    or "B100" in cc
    or "B200" in cc
    or "GB" in cc
), "This example requires SM89 (Ada) or newer"

# START_AUTOCAST_BASIC

import jax
import jax.numpy as jnp
import transformer_engine.jax as te
from transformer_engine.jax.flax import TransformerLayer
from transformer_engine.jax.sharding import MeshResource, global_shard_guard
from transformer_engine.jax.quantize import get_delayed_scaling_recipe

# Set up mesh resource and recipe
recipe = get_delayed_scaling_recipe()

with global_shard_guard(MeshResource()):
    # Model initialization must happen inside autocast
    with te.autocast(enabled=True, recipe=recipe, mesh_resource=MeshResource()):
        layer = TransformerLayer(
            hidden_size=1024,
            mlp_hidden_size=4096,
            num_attention_heads=16,
        )

        init_key, dropout_key = jax.random.split(jax.random.PRNGKey(0))
        x = jax.random.normal(init_key, (32, 128, 1024), dtype=jnp.bfloat16)
        params = layer.init({"params": init_key, "dropout": dropout_key}, x)

        # Forward and backward pass (both inside autocast for JAX)
        def loss_fn(params):
            output = layer.apply(params, x, rngs={"dropout": dropout_key})
            return output.sum()

        loss, grads = jax.value_and_grad(loss_fn)(params)

# END_AUTOCAST_BASIC


# START_AUTOCAST_SEQUENTIAL

from transformer_engine.common.recipe import DelayedScaling

encoder_recipe = DelayedScaling(fp8_format="E4M3")
decoder_recipe = DelayedScaling(fp8_format="HYBRID")

with global_shard_guard(MeshResource()):
    with te.autocast(enabled=True, recipe=encoder_recipe, mesh_resource=MeshResource()):
        encoder = TransformerLayer(hidden_size=1024, mlp_hidden_size=4096, num_attention_heads=16)
        encoder_params = encoder.init({"params": init_key, "dropout": dropout_key}, x)
        hidden = encoder.apply(encoder_params, x, rngs={"dropout": dropout_key})

    with te.autocast(enabled=True, recipe=decoder_recipe, mesh_resource=MeshResource()):
        decoder = TransformerLayer(hidden_size=1024, mlp_hidden_size=4096, num_attention_heads=16)
        decoder_params = decoder.init({"params": init_key, "dropout": dropout_key}, hidden)
        output = decoder.apply(decoder_params, hidden, rngs={"dropout": dropout_key})

# END_AUTOCAST_SEQUENTIAL


# START_AUTOCAST_NESTED

outer_recipe = DelayedScaling(fp8_format="E4M3")
inner_recipe = DelayedScaling(fp8_format="HYBRID")

with global_shard_guard(MeshResource()):
    with te.autocast(enabled=True, recipe=outer_recipe, mesh_resource=MeshResource()):
        # layer1 uses outer_recipe
        layer1 = TransformerLayer(hidden_size=1024, mlp_hidden_size=4096, num_attention_heads=16)
        params1 = layer1.init({"params": init_key, "dropout": dropout_key}, x)
        hidden = layer1.apply(params1, x, rngs={"dropout": dropout_key})

        with te.autocast(enabled=True, recipe=inner_recipe, mesh_resource=MeshResource()):
            # layer2 uses inner_recipe (overrides outer)
            layer2 = TransformerLayer(
                hidden_size=1024, mlp_hidden_size=4096, num_attention_heads=16
            )
            params2 = layer2.init({"params": init_key, "dropout": dropout_key}, hidden)
            hidden = layer2.apply(params2, hidden, rngs={"dropout": dropout_key})

        # layer3 uses outer_recipe again
        layer3 = TransformerLayer(hidden_size=1024, mlp_hidden_size=4096, num_attention_heads=16)
        params3 = layer3.init({"params": init_key, "dropout": dropout_key}, hidden)
        output = layer3.apply(params3, hidden, rngs={"dropout": dropout_key})

# END_AUTOCAST_NESTED
