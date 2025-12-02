# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

# START_BF16_FP16_TRAINING

import jax
import jax.numpy as jnp
import optax
import transformer_engine.jax as te
from transformer_engine.jax.flax import TransformerLayer
from transformer_engine.jax.sharding import MeshResource, global_shard_guard


def run_forward_backward(params_dtype, compute_dtype):
    # Create TransformerLayer
    layer = TransformerLayer(
        hidden_size=1024,
        mlp_hidden_size=4096,
        num_attention_heads=16,
        dtype=params_dtype,
    )

    # Initialize parameters and optimizer
    init_key, dropout_key = jax.random.split(jax.random.PRNGKey(0))
    x = jax.random.normal(init_key, (32, 128, 1024), dtype=compute_dtype)
    params = layer.init({"params": init_key, "dropout": dropout_key}, x)

    # Create optimizer
    optimizer = optax.sgd(learning_rate=0.01)
    opt_state = optimizer.init(params)

    # Forward and backward pass
    def loss_fn(params):
        output = layer.apply(params, x, rngs={"dropout": dropout_key})
        assert output.dtype == compute_dtype
        return output.sum()

    loss, grads = jax.value_and_grad(loss_fn)(params)

    # Update parameters
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)


# Set up mesh resource for single GPU
with global_shard_guard(MeshResource()):
    run_forward_backward(jnp.float32, jnp.float32)  # high precision training
    run_forward_backward(jnp.float32, jnp.bfloat16)  # bfloat16 training with master weights in FP32
    run_forward_backward(jnp.bfloat16, jnp.bfloat16)  # bfloat16 training with weights in BF16

# END_BF16_FP16_TRAINING
