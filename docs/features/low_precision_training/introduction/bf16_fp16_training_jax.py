# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

# START_BF16_FP16_TRAINING

import jax
import jax.numpy as jnp
from transformer_engine.jax.flax import TransformerLayer


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
    var_collect = layer.init({"params": init_key, "dropout": dropout_key}, x)

    # Forward and backward pass
    def loss_fn(var_collect):
        output = layer.apply(var_collect, x, rngs={"dropout": dropout_key})
        assert output.dtype == compute_dtype
        return output.sum()

    loss, grads = jax.value_and_grad(loss_fn)(var_collect)


run_forward_backward(jnp.float32, jnp.float32)  # high precision training
run_forward_backward(jnp.float32, jnp.bfloat16)  # bfloat16 training with master weights in FP32
run_forward_backward(jnp.bfloat16, jnp.bfloat16)  # bfloat16 training with weights in BF16

# END_BF16_FP16_TRAINING
