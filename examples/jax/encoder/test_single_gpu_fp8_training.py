# Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
""" Encoder with FP8 Training on single GPU"""
import jax
import jax.numpy as jnp
import optax
from cuda import cudart
from flax.core.frozen_dict import FrozenDict
from flax.training import train_state

import transformer_engine.jax as te
from transformer_engine.jax.fp8 import FP8Helper
from transformer_engine.common.recipe import Format as FP8Format
from transformer_engine.common.recipe import DelayedScaling

PARAMS_KEY = 'params'

BATCH = 32
SEQLEN = 512
HIDDEN = 1024


def gpu_has_fp8():
    """GPU arch has to support FP8"""
    cudaSuccess = cudart.cudaError_t.cudaSuccess
    ret, gpu_id = cudart.cudaGetDevice()
    assert ret == cudaSuccess
    flag = cudart.cudaDeviceAttr.cudaDevAttrComputeCapabilityMajor
    _, major = cudart.cudaDeviceGetAttribute(flag, gpu_id)
    flag = cudart.cudaDeviceAttr.cudaDevAttrComputeCapabilityMinor
    _, minor = cudart.cudaDeviceGetAttribute(flag, gpu_id)
    sm_arch = major * 10 + minor
    return sm_arch >= 89


def network():
    """NLP Encoder"""
    encoder = te.TransformerLayer(hidden_size=HIDDEN,
                                  mlp_hidden_size=4 * HIDDEN,
                                  hidden_dropout=0.0,
                                  attention_dropout=0.0,
                                  layernorm_type='rmsnorm',
                                  mlp_activations=('gelu', 'linear'),
                                  layer_type=te.TransformerLayerType.ENCODER,
                                  transpose_batch_sequence=True,
                                  dtype=jnp.bfloat16)
    return encoder


def synthesis_data(data_rng):
    """Dataset generator"""
    return jax.random.normal(data_rng, [SEQLEN, BATCH, HIDDEN], jnp.bfloat16)


def train_step(batch, state, others):
    """Training function."""

    def loss_fn(collections):
        logits = state.apply_fn(collections, batch)
        loss = jnp.mean(logits)
        return loss

    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(FrozenDict({PARAMS_KEY: state.params, **others}))
    grads, params_grads = grads.pop(PARAMS_KEY)
    state = state.apply_gradients(grads=params_grads)
    others = FP8Helper.update_fp8_metas(grads)
    return loss, state, others


def test_encoder():
    """Encoder example"""
    if gpu_has_fp8() is False:
        print("GPU doesn't support FP8")
        return

    rng = jax.random.PRNGKey(0)
    rng, init_rng, data_rng = jax.random.split(rng, 3)
    inputs = synthesis_data(data_rng)
    optimizer = optax.sgd(0.001, 0.9)

    with te.fp8_autocast(enabled=True, fp8_recipe=DelayedScaling(fp8_format=FP8Format.HYBRID)):
        encoder = network()
        variables = jax.jit(encoder.init)(init_rng, inputs)
        variables, params = variables.pop(PARAMS_KEY)
        state = train_state.TrainState.create(apply_fn=encoder.apply, params=params, tx=optimizer)
        jitted_train_step = jax.jit(train_step)
        assert "fp8" in str(jax.make_jaxpr(jitted_train_step)(inputs, state, variables))

        for i in range(5):
            rng, data_rng = jax.random.split(rng)
            inputs = synthesis_data(data_rng)
            loss, state, variables = jitted_train_step(inputs, state, variables)
            print(f"Step {i} - Loss: {loss}")


if __name__ == "__main__":
    test_encoder()
