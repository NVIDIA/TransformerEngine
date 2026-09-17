# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""Parity checks for TE-owned DeepSeek-V4 CSA per-kernel VJPs."""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp

from transformer_engine.jax.deepseek_v4 import (
    csa_compressor,
    dsa_indexer,
    dsa_sparse_attention,
)


def _require_sm100():
    if not any(device.platform == "gpu" for device in jax.devices()):
        pytest.skip("JAX has no CUDA device")
    from cudnn.tensor_adapter import get_compute_capability

    if get_compute_capability()[0] < 10:
        pytest.skip("DSv4 CuTeDSL kernels require compute capability >= 10")


@pytest.mark.L0
@pytest.mark.parametrize("head_dim", [128, 512])
def test_csa_compressor_vjp(head_dim):
    _require_sm100()
    key = jax.random.key(head_dim)
    kv = jax.random.normal(key, (16, 2 * head_dim), jnp.bfloat16)
    score = jax.random.normal(jax.random.fold_in(key, 1), kv.shape, jnp.bfloat16)
    ape = jax.random.normal(jax.random.fold_in(key, 2), (4, 2 * head_dim), jnp.float32)
    cu = jnp.array([0, 16], jnp.int32)
    cu_comp = jnp.array([0, 4], jnp.int32)

    def reference(x, scores, bias):
        a_kv, b_kv = jnp.split(x.reshape(4, 4, 2 * head_dim), 2, axis=-1)
        a_score, b_score = jnp.split(scores.reshape(4, 4, 2 * head_dim) + bias, 2, axis=-1)
        shifted_kv = jnp.concatenate([jnp.zeros_like(a_kv[:1]), a_kv[:-1]], axis=0)
        shifted_score = jnp.concatenate([jnp.full_like(a_score[:1], -jnp.inf), a_score[:-1]], axis=0)
        values = jnp.concatenate([shifted_kv, b_kv], axis=1)
        logits = jnp.concatenate([shifted_score, b_score], axis=1)
        return jnp.sum(values * jax.nn.softmax(logits.astype(jnp.float32), axis=1).astype(values.dtype), axis=1)

    got = jax.jit(lambda x, s, a: csa_compressor(x, s, a, cu, cu_comp, total_comp=4))(kv, score, ape)
    np.testing.assert_allclose(np.asarray(got, np.float32), np.asarray(reference(kv, score, ape), np.float32), atol=8e-3, rtol=8e-3)

    grads = jax.grad(
        lambda x, s, a: csa_compressor(x, s, a, cu, cu_comp, total_comp=4).astype(jnp.float32).sum(),
        argnums=(0, 1, 2),
    )(kv, score, ape)
    reference_grads = jax.grad(
        lambda x, s, a: reference(x, s, a).astype(jnp.float32).sum(), argnums=(0, 1, 2)
    )(kv, score, ape)
    for actual, expected in zip(grads, reference_grads, strict=True):
        np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), atol=8e-3, rtol=8e-3)


@pytest.mark.L0
def test_dsa_indexer_vjp():
    _require_sm100()
    key = jax.random.key(4)
    q = jax.random.normal(key, (1, 16, 64, 128), jnp.bfloat16)
    k = jax.random.normal(jax.random.fold_in(key, 1), (1, 4, 1, 128), jnp.bfloat16)
    weights = jax.random.normal(jax.random.fold_in(key, 2), (1, 16, 64), jnp.bfloat16)
    scale = 128**-0.5

    def reference(q_, k_, weights_):
        dots = jnp.einsum("bshd,bwkd->bhsw", q_.astype(jnp.float32), k_.astype(jnp.float32))
        scores = jnp.einsum("bhsw,bsh->bsw", jax.nn.relu(dots) * scale, weights_.astype(jnp.float32))
        q_pos = jnp.arange(q_.shape[1])[:, None]
        block_ends = 4 * (jnp.arange(k_.shape[1])[None, :] + 1)
        return jnp.where(block_ends <= q_pos + 1, scores, -jnp.inf)

    got = dsa_indexer(q, k, weights, ratio=4, sm_scale=scale)
    expected = reference(q, k, weights)
    valid = jnp.isfinite(got)
    np.testing.assert_allclose(np.asarray(got[valid]), np.asarray(expected[valid]), atol=2e-5, rtol=2e-5)

    loss = lambda q_, k_, w_: jnp.where(jnp.isfinite(dsa_indexer(q_, k_, w_, ratio=4, sm_scale=scale)), dsa_indexer(q_, k_, w_, ratio=4, sm_scale=scale), 0.0).sum()
    reference_loss = lambda q_, k_, w_: jnp.where(jnp.isfinite(reference(q_, k_, w_)), reference(q_, k_, w_), 0.0).sum()
    grads = jax.grad(loss, argnums=(0, 1, 2))(q, k, weights)
    expected_grads = jax.grad(reference_loss, argnums=(0, 1, 2))(q, k, weights)
    for actual, expected in zip(grads, expected_grads, strict=True):
        np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), atol=2e-5, rtol=2e-5)


@pytest.mark.L0
def test_dsa_sparse_attention_vjp():
    _require_sm100()
    key = jax.random.key(5)
    q = jax.random.normal(key, (1, 64, 512), jnp.bfloat16) / jnp.sqrt(512)
    kv = jax.random.normal(jax.random.fold_in(key, 1), (640, 512), jnp.bfloat16)
    indices = jnp.arange(640, dtype=jnp.int32)[None]
    lengths = jnp.array([640], jnp.int32)
    sinks = jnp.zeros((64,), jnp.float32)
    got = dsa_sparse_attention(q, kv, indices, lengths, sinks)

    def reference(x, y, s):
        logits = jnp.einsum("thd,kd->thk", x.astype(jnp.float32), y.astype(jnp.float32))
        maximum = jnp.maximum(jnp.max(logits, axis=-1), s)
        probabilities = jnp.exp(logits - maximum[..., None])
        return jnp.einsum("thk,kd->thd", probabilities, y.astype(jnp.float32)) / (
            jnp.sum(probabilities, axis=-1) + jnp.exp(s - maximum)
        )[..., None]

    np.testing.assert_allclose(np.asarray(got, np.float32), np.asarray(reference(q, kv, sinks)), atol=1e-2, rtol=2e-2)
    grads = jax.grad(
        lambda x, y, s: dsa_sparse_attention(x, y, indices, lengths, s).astype(jnp.float32).sum(),
        argnums=(0, 1, 2),
    )(q, kv, sinks)
    expected_grads = jax.grad(
        lambda x, y, s: reference(x, y, s).sum(), argnums=(0, 1, 2)
    )(q, kv, sinks)
    for actual, expected in zip(grads, expected_grads, strict=True):
        np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), atol=4e-2, rtol=2e-2)
