# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Explicit TE cleanup while the XLA runtime and model arrays remain alive."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import transformer_engine.jax as te
import transformer_engine_jax as tex
from transformer_engine.jax.layernorm import layernorm
from transformer_engine.jax import cpp_extensions as extensions
from transformer_engine.jax.quantize.tensor import GroupedNoScaleTensor


def test_context_loss_rejects_framework_without_mutation():
    generation = tex.get_cuda_state_version()
    with pytest.raises(RuntimeError, match="JAX/XLA CUDA runtime"):
        te.reset_cuda_state()
    assert tex.get_cuda_state_version() == generation


def test_live_cleanup_preserves_arrays_and_recreates_normalization():
    device = jax.devices("gpu")[0]
    x = jax.device_put(jnp.arange(32 * 128, dtype=jnp.float32).reshape(32, 128) / 1000, device)
    gamma = jax.device_put(jnp.linspace(0.8, 1.2, 128), device)
    beta = jax.device_put(jnp.linspace(-0.1, 0.1, 128), device)
    epsilon = 1e-5

    @jax.jit
    def compute(value):
        return layernorm(value, gamma, beta, "layernorm", False, epsilon)

    mean = jnp.mean(x, axis=-1, keepdims=True)
    reference = (x - mean) * jax.lax.rsqrt(
        jnp.mean((x - mean) ** 2, axis=-1, keepdims=True) + epsilon
    ) * gamma + beta
    generation = tex.get_cuda_state_version()
    for _ in range(3):
        np.testing.assert_allclose(compute(x), reference, rtol=1e-4, atol=1e-4)
        te.release_cuda_state()
        assert tex.get_cuda_state_version() > generation
        generation = tex.get_cuda_state_version()
    np.testing.assert_allclose(compute(x), reference, rtol=1e-4, atol=1e-4)


def test_live_cleanup_rebuilds_grouped_gemm_pinned_transfers():
    sizes = jnp.array([32, 64], dtype=jnp.int32)
    lhs = jnp.arange(96 * 64, dtype=jnp.float16).reshape(96, 64) / 8192
    rhs = jnp.ones((2, 64, 32), dtype=jnp.float16) / 64
    left = GroupedNoScaleTensor(
        data=lhs, amax=None, first_dims=sizes, last_dims=None, original_shape=lhs.shape
    )
    right = GroupedNoScaleTensor(
        data=rhs, amax=None, first_dims=None, last_dims=None, original_shape=rhs.shape
    )
    copy = jax.jit(extensions.grouped_gemm_copy_group_sizes, static_argnames=("num_gemms",))
    compute = jax.jit(
        extensions.grouped_gemm, static_argnames=("contracting_dims", "use_async_d2h_group_sizes")
    )
    reference = jnp.concatenate((lhs[:32] @ rhs[0], lhs[32:] @ rhs[1]))
    for _ in range(2):
        copy(sizes, num_gemms=2).block_until_ready()
        output = compute(left, right, contracting_dims=((1,), (1,)), use_async_d2h_group_sizes=True)
        np.testing.assert_allclose(output, reference, rtol=2e-3, atol=2e-3)
        te.release_cuda_state()


@pytest.mark.skipif(len(jax.local_devices(backend="gpu")) < 2, reason="Two CUDA devices required")
def test_global_release_preserves_arrays_and_rebuilds_plans_on_two_devices():
    compute = jax.jit(lambda x, gamma, beta: layernorm(x, gamma, beta, "layernorm", False, 1e-5))
    states = []
    for device in jax.local_devices(backend="gpu")[:2]:
        with jax.default_device(device):
            x = jax.device_put(
                jnp.arange(8 * 128, dtype=jnp.float32).reshape(8, 128) / 1000, device
            )
            gamma = jax.device_put(jnp.ones(128), device)
            beta = jax.device_put(jnp.zeros(128), device)
            values = np.asarray(x)
            centered = values - values.mean(axis=-1, keepdims=True)
            reference = centered / np.sqrt((centered**2).mean(axis=-1, keepdims=True) + 1e-5)
            states.append((x, gamma, beta, reference))
    for _ in range(3):
        for x, gamma, beta, reference in states:
            np.testing.assert_allclose(compute(x, gamma, beta), reference, rtol=1e-4, atol=1e-4)
        version = tex.get_cuda_state_version()
        te.release_cuda_state()
        assert tex.get_cuda_state_version() == version + 1
