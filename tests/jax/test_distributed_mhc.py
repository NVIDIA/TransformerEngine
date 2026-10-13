# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Tests for sharded execution of the mHC (manifold Hyper-Connection) primitives.

The mHC kernels are per token, except for the parameter gradients, which are sums over tokens.
With the token dims split across devices, the sharded forward output and gradients must
therefore match an unsharded run, provided the parameter gradients are summed across devices.
"""

import os
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding, PartitionSpec

from distributed_test_base import generate_configs
from utils import assert_allclose, pytest_parametrize_wrapper


@pytest.fixture(autouse=True, scope="function")
def _inject_mhc(request, monkeypatch):
    """Lazy-load the mHC API only for tests marked 'triton'."""
    if not request.node.get_closest_marker("triton"):
        yield
        return
    # Triton autotuning benchmarks on each device's stream at the first run of the multi-device
    # executable, which can deadlock with the cross-device all-reduces in a single process.
    monkeypatch.setenv("NVTE_DISABLE_TRITON_AUTOTUNING", "1")
    from transformer_engine.jax import mhc as te_mhc

    mod = sys.modules[__name__]
    for name in te_mhc.__all__:
        setattr(mod, name, getattr(te_mhc, name))
    yield


ENFORCE_DETERMINISTIC = os.environ.get("NVTE_ALLOW_NONDETERMINISTIC_ALGO", "1") == "0"
N_STREAMS = 4
N_MIX = 2 * N_STREAMS + N_STREAMS * N_STREAMS
# fp32 tolerances of tests/jax/test_mhc.py. The sharded run only changes the order of the
# token reductions, while a missing cross-device sum would be off by about the device count.
TOLS = {"atol": 5e-3, "rtol": 5e-3}
# (b, s, C), with b and s divisible by the dp * fsdp and tpsp mesh sizes.
SHAPES = [(4, 64, 256), (8, 128, 1024)]
PARAM_NAMES = ("phi", "alpha", "beta", "norm_weight", "bias")


def _normal(key, shape, scale=1.0):
    return scale * jax.random.normal(key, shape, jnp.float32)


def _branch_fn(y):
    """A per-token pre-norm and nonlinearity."""
    y = y * jax.lax.rsqrt(jnp.mean(y * y, axis=-1, keepdims=True) + 1e-6)
    return jnp.tanh(2 * y)


def _make_inputs(b, s, C, with_bias):
    keys = jax.random.split(jax.random.PRNGKey(0), 7)
    nC = N_STREAMS * C
    x = _normal(keys[0], (b, s, C, N_STREAMS))
    params = (
        _normal(keys[1], (N_MIX, nC), nC**-0.5),  # phi
        1.0 + _normal(keys[2], (3,), 0.1),  # alpha
        _normal(keys[3], (N_MIX,), 0.5),  # beta
        1.0 + _normal(keys[4], (nC,), 0.1),  # norm_weight
    )
    if with_bias:
        params += (_normal(keys[5], (C,), 0.5),)
    cotangent = _normal(keys[6], x.shape)
    return x, params, cotangent


def _make_loss(use_split_k):
    def forward(x, params):
        phi, alpha, beta, norm_weight, *bias = params
        kwargs = {"use_tf32": False, "use_split_k": use_split_k, "pre_eps": 1e-6}
        if not bias:
            return mhc(x, phi, alpha, beta, _branch_fn, norm_weight, **kwargs)
        # mhc() has no branch bias, so chain the stages to cover the fused bias of expand_combine.
        layer_input, h_post, h_res = mhc_generate_mix_and_aggregate(
            x, phi, alpha, beta, norm_weight, **kwargs
        )
        return mhc_expand_combine(_branch_fn(layer_input), bias[0], h_post, x, h_res, False)

    def loss(x, params, cotangent):
        out = forward(x, params)
        return jnp.sum(out * cotangent), out

    return jax.jit(jax.value_and_grad(loss, argnums=(0, 1), has_aux=True))


@pytest.mark.triton
class TestDistributedMHC:
    """Sharded vs unsharded mHC forward and gradients, with the token dims split across devices."""

    @pytest.mark.parametrize("device_count,mesh_shape,mesh_axes,mesh_resource", generate_configs())
    @pytest_parametrize_wrapper("b,s,C", SHAPES)
    @pytest_parametrize_wrapper("with_bias", [False, True])
    @pytest_parametrize_wrapper("use_split_k", [False, True])
    def test_token_sharded(
        self, device_count, mesh_shape, mesh_axes, mesh_resource, b, s, C, with_bias, use_split_k
    ):
        if ENFORCE_DETERMINISTIC and use_split_k:
            pytest.skip("Split-K is not deterministic, skip the test under deterministic mode")

        x, params, cotangent = _make_inputs(b, s, C, with_bias)
        loss_and_grads = _make_loss(use_split_k)
        (_, ref_out), (ref_dx, ref_dparams) = loss_and_grads(x, params, cotangent)

        devices = np.asarray(jax.devices()[:device_count]).reshape(*mesh_shape)
        mesh = Mesh(devices, mesh_axes)
        # Split the batch over dp/fsdp and the sequence over tpsp, so the cross-device sums span
        # several token dims and mesh axes. The hidden and stream dims stay replicated.
        batch_axes = tuple(a for a in (mesh_resource.dp_resource, mesh_resource.fsdp_resource) if a)
        token_pspec = PartitionSpec(batch_axes or None, mesh_resource.tpsp_resource, None, None)

        with mesh:
            token_sharding = NamedSharding(mesh, token_pspec)
            replicated = NamedSharding(mesh, PartitionSpec())
            x_sharded = jax.device_put(x, token_sharding)
            cotangent_sharded = jax.device_put(cotangent, token_sharding)
            params_sharded = jax.device_put(params, replicated)
            (_, out), (dx, dparams) = loss_and_grads(x_sharded, params_sharded, cotangent_sharded)

        assert out.sharding.is_equivalent_to(token_sharding, out.ndim)
        assert dx.sharding.is_equivalent_to(token_sharding, dx.ndim)
        assert_allclose(out, ref_out, err_msg="out", **TOLS)
        assert_allclose(dx, ref_dx, err_msg="x", **TOLS)
        for name, grad, ref_grad in zip(PARAM_NAMES, dparams, ref_dparams):
            assert grad.sharding.is_fully_replicated, name
            assert_allclose(grad, ref_grad, err_msg=name, **TOLS)
