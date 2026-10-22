# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Per-call dense mesh roles survive VJP tracing and custom partitioning."""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from transformer_engine.jax.cpp_extensions.gemm import GemmPrimitive, CollectiveOp
from transformer_engine.jax.dense import dense
from transformer_engine.jax.layernorm_dense import layernorm_dense
from transformer_engine.jax.layernorm_mlp import layernorm_mlp
from transformer_engine.jax.cpp_extensions.amax import calculate_amax, AmaxScope
from transformer_engine.jax import cpp_extensions as tex
from transformer_engine.common.recipe import Float8CurrentScaling
from transformer_engine.jax.quantize import QuantizerFactory, QuantizeMetaSet, QuantizeMeta
from transformer_engine.jax.quantize import ScalingMode
from transformer_engine.jax.sharding import (
    MeshResource,
    _mesh_resource_snapshot,
    global_mesh_resource,
    global_shard_guard,
)


def test_mesh_override_is_an_immutable_snapshot():
    resource = MeshResource(fsdp_resource=("fsdp", "expert"))
    snapshot = _mesh_resource_snapshot(resource)
    resource.fsdp_resource = "changed"
    assert snapshot.fsdp_resource == ("fsdp", "expert")
    assert hash(snapshot) == hash(
        _mesh_resource_snapshot(MeshResource(fsdp_resource=("fsdp", "expert")))
    )
    with pytest.raises(TypeError, match="MeshResource or None"):
        _mesh_resource_snapshot("expert")


@pytest.mark.parametrize("fsdp_axes", [("fsdp", "expert"), ("expert", "fsdp")])
def test_partition_gathers_weights_instead_of_activations(fsdp_axes):
    def info(spec, ndim, size=1):
        return SimpleNamespace(sharding=SimpleNamespace(spec=P(*spec)), ndim=ndim, size=size)

    args = [
        info((fsdp_axes, None), 2),
        info((None,), 1),
        info((None, "expert"), 2),
        info((None,), 1),
        info((None,), 1, size=0),
    ]
    override = _mesh_resource_snapshot(MeshResource(fsdp_resource=fsdp_axes))
    global_resource = MeshResource(fsdp_resource="fsdp", ep_resource="expert")
    with global_shard_guard(global_resource):
        specs, output, *_ = GemmPrimitive._parse_operand_output_specs(
            args, ((1,), (0,)), False, CollectiveOp.NONE, ScalingMode.NO_SCALING, override
        )
        assert global_mesh_resource() is global_resource
    assert specs[0] == (fsdp_axes, None)
    assert specs[2] == (None, None)
    assert output == (fsdp_axes, None)


@pytest.mark.parametrize("ep,fsdp", [(4, 1), (2, 2)])
def test_dense_override_forward_and_vjp(ep, fsdp):
    if jax.device_count() < 4:
        pytest.skip("Requires four GPUs")
    mesh = Mesh(np.array(jax.devices()[:4]).reshape(fsdp, ep), ("fsdp", "expert"))
    batch_axes = ("fsdp", "expert")
    x_sharding = NamedSharding(mesh, P(batch_axes, None))
    w_sharding = NamedSharding(mesh, P(None, batch_axes))
    resource = MeshResource(fsdp_resource=batch_axes)
    x = (jnp.arange(128 * 128).reshape(128, 128) % 7 / 8).astype(jnp.float32)
    weight = (jnp.arange(128 * 128).reshape(128, 128) % 11 / 16).astype(jnp.float32)
    x, weight = jax.device_put(x, x_sharding), jax.device_put(weight, w_sharding)

    def objective(x, weight):
        y = dense(x, weight, mesh_resource=resource)
        return jnp.mean(y * y), y

    def reference(x, weight):
        y = x @ weight
        return jnp.mean(y * y), y

    # Trace and compile after leaving the surrounding context. Partitioning must
    # use the explicit override rather than whatever global roles exist then.
    with jax.set_mesh(mesh), global_shard_guard(MeshResource(fsdp_resource="fsdp")):
        lowered = jax.jit(jax.value_and_grad(objective, (0, 1), has_aux=True)).lower(x, weight)
    with jax.set_mesh(mesh), global_shard_guard(MeshResource(fsdp_resource="wrong")):
        actual = lowered.compile()(x, weight)
    expected = jax.jit(jax.value_and_grad(reference, (0, 1), has_aux=True))(x, weight)
    for result, ref in zip(jax.tree.leaves(actual), jax.tree.leaves(expected)):
        np.testing.assert_allclose(result, ref, rtol=2e-5, atol=2e-5)


@pytest.mark.parametrize("operation", ["layernorm_dense", "layernorm_mlp"])
def test_fused_dense_override_parameter_gradients(operation):
    if jax.device_count() < 4:
        pytest.skip("Requires four GPUs")
    mesh = Mesh(np.array(jax.devices()[:4]).reshape(2, 2), ("fsdp", "expert"))
    axes = ("fsdp", "expert")
    resource = MeshResource(fsdp_resource=axes)
    x = jax.random.normal(jax.random.key(0), (128, 128))
    w1 = jax.random.normal(jax.random.key(1), (128, 1, 128)) / 16
    w2 = jax.random.normal(jax.random.key(2), (128, 128)) / 16
    gamma, bias = jnp.ones(128), jnp.zeros(128)
    x = jax.device_put(x, NamedSharding(mesh, P(axes, None)))
    w1 = jax.device_put(w1, NamedSharding(mesh, P(None, None, axes)))
    w2 = jax.device_put(w2, NamedSharding(mesh, P(None, axes)))

    def objective(x, w1, w2, gamma, bias):
        if operation == "layernorm_dense":
            y = layernorm_dense(
                x, w2, gamma, None, bias, norm_type="rmsnorm", mesh_resource=resource
            )
        else:
            y = layernorm_mlp(
                x,
                gamma,
                None,
                [w1, w2],
                [bias[None, :], bias],
                norm_type="rmsnorm",
                activation_type=("relu",),
                mesh_resource=resource,
            )
        return jnp.mean(y * y)

    def reference(x, w1, w2, gamma, bias):
        normalized = x * jax.lax.rsqrt(jnp.mean(x * x, axis=-1, keepdims=True) + 1e-6) * gamma
        if operation == "layernorm_mlp":
            normalized = jax.nn.relu(normalized @ w1[:, 0, :] + bias)
        y = normalized @ w2 + bias
        return jnp.mean(y * y)

    args = x, w1, w2, gamma, bias
    with jax.set_mesh(mesh), global_shard_guard(MeshResource(fsdp_resource="wrong")):
        actual = jax.jit(jax.value_and_grad(objective, tuple(range(5))))(*args)
    expected = jax.jit(jax.value_and_grad(reference, tuple(range(5))))(*args)
    for result, ref in zip(jax.tree.leaves(actual), jax.tree.leaves(expected)):
        np.testing.assert_allclose(result, ref, rtol=2e-4, atol=2e-5)


def test_quantization_override_amax_and_bias_reductions():
    if jax.device_count() < 4:
        pytest.skip("Requires four GPUs")
    mesh = Mesh(np.array(jax.devices()[:4]).reshape(2, 2), ("fsdp", "expert"))
    resource = MeshResource(fsdp_resource=("fsdp", "expert"))
    x = jnp.arange(128 * 128, dtype=jnp.float32).reshape(128, 128) / 1024
    x = jax.device_put(x, NamedSharding(mesh, P(("fsdp", "expert"), None)))
    quantizers = QuantizerFactory.create_set(
        fp8_recipe=Float8CurrentScaling(),
        quantize_meta_set=QuantizeMetaSet(
            x=QuantizeMeta(), kernel=QuantizeMeta(), grad=QuantizeMeta()
        ),
    )

    def operation(x):
        amax = calculate_amax(x, AmaxScope.FSDP, False, mesh_resource=resource)
        _, dbias = tex.quantize_dbias(x, quantizers.dgrad, mesh_resource=resource)
        return amax, dbias

    with jax.set_mesh(mesh), global_shard_guard(MeshResource(fsdp_resource="wrong")):
        amax, dbias = jax.jit(operation)(x)
    np.testing.assert_allclose(amax, jnp.max(jnp.abs(x)).reshape(1), rtol=0, atol=0)
    np.testing.assert_allclose(dbias, jnp.sum(x, axis=0), rtol=0, atol=0)
