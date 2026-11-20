# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Per-call dense mesh roles survive VJP tracing and custom partitioning."""

import math
import re
from functools import partial
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from flax.linen import logical_axis_rules
from utils import assert_allclose
from test_custom_call_compute import _ref_jax_norm_impl

from transformer_engine.jax.cpp_extensions.gemm import GemmPrimitive, CollectiveOp
from transformer_engine.jax.dense import dense
from transformer_engine.jax.layernorm_dense import layernorm_dense
from transformer_engine.jax.layernorm_mlp import layernorm_mlp
from transformer_engine.jax.cpp_extensions.amax import calculate_amax, AmaxScope
from transformer_engine.jax import cpp_extensions as tex
from transformer_engine.common.recipe import Float8CurrentScaling
from transformer_engine.jax.quantize import QuantizerFactory, QuantizeMetaSet, QuantizeMeta
from transformer_engine.jax.quantize import (
    ScalingMode,
    QuantizeLayout,
    TensorUsage,
    is_scaling_mode_supported,
)
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


@pytest.mark.parametrize("batch_axes", ["fsdp", ("fsdp", "expert")])
@pytest.mark.parametrize(
    "weight_axes,kept_axes",
    [("fsdp", None), (("expert", "fsdp"), None), (("fsdp", "expert", "tp"), "tp")],
)
def test_tuple_global_fsdp_matches_override(batch_axes, weight_axes, kept_axes):
    def info(spec, ndim, size=1):
        return SimpleNamespace(sharding=SimpleNamespace(spec=P(*spec)), ndim=ndim, size=size)

    args = [
        info((batch_axes, None), 2),
        info((None,), 1),
        info((None, weight_axes), 2),
        info((None,), 1),
        info((None,), 1, size=0),
    ]
    resource = MeshResource(fsdp_resource=("fsdp", "expert"))

    def parse(override):
        return GemmPrimitive._parse_operand_output_specs(
            args, ((1,), (0,)), False, CollectiveOp.NONE, ScalingMode.NO_SCALING, override
        )[:2]

    with global_shard_guard(resource):
        specs, output = parse(None)
    assert specs[0] == (batch_axes, None)
    assert specs[2] == (None, kept_axes)
    assert output == (batch_axes, kept_axes)
    with global_shard_guard(MeshResource()):
        assert parse(_mesh_resource_snapshot(resource)) == (specs, output)


@pytest.mark.parametrize("batch_axes", [("fsdp", "expert"), ("expert", "fsdp")])
def test_partition_gathers_only_activation_axes_kept_on_weight(batch_axes):
    def info(spec, ndim, size=1):
        return SimpleNamespace(sharding=SimpleNamespace(spec=P(*spec)), ndim=ndim, size=size)

    args = [
        info((batch_axes, None), 2),
        info((None,), 1),
        info((None, batch_axes), 2),
        info((None,), 1),
        info((None,), 1, size=0),
    ]

    def parse(override):
        return GemmPrimitive._parse_operand_output_specs(
            args, ((1,), (0,)), False, CollectiveOp.NONE, ScalingMode.NO_SCALING, override
        )[:2]

    resource = MeshResource(fsdp_resource="fsdp", ep_resource="expert")
    with global_shard_guard(MeshResource()):
        specs, output = parse(_mesh_resource_snapshot(resource))
    # The weight keeps "expert", so only "expert" is gathered from the activation.
    assert specs[0] == ("fsdp", None)
    assert specs[2] == (None, "expert")
    assert output == ("fsdp", "expert")
    with global_shard_guard(MeshResource(fsdp_resource=("fsdp",), ep_resource="expert")):
        assert parse(None) == (specs, output)


def _value_and_grads(forward, *args):
    """Mean square of forward's output, the output, and the gradient for every argument."""

    def loss(*args):
        y = forward(*args)
        return jnp.mean(y * y), y

    return jax.value_and_grad(loss, tuple(range(len(args))), has_aux=True)(*args)


def _assert_matches_single_device_and_reference(actual, forward, reference, *args):
    """A sharded run must match forward on one device in FP32, and forward must match an XLA
    reference with BF16 inputs, as in test_custom_call_compute."""
    args = tuple(np.asarray(a) for a in args)
    single = jax.jit(partial(_value_and_grads, forward))
    for result, ref in zip(jax.tree.leaves(actual), jax.tree.leaves(single(*args))):
        assert_allclose(result, ref, dtype=jnp.float32)
    args = tuple(jnp.asarray(a, jnp.bfloat16) for a in args)
    expected = jax.jit(partial(_value_and_grads, reference))(*args)
    for result, ref in zip(jax.tree.leaves(single(*args)), jax.tree.leaves(expected)):
        assert_allclose(result, ref, dtype=jnp.bfloat16)


@pytest.mark.parametrize(
    "resource",
    [
        MeshResource(fsdp_resource=("fsdp", "expert")),
        MeshResource(fsdp_resource="fsdp", ep_resource="expert"),
    ],
    ids=["fsdp_expert", "fsdp_only"],
)
@pytest.mark.parametrize("ep,fsdp", [(4, 1), (2, 2)])
def test_dense_override_forward_and_vjp(ep, fsdp, resource):
    if jax.device_count() < 4:
        pytest.skip("Requires four GPUs")
    mesh = Mesh(np.array(jax.devices()[:4]).reshape(fsdp, ep), ("fsdp", "expert"))
    batch_axes = ("fsdp", "expert")
    x_sharding = NamedSharding(mesh, P(batch_axes, None))
    w_sharding = NamedSharding(mesh, P(None, batch_axes))
    x = (jnp.arange(128 * 128).reshape(128, 128) % 7 / 8).astype(jnp.float32)
    weight = (jnp.arange(128 * 128).reshape(128, 128) % 11 / 16).astype(jnp.float32)
    x, weight = jax.device_put(x, x_sharding), jax.device_put(weight, w_sharding)

    def forward(x, weight, mesh_resource=None):
        return dense(x, weight, mesh_resource=mesh_resource)

    def reference(x, weight):
        return jnp.dot(x, weight)

    # Trace and compile after leaving the surrounding context. Partitioning must
    # use the explicit override rather than whatever global roles exist then.
    fn = jax.jit(partial(_value_and_grads, partial(forward, mesh_resource=resource)))
    with jax.set_mesh(mesh), global_shard_guard(MeshResource(fsdp_resource="fsdp")):
        lowered = fn.lower(x, weight)
    with jax.set_mesh(mesh), global_shard_guard(MeshResource(fsdp_resource="wrong")):
        actual = lowered.compile()(x, weight)
    _assert_matches_single_device_and_reference(actual, forward, reference, x, weight)


def _all_gather_sizes(hlo):
    """Element counts of the all-gather results in compiled HLO text."""
    return [
        math.prod(int(dim) for dim in dims.split(",") if dim)
        for result in re.findall(r"= ([^=\n]*?) all-gather(?:-done)?\(", hlo)
        for dims in re.findall(r"\[([\d,]*)\]", result)
    ]


@pytest.mark.parametrize("operation", ["dense", "layernorm_dense"])
@pytest.mark.parametrize("weight_axes", ["expert", ("expert", "fsdp")])
def test_tuple_global_fsdp_forward_and_vjp(operation, weight_axes):
    """A tuple global fsdp_resource with no override gathers the weight, not the activation."""
    if jax.device_count() < 4:
        pytest.skip("Requires four GPUs")
    mesh = Mesh(np.array(jax.devices()[:4]).reshape(2, 2), ("fsdp", "expert"))
    batch_axes = ("fsdp", "expert")
    x = jax.random.normal(jax.random.key(0), (64, 128))
    weight = jax.random.normal(jax.random.key(1), (128, 256)) / 16
    gamma = jnp.ones(128)

    def forward(x, weight, gamma):
        if operation == "dense":
            return dense(x, weight)
        return layernorm_dense(x, weight, gamma, None, norm_type="rmsnorm")

    def sharded(x, weight, gamma):
        # Keep the activation batch-sharded under both partitioners.
        y = forward(x, weight, gamma)
        return jax.lax.with_sharding_constraint(y, NamedSharding(mesh, P(batch_axes, None)))

    def reference(x, weight, gamma):
        if operation == "layernorm_dense":
            x = _ref_jax_norm_impl(x, gamma, None, "rmsnorm", False, 1e-6, None)
        return jnp.dot(x, weight)

    x = jax.device_put(x, NamedSharding(mesh, P(batch_axes, None)))
    weight = jax.device_put(weight, NamedSharding(mesh, P(None, weight_axes)))
    # Partitioning reads the global resource, so it stays set through compilation.
    with jax.set_mesh(mesh), global_shard_guard(MeshResource(fsdp_resource=batch_axes)):
        fn = jax.jit(partial(_value_and_grads, sharded))
        compiled = fn.lower(x, weight, gamma).compile()
        actual = compiled(x, weight, gamma)
    _assert_matches_single_device_and_reference(actual, forward, reference, x, weight, gamma)
    # The activation keeps its batch sharding; only the whole weight is gathered.
    assert set(_all_gather_sizes(compiled.as_text())) == {weight.size}


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

    def forward(x, w1, w2, gamma, bias, mesh_resource=None):
        if operation == "layernorm_dense":
            return layernorm_dense(
                x, w2, gamma, None, bias, norm_type="rmsnorm", mesh_resource=mesh_resource
            )
        return layernorm_mlp(
            x,
            gamma,
            None,
            [w1, w2],
            [bias[None, :], bias],
            norm_type="rmsnorm",
            activation_type=("relu",),
            mesh_resource=mesh_resource,
        )

    def reference(x, w1, w2, gamma, bias):
        normalized = _ref_jax_norm_impl(x, gamma, None, "rmsnorm", False, 1e-6, None)
        if operation == "layernorm_mlp":
            normalized = jax.nn.relu(jnp.dot(normalized, w1[:, 0, :]) + bias)
        return jnp.dot(normalized, w2) + bias

    args = x, w1, w2, gamma, bias
    fn = jax.jit(partial(_value_and_grads, partial(forward, mesh_resource=resource)))
    with jax.set_mesh(mesh), global_shard_guard(MeshResource(fsdp_resource="wrong")):
        actual = fn(*args)
    _assert_matches_single_device_and_reference(actual, forward, reference, *args)


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


@pytest.mark.parametrize(
    "scaling_mode", [ScalingMode.CURRENT_TENSOR_SCALING, ScalingMode.DELAYED_TENSOR_SCALING]
)
@pytest.mark.parametrize("backward", [False, True])
def test_activation_helper_override_reductions(scaling_mode, backward):
    """Exercise the actual MLP activation helpers, including the 2x recursion."""
    supported, reason = is_scaling_mode_supported(scaling_mode)
    if not supported:
        pytest.skip(reason)
    if jax.device_count() < 4:
        pytest.skip("Requires four GPUs")
    mesh = Mesh(np.array(jax.devices()[:4]).reshape(2, 2), ("fsdp", "expert"))
    current = scaling_mode == ScalingMode.CURRENT_TENSOR_SCALING
    if current:
        resource = MeshResource(tpsp_resource="expert")
        surrounding = MeshResource(tpsp_resource="fsdp")
        spec = P(None, "expert", None, None)
    else:
        resource = MeshResource(fsdp_resource=("fsdp", "expert"))
        surrounding = MeshResource(fsdp_resource="fsdp", pp_resource="expert")
        spec = P(("fsdp", "expert"), None, None, None)
    # Different magnitudes on each shard make missing collectives observable.
    x = jnp.arange(4 * 128 * 2 * 128, dtype=jnp.float32).reshape(4, 128, 2, 128) / 16384 + 1
    dz = jnp.ones((4, 128, 128), jnp.float32)
    x = jax.device_put(x, NamedSharding(mesh, spec))
    dz = jax.device_put(dz, NamedSharding(mesh, P(*spec[:-2], spec[-1])))

    def operation(x, dz):
        quantizer = QuantizerFactory.create(
            scaling_mode=scaling_mode,
            q_dtype=jnp.float8_e4m3fn,
            q_layout=QuantizeLayout.ROWWISE_COLWISE,
        )
        if backward:
            out, dbias = tex.quantize_dact_dbias(
                dz,
                x,
                ("relu", "linear"),
                is_dbias=not current,
                quantizer=quantizer,
                amax_scope=AmaxScope.TPSP,
                mesh_resource=resource,
            )
        else:
            out = tex.act_lu(
                x,
                ("relu", "linear"),
                quantizer=quantizer,
                amax_scope=AmaxScope.TPSP,
                mesh_resource=resource,
            )
            dbias = None
        # Current scale is visible on the tensor; delayed scale is updated for
        # the next iteration. Return it explicitly rather than leaking tracers.
        scale_inv = out.get_tensor(TensorUsage.LHS).scale_inv if current else 1 / quantizer.scale
        return scale_inv, dbias

    with jax.set_mesh(mesh), global_shard_guard(surrounding):
        scale_inv, dbias = jax.jit(operation)(x, dz)
    if backward:
        derivative = jnp.stack((x[..., 1, :], x[..., 0, :]), axis=-2)
        expected_amax = jnp.max(derivative)
        if not current:
            np.testing.assert_allclose(dbias, jnp.sum(derivative, axis=(0, 1)), rtol=1e-6)
    else:
        expected_amax = jnp.max(x[..., 0, :] * x[..., 1, :])
    np.testing.assert_allclose(scale_inv, (expected_amax / 448).reshape(1), rtol=1e-6)


@pytest.mark.parametrize(
    "scaling_mode", [ScalingMode.CURRENT_TENSOR_SCALING, ScalingMode.DELAYED_TENSOR_SCALING]
)
def test_quantized_mlp_override_matches_global_context(scaling_mode):
    """An explicit resource must match the legacy omitted-override VJP."""
    supported, reason = is_scaling_mode_supported(scaling_mode)
    if not supported:
        pytest.skip(reason)
    if jax.device_count() < 4:
        pytest.skip("Requires four GPUs")
    mesh = Mesh(np.array(jax.devices()[:4]).reshape(2, 2), ("fsdp", "expert"))
    current = scaling_mode == ScalingMode.CURRENT_TENSOR_SCALING
    if current:
        resource = MeshResource(tpsp_resource="expert")
        surrounding = MeshResource(tpsp_resource="fsdp")
        batch_axis, seq_axis = None, "expert"
    else:
        resource = MeshResource(fsdp_resource=("fsdp", "expert"))
        surrounding = MeshResource(fsdp_resource="fsdp", ep_resource="expert")
        batch_axis, seq_axis = ("fsdp", "expert"), None
    x = jax.random.normal(jax.random.key(10), (4, 128, 128))
    w1 = jax.random.normal(jax.random.key(11), (128, 2, 128)) / 16
    w2 = jax.random.normal(jax.random.key(12), (128, 128)) / 16
    args = tuple(
        a.astype(jnp.bfloat16)
        for a in (x, w1, w2, jnp.ones(128), jnp.zeros((2, 128)), jnp.zeros(128))
    )
    specs = (P(batch_axis, seq_axis, None), P(), P(), P(), P(), P())
    args = tuple(jax.device_put(a, NamedSharding(mesh, s)) for a, s in zip(args, specs))

    def objective(override, x, w1, w2, gamma, b1, b2):
        quantizers = QuantizerFactory.create_set(
            n_quantizer_sets=2,
            scaling_mode=scaling_mode,
            fwd_dtype=jnp.float8_e4m3fn,
            bwd_dtype=jnp.float8_e5m2,
            is_2x2x=True,
        )
        y = layernorm_mlp(
            x,
            gamma,
            None,
            [w1, w2],
            [b1, b2],
            norm_type="rmsnorm",
            activation_type=("relu", "linear"),
            quantizer_sets=quantizers,
            norm_input_axes=("batch", "seq", "hidden"),
            dot_1_input_axes=("batch", "seq", "hidden"),
            dot_2_input_axes=("batch", "seq", "hidden"),
            mesh_resource=override,
        )
        # A nonuniform cotangent exposes shard-local backward scaling.
        return jnp.mean(y * jnp.linspace(0.25, 2, 128)[None, :, None])

    def run(override, context):
        fn = jax.value_and_grad(lambda *a: objective(override, *a), tuple(range(6)))
        with jax.set_mesh(mesh), logical_axis_rules(
            (("batch", batch_axis), ("seq", seq_axis), ("hidden", None))
        ), global_shard_guard(context):
            return jax.jit(fn)(*args)

    expected = run(None, resource)
    actual = run(resource, surrounding)
    for result, ref in zip(jax.tree.leaves(actual), jax.tree.leaves(expected)):
        np.testing.assert_allclose(result, ref, rtol=1e-6, atol=1e-6)
