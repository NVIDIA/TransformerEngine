# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""MoE resource resolution and MeshResource-only API without EP kernels."""

import importlib
import inspect

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding, PartitionSpec

from utils import dtype_tols
from transformer_engine.jax.cpp_extensions.misc import jax_dtype_to_te_dtype
from transformer_engine.jax.ep import (
    EpLayerConfig,
    ep_bootstrap,
    ep_combine,
    ep_dispatch,
    ep_finalize,
)
from transformer_engine.jax.flax import _MoEBlock
from transformer_engine.jax.moe import (
    RouterComputationInfo,
    RoutingMapInfo,
    _moe_mesh_axes,
    _resolve_moe_mesh_resource,
    get_moe_recv_capacity_per_rank,
    moe,
    record_ep_bootstrap_signature_for_moe,
)
from transformer_engine.jax.sharding import MeshResource, global_mesh_resource, global_shard_guard


@pytest.fixture(autouse=True)
def no_global_resource():
    with global_shard_guard(None):
        yield


def test_resource_required_for_function_and_block():
    module = importlib.import_module("transformer_engine.jax.moe")
    with pytest.raises(ValueError, match="active global_shard_guard"):
        module.moe(None, None, None, None, num_experts=2)
    with pytest.raises(ValueError, match="active global_shard_guard"):
        _MoEBlock().init(jax.random.PRNGKey(0), jnp.ones((1, 1, 4)))


def test_explicit_resource_overrides_global_and_is_snapshotted():
    explicit = MeshResource(dp_resource="dp", fsdp_resource="fsdp", ep_resource="ep")
    enclosing = MeshResource(ep_resource="other")
    with global_shard_guard(enclosing):
        resolved, quantize = _resolve_moe_mesh_resource(explicit, True)
        assert global_mesh_resource() is enclosing
    explicit.ep_resource = "changed"
    assert _moe_mesh_axes(resolved) == ("ep", ("dp", "fsdp"))
    assert quantize is True


def test_global_default_and_duplicate_outer_axis():
    with global_shard_guard(
        MeshResource(dp_resource="fsdp", fsdp_resource="fsdp", ep_resource="ep")
    ):
        resource, quantize = _resolve_moe_mesh_resource()
    assert _moe_mesh_axes(resource) == ("ep", ("fsdp",))
    assert quantize is False


@pytest.mark.parametrize("fsdp_resource", ["fsdp", ("fsdp",), ("fsdp", "ep"), ("ep", "fsdp")])
def test_ep_partitioning_retains_bind_time_resources(fsdp_resource):
    from transformer_engine.jax.cpp_extensions import ep
    from jax.sharding import PartitionSpec as P

    resource = MeshResource(dp_resource="dp", fsdp_resource=fsdp_resource, ep_resource="ep")
    with global_shard_guard(resource):
        captured = ep._capture_ep_resource_axes()
    spec = P(("dp", "fsdp", "ep"), None, None)
    assert ep._leading_axis_ok(spec, captured) == (True, "ep", ("dp", "fsdp"))
    assert ep._ep_output_spec(None, None, resource_axes=captured) == spec
    assert ep._ep_spec_ok(spec, 2, resource_axes=captured)
    assert not ep._ep_spec_ok(P(("wrong", "ep"), None, None), 2, resource_axes=captured)


@pytest.mark.parametrize("ep_size,expected", [(1, True), (2, False)])
def test_ep_leading_axis_canonicalization(ep_size, expected):
    from types import SimpleNamespace
    from jax.sharding import PartitionSpec as P
    from transformer_engine.jax.cpp_extensions import ep

    mesh = SimpleNamespace(shape={"fsdp": 4, "ep": ep_size})
    axes = ("ep", ("fsdp",))
    assert ep._leading_axis_ok(P("fsdp", None, None), axes, mesh)[0] is expected
    assert ep._leading_axis_ok(P(None, None, None), axes, mesh)[0] is expected
    assert ep._leading_axis_ok(P(("fsdp", "ep"), None, None), axes, mesh)[0]
    assert not ep._leading_axis_ok(P("wrong", None, None), axes, mesh)[0]
    assert not ep._leading_axis_ok(P("fsdp", "fsdp", None), axes, mesh)[0]
    assert not ep._leading_axis_ok(P("fsdp", None, None), axes)[0]
    missing_ep_mesh = SimpleNamespace(shape={"fsdp": 4})
    assert not ep._leading_axis_ok(P("fsdp", None, None), axes, missing_ep_mesh)[0]


def test_fused_ep_partitioning_retains_bind_time_resources(monkeypatch):
    from types import SimpleNamespace
    from jax.sharding import NamedSharding, PartitionSpec as P
    from transformer_engine.jax.cpp_extensions import ep

    captured = {}

    def bind(*args, **kwargs):
        captured.update(kwargs)
        return ()

    monkeypatch.setattr(ep.EpPrepareAndDispatchPrimitive.outer_primitive, "bind", bind)
    cfg = SimpleNamespace(top_k=1, dispatch_output_per_expert_alignment=256, topk_idx_dtype=0)
    with global_shard_guard(MeshResource(dp_resource="dp", fsdp_resource="fsdp", ep_resource="ep")):
        inspect.unwrap(ep._ep_fused_prepare_and_dispatch_fwd)(cfg, None, None, None, None, 256)

    mesh = Mesh(np.asarray(jax.devices()[:1]).reshape(1, 1, 1), ("dp", "fsdp", "ep"))
    sharding = NamedSharding(mesh, P(("dp", "fsdp", "ep"), None, None))
    args = [SimpleNamespace(sharding=sharding)] * 4
    _, _, outputs, _ = ep.EpPrepareAndDispatchPrimitive.partition(
        captured["top_k"],
        captured["dispatch_output_per_expert_alignment"],
        captured["topk_idx_dtype"],
        captured["recv_capacity_per_rank"],
        captured["is_outer"],
        captured["resource_axes"],
        mesh,
        args,
        None,
    )
    assert all(output.spec[0] == ("dp", "fsdp", "ep") for output in outputs)


@pytest.mark.parametrize(
    "fsdp_resource,outer_axis",
    [("fsdp", "fsdp"), (("fsdp", "ep"), "fsdp"), (("ep", "fsdp"), "fsdp"), (("ep",), None)],
)
def test_ep_outer_axis_excludes_ep_from_fsdp_tuple(monkeypatch, fsdp_resource, outer_axis):
    from types import SimpleNamespace
    from transformer_engine.jax import sharding
    from transformer_engine.jax.cpp_extensions import ep
    from transformer_engine.jax.ep import _default_out_partition_spec

    monkeypatch.setattr(sharding, "_get_mesh", lambda: SimpleNamespace(shape={"fsdp": 2, "ep": 4}))
    with global_shard_guard(MeshResource(fsdp_resource=fsdp_resource, ep_resource="ep")):
        assert ep._ep_outer_axis() == outer_axis
        leading = (outer_axis, "ep") if outer_axis is not None else "ep"
        assert _default_out_partition_spec() == (leading,)


def test_default_ep_out_spec_flattens_two_outer_fsdp_axes(monkeypatch):
    from types import SimpleNamespace
    from transformer_engine.jax import sharding
    from transformer_engine.jax.cpp_extensions import ep
    from transformer_engine.jax.ep import _default_out_partition_spec

    mesh = SimpleNamespace(shape={"fsdp1": 2, "fsdp2": 2, "ep": 2})
    monkeypatch.setattr(sharding, "_get_mesh", lambda: mesh)
    resource = MeshResource(fsdp_resource=("fsdp1", "fsdp2", "ep"), ep_resource="ep")
    with global_shard_guard(resource):
        assert ep._ep_outer_axis() == ("fsdp1", "fsdp2")
        spec = _default_out_partition_spec()
    assert spec == (("fsdp1", "fsdp2", "ep"),)
    assert ep._leading_per_shard((16, 4), spec[0], mesh) == (2, 4)


def dp_fsdp_ep1_mesh():
    if len(jax.devices()) < 4:
        pytest.skip("Requires four JAX devices for a dp=2 x fsdp=2 x ep=1 mesh")
    return Mesh(np.asarray(jax.devices()[:4]).reshape(2, 2, 1), ("dp", "fsdp", "ep"))


def test_ep1_local_dispatch_groups_every_outer_axis():
    """EP=1 dispatch keeps one group per dp x fsdp shard and round-trips through combine."""
    mesh = dp_fsdp_ep1_mesh()
    resource = MeshResource(dp_resource="dp", fsdp_resource="fsdp", ep_resource="ep")
    num_groups, num_experts, top_k = 4, 4, 2
    batch, sequence, hidden = 8, 3, 8
    tokens_per_group = batch * sequence // num_groups
    recv_capacity = tokens_per_group * top_k
    cfg = EpLayerConfig(top_k=top_k, topk_idx_dtype=int(jax_dtype_to_te_dtype(jnp.int32)))
    rng = np.random.default_rng(0)
    topk_idx = np.stack(
        [rng.permutation(num_experts)[:top_k] for _ in range(batch * sequence)]
    ).reshape(batch, sequence, top_k)
    tokens = jnp.asarray(rng.uniform(-1, 1, (batch, sequence, hidden)), jnp.float32)
    topk_weights = jnp.asarray(rng.uniform(size=(batch, sequence, top_k)), jnp.float32)

    def run(x, weights):
        recv, recv_weights, handle, counts, total = ep_dispatch(
            cfg, jnp.asarray(topk_idx, jnp.int32), x, weights, recv_capacity
        )
        output = ep_combine(
            cfg,
            handle,
            counts,
            recv * recv_weights[..., None],
            num_local_tokens=(batch, sequence),
        )
        return output, recv, counts, total

    def loss(x, weights):
        return jnp.sum(run(x, weights)[0])

    with mesh, global_shard_guard(resource):
        ep_bootstrap(
            world_size=4,
            rank=0,
            num_experts=num_experts,
            max_tokens_per_rank=tokens_per_group,
            recv_capacity_per_rank=recv_capacity,
            hidden_dim=hidden,
        )
        try:
            sharding = NamedSharding(mesh, PartitionSpec(("dp", "fsdp", "ep"), None, None))
            x, weights = jax.device_put((tokens, topk_weights), sharding)
            output, recv, counts, total = jax.jit(run)(x, weights)
            grad_tokens, grad_weights = jax.jit(jax.grad(loss, argnums=(0, 1)))(x, weights)
        finally:
            ep_finalize()

    assert recv.shape == (num_groups, recv_capacity, hidden)
    # Each group counts the assignments of its own quarter of the batch.
    expected_counts = [
        np.bincount(group, minlength=num_experts) for group in topk_idx.reshape(num_groups, -1)
    ]
    np.testing.assert_array_equal(counts, expected_counts)
    np.testing.assert_array_equal(total, np.full((num_groups, 1), recv_capacity))
    weight_sums = jnp.sum(topk_weights, axis=-1, keepdims=True)
    np.testing.assert_allclose(output, tokens * weight_sums, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(
        grad_tokens, jnp.broadcast_to(weight_sums, tokens.shape), rtol=1e-6, atol=1e-6
    )
    np.testing.assert_allclose(
        grad_weights,
        jnp.repeat(jnp.sum(tokens, axis=-1, keepdims=True), top_k, axis=-1),
        rtol=1e-5,
        atol=1e-5,
    )


def test_ep1_moe_on_dp_fsdp_mesh_matches_dense_reference():
    """moe() at dp=2 x fsdp=2 x ep=1 matches a dense reference in forward and backward."""
    mesh = dp_fsdp_ep1_mesh()
    resource = MeshResource(dp_resource="dp", fsdp_resource="fsdp", ep_resource="ep")
    num_experts, top_k = 4, 2
    batch, sequence, hidden, intermediate = 4, 16, 128, 128
    max_tokens_per_rank = batch * sequence // 4
    recv_capacity = get_moe_recv_capacity_per_rank(
        num_experts=num_experts,
        num_experts_per_tok=top_k,
        max_tokens_per_rank=max_tokens_per_rank,
        ep_size=1,
        use_cudnn_fusion=False,
    )
    keys = jax.random.split(jax.random.PRNGKey(0), 6)
    x = jax.random.normal(keys[0], (batch, sequence, hidden), jnp.bfloat16)
    wi = jax.random.normal(keys[1], (num_experts, hidden, 2 * intermediate)) / hidden**0.5
    wo = jax.random.normal(keys[2], (num_experts, intermediate, hidden)) / intermediate**0.5
    wi, wo = wi.astype(jnp.bfloat16), wo.astype(jnp.bfloat16)
    scores = jax.random.uniform(keys[3], (batch, sequence, num_experts))
    topk_idx = jnp.argsort(scores, axis=-1)[..., :top_k].astype(jnp.int32)
    topk_weights = jax.nn.softmax(jax.random.normal(keys[4], (batch, sequence, top_k)), axis=-1)
    cotangent = jax.random.normal(keys[5], (batch, sequence, hidden))

    def te_loss(x, wi, wo):
        output = moe(
            x,
            RoutingMapInfo(routing_indices=topk_idx, routing_weights=topk_weights),
            wi,
            wo,
            num_experts=num_experts,
            mesh_resource=resource,
            dtype=jnp.bfloat16,
            collect_expert_counts=False,
            use_cudnn_fusion=False,
        )[0]
        return jnp.sum(output.astype(jnp.float32) * cotangent), output

    def reference_loss(x, wi, wo):
        gate, up = jnp.split(jnp.einsum("bsh,ehf->bsef", x, wi), 2, axis=-1)
        expert_out = jnp.einsum("bsei,eih->bseh", jax.nn.silu(gate) * up, wo)
        routing = jax.nn.one_hot(topk_idx, num_experts, dtype=x.dtype)
        routing = jnp.sum(routing * topk_weights[..., None].astype(x.dtype), axis=-2)
        output = jnp.einsum("bse,bseh->bsh", routing, expert_out)
        return jnp.sum(output.astype(jnp.float32) * cotangent), output

    with mesh, global_shard_guard(resource):
        ep_bootstrap(
            world_size=4,
            rank=0,
            num_experts=num_experts,
            max_tokens_per_rank=max_tokens_per_rank,
            recv_capacity_per_rank=recv_capacity,
            hidden_dim=hidden,
        )
        record_ep_bootstrap_signature_for_moe(
            num_experts=num_experts,
            max_tokens_per_rank=max_tokens_per_rank,
            recv_capacity_per_rank=recv_capacity,
            hidden_dim=hidden,
            ep_size=1,
        )
        try:
            (_, output), grads = jax.jit(
                jax.value_and_grad(te_loss, argnums=(0, 1, 2), has_aux=True)
            )(x, wi, wo)
        finally:
            ep_finalize()
    (_, ref_output), ref_grads = jax.jit(
        jax.value_and_grad(reference_loss, argnums=(0, 1, 2), has_aux=True)
    )(x, wi, wo)

    # Both sides round BF16 intermediates, at different points.
    tols = dtype_tols(jnp.bfloat16)
    np.testing.assert_allclose(output.astype(jnp.float32), ref_output.astype(jnp.float32), **tols)
    for grad, ref_grad in zip(grads, ref_grads):
        np.testing.assert_allclose(grad.astype(jnp.float32), ref_grad.astype(jnp.float32), **tols)


@pytest.mark.parametrize(
    "resource,quantize,error",
    [
        (MeshResource(), False, "ep_resource"),
        (MeshResource(ep_resource="ep", dp_resource="ep"), False, "distinct"),
        (MeshResource(ep_resource="ep"), True, "fsdp_resource"),
    ],
)
def test_invalid_resource(resource, quantize, error):
    with pytest.raises(ValueError, match=error):
        _resolve_moe_mesh_resource(resource, quantize)


def test_bool_and_resource_types():
    with pytest.raises(TypeError, match="must be a bool"):
        _resolve_moe_mesh_resource(MeshResource(ep_resource="ep"), "true")
    with pytest.raises(TypeError, match="must be a MeshResource"):
        _resolve_moe_mesh_resource("ep")


@pytest.mark.parametrize("quantize", [False, True])
def test_public_api_uses_selected_resource(monkeypatch, quantize):
    module = importlib.import_module("transformer_engine.jax.moe")
    signature = inspect.signature(module._moe)
    captured = {}

    def fake_vjp(*args):
        captured.update(signature.bind(*args).arguments)
        assert global_mesh_resource().ep_resource == "ep"
        return args[0], None, jnp.zeros((1,), jnp.int32), jnp.zeros((2,), jnp.int32)

    monkeypatch.setattr(module, "_moe", fake_vjp)
    mesh = Mesh(np.asarray(jax.devices()[:1]).reshape(1, 1, 1), ("dp", "fsdp", "ep"))
    kwargs = dict(
        num_experts=2,
        mesh_resource=MeshResource(dp_resource="dp", fsdp_resource="fsdp", ep_resource="ep"),
        quant_before_fsdp_ag=quantize,
    )
    with jax.set_mesh(mesh):
        module.moe(
            jnp.ones((1, 1, 4)),
            RouterComputationInfo(gate_kernel=jnp.ones((4, 2)), num_experts_per_tok=1),
            jnp.ones((2, 4, 8)),
            jnp.ones((2, 4, 4)),
            **kwargs,
        )
    assert _moe_mesh_axes(captured["mesh_resource"]) == ("ep", ("dp", "fsdp"))
    assert captured["quant_before_fsdp_ag"] is quantize
    assert "ep_axis" not in captured
    with pytest.raises(AssertionError, match="Global mesh resource is not set"):
        global_mesh_resource()


@pytest.mark.parametrize("quantize", [False, True])
def test_flax_block_forwards_boolean(monkeypatch, quantize):
    module = importlib.import_module("transformer_engine.jax.flax.moe")
    resource = MeshResource(fsdp_resource="fsdp", ep_resource="ep")
    captured = {}

    def fake_moe(inputs, *args, **kwargs):
        captured.update(kwargs)
        return inputs, None, jnp.zeros((1,), jnp.int32), jnp.zeros((2,), jnp.int32)

    monkeypatch.setattr(module, "moe", fake_moe)
    mesh = Mesh(np.asarray(jax.devices()[:1]).reshape(1, 1), ("fsdp", "ep"))
    block = _MoEBlock(
        num_experts=2,
        intermediate_size=4,
        mesh_resource=resource,
        quant_before_fsdp_ag=quantize,
    )
    with jax.set_mesh(mesh):
        block.init(jax.random.PRNGKey(0), jnp.ones((1, 1, 4)))
    assert captured["quant_before_fsdp_ag"] is quantize
    assert _moe_mesh_axes(captured["mesh_resource"]) == ("ep", ("fsdp",))
