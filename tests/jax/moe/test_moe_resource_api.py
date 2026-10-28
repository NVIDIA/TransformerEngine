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
from jax.sharding import Mesh

from transformer_engine.jax.flax import _MoEBlock
from transformer_engine.jax.moe import (
    RouterComputationInfo,
    _moe_mesh_axes,
    _resolve_moe_mesh_resource,
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


def test_ep_partitioning_retains_bind_time_resources():
    from transformer_engine.jax.cpp_extensions import ep
    from jax.sharding import PartitionSpec as P

    with global_shard_guard(MeshResource(dp_resource="dp", fsdp_resource="fsdp", ep_resource="ep")):
        captured = ep._capture_ep_resource_axes()
    spec = P(("dp", "fsdp", "ep"), None, None)
    assert ep._leading_axis_ok(spec, captured) == (True, "ep", ("dp", "fsdp"))
    assert ep._ep_output_spec(None, None, resource_axes=captured) == spec
    assert ep._ep_spec_ok(spec, 2, resource_axes=captured)
    assert not ep._ep_spec_ok(P(("wrong", "ep"), None, None), 2, resource_axes=captured)


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
        inspect.unwrap(ep._ep_fused_prepare_and_dispatch_fwd)(cfg, None, None, None, 256)

    mesh = Mesh(np.asarray(jax.devices()[:1]).reshape(1, 1, 1), ("dp", "fsdp", "ep"))
    sharding = NamedSharding(mesh, P(("dp", "fsdp", "ep"), None, None))
    args = [SimpleNamespace(sharding=sharding)] * 3
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
