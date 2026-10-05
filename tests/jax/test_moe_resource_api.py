# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""MoE resource resolution and deprecated API compatibility without EP kernels."""

import importlib
import inspect
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh

from transformer_engine.jax.flax import _MoEBlock
from transformer_engine.jax.moe import WeightGather, _moe_mesh_axes, _resolve_moe_mesh_resource
from transformer_engine.jax.sharding import MeshResource, global_mesh_resource, global_shard_guard


@pytest.fixture(autouse=True)
def no_global_resource():
    with global_shard_guard(None):
        yield


def test_resource_required_for_function_and_block():
    module = importlib.import_module("transformer_engine.jax.moe")
    with pytest.raises(ValueError, match="active global_shard_guard"):
        module.moe(None, None, None, None, num_experts=2, num_experts_per_tok=1)
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


def test_legacy_preserves_arbitrary_outer_axis_order():
    with pytest.warns(DeprecationWarning):
        resource, quantize = _resolve_moe_mesh_resource(
            ep_axis="ep",
            data_parallelism_axes=("outer", "fsdp", "replica"),
            weight_gather=WeightGather.quantized(axis="fsdp"),
        )
    assert _moe_mesh_axes(resource) == ("ep", ("outer", "fsdp", "replica"))
    assert resource.fsdp_resource == "fsdp"
    assert quantize is True


def test_legacy_defaults_to_no_outer_axes():
    with global_shard_guard(MeshResource(dp_resource="dp", fsdp_resource="fsdp", ep_resource="ep")):
        with pytest.warns(DeprecationWarning):
            resource, _ = _resolve_moe_mesh_resource(ep_axis="ep")
    assert _moe_mesh_axes(resource) == ("ep", ())


@pytest.mark.parametrize(
    "kwargs,error",
    [
        ({"ep_axis": "other"}, "ep_axis conflicts"),
        ({"data_parallelism_axes": ("other",)}, "data_parallelism_axes conflicts"),
        ({"weight_gather": WeightGather.quantized(axis="other")}, "weight_gather axis conflicts"),
    ],
)
def test_conflicting_old_and_new_args(kwargs, error):
    with pytest.warns(DeprecationWarning):
        with pytest.raises(ValueError, match=error):
            _resolve_moe_mesh_resource(
                MeshResource(fsdp_resource="fsdp", ep_resource="ep"), **kwargs
            )


@pytest.mark.parametrize("legacy", [False, True])
def test_public_api_delegates_with_selected_resource(monkeypatch, legacy):
    module = importlib.import_module("transformer_engine.jax.moe")
    original_moe = module.moe
    signature = inspect.signature(module._moe)
    captured = {}

    def fake_vjp(*args):
        captured.update(signature.bind(*args).arguments)
        assert global_mesh_resource().ep_resource == "ep"
        return args[0], None, jnp.zeros((1,), jnp.int32)

    def delegated_moe(*args, **kwargs):
        assert "mesh_resource" in kwargs
        assert not {"ep_axis", "data_parallelism_axes", "weight_gather"}.intersection(kwargs)
        return original_moe(*args, **kwargs)

    monkeypatch.setattr(module, "_moe", fake_vjp)
    monkeypatch.setattr(module, "moe", delegated_moe)
    mesh = Mesh(np.asarray(jax.devices()[:1]).reshape(1, 1, 1), ("dp", "fsdp", "ep"))
    kwargs = dict(num_experts=2, num_experts_per_tok=1)
    if legacy:
        kwargs.update(
            ep_axis="ep",
            data_parallelism_axes=("dp", "fsdp"),
            weight_gather=WeightGather.quantized(axis="fsdp"),
        )
    else:
        kwargs.update(
            mesh_resource=MeshResource(dp_resource="dp", fsdp_resource="fsdp", ep_resource="ep"),
            quant_before_fsdp_ag=True,
        )
    with jax.set_mesh(mesh), warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always", DeprecationWarning)
        original_moe(
            jnp.ones((1, 1, 4)),
            jnp.ones((4, 2)),
            jnp.ones((2, 4, 8)),
            jnp.ones((2, 4, 4)),
            **kwargs,
        )
    assert any("deprecated for TE MoE" in str(w.message) for w in recorded) == legacy
    assert _moe_mesh_axes(captured["mesh_resource"]) == ("ep", ("dp", "fsdp"))
    assert captured["quant_before_fsdp_ag"] is True
    assert "ep_axis" not in captured
    with pytest.raises(AssertionError, match="Global mesh resource is not set"):
        global_mesh_resource()
