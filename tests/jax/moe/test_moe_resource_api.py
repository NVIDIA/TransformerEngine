# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""MoE resource resolution and MeshResource-only API without EP kernels."""

import importlib
import inspect
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh

from transformer_engine.jax.cpp_extensions.activation import ActivationParams
from transformer_engine.jax.flax import _MoEBlock
from transformer_engine.jax.moe import (
    RouterComputationInfo,
    _moe_mesh_axes,
    _resolve_moe_mesh_resource,
)
from transformer_engine.jax.sharding import MeshResource, global_mesh_resource, global_shard_guard


@pytest.mark.parametrize("soft_caps", [None, {"gate_beta": 2.0, "up_beta": 8.0}])
def test_situ_activation_params_reach_grouped_gemm_adapters(monkeypatch, soft_caps):
    module = importlib.import_module("transformer_engine.jax.moe")
    adapter = importlib.import_module("transformer_engine.jax.cpp_extensions.grouped_gemm_glu")
    params = module._resolve_moe_activation_params("situglu", soft_caps)
    assert isinstance(params, ActivationParams)
    calls = []
    scales = jnp.ones((512,), dtype=jnp.uint8)

    def fused(**kwargs):
        calls.append(kwargs)
        return dict(
            c_tensor=jnp.ones((1, 64)),
            d_tensor=jnp.ones((1, 32)),
            d_col_tensor=jnp.ones((1, 32)),
            d_row_tensor=jnp.ones((1, 64)),
            dprob_tensor=jnp.ones((1,)),
            sfd_row_tensor=scales,
            sfd_col_tensor=scales,
        )

    cudnn = SimpleNamespace(
        grouped_gemm_glu_wrapper_sm100=fused, grouped_gemm_dglu_wrapper_sm100=fused
    )
    original_import = importlib.import_module
    monkeypatch.setattr(
        adapter.importlib,
        "import_module",
        lambda name: cudnn if name == "cudnn" else original_import(name),
    )
    adapter.grouped_gemm_glu(
        jnp.ones((1, 32, 1)),
        jnp.ones((1, 64, 32)),
        scales,
        scales,
        jnp.asarray([1]),
        jnp.ones((1,)),
        compute_dtype=jnp.bfloat16,
        output_dtype=jnp.float8_e4m3fn,
        **module._fused_glu_activation_kwargs("situglu", params),
    )
    adapter.grouped_gemm_dglu(
        jnp.ones((1, 32)),
        jnp.ones((1, 32, 32)),
        jnp.ones((1, 64)),
        scales,
        scales,
        jnp.asarray([1]),
        jnp.ones((1,)),
        output_dtype=jnp.float8_e4m3fn,
        **module._fused_glu_activation_kwargs("situglu", params, backward=True),
    )
    expected = soft_caps or {"gate_beta": 4.0, "up_beta": 25.0}
    assert [call["act_func"] for call in calls] == ["situglu", "dsituglu"]
    for call in calls:
        assert call["situ_beta1"] == expected["gate_beta"]
        assert call["situ_beta2"] == expected["up_beta"]


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


def test_tuple_ep_resource():
    resource, _ = _resolve_moe_mesh_resource(
        MeshResource(dp_resource="dp", fsdp_resource="fsdp", ep_resource=("ep", "tp"))
    )
    assert _moe_mesh_axes(resource) == (("ep", "tp"), ("dp", "fsdp"))
    with pytest.raises(ValueError, match="distinct"):
        _resolve_moe_mesh_resource(MeshResource(fsdp_resource="tp", ep_resource=("ep", "tp")))


@pytest.mark.parametrize(
    "resource,outer_axes",
    [
        (
            MeshResource(
                dp_resource="dp", fsdp_resource=("ep", "fsdp", "tp"), ep_resource=("ep", "tp")
            ),
            ("dp", "fsdp"),
        ),
        (MeshResource(dp_resource="tp", fsdp_resource="ep", ep_resource=("ep", "tp")), ()),
    ],
)
def test_ep_outer_axes_flatten_tuple_resources(resource, outer_axes):
    """EP outer axes flatten a tuple FSDP resource and drop every axis of a tuple EP."""
    from transformer_engine.jax.cpp_extensions import ep

    with global_shard_guard(resource):
        assert ep._capture_ep_resource_axes() == (("ep", "tp"), outer_axes)


def test_tuple_resources_flatten_in_default_spec_and_sizes(monkeypatch):
    """Tuple FSDP and EP resources give a flat default EP output spec and product sizes."""
    from types import SimpleNamespace
    from transformer_engine.jax import sharding
    from transformer_engine.jax.ep import _default_out_partition_spec

    mesh = SimpleNamespace(shape={"fsdp": 2, "data": 2, "ep": 2, "tp": 2}, empty=False)
    monkeypatch.setattr(sharding, "_get_mesh", lambda: mesh)
    resource = MeshResource(fsdp_resource=("fsdp", "data", "ep"), ep_resource=("ep", "tp"))
    with global_shard_guard(resource):
        assert _default_out_partition_spec() == (("fsdp", "data", "ep", "tp"),)
        assert sharding.ep_axis_size() == 4
        assert sharding.get_active_resource_axis("ep_resource") == ("ep", "tp")


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


@pytest.mark.parametrize("tp_size,expected", [(1, True), (2, False)])
def test_tuple_ep_leading_axis_canonicalization(tp_size, expected):
    """A tuple EP axis may be absent from the spec only when its mesh size is one."""
    from types import SimpleNamespace
    from jax.sharding import PartitionSpec as P
    from transformer_engine.jax.cpp_extensions import ep

    mesh = SimpleNamespace(shape={"fsdp": 2, "ep": 2, "tp": tp_size})
    axes = (("ep", "tp"), ("fsdp",))
    assert ep._leading_axis_ok(P(("fsdp", "ep"), None, None), axes, mesh)[0] is expected
    assert ep._leading_axis_ok(P("ep", None, None), axes, mesh)[0] is expected
    assert ep._leading_axis_ok(P(("fsdp", "ep", "tp"), None, None), axes, mesh)[0]
    assert not ep._leading_axis_ok(P(("fsdp", "tp"), None, None), axes, mesh)[0]
    assert not ep._leading_axis_ok(P(None, None, None), axes, mesh)[0]
    assert not ep._leading_axis_ok(P(("fsdp", "ep"), None, None), axes)[0]


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
def test_default_ep_out_spec_excludes_ep_from_fsdp_tuple(fsdp_resource, outer_axis):
    from transformer_engine.jax.ep import _default_out_partition_spec

    with global_shard_guard(MeshResource(fsdp_resource=fsdp_resource, ep_resource="ep")):
        leading = (outer_axis, "ep") if outer_axis is not None else "ep"
        assert _default_out_partition_spec() == (leading,)


def test_default_ep_out_spec_flattens_two_outer_fsdp_axes():
    from types import SimpleNamespace
    from transformer_engine.jax.cpp_extensions import ep
    from transformer_engine.jax.ep import _default_out_partition_spec

    mesh = SimpleNamespace(shape={"fsdp1": 2, "fsdp2": 2, "ep": 2})
    resource = MeshResource(fsdp_resource=("fsdp1", "fsdp2", "ep"), ep_resource="ep")
    with global_shard_guard(resource):
        spec = _default_out_partition_spec()
    assert spec == (("fsdp1", "fsdp2", "ep"),)
    assert ep._leading_per_shard((16, 4), spec[0], mesh) == (2, 4)


def test_default_ep_out_spec_uses_dp_and_fsdp():
    """The default EP out spec shards over dp and fsdp, as ep_bootstrap counts EP groups."""
    from types import SimpleNamespace
    from transformer_engine.jax.cpp_extensions import ep
    from transformer_engine.jax.ep import _default_out_partition_spec

    mesh = SimpleNamespace(shape={"dp": 2, "fsdp": 2, "ep": 2})
    resource = MeshResource(dp_resource="dp", fsdp_resource="fsdp", ep_resource="ep")
    with global_shard_guard(resource):
        spec = _default_out_partition_spec()
    assert spec == (("dp", "fsdp", "ep"),)
    assert ep._leading_per_shard((16, 4), spec[0], mesh) == (2, 4)


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
    from transformer_engine.jax.ep import ep_bootstrap

    with pytest.raises(TypeError, match="must be a bool"):
        _resolve_moe_mesh_resource(MeshResource(ep_resource="ep"), "true")
    with pytest.raises(TypeError, match="must be a MeshResource"):
        _resolve_moe_mesh_resource("ep")
    with pytest.raises(TypeError, match="must be a MeshResource"):
        ep_bootstrap(None, None, 2, 1, 1, 1, mesh_resource="ep")


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
