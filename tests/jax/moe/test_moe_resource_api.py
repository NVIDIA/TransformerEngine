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


@pytest.mark.parametrize(
    "apply_topk_weights_early,counts",
    [(False, [3, 0, 5]), (True, [3, 0, 5]), (True, [0, 0, 0])],
)
def test_unfused_situ_ffn_masks_inactive_rows(monkeypatch, apply_topk_weights_early, counts):
    module = importlib.import_module("transformer_engine.jax.moe")
    captured = {}

    class _Tensor:
        def __init__(self, data, first_dims=None):
            self.data, self.first_dims = data, first_dims

        def get_tensor(self, usage=None):
            del usage
            return self

        def checkpoint(self, quantizer=None):
            del quantizer
            return self

    def quantize(data, quantizer, group_sizes=None, **_):
        if quantizer == "fc1_dgrad":
            captured["d_combined"] = data
        return _Tensor(data, group_sizes)

    def gemm(lhs, rhs, *, contracting_dims, bias=None):
        del bias
        if contracting_dims == ((0,), (0,)):
            return jnp.zeros((len(counts), lhs.data.shape[1], rhs.data.shape[1]), lhs.data.dtype)
        active = jnp.arange(lhs.data.shape[0]) < lhs.first_dims.sum()
        rhs_mat = rhs.data[0] if contracting_dims == ((1,), (1,)) else rhs.data[0].T
        return jnp.where(active[:, None], lhs.data @ rhs_mat, jnp.nan)

    monkeypatch.setattr(module.tex, "grouped_quantize", quantize)
    monkeypatch.setattr(module.tex, "grouped_gemm", gemm)

    capacity, active, inter = 10, sum(counts), 4
    hidden = 2 * inter
    rng = np.random.default_rng(72)
    gate, up, dy = [rng.uniform(-4, 4, (capacity, inter)).astype(np.float32) for _ in range(3)]
    weights_np = rng.uniform(0.1, 0.9, (1, capacity)).astype(np.float32)
    for arr in (gate, up, dy):
        arr[active:] = np.nan
    weights_np[:, active:] = np.nan

    tokens = jnp.asarray(np.concatenate([gate, up], axis=-1))[None]
    d_out = jnp.asarray(np.concatenate([dy, np.zeros_like(dy)], axis=-1))[None]
    weights = jnp.asarray(weights_np)
    sizes = jnp.asarray(counts, jnp.int32)
    wi = jnp.broadcast_to(jnp.eye(hidden, dtype=jnp.float32), (len(counts), hidden, hidden))
    wo = jnp.broadcast_to(jnp.eye(inter, hidden, dtype=jnp.float32), (len(counts), inter, hidden))
    qsets = (
        SimpleNamespace(x=None, kernel=None, dgrad="fc1_dgrad"),
        SimpleNamespace(x=None, kernel=None, dgrad=None),
    )
    params = ActivationParams.create("situglu", gate_beta=2.0, up_beta=8.0)

    _, residuals = module._ffn_fwd_per_shard(
        tokens,
        weights,
        sizes,
        wi,
        wo,
        None,
        None,
        None,
        None,
        qsets,
        compute_dtype=jnp.float32,
        num_local_experts=len(counts),
        activation_type="situglu",
        apply_topk_weights_early=apply_topk_weights_early,
        use_cudnn_jax_fusion=False,
        wi_0_checkpoint_name=None,
        wi_1_checkpoint_name=None,
        wo_checkpoint_name=None,
        cudnn_native_weight_layout=False,
        quant_before_fsdp_ag=False,
        fsdp_axis=None,
        fsdp_size=1,
        wi_fsdp_axis=None,
        wo_fsdp_axis=None,
        act_params=params,
    )
    _, d_w, *_ = module._ffn_bwd_per_shard(
        d_out,
        *residuals,
        weights,
        qsets,
        activation_type="situglu",
        apply_topk_weights_early=apply_topk_weights_early,
        has_bias=False,
        use_cudnn_jax_fusion=False,
        cudnn_native_weight_layout=False,
        act_params=params,
    )

    g, u, d = gate[:active], up[:active], dy[:active]
    tg, tu, sig = np.tanh(g / 2.0), np.tanh(u / 8.0), 1 / (1 + np.exp(-g))
    mid = (2.0 * tg * sig) * (8.0 * tu)
    w = weights_np[0, :active, None] if apply_topk_weights_early else 1.0
    d_combined = captured["d_combined"]
    expected = (
        mid * w,
        (d * w) * ((1 - tg**2) * sig + 2.0 * tg * sig * (1 - sig)) * (8.0 * tu),
        (d * w) * (2.0 * tg * sig) * (1 - tu**2),
        np.sum(d * mid, axis=-1) if apply_topk_weights_early else np.zeros(active),
    )
    for actual, ref in zip(
        (residuals[4].data, d_combined[:, :inter], d_combined[:, inter:], d_w[0]), expected
    ):
        arr = np.asarray(actual, np.float32)
        assert np.isfinite(arr).all()
        np.testing.assert_array_equal(arr[active:], 0)
        np.testing.assert_allclose(arr[:active], ref, atol=2e-05, rtol=2e-05)


@pytest.mark.parametrize(
    "capability,available,expected",
    [(89, True, False), (100, True, True), (100, False, False), (107, True, False)],
)
def test_situ_fusion_or_jax_fallback(monkeypatch, capability, available, expected):
    import transformer_engine_jax

    module = importlib.import_module("transformer_engine.jax.moe")
    monkeypatch.setattr(
        transformer_engine_jax, "get_device_compute_capability", lambda _: capability
    )
    monkeypatch.setattr(
        module.tex,
        "grouped_gemm_glu_dependencies_available",
        lambda **_: (available, "unavailable frontend"),
    )
    if expected:
        assert module._select_cudnn_jax_fusion([], activation_type="situglu")
    else:
        with pytest.warns(UserWarning, match="falling back"):
            assert not module._select_cudnn_jax_fusion([], activation_type="situglu")


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
