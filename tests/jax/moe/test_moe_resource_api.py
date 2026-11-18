# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""MoE resource resolution and MeshResource-only API without EP kernels."""

import importlib
import inspect
from functools import partial
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
    _apply_ffn_activation,
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


@pytest.mark.parametrize("betas", [(4.0, 25.0), (2.0, 7.0)])
def test_situ_value_and_both_branch_derivatives(betas):
    (gate_beta, up_beta) = betas
    gate = jnp.asarray([-10000.0, -12.0, -0.5, 0.0, 0.5, 12.0, 10000.0])
    up = jnp.asarray([10000.0, 40.0, -3.0, 2.0, 3.0, -40.0, -10000.0])
    params = ActivationParams.create("situglu", gate_beta=gate_beta, up_beta=up_beta)
    counts = jnp.asarray([1], jnp.int32)

    def activate(g, u):
        return _apply_ffn_activation("situglu", g[None, :], u[None, :], counts, params)[0]

    (output, pullback) = jax.vjp(activate, gate, up)
    (dg, du) = pullback(jnp.ones_like(output))
    (g, u) = (np.asarray(gate, np.float64), np.asarray(up, np.float64))
    sigmoid = 0.5 + 0.5 * np.tanh(g / 2)
    (tg, tu) = (np.tanh(g / gate_beta), np.tanh(u / up_beta))
    (capped_gate, capped_up) = (gate_beta * tg * sigmoid, up_beta * tu)
    derivative_gate = (1 - tg**2) * sigmoid + gate_beta * tg * sigmoid * (1 - sigmoid)
    for actual, expected in (
        (output, capped_gate * capped_up),
        (dg, derivative_gate * capped_up),
        (du, capped_gate * (1 - tu**2)),
    ):
        np.testing.assert_allclose(actual, expected, atol=2e-05, rtol=2e-05)


def test_situ_bf16_uses_fp32_before_final_cast():
    gate = jnp.asarray([0.25, 1.5, -3.0, 12.0, 64.0], dtype=jnp.bfloat16)
    up = jnp.asarray([0.75, -2.0, 32.0, 64.0, -128.0], dtype=jnp.bfloat16)
    actual = _apply_ffn_activation(
        "situglu", gate[None, :], up[None, :], jnp.asarray([1], jnp.int32)
    )[0]
    (g, u) = (np.asarray(gate, np.float64), np.asarray(up, np.float64))
    expected = jnp.asarray(100 * np.tanh(g / 4) / (1 + np.exp(-g)) * np.tanh(u / 25), jnp.bfloat16)
    assert actual.dtype == jnp.bfloat16
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float16, jnp.bfloat16])
@pytest.mark.parametrize("betas", [(4.0, 25.0), (2.0, 8.0)])
@pytest.mark.parametrize("counts", [[3, 0, 5], [0, 0, 0], [0] * 257 + [13], [0]])
def test_situ_masks_inactive_values_and_gradients(dtype, betas, counts):
    capacity = 0 if counts == [0] else 13
    active = sum(counts)
    rng = np.random.default_rng(72)
    (gate, up, dy) = [rng.uniform(-5, 5, (capacity, 6)).astype(np.float32) for _ in range(3)]
    for value in (gate, up, dy):
        value[active:] = np.nan
    (gate, up, dy) = [jnp.asarray(value, dtype) for value in (gate, up, dy)]
    sizes = jnp.asarray(counts, jnp.int32)
    activate = lambda g, u: _apply_ffn_activation(
        "situglu",
        g,
        u,
        sizes,
        ActivationParams.create("situglu", gate_beta=betas[0], up_beta=betas[1]),
    )
    output = jax.jit(activate)(gate, up)
    (_, pullback) = jax.vjp(activate, gate, up)
    (dg, du) = jax.jit(pullback)(dy)
    (g, u, d) = [np.asarray(value, np.float64)[:active] for value in (gate, up, dy)]
    (gate_beta, up_beta) = betas
    (tg, tu, sigmoid) = (np.tanh(g / gate_beta), np.tanh(u / up_beta), 1 / (1 + np.exp(-g)))
    ref = (
        gate_beta * tg * sigmoid * up_beta * tu,
        d * ((1 - tg**2) * sigmoid + gate_beta * tg * sigmoid * (1 - sigmoid)) * up_beta * tu,
        d * gate_beta * tg * sigmoid * (1 - tu**2),
    )
    tolerance = {jnp.float32: 2e-05, jnp.float16: 0.003, jnp.bfloat16: 0.02}[dtype]
    for actual, expected in zip((output, dg, du), ref):
        actual = np.asarray(actual, np.float32)
        assert np.isfinite(actual).all()
        np.testing.assert_array_equal(actual[active:], 0)
        np.testing.assert_allclose(actual[:active], expected, atol=tolerance, rtol=tolerance)


def test_situ_dynamic_counts_remat_and_vmap():
    gate = jnp.arange(13 * 6, dtype=jnp.float32).reshape(13, 6) / 20 - 2
    up = -gate
    activation = "situglu"
    activate = partial(_apply_ffn_activation, activation)
    compiled = jax.jit(activate)
    for counts in (jnp.array([3, 0, 5]), jnp.array([1, 2, 2])):
        expected = activate(gate, up, counts)
        np.testing.assert_allclose(compiled(gate, up, counts), expected, atol=2e-05, rtol=2e-05)
        loss = lambda g, u: activate(g, u, counts).sum()
        gradients = jax.jit(jax.grad(loss, (0, 1)))(gate, up)
        remat = jax.jit(jax.grad(jax.checkpoint(loss), (0, 1)))(gate, up)
        for actual, reference in zip(remat, gradients):
            np.testing.assert_allclose(actual, reference, atol=2e-05, rtol=2e-05)
    counts = jnp.array([[3, 0, 5], [1, 2, 2]], dtype=jnp.int32)
    (gates, ups) = (jnp.stack([gate, gate]), jnp.stack([up, up]))
    mapped = jax.jit(jax.vmap(activate))(gates, ups, counts)
    for actual, sizes in zip(mapped, counts):
        np.testing.assert_allclose(actual, activate(gate, up, sizes), atol=2e-05, rtol=2e-05)
    hlo = compiled.lower(gate, up, counts[0]).compile().as_text()
    assert "te_grouped_situglu_ffi" not in hlo


@pytest.mark.parametrize(
    "capability,available,expected",
    [(89, True, False), (100, False, False), (103, True, True), (107, True, False)],
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


def test_situ_can_disable_fusion():
    module = importlib.import_module("transformer_engine.jax.moe")
    with pytest.raises(ValueError, match="active global_shard_guard"):
        module.moe(
            None, None, None, None, num_experts=2, activation_type="situglu", use_cudnn_fusion=False
        )


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
