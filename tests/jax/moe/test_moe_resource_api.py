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

from transformer_engine.jax.cpp_extensions.activation import (
    ActivationParams,
    _jax_act_lu,
)
from transformer_engine.jax.flax import _MoEBlock
from transformer_engine.jax.moe import (
    RouterComputationInfo,
    _moe_mesh_axes,
    _resolve_moe_mesh_resource,
)
from transformer_engine.jax.sharding import MeshResource, global_mesh_resource, global_shard_guard


@jax.tree_util.register_pytree_node_class
class _FakeGroupedTensor:
    def __init__(self, data, first_dims=None):
        self.data = data
        self.first_dims = first_dims

    def get_tensor(self, usage=None):
        del usage
        return self

    def checkpoint(self, quantizer=None):
        del quantizer
        return self

    def tree_flatten(self):
        return (self.data, self.first_dims), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        del aux_data
        return cls(*children)


def _install_unfused_ffn_doubles(monkeypatch, *, poison_inactive=False, captured_d_combined=None):
    module = importlib.import_module("transformer_engine.jax.moe")

    def row_expert_mask(rows, group_sizes):
        ends = jnp.cumsum(group_sizes)
        starts = ends - group_sizes
        idx = jnp.arange(rows)[:, None]
        return (idx >= starts[None, :]) & (idx < ends[None, :])

    def quantize(data, quantizer, group_sizes=None, **kwargs):
        del kwargs
        if captured_d_combined is not None and quantizer == "fc1_dgrad":
            captured_d_combined.append(data)
        return _FakeGroupedTensor(data, group_sizes)

    def gemm(lhs, rhs, *, contracting_dims, bias=None):
        group_sizes = lhs.first_dims
        mask = row_expert_mask(lhs.data.shape[0], group_sizes)
        active = mask.any(axis=-1)
        lhs_clean = jnp.where(active[:, None], lhs.data, 0)
        if contracting_dims == ((0,), (0,)):
            rhs_clean = jnp.where(active[:, None], rhs.data, 0)
            return jnp.einsum(
                "re,rk,rn->ekn",
                mask.astype(lhs.data.dtype),
                lhs_clean,
                rhs_clean,
                preferred_element_type=lhs.data.dtype,
            )
        expert_ids = jnp.argmax(mask, axis=-1)
        rhs_sel = rhs.data[expert_ids]
        if contracting_dims == ((1,), (1,)):
            out = jnp.einsum(
                "rk,rkn->rn", lhs_clean, rhs_sel, preferred_element_type=lhs.data.dtype
            )
            if bias is not None:
                out = out + bias[expert_ids].astype(out.dtype)
        elif contracting_dims == ((1,), (2,)):
            out = jnp.einsum(
                "rn,rkn->rk", lhs_clean, rhs_sel, preferred_element_type=lhs.data.dtype
            )
        else:
            raise AssertionError(f"Unexpected contracting_dims={contracting_dims}")
        fill = jnp.asarray(jnp.nan if poison_inactive else 0, out.dtype)
        return jnp.where(active[:, None], out, fill)

    def dbias(grad, group_sizes):
        mask = row_expert_mask(grad.shape[0], group_sizes)
        active = mask.any(axis=-1)
        grad_clean = jnp.where(active[:, None], grad, 0)
        return jnp.einsum(
            "re,rn->en",
            mask.astype(grad.dtype),
            grad_clean,
            preferred_element_type=grad.dtype,
        )

    def unexpected_fused(*args, **kwargs):
        raise AssertionError("Unfused fallback must not invoke cuDNN fused GLU/dGLU")

    monkeypatch.setattr(module.tex, "grouped_quantize", quantize)
    monkeypatch.setattr(module.tex, "grouped_gemm", gemm)
    monkeypatch.setattr(module.tex, "grouped_dbias", dbias)
    monkeypatch.setattr(module.tex, "grouped_gemm_glu", unexpected_fused)
    monkeypatch.setattr(module.tex, "grouped_gemm_dglu", unexpected_fused)
    return module


def _fake_quantizer_sets():
    fc1 = SimpleNamespace(x="fc1_x", kernel="fc1_k", dgrad="fc1_dgrad")
    fc2 = SimpleNamespace(x="fc2_x", kernel="fc2_k", dgrad="fc2_dgrad")
    return (fc1, fc2)


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

    def activate(g, u):
        return _jax_act_lu(
            jnp.stack([g, u], axis=-2), ("situ", "situ_linear"), act_params=params
        ).data

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
    bf16_out = _jax_act_lu(
        jnp.stack([gate, up], axis=-2).astype(jnp.bfloat16),
        ("situ", "situ_linear"),
        act_params=params,
    ).data
    assert bf16_out.dtype == jnp.bfloat16


@pytest.mark.parametrize("activation_type", ["situglu", "silu"])
def test_unfused_ffn_bf16_uses_fp32_activation_math(monkeypatch, activation_type):
    module = _install_unfused_ffn_doubles(monkeypatch)
    gate = jnp.asarray([0.25, 1.5, -3.0, 12.0, 64.0], dtype=jnp.bfloat16)
    up = jnp.asarray([0.75, -2.0, 32.0, 64.0, -128.0], dtype=jnp.bfloat16)
    tokens = jnp.stack([gate, up], axis=-1)[None, :, :]
    wi = jnp.eye(2, dtype=jnp.bfloat16)[None, :, :]
    wo = jnp.ones((1, 1, 1), dtype=jnp.bfloat16)
    out, residuals = module._ffn_fwd_per_shard(
        tokens,
        jnp.ones((1, gate.size), jnp.float32),
        jnp.asarray([gate.size], jnp.int32),
        wi,
        wo,
        None,
        None,
        None,
        None,
        _fake_quantizer_sets(),
        compute_dtype=jnp.bfloat16,
        num_local_experts=1,
        activation_type=activation_type,
        apply_topk_weights_early=False,
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
    )
    actual = residuals[4].data[:, 0]
    (g, u) = (np.asarray(gate, np.float64), np.asarray(up, np.float64))
    if activation_type == "situglu":
        expected = jnp.asarray(
            100 * np.tanh(g / 4) / (1 + np.exp(-g)) * np.tanh(u / 25), jnp.bfloat16
        )
    else:
        expected = (jax.nn.silu(gate.astype(jnp.float32)) * up.astype(jnp.float32)).astype(
            jnp.bfloat16
        )
    assert out.dtype == jnp.bfloat16
    assert actual.dtype == jnp.bfloat16
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float16, jnp.bfloat16])
@pytest.mark.parametrize("betas", [(4.0, 25.0), (2.0, 8.0)])
@pytest.mark.parametrize("counts", [[3, 0, 5], [0, 0, 0], [0] * 257 + [13], [0]])
def test_situ_masks_inactive_values_and_gradients(monkeypatch, dtype, betas, counts):
    captured_d_combined = []
    module = _install_unfused_ffn_doubles(
        monkeypatch, poison_inactive=True, captured_d_combined=captured_d_combined
    )
    capacity = 0 if counts == [0] else 13
    active = sum(counts)
    num_experts = len(counts)
    inter = 6
    hidden = 2 * inter
    rng = np.random.default_rng(72)
    (gate, up, dy) = [rng.uniform(-5, 5, (capacity, inter)).astype(np.float32) for _ in range(3)]
    for value in (gate, up, dy):
        value[active:] = np.nan
    tokens = jnp.asarray(np.concatenate([gate, up], axis=-1), dtype)[None, :, :]
    d_out = jnp.asarray(np.concatenate([dy, np.zeros_like(dy)], axis=-1), dtype)[None, :, :]
    weights = np.ones((1, capacity), dtype=np.float32)
    weights[:, active:] = np.nan
    weights = jnp.asarray(weights)
    sizes = jnp.asarray(counts, jnp.int32)
    wi = jnp.broadcast_to(jnp.eye(hidden, dtype=dtype), (num_experts, hidden, hidden))
    wo = jnp.broadcast_to(jnp.eye(inter, hidden, dtype=dtype), (num_experts, inter, hidden))
    params = ActivationParams.create("situglu", gate_beta=betas[0], up_beta=betas[1])
    quantizers = _fake_quantizer_sets()

    _, residuals = jax.jit(
        lambda tok, w, cnt: module._ffn_fwd_per_shard(
            tok,
            w,
            cnt,
            wi,
            wo,
            None,
            None,
            None,
            None,
            quantizers,
            compute_dtype=dtype,
            num_local_experts=num_experts,
            activation_type="situglu",
            apply_topk_weights_early=False,
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
    )(tokens, weights, sizes)
    output = residuals[4].data

    def run_bwd(d_eo, res, w):
        d_res = module._ffn_bwd_per_shard(
            d_eo,
            *res,
            w,
            quantizers,
            activation_type="situglu",
            apply_topk_weights_early=False,
            has_bias=False,
            use_cudnn_jax_fusion=False,
            cudnn_native_weight_layout=False,
            act_params=params,
        )
        return d_res, captured_d_combined[-1]

    grads, d_combined = jax.jit(run_bwd)(d_out, residuals, weights)
    dg, du = d_combined[:, :inter], d_combined[:, inter:]
    (g, u, d) = [
        np.asarray(jnp.asarray(value, dtype), np.float64)[:active] for value in (gate, up, dy)
    ]
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
    assert np.isfinite(np.asarray(grads[0][0, :active], np.float32)).all()
    for tensor in grads[1:4]:
        assert np.isfinite(np.asarray(tensor, np.float32)).all()


@pytest.mark.parametrize("apply_topk_weights_early", [False, True])
@pytest.mark.parametrize("has_bias", [False, True])
def test_unfused_situ_ffn_early_weighting_and_bias(monkeypatch, apply_topk_weights_early, has_bias):
    captured_d_combined = []
    module = _install_unfused_ffn_doubles(
        monkeypatch, poison_inactive=True, captured_d_combined=captured_d_combined
    )
    counts = [2, 0, 3]
    active = sum(counts)
    capacity, num_experts, hidden, inter = 8, len(counts), 4, 3
    rng = np.random.default_rng(19)
    tokens = rng.uniform(-1, 1, (1, capacity, hidden)).astype(np.float32)
    weights = rng.uniform(0.1, 0.9, (1, capacity)).astype(np.float32)
    d_out = rng.uniform(-1, 1, (1, capacity, hidden)).astype(np.float32)
    tokens[:, active:] = np.nan
    weights[:, active:] = np.nan
    d_out[:, active:] = np.nan
    tokens, weights, d_out = map(jnp.asarray, (tokens, weights, d_out))
    wi = jnp.asarray(rng.uniform(-0.5, 0.5, (num_experts, hidden, 2 * inter)).astype(np.float32))
    wo = jnp.asarray(rng.uniform(-0.5, 0.5, (num_experts, inter, hidden)).astype(np.float32))
    wi_0_bias = (
        jnp.asarray(rng.uniform(-0.2, 0.2, (num_experts, inter)).astype(np.float32))
        if has_bias
        else None
    )
    wi_1_bias = (
        jnp.asarray(rng.uniform(-0.2, 0.2, (num_experts, inter)).astype(np.float32))
        if has_bias
        else None
    )
    wo_bias = (
        jnp.asarray(rng.uniform(-0.2, 0.2, (num_experts, hidden)).astype(np.float32))
        if has_bias
        else None
    )
    sizes = jnp.asarray(counts, jnp.int32)
    params = ActivationParams.create("situglu", gate_beta=2.5, up_beta=10.0)
    quantizers = _fake_quantizer_sets()

    out, residuals = module._ffn_fwd_per_shard(
        tokens,
        weights,
        sizes,
        wi,
        wo,
        None,
        wi_0_bias,
        wi_1_bias,
        wo_bias,
        quantizers,
        compute_dtype=jnp.float32,
        num_local_experts=num_experts,
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
    grads = module._ffn_bwd_per_shard(
        d_out,
        *residuals,
        weights,
        quantizers,
        activation_type="situglu",
        apply_topk_weights_early=apply_topk_weights_early,
        has_bias=has_bias,
        use_cudnn_jax_fusion=False,
        cudnn_native_weight_layout=False,
        act_params=params,
    )
    d_x, d_w, d_wi, d_wo, d_w0b, d_w1b, d_wob = grads
    intermediate = np.asarray(residuals[4].data)
    d_combined = np.asarray(captured_d_combined[-1])
    assert np.isfinite(intermediate).all()
    assert np.isfinite(d_combined).all()
    np.testing.assert_array_equal(intermediate[active:], 0)
    np.testing.assert_array_equal(d_combined[active:], 0)
    np.testing.assert_array_equal(np.asarray(d_w)[0, active:], 0)

    def ref_loss(x_act, w_act, wi_p, wo_p, b0, b1, bo):
        expert_ids = jnp.repeat(jnp.arange(num_experts), sizes, total_repeat_length=active)
        wi_g, wi_u = jnp.split(wi_p, 2, axis=-1)
        g = jnp.einsum("rk,rkn->rn", x_act, wi_g[expert_ids])
        u = jnp.einsum("rk,rkn->rn", x_act, wi_u[expert_ids])
        if has_bias:
            g = g + b0[expert_ids]
            u = u + b1[expert_ids]
        mid = (2.5 * jnp.tanh(g / 2.5) * jax.nn.sigmoid(g)) * (10.0 * jnp.tanh(u / 10.0))
        if apply_topk_weights_early:
            mid = mid * w_act[:, None]
        y = jnp.einsum("rn,rnh->rh", mid, wo_p[expert_ids])
        if has_bias:
            y = y + bo[expert_ids]
        return jnp.sum(y * d_out[0, :active])

    ref_args = (tokens[0, :active], weights[0, :active], wi, wo, wi_0_bias, wi_1_bias, wo_bias)
    argnums = (0, 1, 2, 3, 4, 5, 6) if has_bias else (0, 1, 2, 3)
    ref_grads = jax.grad(ref_loss, argnums=argnums)(*ref_args)
    np.testing.assert_allclose(d_x[0, :active], ref_grads[0], atol=2e-05, rtol=2e-05)
    np.testing.assert_allclose(d_w[0, :active], ref_grads[1], atol=2e-05, rtol=2e-05)
    np.testing.assert_allclose(d_wi, ref_grads[2], atol=2e-05, rtol=2e-05)
    np.testing.assert_allclose(d_wo, ref_grads[3], atol=2e-05, rtol=2e-05)
    if has_bias:
        np.testing.assert_allclose(d_w0b, ref_grads[4], atol=2e-05, rtol=2e-05)
        np.testing.assert_allclose(d_w1b, ref_grads[5], atol=2e-05, rtol=2e-05)
        np.testing.assert_allclose(d_wob, ref_grads[6], atol=2e-05, rtol=2e-05)
    assert np.isfinite(np.asarray(out[0, :active])).all()


def test_situ_dynamic_counts_remat_and_vmap(monkeypatch):
    module = _install_unfused_ffn_doubles(monkeypatch)
    capacity, num_experts, hidden, inter = 13, 3, 6, 4
    tokens = (
        jnp.arange(capacity * hidden, dtype=jnp.float32).reshape(1, capacity, hidden) / 20 - 1.5
    )
    weights = jnp.ones((1, capacity), dtype=jnp.float32)
    wi = jnp.ones((num_experts, hidden, 2 * inter), dtype=jnp.float32) * 0.25
    wo = jnp.ones((num_experts, inter, hidden), dtype=jnp.float32) * 0.25
    quantizers = _fake_quantizer_sets()

    @jax.custom_vjp
    def run_ffn(tok, w_in, w_out, sizes):
        out, _ = module._ffn_fwd_per_shard(
            tok,
            weights,
            sizes,
            w_in,
            w_out,
            None,
            None,
            None,
            None,
            quantizers,
            compute_dtype=jnp.float32,
            num_local_experts=num_experts,
            activation_type="situglu",
            apply_topk_weights_early=False,
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
        )
        return out

    def run_ffn_fwd(tok, w_in, w_out, sizes):
        return module._ffn_fwd_per_shard(
            tok,
            weights,
            sizes,
            w_in,
            w_out,
            None,
            None,
            None,
            None,
            quantizers,
            compute_dtype=jnp.float32,
            num_local_experts=num_experts,
            activation_type="situglu",
            apply_topk_weights_early=False,
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
        )

    def run_ffn_bwd(res, g_out):
        d_x, _, d_wi, d_wo, _, _, _ = module._ffn_bwd_per_shard(
            g_out,
            *res,
            weights,
            quantizers,
            activation_type="situglu",
            apply_topk_weights_early=False,
            has_bias=False,
            use_cudnn_jax_fusion=False,
            cudnn_native_weight_layout=False,
        )
        return d_x, d_wi, d_wo, None

    run_ffn.defvjp(run_ffn_fwd, run_ffn_bwd)

    compiled = jax.jit(run_ffn)
    for counts in (jnp.array([3, 0, 5], jnp.int32), jnp.array([1, 2, 2], jnp.int32)):
        expected = run_ffn(tokens, wi, wo, counts)
        np.testing.assert_allclose(
            compiled(tokens, wi, wo, counts), expected, atol=2e-05, rtol=2e-05
        )
        loss = lambda tok, w_in, w_out: run_ffn(tok, w_in, w_out, counts).sum()
        gradients = jax.jit(jax.grad(loss, (0, 1, 2)))(tokens, wi, wo)
        remat = jax.jit(jax.grad(jax.checkpoint(loss), (0, 1, 2)))(tokens, wi, wo)
        for actual, reference in zip(remat, gradients):
            np.testing.assert_allclose(actual, reference, atol=2e-05, rtol=2e-05)

    batched_counts = jnp.array([[3, 0, 5], [1, 2, 2]], dtype=jnp.int32)
    batched_tokens = jnp.stack([tokens, -tokens])
    mapped = jax.jit(jax.vmap(lambda tok, cnt: run_ffn(tok, wi, wo, cnt)))(
        batched_tokens, batched_counts
    )
    for actual, tok, sizes in zip(mapped, batched_tokens, batched_counts):
        np.testing.assert_allclose(actual, run_ffn(tok, wi, wo, sizes), atol=2e-05, rtol=2e-05)


@pytest.mark.parametrize(
    "capability,available,expected",
    [
        (89, True, False),
        (100, True, True),
        (100, False, False),
        (103, True, True),
        (107, True, False),
    ],
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
