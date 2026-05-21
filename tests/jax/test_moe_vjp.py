# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Single-device tests for the unified MoE custom_vjp at
``transformer_engine.jax.moe.moe`` (and its Flax wrapper
``transformer_engine.jax.flax._MoEBlock``).

Strategy
--------

Rather than reproducing every internal kernel residual, we rely on a
single end-to-end pure-JAX *reference* implementation of the whole
MoE block (``_pure_jax_moe_reference`` below) and compare the TE
``moe(...)`` forward output AND parameter gradients against it. This
gives us coverage of:

* the gate GEMM,
* the fused top-k routing primitive (and its bwd),
* the dispatch / per-expert FFN / combine pipeline (and their bwds
  threaded through the absorbed primitives),
* the optional aux-loss path (and its bwd).

The reference uses only ``jnp`` ops + ``jax.vjp``, so we get a
"definitive" pullback to compare against without needing the TE
primitive bwd kernels.

Distributed (EP + FSDP) testing is intentionally NOT in this file --
that needs a multi-device setup and lives in
``tests/jax/test_distributed_moe_vjp.py`` (follow-up).
"""

from functools import partial
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest


# Lazy import (mirrors the gating in the old test file): the underlying
# kernels require triton + the fused-router CUDA kernel.
@pytest.fixture(autouse=True, scope="function")
def _inject_moe(request):
    if not request.node.get_closest_marker("triton"):
        yield
        return
    import sys
    from transformer_engine.jax.flax import _MoEBlock as MoEBlock
    from transformer_engine.jax.moe import PermutationBackend, moe

    mod = sys.modules[__name__]
    mod.MoEBlock = MoEBlock
    mod.PermutationBackend = PermutationBackend
    mod.moe = moe
    yield


# -----------------------------------------------------------------------------
# Test config
# -----------------------------------------------------------------------------

DTYPE = jnp.float32  # use fp32 for tighter parity assertions
BATCH_SIZE = 2
SEQUENCE_LENGTH = 16
HIDDEN_SIZE = 32
INTERMEDIATE_SIZE = 64
NUM_EXPERTS = 8
NUM_EXPERTS_PER_TOK = 2


def _make_inputs(key: jax.Array, *, batch=BATCH_SIZE, seq=SEQUENCE_LENGTH) -> jax.Array:
    return jax.random.normal(key, (batch, seq, HIDDEN_SIZE), dtype=DTYPE)


# -----------------------------------------------------------------------------
# Pure-JAX reference MoE
# -----------------------------------------------------------------------------
#
# Implements EXACTLY the same math as ``moe(...)`` for the no-EP,
# softmax-routing, no-bias, silu activation, no-quantization path.
# Returns ``(output, aux_loss_or_zero)``. Used as ground truth for both
# fwd and bwd parity.


@partial(
    jax.jit,
    static_argnames=("num_experts", "num_experts_per_tok", "aux_loss_coeff"),
)
def _pure_jax_moe_reference(
    x: jnp.ndarray,
    gate_kernel: jnp.ndarray,
    wi_0: jnp.ndarray,
    wi_1: jnp.ndarray,
    wo: jnp.ndarray,
    *,
    num_experts: int,
    num_experts_per_tok: int,
    aux_loss_coeff: float = 0.0,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Reference no-EP MoE forward (pure JAX, no TE primitives).

    Mirrors :func:`transformer_engine.jax.moe._body_fwd` for the
    PURE_JAX backend, no biases, softmax routing, silu activation,
    no quantization. Linear ops only -- ``jax.vjp`` over this gives
    the canonical bwd to compare against.
    """
    B, S, H = x.shape
    T = B * S
    x_2d = x.reshape(T, H)

    # Gate
    logits = x_2d @ gate_kernel  # [T, E]

    # Softmax + topk (no expert_bias, no grouping, scale=1.0)
    probs_full = jax.nn.softmax(logits, axis=-1)  # [T, E]
    # top-k by probability:
    sorted_idx = jnp.argsort(probs_full, axis=-1)  # ascending
    selected = sorted_idx[:, -num_experts_per_tok:]  # [T, K]
    weights = jnp.take_along_axis(probs_full, selected, axis=-1)  # [T, K]
    # Normalize topk weights to sum to 1 (matches softmax->topk semantics
    # of fused_topk_with_score_function with use_pre_softmax=False):
    weights = weights / jnp.sum(weights, axis=-1, keepdims=True)

    # Build a sparse routing_map [T, E] with weights at selected positions
    routing_weights_full = jnp.zeros_like(probs_full)
    routing_weights_full = routing_weights_full.at[jnp.arange(T)[:, None], selected].set(weights)

    # Per-expert FFN: replicate each token K times, gather by expert,
    # run through wi_0 / wi_1 / wo, gather back, weighted-sum.
    #
    # Vectorize the gather without sorting: for each (token, slot k),
    # multiply the corresponding expert's FFN by routing_weights[t, k]
    # and sum over experts.
    # x_2d: [T, H], wi_0: [E, H, M], wi_1: [E, H, M], wo: [E, M, H]
    # For each expert e: layer_w0_e = x_2d @ wi_0[e]; layer_w1_e = x_2d @ wi_1[e]
    #                    intermediate_e = silu(layer_w0_e) * layer_w1_e
    #                    expert_out_e = intermediate_e @ wo[e]
    # output[t, h] = sum_e routing_weights_full[t, e] * expert_out_e[t, h]
    layer_w0 = jnp.einsum("th,ehm->tem", x_2d, wi_0)  # [T, E, M]
    layer_w1 = jnp.einsum("th,ehm->tem", x_2d, wi_1)  # [T, E, M]
    intermediate = jax.nn.silu(layer_w0) * layer_w1  # [T, E, M]
    expert_out = jnp.einsum("tem,emh->teh", intermediate, wo)  # [T, E, H]
    output_2d = jnp.einsum("te,teh->th", routing_weights_full, expert_out)  # [T, H]
    output = output_2d.reshape(B, S, H)

    if aux_loss_coeff > 0.0:
        # aux scores: clean per-expert softmax (compute_aux_scores=True
        # kernel uses a clean softmax, no bias, scale=1, no grouping).
        aux_probs = jax.nn.softmax(logits.astype(jnp.float32), axis=-1)
        # tokens_per_expert from REAL routing_map (post-grouping); here
        # there's no grouping so == count of non-zero positions per expert.
        routing_map = (routing_weights_full > 0).astype(jnp.int32)
        tokens_per_expert = jnp.sum(routing_map, axis=0)  # [E]
        # aux_loss formula: (E * coeff / (k * T^2)) * sum_e
        # (sum_t aux_probs[t, e]) * tokens_per_expert[e]
        sum_probs_per_expert = jnp.sum(aux_probs, axis=0)  # [E]
        aux_loss = (num_experts * aux_loss_coeff / (num_experts_per_tok * (T**2))) * jnp.sum(
            sum_probs_per_expert * tokens_per_expert.astype(jnp.float32)
        )
    else:
        aux_loss = jnp.zeros((), dtype=DTYPE)

    return output, aux_loss


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def _init_params(key: jax.Array) -> dict:
    k_g, k_w0, k_w1, k_wo = jax.random.split(key, 4)
    init = jax.nn.initializers.variance_scaling(1.0, "fan_in", "truncated_normal")
    return dict(
        gate_kernel=init(k_g, (HIDDEN_SIZE, NUM_EXPERTS), DTYPE),
        wi_0=init(k_w0, (NUM_EXPERTS, HIDDEN_SIZE, INTERMEDIATE_SIZE), DTYPE),
        wi_1=init(k_w1, (NUM_EXPERTS, HIDDEN_SIZE, INTERMEDIATE_SIZE), DTYPE),
        wo=init(k_wo, (NUM_EXPERTS, INTERMEDIATE_SIZE, HIDDEN_SIZE), DTYPE),
    )


@partial(jax.jit, static_argnames=("permutation_backend", "aux_loss_coeff"))
def _run_te_moe(
    x: jnp.ndarray,
    params: dict,
    *,
    permutation_backend,
    aux_loss_coeff: float = 0.0,
) -> Tuple[jnp.ndarray, Optional[jnp.ndarray]]:
    return moe(  # noqa: F821 -- injected by fixture
        x,
        params["gate_kernel"],
        params["wi_0"],
        params["wi_1"],
        params["wo"],
        num_experts=NUM_EXPERTS,
        num_experts_per_tok=NUM_EXPERTS_PER_TOK,
        activation_type="silu",
        score_function="softmax",
        use_pre_softmax=False,
        scaling_factor=1.0,
        aux_loss_coeff=aux_loss_coeff,
        permutation_backend=permutation_backend,
        align_size=0,
        dtype=DTYPE,
    )


@partial(jax.jit, static_argnames=("permutation_backend", "aux_loss_coeff"))
def _grads_te_main_loss(params, x, *, permutation_backend, aux_loss_coeff: float = 0.0):
    """jit'd grad of ``mean(out**2)`` w.r.t. params (no aux contribution)."""

    def loss(params, x):
        out, _ = _run_te_moe(
            x, params, permutation_backend=permutation_backend, aux_loss_coeff=aux_loss_coeff
        )
        return jnp.mean(out**2)

    return jax.grad(loss)(params, x)


@partial(jax.jit, static_argnames=("num_experts", "num_experts_per_tok", "aux_loss_coeff"))
def _grads_ref_main_loss(params, x, *, num_experts, num_experts_per_tok, aux_loss_coeff=0.0):
    """jit'd grad of ``mean(out**2)`` w.r.t. params on the pure-JAX ref."""

    def loss(params, x):
        out, _ = _pure_jax_moe_reference(
            x,
            **params,
            num_experts=num_experts,
            num_experts_per_tok=num_experts_per_tok,
            aux_loss_coeff=aux_loss_coeff,
        )
        return jnp.mean(out**2)

    return jax.grad(loss)(params, x)


@partial(jax.jit, static_argnames=("permutation_backend",))
def _grad_te_aux_only(params, x, *, permutation_backend):
    """jit'd grad of just the aux loss scalar (no main contribution)."""

    def aux_only(params, x):
        _, aux = _run_te_moe(
            x, params, permutation_backend=permutation_backend, aux_loss_coeff=1e-2
        )
        return aux.astype(jnp.float32)

    return jax.grad(aux_only)(params, x)


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------


@pytest.mark.triton
class TestMoeVjpForward:
    """Forward shape / finiteness / parity vs pure-JAX reference."""

    @pytest.mark.parametrize("backend_name", ["pure_jax", "triton"])
    def test_forward_shape_and_finite(self, backend_name):
        backend = PermutationBackend(backend_name)  # noqa: F821
        key = jax.random.PRNGKey(0)
        kp, kx = jax.random.split(key)
        params = _init_params(kp)
        x = _make_inputs(kx)
        out, aux = _run_te_moe(x, params, permutation_backend=backend)
        assert out.shape == x.shape
        assert out.dtype == x.dtype
        assert jnp.all(jnp.isfinite(out))
        assert aux is None

    @pytest.mark.parametrize("backend_name", ["pure_jax", "triton"])
    def test_forward_parity_vs_pure_jax_reference(self, backend_name):
        backend = PermutationBackend(backend_name)  # noqa: F821
        key = jax.random.PRNGKey(1)
        kp, kx = jax.random.split(key)
        params = _init_params(kp)
        x = _make_inputs(kx)
        out_te, _ = _run_te_moe(x, params, permutation_backend=backend)
        out_ref, _ = _pure_jax_moe_reference(
            x,
            **params,
            num_experts=NUM_EXPERTS,
            num_experts_per_tok=NUM_EXPERTS_PER_TOK,
        )
        # FP32, small shapes -> tight tolerance
        np.testing.assert_allclose(np.array(out_te), np.array(out_ref), atol=2e-5, rtol=2e-5)

    def test_pure_jax_triton_equivalence(self):
        key = jax.random.PRNGKey(2)
        kp, kx = jax.random.split(key)
        params = _init_params(kp)
        x = _make_inputs(kx)
        out_pj, _ = _run_te_moe(
            x, params, permutation_backend=PermutationBackend.PURE_JAX  # noqa: F821
        )
        out_tr, _ = _run_te_moe(
            x, params, permutation_backend=PermutationBackend.TRITON  # noqa: F821
        )
        np.testing.assert_allclose(np.array(out_pj), np.array(out_tr), atol=2e-5, rtol=2e-5)


@pytest.mark.triton
class TestMoeVjpBackward:
    """Backward parity vs pure-JAX reference (which uses ``jax.vjp`` over
    plain JAX ops, giving us the canonical pullback)."""

    @pytest.mark.parametrize("backend_name", ["pure_jax", "triton"])
    def test_grads_finite_and_nonzero(self, backend_name):
        backend = PermutationBackend(backend_name)  # noqa: F821
        key = jax.random.PRNGKey(3)
        kp, kx = jax.random.split(key)
        params = _init_params(kp)
        x = _make_inputs(kx)
        grads = _grads_te_main_loss(params, x, permutation_backend=backend)
        for name in ("gate_kernel", "wi_0", "wi_1", "wo"):
            g = grads[name]
            assert jnp.all(jnp.isfinite(g)), f"{name} grad has NaN/Inf"
            assert jnp.any(g != 0.0), f"{name} grad is identically zero"

    @pytest.mark.parametrize("backend_name", ["pure_jax", "triton"])
    def test_grads_match_pure_jax_reference(self, backend_name):
        backend = PermutationBackend(backend_name)  # noqa: F821
        key = jax.random.PRNGKey(4)
        kp, kx = jax.random.split(key)
        params = _init_params(kp)
        x = _make_inputs(kx)
        grads_te = _grads_te_main_loss(params, x, permutation_backend=backend)
        grads_ref = _grads_ref_main_loss(
            params,
            x,
            num_experts=NUM_EXPERTS,
            num_experts_per_tok=NUM_EXPERTS_PER_TOK,
        )
        # Loose-ish tol on grads: routing path has discrete topk so the
        # softmax cotangent paths through the non-topk experts diverge
        # slightly between TE (which uses the fused topk bwd) and the
        # reference (which uses argsort-based take_along_axis).
        # Tighter than the bf16 tests.
        for name in ("wi_0", "wi_1", "wo"):
            np.testing.assert_allclose(
                np.array(grads_te[name]),
                np.array(grads_ref[name]),
                atol=5e-5,
                rtol=5e-5,
                err_msg=f"grad mismatch on {name}",
            )
        # Gate grad has more error budget because it propagates through
        # the topk derivative kernel (which differs in zero-pattern
        # treatment from a plain take_along_axis).
        np.testing.assert_allclose(
            np.array(grads_te["gate_kernel"]),
            np.array(grads_ref["gate_kernel"]),
            atol=5e-4,
            rtol=5e-4,
            err_msg="grad mismatch on gate_kernel",
        )


@pytest.mark.triton
class TestMoeVjpAuxLoss:
    """Aux-loss path: forward + grad parity."""

    @pytest.mark.parametrize("backend_name", ["pure_jax", "triton"])
    def test_aux_loss_returned_and_finite(self, backend_name):
        backend = PermutationBackend(backend_name)  # noqa: F821
        key = jax.random.PRNGKey(5)
        kp, kx = jax.random.split(key)
        params = _init_params(kp)
        x = _make_inputs(kx)
        _, aux = _run_te_moe(x, params, permutation_backend=backend, aux_loss_coeff=1e-2)
        assert aux is not None
        assert aux.shape == ()
        assert jnp.isfinite(aux)
        assert jnp.abs(aux) < 1e2

    @pytest.mark.parametrize("backend_name", ["pure_jax", "triton"])
    def test_aux_loss_parity_vs_reference(self, backend_name):
        backend = PermutationBackend(backend_name)  # noqa: F821
        key = jax.random.PRNGKey(6)
        kp, kx = jax.random.split(key)
        params = _init_params(kp)
        x = _make_inputs(kx)
        _, aux_te = _run_te_moe(x, params, permutation_backend=backend, aux_loss_coeff=1e-2)
        _, aux_ref = _pure_jax_moe_reference(
            x,
            **params,
            num_experts=NUM_EXPERTS,
            num_experts_per_tok=NUM_EXPERTS_PER_TOK,
            aux_loss_coeff=1e-2,
        )
        np.testing.assert_allclose(float(aux_te), float(aux_ref), atol=1e-5, rtol=1e-5)

    @pytest.mark.parametrize("backend_name", ["pure_jax", "triton"])
    def test_aux_loss_grads_propagate_to_logits(self, backend_name):
        """The aux-loss bwd path must produce non-zero gate-kernel grads
        when only the aux-loss scalar is differentiated (no main-output
        contribution)."""
        backend = PermutationBackend(backend_name)  # noqa: F821
        key = jax.random.PRNGKey(7)
        kp, kx = jax.random.split(key)
        params = _init_params(kp)
        x = _make_inputs(kx)
        g_gate = _grad_te_aux_only(params, x, permutation_backend=backend)["gate_kernel"]
        assert jnp.all(jnp.isfinite(g_gate))
        assert jnp.any(
            g_gate != 0.0
        ), "aux_loss bwd should propagate to gate_kernel via fused_topk bwd"


# -----------------------------------------------------------------------------
# Flax wrapper smoke test
# -----------------------------------------------------------------------------


@pytest.mark.triton
class TestMoEBlockFlaxWrapper:
    """Sanity-check the thin Flax wrapper: forward + grad on init."""

    def test_init_and_apply(self):
        block = MoEBlock(  # noqa: F821
            num_experts=NUM_EXPERTS,
            num_experts_per_tok=NUM_EXPERTS_PER_TOK,
            intermediate_size=INTERMEDIATE_SIZE,
            permutation_backend=PermutationBackend.PURE_JAX,  # noqa: F821
            dtype=DTYPE,
        )
        key = jax.random.PRNGKey(8)
        ki, kx = jax.random.split(key)
        x = _make_inputs(kx)
        variables = jax.jit(block.init)(ki, x)
        out, aux = jax.jit(block.apply)(variables, x)
        assert out.shape == x.shape
        assert aux is None

        @jax.jit
        def grad_fn(variables, x):
            return jax.grad(lambda v, x: jnp.mean(block.apply(v, x)[0] ** 2))(variables, x)

        grads = grad_fn(variables, x)
        for name in ("gate_kernel", "wi_0", "wi_1", "wo"):
            g = grads["params"][name]
            g = g.value if hasattr(g, "value") else g
            assert jnp.all(jnp.isfinite(g)), f"{name} grad NaN/Inf"
            assert jnp.any(g != 0.0), f"{name} grad zero"
