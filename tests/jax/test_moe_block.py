# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Basic tests for ``transformer_engine.jax.flax.MoEBlock``.

These tests exercise the MoEBlock on a single device (no expert parallelism)
and verify:

* Forward pass runs end-to-end and produces the expected output shape.
* Backward pass yields finite, non-trivial parameter gradients.
* The two permutation backends (``"pure_jax"`` and ``"triton"``) produce
  numerically equivalent outputs and gradients when given the same routing
  decisions.
* Auxiliary load-balancing loss is returned when ``aux_loss_coeff > 0``.
* DeepSeek-style grouped top-k (``num_groups`` / ``group_topk``) runs.
* ``_align_size > 0`` produces numerically-equivalent outputs to ``_align_size = 0``
  for the pure-JAX backend (padding must not change the result).
"""

import sys
from typing import Tuple

import jax
import jax.numpy as jnp
import pytest


# The MoEBlock pulls in both the fused-router CUDA kernel and the Triton
# permutation kernels, so it can only run in the environment where those are
# available. We gate the test on the ``triton`` marker (the Triton permutation
# backend is stricter than the CUDA router). See ``conftest.py``.


@pytest.fixture(autouse=True, scope="function")
def _inject_moe(request):
    """Lazy-load ``MoEBlock`` only for tests marked ``triton``."""
    if not request.node.get_closest_marker("triton"):
        yield
        return

    from transformer_engine.jax.flax import MoEBlock
    from transformer_engine.jax.flax.moe import PermutationBackend

    mod = sys.modules[__name__]
    mod.MoEBlock = MoEBlock
    mod.PermutationBackend = PermutationBackend
    yield


# -----------------------------------------------------------------------------
# Configurations
# -----------------------------------------------------------------------------
#
# Keep shapes small so the tests are cheap but still exercise every code path.

DTYPE = jnp.bfloat16
BATCH_SIZE = 2
SEQUENCE_LENGTH = 16
HIDDEN_SIZE = 64
INTERMEDIATE_SIZE = 128
NUM_EXPERTS = 8
NUM_EXPERTS_PER_TOK = 2


def _make_inputs(
    key: jax.Array, batch_size: int = BATCH_SIZE, sequence_length: int = SEQUENCE_LENGTH
) -> jax.Array:
    return jax.random.normal(key, (batch_size, sequence_length, HIDDEN_SIZE), dtype=DTYPE)


def _init_and_apply(
    block,
    inputs: jax.Array,
    init_key: jax.Array,
) -> Tuple[dict, jax.Array, jax.Array]:
    variables = block.init(init_key, inputs)
    output, aux_loss = block.apply(variables, inputs)
    return variables, output, aux_loss


def _unwrap_partitioned(x):
    """Strip Flax logical-partition wrappers for numeric assertions."""
    return x.value if hasattr(x, "value") else x


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------


@pytest.mark.triton
class TestMoEBlockSingleDevice:
    """Single-device smoke tests for :class:`MoEBlock`."""

    @pytest.mark.parametrize("permutation_backend", ["pure_jax", "triton"])
    def test_forward_shape_and_finite(self, permutation_backend):
        permutation_backend = PermutationBackend(permutation_backend)
        key = jax.random.PRNGKey(0)
        init_key, data_key = jax.random.split(key)

        block = MoEBlock(
            num_experts=NUM_EXPERTS,
            num_experts_per_tok=NUM_EXPERTS_PER_TOK,
            intermediate_size=INTERMEDIATE_SIZE,
            permutation_backend=permutation_backend,
            dtype=DTYPE,
        )
        inputs = _make_inputs(data_key)
        _variables, output, aux_loss = _init_and_apply(block, inputs, init_key)

        assert (
            output.shape == inputs.shape
        ), f"Unexpected output shape {output.shape} for backend {permutation_backend}"
        assert output.dtype == inputs.dtype
        assert jnp.all(jnp.isfinite(output)), "Output contains NaN/Inf"
        assert aux_loss is None, "aux_loss should be None when aux_loss_coeff=0"

    @pytest.mark.parametrize("permutation_backend", ["pure_jax", "triton"])
    def test_backward_grad_is_finite_and_nonzero(self, permutation_backend):
        permutation_backend = PermutationBackend(permutation_backend)
        key = jax.random.PRNGKey(1)
        init_key, data_key = jax.random.split(key)

        block = MoEBlock(
            num_experts=NUM_EXPERTS,
            num_experts_per_tok=NUM_EXPERTS_PER_TOK,
            intermediate_size=INTERMEDIATE_SIZE,
            permutation_backend=permutation_backend,
            dtype=DTYPE,
        )
        inputs = _make_inputs(data_key)
        variables = block.init(init_key, inputs)

        def loss_fn(variables, inputs):
            output, _ = block.apply(variables, inputs)
            return jnp.mean(output.astype(jnp.float32) ** 2)

        grads = jax.grad(loss_fn)(variables, inputs)
        # All trainable kernels should receive a non-trivial gradient.
        for name in ("gate_kernel", "wi_0", "wi_1", "wo"):
            g = _unwrap_partitioned(grads["params"][name])
            assert jnp.all(jnp.isfinite(g)), f"{name} gradient has NaN/Inf"
            assert jnp.any(g != 0.0), f"{name} gradient is identically zero"

    def test_pure_jax_triton_equivalence(self):
        """Both permutation backends must produce the same forward + grads
        under identical routing decisions.

        Since the two backends share the same routing path (TE's fused
        top-k), fixing the gate kernel gives both the same routing decisions
        and the remainder of the network is identical modulo the permutation
        implementation, whose semantics are equivalent.
        """
        key = jax.random.PRNGKey(2)
        init_key, data_key = jax.random.split(key)

        base_kwargs = dict(
            num_experts=NUM_EXPERTS,
            num_experts_per_tok=NUM_EXPERTS_PER_TOK,
            intermediate_size=INTERMEDIATE_SIZE,
            dtype=DTYPE,
        )
        pure_block = MoEBlock(permutation_backend=PermutationBackend.PURE_JAX, **base_kwargs)
        triton_block = MoEBlock(permutation_backend=PermutationBackend.TRITON, **base_kwargs)
        inputs = _make_inputs(data_key)

        # Share a single parameter tree so routing decisions and expert
        # weights are identical for both backends.
        variables = pure_block.init(init_key, inputs)

        def loss_fn(block, variables, inputs):
            output, _ = block.apply(variables, inputs)
            return jnp.mean(output.astype(jnp.float32) ** 2), output

        (loss_pj, out_pj), grads_pj = jax.value_and_grad(loss_fn, argnums=1, has_aux=True)(
            pure_block, variables, inputs
        )
        (loss_tr, out_tr), grads_tr = jax.value_and_grad(loss_fn, argnums=1, has_aux=True)(
            triton_block, variables, inputs
        )

        # BF16 tolerances: outputs come out of the grouped-GEMM + weighted
        # sum so they accumulate error; we use ~2 ULPs worth of slack.
        atol_out, rtol_out = 5e-2, 5e-2
        assert jnp.allclose(
            out_pj, out_tr, atol=atol_out, rtol=rtol_out
        ), f"Forward outputs differ across backends: max diff {jnp.max(jnp.abs(out_pj - out_tr))}"
        assert jnp.allclose(loss_pj, loss_tr, atol=atol_out, rtol=rtol_out)

        # The two backends share the routing path (same fused top-k) and
        # the same expert FFN; the only difference is the order of the
        # gather + scatter ops in dispatch/combine. Under bf16 with these
        # small shapes, observed grad max-abs-diff is on the order of a
        # few-units-of-bf16-eps (~1e-2). 5e-2 / 5e-2 leaves headroom for
        # accumulation jitter without masking real divergence. If this
        # tightens too far on a particular GPU, print
        # ``jnp.max(jnp.abs(g_pj - g_tr))`` from the failing assertion
        # and bump to the next safe value with a comment recording the
        # measured gap.
        atol_grad, rtol_grad = 5e-2, 5e-2
        for name in ("gate_kernel", "wi_0", "wi_1", "wo"):
            g_pj = _unwrap_partitioned(grads_pj["params"][name])
            g_tr = _unwrap_partitioned(grads_tr["params"][name])
            assert jnp.allclose(g_pj, g_tr, atol=atol_grad, rtol=rtol_grad), (
                f"Gradient for {name} differs across backends: max diff"
                f" {jnp.max(jnp.abs(g_pj - g_tr))} (atol={atol_grad},"
                f" rtol={rtol_grad})"
            )

    @pytest.mark.parametrize("permutation_backend", ["pure_jax", "triton"])
    def test_aux_loss_returned(self, permutation_backend):
        permutation_backend = PermutationBackend(permutation_backend)
        key = jax.random.PRNGKey(3)
        init_key, data_key = jax.random.split(key)

        block = MoEBlock(
            num_experts=NUM_EXPERTS,
            num_experts_per_tok=NUM_EXPERTS_PER_TOK,
            intermediate_size=INTERMEDIATE_SIZE,
            permutation_backend=permutation_backend,
            aux_loss_coeff=1e-2,
            dtype=DTYPE,
        )
        inputs = _make_inputs(data_key)
        _variables, output, aux_loss = _init_and_apply(block, inputs, init_key)

        assert output.shape == inputs.shape
        assert aux_loss is not None, "aux_loss should be returned when coeff > 0"
        assert aux_loss.shape == (), "aux_loss should be a scalar"
        assert jnp.isfinite(aux_loss)
        # With uniform-ish routing the loss should be small-positive, not huge.
        assert jnp.abs(aux_loss) < 1e2

    def test_aux_loss_uses_real_routing_under_group_topk(self):
        """Aux loss must reflect the real (post-group) routing decisions.

        Under DeepSeek-style ``num_groups`` / ``group_topk`` routing,
        the auxiliary load-balancing loss must be computed using the
        per-expert token counts from the *real* routing_map (post
        grouping), not from the clean top-k that the
        ``compute_aux_scores=True`` kernel returns. Otherwise the aux
        objective trains against the wrong distribution.

        We compute three values:
          * ``corrected_ref`` -- ``fused_moe_aux_loss(aux_scores,
            tokens_from_real_routing_map, ...)`` (what the block
            should produce after the fix).
          * ``buggy_ref`` -- ``fused_moe_aux_loss(aux_scores,
            tokens_from_aux_routing_map, ...)`` (what the block used
            to produce before the fix).
          * ``block_aux_loss`` -- what the block actually produces.

        Block must match the corrected reference. We also assert that
        the corrected and buggy references differ for this config so
        the test is not vacuously satisfied by them coinciding.
        """
        from transformer_engine.jax.router import (
            fused_moe_aux_loss,
            fused_topk_with_score_function,
        )

        key = jax.random.PRNGKey(7)
        init_key, data_key = jax.random.split(key)

        # Pick a config that *reliably* exercises grouped-vs-clean
        # divergence: with ``group_topk=1`` only ONE group's experts
        # can be selected by grouped routing, so the routing diverges
        # from a plain top-k whenever the global top-K experts are
        # spread across multiple groups (which is almost always the
        # case for random init + ``num_experts_per_tok > 1``).
        num_groups = 2
        group_topk = 1
        aux_loss_coeff = 1e-2

        block = MoEBlock(
            num_experts=NUM_EXPERTS,
            num_experts_per_tok=NUM_EXPERTS_PER_TOK,
            intermediate_size=INTERMEDIATE_SIZE,
            permutation_backend=PermutationBackend.PURE_JAX,
            score_function="sigmoid",
            num_groups=num_groups,
            group_topk=group_topk,
            aux_loss_coeff=aux_loss_coeff,
            dtype=DTYPE,
        )
        inputs = _make_inputs(data_key)
        variables = block.init(init_key, inputs)
        _output, block_aux_loss = block.apply(variables, inputs)

        assert block_aux_loss is not None

        # Reproduce the gating GEMM and routing externally so we can
        # build the references against the same logits the block sees.
        gate_kernel = _unwrap_partitioned(variables["params"]["gate_kernel"])
        gate_kernel = gate_kernel.astype(inputs.dtype)
        logits = jnp.einsum("bsh,he->bse", inputs, gate_kernel)
        logits_2d = logits.reshape(-1, NUM_EXPERTS)

        # Real routing (with grouping). This is what _route_topk
        # would produce inside the block.
        _, real_routing_map = fused_topk_with_score_function(
            logits_2d,
            topk=NUM_EXPERTS_PER_TOK,
            score_function="sigmoid",
            num_groups=num_groups,
            group_topk=group_topk,
        )
        real_tokens = jnp.sum(real_routing_map.astype(jnp.int32), axis=0)

        # Aux scores + the (clean topk) aux_routing_map that the old
        # buggy code used for tokens_per_expert.
        aux_scores, aux_routing_map = fused_topk_with_score_function(
            logits_2d.astype(jnp.float32),
            topk=NUM_EXPERTS_PER_TOK,
            score_function="sigmoid",
            compute_aux_scores=True,
        )
        buggy_tokens = jnp.sum(aux_routing_map.astype(jnp.int32), axis=0)

        corrected_ref = fused_moe_aux_loss(
            aux_scores.astype(jnp.float32),
            real_tokens,
            topk=NUM_EXPERTS_PER_TOK,
            coeff=aux_loss_coeff,
        )
        buggy_ref = fused_moe_aux_loss(
            aux_scores.astype(jnp.float32),
            buggy_tokens,
            topk=NUM_EXPERTS_PER_TOK,
            coeff=aux_loss_coeff,
        )

        # Sanity: the test config must actually exercise the bug
        # (otherwise both references coincide and the assertion below
        # would silently pass even with the old code).
        assert not jnp.allclose(real_tokens, buggy_tokens), (
            "Test config does not exercise grouped-topk vs clean-topk"
            " divergence; pick a config where they differ"
        )

        assert jnp.allclose(
            block_aux_loss, corrected_ref, atol=1e-5, rtol=1e-5
        ), f"Block aux_loss {block_aux_loss} does not match real-routing reference {corrected_ref}"
        # The corrected and buggy refs can be numerically close
        # (only the mis-routed tokens contribute to the difference),
        # so assert that the block is *strictly closer* to the
        # corrected ref than to the buggy one. This catches the
        # regression robustly even when the absolute gap between
        # corrected_ref and buggy_ref is sub-tolerance.
        diff_to_corrected = jnp.abs(block_aux_loss - corrected_ref)
        diff_to_buggy = jnp.abs(block_aux_loss - buggy_ref)
        gap = jnp.abs(corrected_ref - buggy_ref)
        assert diff_to_corrected < diff_to_buggy, (
            f"Block aux_loss {block_aux_loss} is closer to the *old"
            f" buggy* reference ({buggy_ref}, diff={diff_to_buggy})"
            f" than to the corrected reference ({corrected_ref},"
            f" diff={diff_to_corrected}); the regression has"
            f" reappeared. corrected-buggy gap = {gap}"
        )

    @pytest.mark.parametrize("permutation_backend", ["pure_jax", "triton"])
    def test_group_topk_deepseek(self, permutation_backend):
        """Exercise DeepSeek-style grouped top-k routing."""
        permutation_backend = PermutationBackend(permutation_backend)
        key = jax.random.PRNGKey(4)
        init_key, data_key = jax.random.split(key)

        # num_groups must divide num_experts.
        num_groups = 4
        group_topk = 2
        block = MoEBlock(
            num_experts=NUM_EXPERTS,
            num_experts_per_tok=NUM_EXPERTS_PER_TOK,
            intermediate_size=INTERMEDIATE_SIZE,
            permutation_backend=permutation_backend,
            score_function="sigmoid",
            num_groups=num_groups,
            group_topk=group_topk,
            dtype=DTYPE,
        )
        inputs = _make_inputs(data_key)
        _variables, output, _aux_loss = _init_and_apply(block, inputs, init_key)

        assert output.shape == inputs.shape
        assert jnp.all(jnp.isfinite(output))

    def test_align_size_equivalence_pure_jax(self, monkeypatch):
        """For the pure-JAX backend, ``_align_size > 0`` must not change the
        numerical output of the forward pass: padding tokens contribute zero
        to every expert GEMM output (their input rows are zeros) and are
        stripped before the weighted sum.

        Why the env knob: the V1 TE grouped GEMM FFI asserts strict
        equality ``sum(group_sizes) == M``. With ``_align_size > 0`` the
        pure-JAX backend produces a buffer where ``M >= sum(group_sizes)``
        (the slack is structural padding for JIT), so V1 is incompatible.
        The V2 cuBLASLt-backed grouped GEMM relaxes the assertion to
        ``M >= sum(group_sizes)`` and is selected when
        ``NVTE_JAX_ENFORCE_V2_GROUPED_GEMM=1``. If V2 isn't supported on
        this hardware / for this dtype, the dispatch raises a
        ``RuntimeError`` whose message is matched here so the test
        ``skip``-s instead of failing.
        """
        monkeypatch.setenv("NVTE_JAX_ENFORCE_V2_GROUPED_GEMM", "1")

        key = jax.random.PRNGKey(5)
        init_key, data_key = jax.random.split(key)

        base_kwargs = dict(
            num_experts=NUM_EXPERTS,
            num_experts_per_tok=NUM_EXPERTS_PER_TOK,
            intermediate_size=INTERMEDIATE_SIZE,
            permutation_backend=PermutationBackend.PURE_JAX,
            dtype=DTYPE,
        )
        block_no_pad = MoEBlock(_align_size=0, **base_kwargs)
        block_pad = MoEBlock(_align_size=16, **base_kwargs)
        inputs = _make_inputs(data_key)

        try:
            variables = block_no_pad.init(init_key, inputs)
            out_no_pad, _ = block_no_pad.apply(variables, inputs)
            out_pad, _ = block_pad.apply(variables, inputs)
        except RuntimeError as exc:
            if "V2 grouped GEMM is not supported" in str(exc):
                pytest.skip(f"V2 grouped GEMM unavailable on this hardware: {exc}")
            raise

        assert jnp.allclose(out_no_pad, out_pad, atol=5e-2, rtol=5e-2), (
            "_align_size > 0 must not change pure_jax forward output; max diff"
            f" {jnp.max(jnp.abs(out_no_pad - out_pad))}"
        )

    @pytest.mark.parametrize("permutation_backend", ["pure_jax", "triton"])
    def test_jit_and_determinism(self, permutation_backend):
        """The block must be JIT-compilable and produce a deterministic
        forward pass across repeat calls with the same params."""
        permutation_backend = PermutationBackend(permutation_backend)
        key = jax.random.PRNGKey(6)
        init_key, data_key = jax.random.split(key)

        block = MoEBlock(
            num_experts=NUM_EXPERTS,
            num_experts_per_tok=NUM_EXPERTS_PER_TOK,
            intermediate_size=INTERMEDIATE_SIZE,
            permutation_backend=permutation_backend,
            dtype=DTYPE,
        )
        inputs = _make_inputs(data_key)
        variables = block.init(init_key, inputs)

        @jax.jit
        def forward(variables, inputs):
            return block.apply(variables, inputs)[0]

        out_a = forward(variables, inputs)
        out_b = forward(variables, inputs)
        assert jnp.array_equal(out_a, out_b), "JITted forward is non-deterministic"
