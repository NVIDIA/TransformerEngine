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
* ``align_size > 0`` produces numerically-equivalent outputs to ``align_size = 0``
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

    mod = sys.modules[__name__]
    mod.MoEBlock = MoEBlock
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
    return jax.random.normal(
        key, (batch_size, sequence_length, HIDDEN_SIZE), dtype=DTYPE
    )


def _init_and_apply(
    block,
    inputs: jax.Array,
    init_key: jax.Array,
) -> Tuple[dict, jax.Array, jax.Array]:
    variables = block.init(init_key, inputs)
    output, aux_loss = block.apply(variables, inputs)
    return variables, output, aux_loss


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------


@pytest.mark.triton
class TestMoEBlockSingleDevice:
    """Single-device smoke tests for :class:`MoEBlock`."""

    @pytest.mark.parametrize("permutation_backend", ["pure_jax", "triton"])
    def test_forward_shape_and_finite(self, permutation_backend):
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

        assert output.shape == inputs.shape, (
            f"Unexpected output shape {output.shape} for backend {permutation_backend}"
        )
        assert output.dtype == inputs.dtype
        assert jnp.all(jnp.isfinite(output)), "Output contains NaN/Inf"
        assert aux_loss is None, "aux_loss should be None when aux_loss_coeff=0"

    @pytest.mark.parametrize("permutation_backend", ["pure_jax", "triton"])
    def test_backward_grad(self, permutation_backend):
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
            g = grads["params"][name]
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
        pure_block = MoEBlock(permutation_backend="pure_jax", **base_kwargs)
        triton_block = MoEBlock(permutation_backend="triton", **base_kwargs)
        inputs = _make_inputs(data_key)

        # Share a single parameter tree so routing decisions and expert
        # weights are identical for both backends.
        variables = pure_block.init(init_key, inputs)

        def loss_fn(block, variables, inputs):
            output, _ = block.apply(variables, inputs)
            return jnp.mean(output.astype(jnp.float32) ** 2), output

        (loss_pj, out_pj), grads_pj = jax.value_and_grad(
            loss_fn, argnums=1, has_aux=True
        )(pure_block, variables, inputs)
        (loss_tr, out_tr), grads_tr = jax.value_and_grad(
            loss_fn, argnums=1, has_aux=True
        )(triton_block, variables, inputs)

        # BF16 tolerances: outputs come out of the grouped-GEMM + weighted
        # sum so they accumulate error; we use ~2 ULPs worth of slack.
        atol_out, rtol_out = 5e-2, 5e-2
        assert jnp.allclose(out_pj, out_tr, atol=atol_out, rtol=rtol_out), (
            f"Forward outputs differ across backends: max diff"
            f" {jnp.max(jnp.abs(out_pj - out_tr))}"
        )
        assert jnp.allclose(loss_pj, loss_tr, atol=atol_out, rtol=rtol_out)

        for name in ("gate_kernel", "wi_0", "wi_1", "wo"):
            g_pj = grads_pj["params"][name]
            g_tr = grads_tr["params"][name]
            assert jnp.allclose(g_pj, g_tr, atol=1e-1, rtol=1e-1), (
                f"Gradient for {name} differs across backends: max diff"
                f" {jnp.max(jnp.abs(g_pj - g_tr))}"
            )

    @pytest.mark.parametrize("permutation_backend", ["pure_jax", "triton"])
    def test_aux_loss_returned(self, permutation_backend):
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

    @pytest.mark.parametrize("permutation_backend", ["pure_jax", "triton"])
    def test_group_topk_deepseek(self, permutation_backend):
        """Exercise DeepSeek-style grouped top-k routing."""
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

    def test_align_size_equivalence_pure_jax(self):
        """For the pure-JAX backend, ``align_size > 0`` must not change the
        numerical output of the forward pass: padding tokens contribute zero
        to every expert GEMM output (their input rows are zeros) and are
        stripped before the weighted sum.
        """
        key = jax.random.PRNGKey(5)
        init_key, data_key = jax.random.split(key)

        base_kwargs = dict(
            num_experts=NUM_EXPERTS,
            num_experts_per_tok=NUM_EXPERTS_PER_TOK,
            intermediate_size=INTERMEDIATE_SIZE,
            permutation_backend="pure_jax",
            dtype=DTYPE,
        )
        block_no_pad = MoEBlock(align_size=0, **base_kwargs)
        block_pad = MoEBlock(align_size=16, **base_kwargs)
        inputs = _make_inputs(data_key)
        variables = block_no_pad.init(init_key, inputs)

        out_no_pad, _ = block_no_pad.apply(variables, inputs)
        out_pad, _ = block_pad.apply(variables, inputs)
        assert jnp.allclose(out_no_pad, out_pad, atol=5e-2, rtol=5e-2), (
            "align_size > 0 must not change pure_jax forward output; max diff"
            f" {jnp.max(jnp.abs(out_no_pad - out_pad))}"
        )

    @pytest.mark.parametrize("permutation_backend", ["pure_jax", "triton"])
    def test_jit_and_determinism(self, permutation_backend):
        """The block must be JIT-compilable and produce a deterministic
        forward pass across repeat calls with the same params."""
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
