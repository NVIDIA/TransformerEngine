# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Tests for fused MoE router CUDA kernels (JAX wrappers)."""

from functools import partial
from typing import Optional

import jax
import jax.numpy as jnp
import pytest

from utils import pytest_parametrize_wrapper

from transformer_engine.jax.router import (
    fused_topk_with_score_function,
    fused_compute_score_for_moe_aux_loss,
    fused_moe_aux_loss,
)

# =============================================================================
# Test case definitions (L0 = fast smoke, L2 = comprehensive)
# =============================================================================

# (num_tokens, num_experts, topk)
ALL_TOPK_CASES = [
    (128, 32, 4),
    (2048, 32, 4),
    (2048, 128, 8),
    (7168, 128, 4),
    (7168, 32, 8),
]
TOPK_CASES = {
    "L0": ALL_TOPK_CASES[0:2],
    "L2": ALL_TOPK_CASES,
}

ALL_GROUP_TOPK_OPTIONS = [None, 4]
GROUP_TOPK_OPTIONS = {
    "L0": [None],
    "L2": ALL_GROUP_TOPK_OPTIONS,
}

ALL_SCALING_FACTOR_OPTIONS = [None, 1.2]
SCALING_FACTOR_OPTIONS = {
    "L0": [None],
    "L2": ALL_SCALING_FACTOR_OPTIONS,
}

ALL_ENABLE_BIAS_OPTIONS = [True, False]
ENABLE_BIAS_OPTIONS = {
    "L0": [False],
    "L2": ALL_ENABLE_BIAS_OPTIONS,
}

ALL_USE_PRE_SOFTMAX_OPTIONS = [True, False]
USE_PRE_SOFTMAX_OPTIONS = {
    "L0": [False],
    "L2": ALL_USE_PRE_SOFTMAX_OPTIONS,
}

# (num_tokens, num_experts, topk)
ALL_SCORE_AUX_LOSS_CASES = [
    (128, 32, 4),
    (2048, 128, 4),
    (2048, 256, 8),
    (7168, 128, 8),
    (7168, 32, 4),
]
SCORE_AUX_LOSS_CASES = {
    "L0": ALL_SCORE_AUX_LOSS_CASES[0:2],
    "L2": ALL_SCORE_AUX_LOSS_CASES,
}

ALL_SCORE_FUNCTIONS = ["softmax", "sigmoid"]
SCORE_FUNCTIONS = {
    "L0": ["softmax"],
    "L2": ALL_SCORE_FUNCTIONS,
}

# (num_tokens, num_experts, topk)
ALL_AUX_LOSS_CASES = [
    (128, 32, 4),
    (2048, 128, 4),
    (2048, 256, 4),
    (7168, 128, 4),
    (7168, 32, 4),
]
AUX_LOSS_CASES = {
    "L0": ALL_AUX_LOSS_CASES[0:2],
    "L2": ALL_AUX_LOSS_CASES,
}

ALL_DTYPES = [jnp.float32]
DTYPES = {
    "L0": [jnp.float32],
    "L2": ALL_DTYPES,
}

SEED = 42


# =============================================================================
# Reference Implementations
# =============================================================================


def reference_group_limited_topk(
    scores: jnp.ndarray,
    topk: int,
    num_tokens: int,
    num_experts: int,
    num_groups: int,
    group_topk: int,
):
    """Reference implementation for grouped top-k.

    Only valid when num_groups and group_topk are both positive integers.
    For plain top-k without grouping, use jax.lax.top_k directly.
    """
    assert num_groups is not None and num_groups > 0, (
        "reference_group_limited_topk requires valid num_groups > 0. "
        "For plain top-k, use jax.lax.top_k directly."
    )
    assert group_topk is not None and group_topk > 0, (
        "reference_group_limited_topk requires valid group_topk > 0."
    )
    assert num_experts % num_groups == 0, (
        f"num_experts ({num_experts}) must be divisible by num_groups ({num_groups})"
    )
    group_size = num_experts // num_groups
    experts_per_group = topk // group_topk

    group_scores = (
        scores.reshape(num_tokens, num_groups, group_size)
        .sort(axis=-1)[..., -experts_per_group:]
        .sum(axis=-1)
    )
    group_idx = jax.lax.top_k(group_scores, k=group_topk)[1]
    group_mask = jnp.zeros_like(group_scores).at[
        jnp.arange(num_tokens)[:, None], group_idx
    ].set(1)

    score_mask = (
        group_mask[:, :, None]
        * jnp.ones((num_tokens, num_groups, group_size))
    ).reshape(num_tokens, -1)

    masked_scores = jnp.where(score_mask.astype(bool), scores, -jnp.inf)
    probs, top_indices = jax.lax.top_k(masked_scores, k=topk)
    return probs, top_indices


def reference_topk_softmax_sigmoid(
    logits: jnp.ndarray,
    topk: int,
    use_pre_softmax: bool = False,
    num_groups: Optional[int] = None,
    group_topk: Optional[int] = None,
    scaling_factor: Optional[float] = None,
    score_function: str = "softmax",
    expert_bias: Optional[jnp.ndarray] = None,
):
    """Reference implementation for topk + softmax/sigmoid."""
    num_tokens, num_experts = logits.shape

    def compute_topk(scores, topk, num_groups=None, group_topk=None):
        if group_topk:
            return reference_group_limited_topk(
                scores=scores,
                topk=topk,
                num_tokens=num_tokens,
                num_experts=num_experts,
                num_groups=num_groups,
                group_topk=group_topk,
            )
        else:
            return jax.lax.top_k(scores, k=topk)

    if score_function == "softmax":
        if use_pre_softmax:
            scores = jax.nn.softmax(logits.astype(jnp.float32), axis=-1).astype(logits.dtype)
            probs, top_indices = compute_topk(scores, topk, num_groups, group_topk)
        else:
            scores, top_indices = compute_topk(logits, topk, num_groups, group_topk)
            probs = jax.nn.softmax(scores.astype(jnp.float32), axis=-1).astype(logits.dtype)
    elif score_function == "sigmoid":
        scores = jax.nn.sigmoid(logits.astype(jnp.float32)).astype(logits.dtype)
        if expert_bias is not None:
            scores_for_routing = scores + expert_bias
            _, top_indices = compute_topk(scores_for_routing, topk, num_groups, group_topk)
            scores = jnp.take_along_axis(scores, top_indices, axis=1).astype(logits.dtype)
        else:
            scores, top_indices = compute_topk(scores, topk, num_groups, group_topk)
        probs = scores / (scores.sum(axis=-1, keepdims=True) + 1e-20) if topk > 1 else scores
    else:
        raise ValueError(f"Invalid score_function: {score_function}")

    if scaling_factor:
        probs = probs * scaling_factor

    topk_masked_gates = jnp.zeros_like(logits).at[
        jnp.arange(num_tokens)[:, None], top_indices
    ].set(probs)
    topk_map = jnp.zeros_like(logits, dtype=jnp.bool_).at[
        jnp.arange(num_tokens)[:, None], top_indices
    ].set(True)

    return topk_masked_gates, topk_map


def reference_compute_scores_for_aux_loss(
    logits: jnp.ndarray, topk: int, score_function: str
):
    """Reference implementation for computing routing scores for aux loss."""
    if score_function == "softmax":
        scores = jax.nn.softmax(logits.astype(jnp.float32), axis=-1)
    elif score_function == "sigmoid":
        scores = jax.nn.sigmoid(logits.astype(jnp.float32))
        scores = scores / (scores.sum(axis=-1, keepdims=True) + 1e-20) if topk > 1 else scores
    else:
        raise ValueError(f"Invalid score_function: {score_function}")

    _, top_indices = jax.lax.top_k(scores, k=topk)
    num_tokens = logits.shape[0]
    routing_map = jnp.zeros_like(logits, dtype=jnp.bool_).at[
        jnp.arange(num_tokens)[:, None], top_indices
    ].set(True)
    return routing_map, scores


def reference_aux_loss(
    probs: jnp.ndarray,
    tokens_per_expert: jnp.ndarray,
    total_num_tokens: int,
    topk: int,
    num_experts: int,
    moe_aux_loss_coeff: float,
):
    """Reference implementation for MoE auxiliary loss."""
    aggregated_probs_per_expert = probs.sum(axis=0)
    aux_loss = jnp.sum(aggregated_probs_per_expert * tokens_per_expert) * (
        num_experts * moe_aux_loss_coeff / (topk * total_num_tokens * total_num_tokens)
    )
    return aux_loss


# =============================================================================
# Helper: logits generation
# =============================================================================


def make_logits(num_tokens, num_experts, score_function, dtype=jnp.float32):
    """Create deterministic logits for testing."""
    if score_function == "sigmoid":
        offset = jnp.arange(-num_tokens // 2, num_tokens // 2, dtype=dtype) * 1e-4
        logits = jnp.arange(-num_experts // 2, num_experts // 2, dtype=dtype) * 1e-2
        logits = logits[None, :].repeat(num_tokens, axis=0) + offset[:, None]
    else:
        logits = (
            jnp.arange(
                -num_tokens * num_experts // 2,
                num_tokens * num_experts // 2,
                dtype=dtype,
            )
            * 1e-4
        )
        logits = logits.reshape(num_tokens, num_experts)
    return logits


# =============================================================================
# Test: Fused Top-K with Score Function
# =============================================================================


def run_topk_comparison(
    dtype,
    num_tokens,
    num_experts,
    topk,
    use_pre_softmax,
    num_groups,
    group_topk,
    scaling_factor,
    score_function,
    enable_bias,
):
    """Compare fused vs reference top-k implementation, both jitted."""
    logits = make_logits(num_tokens, num_experts, score_function, dtype)

    if enable_bias and score_function == "sigmoid":
        expert_bias = jnp.arange(num_experts, dtype=jnp.float32) * 0.1
        expert_bias = jnp.flip(expert_bias)
    else:
        expert_bias = None

    # Forward: reference (jitted)
    ref_fwd_fn = jax.jit(partial(
        reference_topk_softmax_sigmoid,
        topk=topk,
        use_pre_softmax=use_pre_softmax,
        num_groups=num_groups,
        group_topk=group_topk,
        scaling_factor=scaling_factor,
        score_function=score_function,
        expert_bias=expert_bias,
    ))
    probs_ref, routing_map_ref = ref_fwd_fn(logits)

    # Forward: fused (jitted)
    fused_fwd_fn = jax.jit(partial(
        fused_topk_with_score_function,
        topk=topk,
        use_pre_softmax=use_pre_softmax,
        num_groups=num_groups if num_groups else -1,
        group_topk=group_topk if group_topk else -1,
        scaling_factor=scaling_factor if scaling_factor else 1.0,
        score_function=score_function,
        expert_bias=expert_bias,
    ))
    probs_fused, routing_map_fused = fused_fwd_fn(logits)

    assert jnp.allclose(probs_ref, probs_fused, atol=1e-5, rtol=1e-5), \
        f"Probs mismatch: max diff = {jnp.abs(probs_ref - probs_fused).max()}"
    assert jnp.array_equal(routing_map_ref, routing_map_fused), "Routing map mismatch"

    # Backward: reference (jitted)
    def loss_ref(logits_):
        p, _ = reference_topk_softmax_sigmoid(
            logits_, topk, use_pre_softmax, num_groups, group_topk,
            scaling_factor, score_function, expert_bias,
        )
        return p.sum()

    def loss_fused(logits_):
        p, _ = fused_topk_with_score_function(
            logits_, topk, use_pre_softmax,
            num_groups if num_groups else -1,
            group_topk if group_topk else -1,
            scaling_factor if scaling_factor else 1.0,
            score_function, expert_bias,
        )
        return p.sum()

    grad_ref = jax.jit(jax.grad(loss_ref))(logits)
    grad_fused = jax.jit(jax.grad(loss_fused))(logits)
    assert jnp.allclose(grad_ref, grad_fused, atol=1e-5, rtol=1e-5), \
        f"Grad mismatch: max diff = {jnp.abs(grad_ref - grad_fused).max()}"


@pytest_parametrize_wrapper("dtype", DTYPES)
@pytest_parametrize_wrapper(
    "num_tokens,num_experts,topk", TOPK_CASES,
)
@pytest_parametrize_wrapper("group_topk", GROUP_TOPK_OPTIONS)
@pytest_parametrize_wrapper("scaling_factor", SCALING_FACTOR_OPTIONS)
@pytest_parametrize_wrapper("enable_bias", ENABLE_BIAS_OPTIONS)
def test_topk_sigmoid(
    dtype, num_tokens, num_experts, topk, group_topk, scaling_factor, enable_bias
):
    num_groups = 8 if group_topk else None
    run_topk_comparison(
        dtype=dtype,
        num_tokens=num_tokens,
        num_experts=num_experts,
        topk=topk,
        use_pre_softmax=False,
        num_groups=num_groups,
        group_topk=group_topk,
        scaling_factor=scaling_factor,
        score_function="sigmoid",
        enable_bias=enable_bias,
    )


@pytest_parametrize_wrapper("dtype", DTYPES)
@pytest_parametrize_wrapper(
    "num_tokens,num_experts,topk", TOPK_CASES,
)
@pytest_parametrize_wrapper("use_pre_softmax", USE_PRE_SOFTMAX_OPTIONS)
@pytest_parametrize_wrapper("group_topk", GROUP_TOPK_OPTIONS)
@pytest_parametrize_wrapper("scaling_factor", SCALING_FACTOR_OPTIONS)
def test_topk_softmax(
    dtype, num_tokens, num_experts, topk, use_pre_softmax, group_topk, scaling_factor
):
    num_groups = 8 if group_topk else None
    run_topk_comparison(
        dtype=dtype,
        num_tokens=num_tokens,
        num_experts=num_experts,
        topk=topk,
        use_pre_softmax=use_pre_softmax,
        num_groups=num_groups,
        group_topk=group_topk,
        scaling_factor=scaling_factor,
        score_function="softmax",
        enable_bias=False,
    )


# =============================================================================
# Test: Fused Score for MoE Aux Loss
# =============================================================================


@pytest_parametrize_wrapper("dtype", DTYPES)
@pytest_parametrize_wrapper(
    "num_tokens,num_experts,topk", SCORE_AUX_LOSS_CASES,
)
@pytest_parametrize_wrapper("score_function", SCORE_FUNCTIONS)
def test_fused_scores_for_aux_loss(dtype, num_tokens, num_experts, topk, score_function):
    logits = make_logits(num_tokens, num_experts, score_function, dtype)

    # Forward: reference (jitted)
    ref_fwd_fn = jax.jit(partial(
        reference_compute_scores_for_aux_loss,
        topk=topk,
        score_function=score_function,
    ))
    routing_map_ref, scores_ref = ref_fwd_fn(logits)

    # Forward: fused (jitted)
    fused_fwd_fn = jax.jit(partial(
        fused_compute_score_for_moe_aux_loss,
        topk=topk,
        score_function=score_function,
    ))
    routing_map_fused, scores_fused = fused_fwd_fn(logits)

    assert jnp.allclose(scores_ref, scores_fused, atol=1e-5, rtol=1e-5), \
        f"Scores mismatch: max diff = {jnp.abs(scores_ref - scores_fused).max()}"
    assert jnp.array_equal(routing_map_ref, routing_map_fused), "Routing map mismatch"

    # Backward (jitted)
    def loss_ref(logits_):
        _, s = reference_compute_scores_for_aux_loss(logits_, topk, score_function)
        return s.sum()

    def loss_fused(logits_):
        _, s = fused_compute_score_for_moe_aux_loss(logits_, topk, score_function)
        return s.sum()

    grad_ref = jax.jit(jax.grad(loss_ref))(logits)
    grad_fused = jax.jit(jax.grad(loss_fused))(logits)
    assert jnp.allclose(grad_ref, grad_fused, atol=1e-5, rtol=1e-5), \
        f"Grad mismatch: max diff = {jnp.abs(grad_ref - grad_fused).max()}"


# =============================================================================
# Test: Fused MoE Aux Loss
# =============================================================================


@pytest_parametrize_wrapper("dtype", DTYPES)
@pytest_parametrize_wrapper(
    "num_tokens,num_experts,topk", AUX_LOSS_CASES,
)
def test_fused_moe_aux_loss(dtype, num_tokens, num_experts, topk):
    key = jax.random.PRNGKey(SEED)

    offset = jnp.arange(-num_tokens // 2, num_tokens // 2, dtype=dtype) * 1e-4
    probs = jnp.arange(-num_experts // 2, num_experts // 2, dtype=dtype) * 1e-2
    probs = probs[None, :].repeat(num_tokens, axis=0) + offset[:, None]
    probs = probs.reshape(num_tokens, num_experts)

    tokens_per_expert = jax.random.randint(key, (num_experts,), 1, 1000).astype(jnp.int32)
    coeff = 0.01

    # Forward: reference (jitted)
    ref_fwd_fn = jax.jit(partial(
        reference_aux_loss,
        tokens_per_expert=tokens_per_expert,
        total_num_tokens=num_tokens,
        topk=topk,
        num_experts=num_experts,
        moe_aux_loss_coeff=coeff,
    ))
    aux_loss_ref = ref_fwd_fn(probs)

    # Forward: fused (jitted)
    fused_fwd_fn = jax.jit(partial(
        fused_moe_aux_loss,
        tokens_per_expert=tokens_per_expert,
        total_num_tokens=num_tokens,
        num_experts=num_experts,
        topk=topk,
        coeff=coeff,
    ))
    aux_loss_fused = fused_fwd_fn(probs)

    assert jnp.allclose(aux_loss_ref, aux_loss_fused, atol=1e-5, rtol=1e-5), \
        f"Aux loss mismatch: ref={aux_loss_ref}, fused={aux_loss_fused}"

    # Backward (jitted)
    def loss_ref_fn(probs_):
        return reference_aux_loss(probs_, tokens_per_expert, num_tokens, topk, num_experts, coeff)

    def loss_fused_fn(probs_):
        return fused_moe_aux_loss(probs_, tokens_per_expert, num_tokens, num_experts, topk, coeff)

    grad_ref = jax.jit(jax.grad(loss_ref_fn))(probs)
    grad_fused = jax.jit(jax.grad(loss_fused_fn))(probs)
    assert jnp.allclose(grad_ref, grad_fused, atol=1e-5, rtol=1e-5), \
        f"Grad mismatch: max diff = {jnp.abs(grad_ref - grad_fused).max()}"
