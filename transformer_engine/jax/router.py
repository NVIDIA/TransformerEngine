# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Fused MoE Router API for JAX.

This module provides high-level fused router operations for Mixture of Experts (MoE)
models with proper automatic differentiation support. These wrap the CUDA kernels in
transformer_engine/common/fused_router/.

Functions:
    fused_topk_with_score_function:
        Fused score_function + top-k selection. Supports softmax/sigmoid,
        grouped top-k, expert bias, and scaling factor. When compute_aux_scores=True,
        switches to the clean score-for-aux-loss kernel (no bias/groups/scaling,
        dense output).

    fused_moe_aux_loss:
        Compute the MoE auxiliary load-balancing loss scalar.
"""

from functools import partial
from typing import Optional, Tuple, Union

import jax
import jax.numpy as jnp

from transformer_engine.jax.cpp_extensions.router import (
    ScoreFunction,
    fused_topk_with_score_function_fwd,
    fused_topk_with_score_function_bwd,
    fused_moe_aux_loss_fwd,
    fused_moe_aux_loss_bwd,
)

__all__ = [
    "ScoreFunction",
    "fused_topk_with_score_function",
    "fused_moe_aux_loss",
]


def _validate_score_function(score_function: Union[str, ScoreFunction]) -> ScoreFunction:
    """Validate and convert score_function to a ScoreFunction enum."""
    if isinstance(score_function, ScoreFunction):
        return score_function
    try:
        return ScoreFunction[score_function.upper()]
    except (KeyError, AttributeError):
        raise ValueError(
            "score_function must be 'softmax', 'sigmoid', or a ScoreFunction enum, "
            f"got {score_function!r}"
        ) from None


# =============================================================================
# Fused Top-K with Score Function
# =============================================================================


def fused_topk_with_score_function(
    logits: jnp.ndarray,
    topk: int,
    use_pre_softmax: bool = False,
    num_groups: int = -1,
    group_topk: int = -1,
    scaling_factor: float = 1.0,
    score_function: Union[str, ScoreFunction] = ScoreFunction.SOFTMAX,
    expert_bias: Optional[jnp.ndarray] = None,
    compute_aux_scores: bool = False,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Fused top-k with score function router.

    When compute_aux_scores=False (default), runs the main routing kernel:
    score_function(logits) -> [optional bias] -> top-k -> [optional post-softmax] -> scale.
    Returns sparse probs (only top-k positions nonzero) and routing_map.

    When compute_aux_scores=True, runs the score-for-aux-loss kernel instead:
    score_function(logits) -> top-k (clean, no bias/groups/scaling).
    Returns dense scores (all expert positions) and routing_map.
    The expert_bias, use_pre_softmax, num_groups, group_topk, and scaling_factor
    parameters are ignored in this mode.

    Parameters
    ----------
    logits : jnp.ndarray
        Logits from the gating GEMM, shape [num_tokens, num_experts].
    topk : int
        Number of top experts to select per token.
    use_pre_softmax : bool
        If True, apply softmax before top-k (only for softmax score function). Else, apply post top-k.
        Ignored when compute_aux_scores=True.
    num_groups : int
        Number of groups for grouped top-k. <= 0 disables grouping (default).
        Ignored when compute_aux_scores=True.
    group_topk : int
        Top-k at group level. <= 0 disables group-level selection (default).
        Ignored when compute_aux_scores=True.
    scaling_factor : float
        Scaling factor applied to output probs.
        Ignored when compute_aux_scores=True.
    score_function : Union[str, ScoreFunction]
        Score function: "softmax" / "sigmoid" or ScoreFunction.SOFTMAX / ScoreFunction.SIGMOID.
    expert_bias : Optional[jnp.ndarray]
        Expert bias, shape [num_experts]. Only used with sigmoid.
        Ignored when compute_aux_scores=True.
    compute_aux_scores : bool
        If True, use the clean score-for-aux-loss kernel. Returns dense scores
        over all experts instead of sparse probs.

    Returns
    -------
    probs_or_scores : jnp.ndarray
        When compute_aux_scores=False: Sparse probability tensor, shape [num_tokens, num_experts].
            Non-zero only at selected expert positions.
        When compute_aux_scores=True: Dense score tensor, shape [num_tokens, num_experts].
            All expert positions contain scores.
    routing_map : jnp.ndarray
        Boolean mask, shape [num_tokens, num_experts].
        True at selected expert positions.
    """
    if not isinstance(scaling_factor, (int, float)):
        raise TypeError(
            f"scaling_factor must be a Python float or int, not {type(scaling_factor).__name__}. "
            "If you used jnp.sqrt() or similar, use math.sqrt() instead."
        )

    score_function = _validate_score_function(score_function)

    if compute_aux_scores:
        expert_bias = jnp.empty((0,), dtype=logits.dtype)
        use_pre_softmax = False
        num_groups = -1
        group_topk = -1
        scaling_factor = 1.0
    else:
        if expert_bias is not None and score_function != ScoreFunction.SIGMOID:
            raise ValueError(
                "expert_bias is only supported with score_function='sigmoid'. "
                f"Got score_function='{score_function.name}'."
            )
        if expert_bias is None:
            expert_bias = jnp.empty((0,), dtype=logits.dtype)

    probs_or_scores, routing_map = _fused_topk_with_score_function(
        logits,
        expert_bias,
        topk,
        use_pre_softmax,
        num_groups,
        group_topk,
        scaling_factor,
        score_function,
        compute_aux_scores,
    )

    return probs_or_scores, routing_map


@partial(jax.custom_vjp, nondiff_argnums=(2, 3, 4, 5, 6, 7, 8))
def _fused_topk_with_score_function(
    logits: jnp.ndarray,
    expert_bias: jnp.ndarray,
    topk: int,
    use_pre_softmax: bool,
    num_groups: int,
    group_topk: int,
    scaling_factor: float,
    score_function: ScoreFunction,
    compute_aux_scores: bool,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    (probs, routing_map), _ = _fused_topk_with_score_function_fwd(
        logits,
        expert_bias,
        topk,
        use_pre_softmax,
        num_groups,
        group_topk,
        scaling_factor,
        score_function,
        compute_aux_scores,
    )
    return probs, routing_map


def _fused_topk_with_score_function_fwd(
    logits,
    expert_bias,
    topk,
    use_pre_softmax,
    num_groups,
    group_topk,
    scaling_factor,
    score_function,
    compute_aux_scores,
):
    probs, routing_map, saved_scores = fused_topk_with_score_function_fwd(
        logits,
        topk,
        use_pre_softmax,
        num_groups,
        group_topk,
        scaling_factor,
        score_function,
        expert_bias,
        compute_aux_scores,
    )
    residuals = (routing_map, saved_scores)
    return (probs, routing_map), residuals


def _fused_topk_with_score_function_bwd(
    topk,
    use_pre_softmax,
    num_groups,
    group_topk,
    scaling_factor,
    score_function,
    compute_aux_scores,
    residuals,
    g,
):
    del num_groups, group_topk
    routing_map, saved_scores = residuals
    grad_probs, _ = g

    grad_logits = fused_topk_with_score_function_bwd(
        routing_map,
        saved_scores,
        grad_probs,
        topk,
        use_pre_softmax,
        scaling_factor,
        score_function,
        compute_aux_scores,
    )
    return grad_logits, None


_fused_topk_with_score_function.defvjp(
    _fused_topk_with_score_function_fwd,
    _fused_topk_with_score_function_bwd,
)


# =============================================================================
# Fused MoE Aux Loss
# =============================================================================


def fused_moe_aux_loss(
    probs: jnp.ndarray,
    tokens_per_expert: jnp.ndarray,
    topk: int,
    coeff: float,
) -> jnp.ndarray:
    """
    Compute the MoE auxiliary load-balancing loss.

    loss = (E * coeff / (k * T^2)) * sum_i(sum_t(probs[t,i]) * tokens_per_expert[i])

    where T = probs.shape[0] (num_tokens) and E = probs.shape[1] (num_experts).

    Parameters
    ----------
    probs : jnp.ndarray
        Probability/score tensor, shape [num_tokens, num_experts].
    tokens_per_expert : jnp.ndarray
        Token counts per expert, shape [num_experts]. Integer tensor.
    topk : int
        Top-k value.
    coeff : float
        Loss coefficient.

    Returns
    -------
    aux_loss : jnp.ndarray
        Scalar loss value.
    """
    return _fused_moe_aux_loss(probs, tokens_per_expert, topk, coeff)


@partial(jax.custom_vjp, nondiff_argnums=(2, 3))
def _fused_moe_aux_loss(
    probs: jnp.ndarray,
    tokens_per_expert: jnp.ndarray,
    topk: int,
    coeff: float,
) -> jnp.ndarray:
    aux_loss, _ = _fused_moe_aux_loss_fwd(probs, tokens_per_expert, topk, coeff)
    return aux_loss


def _fused_moe_aux_loss_fwd(probs, tokens_per_expert, topk, coeff):
    aux_loss, const_buf = fused_moe_aux_loss_fwd(probs, tokens_per_expert, topk, coeff)
    residuals = (const_buf, tokens_per_expert, probs.shape[0])
    return aux_loss, residuals


def _fused_moe_aux_loss_bwd(topk, coeff, residuals, g):
    del topk, coeff
    const_buf, tokens_per_expert, num_tokens = residuals
    grad_aux_loss = g.reshape(1)

    grad_probs = fused_moe_aux_loss_bwd(
        const_buf,
        tokens_per_expert,
        grad_aux_loss,
        num_tokens,
    )
    return grad_probs, None


_fused_moe_aux_loss.defvjp(
    _fused_moe_aux_loss_fwd,
    _fused_moe_aux_loss_bwd,
)
