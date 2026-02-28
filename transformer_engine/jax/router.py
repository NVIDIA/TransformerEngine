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
        grouped top-k, expert bias, and scaling factor.

    fused_compute_score_for_moe_aux_loss:
        Compute clean scores and routing map for the auxiliary load-balancing loss.

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
    "fused_compute_score_for_moe_aux_loss",
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
            f"score_function must be 'softmax', 'sigmoid', or a ScoreFunction enum, "
            f"got {score_function!r}"
        ) from None


# =============================================================================
# Fused Top-K with Score Function
# =============================================================================


def fused_topk_with_score_function(
    logits: jnp.ndarray,
    topk: int,
    use_pre_softmax: bool = False,
    num_groups: int = 1,
    group_topk: int = 1,
    scaling_factor: float = 1.0,
    score_function: Union[str, ScoreFunction] = ScoreFunction.SOFTMAX,
    expert_bias: Optional[jnp.ndarray] = None,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Fused top-k with score function router.

    Parameters
    ----------
    logits : jnp.ndarray
        Logits from the gating GEMM, shape [num_tokens, num_experts].
    topk : int
        Number of top experts to select per token.
    use_pre_softmax : bool
        If True, apply softmax before top-k (only for softmax score function). Else, apply post top-k
    num_groups : int
        Number of groups for grouped top-k. 1 means no grouping.
    group_topk : int
        Top-k at group level. 1 means no group-level selection.
    scaling_factor : float
        Scaling factor applied to output probs.
    score_function : Union[str, ScoreFunction]
        Score function: "softmax" / "sigmoid" or ScoreFunction.SOFTMAX / ScoreFunction.SIGMOID.
    expert_bias : Optional[jnp.ndarray]
        Expert bias, shape [num_experts]. Only used with sigmoid.

    Returns
    -------
    probs : jnp.ndarray
        Sparse probability tensor, shape [num_tokens, num_experts].
        Non-zero only at selected expert positions.
    routing_map : jnp.ndarray
        Boolean mask, shape [num_tokens, num_experts].
        True at selected expert positions.
    """
    score_function = _validate_score_function(score_function)

    if expert_bias is not None and score_function != ScoreFunction.SIGMOID:
        raise ValueError(
            "expert_bias is only supported with score_function='sigmoid'. "
            f"Got score_function='{score_function.name}'."
        )

    if expert_bias is None:
        expert_bias = jnp.empty((0,), dtype=logits.dtype)

    probs, routing_map = _fused_topk_with_score_function(
        logits,
        expert_bias,
        topk,
        use_pre_softmax,
        num_groups,
        group_topk,
        scaling_factor,
        score_function,
        False,  # compute_aux_scores
    )

    return probs, routing_map


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
        logits, expert_bias, topk, use_pre_softmax, num_groups, group_topk,
        scaling_factor, score_function, compute_aux_scores,
    )
    return probs, routing_map


def _fused_topk_with_score_function_fwd(
    logits, expert_bias, topk, use_pre_softmax, num_groups, group_topk,
    scaling_factor, score_function, compute_aux_scores,
):
    probs, routing_map, intermediate_output = fused_topk_with_score_function_fwd(
        logits, topk, use_pre_softmax, num_groups, group_topk,
        scaling_factor, score_function, expert_bias, compute_aux_scores,
    )
    residuals = (routing_map, intermediate_output)
    return (probs, routing_map), residuals


def _fused_topk_with_score_function_bwd(
    topk, use_pre_softmax, num_groups, group_topk, scaling_factor, score_function,  # pylint: disable=unused-argument
    compute_aux_scores, residuals, g,
):
    routing_map, intermediate_output = residuals
    grad_probs, _ = g  # routing_map gradient is None (boolean)

    grad_logits = fused_topk_with_score_function_bwd(
        routing_map, intermediate_output, grad_probs,
        topk, use_pre_softmax, scaling_factor, score_function,
        compute_aux_scores,
    )
    # expert_bias gradient is None: bias is not differentiated through this kernel
    return grad_logits, None


_fused_topk_with_score_function.defvjp(
    _fused_topk_with_score_function_fwd,
    _fused_topk_with_score_function_bwd,
)


# =============================================================================
# Fused Score for MoE Aux Loss
# =============================================================================


def fused_compute_score_for_moe_aux_loss(
    logits: jnp.ndarray,
    topk: int,
    score_function: Union[str, ScoreFunction] = ScoreFunction.SOFTMAX,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Compute scores and routing map for MoE auxiliary loss.

    This uses clean softmax/sigmoid + plain top-k (no group constraints,
    no expert bias, no scaling) to produce the scores and routing map
    used for the load-balancing auxiliary loss.

    Internally delegates to the same primitive as fused_topk_with_score_function
    with compute_aux_scores=True, selecting the score-for-aux-loss CUDA kernel.

    Parameters
    ----------
    logits : jnp.ndarray
        Logits from the gating GEMM, shape [num_tokens, num_experts].
    topk : int
        Number of top experts to select.
    score_function : Union[str, ScoreFunction]
        Score function: "softmax" / "sigmoid" or ScoreFunction.SOFTMAX / ScoreFunction.SIGMOID.

    Returns
    -------
    routing_map : jnp.ndarray
        Boolean mask, shape [num_tokens, num_experts].
    scores : jnp.ndarray
        Dense score tensor, shape [num_tokens, num_experts].
    """
    score_function = _validate_score_function(score_function)

    scores, routing_map = _fused_topk_with_score_function(
        logits,
        jnp.empty((0,), dtype=logits.dtype),
        topk,
        False,  # use_pre_softmax (unused for aux scores)
        1,      # num_groups (unused for aux scores)
        1,      # group_topk (unused for aux scores)
        1.0,    # scaling_factor (unused for aux scores)
        score_function,
        True,   # compute_aux_scores
    )
    return routing_map, scores


# =============================================================================
# Fused MoE Aux Loss
# =============================================================================


def fused_moe_aux_loss(
    probs: jnp.ndarray,
    tokens_per_expert: jnp.ndarray,
    total_num_tokens: int,
    num_experts: int,
    topk: int,
    coeff: float,
) -> jnp.ndarray:
    """
    Compute the MoE auxiliary load-balancing loss.

    loss = (E * coeff / (k * T^2)) * sum_i(sum_t(probs[t,i]) * tokens_per_expert[i])

    Parameters
    ----------
    probs : jnp.ndarray
        Probability/score tensor, shape [num_tokens, num_experts].
    tokens_per_expert : jnp.ndarray
        Token counts per expert, shape [num_experts]. Integer tensor.
    total_num_tokens : int
        Total token count for normalization.
    num_experts : int
        Number of experts.
    topk : int
        Top-k value.
    coeff : float
        Loss coefficient.

    Returns
    -------
    aux_loss : jnp.ndarray
        Scalar loss value.
    """
    return _fused_moe_aux_loss(
        probs, tokens_per_expert, total_num_tokens, num_experts, topk, coeff,
    )


@partial(jax.custom_vjp, nondiff_argnums=(2, 3, 4, 5))
def _fused_moe_aux_loss(
    probs: jnp.ndarray,
    tokens_per_expert: jnp.ndarray,
    total_num_tokens: int,
    num_experts: int,
    topk: int,
    coeff: float,
) -> jnp.ndarray:
    aux_loss, _ = _fused_moe_aux_loss_fwd(
        probs, tokens_per_expert, total_num_tokens, num_experts, topk, coeff,
    )
    return aux_loss


def _fused_moe_aux_loss_fwd(
    probs, tokens_per_expert, total_num_tokens, num_experts, topk, coeff,
):
    aux_loss, const_buf = fused_moe_aux_loss_fwd(
        probs, tokens_per_expert, total_num_tokens, num_experts, topk, coeff,
    )
    residuals = (const_buf, tokens_per_expert, probs.shape[0], probs.shape[1])
    return aux_loss.squeeze(), residuals


def _fused_moe_aux_loss_bwd(
    total_num_tokens, num_experts, topk, coeff, residuals, g,  # pylint: disable=unused-argument
):
    const_buf, tokens_per_expert, num_rows, num_cols = residuals
    # g is a scalar matching the squeezed output of _fwd
    grad_aux_loss = g.reshape(1)

    grad_probs = fused_moe_aux_loss_bwd(
        const_buf, tokens_per_expert, grad_aux_loss, num_rows, num_cols,
    )
    return grad_probs, None


_fused_moe_aux_loss.defvjp(
    _fused_moe_aux_loss_fwd,
    _fused_moe_aux_loss_bwd,
)
