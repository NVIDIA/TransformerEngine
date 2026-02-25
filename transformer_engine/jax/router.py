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
from typing import Optional, Tuple

import jax
import jax.numpy as jnp

from transformer_engine.jax.cpp_extensions.router import (
    fused_topk_with_score_function_fwd,
    fused_topk_with_score_function_bwd,
    fused_score_for_moe_aux_loss_fwd,
    fused_score_for_moe_aux_loss_bwd,
    fused_moe_aux_loss_fwd,
    fused_moe_aux_loss_bwd,
)

__all__ = [
    "fused_topk_with_score_function",
    "fused_compute_score_for_moe_aux_loss",
    "fused_moe_aux_loss",
]


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
    score_function: str = "softmax",
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
        Number of groups for grouped top-k. -1 to disable.
    group_topk : int
        Top-k at group level. -1 to disable.
    scaling_factor : float
        Scaling factor applied to output probs.
    score_function : str
        Score function: "softmax" or "sigmoid".
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
    if score_function not in ("softmax", "sigmoid"):
        raise ValueError(
            f"score_function must be 'softmax' or 'sigmoid', got '{score_function}'"
        )

    if expert_bias is not None and score_function != "sigmoid":
        raise ValueError(
            "expert_bias is only supported with score_function='sigmoid'. "
            f"Got score_function='{score_function}'."
        )

    # Flatten to 2D if shape is [B, S, H]
    original_shape = logits.shape
    if logits.ndim > 2:
        logits = logits.reshape(-1, original_shape[-1])

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
    )

    # Restore shape if needed
    if len(original_shape) > 2:
        probs = probs.reshape(original_shape)
        routing_map = routing_map.reshape(original_shape)

    return probs, routing_map


@partial(jax.custom_vjp, nondiff_argnums=(2, 3, 4, 5, 6, 7))
def _fused_topk_with_score_function(
    logits: jnp.ndarray,
    expert_bias: jnp.ndarray,
    topk: int,
    use_pre_softmax: bool,
    num_groups: int,
    group_topk: int,
    scaling_factor: float,
    score_function: str,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    (probs, routing_map), _ = _fused_topk_with_score_function_fwd(
        logits, expert_bias, topk, use_pre_softmax, num_groups, group_topk,
        scaling_factor, score_function,
    )
    return probs, routing_map


def _fused_topk_with_score_function_fwd(
    logits, expert_bias, topk, use_pre_softmax, num_groups, group_topk,
    scaling_factor, score_function,
):
    probs, routing_map, intermediate_output = fused_topk_with_score_function_fwd(
        logits, topk, use_pre_softmax, num_groups, group_topk,
        scaling_factor, score_function, expert_bias,
    )
    residuals = (routing_map, intermediate_output)
    return (probs, routing_map), residuals


def _fused_topk_with_score_function_bwd(
    topk, use_pre_softmax, num_groups, group_topk, scaling_factor, score_function,
    residuals, g,
):
    routing_map, intermediate_output = residuals
    grad_probs, _ = g  # routing_map gradient is None (boolean)

    grad_logits = fused_topk_with_score_function_bwd(
        routing_map, intermediate_output, grad_probs,
        topk, use_pre_softmax, scaling_factor, score_function,
    )
    # Return gradients for (logits, expert_bias)
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
    score_function: str = "softmax",
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Compute scores and routing map for MoE auxiliary loss.

    This uses clean softmax/sigmoid + plain top-k (no group constraints,
    no expert bias, no scaling) to produce the scores and routing map
    used for the load-balancing auxiliary loss.

    Parameters
    ----------
    logits : jnp.ndarray
        Logits from the gating GEMM, shape [num_tokens, num_experts].
    topk : int
        Number of top experts to select.
    score_function : str
        Score function: "softmax" or "sigmoid".

    Returns
    -------
    routing_map : jnp.ndarray
        Boolean mask, shape [num_tokens, num_experts].
    scores : jnp.ndarray
        Dense score tensor, shape [num_tokens, num_experts].
    """
    if score_function not in ("softmax", "sigmoid"):
        raise ValueError(
            f"score_function must be 'softmax' or 'sigmoid', got '{score_function}'"
        )

    original_shape = logits.shape
    if logits.ndim > 2:
        logits = logits.reshape(-1, original_shape[-1])

    routing_map, scores = _fused_compute_score_for_moe_aux_loss(logits, topk, score_function)

    if len(original_shape) > 2:
        routing_map = routing_map.reshape(original_shape)
        scores = scores.reshape(original_shape)

    return routing_map, scores


@partial(jax.custom_vjp, nondiff_argnums=(1, 2))
def _fused_compute_score_for_moe_aux_loss(
    logits: jnp.ndarray,
    topk: int,
    score_function: str,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    (routing_map, scores), _ = _fused_compute_score_for_moe_aux_loss_fwd(
        logits, topk, score_function,
    )
    return routing_map, scores


def _fused_compute_score_for_moe_aux_loss_fwd(logits, topk, score_function):
    scores, routing_map, intermediate_output = fused_score_for_moe_aux_loss_fwd(
        logits, topk, score_function,
    )
    residuals = (intermediate_output,)
    return (routing_map, scores), residuals


def _fused_compute_score_for_moe_aux_loss_bwd(topk, score_function, residuals, g):
    (intermediate_output,) = residuals
    _, grad_scores = g  # routing_map gradient is None (boolean)

    grad_logits = fused_score_for_moe_aux_loss_bwd(
        intermediate_output, grad_scores, topk, score_function,
    )
    return (grad_logits,)


_fused_compute_score_for_moe_aux_loss.defvjp(
    _fused_compute_score_for_moe_aux_loss_fwd,
    _fused_compute_score_for_moe_aux_loss_bwd,
)


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
    (aux_loss,), _ = _fused_moe_aux_loss_fwd(
        probs, tokens_per_expert, total_num_tokens, num_experts, topk, coeff,
    )
    # Squeeze from shape (1,) to scalar
    return aux_loss.squeeze()


def _fused_moe_aux_loss_fwd(
    probs, tokens_per_expert, total_num_tokens, num_experts, topk, coeff,
):
    num_rows = probs.shape[0]
    num_cols = probs.shape[1]
    aux_loss, const_buf = fused_moe_aux_loss_fwd(
        probs, tokens_per_expert, total_num_tokens, num_experts, topk, coeff,
    )
    residuals = (const_buf, tokens_per_expert, num_rows, num_cols)
    return (aux_loss,), residuals


def _fused_moe_aux_loss_bwd(
    total_num_tokens, num_experts, topk, coeff, residuals, g,
):
    const_buf, tokens_per_expert, num_rows, num_cols = residuals
    # g is a tuple matching the output of fwd; the squeeze means g is a scalar
    (grad_aux_loss,) = g
    # Ensure grad_aux_loss has shape (1,) for the C kernel
    grad_aux_loss = grad_aux_loss.reshape(1)

    grad_probs = fused_moe_aux_loss_bwd(
        const_buf, tokens_per_expert, grad_aux_loss, num_rows, num_cols,
    )
    # Return gradients for (probs, tokens_per_expert)
    # tokens_per_expert is integer, no gradient
    return grad_probs, None


_fused_moe_aux_loss.defvjp(
    _fused_moe_aux_loss_fwd,
    _fused_moe_aux_loss_bwd,
)
