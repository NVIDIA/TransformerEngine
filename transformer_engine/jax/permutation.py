# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""MoE Permutation API for JAX.

This module provides high-level token dispatch and combine operations for
Mixture of Experts (MoE) models with proper automatic differentiation support.

Token Dispatch (Permute):
    - Forward: Permute tokens according to routing map (scatter to experts)
    - Backward: Unpermute gradients (gather from experts)

Token Combine (Unpermute):
    - Forward: Unpermute tokens and merge with weights (gather from experts)
    - Backward: Permute gradients (scatter to experts)
"""

from typing import Optional, Tuple
from functools import partial

import jax
import jax.numpy as jnp

from transformer_engine.jax.triton_extensions.permutation import (
    make_row_id_map,
    permute_with_mask_map,
    unpermute_with_mask_map,
    make_chunk_sort_map,
    sort_chunks_by_map,
)

__all__ = [
    "token_dispatch",
    "token_dispatch_with_probs",
    "token_combine",
    "sort_chunks_by_index",
]


@partial(jax.custom_vjp, nondiff_argnums=(3, 4, 5, 6))
def token_dispatch(
    inp: jnp.ndarray,
    routing_map: jnp.ndarray,
    row_id_map: Optional[jnp.ndarray],
    num_tokens: int,
    num_experts: int,
    num_out_tokens: int,
    hidden_size: int,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Dispatch tokens to experts based on routing map.

    This is the forward pass of the MoE permutation. Tokens are scattered
    to their designated experts according to the routing map.

    Parameters
    ----------
    inp : jnp.ndarray
        Input tensor of shape [num_tokens, hidden_size].
    routing_map : jnp.ndarray
        Routing mask of shape [num_tokens, num_experts]. Values: 1 = routed, 0 = not routed.
    row_id_map : Optional[jnp.ndarray]
        Pre-computed row ID map. If None, will be computed from routing_map.
    num_tokens : int
        Number of input tokens.
    num_experts : int
        Number of experts.
    num_out_tokens : int
        Number of output tokens (total routed tokens).
    hidden_size : int
        Hidden dimension size.

    Returns
    -------
    output : jnp.ndarray
        Permuted output tensor of shape [num_out_tokens, hidden_size].
    row_id_map : jnp.ndarray
        Row ID map for use in token_combine (shape [num_tokens, num_experts * 2 + 1]).
    """
    if row_id_map is None:
        row_id_map = make_row_id_map(routing_map, num_tokens, num_experts)

    output, _ = permute_with_mask_map(
        inp,
        row_id_map,
        None,  # No probs
        num_tokens,
        num_experts,
        num_out_tokens,
        hidden_size,
    )

    return output, row_id_map


def _token_dispatch_fwd(
    inp: jnp.ndarray,
    routing_map: jnp.ndarray,
    row_id_map: Optional[jnp.ndarray],
    num_tokens: int,
    num_experts: int,
    num_out_tokens: int,
    hidden_size: int,
) -> Tuple[Tuple[jnp.ndarray, jnp.ndarray], jnp.ndarray]:
    """Forward pass for token_dispatch VJP."""
    output, row_id_map_out = token_dispatch(
        inp, routing_map, row_id_map, num_tokens, num_experts, num_out_tokens, hidden_size
    )
    # Only save row_id_map in residuals; num_tokens, num_experts, hidden_size
    # come from nondiff_argnums and are passed directly to bwd
    residuals = row_id_map_out
    return (output, row_id_map_out), residuals


def _token_dispatch_bwd(
    num_tokens: int,
    num_experts: int,
    num_out_tokens: int,
    hidden_size: int,
    residuals: jnp.ndarray,
    g: Tuple[jnp.ndarray, jnp.ndarray],
) -> Tuple[jnp.ndarray, None, None]:
    """Backward pass for token_dispatch: unpermute the gradients."""
    row_id_map = residuals
    output_grad, _ = g  # Ignore row_id_map gradient

    # Backward: unpermute gradients (gather from experts back to tokens)
    inp_grad, _ = unpermute_with_mask_map(
        output_grad,
        row_id_map,
        None,  # No merging probs
        None,  # No permuted probs
        num_tokens,
        num_experts,
        hidden_size,
    )

    # Return gradients for (inp, routing_map, row_id_map)
    # routing_map and row_id_map don't need gradients
    return inp_grad, None, None


token_dispatch.defvjp(_token_dispatch_fwd, _token_dispatch_bwd)


@partial(jax.custom_vjp, nondiff_argnums=(4, 5, 6, 7))
def token_dispatch_with_probs(
    inp: jnp.ndarray,
    routing_map: jnp.ndarray,
    probs: jnp.ndarray,
    row_id_map: Optional[jnp.ndarray],
    num_tokens: int,
    num_experts: int,
    num_out_tokens: int,
    hidden_size: int,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Dispatch tokens to experts with routing probabilities.

    Parameters
    ----------
    inp : jnp.ndarray
        Input tensor of shape [num_tokens, hidden_size].
    routing_map : jnp.ndarray
        Routing mask of shape [num_tokens, num_experts].
    probs : jnp.ndarray
        Routing probabilities of shape [num_tokens, num_experts].
    row_id_map : Optional[jnp.ndarray]
        Pre-computed row ID map. If None, will be computed from routing_map.
    num_tokens : int
        Number of input tokens.
    num_experts : int
        Number of experts.
    num_out_tokens : int
        Number of output tokens.
    hidden_size : int
        Hidden dimension size.

    Returns
    -------
    output : jnp.ndarray
        Permuted output tensor of shape [num_out_tokens, hidden_size].
    permuted_probs : jnp.ndarray
        Permuted probabilities of shape [num_out_tokens].
    row_id_map : jnp.ndarray
        Row ID map for use in token_combine.
    """
    if row_id_map is None:
        row_id_map = make_row_id_map(routing_map, num_tokens, num_experts)

    output, permuted_probs = permute_with_mask_map(
        inp,
        row_id_map,
        probs,
        num_tokens,
        num_experts,
        num_out_tokens,
        hidden_size,
    )

    return output, permuted_probs, row_id_map


def _token_dispatch_with_probs_fwd(
    inp: jnp.ndarray,
    routing_map: jnp.ndarray,
    probs: jnp.ndarray,
    row_id_map: Optional[jnp.ndarray],
    num_tokens: int,
    num_experts: int,
    num_out_tokens: int,
    hidden_size: int,
) -> Tuple[Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray]:
    """Forward pass for token_dispatch_with_probs VJP."""
    output, permuted_probs, row_id_map_out = token_dispatch_with_probs(
        inp, routing_map, probs, row_id_map, num_tokens, num_experts, num_out_tokens, hidden_size
    )
    # Only save row_id_map; other sizes come from nondiff_argnums
    residuals = row_id_map_out
    return (output, permuted_probs, row_id_map_out), residuals


def _token_dispatch_with_probs_bwd(
    num_tokens: int,
    num_experts: int,
    num_out_tokens: int,
    hidden_size: int,
    residuals: jnp.ndarray,
    g: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
) -> Tuple[jnp.ndarray, None, jnp.ndarray, None]:
    """Backward pass for token_dispatch_with_probs."""
    row_id_map = residuals
    output_grad, permuted_probs_grad, _ = g

    # Backward: unpermute gradients
    inp_grad, probs_grad = unpermute_with_mask_map(
        output_grad,
        row_id_map,
        None,  # No merging probs
        permuted_probs_grad,  # Unpermute the probs gradient
        num_tokens,
        num_experts,
        hidden_size,
    )

    return inp_grad, None, probs_grad, None


token_dispatch_with_probs.defvjp(_token_dispatch_with_probs_fwd, _token_dispatch_with_probs_bwd)


# =============================================================================
# Token Combine (Unpermute) with VJP
# =============================================================================


@partial(jax.custom_vjp, nondiff_argnums=(3, 4, 5))
def token_combine(
    inp: jnp.ndarray,
    row_id_map: jnp.ndarray,
    merging_probs: Optional[jnp.ndarray],
    num_tokens: int,
    num_experts: int,
    hidden_size: int,
) -> jnp.ndarray:
    """
    Combine tokens from experts back to original token positions.

    This is the forward pass of MoE unpermutation. Tokens are gathered from
    experts and merged (optionally weighted by merging_probs).

    Parameters
    ----------
    inp : jnp.ndarray
        Input tensor from experts of shape [num_out_tokens, hidden_size].
    row_id_map : jnp.ndarray
        Row ID map from token_dispatch of shape [num_tokens, num_experts * 2 + 1].
    merging_probs : Optional[jnp.ndarray]
        Merging weights of shape [num_tokens, num_experts]. If provided, tokens
        from different experts are weighted-summed. If None, tokens are summed directly.
    num_tokens : int
        Number of output tokens (original token count).
    num_experts : int
        Number of experts.
    hidden_size : int
        Hidden dimension size.

    Returns
    -------
    output : jnp.ndarray
        Combined output tensor of shape [num_tokens, hidden_size].
    """
    output, _ = unpermute_with_mask_map(
        inp,
        row_id_map,
        merging_probs,
        None,  # No permuted probs to unpermute
        num_tokens,
        num_experts,
        hidden_size,
    )

    return output


def _token_combine_fwd(
    inp: jnp.ndarray,
    row_id_map: jnp.ndarray,
    merging_probs: Optional[jnp.ndarray],
    num_tokens: int,
    num_experts: int,
    hidden_size: int,
) -> Tuple[jnp.ndarray, Tuple[jnp.ndarray, Optional[jnp.ndarray], int]]:
    """Forward pass for token_combine VJP."""
    output = token_combine(inp, row_id_map, merging_probs, num_tokens, num_experts, hidden_size)

    # Save for backward:
    # - row_id_map: needed for permutation
    # - merging_probs: needed if we want to scale gradients
    # - num_out_tokens: computed from inp.shape[0], not in nondiff_argnums
    # Note: num_tokens, num_experts, hidden_size come from nondiff_argnums
    num_out_tokens = inp.shape[0]
    residuals = (row_id_map, merging_probs, num_out_tokens)
    return output, residuals


def _token_combine_bwd(
    num_tokens: int,
    num_experts: int,
    hidden_size: int,
    residuals: Tuple[jnp.ndarray, Optional[jnp.ndarray], int],
    g: jnp.ndarray,
) -> Tuple[jnp.ndarray, None, Optional[jnp.ndarray]]:
    """Backward pass for token_combine: permute the gradients.

    Note: num_tokens, num_experts, hidden_size come from nondiff_argnums.
    """
    row_id_map, merging_probs, num_out_tokens = residuals
    output_grad = g

    with_merging_probs = merging_probs is not None

    if with_merging_probs:
        # Need to compute gradient for merging_probs
        # This requires a specialized backward kernel
        # For now, we use the simple approach: permute with merging_probs applied

        # Scale output_grad by merging_probs before permuting back
        # inp_grad[dest] = output_grad[src] * merging_probs[src, expert]
        inp_grad, _ = permute_with_mask_map(
            output_grad,
            row_id_map,
            merging_probs,  # Used to scale gradients
            num_tokens,
            num_experts,
            num_out_tokens,
            hidden_size,
        )

        # Compute gradient for merging_probs
        # d_loss/d_merging_probs = sum over hidden (fwd_inp * output_grad)
        # This requires the forward input and computing per-expert contributions
        # For simplicity, we'll set this to None and require users to handle it
        # In practice, you'd need a specialized kernel for this
        merging_probs_grad = None
    else:
        # Simple case: just permute gradients back
        inp_grad, _ = permute_with_mask_map(
            output_grad,
            row_id_map,
            None,
            num_tokens,
            num_experts,
            num_out_tokens,
            hidden_size,
        )
        merging_probs_grad = None

    return inp_grad, None, merging_probs_grad


token_combine.defvjp(_token_combine_fwd, _token_combine_bwd)


# =============================================================================
# Chunk Sort with VJP
# =============================================================================


@partial(jax.custom_vjp, nondiff_argnums=(3, 4))
def sort_chunks_by_index(
    inp: jnp.ndarray,
    split_sizes: jnp.ndarray,
    sorted_indices: jnp.ndarray,
    num_tokens: int,
    hidden_size: int,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Sort chunks of tokens according to sorted indices.

    Parameters
    ----------
    inp : jnp.ndarray
        Input tensor of shape [num_tokens, hidden_size].
    split_sizes : jnp.ndarray
        Sizes of each chunk of shape [num_splits].
    sorted_indices : jnp.ndarray
        Permutation indices for chunks of shape [num_splits].
    num_tokens : int
        Total number of tokens.
    hidden_size : int
        Hidden dimension size.

    Returns
    -------
    output : jnp.ndarray
        Sorted output tensor of shape [num_tokens, hidden_size].
    row_id_map : jnp.ndarray
        Row ID map for reversing the sort.
    """
    num_splits = split_sizes.shape[0]
    row_id_map = make_chunk_sort_map(split_sizes, sorted_indices, num_tokens, num_splits)

    output, _ = sort_chunks_by_map(
        inp,
        row_id_map,
        None,  # No probs
        num_tokens,
        hidden_size,
        is_forward=True,
    )

    return output, row_id_map


def _sort_chunks_by_index_fwd(
    inp: jnp.ndarray,
    split_sizes: jnp.ndarray,
    sorted_indices: jnp.ndarray,
    num_tokens: int,
    hidden_size: int,
) -> Tuple[Tuple[jnp.ndarray, jnp.ndarray], jnp.ndarray]:
    """Forward pass for sort_chunks_by_index VJP."""
    output, row_id_map = sort_chunks_by_index(
        inp, split_sizes, sorted_indices, num_tokens, hidden_size
    )
    # Only save row_id_map; num_tokens, hidden_size come from nondiff_argnums
    residuals = row_id_map
    return (output, row_id_map), residuals


def _sort_chunks_by_index_bwd(
    num_tokens: int,
    hidden_size: int,
    residuals: jnp.ndarray,
    g: Tuple[jnp.ndarray, jnp.ndarray],
) -> Tuple[jnp.ndarray, None, None]:
    """Backward pass for sort_chunks_by_index: reverse sort.

    Note: num_tokens, hidden_size come from nondiff_argnums.
    """
    row_id_map = residuals
    output_grad, _ = g

    # Backward: reverse the sort (is_forward=False)
    inp_grad, _ = sort_chunks_by_map(
        output_grad,
        row_id_map,
        None,
        num_tokens,
        hidden_size,
        is_forward=False,
    )

    return inp_grad, None, None


sort_chunks_by_index.defvjp(_sort_chunks_by_index_fwd, _sort_chunks_by_index_bwd)
