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

from functools import partial
from typing import Optional, Tuple

import jax
import jax.numpy as jnp

from transformer_engine.jax.triton_extensions.permutation import (
    make_row_id_map,
    permute_with_mask_map,
    unpermute_with_mask_map,
    unpermute_bwd_with_merging_probs,
    make_chunk_sort_map,
    sort_chunks_by_map,
)

__all__ = [
    "token_dispatch",
    "token_combine",
    "sort_chunks_by_index",
]


def token_dispatch(
    inp: jnp.ndarray,
    routing_map: jnp.ndarray,
    num_out_tokens: int,
    probs: Optional[jnp.ndarray] = None,
) -> Tuple[jnp.ndarray, Optional[jnp.ndarray], jnp.ndarray]:
    """
    Dispatch tokens to experts based on routing map.

    This is the forward pass of the MoE permutation. Tokens are scattered
    to their designated experts according to the routing map. The row_id_map
    is computed internally from the routing_map.

    Parameters
    ----------
    inp : jnp.ndarray
        Input tensor of shape [batch, sequence, hidden_size] or [num_tokens, hidden_size].
    routing_map : jnp.ndarray
        Routing mask of shape [batch, sequence, num_experts] or [num_tokens, num_experts].
        Values: 1 = routed, 0 = not routed.
    num_out_tokens : int
        The number of output tokens after permutation. This should equal the sum of
        routing_map and must be provided explicitly for JIT compatibility.
    probs : Optional[jnp.ndarray]
        Optional routing probabilities of shape [batch, sequence, num_experts] or
        [num_tokens, num_experts]. If provided, permuted_probs will be returned.

    Returns
    -------
    output : jnp.ndarray
        Permuted output tensor of shape [num_out_tokens, hidden_size].
    permuted_probs : Optional[jnp.ndarray]
        Permuted probabilities of shape [num_out_tokens], or None if probs was not provided.
    row_id_map : jnp.ndarray
        Row ID map for use in token_combine (shape [num_tokens, num_experts * 2 + 1]).
    """
    return _token_dispatch(inp, routing_map, probs, num_out_tokens)


@partial(jax.custom_vjp, nondiff_argnums=(1, 3))
def _token_dispatch(
    inp: jnp.ndarray,
    routing_map: jnp.ndarray,
    probs: Optional[jnp.ndarray],
    num_out_tokens: int,
) -> Tuple[jnp.ndarray, Optional[jnp.ndarray], jnp.ndarray]:
    """Internal token_dispatch with custom VJP."""
    (output, permuted_probs, row_id_map), _ = _token_dispatch_fwd_rule(
        inp, routing_map, probs, num_out_tokens
    )
    return output, permuted_probs, row_id_map


def _token_dispatch_fwd_rule(
    inp: jnp.ndarray,
    routing_map: jnp.ndarray,
    probs: Optional[jnp.ndarray],
    num_out_tokens: int,
) -> Tuple[
    Tuple[jnp.ndarray, Optional[jnp.ndarray], jnp.ndarray],
    Tuple[jnp.ndarray, int, int, int, bool],
]:
    """Forward pass rule for token_dispatch."""
    # Validate input dimensions
    assert inp.ndim in [2, 3], f"inp must be 2D or 3D, got {inp.ndim}D"
    assert routing_map.ndim in [2, 3], f"routing_map must be 2D or 3D, got {routing_map.ndim}D"

    # Infer dimensions from input shapes
    num_tokens = inp.shape[0] * inp.shape[1] if inp.ndim == 3 else inp.shape[0]
    hidden_size = inp.shape[-1]
    num_experts = routing_map.shape[-1]

    # Verify consistency between inp and routing_map
    routing_num_tokens = (
        routing_map.shape[0] * routing_map.shape[1]
        if routing_map.ndim == 3
        else routing_map.shape[0]
    )
    assert num_tokens == routing_num_tokens, (
        f"Token count mismatch: inp has {num_tokens} tokens, "
        f"routing_map has {routing_num_tokens} tokens"
    )

    # Always compute row_id_map internally from routing_map
    row_id_map = make_row_id_map(routing_map, num_tokens, num_experts)

    with_probs = probs is not None

    output, permuted_probs = permute_with_mask_map(
        inp,
        row_id_map,
        probs,
        num_tokens,
        num_experts,
        num_out_tokens,
        hidden_size,
    )

    # Return (primals, residuals)
    # Include with_probs flag to know how to handle backward pass
    residuals = (row_id_map, num_tokens, num_experts, hidden_size, with_probs)
    return (output, permuted_probs, row_id_map), residuals


def _token_dispatch_bwd_rule(
    _routing_map: jnp.ndarray,
    _num_out_tokens: int,
    residuals: Tuple[jnp.ndarray, int, int, int, bool],
    g: Tuple[jnp.ndarray, Optional[jnp.ndarray], jnp.ndarray],
) -> Tuple[jnp.ndarray, Optional[jnp.ndarray]]:
    """Backward pass rule for token_dispatch."""
    row_id_map, num_tokens, num_experts, hidden_size, with_probs = residuals
    output_grad, permuted_probs_grad, _ = g  # Ignore row_id_map gradient

    # Backward: unpermute gradients (gather from experts back to tokens)
    inp_grad, probs_grad = unpermute_with_mask_map(
        output_grad,
        row_id_map,
        None,  # No merging probs
        permuted_probs_grad if with_probs else None,
        num_tokens,
        num_experts,
        hidden_size,
    )

    return inp_grad, probs_grad if with_probs else None


_token_dispatch.defvjp(_token_dispatch_fwd_rule, _token_dispatch_bwd_rule)


# =============================================================================
# Token Combine (Unpermute) with VJP
# =============================================================================


def token_combine(
    inp: jnp.ndarray,
    row_id_map: jnp.ndarray,
    merging_probs: Optional[jnp.ndarray] = None,
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
        Merging weights of shape [batch, sequence, num_experts] or [num_tokens, num_experts].
        If provided, tokens from different experts are weighted-summed.
        If None, tokens are summed directly.

    Returns
    -------
    output : jnp.ndarray
        Combined output tensor of shape [num_tokens, hidden_size].
    """
    return _token_combine(inp, row_id_map, merging_probs)


@partial(jax.custom_vjp, nondiff_argnums=(1,))
def _token_combine(
    inp: jnp.ndarray,
    row_id_map: jnp.ndarray,
    merging_probs: Optional[jnp.ndarray],
) -> jnp.ndarray:
    """Internal token_combine with custom VJP."""
    output, _ = _token_combine_fwd_rule(inp, row_id_map, merging_probs)
    return output


def _token_combine_fwd_rule(
    inp: jnp.ndarray,
    row_id_map: jnp.ndarray,
    merging_probs: Optional[jnp.ndarray],
) -> Tuple[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray, Optional[jnp.ndarray], int, int, int, int]]:
    """Forward pass rule for token_combine."""
    # Infer dimensions from row_id_map shape: [num_tokens, num_experts * 2 + 1]
    num_tokens = row_id_map.shape[0]
    num_experts = (row_id_map.shape[1] - 1) // 2
    hidden_size = inp.shape[-1]
    num_out_tokens = inp.shape[0]

    # Call triton extension
    output, _ = unpermute_with_mask_map(
        inp,
        row_id_map,
        merging_probs,
        None,  # No permuted probs to unpermute
        num_tokens,
        num_experts,
        hidden_size,
    )

    # Return (primal, residuals)
    # Include inp in residuals for backward with merging_probs
    residuals = (
        row_id_map,
        inp,
        merging_probs,
        num_tokens,
        num_experts,
        hidden_size,
        num_out_tokens,
    )
    return output, residuals


def _token_combine_bwd_rule(
    row_id_map: jnp.ndarray,
    residuals: Tuple[jnp.ndarray, jnp.ndarray, Optional[jnp.ndarray], int, int, int, int],
    g: jnp.ndarray,
) -> Tuple[jnp.ndarray, Optional[jnp.ndarray]]:
    """Backward pass rule for token_combine."""
    (
        row_id_map,
        fwd_input,
        merging_probs,
        num_tokens,
        num_experts,
        hidden_size,
        num_out_tokens,
    ) = residuals
    output_grad = g

    with_merging_probs = merging_probs is not None

    if with_merging_probs:
        # Use specialized backward kernel that properly scales by merging_probs
        inp_grad, merging_probs_grad = unpermute_bwd_with_merging_probs(
            output_grad,
            row_id_map,
            fwd_input,
            merging_probs,
            num_tokens,
            num_experts,
            num_out_tokens,
            hidden_size,
        )
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

    return inp_grad, merging_probs_grad


_token_combine.defvjp(_token_combine_fwd_rule, _token_combine_bwd_rule)


# =============================================================================
# Chunk Sort with VJP
# =============================================================================


def sort_chunks_by_index(
    inp: jnp.ndarray,
    split_sizes: jnp.ndarray,
    sorted_indices: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Sort chunks of tokens according to sorted indices.

    Parameters
    ----------
    inp : jnp.ndarray
        Input tensor of shape [batch, sequence, hidden_size] or [num_tokens, hidden_size].
    split_sizes : jnp.ndarray
        Sizes of each chunk of shape [num_splits].
    sorted_indices : jnp.ndarray
        Permutation indices for chunks of shape [num_splits].

    Returns
    -------
    output : jnp.ndarray
        Sorted output tensor of shape [num_tokens, hidden_size].
    row_id_map : jnp.ndarray
        Row ID map for reversing the sort.
    """
    return _sort_chunks_by_index(inp, split_sizes, sorted_indices)


@partial(jax.custom_vjp, nondiff_argnums=(1, 2))
def _sort_chunks_by_index(
    inp: jnp.ndarray,
    split_sizes: jnp.ndarray,
    sorted_indices: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Internal sort_chunks_by_index with custom VJP."""
    (output, row_id_map), _ = _sort_chunks_by_index_fwd_rule(inp, split_sizes, sorted_indices)
    return output, row_id_map


def _sort_chunks_by_index_fwd_rule(
    inp: jnp.ndarray,
    split_sizes: jnp.ndarray,
    sorted_indices: jnp.ndarray,
) -> Tuple[Tuple[jnp.ndarray, jnp.ndarray], Tuple[jnp.ndarray, int, int]]:
    """Forward pass rule for sort_chunks_by_index."""
    # Validate input dimensions
    assert inp.ndim in [2, 3], f"inp must be 2D or 3D, got {inp.ndim}D"

    # Infer dimensions from input shape
    num_tokens = inp.shape[0] * inp.shape[1] if inp.ndim == 3 else inp.shape[0]
    hidden_size = inp.shape[-1]
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

    # Return (primals, residuals)
    residuals = (row_id_map, num_tokens, hidden_size)
    return (output, row_id_map), residuals


def _sort_chunks_by_index_bwd_rule(
    _split_sizes: jnp.ndarray,
    _sorted_indices: jnp.ndarray,
    residuals: Tuple[jnp.ndarray, int, int],
    g: Tuple[jnp.ndarray, jnp.ndarray],
) -> Tuple[jnp.ndarray]:
    """Backward pass rule for sort_chunks_by_index."""
    row_id_map, num_tokens, hidden_size = residuals
    output_grad, _ = g

    # Backward: reverse the sort
    inp_grad, _ = sort_chunks_by_map(
        output_grad,
        row_id_map,
        None,
        num_tokens,
        hidden_size,
        is_forward=False,
    )

    return (inp_grad,)


_sort_chunks_by_index.defvjp(_sort_chunks_by_index_fwd_rule, _sort_chunks_by_index_bwd_rule)
