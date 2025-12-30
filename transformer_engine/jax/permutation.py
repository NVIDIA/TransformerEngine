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
    permute_with_mask_map_and_pad,
    unpermute_with_mask_map,
    unpermute_with_mask_map_and_unpad,
    unpermute_bwd_with_merging_probs,
    unpermute_bwd_with_merging_probs_and_unpad,
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
    align_size: Optional[int] = None,
) -> Tuple[
    jnp.ndarray,
    Optional[jnp.ndarray],
    jnp.ndarray,
    Optional[jnp.ndarray],
    Optional[jnp.ndarray],
]:
    """
    Dispatch tokens to experts based on routing map.

    This is the forward pass of the MoE permutation. Tokens are scattered
    to their designated experts according to the routing map. The row_id_map
    is computed internally from the routing_map.

    Optionally supports fused padding for alignment when `align_size` is provided.
    This is useful for efficient matrix multiplications that require aligned tensor
    dimensions. The padding is computed internally from the routing_map.

    Parameters
    ----------
    inp : jnp.ndarray
        Input tensor of shape [batch, sequence, hidden_size] or [num_tokens, hidden_size].
    routing_map : jnp.ndarray
        Routing mask of shape [batch, sequence, num_experts] or [num_tokens, num_experts].
        Values: 1 = routed, 0 = not routed.
    num_out_tokens : int
        The number of output tokens after permutation (before padding). For the dropless
        case, this should be equal to the sum of routing_map. Must be provided explicitly
        for JIT compatibility since output shape must be known at compile time.
    probs : Optional[jnp.ndarray]
        Optional routing probabilities of shape [batch, sequence, num_experts] or
        [num_tokens, num_experts]. If provided, permuted_probs will be returned.
    align_size : Optional[int]
        Optional alignment size for padding. If provided, outputs will be padded to
        align each expert's tokens to a multiple of this size. The output buffer is
        allocated with worst-case size, rounded down to align_size:
        ((num_out_tokens + num_experts * (align_size - 1)) // align_size) * align_size
        This enables full JIT compatibility.

    Returns
    -------
    output : jnp.ndarray
        Permuted output tensor of shape [num_out_tokens, hidden_size] without padding,
        or [worst_case_padded_size, hidden_size] when using padding fusion.
        With padding, the actual used portion may be smaller than the buffer; check
        actual_num_out_tokens (sum of target_tokens_per_expert) for the actual size.
    permuted_probs : Optional[jnp.ndarray]
        Permuted probabilities of shape [num_out_tokens] or [worst_case_padded_size],
        or None if probs was not provided.
    row_id_map : jnp.ndarray
        Row ID map for use in token_combine (shape [num_tokens, num_experts * 2 + 1]).
    pad_offsets : Optional[jnp.ndarray]
        Per-expert cumulative padding offsets of shape [num_experts] when using padding,
        None otherwise. Pass this to token_combine when unpadding is needed.
    target_tokens_per_expert : Optional[jnp.ndarray]
        Aligned token counts per expert of shape [num_experts] when using padding,
        None otherwise.

    Note
    ----
    **JIT Compatibility:**

    This function is fully JIT-compatible. When using padding (align_size provided),
    the output buffer is allocated with a fixed worst-case size that depends only on
    compile-time constants (num_out_tokens, num_experts, align_size). The actual
    padding offsets (pad_offsets) and aligned token counts (target_tokens_per_expert)
    are computed internally from the routing_map and can be traced values.

    The worst-case output size is:
    ((num_out_tokens + num_experts * (align_size - 1)) // align_size) * align_size
    This accounts for the maximum possible padding when each expert needs (align_size - 1)
    extra tokens to align, rounded down to align_size for buffer alignment.
    """
    use_padding = align_size is not None
    num_experts = routing_map.shape[-1]

    if use_padding:
        # Compute worst-case output size (compile-time constant)
        # This is the maximum possible size when each expert needs max padding
        worst_case_out_tokens = (
            (num_out_tokens + num_experts * (align_size - 1)) // align_size
        ) * align_size
    else:
        worst_case_out_tokens = num_out_tokens

    return _token_dispatch(
        inp, routing_map, probs, num_out_tokens, worst_case_out_tokens, align_size, use_padding
    )


@partial(jax.custom_vjp, nondiff_argnums=(1, 3, 4, 5, 6))
def _token_dispatch(
    inp: jnp.ndarray,
    routing_map: jnp.ndarray,
    probs: Optional[jnp.ndarray],
    num_out_tokens: int,
    worst_case_out_tokens: int,
    align_size: Optional[int],
    use_padding: bool,
) -> Tuple[
    jnp.ndarray,
    Optional[jnp.ndarray],
    jnp.ndarray,
    Optional[jnp.ndarray],
    Optional[jnp.ndarray],
]:
    """Internal token_dispatch with custom VJP."""
    (output, permuted_probs, row_id_map, pad_offsets, target_tokens_per_expert), _ = (
        _token_dispatch_fwd_rule(
            inp,
            routing_map,
            probs,
            num_out_tokens,
            worst_case_out_tokens,
            align_size,
            use_padding,
        )
    )
    return output, permuted_probs, row_id_map, pad_offsets, target_tokens_per_expert


def _token_dispatch_fwd_rule(
    inp: jnp.ndarray,
    routing_map: jnp.ndarray,
    probs: Optional[jnp.ndarray],
    num_out_tokens: int,
    worst_case_out_tokens: int,
    align_size: Optional[int],
    use_padding: bool,
) -> Tuple[
    Tuple[
        jnp.ndarray,
        Optional[jnp.ndarray],
        jnp.ndarray,
        Optional[jnp.ndarray],
        Optional[jnp.ndarray],
    ],
    Tuple[jnp.ndarray, Optional[jnp.ndarray], int, int, int, bool],
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

    if use_padding:
        # Compute tokens_per_expert internally from routing_map
        # This can be a traced value since output shape uses worst_case_out_tokens
        tokens_per_expert = jnp.sum(routing_map, axis=0).astype(jnp.int32)

        # Calculate aligned token counts per expert
        target_tokens_per_expert = (jnp.ceil(tokens_per_expert / align_size) * align_size).astype(
            jnp.int32
        )

        # Compute pad_offsets: cumulative padding for each expert
        # pad_offsets[i] = sum of (target - actual) for experts 0..i-1
        pad_lengths = target_tokens_per_expert - tokens_per_expert
        cum_pad = jnp.cumsum(pad_lengths)
        pad_offsets = jnp.concatenate([jnp.array([0], dtype=cum_pad.dtype), cum_pad[:-1]])

        # Use worst_case_out_tokens as the output buffer size (compile-time constant)
        # The actual used size is sum(target_tokens_per_expert), which may be smaller.
        # Unused positions will be zero-initialized by the kernel.
        output, permuted_probs = permute_with_mask_map_and_pad(
            inp,
            row_id_map,
            probs,
            pad_offsets,
            num_tokens,
            num_experts,
            worst_case_out_tokens,
            hidden_size,
        )
    else:
        # No padding
        pad_offsets = None
        target_tokens_per_expert = None

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
    residuals = (row_id_map, pad_offsets, num_tokens, num_experts, hidden_size, with_probs)
    return (
        output,
        permuted_probs,
        row_id_map,
        pad_offsets,
        target_tokens_per_expert,
    ), residuals


def _token_dispatch_bwd_rule(
    _routing_map: jnp.ndarray,
    _num_out_tokens: int,
    _worst_case_out_tokens: int,
    _align_size: Optional[int],
    _use_padding: bool,
    residuals: Tuple[jnp.ndarray, Optional[jnp.ndarray], int, int, int, bool],
    g: Tuple[
        jnp.ndarray,
        Optional[jnp.ndarray],
        jnp.ndarray,
        Optional[jnp.ndarray],
        Optional[jnp.ndarray],
    ],
) -> Tuple[jnp.ndarray, Optional[jnp.ndarray]]:
    """Backward pass rule for token_dispatch."""
    row_id_map, pad_offsets, num_tokens, num_experts, hidden_size, with_probs = residuals
    output_grad, permuted_probs_grad, _, _, _ = g  # Ignore row_id_map, pad_offsets, target grads

    # Backward: unpermute gradients (gather from experts back to tokens)
    if pad_offsets is not None:
        inp_grad, probs_grad = unpermute_with_mask_map_and_unpad(
            output_grad,
            row_id_map,
            None,  # No merging probs
            permuted_probs_grad if with_probs else None,
            pad_offsets,
            num_tokens,
            num_experts,
            hidden_size,
        )
    else:
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
    pad_offsets: Optional[jnp.ndarray] = None,
) -> jnp.ndarray:
    """
    Combine tokens from experts back to original token positions.

    This is the forward pass of MoE unpermutation. Tokens are gathered from
    experts and merged (optionally weighted by merging_probs).

    Optionally supports fused unpadding when `pad_offsets` is provided (from
    token_dispatch with padding enabled).

    Parameters
    ----------
    inp : jnp.ndarray
        Input tensor from experts of shape [num_out_tokens, hidden_size]
        (or [num_out_tokens_padded, hidden_size] when using unpadding).
    row_id_map : jnp.ndarray
        Row ID map from token_dispatch of shape [num_tokens, num_experts * 2 + 1].
    merging_probs : Optional[jnp.ndarray]
        Merging weights of shape [batch, sequence, num_experts] or [num_tokens, num_experts].
        If provided, tokens from different experts are weighted-summed.
        If None, tokens are summed directly.
    pad_offsets : Optional[jnp.ndarray]
        Per-expert cumulative padding offsets of shape [num_experts] from token_dispatch.
        If provided, fused unpadding will be performed. This should be the pad_offsets
        returned by token_dispatch when using padding.

    Returns
    -------
    output : jnp.ndarray
        Combined output tensor of shape [num_tokens, hidden_size].
    """
    return _token_combine(inp, row_id_map, merging_probs, pad_offsets)


@jax.custom_vjp
def _token_combine(
    inp: jnp.ndarray,
    row_id_map: jnp.ndarray,
    merging_probs: Optional[jnp.ndarray],
    pad_offsets: Optional[jnp.ndarray],
) -> jnp.ndarray:
    """Internal token_combine with custom VJP."""
    output, _ = _token_combine_fwd_rule(inp, row_id_map, merging_probs, pad_offsets)
    return output


def _token_combine_fwd_rule(
    inp: jnp.ndarray,
    row_id_map: jnp.ndarray,
    merging_probs: Optional[jnp.ndarray],
    pad_offsets: Optional[jnp.ndarray],
) -> Tuple[
    jnp.ndarray,
    Tuple[
        jnp.ndarray,
        Optional[jnp.ndarray],
        jnp.ndarray,
        Optional[jnp.ndarray],
        int,
        int,
        int,
        int,
    ],
]:
    """Forward pass rule for token_combine."""
    # Infer dimensions from row_id_map shape: [num_tokens, num_experts * 2 + 1]
    num_tokens = row_id_map.shape[0]
    num_experts = (row_id_map.shape[1] - 1) // 2
    hidden_size = inp.shape[-1]
    num_out_tokens = inp.shape[0]

    # Call triton extension with or without unpadding
    if pad_offsets is not None:
        output, _ = unpermute_with_mask_map_and_unpad(
            inp,
            row_id_map,
            merging_probs,
            None,  # No permuted probs to unpermute
            pad_offsets,
            num_tokens,
            num_experts,
            hidden_size,
        )
    else:
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
        pad_offsets,
        inp,
        merging_probs,
        num_tokens,
        num_experts,
        hidden_size,
        num_out_tokens,
    )
    return output, residuals


def _token_combine_bwd_rule(
    residuals: Tuple[
        jnp.ndarray,
        Optional[jnp.ndarray],
        jnp.ndarray,
        Optional[jnp.ndarray],
        int,
        int,
        int,
        int,
    ],
    g: jnp.ndarray,
) -> Tuple[jnp.ndarray, None, Optional[jnp.ndarray], None]:
    """Backward pass rule for token_combine.

    Returns gradients for: (inp, row_id_map, merging_probs, pad_offsets)
    row_id_map and pad_offsets are integer arrays, so their gradients are None.
    """
    (
        row_id_map,
        pad_offsets,
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
        if pad_offsets is not None:
            inp_grad, merging_probs_grad = unpermute_bwd_with_merging_probs_and_unpad(
                output_grad,
                row_id_map,
                fwd_input,
                merging_probs,
                pad_offsets,
                num_tokens,
                num_experts,
                num_out_tokens,
                hidden_size,
            )
            # The backward kernel only writes to positions that tokens map to.
            # Padded positions may contain uninitialized (NaN) values - replace with zeros.
            inp_grad = jnp.where(jnp.isnan(inp_grad), 0.0, inp_grad)
        else:
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
        if pad_offsets is not None:
            inp_grad, _ = permute_with_mask_map_and_pad(
                output_grad,
                row_id_map,
                None,
                pad_offsets,
                num_tokens,
                num_experts,
                num_out_tokens,
                hidden_size,
            )
            # The permute kernel only writes to positions that tokens map to.
            # Padded positions may contain uninitialized (NaN) values - replace with zeros.
            inp_grad = jnp.where(jnp.isnan(inp_grad), 0.0, inp_grad)
        else:
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

    # Return gradients for: inp, row_id_map, merging_probs, pad_offsets
    # row_id_map and pad_offsets are integer arrays, so their gradients are None
    return inp_grad, None, merging_probs_grad, None


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
