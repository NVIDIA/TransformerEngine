# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""MoE Permutation API for JAX.

This module provides high-level token dispatch and combine operations for
Mixture of Experts (MoE) models with proper automatic differentiation support.

Two backends are offered:

* Fused, Triton-backed ``token_dispatch`` / ``token_combine`` - uses the
  Triton kernels in ``transformer_engine.jax.triton_extensions.permutation``.
* Unfused, pure-JAX ``unfused_token_dispatch`` / ``unfused_token_combine`` -
  uses only ``jnp.argsort`` + gather and is therefore compiled as plain XLA.

Both backends support optional alignment padding (``align_size > 0``) so each
expert's group size is a multiple of ``align_size``, which is required for
quantized grouped GEMMs.

Token Dispatch (Permute):
    - Forward: Permute tokens according to routing map (scatter to experts)
    - Backward: Unpermute gradients (gather from experts)

Token Combine (Unpermute):
    - Forward: Unpermute tokens and merge with weights (gather from experts)
    - Backward: Permute gradients (scatter to experts)
"""

from functools import partial
from typing import NamedTuple, Optional, Tuple

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
    "unfused_token_dispatch",
    "unfused_token_combine",
    "UnfusedPermState",
    # Ragged-all-to-all expert-parallelism helpers
    "compute_ragged_all_to_all_params",
    "compute_reverse_ragged_all_to_all_params",
    "local_permute_after_a2a",
    "local_unpermute_before_a2a",
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
    jnp.ndarray,
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
        Number of output tokens (rows in the permuted buffer, before padding). Must be > 0, e.g. int(jnp.sum(routing_map)) or num_tokens * top_k. Must be a compile-time constant for JIT.
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
    tokens_per_expert : jnp.ndarray
        Token counts per expert of shape [num_experts]:
        - Without padding: actual token counts (sum of routing_map columns)
        - With padding: aligned token counts (ceil(actual / align_size) * align_size)
        This gives the effective number of tokens per expert in the output buffer.

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

    Non-positive num_out_tokens (e.g. -1) raises AssertionError.
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

    assert num_out_tokens > 0, (
        f"token_dispatch requires num_out_tokens > 0, got {num_out_tokens}. "
        "Use int(jnp.sum(routing_map)) or num_tokens * top_k."
    )

    return _token_dispatch(
        inp, routing_map, probs, num_out_tokens, worst_case_out_tokens, align_size, use_padding
    )


@partial(jax.custom_vjp, nondiff_argnums=(3, 4, 5, 6))
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
    jnp.ndarray,
]:
    """Internal token_dispatch with custom VJP."""
    (output, permuted_probs, row_id_map, pad_offsets, tokens_per_expert), _ = (
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
    return output, permuted_probs, row_id_map, pad_offsets, tokens_per_expert


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
        jnp.ndarray,
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

    # Compute tokens_per_expert from routing_map (actual counts)
    # This is well-optimized by XLA as a simple column-wise reduction
    tokens_per_expert = jnp.sum(routing_map, axis=0).astype(jnp.int32)

    if use_padding:
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
            align_size=align_size,
        )

        # Return aligned counts when using padding
        out_tokens_per_expert = target_tokens_per_expert
    else:
        # No padding
        pad_offsets = None

        output, permuted_probs = permute_with_mask_map(
            inp,
            row_id_map,
            probs,
            num_tokens,
            num_experts,
            num_out_tokens,
            hidden_size,
        )

        # Return actual counts when not using padding
        out_tokens_per_expert = tokens_per_expert

    # Return (primals, residuals)
    # out_tokens_per_expert is:
    #   - target_tokens_per_expert (aligned) when using padding
    #   - tokens_per_expert (actual) when not using padding
    residuals = (row_id_map, pad_offsets, num_tokens, num_experts, hidden_size, with_probs)
    return (
        output,
        permuted_probs,
        row_id_map,
        pad_offsets,
        out_tokens_per_expert,
    ), residuals


def _token_dispatch_bwd_rule(
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
) -> Tuple[jnp.ndarray, None, Optional[jnp.ndarray]]:
    """Backward pass rule for token_dispatch.

    Returns gradients for (inp, routing_map, probs).
    routing_map gradient is None since it's a discrete routing decision.
    """
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

    # Return gradients for (inp, routing_map, probs)
    # routing_map is non-differentiable (discrete routing), so return None
    return inp_grad, None, probs_grad if with_probs else None


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
            # Note: align_size uses default (128) since buffer sizes are already
            # determined from forward pass (stored in residuals as num_out_tokens)
            inp_grad, _ = permute_with_mask_map_and_pad(
                output_grad,
                row_id_map,
                None,
                pad_offsets,
                num_tokens,
                num_experts,
                num_out_tokens,
                hidden_size,
                align_size=128,  # Default, sizes already computed in forward
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


@jax.custom_vjp
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
) -> Tuple[Tuple[jnp.ndarray, jnp.ndarray], Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, int, int]]:
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
    # Include split_sizes and sorted_indices in residuals since we removed nondiff_argnums
    residuals = (row_id_map, split_sizes, sorted_indices, num_tokens, hidden_size)
    return (output, row_id_map), residuals


def _sort_chunks_by_index_bwd_rule(
    residuals: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, int, int],
    g: Tuple[jnp.ndarray, jnp.ndarray],
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Backward pass rule for sort_chunks_by_index."""
    row_id_map, split_sizes, sorted_indices, num_tokens, hidden_size = residuals
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

    # Return gradients for all inputs: (inp, split_sizes, sorted_indices)
    # split_sizes and sorted_indices are integer arrays, so their gradients are zeros
    split_sizes_grad = jnp.zeros_like(split_sizes, dtype=split_sizes.dtype)
    sorted_indices_grad = jnp.zeros_like(sorted_indices, dtype=sorted_indices.dtype)

    return (inp_grad, split_sizes_grad, sorted_indices_grad)


_sort_chunks_by_index.defvjp(_sort_chunks_by_index_fwd_rule, _sort_chunks_by_index_bwd_rule)


# =============================================================================
# Unfused (pure-JAX) token dispatch / combine
# =============================================================================
#
# The following implementations use only ``jnp.argsort`` + gather and compile
# to plain XLA. They are a drop-in alternative to ``token_dispatch`` /
# ``token_combine`` above, differing only in input/output conventions (the
# fused path takes ``routing_map`` and ``sparse_probs`` over all experts; the
# unfused path takes dense ``selected_experts`` and per-token ``weights`` of
# shape ``[..., topk]``).


# -----------------------------------------------------------------------------
# Custom-VJP argsort-based gather.
#
# ``inputs[sort_indices]`` has a known inverse: ``output[argsort(sort_indices)]``.
# Using a custom VJP lets the backward pass exploit that inverse instead of
# relying on the compiler to discover it from the scatter-style default
# gradient of a gather, which is typically less efficient.


@jax.custom_vjp
def _sort_activations(inputs: jax.Array, sort_indices: jax.Array) -> jax.Array:
    """Sort ``inputs`` along the leading dim by ``sort_indices``."""
    assert inputs.shape[0] == sort_indices.shape[0], (
        f"inputs.shape[0]={inputs.shape[0]} must match"
        f" sort_indices.shape[0]={sort_indices.shape[0]}"
    )
    with jax.named_scope("unfused_sort_activations"):
        return inputs[sort_indices, ...]


def _sort_activations_fwd(
    inputs: jax.Array, sort_indices: jax.Array
) -> Tuple[jax.Array, jax.Array]:
    return _sort_activations(inputs, sort_indices), sort_indices


def _sort_activations_bwd(
    residuals: jax.Array, grads: jax.Array
) -> Tuple[jax.Array, None]:
    sort_indices = residuals
    # Inverse permutation: gather-by-argsort undoes the forward gather.
    return _sort_activations(grads, jnp.argsort(sort_indices)), None


_sort_activations.defvjp(_sort_activations_fwd, _sort_activations_bwd)


def _routing_map_to_selected_experts(
    sparse_probs: jnp.ndarray,
    routing_map: jnp.ndarray,
    topk: int,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Convert ``(sparse_probs, routing_map)`` from TE's fused router to the
    ``(selected_experts, weights)`` format consumed by
    :func:`unfused_token_dispatch`.

    ``routing_map`` is a boolean mask of shape ``[num_tokens, num_experts]``
    with exactly ``topk`` ``True`` positions per row.
    """
    # Argsort on a bool tensor places ``True`` rows last (False=0 < True=1),
    # so the last ``topk`` indices are the selected expert IDs.
    selected_experts = jnp.argsort(routing_map, axis=-1)[..., -topk:]
    weights = jnp.take_along_axis(sparse_probs, selected_experts, axis=-1)
    return selected_experts, weights


# -----------------------------------------------------------------------------
# Permutation state carried from dispatch to combine.


class UnfusedPermState(NamedTuple):
    """Opaque state produced by :func:`unfused_token_dispatch`.

    Attributes
    ----------
    sorted_indices : jnp.ndarray
        The argsort indices used in the forward sort. Needed to reverse the
        permutation in :func:`unfused_token_combine`. Shape
        ``[num_real_tokens + padding_size]``.
    num_real_tokens : int
        Number of real (non-padding) permuted tokens, i.e.
        ``batch_size * sequence_length * num_experts_per_tok``. Compile-time
        constant.
    padding_size : int
        Number of alignment-padding tokens appended to the sort buffer. Equals
        ``num_experts * (align_size - 1)`` when ``align_size > 0``, else ``0``.
        Compile-time constant.
    """

    sorted_indices: jax.Array
    num_real_tokens: int
    padding_size: int


# -----------------------------------------------------------------------------
# Dispatch (permute)


def unfused_token_dispatch(
    inputs: jnp.ndarray,
    selected_experts: jnp.ndarray,
    num_experts: int,
    num_experts_per_tok: int,
    align_size: int = 0,
    roll_to_expert_id: Optional[int] = None,
) -> Tuple[jnp.ndarray, UnfusedPermState, jnp.ndarray]:
    """Pure-JAX ``argsort``-based token dispatch.

    Parameters
    ----------
    inputs : jnp.ndarray
        Input tensor of shape ``[num_tokens, hidden_size]`` (or
        ``[batch, seq, hidden]``; it will be flattened).
    selected_experts : jnp.ndarray
        Per-token expert IDs, shape ``[num_tokens, num_experts_per_tok]`` (or
        ``[batch, seq, num_experts_per_tok]``). Integer dtype.
    num_experts : int
        Total number of experts.
    num_experts_per_tok : int
        Top-k. Must equal ``selected_experts.shape[-1]``.
    align_size : int, default 0
        Alignment for each expert's group size. ``0`` disables padding; a value
        ``> 0`` appends a static-size padding buffer so each resulting group
        size is a multiple of ``align_size`` (required for quantized grouped
        GEMM).
    roll_to_expert_id : Optional[int]
        If provided, rotates expert IDs by ``-roll_to_expert_id`` modulo
        ``num_experts`` before the sort (ring-of-experts EP). The returned
        ``group_sizes`` is rolled to match.

    Returns
    -------
    sorted_inputs : jnp.ndarray
        Permuted tokens grouped by expert, shape
        ``[num_real_tokens + padding_size, hidden_size]``.
    perm_state : UnfusedPermState
        State needed by :func:`unfused_token_combine`.
    group_sizes : jnp.ndarray
        Token count per expert, shape ``[num_experts]``. Each entry is a
        multiple of ``align_size`` when ``align_size > 0``.
    """
    assert num_experts_per_tok == selected_experts.shape[-1], (
        f"num_experts_per_tok={num_experts_per_tok} must match"
        f" selected_experts.shape[-1]={selected_experts.shape[-1]}"
    )
    assert align_size >= 0, f"align_size must be >= 0, got {align_size}"

    hidden_size = inputs.shape[-1]
    inputs_2d = inputs.reshape(-1, hidden_size)
    num_tokens = inputs_2d.shape[0]
    num_real_tokens = num_tokens * num_experts_per_tok

    flatten_selected_experts = jnp.ravel(selected_experts)

    if align_size > 0:
        # Per-expert token count, and how many extra tokens each expert needs
        # to become aligned to ``align_size``. Using
        # ``(align - count % align) % align`` gives 0 (not ``align``) when
        # already aligned, so we never exceed the per-expert slot capacity of
        # ``align_size - 1``.
        token_count_per_expert = jnp.bincount(
            flatten_selected_experts, length=num_experts
        )
        padding_tokens_required_per_expert = (
            (align_size - (token_count_per_expert % align_size)) % align_size
        )

        # Build a static-size padding buffer of shape
        # ``[num_experts * (align_size - 1)]``. Each expert ``i`` owns a slot
        # of ``align_size - 1`` positions (worst-case padding, which occurs
        # when ``token_count[i] % align_size == 1``). Within slot ``i``,
        # positions ``[0, padding_needed)`` are assigned expert ``i`` and act
        # as real padding; the rest are assigned to ``num_experts - 1`` as
        # overflow placeholders that keep the buffer statically sized for JIT.
        max_padding_per_expert = align_size - 1
        max_total_padding_size = num_experts * max_padding_per_expert
        positions = jnp.arange(max_total_padding_size)
        expert_for_pos = positions // max_padding_per_expert
        offset_in_slot = positions % max_padding_per_expert
        padding_needed = padding_tokens_required_per_expert[expert_for_pos]
        flatten_padding_selected_experts = jnp.where(
            offset_in_slot < padding_needed,
            expert_for_pos,
            num_experts - 1,
        )

        flatten_selected_experts = jnp.concatenate(
            [flatten_selected_experts, flatten_padding_selected_experts], axis=0
        )

        if roll_to_expert_id is not None:
            flatten_selected_experts = (
                flatten_selected_experts - roll_to_expert_id
            ) % num_experts

        sorted_selected_experts = jnp.argsort(flatten_selected_experts)

        replicated_inputs_2d = jnp.repeat(inputs_2d, num_experts_per_tok, axis=0)
        # Pad inputs with zeros so the sort operand shape matches the expanded
        # selected-experts vector.
        replicated_inputs_2d = jnp.pad(
            replicated_inputs_2d,
            pad_width=((0, max_total_padding_size), (0, 0)),
            mode="constant",
            constant_values=0.0,
        )

        sorted_inputs = _sort_activations(replicated_inputs_2d, sorted_selected_experts)

        # Compute ``group_sizes`` directly from counts rather than via
        # ``bincount(flatten_selected_experts)``: the overflow placeholder
        # tokens would inflate ``group_sizes[num_experts - 1]``, breaking the
        # alignment guarantee. Direct computation gives each expert exactly
        # ``ceil(count / align) * align`` tokens.
        group_sizes = token_count_per_expert + padding_tokens_required_per_expert

        if roll_to_expert_id is not None:
            group_sizes = jnp.roll(group_sizes, -roll_to_expert_id)

        padding_size = max_total_padding_size
    else:
        if roll_to_expert_id is not None:
            flatten_selected_experts = (
                flatten_selected_experts - roll_to_expert_id
            ) % num_experts

        sorted_selected_experts = jnp.argsort(flatten_selected_experts)

        replicated_inputs_2d = jnp.repeat(inputs_2d, num_experts_per_tok, axis=0)
        sorted_inputs = _sort_activations(replicated_inputs_2d, sorted_selected_experts)

        group_sizes = jnp.bincount(flatten_selected_experts, length=num_experts)
        if roll_to_expert_id is not None:
            group_sizes = jnp.roll(group_sizes, -roll_to_expert_id)

        padding_size = 0

    perm_state = UnfusedPermState(
        sorted_indices=sorted_selected_experts,
        num_real_tokens=num_real_tokens,
        padding_size=padding_size,
    )
    return sorted_inputs, perm_state, group_sizes


# -----------------------------------------------------------------------------
# Combine (unpermute + weighted sum)


def unfused_token_combine(
    expert_outputs: jnp.ndarray,
    perm_state: UnfusedPermState,
    routing_weights: jnp.ndarray,
    num_experts_per_tok: int,
    batch_size: int,
    sequence_length: int,
) -> jnp.ndarray:
    """Pure-JAX ``argsort``-based token combine.

    Reverses the permutation performed by :func:`unfused_token_dispatch`,
    strips any alignment-padding rows appended during dispatch, and applies a
    per-token weighted sum across the top-k experts.

    Parameters
    ----------
    expert_outputs : jnp.ndarray
        Output of the expert FFN, shape
        ``[num_real_tokens + padding_size, hidden_size]``.
    perm_state : UnfusedPermState
        State returned by :func:`unfused_token_dispatch`.
    routing_weights : jnp.ndarray
        Top-k routing weights, shape ``[batch*seq, num_experts_per_tok]``
        (or broadcastable to it after a ``reshape``).
    num_experts_per_tok : int
        Top-k.
    batch_size : int
        Original batch size.
    sequence_length : int
        Original sequence length.

    Returns
    -------
    output : jnp.ndarray
        Combined output tensor of shape ``[batch_size, sequence_length, hidden_size]``.
    """
    # Reverse the permutation: ``output[argsort(sorted_indices)]`` undoes
    # ``input[sorted_indices]``.
    unsort_intermediate = _sort_activations(
        expert_outputs,
        jnp.argsort(perm_state.sorted_indices),
    )

    # Strip alignment padding tokens appended during dispatch. After unsorting,
    # the first ``num_real_tokens`` rows hold the real per-(token, top-k)
    # outputs; any trailing rows are padding placeholders (zeros) and must be
    # discarded before the reshape below.
    if perm_state.padding_size > 0:
        unsort_intermediate = unsort_intermediate[: perm_state.num_real_tokens]

    hidden_size = unsort_intermediate.shape[-1]
    reshaped_weights = jnp.reshape(routing_weights, (-1, num_experts_per_tok))
    reshaped_intermediate = jnp.reshape(
        unsort_intermediate, (reshaped_weights.shape[0], num_experts_per_tok, hidden_size)
    )

    # Cast weights to match intermediate dtype (weighted sum happens in
    # intermediate dtype; callers can upcast before calling if higher
    # precision weight-sum is desired).
    reshaped_weights = reshaped_weights.astype(reshaped_intermediate.dtype)
    with jax.named_scope("unfused_weight_sum"):
        output = jnp.einsum(
            "BKE,BK -> BE",
            reshaped_intermediate,
            reshaped_weights,
        )
    return output.reshape(batch_size, sequence_length, hidden_size)


# =============================================================================
# Ragged-all-to-all expert-parallelism helpers
# =============================================================================
#
# These helpers support the ragged-all-to-all (A2A / A2Av) EP strategy used by
# :class:`transformer_engine.jax.flax.MoEBlock`. The forward EP path looks
# like::
#
#     route -> global_permute -> AG(group_sizes, ep)
#                             -> ragged_all_to_all(fwd, ep)
#                             -> local_permute_after_a2a
#                             -> grouped_dense x3 + activation
#                             -> local_unpermute_before_a2a
#                             -> ragged_all_to_all(reverse, ep)
#                             -> global_combine
#
# The two ``compute_*_ragged_all_to_all_params`` functions translate
# ``all_shards_tokens_per_expert`` (an EP-axis ``all_gather`` of each shard's
# global ``group_sizes``) into the four ``ragged_all_to_all`` arguments
# (``input_offsets``, ``send_sizes``, ``output_offsets``, ``recv_sizes``).
# ``shard_id`` may be a traced value (e.g. from :func:`jax.lax.axis_index`),
# which is why every slice into ``all_shards_tokens_per_expert`` uses
# :func:`jax.lax.dynamic_slice`.
#
# These functions are pure JAX (no MaxText / TE dependencies) and equivalent
# to :func:`maxtext.layers.te_permutation.compute_ragged_all_to_all_params`
# / :func:`compute_reverse_ragged_all_to_all_params`.


def compute_ragged_all_to_all_params(
    all_shards_tokens_per_expert: jnp.ndarray,
    shard_id: jnp.ndarray,
    num_expert_shards: int,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Forward-direction ragged_all_to_all parameters.

    Computes the four index/size arrays that :func:`jax.lax.ragged_all_to_all`
    consumes for the **forward** EP shuffle, where each shard sends its
    expert-grouped tokens to the shard that owns those experts.

    Parameters
    ----------
    all_shards_tokens_per_expert : jnp.ndarray
        Per-shard, per-expert token counts gathered across the EP axis. Shape
        ``[num_expert_shards, num_experts]`` and integer dtype.
    shard_id : jnp.ndarray
        Index of the current shard along the EP axis (typically
        :func:`jax.lax.axis_index` of the EP axis). Must be a 0-d integer.
    num_expert_shards : int
        Static EP-axis size. Must match
        ``all_shards_tokens_per_expert.shape[0]``.

    Returns
    -------
    input_offsets : jnp.ndarray
        Shape ``[num_expert_shards]``. Cumulative ``send_sizes`` (with a
        leading 0) -- where in the local source buffer each destination
        shard's chunk begins.
    send_sizes : jnp.ndarray
        Shape ``[num_expert_shards]``. ``send_sizes[i]`` is the number of
        tokens this shard sends to shard ``i`` (= the sum of token counts
        for the experts owned by shard ``i``).
    output_offsets : jnp.ndarray
        Shape ``[num_expert_shards]``. ``output_offsets[i]`` is the row in
        shard ``i``'s receive buffer where this shard's contribution should
        land. Sender-side semantics, per :func:`jax.lax.ragged_all_to_all`.
    recv_sizes : jnp.ndarray
        Shape ``[num_expert_shards]``. ``recv_sizes[i]`` is the number of
        tokens shard ``i`` sends to this shard.
    """
    num_experts = all_shards_tokens_per_expert.shape[1]
    assert num_experts % num_expert_shards == 0, (
        f"num_experts={num_experts} must be divisible by num_expert_shards"
        f"={num_expert_shards}"
    )
    local_expert_size = num_experts // num_expert_shards

    # This shard's row of the gathered table, reshaped so axis 0 indexes the
    # destination shard and axis 1 indexes its local experts.
    local_tokens_per_expert = jax.lax.dynamic_slice(
        all_shards_tokens_per_expert,
        start_indices=(shard_id, 0),
        slice_sizes=(1, num_experts),
    ).squeeze(0)
    local_reshaped = local_tokens_per_expert.reshape(
        num_expert_shards, local_expert_size
    )

    # send_sizes[i] = sum of token counts for shard i's experts in our buffer.
    send_sizes = jnp.sum(local_reshaped, axis=1)
    input_offsets = jnp.concatenate(
        [
            jnp.array([0], dtype=send_sizes.dtype),
            jnp.cumsum(send_sizes)[:-1],
        ]
    )

    # recv_sizes[i] = how many tokens shard i sends to this shard, i.e. the
    # sum across our local-expert columns of shard i's row.
    local_expert_start = shard_id * local_expert_size
    local_expert_columns = jax.lax.dynamic_slice(
        all_shards_tokens_per_expert,
        start_indices=(0, local_expert_start),
        slice_sizes=(num_expert_shards, local_expert_size),
    )
    recv_sizes = jnp.sum(local_expert_columns, axis=1)

    # output_offsets uses sender-side semantics for ragged_all_to_all:
    # output_offsets[j] = row in shard j's buffer where THIS shard's chunk
    # should be placed. That's the cumulative sum (over source shards 0..j-1)
    # of how many tokens those earlier source shards already sent to shard j.
    sends_to_target = jnp.sum(
        all_shards_tokens_per_expert.reshape(
            num_expert_shards, num_expert_shards, local_expert_size
        ),
        axis=2,
    )  # [src_shard, dst_shard]
    zero_row = jnp.zeros((1, num_expert_shards), dtype=sends_to_target.dtype)
    cumulated = jnp.cumsum(
        jnp.concatenate([zero_row, sends_to_target], axis=0),
        axis=0,
        dtype=sends_to_target.dtype,
    )  # [src_shard + 1, dst_shard]; row r = total sent by sources 0..r-1
    output_offsets = jax.lax.dynamic_slice(
        cumulated,
        start_indices=(shard_id, 0),
        slice_sizes=(1, num_expert_shards),
    ).squeeze(0)

    return input_offsets, send_sizes, output_offsets, recv_sizes


def compute_reverse_ragged_all_to_all_params(
    all_shards_tokens_per_expert: jnp.ndarray,
    shard_id: jnp.ndarray,
    num_expert_shards: int,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Reverse-direction ragged_all_to_all parameters.

    Mirror of :func:`compute_ragged_all_to_all_params` for the **reverse**
    EP shuffle that returns expert outputs to their source shards. The
    sender / receiver roles are swapped: what we received in the forward
    shuffle we now send back, and vice versa.

    Parameters and shapes are identical to
    :func:`compute_ragged_all_to_all_params`.
    """
    num_experts = all_shards_tokens_per_expert.shape[1]
    assert num_experts % num_expert_shards == 0, (
        f"num_experts={num_experts} must be divisible by num_expert_shards"
        f"={num_expert_shards}"
    )
    local_expert_size = num_experts // num_expert_shards

    local_expert_start = shard_id * local_expert_size

    # In reverse, what we received becomes what we send. send_sizes[i] is how
    # many tokens we send back to source shard i (= what shard i originally
    # sent us, summed across our local experts).
    local_expert_columns = jax.lax.dynamic_slice(
        all_shards_tokens_per_expert,
        start_indices=(0, local_expert_start),
        slice_sizes=(num_expert_shards, local_expert_size),
    )
    send_sizes = jnp.sum(local_expert_columns, axis=1)
    input_offsets = jnp.concatenate(
        [
            jnp.array([0], dtype=send_sizes.dtype),
            jnp.cumsum(send_sizes)[:-1],
        ]
    )

    # recv_sizes[i] = how many tokens we receive back from shard i (= what
    # we originally sent to shard i in the forward).
    local_tokens_per_expert = jax.lax.dynamic_slice(
        all_shards_tokens_per_expert,
        start_indices=(shard_id, 0),
        slice_sizes=(1, num_experts),
    ).squeeze(0)
    local_reshaped = local_tokens_per_expert.reshape(
        num_expert_shards, local_expert_size
    )
    recv_sizes = jnp.sum(local_reshaped, axis=1)

    # output_offsets: the reverse sends-to-target matrix is the transpose of
    # the forward one (row i = what shard i sends in reverse = what shard i
    # received in forward). Cumsum down source-shard axis, then index our row.
    fwd_sends_to = jnp.sum(
        all_shards_tokens_per_expert.reshape(
            num_expert_shards, num_expert_shards, local_expert_size
        ),
        axis=2,
    )  # forward: [src, dst]
    rev_sends_to = jnp.transpose(fwd_sends_to)  # reverse: [src, dst]
    zero_row = jnp.zeros((1, num_expert_shards), dtype=rev_sends_to.dtype)
    rev_cumulated = jnp.cumsum(
        jnp.concatenate([zero_row, rev_sends_to], axis=0),
        axis=0,
        dtype=rev_sends_to.dtype,
    )
    output_offsets = jax.lax.dynamic_slice(
        rev_cumulated,
        start_indices=(shard_id, 0),
        slice_sizes=(1, num_expert_shards),
    ).squeeze(0)

    return input_offsets, send_sizes, output_offsets, recv_sizes


# -----------------------------------------------------------------------------
# Local permute / unpermute
# -----------------------------------------------------------------------------
#
# After the forward ragged_all_to_all the receive buffer is laid out as
# ``[from_shard_0_chunk | from_shard_1_chunk | ... ]`` and within each chunk
# tokens are sorted by local-expert id. To feed ``grouped_dense`` we want
# ``[expert_0_block | expert_1_block | ... ]`` where each expert's block
# contains tokens from every source shard. ``local_permute_after_a2a``
# performs that reorder; ``local_unpermute_before_a2a`` undoes it before the
# reverse ragged_all_to_all.
#
# Implementation uses :func:`sort_chunks_by_index`, which is Triton-backed
# (see ``transformer_engine.jax.triton_extensions.permutation``) and has a
# paired custom-VJP backward. There is no pure-JAX alternative here -- the
# global :func:`unfused_token_dispatch` / :func:`token_dispatch` choice is
# unaffected by this; only the (small) post-A2A chunk reorder uses Triton
# unconditionally.


def local_permute_after_a2a(
    x_recv: jnp.ndarray,
    all_shards_tokens_per_expert: jnp.ndarray,
    shard_id: jnp.ndarray,
    num_expert_shards: int,
) -> Tuple[jnp.ndarray, jnp.ndarray, dict]:
    """Reorder tokens received via ragged_all_to_all so each local expert's
    tokens are contiguous.

    This is the EP-side complement to the global :func:`token_dispatch` /
    :func:`unfused_token_dispatch`. Internally uses
    :func:`sort_chunks_by_index` (Triton-backed) for both the forward sort
    and -- via :func:`local_unpermute_before_a2a` -- the inverse.

    Parameters
    ----------
    x_recv : jnp.ndarray
        Output of the forward ``ragged_all_to_all`` of shape
        ``[buffer_size, hidden_size]``. Layout: source-shard major, then
        local-expert id within each source chunk.
    all_shards_tokens_per_expert : jnp.ndarray
        Per-shard, per-expert token counts of shape
        ``[num_expert_shards, num_experts]``.
    shard_id : jnp.ndarray
        Current EP shard index (typically a traced
        :func:`jax.lax.axis_index`).
    num_expert_shards : int
        Static EP-axis size.

    Returns
    -------
    sorted_x : jnp.ndarray
        Tokens reordered into expert-major layout. Same shape as ``x_recv``.
    local_group_sizes : jnp.ndarray
        Per-local-expert token counts of shape ``[local_expert_size]``.
    state : dict
        Opaque state for :func:`local_unpermute_before_a2a`.
    """
    num_experts = all_shards_tokens_per_expert.shape[1]
    assert num_experts % num_expert_shards == 0, (
        f"num_experts={num_experts} must be divisible by num_expert_shards"
        f"={num_expert_shards}"
    )
    local_expert_size = num_experts // num_expert_shards
    local_expert_start = shard_id * local_expert_size
    local_expert_columns = jax.lax.dynamic_slice(
        all_shards_tokens_per_expert,
        start_indices=(0, local_expert_start),
        slice_sizes=(num_expert_shards, local_expert_size),
    )

    # Flat sizes in source-major order, matching the receive buffer layout:
    # [(s0,e0), (s0,e1), ..., (s1,e0), (s1,e1), ...]
    split_sizes = local_expert_columns.reshape(-1)

    # Permutation that maps source-major -> expert-major:
    #   original index = s * E_local + e
    #   target   index = e * num_shards + s
    indices_matrix = jnp.arange(
        num_expert_shards * local_expert_size, dtype=jnp.int32
    ).reshape(num_expert_shards, local_expert_size)
    sorted_chunk_indices = indices_matrix.T.reshape(-1)

    sorted_x, _ = sort_chunks_by_index(x_recv, split_sizes, sorted_chunk_indices)
    sorted_split_sizes = split_sizes[sorted_chunk_indices]
    inverse_chunk_indices = jnp.argsort(sorted_chunk_indices)
    local_group_sizes = jnp.sum(local_expert_columns, axis=0)
    state = {
        "sorted_split_sizes": sorted_split_sizes,
        "inverse_chunk_indices": inverse_chunk_indices,
    }
    return sorted_x, local_group_sizes, state


def local_unpermute_before_a2a(
    expert_outputs: jnp.ndarray,
    state: dict,
) -> jnp.ndarray:
    """Inverse of :func:`local_permute_after_a2a`.

    Parameters
    ----------
    expert_outputs : jnp.ndarray
        Output of the local expert FFN of shape ``[buffer_size, hidden_size]``,
        in expert-major layout.
    state : dict
        Opaque state returned by :func:`local_permute_after_a2a`.

    Returns
    -------
    unsorted_x : jnp.ndarray
        Tokens reordered back into source-shard-major layout, ready for the
        reverse ``ragged_all_to_all``. Same shape as ``expert_outputs``.
    """
    out, _ = sort_chunks_by_index(
        expert_outputs,
        state["sorted_split_sizes"],
        state["inverse_chunk_indices"],
    )
    return out
