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
