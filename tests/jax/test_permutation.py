# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Tests for permutation Triton kernels and high-level APIs"""

import functools

import jax
import jax.numpy as jnp
import pytest

# High-level API with VJP support
from transformer_engine.jax.permutation import (
    token_dispatch,
    token_combine,
    sort_chunks_by_index,
)
from utils import assert_allclose, pytest_parametrize_wrapper


# =============================================================================
# Test parameter definitions with L0 (fast) and L2 (comprehensive) levels
# =============================================================================

# All dispatch/combine test cases
ALL_DISPATCH_COMBINE_CASES = [
    (128, 5, 128, 3),
    (1024, 8, 128, 8),
    (4096, 32, 1280, 2),
    (4096, 256, 4096, 6),
]
DISPATCH_COMBINE_CASES = {
    "L0": ALL_DISPATCH_COMBINE_CASES[0:2],
    "L2": ALL_DISPATCH_COMBINE_CASES,
}

# All sort chunks test cases
ALL_SORT_CHUNKS_CASES = [
    (8, 4096, 1280),
    (64, 4096, 4096),
    (256, 4096, 9216),
]
SORT_CHUNKS_CASES = {
    "L0": ALL_SORT_CHUNKS_CASES[0:2],
    "L2": ALL_SORT_CHUNKS_CASES,
}

# All dispatch/combine with padding test cases
ALL_DISPATCH_COMBINE_PADDING_CASES = [
    (128, 5, 128, 3, 8),
    (1024, 8, 128, 8, 16),
    (4096, 32, 1280, 2, 128),
    (4096, 256, 4096, 6, 16),
]
DISPATCH_COMBINE_PADDING_CASES = {
    "L0": ALL_DISPATCH_COMBINE_PADDING_CASES[0:2],
    "L2": ALL_DISPATCH_COMBINE_PADDING_CASES,
}

# Dtypes for testing
ALL_DTYPES = [jnp.float32, jnp.bfloat16]
DTYPES = {
    "L0": ALL_DTYPES,
    "L2": ALL_DTYPES,
}

# With probs options
ALL_WITH_PROBS = [True, False]
WITH_PROBS = {
    "L0": [True],
    "L2": ALL_WITH_PROBS,
}


def reference_make_row_id_map(
    routing_map: jnp.ndarray,
) -> jnp.ndarray:
    """
    Vectorized reference implementation of make_row_id_map using JAX primitives.

    Parameters
    ----------
    routing_map : jnp.ndarray
        Input tensor of shape [num_tokens, num_experts]. Mask indicating which experts
        are routed to which tokens (1 = routed, 0 = not routed).

    Returns
    -------
    row_id_map : jnp.ndarray
        The row_id_map for the permutation of shape [num_tokens, num_experts * 2 + 1].
    """
    num_tokens, num_experts = routing_map.shape

    # For each expert, compute cumulative sum to get destination indices
    cumsum_per_expert = jnp.cumsum(routing_map, axis=0)

    # Compute total tokens per expert and expert offsets
    tokens_per_expert = jnp.sum(routing_map, axis=0)
    expert_offsets = jnp.concatenate([jnp.array([0]), jnp.cumsum(tokens_per_expert)[:-1]])

    # Compute destination rows for all (token, expert) pairs
    # dest_row[i, j] = expert_offsets[j] + cumsum_per_expert[i, j] - 1 if routed, else -1
    dest_rows_all = (expert_offsets[None, :] + cumsum_per_expert - 1) * routing_map + (-1) * (
        1 - routing_map
    )

    # Count routed experts per token
    n_routed_per_token = jnp.sum(routing_map, axis=1)

    # For each token, we need to sort by descending dest_row and pack into row_id_map
    # Use a large negative value for non-routed experts so they sort to the end
    sort_keys = jnp.where(routing_map == 1, -dest_rows_all, jnp.iinfo(jnp.int32).max)
    sorted_expert_indices = jnp.argsort(sort_keys, axis=1)

    # Gather the sorted destination rows and expert indices using advanced indexing
    # Create indices for gathering
    token_idx = jnp.broadcast_to(jnp.arange(num_tokens)[:, None], (num_tokens, num_experts))
    sorted_dest_rows = dest_rows_all[token_idx, sorted_expert_indices]

    # Build row_id_map: [dest_row_0, ..., dest_row_{E-1}, expert_idx_0, ..., expert_idx_{E-1}, n_routed]
    row_id_map = jnp.concatenate(
        [
            sorted_dest_rows.astype(jnp.int32),
            sorted_expert_indices.astype(jnp.int32),
            n_routed_per_token.astype(jnp.int32)[:, None],
        ],
        axis=1,
    )

    return row_id_map


def _reference_permute_impl(
    inp: jnp.ndarray,
    row_id_map: jnp.ndarray,
    probs: jnp.ndarray,
    num_out_tokens: int,
) -> tuple:
    """
    Vectorized internal helper for reference permutation implementation.

    Parameters
    ----------
    inp : jnp.ndarray
        Input tensor of shape [num_tokens, hidden_size].
    row_id_map : jnp.ndarray
        The token to expert mapping tensor of shape [num_tokens, num_experts * 2 + 1].
    probs : jnp.ndarray
        The probabilities of the input tensor.
    num_out_tokens : int
        Number of tokens in the permuted tensor.

    Returns
    -------
    output : jnp.ndarray
        Permuted output tensor of shape [num_out_tokens, hidden_size].
    permuted_probs : jnp.ndarray
        Permuted probabilities if probs was provided, None otherwise.
    """
    num_tokens, hidden_size = inp.shape
    num_experts = (row_id_map.shape[1] - 1) // 2

    # Extract destination rows, expert indices, and n_routed from row_id_map
    dest_rows = row_id_map[:, :num_experts]  # [num_tokens, num_experts]
    expert_indices = row_id_map[:, num_experts : 2 * num_experts]  # [num_tokens, num_experts]
    n_routed = row_id_map[:, 2 * num_experts]  # [num_tokens]

    # Create mask for valid entries: slot_idx < n_routed[token]
    # The kernel's row_id_map only guarantees valid data in the first n_routed slots
    # (slots beyond n_routed may contain garbage, not -1)
    slot_indices = jnp.arange(num_experts)[None, :]  # [1, num_experts]
    valid_mask = slot_indices < n_routed[:, None]  # [num_tokens, num_experts]

    # Flatten for scatter operations
    flat_dest_rows = dest_rows.flatten()  # [num_tokens * num_experts]
    flat_valid_mask = valid_mask.flatten()
    flat_token_indices = jnp.repeat(jnp.arange(num_tokens), num_experts)
    flat_expert_indices = expert_indices.flatten()

    # Set invalid dest_rows to num_out_tokens (out of bounds, will be dropped)
    # This avoids overwriting valid entries at index 0 with zeros
    flat_dest_rows_clamped = jnp.where(flat_valid_mask, flat_dest_rows, num_out_tokens)

    # Gather input tokens and scatter to output
    output = jnp.zeros((num_out_tokens, hidden_size), dtype=inp.dtype)
    gathered_inp = inp[flat_token_indices]  # [num_tokens * num_experts, hidden_size]

    # Use segment_sum-like operation via scatter
    # For each valid (token, expert) pair, write inp[token] to output[dest_row]
    # Invalid entries target num_out_tokens and get dropped by mode="drop"
    output = output.at[flat_dest_rows_clamped].set(
        gathered_inp,
        mode="drop",
    )

    permuted_probs = None
    if probs is not None:
        permuted_probs = jnp.zeros((num_out_tokens,), dtype=probs.dtype)

        # Vectorized approach: gather probs and scatter to permuted_probs
        if probs.ndim == 1:
            flat_probs = probs[flat_token_indices]
        else:
            # Clamp invalid expert indices to 0 to avoid wraparound indexing with -1
            # The result for invalid entries will be ignored anyway since they target num_out_tokens
            # Cast to int32 explicitly for consistent indexing behavior
            flat_expert_indices_clamped = jnp.where(flat_valid_mask, flat_expert_indices, 0).astype(
                jnp.int32
            )
            flat_probs = probs[flat_token_indices.astype(jnp.int32), flat_expert_indices_clamped]

        # Invalid entries target num_out_tokens and get dropped by mode="drop"
        permuted_probs = permuted_probs.at[flat_dest_rows_clamped.astype(jnp.int32)].set(
            flat_probs,
            mode="drop",
        )

    return output, permuted_probs


def _reference_unpermute_impl(
    inp: jnp.ndarray,
    row_id_map: jnp.ndarray,
    merging_probs: jnp.ndarray,
    permuted_probs: jnp.ndarray,
) -> tuple:
    """
    Vectorized internal helper for reference unpermutation implementation.

    Parameters
    ----------
    inp : jnp.ndarray
        Input tensor of shape [num_out_tokens, hidden_size].
    row_id_map : jnp.ndarray
        The token to expert mapping tensor of shape [num_tokens, num_experts * 2 + 1].
    merging_probs : jnp.ndarray
        The merging probabilities for weighted reduction.
    permuted_probs : jnp.ndarray
        The permuted probabilities.

    Returns
    -------
    output : jnp.ndarray
        Unpermuted output tensor of shape [num_tokens, hidden_size].
    unpermuted_probs : jnp.ndarray
        Unpermuted probabilities if permuted_probs was provided, None otherwise.
    """
    num_tokens = row_id_map.shape[0]
    num_experts = (row_id_map.shape[1] - 1) // 2

    # Extract source rows, expert indices, and n_routed from row_id_map
    src_rows = row_id_map[:, :num_experts]  # [num_tokens, num_experts]
    expert_indices = row_id_map[:, num_experts : 2 * num_experts]  # [num_tokens, num_experts]
    n_routed = row_id_map[:, 2 * num_experts]  # [num_tokens]

    # Create mask for valid entries: slot_idx < n_routed[token]
    # The kernel's row_id_map only guarantees valid data in the first n_routed slots
    slot_indices = jnp.arange(num_experts)[None, :]  # [1, num_experts]
    valid_mask = slot_indices < n_routed[:, None]  # [num_tokens, num_experts]

    # Clamp invalid src_rows to 0 (they won't be used due to masking)
    src_rows_clamped = jnp.where(valid_mask, src_rows, 0)

    # Gather input from permuted positions
    gathered_inp = inp[src_rows_clamped]  # [num_tokens, num_experts, hidden_size]

    # Apply merging probs if provided
    if merging_probs is not None:
        # Gather the merging weights for each (token, expert) pair using advanced indexing
        token_idx = jnp.broadcast_to(jnp.arange(num_tokens)[:, None], (num_tokens, num_experts))
        weights = merging_probs[token_idx, expert_indices]  # [num_tokens, num_experts]
        gathered_inp = gathered_inp * weights[:, :, None]

    # Mask out invalid entries and sum across experts
    gathered_inp = jnp.where(valid_mask[:, :, None], gathered_inp, 0.0)
    output = jnp.sum(gathered_inp, axis=1)  # [num_tokens, hidden_size]

    unpermuted_probs = None
    if permuted_probs is not None:
        gathered_probs = permuted_probs[src_rows_clamped]  # [num_tokens, num_experts]
        unpermuted_probs = jnp.zeros((num_tokens, num_experts), dtype=permuted_probs.dtype)
        token_idx = jnp.broadcast_to(jnp.arange(num_tokens)[:, None], (num_tokens, num_experts))
        unpermuted_probs = unpermuted_probs.at[token_idx, expert_indices].set(
            jnp.where(valid_mask, gathered_probs, 0.0)
        )

    return output, unpermuted_probs


def reference_token_dispatch(
    inp: jnp.ndarray,
    routing_map: jnp.ndarray,
    num_out_tokens: int,
    probs: jnp.ndarray = None,
) -> tuple:
    """
    Reference implementation of token_dispatch using JAX primitives.

    Parameters
    ----------
    inp : jnp.ndarray
        Input tensor of shape [num_tokens, hidden_size].
    routing_map : jnp.ndarray
        Routing mask of shape [num_tokens, num_experts].
    num_out_tokens : int
        Number of tokens in the permuted tensor.
    probs : jnp.ndarray, optional
        The probabilities of shape [num_tokens, num_experts].

    Returns
    -------
    output : jnp.ndarray
        Permuted output tensor of shape [num_out_tokens, hidden_size].
    permuted_probs : jnp.ndarray or None
        Permuted probabilities of shape [num_out_tokens], or None if probs not provided.
    row_id_map : jnp.ndarray
        The row_id_map for the permutation.
    """
    row_id_map = reference_make_row_id_map(routing_map)
    output, permuted_probs = _reference_permute_impl(inp, row_id_map, probs, num_out_tokens)

    return output, permuted_probs, row_id_map


def reference_token_combine(
    inp: jnp.ndarray,
    row_id_map: jnp.ndarray,
    merging_probs: jnp.ndarray,
) -> jnp.ndarray:
    """
    Reference implementation of token_combine using JAX primitives.

    Parameters
    ----------
    inp : jnp.ndarray
        Input tensor of shape [num_out_tokens, hidden_size].
    row_id_map : jnp.ndarray
        The token to expert mapping tensor of shape [num_tokens, num_experts * 2 + 1].
    merging_probs : jnp.ndarray
        The merging probabilities for weighted reduction.

    Returns
    -------
    output : jnp.ndarray
        Unpermuted output tensor of shape [num_tokens, hidden_size].
    """
    output, _ = _reference_unpermute_impl(inp, row_id_map, merging_probs, None)

    return output


def reference_make_chunk_sort_map(
    split_sizes: jnp.ndarray,
    sorted_indices: jnp.ndarray,
    num_tokens: int,
) -> jnp.ndarray:
    """
    Vectorized reference implementation of make_chunk_sort_map using JAX primitives.

    Parameters
    ----------
    split_sizes : jnp.ndarray
        The sizes of the chunks of shape [num_splits,].
    sorted_indices : jnp.ndarray
        The indices of the sorted chunks of shape [num_splits,].
    num_tokens : int
        Number of tokens.

    Returns
    -------
    row_id_map : jnp.ndarray
        Row ID map for chunk sorting of shape [num_tokens,].
    """
    # Compute source chunk boundaries (cumulative sum of original split_sizes)
    src_cumsum = jnp.concatenate([jnp.array([0]), jnp.cumsum(split_sizes)])

    # Compute destination chunk boundaries based on sorted order
    sorted_sizes = split_sizes[sorted_indices]
    dest_cumsum = jnp.concatenate([jnp.array([0]), jnp.cumsum(sorted_sizes)])

    # For each source chunk, compute its destination offset
    # inverse_indices[i] = position of chunk i in sorted order
    inverse_indices = jnp.argsort(sorted_indices)
    dest_offsets = dest_cumsum[inverse_indices]

    # Create row_id_map: for each token position, compute its destination
    # First, figure out which chunk each position belongs to
    position_indices = jnp.arange(num_tokens)

    # chunk_ids[i] = which chunk position i belongs to
    chunk_ids = jnp.searchsorted(src_cumsum[1:], position_indices, side="right")

    # within_chunk_offset[i] = position i's offset within its chunk
    within_chunk_offset = position_indices - src_cumsum[chunk_ids]

    # destination[i] = dest_offsets[chunk_ids[i]] + within_chunk_offset[i]
    row_id_map = dest_offsets[chunk_ids] + within_chunk_offset

    return row_id_map.astype(jnp.int32)


def reference_sort_chunks_by_map(
    inp: jnp.ndarray,
    row_id_map: jnp.ndarray,
    probs: jnp.ndarray,
    is_forward: bool,
) -> tuple:
    """
    Vectorized reference implementation of sort_chunks_by_map using JAX primitives.

    Parameters
    ----------
    inp : jnp.ndarray
        Input tensor of shape [num_tokens, hidden_size].
    row_id_map : jnp.ndarray
        The token to destination mapping of shape [num_tokens,].
    probs : jnp.ndarray
        The probabilities.
    is_forward : bool
        Whether this is forward or backward.

    Returns
    -------
    output : jnp.ndarray
        Sorted output tensor of shape [num_tokens, hidden_size].
    permuted_probs : jnp.ndarray
        Sorted probabilities if probs was provided, None otherwise.
    """
    num_tokens = inp.shape[0]
    hidden_size = inp.shape[1]

    if is_forward:
        # Forward: scatter inp[src] to output[dest] where dest = row_id_map[src]
        output = jnp.zeros((num_tokens, hidden_size), dtype=inp.dtype)
        output = output.at[row_id_map].set(inp)
        if probs is not None:
            permuted_probs = jnp.zeros((num_tokens,), dtype=probs.dtype)
            permuted_probs = permuted_probs.at[row_id_map].set(probs)
        else:
            permuted_probs = None
    else:
        # Backward: gather output[dest] = inp[src] where src = row_id_map[dest]
        output = inp[row_id_map]
        if probs is not None:
            permuted_probs = probs[row_id_map]
        else:
            permuted_probs = None

    return output, permuted_probs


class TestHighLevelPermutationAPI:
    """Test high-level permutation APIs (token_dispatch, token_combine, etc.)

    These tests compare the high-level APIs against reference implementations
    to verify correctness of both forward and backward passes.
    """

    @staticmethod
    def generate_routing_map(
        num_tokens: int,
        num_experts: int,
        tokens_per_expert: int = 2,
        key: jax.Array = None,
    ):
        """Generate random routing map for testing"""
        if key is None:
            key = jax.random.PRNGKey(0)

        routing_map = jnp.zeros((num_tokens, num_experts), dtype=jnp.int32)
        for token_idx in range(num_tokens):
            key, subkey = jax.random.split(key)
            expert_indices = jax.random.choice(
                subkey, num_experts, shape=(tokens_per_expert,), replace=False
            )
            routing_map = routing_map.at[token_idx, expert_indices].set(1)

        return routing_map

    @pytest_parametrize_wrapper(
        "num_tokens,num_experts,hidden_size,tokens_per_expert",
        DISPATCH_COMBINE_CASES,
    )
    @pytest_parametrize_wrapper("dtype", DTYPES)
    @pytest_parametrize_wrapper("with_probs", WITH_PROBS)
    def test_token_dispatch(
        self, num_tokens, num_experts, hidden_size, tokens_per_expert, dtype, with_probs
    ):
        """
        Individual test for token_dispatch forward and backward passes.

        This test validates dispatch in isolation to catch errors that might be
        masked when combined with token_combine in the roundtrip test.

        Uses value_and_grad to validate both forward (via loss comparison) and
        backward (via gradient comparison) passes against reference implementation.
        """
        key = jax.random.PRNGKey(42)

        # Generate routing map
        routing_map = self.generate_routing_map(num_tokens, num_experts, tokens_per_expert, key)
        num_out_tokens = int(jnp.sum(routing_map))

        # Generate input data
        key, inp_key, prob_key = jax.random.split(key, 3)
        inp = jax.random.uniform(
            inp_key, (num_tokens, hidden_size), dtype=dtype, minval=-1.0, maxval=1.0
        )

        # Generate probs if needed (minval > 0 to avoid kernel's special prob==0 handling)
        probs = None
        if with_probs:
            probs = jax.random.uniform(
                prob_key, (num_tokens, num_experts), dtype=dtype, minval=0.1, maxval=1.0
            )

        # Generate reference row_id_map for comparison
        ref_row_id_map = reference_make_row_id_map(routing_map)

        # =====================================================================
        # Test forward and backward pass using value_and_grad
        # (value validates forward, grad validates backward)
        # =====================================================================
        if with_probs:

            @jax.jit
            def dispatch_loss(x, p):
                out, perm_probs, _, _, _ = token_dispatch(x, routing_map, num_out_tokens, probs=p)
                return jnp.sum(out**2) + jnp.sum(perm_probs**2)

            @jax.jit
            def ref_dispatch_loss(x, p):
                out, perm_probs = _reference_permute_impl(x, ref_row_id_map, p, num_out_tokens)
                return jnp.sum(out**2) + jnp.sum(perm_probs**2)

            loss_val, (inp_grad, probs_grad) = jax.value_and_grad(dispatch_loss, argnums=(0, 1))(
                inp, probs
            )
            ref_loss_val, (ref_inp_grad, ref_probs_grad) = jax.value_and_grad(
                ref_dispatch_loss, argnums=(0, 1)
            )(inp, probs)

            # Validate forward loss matches
            assert_allclose(loss_val, ref_loss_val, dtype=dtype)
            # Validate gradients
            assert_allclose(inp_grad, ref_inp_grad, dtype=dtype)
            assert_allclose(probs_grad, ref_probs_grad, dtype=dtype)
        else:

            @jax.jit
            def dispatch_loss_no_probs(x):
                out, _, _, _, _ = token_dispatch(x, routing_map, num_out_tokens)
                return jnp.sum(out**2)

            @jax.jit
            def ref_dispatch_loss_no_probs(x):
                out, _ = _reference_permute_impl(x, ref_row_id_map, None, num_out_tokens)
                return jnp.sum(out**2)

            loss_val, inp_grad = jax.value_and_grad(dispatch_loss_no_probs)(inp)
            ref_loss_val, ref_inp_grad = jax.value_and_grad(ref_dispatch_loss_no_probs)(inp)

            # Validate forward loss matches
            assert_allclose(loss_val, ref_loss_val, dtype=dtype)
            # Validate gradients
            assert_allclose(inp_grad, ref_inp_grad, dtype=dtype)

    # =========================================================================
    # Consolidated dispatch + combine tests
    # =========================================================================

    @pytest_parametrize_wrapper(
        "num_tokens,num_experts,hidden_size,tokens_per_expert",
        DISPATCH_COMBINE_CASES,
    )
    @pytest_parametrize_wrapper("dtype", DTYPES)
    @pytest_parametrize_wrapper("with_probs", WITH_PROBS)
    def test_dispatch_and_combine(
        self, num_tokens, num_experts, hidden_size, tokens_per_expert, dtype, with_probs
    ):
        """
        Comprehensive test for token_dispatch and token_combine.

        Tests:
        1. Dispatch forward pass against reference (element-by-element)
        2. Dispatch backward pass against reference
        3. Combine forward pass against reference (element-by-element)
        4. Combine backward pass against reference
        5. Roundtrip: dispatch + combine recovers original input
        6. row_id_map n_routed column validation
        7. Probs permutation (when with_probs=True)
        """
        key = jax.random.PRNGKey(42)

        # Generate routing map
        routing_map = self.generate_routing_map(num_tokens, num_experts, tokens_per_expert, key)
        num_out_tokens = int(jnp.sum(routing_map))

        # Generate input data
        key, inp_key, prob_key, merge_key = jax.random.split(key, 4)
        inp = jax.random.uniform(
            inp_key, (num_tokens, hidden_size), dtype=dtype, minval=-1.0, maxval=1.0
        )

        # Generate probs if needed (minval > 0 to avoid kernel's special prob==0 handling)
        probs = None
        if with_probs:
            probs = jax.random.uniform(
                prob_key, (num_tokens, num_experts), dtype=dtype, minval=0.1, maxval=1.0
            )

        # Generate merging probs (normalized per token)
        merging_probs = jax.random.uniform(
            merge_key, (num_tokens, num_experts), dtype=dtype, minval=0.1, maxval=1.0
        )
        merging_probs = merging_probs * routing_map.astype(dtype)  # Zero out non-routed
        merging_probs = merging_probs / jnp.maximum(
            jnp.sum(merging_probs, axis=1, keepdims=True), 1e-8
        )

        # =====================================================================
        # Test 1: Dispatch forward pass
        # =====================================================================
        output, permuted_probs, row_id_map, _, _ = token_dispatch(
            inp, routing_map, num_out_tokens, probs=probs
        )
        ref_output, ref_permuted_probs = _reference_permute_impl(
            inp, row_id_map, probs, num_out_tokens
        )

        # Validate row_id_map structure: n_routed column should match routing_map sum
        n_routed_actual = row_id_map[:, -1]
        n_routed_expected = jnp.sum(routing_map, axis=1)
        assert jnp.array_equal(
            n_routed_actual, n_routed_expected
        ), "make_row_id_map n_routed column mismatch"

        # Compare dispatch output
        assert_allclose(output, ref_output, dtype=dtype)
        if with_probs:
            assert_allclose(permuted_probs, ref_permuted_probs, dtype=dtype)

        # =====================================================================
        # Test 2: Dispatch backward pass
        # =====================================================================
        if with_probs:

            @jax.jit
            def dispatch_loss(x, p):
                out, perm_probs, _, _, _ = token_dispatch(x, routing_map, num_out_tokens, probs=p)
                return jnp.sum(out**2) + jnp.sum(perm_probs**2)

            @jax.jit
            def ref_dispatch_loss(x, p):
                out, perm_probs = _reference_permute_impl(x, row_id_map, p, num_out_tokens)
                return jnp.sum(out**2) + jnp.sum(perm_probs**2)

            _, (inp_grad, probs_grad) = jax.value_and_grad(dispatch_loss, argnums=(0, 1))(
                inp, probs
            )
            _, (ref_inp_grad, ref_probs_grad) = jax.value_and_grad(
                ref_dispatch_loss, argnums=(0, 1)
            )(inp, probs)
            assert_allclose(inp_grad, ref_inp_grad, dtype=dtype)
            assert_allclose(probs_grad, ref_probs_grad, dtype=dtype)
        else:

            @jax.jit
            def dispatch_loss_no_probs(x):
                out, _, _, _, _ = token_dispatch(x, routing_map, num_out_tokens)
                return jnp.sum(out**2)

            @jax.jit
            def ref_dispatch_loss_no_probs(x):
                out, _ = _reference_permute_impl(x, row_id_map, None, num_out_tokens)
                return jnp.sum(out**2)

            _, inp_grad = jax.value_and_grad(dispatch_loss_no_probs)(inp)
            _, ref_inp_grad = jax.value_and_grad(ref_dispatch_loss_no_probs)(inp)
            assert_allclose(inp_grad, ref_inp_grad, dtype=dtype)

        # =====================================================================
        # Test 3: Combine forward pass
        # =====================================================================
        combined = token_combine(output, row_id_map, merging_probs)
        ref_combined = _reference_unpermute_impl(output, row_id_map, merging_probs, None)[0]
        assert_allclose(combined, ref_combined, dtype=dtype)

        # =====================================================================
        # Test 4: Combine backward pass
        # =====================================================================

        @jax.jit
        def combine_loss(x):
            return jnp.sum(token_combine(x, row_id_map, merging_probs) ** 2)

        @jax.jit
        def ref_combine_loss(x):
            return jnp.sum(_reference_unpermute_impl(x, row_id_map, merging_probs, None)[0] ** 2)

        _, combine_grad = jax.value_and_grad(combine_loss)(output)
        _, ref_combine_grad = jax.value_and_grad(ref_combine_loss)(output)
        assert_allclose(combine_grad, ref_combine_grad, dtype=dtype)

        # =====================================================================
        # Test 5: Roundtrip (dispatch + combine = original)
        # =====================================================================
        # Use uniform merging probs for perfect roundtrip
        uniform_merging_probs = routing_map.astype(dtype) / jnp.maximum(
            jnp.sum(routing_map, axis=1, keepdims=True), 1.0
        )

        @jax.jit
        def roundtrip(x):
            dispatched, _, rid_map, _, _ = token_dispatch(x, routing_map, num_out_tokens)
            return token_combine(dispatched, rid_map, uniform_merging_probs)

        roundtrip_output = roundtrip(inp)
        assert_allclose(roundtrip_output, inp, dtype=dtype)

    # =========================================================================
    # sort_chunks_by_index tests
    # =========================================================================

    @pytest_parametrize_wrapper(
        "num_splits,total_tokens,hidden_size",
        SORT_CHUNKS_CASES,
    )
    @pytest_parametrize_wrapper("dtype", DTYPES)
    def test_sort_chunks_by_index(self, num_splits, total_tokens, hidden_size, dtype):
        """Test sort_chunks_by_index forward and backward pass against reference"""
        key = jax.random.PRNGKey(42)

        # Generate random split sizes
        key, size_key = jax.random.split(key)
        split_sizes = jax.random.randint(size_key, (num_splits,), 10, total_tokens // num_splits)
        split_sizes = split_sizes.at[-1].set(total_tokens - jnp.sum(split_sizes[:-1]))

        # Generate sorted indices
        key, sort_key = jax.random.split(key)
        sorted_indices = jax.random.permutation(sort_key, num_splits)

        # Generate input data
        key, inp_key = jax.random.split(key)
        inp = jax.random.uniform(
            inp_key, (total_tokens, hidden_size), dtype=dtype, minval=-1.0, maxval=1.0
        )

        # Get reference row_id_map
        row_id_map = reference_make_chunk_sort_map(split_sizes, sorted_indices, total_tokens)

        # Define loss functions (JIT compiled for performance)
        @jax.jit
        def loss_fn(x):
            output, _ = sort_chunks_by_index(x, split_sizes, sorted_indices)
            return jnp.sum(output**2)

        @jax.jit
        def ref_loss_fn(x):
            output, _ = reference_sort_chunks_by_map(x, row_id_map, None, is_forward=True)
            return jnp.sum(output**2)

        # Test forward pass
        output, _ = sort_chunks_by_index(inp, split_sizes, sorted_indices)
        ref_output, _ = reference_sort_chunks_by_map(inp, row_id_map, None, is_forward=True)

        # Test backward pass with JIT
        loss_val, computed_grad = jax.value_and_grad(loss_fn)(inp)
        ref_loss_val, ref_grad = jax.value_and_grad(ref_loss_fn)(inp)

        # Compare forward and backward
        assert_allclose(output, ref_output)
        assert_allclose(loss_val, ref_loss_val)
        assert_allclose(computed_grad, ref_grad)

    # =========================================================================
    # Consolidated dispatch + combine with padding tests
    # =========================================================================

    @pytest_parametrize_wrapper(
        "num_tokens,num_experts,hidden_size,topk,align_size",
        DISPATCH_COMBINE_PADDING_CASES,
    )
    @pytest_parametrize_wrapper("dtype", DTYPES)
    @pytest_parametrize_wrapper("with_probs", WITH_PROBS)
    def test_dispatch_and_combine_with_padding(
        self, num_tokens, num_experts, hidden_size, topk, align_size, dtype, with_probs
    ):
        """
        Comprehensive test for token_dispatch and token_combine with padding/unpadding.

        Tests:
        1. Dispatch with padding: output shape and alignment
        2. Dispatch backward pass with padding
        3. Combine with unpad: output shape
        4. Combine backward pass with unpad
        5. Roundtrip with padding: dispatch + combine recovers original
        6. Probs permutation with padding (when with_probs=True)
        """
        key = jax.random.PRNGKey(42)

        # Generate routing map
        routing_map = self.generate_routing_map(num_tokens, num_experts, topk, key)
        num_out_tokens = int(jnp.sum(routing_map))

        # Compute worst-case padded size
        worst_case_size = (
            (num_out_tokens + num_experts * (align_size - 1)) // align_size
        ) * align_size

        # Generate input data
        key, inp_key, prob_key, merge_key = jax.random.split(key, 4)
        inp = jax.random.uniform(
            inp_key, (num_tokens, hidden_size), dtype=dtype, minval=-1.0, maxval=1.0
        )

        # Generate probs if needed (minval > 0 to avoid kernel's special prob==0 handling)
        probs = None
        if with_probs:
            probs = jax.random.uniform(
                prob_key, (num_tokens, num_experts), dtype=dtype, minval=0.1, maxval=1.0
            )

        # Generate merging probs (normalized per token)
        merging_probs = jax.random.uniform(
            merge_key, (num_tokens, num_experts), dtype=dtype, minval=0.1, maxval=1.0
        )
        merging_probs = merging_probs * routing_map.astype(dtype)  # Zero out non-routed
        merging_probs = merging_probs / jnp.maximum(
            jnp.sum(merging_probs, axis=1, keepdims=True), 1e-8
        )

        # =====================================================================
        # Test 1: Dispatch with padding - forward pass
        # =====================================================================
        output, permuted_probs, row_id_map, pad_offsets, target_tokens_per_expert = token_dispatch(
            inp, routing_map, num_out_tokens, probs=probs, align_size=align_size
        )

        # Check output shape
        assert output.shape == (worst_case_size, hidden_size)
        if with_probs:
            assert permuted_probs is not None
            assert permuted_probs.shape == (worst_case_size,)
        else:
            assert permuted_probs is None

        # Check alignment: each expert's tokens should be aligned
        for expert_idx in range(num_experts):
            expert_tokens = int(target_tokens_per_expert[expert_idx])
            assert expert_tokens % align_size == 0 or expert_tokens == 0

        # =====================================================================
        # Test 2: Dispatch with padding - backward pass
        # =====================================================================
        if with_probs:

            @jax.jit
            def dispatch_loss(x, p):
                out, perm_probs, _, _, _ = token_dispatch(
                    x, routing_map, num_out_tokens, probs=p, align_size=align_size
                )
                return jnp.sum(out**2) + jnp.sum(perm_probs**2)

            inp_grad, probs_grad = jax.grad(dispatch_loss, argnums=(0, 1))(inp, probs)
            assert inp_grad.shape == inp.shape
            assert probs_grad.shape == probs.shape
            assert not jnp.any(jnp.isnan(inp_grad))
            assert not jnp.any(jnp.isnan(probs_grad))
        else:

            @jax.jit
            def dispatch_loss_no_probs(x):
                out, _, _, _, _ = token_dispatch(
                    x, routing_map, num_out_tokens, align_size=align_size
                )
                return jnp.sum(out**2)

            inp_grad = jax.grad(dispatch_loss_no_probs)(inp)
            assert inp_grad.shape == inp.shape
            assert not jnp.any(jnp.isnan(inp_grad))

        # =====================================================================
        # Test 3: Combine with unpad - forward pass
        # =====================================================================
        combined = token_combine(output, row_id_map, merging_probs, pad_offsets)
        assert combined.shape == (num_tokens, hidden_size)

        # =====================================================================
        # Test 4: Combine with unpad - backward pass
        # =====================================================================

        @jax.jit
        def combine_loss(x):
            return jnp.sum(token_combine(x, row_id_map, merging_probs, pad_offsets) ** 2)

        combine_grad = jax.grad(combine_loss)(output)
        assert combine_grad.shape == output.shape
        assert not jnp.any(jnp.isnan(combine_grad))

        # =====================================================================
        # Test 5: Roundtrip with padding (dispatch + combine = original)
        # =====================================================================
        # Use uniform merging probs for perfect roundtrip
        uniform_merging_probs = routing_map.astype(dtype) / jnp.maximum(
            jnp.sum(routing_map, axis=1, keepdims=True), 1.0
        )

        @jax.jit
        def roundtrip(x):
            dispatched, _, rid_map, p_offsets, _ = token_dispatch(
                x, routing_map, num_out_tokens, align_size=align_size
            )
            return token_combine(dispatched, rid_map, uniform_merging_probs, p_offsets)

        roundtrip_output = roundtrip(inp)
        assert_allclose(roundtrip_output, inp, dtype=dtype)

        # Test roundtrip gradient
        @jax.jit
        def roundtrip_loss(x):
            return jnp.sum(roundtrip(x) ** 2)

        roundtrip_grad = jax.grad(roundtrip_loss)(inp)
        assert roundtrip_grad.shape == inp.shape
        assert not jnp.any(jnp.isnan(roundtrip_grad))
