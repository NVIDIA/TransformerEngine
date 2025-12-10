# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Tests for permutation Triton kernels and high-level APIs"""

import jax
import jax.numpy as jnp
import pytest

# High-level API with VJP support
from transformer_engine.jax.permutation import (
    token_dispatch,
    token_combine,
    sort_chunks_by_index,
)
from utils import assert_allclose


def reference_make_row_id_map(
    routing_map: jnp.ndarray,
    num_tokens: int,
    num_experts: int,
) -> jnp.ndarray:
    """
    Reference implementation of make_row_id_map using JAX primitives.

    Parameters
    ----------
    routing_map : jnp.ndarray
        Input tensor of shape [num_tokens, num_experts]. Mask indicating which experts
        are routed to which tokens (1 = routed, 0 = not routed).
    num_tokens : int
        Number of tokens in the input tensor.
    num_experts : int
        Number of experts in the input tensor.

    Returns
    -------
    row_id_map : jnp.ndarray
        The row_id_map for the permutation of shape [num_tokens, num_experts * 2 + 1].
    """
    row_id_map = jnp.full((num_tokens, num_experts * 2 + 1), -1, dtype=jnp.int32)

    # For each expert, compute cumulative sum to get destination indices
    cumsum_per_expert = jnp.cumsum(routing_map, axis=0)

    # Compute total tokens per expert
    tokens_per_expert = jnp.sum(routing_map, axis=0)
    expert_offsets = jnp.concatenate([jnp.array([0]), jnp.cumsum(tokens_per_expert)[:-1]])

    # Build the row_id_map
    for token_idx in range(num_tokens):
        routed_experts = jnp.where(routing_map[token_idx] == 1)[0]
        n_routed = len(routed_experts)

        # Store number of routed experts in the last position
        row_id_map = row_id_map.at[token_idx, -1].set(n_routed)

        # For each routed expert, compute destination row and store it
        dest_rows = []
        expert_indices = []
        for expert_idx in routed_experts:
            # Destination row = expert offset + (cumsum - 1)
            dest_row = expert_offsets[expert_idx] + cumsum_per_expert[token_idx, expert_idx] - 1
            dest_rows.append(dest_row)
            expert_indices.append(expert_idx)

        # Sort by destination row
        if n_routed > 0:
            sort_indices = jnp.argsort(-jnp.array(dest_rows))  # Negative for descending sort
            sorted_dest_rows = jnp.array(dest_rows)[sort_indices]
            sorted_expert_indices = jnp.array(expert_indices)[sort_indices]

            # Store sorted destination rows and expert indices
            for i in range(n_routed):
                row_id_map = row_id_map.at[token_idx, i].set(sorted_dest_rows[i])
                row_id_map = row_id_map.at[token_idx, num_experts + i].set(sorted_expert_indices[i])

    return row_id_map


def _reference_permute_impl(
    inp: jnp.ndarray,
    row_id_map: jnp.ndarray,
    probs: jnp.ndarray,
    num_tokens: int,
    num_experts: int,
    num_out_tokens: int,
    hidden_size: int,
) -> tuple:
    """
    Internal helper for reference permutation implementation.

    Parameters
    ----------
    inp : jnp.ndarray
        Input tensor of shape [num_tokens, hidden_size].
    row_id_map : jnp.ndarray
        The token to expert mapping tensor of shape [num_tokens, num_experts * 2 + 1].
    probs : jnp.ndarray
        The probabilities of the input tensor.
    num_tokens : int
        Number of tokens in the input tensor.
    num_experts : int
        Number of experts.
    num_out_tokens : int
        Number of tokens in the permuted tensor.
    hidden_size : int
        Hidden size of the input tensor.

    Returns
    -------
    output : jnp.ndarray
        Permuted output tensor of shape [num_out_tokens, hidden_size].
    permuted_probs : jnp.ndarray
        Permuted probabilities if probs was provided, None otherwise.
    """
    output = jnp.zeros((num_out_tokens, hidden_size), dtype=inp.dtype)
    permuted_probs = None if probs is None else jnp.zeros((num_out_tokens,), dtype=probs.dtype)

    for token_idx in range(num_tokens):
        n_routed = int(row_id_map[token_idx, -1])  # int() needed for Python range()
        for i in range(n_routed):
            # Don't use int() here - JAX can index with traced values,
            # and int() breaks autodiff gradient tracking
            dest_row = row_id_map[token_idx, i]
            expert_idx = row_id_map[token_idx, num_experts + i]

            # Get probability for this expert
            if probs is not None:
                if probs.ndim == 1:
                    prob = probs[token_idx]
                else:
                    prob = probs[token_idx, expert_idx]

                # Match kernel behavior: if prob == 0.0, zero out the output (padding indicator)
                if prob == 0.0:
                    output = output.at[dest_row].set(0.0)
                else:
                    output = output.at[dest_row].set(inp[token_idx])

                permuted_probs = permuted_probs.at[dest_row].set(prob)
            else:
                output = output.at[dest_row].set(inp[token_idx])

    return output, permuted_probs


def _reference_unpermute_impl(
    inp: jnp.ndarray,
    row_id_map: jnp.ndarray,
    merging_probs: jnp.ndarray,
    permuted_probs: jnp.ndarray,
    num_tokens: int,
    num_experts: int,
    hidden_size: int,
) -> tuple:
    """
    Internal helper for reference unpermutation implementation.

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
    num_tokens : int
        Number of tokens.
    num_experts : int
        Number of experts.
    hidden_size : int
        Hidden size.

    Returns
    -------
    output : jnp.ndarray
        Unpermuted output tensor of shape [num_tokens, hidden_size].
    unpermuted_probs : jnp.ndarray
        Unpermuted probabilities if permuted_probs was provided, None otherwise.
    """
    output = jnp.zeros((num_tokens, hidden_size), dtype=inp.dtype)
    unpermuted_probs = (
        None
        if permuted_probs is None
        else jnp.zeros((num_tokens, num_experts), dtype=permuted_probs.dtype)
    )

    for token_idx in range(num_tokens):
        n_routed = int(row_id_map[token_idx, -1])  # int() needed for Python range()
        for i in range(n_routed):
            # Don't use int() here - JAX can index with traced values,
            # and int() breaks autodiff gradient tracking
            src_row = row_id_map[token_idx, i]
            expert_idx = row_id_map[token_idx, num_experts + i]

            if merging_probs is not None:
                weight = merging_probs[token_idx, expert_idx]
                output = output.at[token_idx].add(inp[src_row] * weight)
            else:
                output = output.at[token_idx].add(inp[src_row])

            if permuted_probs is not None:
                unpermuted_probs = unpermuted_probs.at[token_idx, expert_idx].set(
                    permuted_probs[src_row]
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
    num_tokens, num_experts = routing_map.shape
    hidden_size = inp.shape[1]

    row_id_map = reference_make_row_id_map(routing_map, num_tokens, num_experts)
    output, permuted_probs = _reference_permute_impl(
        inp, row_id_map, probs, num_tokens, num_experts, num_out_tokens, hidden_size
    )

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
    num_tokens = row_id_map.shape[0]
    num_experts = (row_id_map.shape[1] - 1) // 2
    hidden_size = inp.shape[1]

    output, _ = _reference_unpermute_impl(
        inp, row_id_map, merging_probs, None, num_tokens, num_experts, hidden_size
    )

    return output


def reference_make_chunk_sort_map(
    split_sizes: jnp.ndarray,
    sorted_indices: jnp.ndarray,
    num_tokens: int,
    num_splits: int,
) -> jnp.ndarray:
    """
    Reference implementation of make_chunk_sort_map using JAX primitives.

    Parameters
    ----------
    split_sizes : jnp.ndarray
        The sizes of the chunks of shape [num_splits,].
    sorted_indices : jnp.ndarray
        The indices of the sorted chunks of shape [num_splits,].
    num_tokens : int
        Number of tokens.
    num_splits : int
        Number of splits.

    Returns
    -------
    row_id_map : jnp.ndarray
        Row ID map for chunk sorting of shape [num_tokens,].
    """
    row_id_map = jnp.zeros((num_tokens,), dtype=jnp.int32)

    # Compute cumulative positions
    cumsum_sizes = jnp.concatenate([jnp.array([0]), jnp.cumsum(split_sizes)])

    # For each chunk, compute the destination indices
    dest_offset = 0
    for sorted_idx in sorted_indices:
        chunk_start = cumsum_sizes[sorted_idx]
        chunk_end = cumsum_sizes[sorted_idx + 1]
        chunk_size = chunk_end - chunk_start

        # Map source positions to destination positions
        for i in range(chunk_size):
            row_id_map = row_id_map.at[chunk_start + i].set(dest_offset + i)

        dest_offset += chunk_size

    return row_id_map


def reference_sort_chunks_by_map(
    inp: jnp.ndarray,
    row_id_map: jnp.ndarray,
    probs: jnp.ndarray,
    num_tokens: int,
    hidden_size: int,
    is_forward: bool,
) -> tuple:
    """
    Reference implementation of sort_chunks_by_map using JAX primitives.

    Parameters
    ----------
    inp : jnp.ndarray
        Input tensor of shape [num_tokens, hidden_size].
    row_id_map : jnp.ndarray
        The token to destination mapping of shape [num_tokens,].
    probs : jnp.ndarray
        The probabilities.
    num_tokens : int
        Number of tokens.
    hidden_size : int
        Hidden size.
    is_forward : bool
        Whether this is forward or backward.

    Returns
    -------
    output : jnp.ndarray
        Sorted output tensor of shape [num_tokens, hidden_size].
    permuted_probs : jnp.ndarray
        Sorted probabilities if probs was provided, None otherwise.
    """
    output = jnp.zeros((num_tokens, hidden_size), dtype=inp.dtype)
    permuted_probs = None if probs is None else jnp.zeros((num_tokens,), dtype=probs.dtype)

    if is_forward:
        # Forward: src -> dest
        for src_idx in range(num_tokens):
            # Don't use int() - JAX can index with traced values
            dest_idx = row_id_map[src_idx]
            output = output.at[dest_idx].set(inp[src_idx])
            if probs is not None:
                permuted_probs = permuted_probs.at[dest_idx].set(probs[src_idx])
    else:
        # Backward: dest -> src
        for dest_idx in range(num_tokens):
            # Don't use int() - JAX can index with traced values
            src_idx = row_id_map[dest_idx]
            output = output.at[dest_idx].set(inp[src_idx])
            if probs is not None:
                permuted_probs = permuted_probs.at[dest_idx].set(probs[src_idx])

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

    # =========================================================================
    # token_dispatch tests
    # =========================================================================

    @pytest.mark.parametrize(
        "num_tokens,num_experts,hidden_size,tokens_per_expert",
        [
            (32, 8, 256, 2),
            (64, 16, 512, 3),
        ],
    )
    @pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
    def test_token_dispatch(self, num_tokens, num_experts, hidden_size, tokens_per_expert, dtype):
        """Test token_dispatch forward and backward pass against reference"""
        key = jax.random.PRNGKey(42)

        # Generate routing map
        routing_map = self.generate_routing_map(num_tokens, num_experts, tokens_per_expert, key)
        num_out_tokens = int(jnp.sum(routing_map))

        # Generate input data
        key, inp_key = jax.random.split(key)
        inp = jax.random.uniform(
            inp_key, (num_tokens, hidden_size), dtype=dtype, minval=-1.0, maxval=1.0
        )

        # Define loss functions
        def loss_fn(x):
            output, _, _ = token_dispatch(x, routing_map, num_out_tokens)
            return jnp.sum(output**2)

        def ref_loss_fn(x):
            output, _, _ = reference_token_dispatch(x, routing_map, num_out_tokens)
            return jnp.sum(output**2)

        loss_val, computed_grad = jax.value_and_grad(loss_fn)(inp)
        ref_loss_val, ref_grad = jax.value_and_grad(ref_loss_fn)(inp)

        # Compare forward outputs
        output, _, _ = token_dispatch(inp, routing_map, num_out_tokens)
        ref_output, _, _ = reference_token_dispatch(inp, routing_map, num_out_tokens)
        assert_allclose(output, ref_output)

        # Compare loss and gradient
        assert_allclose(loss_val, ref_loss_val)
        assert_allclose(computed_grad, ref_grad)

    # =========================================================================
    # token_dispatch with probs tests
    # =========================================================================

    @pytest.mark.parametrize(
        "num_tokens,num_experts,hidden_size,tokens_per_expert",
        [
            (32, 8, 256, 2),
            (64, 16, 512, 3),
        ],
    )
    @pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
    def test_token_dispatch_with_probs(
        self, num_tokens, num_experts, hidden_size, tokens_per_expert, dtype
    ):
        """Test token_dispatch with probs forward and backward pass against reference"""
        key = jax.random.PRNGKey(42)

        # Generate routing map
        routing_map = self.generate_routing_map(num_tokens, num_experts, tokens_per_expert, key)
        num_out_tokens = int(jnp.sum(routing_map))

        # Generate input data and probs
        key, inp_key, prob_key = jax.random.split(key, 3)
        inp = jax.random.uniform(
            inp_key, (num_tokens, hidden_size), dtype=dtype, minval=-1.0, maxval=1.0
        )
        probs = jax.random.uniform(
            prob_key, (num_tokens, num_experts), dtype=dtype, minval=0.0, maxval=1.0
        )

        # Define loss function that uses token_dispatch with probs
        # We compute gradients w.r.t. both inp and probs
        def loss_fn(x, p):
            output, permuted_probs, _ = token_dispatch(x, routing_map, num_out_tokens, probs=p)
            return jnp.sum(output**2) + jnp.sum(permuted_probs**2)

        def ref_loss_fn(x, p):
            output, permuted_probs, _ = reference_token_dispatch(
                x, routing_map, num_out_tokens, probs=p
            )
            return jnp.sum(output**2) + jnp.sum(permuted_probs**2)

        loss_val, (inp_grad, probs_grad) = jax.value_and_grad(loss_fn, argnums=(0, 1))(inp, probs)
        ref_loss_val, (ref_inp_grad, ref_probs_grad) = jax.value_and_grad(
            ref_loss_fn, argnums=(0, 1)
        )(inp, probs)

        output, permuted_probs, _ = token_dispatch(inp, routing_map, num_out_tokens, probs=probs)

        ref_output, ref_permuted_probs, _ = reference_token_dispatch(
            inp, routing_map, num_out_tokens, probs=probs
        )

        # Compare forward outputs
        assert_allclose(output, ref_output)
        assert_allclose(permuted_probs, ref_permuted_probs)

        # Compare loss and gradients
        assert_allclose(loss_val, ref_loss_val)
        assert_allclose(inp_grad, ref_inp_grad)
        assert_allclose(probs_grad, ref_probs_grad)

    # =========================================================================
    # token_combine tests
    # =========================================================================

    @pytest.mark.parametrize(
        "num_tokens,num_experts,hidden_size,tokens_per_expert",
        [
            (32, 8, 256, 2),
            (64, 16, 512, 3),
        ],
    )
    @pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
    @pytest.mark.parametrize("with_merging_probs", [True, False])
    def test_token_combine(
        self, num_tokens, num_experts, hidden_size, tokens_per_expert, dtype, with_merging_probs
    ):
        """Test token_combine forward and backward pass against reference"""
        key = jax.random.PRNGKey(42)

        # Generate routing map
        routing_map = self.generate_routing_map(num_tokens, num_experts, tokens_per_expert, key)
        num_out_tokens = int(jnp.sum(routing_map))

        # Get row_id_map from reference_token_dispatch
        key, dummy_key = jax.random.split(key)
        dummy_inp = jax.random.uniform(
            dummy_key, (num_tokens, hidden_size), dtype=dtype, minval=-1.0, maxval=1.0
        )
        _, _, row_id_map = reference_token_dispatch(dummy_inp, routing_map, num_out_tokens)

        # Generate input data (from expert outputs)
        key, inp_key, merge_key = jax.random.split(key, 3)
        inp = jax.random.uniform(
            inp_key, (num_out_tokens, hidden_size), dtype=dtype, minval=-1.0, maxval=1.0
        )

        if with_merging_probs:
            merging_probs = jax.random.uniform(
                merge_key, (num_tokens, num_experts), dtype=dtype, minval=0.0, maxval=1.0
            )
            # Normalize per token
            merging_probs = merging_probs / (jnp.sum(merging_probs, axis=1, keepdims=True) + 1e-8)
        else:
            merging_probs = None

        # Define loss functions
        def loss_fn(x):
            output = token_combine(x, row_id_map, merging_probs)
            return jnp.sum(output**2)

        def ref_loss_fn(x):
            output = reference_token_combine(x, row_id_map, merging_probs)
            return jnp.sum(output**2)

        loss_val, computed_grad = jax.value_and_grad(loss_fn)(inp)
        ref_loss_val, ref_grad = jax.value_and_grad(ref_loss_fn)(inp)

        # Compare forward outputs
        output = token_combine(inp, row_id_map, merging_probs)
        ref_output = reference_token_combine(inp, row_id_map, merging_probs)
        assert_allclose(output, ref_output)

        # Compare loss and gradient
        assert_allclose(loss_val, ref_loss_val)
        assert_allclose(computed_grad, ref_grad)

    # =========================================================================
    # sort_chunks_by_index tests
    # =========================================================================

    @pytest.mark.parametrize(
        "num_splits,total_tokens,hidden_size",
        [
            (4, 128, 256),
            (8, 256, 512),
        ],
    )
    @pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
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

        row_id_map = reference_make_chunk_sort_map(
            split_sizes, sorted_indices, total_tokens, num_splits
        )

        # Define loss functions
        def loss_fn(x):
            output, _ = sort_chunks_by_index(x, split_sizes, sorted_indices)
            return jnp.sum(output**2)

        def ref_loss_fn(x):
            output, _ = reference_sort_chunks_by_map(
                x, row_id_map, None, total_tokens, hidden_size, is_forward=True
            )
            return jnp.sum(output**2)

        loss_val, computed_grad = jax.value_and_grad(loss_fn)(inp)
        ref_loss_val, ref_grad = jax.value_and_grad(ref_loss_fn)(inp)

        # Compare forward outputs
        output, _ = sort_chunks_by_index(inp, split_sizes, sorted_indices)
        ref_output, _ = reference_sort_chunks_by_map(
            inp, row_id_map, None, total_tokens, hidden_size, is_forward=True
        )
        assert_allclose(output, ref_output)

        # Compare loss and gradient
        assert_allclose(loss_val, ref_loss_val)
        assert_allclose(computed_grad, ref_grad)

    # =========================================================================
    # Round-trip tests (token_dispatch -> expert processing -> token_combine)
    # =========================================================================

    @pytest.mark.parametrize(
        "num_tokens,num_experts,hidden_size,tokens_per_expert",
        [
            (32, 8, 256, 2),
            (64, 16, 512, 3),
        ],
    )
    @pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
    def test_dispatch_combine_roundtrip(
        self, num_tokens, num_experts, hidden_size, tokens_per_expert, dtype
    ):
        """Test that token_dispatch followed by token_combine recovers original input"""
        key = jax.random.PRNGKey(42)

        # Generate routing map
        routing_map = self.generate_routing_map(num_tokens, num_experts, tokens_per_expert, key)
        num_out_tokens = int(jnp.sum(routing_map))

        # Generate input data
        key, inp_key = jax.random.split(key)
        inp = jax.random.uniform(
            inp_key, (num_tokens, hidden_size), dtype=dtype, minval=-1.0, maxval=1.0
        )

        # Create uniform merging probs (equal weight for all routed experts)
        merging_probs = routing_map.astype(dtype) / jnp.maximum(
            jnp.sum(routing_map, axis=1, keepdims=True), 1.0
        )

        # Dispatch tokens to experts (returns output, permuted_probs, row_id_map)
        dispatched, _, row_id_map = token_dispatch(inp, routing_map, num_out_tokens)

        # Combine tokens back (with uniform merging) (new signature)
        combined = token_combine(dispatched, row_id_map, merging_probs)

        # Compare with original input
        assert_allclose(combined, inp)
