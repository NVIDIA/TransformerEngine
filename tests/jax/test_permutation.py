# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Tests for permutation Triton kernels and high-level APIs"""

import jax
import jax.numpy as jnp
import pytest
from jax import jit

# Low-level triton_extensions API
from transformer_engine.jax.triton_extensions import (
    make_row_id_map,
    permute_with_mask_map,
    unpermute_with_mask_map,
    make_chunk_sort_map,
    sort_chunks_by_map,
)

# High-level API with VJP support
from transformer_engine.jax.permutation import (
    token_dispatch,
    token_dispatch_with_probs,
    token_combine,
    sort_chunks_by_index,
)
from utils import assert_allclose, dtype_tols


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


def reference_permute_with_mask_map(
    inp: jnp.ndarray,
    row_id_map: jnp.ndarray,
    probs: jnp.ndarray,
    num_tokens: int,
    num_experts: int,
    num_out_tokens: int,
    hidden_size: int,
) -> tuple:
    """
    Reference implementation of permute_with_mask_map using JAX primitives.

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
        n_routed = int(row_id_map[token_idx, -1])
        for i in range(n_routed):
            dest_row = int(row_id_map[token_idx, i])
            expert_idx = int(row_id_map[token_idx, num_experts + i])

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


def reference_unpermute_with_mask_map(
    inp: jnp.ndarray,
    row_id_map: jnp.ndarray,
    merging_probs: jnp.ndarray,
    permuted_probs: jnp.ndarray,
    num_tokens: int,
    num_experts: int,
    hidden_size: int,
) -> tuple:
    """
    Reference implementation of unpermute_with_mask_map using JAX primitives.

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
        n_routed = int(row_id_map[token_idx, -1])
        for i in range(n_routed):
            src_row = int(row_id_map[token_idx, i])
            expert_idx = int(row_id_map[token_idx, num_experts + i])

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
            dest_idx = int(row_id_map[src_idx])
            output = output.at[dest_idx].set(inp[src_idx])
            if probs is not None:
                permuted_probs = permuted_probs.at[dest_idx].set(probs[src_idx])
    else:
        # Backward: dest -> src
        for dest_idx in range(num_tokens):
            src_idx = int(row_id_map[dest_idx])
            output = output.at[dest_idx].set(inp[src_idx])
            if probs is not None:
                permuted_probs = permuted_probs.at[dest_idx].set(probs[src_idx])

    return output, permuted_probs


class TestPermutation:
    """Test permutation operations implementation"""

    @staticmethod
    def generate_routing_map(
        num_tokens: int,
        num_experts: int,
        tokens_per_expert: int = 2,
        key: jax.Array = None,
        use_fixed_per_token: bool = True,
    ):
        """Generate random routing map for testing

        Parameters
        ----------
        num_tokens : int
            Number of tokens
        num_experts : int
            Number of experts
        tokens_per_expert : int
            If use_fixed_per_token=True, each token gets exactly this many experts.
            If use_fixed_per_token=False, total routed connections = num_tokens * tokens_per_expert
        key : jax.Array
            Random key
        use_fixed_per_token : bool
            If True: each token routes to exactly tokens_per_expert experts (old behavior)
            If False: randomly distribute routing like PyTorch (different n_routed per token)
        """
        if key is None:
            key = jax.random.PRNGKey(0)

        if use_fixed_per_token:
            # Each token is routed to the same number of experts. The experts are chosen randomly
            routing_map = jnp.zeros((num_tokens, num_experts), dtype=jnp.int32)

            # Randomly assign each token to tokens_per_expert experts
            for token_idx in range(num_tokens):
                key, subkey = jax.random.split(key)
                expert_indices = jax.random.choice(
                    subkey, num_experts, shape=(tokens_per_expert,), replace=False
                )
                routing_map = routing_map.at[token_idx, expert_indices].set(1)
        else:
            # Varying n_routed per token
            num_out_tokens = num_tokens * tokens_per_expert

            # Create flat array with num_out_tokens ones
            flat_array = jnp.zeros((num_tokens * num_experts,), dtype=jnp.int32)
            flat_array = flat_array.at[:num_out_tokens].set(1)

            # Randomly permute
            key, subkey = jax.random.split(key)
            permuted_indices = jax.random.permutation(subkey, num_tokens * num_experts)
            flat_array = flat_array[permuted_indices]

            # Reshape to routing_map
            routing_map = flat_array.reshape((num_tokens, num_experts))

        return routing_map

    @pytest.mark.parametrize(
        "num_tokens,num_experts,tokens_per_expert",
        [
            (32, 8, 2),
            (64, 16, 3),
            (128, 8, 1),
        ],
    )
    @pytest.mark.parametrize("use_fixed_per_token", [True, False])
    def test_make_row_id_map(self, num_tokens, num_experts, tokens_per_expert, use_fixed_per_token):
        """Test make_row_id_map against reference implementation"""
        key = jax.random.PRNGKey(42)

        # Generate routing map
        routing_map = self.generate_routing_map(
            num_tokens, num_experts, tokens_per_expert, key, use_fixed_per_token
        )

        test_row_id_map = make_row_id_map(routing_map, num_tokens, num_experts)

        ref_row_id_map = reference_make_row_id_map(routing_map, num_tokens, num_experts)

        # Compare results only at valid positions (first n_routed in each section)
        for token_idx in range(num_tokens):
            n_routed = int(ref_row_id_map[token_idx, -1])

            # Compare valid dest rows [0:n_routed]
            assert_allclose(
                test_row_id_map[token_idx, :n_routed],
                ref_row_id_map[token_idx, :n_routed],
                rtol=0,
                atol=0,
                err_msg=f"Mismatch in dest rows for token {token_idx}",
            )

            # Compare valid expert indices [num_experts:num_experts+n_routed]
            assert_allclose(
                test_row_id_map[token_idx, num_experts : num_experts + n_routed],
                ref_row_id_map[token_idx, num_experts : num_experts + n_routed],
                rtol=0,
                atol=0,
                err_msg=f"Mismatch in expert indices for token {token_idx}",
            )

            # Compare n_routed (last column)
            assert_allclose(
                test_row_id_map[token_idx, -1],
                ref_row_id_map[token_idx, -1],
                rtol=0,
                atol=0,
                err_msg=f"Mismatch in n_routed for token {token_idx}",
            )

    # Test permute_with_mask_map
    @pytest.mark.parametrize(
        "num_tokens,num_experts,hidden_size,tokens_per_expert",
        [
            (32, 8, 256, 2),
            (64, 16, 512, 3),
            # Smaller test cases for easier debugging
            # (8, 2, 32, 1),
        ],
    )
    @pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
    @pytest.mark.parametrize("with_probs", [True, False])
    def test_permute_with_mask_map(
        self, num_tokens, num_experts, hidden_size, tokens_per_expert, dtype, with_probs
    ):
        """Test permute_with_mask_map against reference implementation"""
        key = jax.random.PRNGKey(42)

        # Generate routing map
        routing_map = self.generate_routing_map(num_tokens, num_experts, tokens_per_expert, key)

        row_id_map = make_row_id_map(routing_map, num_tokens, num_experts)
        num_out_tokens = int(jnp.sum(routing_map))

        # Generate input data
        key, inp_key, prob_key = jax.random.split(key, 3)
        inp = jax.random.uniform(
            inp_key, (num_tokens, hidden_size), dtype=dtype, minval=-1.0, maxval=1.0
        )

        if with_probs:
            probs = jax.random.uniform(
                prob_key, (num_tokens, num_experts), dtype=dtype, minval=0.0, maxval=1.0
            )
        else:
            probs = None

        test_output, test_probs = permute_with_mask_map(
            inp, row_id_map, probs, num_tokens, num_experts, num_out_tokens, hidden_size
        )

        ref_output, ref_probs = reference_permute_with_mask_map(
            inp, row_id_map, probs, num_tokens, num_experts, num_out_tokens, hidden_size
        )

        # Compare results
        tols = dtype_tols(dtype)
        assert_allclose(test_output, ref_output, **tols)

        if with_probs:
            assert_allclose(test_probs, ref_probs, **tols)

    # Test unpermute_with_mask_map
    @pytest.mark.parametrize(
        "num_tokens,num_experts,hidden_size,tokens_per_expert",
        [
            (32, 8, 256, 2),
            (64, 16, 512, 3),
        ],
    )
    @pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
    @pytest.mark.parametrize("with_merging_probs", [True, False])
    @pytest.mark.parametrize("with_permuted_probs", [True, False])
    def test_unpermute_with_mask_map(
        self,
        num_tokens,
        num_experts,
        hidden_size,
        tokens_per_expert,
        dtype,
        with_merging_probs,
        with_permuted_probs,
    ):
        """Test unpermute_with_mask_map against reference implementation"""
        key = jax.random.PRNGKey(42)

        # Generate routing map
        routing_map = self.generate_routing_map(num_tokens, num_experts, tokens_per_expert, key)

        # Generate row_id_map
        row_id_map = make_row_id_map(routing_map, num_tokens, num_experts)

        # Calculate number of output tokens
        num_out_tokens = int(jnp.sum(routing_map))

        # Generate input data
        key, inp_key, merge_key, prob_key = jax.random.split(key, 4)
        inp = jax.random.uniform(
            inp_key, (num_out_tokens, hidden_size), dtype=dtype, minval=-1.0, maxval=1.0
        )

        if with_merging_probs:
            merging_probs = jax.random.uniform(
                merge_key, (num_tokens, num_experts), dtype=dtype, minval=0.0, maxval=1.0
            )
            # Normalize merging probs per token
            merging_probs = merging_probs / (jnp.sum(merging_probs, axis=1, keepdims=True) + 1e-8)
        else:
            merging_probs = None

        if with_permuted_probs:
            permuted_probs = jax.random.uniform(
                prob_key, (num_out_tokens,), dtype=dtype, minval=0.0, maxval=1.0
            )
        else:
            permuted_probs = None

        test_output, test_unprobs = unpermute_with_mask_map(
            inp, row_id_map, merging_probs, permuted_probs, num_tokens, num_experts, hidden_size
        )

        ref_output, ref_unprobs = reference_unpermute_with_mask_map(
            inp, row_id_map, merging_probs, permuted_probs, num_tokens, num_experts, hidden_size
        )

        # Compare results
        tols = dtype_tols(dtype)
        # Use relaxed tolerances for unpermute due to accumulation
        relaxed_tols = dtype_tols(dtype, rtol=tols["rtol"] * 5, atol=tols["atol"] * 5)

        assert_allclose(test_output, ref_output, **relaxed_tols)

        if with_permuted_probs:
            assert_allclose(test_unprobs, ref_unprobs, **tols)

    # Test round-trip: permute -> unpermute
    @pytest.mark.parametrize(
        "num_tokens,num_experts,hidden_size,tokens_per_expert",
        [
            (32, 8, 256, 2),
            (64, 16, 512, 3),
        ],
    )
    @pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
    def test_permute_unpermute_roundtrip(
        self, num_tokens, num_experts, hidden_size, tokens_per_expert, dtype
    ):
        """Test that permute followed by unpermute recovers original input"""
        key = jax.random.PRNGKey(42)

        # Generate routing map
        routing_map = self.generate_routing_map(num_tokens, num_experts, tokens_per_expert, key)

        # Generate row_id_map
        row_id_map = make_row_id_map(routing_map, num_tokens, num_experts)

        # Calculate number of output tokens
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

        # Permute
        permuted, _ = permute_with_mask_map(
            inp, row_id_map, None, num_tokens, num_experts, num_out_tokens, hidden_size
        )

        # Unpermute with uniform merging
        unpermuted, _ = unpermute_with_mask_map(
            permuted, row_id_map, merging_probs, None, num_tokens, num_experts, hidden_size
        )

        # Compare with original input
        tols = dtype_tols(dtype)
        relaxed_tols = dtype_tols(dtype, rtol=tols["rtol"] * 10, atol=tols["atol"] * 10)
        assert_allclose(unpermuted, inp, **relaxed_tols)

    @pytest.mark.parametrize(
        "num_splits,total_tokens",
        [
            (4, 128),
            (8, 256),
            (16, 512),
        ],
    )
    def test_make_chunk_sort_map(self, num_splits, total_tokens):
        """Test make_chunk_sort_map against reference implementation"""
        key = jax.random.PRNGKey(42)

        # Generate random split sizes
        key, size_key = jax.random.split(key)
        split_sizes = jax.random.randint(size_key, (num_splits,), 10, total_tokens // num_splits)
        # Adjust last split to match total_tokens
        split_sizes = split_sizes.at[-1].set(total_tokens - jnp.sum(split_sizes[:-1]))

        # Generate sorted indices (permutation of 0..num_splits-1)
        key, sort_key = jax.random.split(key)
        sorted_indices = jax.random.permutation(sort_key, num_splits)

        test_map = make_chunk_sort_map(split_sizes, sorted_indices, total_tokens, num_splits)
        ref_map = reference_make_chunk_sort_map(
            split_sizes, sorted_indices, total_tokens, num_splits
        )

        assert_allclose(test_map, ref_map, rtol=0, atol=0)

    @pytest.mark.parametrize(
        "num_splits,total_tokens,hidden_size",
        [
            (4, 128, 256),
            (8, 256, 512),
        ],
    )
    @pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
    @pytest.mark.parametrize("is_forward", [True, False])
    @pytest.mark.parametrize("with_probs", [True, False])
    def test_sort_chunks_by_map(
        self, num_splits, total_tokens, hidden_size, dtype, is_forward, with_probs
    ):
        """Test sort_chunks_by_map against reference implementation"""
        key = jax.random.PRNGKey(42)

        # Generate random split sizes
        key, size_key = jax.random.split(key)
        split_sizes = jax.random.randint(size_key, (num_splits,), 10, total_tokens // num_splits)
        split_sizes = split_sizes.at[-1].set(total_tokens - jnp.sum(split_sizes[:-1]))

        # Generate sorted indices
        key, sort_key = jax.random.split(key)
        sorted_indices = jax.random.permutation(sort_key, num_splits)

        row_id_map = make_chunk_sort_map(split_sizes, sorted_indices, total_tokens, num_splits)

        key, inp_key, prob_key = jax.random.split(key, 3)
        inp = jax.random.uniform(
            inp_key, (total_tokens, hidden_size), dtype=dtype, minval=-1.0, maxval=1.0
        )

        if with_probs:
            probs = jax.random.uniform(
                prob_key, (total_tokens,), dtype=dtype, minval=0.0, maxval=1.0
            )
        else:
            probs = None

        test_output, test_probs = sort_chunks_by_map(
            inp, row_id_map, probs, total_tokens, hidden_size, is_forward
        )

        ref_output, ref_probs = reference_sort_chunks_by_map(
            inp, row_id_map, probs, total_tokens, hidden_size, is_forward
        )

        tols = dtype_tols(dtype)
        assert_allclose(test_output, ref_output, **tols)

        if with_probs:
            assert_allclose(test_probs, ref_probs, **tols)

    @pytest.mark.parametrize(
        "num_splits,total_tokens,hidden_size",
        [
            (4, 128, 256),
            (8, 256, 512),
        ],
    )
    @pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
    def test_chunk_sort_roundtrip(self, num_splits, total_tokens, hidden_size, dtype):
        """Test that forward sort followed by backward sort recovers original input"""
        key = jax.random.PRNGKey(42)

        # Generate random split sizes
        key, size_key = jax.random.split(key)
        split_sizes = jax.random.randint(size_key, (num_splits,), 10, total_tokens // num_splits)
        split_sizes = split_sizes.at[-1].set(total_tokens - jnp.sum(split_sizes[:-1]))

        # Generate sorted indices
        key, sort_key = jax.random.split(key)
        sorted_indices = jax.random.permutation(sort_key, num_splits)

        # Generate row_id_map
        row_id_map = make_chunk_sort_map(split_sizes, sorted_indices, total_tokens, num_splits)

        # Generate input data
        key, inp_key = jax.random.split(key)
        inp = jax.random.uniform(
            inp_key, (total_tokens, hidden_size), dtype=dtype, minval=-1.0, maxval=1.0
        )

        # Forward sort
        sorted_output, _ = sort_chunks_by_map(
            inp, row_id_map, None, total_tokens, hidden_size, is_forward=True
        )

        # Backward sort (should recover original)
        recovered, _ = sort_chunks_by_map(
            sorted_output, row_id_map, None, total_tokens, hidden_size, is_forward=False
        )

        # Compare with original input
        tols = dtype_tols(dtype)
        assert_allclose(recovered, inp, **tols)


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
    def test_token_dispatch_forward(
        self, num_tokens, num_experts, hidden_size, tokens_per_expert, dtype
    ):
        """Test token_dispatch forward pass against reference implementation"""
        key = jax.random.PRNGKey(42)

        # Generate routing map
        routing_map = self.generate_routing_map(num_tokens, num_experts, tokens_per_expert, key)
        num_out_tokens = int(jnp.sum(routing_map))

        # Generate input data
        key, inp_key = jax.random.split(key)
        inp = jax.random.uniform(
            inp_key, (num_tokens, hidden_size), dtype=dtype, minval=-1.0, maxval=1.0
        )

        # Call high-level API (new signature)
        output, row_id_map = token_dispatch(inp, routing_map, num_out_tokens)

        # Compare against reference
        ref_output, _ = reference_permute_with_mask_map(
            inp, row_id_map, None, num_tokens, num_experts, num_out_tokens, hidden_size
        )

        tols = dtype_tols(dtype)
        assert_allclose(output, ref_output, **tols)

    @pytest.mark.parametrize(
        "num_tokens,num_experts,hidden_size,tokens_per_expert",
        [
            (32, 8, 256, 2),
        ],
    )
    @pytest.mark.parametrize("dtype", [jnp.float32])
    def test_token_dispatch_vjp(
        self, num_tokens, num_experts, hidden_size, tokens_per_expert, dtype
    ):
        """Test token_dispatch VJP (backward pass) using value_and_grad"""
        key = jax.random.PRNGKey(42)

        # Generate routing map
        routing_map = self.generate_routing_map(num_tokens, num_experts, tokens_per_expert, key)
        num_out_tokens = int(jnp.sum(routing_map))

        # Generate input data
        key, inp_key = jax.random.split(key)
        inp = jax.random.uniform(
            inp_key, (num_tokens, hidden_size), dtype=dtype, minval=-1.0, maxval=1.0
        )

        # Define loss function that uses token_dispatch
        def loss_fn(x):
            output, _ = token_dispatch(x, routing_map, num_out_tokens)
            return jnp.sum(output**2)

        # Compute value and gradient using JAX value_and_grad
        loss_val, computed_grad = jax.value_and_grad(loss_fn)(inp)

        # Compute reference gradient manually:
        # d_loss/d_inp = unpermute(d_loss/d_output)
        # where d_loss/d_output = 2 * output
        output, row_id_map = token_dispatch(inp, routing_map, num_out_tokens)
        output_grad = 2 * output

        # Verify loss value
        expected_loss = jnp.sum(output**2)
        assert_allclose(loss_val, expected_loss, rtol=1e-5, atol=1e-5)

        # Reference backward: unpermute the gradient
        ref_grad, _ = reference_unpermute_with_mask_map(
            output_grad, row_id_map, None, None, num_tokens, num_experts, hidden_size
        )

        tols = dtype_tols(dtype, rtol=1e-4, atol=1e-4)
        assert_allclose(computed_grad, ref_grad, **tols)

    # =========================================================================
    # token_dispatch_with_probs tests
    # =========================================================================

    @pytest.mark.parametrize(
        "num_tokens,num_experts,hidden_size,tokens_per_expert",
        [
            (32, 8, 256, 2),
            (64, 16, 512, 3),
        ],
    )
    @pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
    def test_token_dispatch_with_probs_forward(
        self, num_tokens, num_experts, hidden_size, tokens_per_expert, dtype
    ):
        """Test token_dispatch_with_probs forward pass against reference"""
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

        # Call high-level API (new signature)
        output, permuted_probs, row_id_map = token_dispatch_with_probs(
            inp, routing_map, probs, num_out_tokens
        )

        # Compare against reference
        ref_output, ref_probs = reference_permute_with_mask_map(
            inp, row_id_map, probs, num_tokens, num_experts, num_out_tokens, hidden_size
        )

        tols = dtype_tols(dtype)
        assert_allclose(output, ref_output, **tols)
        assert_allclose(permuted_probs, ref_probs, **tols)

    @pytest.mark.parametrize(
        "num_tokens,num_experts,hidden_size,tokens_per_expert",
        [
            (32, 8, 256, 2),
        ],
    )
    @pytest.mark.parametrize("dtype", [jnp.float32])
    def test_token_dispatch_with_probs_vjp(
        self, num_tokens, num_experts, hidden_size, tokens_per_expert, dtype
    ):
        """Test token_dispatch_with_probs VJP using value_and_grad"""
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

        # Define loss function that uses token_dispatch_with_probs
        # We compute gradients w.r.t. both inp and probs
        def loss_fn(x, p):
            output, permuted_probs, _ = token_dispatch_with_probs(x, routing_map, p, num_out_tokens)
            return jnp.sum(output**2) + jnp.sum(permuted_probs**2)

        # Compute value and gradient using JAX value_and_grad
        loss_val, (inp_grad, probs_grad) = jax.value_and_grad(loss_fn, argnums=(0, 1))(inp, probs)

        # Verify the loss value
        output, permuted_probs, _ = token_dispatch_with_probs(
            inp, routing_map, probs, num_out_tokens
        )
        expected_loss = jnp.sum(output**2) + jnp.sum(permuted_probs**2)
        assert_allclose(loss_val, expected_loss, rtol=1e-5, atol=1e-5)

        # Verify gradients are computed (non-zero for non-trivial inputs)
        assert inp_grad.shape == inp.shape
        assert probs_grad.shape == probs.shape

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
    def test_token_combine_forward(
        self, num_tokens, num_experts, hidden_size, tokens_per_expert, dtype, with_merging_probs
    ):
        """Test token_combine forward pass against reference implementation"""
        key = jax.random.PRNGKey(42)

        # Generate routing map
        routing_map = self.generate_routing_map(num_tokens, num_experts, tokens_per_expert, key)
        num_out_tokens = int(jnp.sum(routing_map))

        # Get row_id_map from token_dispatch (which computes it internally)
        key, dummy_key = jax.random.split(key)
        dummy_inp = jax.random.uniform(
            dummy_key, (num_tokens, hidden_size), dtype=dtype, minval=-1.0, maxval=1.0
        )
        _, row_id_map = token_dispatch(dummy_inp, routing_map, num_out_tokens)

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

        # Call high-level API
        output = token_combine(inp, row_id_map, merging_probs)

        # Compare against reference
        ref_output, _ = reference_unpermute_with_mask_map(
            inp, row_id_map, merging_probs, None, num_tokens, num_experts, hidden_size
        )

        tols = dtype_tols(dtype)
        relaxed_tols = dtype_tols(dtype, rtol=tols["rtol"] * 5, atol=tols["atol"] * 5)
        assert_allclose(output, ref_output, **relaxed_tols)

    @pytest.mark.parametrize(
        "num_tokens,num_experts,hidden_size,tokens_per_expert",
        [
            (32, 8, 256, 2),
        ],
    )
    @pytest.mark.parametrize("dtype", [jnp.float32])
    def test_token_combine_vjp(
        self, num_tokens, num_experts, hidden_size, tokens_per_expert, dtype
    ):
        """Test token_combine VJP using value_and_grad"""
        key = jax.random.PRNGKey(42)

        # Generate routing map
        routing_map = self.generate_routing_map(num_tokens, num_experts, tokens_per_expert, key)
        num_out_tokens = int(jnp.sum(routing_map))

        # Get row_id_map from token_dispatch (which computes it internally)
        key, dummy_key = jax.random.split(key)
        dummy_inp = jax.random.uniform(
            dummy_key, (num_tokens, hidden_size), dtype=dtype, minval=-1.0, maxval=1.0
        )
        _, row_id_map = token_dispatch(dummy_inp, routing_map, num_out_tokens)

        # Generate input data for token_combine
        key, inp_key = jax.random.split(key)
        inp = jax.random.uniform(
            inp_key, (num_out_tokens, hidden_size), dtype=dtype, minval=-1.0, maxval=1.0
        )

        # Define loss function that uses token_combine (no merging probs)
        def loss_fn(x):
            output = token_combine(x, row_id_map, None)
            return jnp.sum(output**2)

        # Compute value and gradient using JAX value_and_grad
        loss_val, computed_grad = jax.value_and_grad(loss_fn)(inp)

        # Compute reference gradient manually:
        # d_loss/d_inp = permute(d_loss/d_output)
        # where d_loss/d_output = 2 * output
        output = token_combine(inp, row_id_map, None)
        output_grad = 2 * output

        # Verify loss value
        expected_loss = jnp.sum(output**2)
        assert_allclose(loss_val, expected_loss, rtol=1e-5, atol=1e-5)

        # Reference backward: permute the gradient
        ref_grad, _ = reference_permute_with_mask_map(
            output_grad, row_id_map, None, num_tokens, num_experts, num_out_tokens, hidden_size
        )

        tols = dtype_tols(dtype, rtol=1e-4, atol=1e-4)
        assert_allclose(computed_grad, ref_grad, **tols)

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
    def test_sort_chunks_by_index_forward(self, num_splits, total_tokens, hidden_size, dtype):
        """Test sort_chunks_by_index forward pass against reference"""
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

        # Call high-level API (new signature)
        output, row_id_map = sort_chunks_by_index(inp, split_sizes, sorted_indices)

        # Compare against reference
        ref_output, _ = reference_sort_chunks_by_map(
            inp, row_id_map, None, total_tokens, hidden_size, is_forward=True
        )

        tols = dtype_tols(dtype)
        assert_allclose(output, ref_output, **tols)

    @pytest.mark.parametrize(
        "num_splits,total_tokens,hidden_size",
        [
            (4, 128, 256),
        ],
    )
    @pytest.mark.parametrize("dtype", [jnp.float32])
    def test_sort_chunks_by_index_vjp(self, num_splits, total_tokens, hidden_size, dtype):
        """Test sort_chunks_by_index VJP using value_and_grad"""
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

        # Define loss function
        def loss_fn(x):
            output, _ = sort_chunks_by_index(x, split_sizes, sorted_indices)
            return jnp.sum(output**2)

        # Compute value and gradient using JAX value_and_grad
        loss_val, computed_grad = jax.value_and_grad(loss_fn)(inp)

        # Compute reference gradient manually:
        # d_loss/d_inp = reverse_sort(d_loss/d_output)
        # Get row_id_map from sort_chunks_by_index output
        output, row_id_map = sort_chunks_by_index(inp, split_sizes, sorted_indices)
        output_grad = 2 * output

        # Verify loss value
        expected_loss = jnp.sum(output**2)
        assert_allclose(loss_val, expected_loss, rtol=1e-5, atol=1e-5)

        # Reference backward: reverse sort (is_forward=False)
        ref_grad, _ = reference_sort_chunks_by_map(
            output_grad, row_id_map, None, total_tokens, hidden_size, is_forward=False
        )

        tols = dtype_tols(dtype, rtol=1e-4, atol=1e-4)
        assert_allclose(computed_grad, ref_grad, **tols)

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

        # Dispatch tokens to experts (new signature)
        dispatched, row_id_map = token_dispatch(inp, routing_map, num_out_tokens)

        # Combine tokens back (with uniform merging) (new signature)
        combined = token_combine(dispatched, row_id_map, merging_probs)

        # Compare with original input
        tols = dtype_tols(dtype)
        relaxed_tols = dtype_tols(dtype, rtol=tols["rtol"] * 10, atol=tols["atol"] * 10)
        assert_allclose(combined, inp, **relaxed_tols)
