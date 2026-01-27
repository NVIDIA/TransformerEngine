# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Tests for distributed/sharded execution of MoE permutation primitives.

Testing Strategy:
=================
MoE permutation is data-dependent - the destination index for each token depends
on how many tokens before it are routed to the same expert. This means:

1. We CANNOT compare sharded output against global reference directly
2. Instead, we verify that each GPU's LOCAL output is correct according to its
   LOCAL routing (which produces LOCAL row_id_map with LOCAL indices)

For data-parallel MoE without expert parallelism:
- Each GPU has ALL experts replicated
- Each GPU processes a subset of tokens (sharded on token/batch dimension)
- Each GPU computes its own local row_id_map from its local routing_map slice
- Each GPU's output is local and doesn't need to match global output

These tests verify:
1. Local token_dispatch: sharded input -> local row_id_map -> local permute (forward + backward)
2. Local roundtrip: dispatch + combine recovers original input (forward + backward)
"""

import pytest

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec

from distributed_test_base import generate_configs
from utils import assert_allclose, pytest_parametrize_wrapper

# High-level API with VJP support
from transformer_engine.jax.permutation import (
    token_dispatch,
    token_combine,
)

# Reference implementations from test_permutation.py
from test_permutation import (
    reference_make_row_id_map,
    _reference_permute_impl,
    _reference_unpermute_impl,
    reference_token_combine,
)

# Dispatch/combine test cases: (num_tokens, num_experts, hidden_size, topk)
# topk = number of experts each token is routed to
# Includes small, medium-large, and largest stress test cases.
ALL_DISPATCH_COMBINE_CASES = [
    (128, 4, 64, 2),
    (4096, 32, 1280, 2),
    (4096, 256, 4096, 6),
]
DISPATCH_COMBINE_CASES = {
    "L0": ALL_DISPATCH_COMBINE_CASES[0:1],
    "L2": ALL_DISPATCH_COMBINE_CASES,
}

# Dispatch/combine with padding test cases: (num_tokens, num_experts, hidden_size, topk, align_size)
ALL_DISPATCH_COMBINE_PADDING_CASES = [
    (128, 4, 64, 2, 8),
    (4096, 32, 1280, 2, 128),
    (4096, 256, 4096, 6, 16),
]
DISPATCH_COMBINE_PADDING_CASES = {
    "L0": ALL_DISPATCH_COMBINE_PADDING_CASES[0:1],
    "L2": ALL_DISPATCH_COMBINE_PADDING_CASES,
}

# Dtypes for testing
ALL_DTYPES = [jnp.float32, jnp.bfloat16]
DTYPES = {
    "L0": [jnp.float32],
    "L2": ALL_DTYPES,
}


class TestDistributedPermutation:
    """Test distributed/sharded execution of MoE permutation primitives.

    These tests validate that custom partitioning produces correct LOCAL results
    when inputs are sharded across multiple devices.

    Key insight: With data-parallel MoE, each GPU independently processes its
    local tokens. The row_id_map is generated locally and contains LOCAL indices.
    We verify correctness by comparing each shard's output against the reference
    implementation run on that shard's local data.
    """

    @staticmethod
    def compute_padded_output_size(
        num_tokens: int,
        num_experts: int,
        topk: int,
        align_size: int,
        num_dp_devices: int,
    ) -> int:
        """Compute global_num_out_tokens for distributed padding tests.

        Each device processes local_num_tokens tokens. We compute the worst-case
        padded output size per device, then multiply by num_dp_devices to get
        a global size that ensures global / num_dp >= local_worst.
        """
        local_num_tokens = num_tokens // num_dp_devices
        local_raw_out = local_num_tokens * topk
        local_worst = ((local_raw_out + num_experts * (align_size - 1)) // align_size) * align_size
        return local_worst * num_dp_devices

    @staticmethod
    def generate_routing_map(
        num_tokens: int,
        num_experts: int,
        topk: int = 2,  # Number of experts each token is routed to (max 1s per row).
        key: jax.Array = None,
    ):
        if key is None:
            key = jax.random.PRNGKey(0)

        routing_map = jnp.zeros((num_tokens, num_experts), dtype=jnp.int32)
        for token_idx in range(num_tokens):
            key, subkey = jax.random.split(key)
            expert_indices = jax.random.choice(subkey, num_experts, shape=(topk,), replace=False)
            routing_map = routing_map.at[token_idx, expert_indices].set(1)

        return routing_map

    @pytest.mark.parametrize("device_count,mesh_shape,mesh_axes,mesh_resource", generate_configs())
    @pytest_parametrize_wrapper(
        "num_tokens,num_experts,hidden_size,topk",
        DISPATCH_COMBINE_CASES,
    )
    @pytest_parametrize_wrapper("dtype", DTYPES)
    @pytest_parametrize_wrapper("use_shardy", [False, True])
    def test_local_token_dispatch(
        self,
        device_count,
        mesh_shape,
        mesh_axes,
        mesh_resource,
        num_tokens,
        num_experts,
        hidden_size,
        topk,
        dtype,
        use_shardy,
    ):
        """
        Test token_dispatch with sharded inputs.

        Verifies that sharded execution produces the same result as chunk-wise
        reference execution. The sharded primitive:
        1. Receives global num_out_tokens (partition function divides it)
        2. Each GPU operates on its local shard independently
        3. Results are gathered (concatenated) across GPUs

        Output ordering: [GPU0_expert0, GPU0_expert1, ... | GPU1_expert0, ...]

        The reference processes each chunk independently and concatenates,
        matching the sharded execution's output ordering.
        Tests both forward pass (output values) and backward pass (gradients).
        """
        jax.config.update("jax_use_shardy_partitioner", use_shardy)
        key = jax.random.PRNGKey(42)

        # Generate global inputs
        key, inp_key, prob_key = jax.random.split(key, 3)
        inp = jax.random.uniform(
            inp_key, (num_tokens, hidden_size), dtype=dtype, minval=-1.0, maxval=1.0
        )
        routing_map = self.generate_routing_map(num_tokens, num_experts, topk, key)
        probs = jax.random.uniform(
            prob_key, (num_tokens, num_experts), dtype=dtype, minval=0.1, maxval=1.0
        )

        devices = np.asarray(jax.devices()[:device_count]).reshape(*mesh_shape)
        mesh = Mesh(devices, mesh_axes)

        # Shard on token (batch) dimension
        dp_axis = mesh_resource.dp_resource
        sharded_pspec = PartitionSpec(dp_axis, None)

        # Compute num_out_tokens as concrete values
        # Global num_out_tokens is passed to token_dispatch (partition function divides it)
        # Local num_out_tokens is used for reference implementation
        num_dp_devices = mesh.shape[dp_axis] if dp_axis else 1
        global_num_out_tokens = num_tokens * topk
        local_num_tokens = num_tokens // num_dp_devices
        local_num_out_tokens = local_num_tokens * topk

        with mesh:
            inp_sharding = NamedSharding(mesh, sharded_pspec)
            routing_sharding = NamedSharding(mesh, sharded_pspec)
            probs_sharding = NamedSharding(mesh, sharded_pspec)

            # Shard the inputs
            inp_sharded = jax.device_put(inp, inp_sharding)
            routing_sharded = jax.device_put(routing_map, routing_sharding)
            probs_sharded = jax.device_put(probs, probs_sharding)

            # ================================================================
            # Forward pass test
            # ================================================================
            @jax.jit
            def target_dispatch(x, rm, p):
                # Pass global num_out_tokens - partition function divides it
                out, perm_probs, rid_map, _, _ = token_dispatch(
                    x, rm, global_num_out_tokens, probs=p
                )
                return out, perm_probs, rid_map

            # Reference: process each GPU's shard independently, then concatenate
            # This matches how the sharded primitive operates:
            # - Each GPU processes its local shard
            # - Results are gathered (concatenated) across GPUs
            # Output ordering: [GPU0_exp0, GPU0_exp1, ... | GPU1_exp0, GPU1_exp1, ...]
            inp_shards = jnp.reshape(inp, (num_dp_devices, local_num_tokens, hidden_size))
            routing_shards = jnp.reshape(
                routing_map, (num_dp_devices, local_num_tokens, num_experts)
            )
            probs_shards = jnp.reshape(probs, (num_dp_devices, local_num_tokens, num_experts))

            ref_outputs = []
            ref_perm_probs_list = []
            ref_rid_maps = []
            for i in range(num_dp_devices):
                shard_rid_map = reference_make_row_id_map(routing_shards[i])
                shard_out, shard_perm_probs = _reference_permute_impl(
                    inp_shards[i], shard_rid_map, probs_shards[i], local_num_out_tokens
                )
                ref_outputs.append(shard_out)
                ref_perm_probs_list.append(shard_perm_probs)
                ref_rid_maps.append(shard_rid_map)

            # Concatenate like all_gather would
            ref_out = jnp.concatenate(ref_outputs, axis=0)
            ref_perm_probs = jnp.concatenate(ref_perm_probs_list, axis=0)
            ref_rid_map = jnp.concatenate(ref_rid_maps, axis=0)

            # Run target on sharded inputs
            target_out, target_perm_probs, target_rid_map = target_dispatch(
                inp_sharded, routing_sharded, probs_sharded
            )

            # Compare forward outputs
            assert_allclose(jax.device_get(target_out), ref_out, dtype=dtype)
            assert_allclose(jax.device_get(target_perm_probs), ref_perm_probs, dtype=dtype)

            # Verify row_id_map n_routed column matches routing_map sum
            target_rid_map_np = jax.device_get(target_rid_map)
            assert jnp.array_equal(
                target_rid_map_np[:, -1], ref_rid_map[:, -1]
            ), "n_routed column mismatch"

            # Sanity checks
            target_out_np = jax.device_get(target_out)
            target_perm_probs_np = jax.device_get(target_perm_probs)
            assert not np.any(np.isnan(target_out_np)), "Output contains NaN"
            assert not np.any(np.isnan(target_perm_probs_np)), "Permuted probs contain NaN"
            assert np.all(target_perm_probs_np >= 0), "Permuted probs contain negative values"

            # ================================================================
            # Backward pass test (gradients)
            # ================================================================
            def target_loss(x, rm, p):
                out, perm_probs, _, _, _ = token_dispatch(x, rm, global_num_out_tokens, probs=p)
                return jnp.sum(out**2) + jnp.sum(perm_probs**2)

            # Reference loss: process chunks independently and sum
            def ref_chunk_loss(inp_chunk, routing_chunk, probs_chunk):
                rid_map = reference_make_row_id_map(routing_chunk)
                out, perm_probs = _reference_permute_impl(
                    inp_chunk, rid_map, probs_chunk, local_num_out_tokens
                )
                return jnp.sum(out**2) + jnp.sum(perm_probs**2)

            target_grad_fn = jax.jit(jax.grad(target_loss, argnums=(0, 2)))
            ref_chunk_grad_fn = jax.jit(jax.grad(ref_chunk_loss, argnums=(0, 2)))

            target_inp_grad, target_probs_grad = target_grad_fn(
                inp_sharded, routing_sharded, probs_sharded
            )

            # Compute reference gradients per chunk, then concatenate
            ref_inp_grads = []
            ref_probs_grads = []
            for i in range(num_dp_devices):
                chunk_inp_grad, chunk_probs_grad = ref_chunk_grad_fn(
                    inp_shards[i], routing_shards[i], probs_shards[i]
                )
                ref_inp_grads.append(chunk_inp_grad)
                ref_probs_grads.append(chunk_probs_grad)

            ref_inp_grad = jnp.concatenate(ref_inp_grads, axis=0)
            ref_probs_grad = jnp.concatenate(ref_probs_grads, axis=0)

            assert_allclose(jax.device_get(target_inp_grad), ref_inp_grad, dtype=dtype)
            assert_allclose(jax.device_get(target_probs_grad), ref_probs_grad, dtype=dtype)

    @pytest.mark.parametrize("device_count,mesh_shape,mesh_axes,mesh_resource", generate_configs())
    @pytest_parametrize_wrapper(
        "num_tokens,num_experts,hidden_size,topk",
        DISPATCH_COMBINE_CASES,
    )
    @pytest_parametrize_wrapper("dtype", DTYPES)
    @pytest_parametrize_wrapper("use_shardy", [False, True])
    def test_local_roundtrip(
        self,
        device_count,
        mesh_shape,
        mesh_axes,
        mesh_resource,
        num_tokens,
        num_experts,
        hidden_size,
        topk,
        dtype,
        use_shardy,
    ):
        """
        Test roundtrip: token_dispatch followed by token_combine with sharded inputs.

        Each GPU:
        1. Gets a shard of the input and routing_map
        2. Performs local dispatch (permute)
        3. Performs local combine (unpermute)
        4. With uniform merging probs, should recover original input

        Tests both forward pass and backward pass (gradient should be 2*x).
        """
        jax.config.update("jax_use_shardy_partitioner", use_shardy)
        key = jax.random.PRNGKey(42)

        # Generate global inputs
        key, inp_key = jax.random.split(key, 2)
        inp = jax.random.uniform(
            inp_key, (num_tokens, hidden_size), dtype=dtype, minval=-1.0, maxval=1.0
        )
        routing_map = self.generate_routing_map(num_tokens, num_experts, topk, key)

        # Uniform merging probs for perfect roundtrip
        uniform_merging_probs = routing_map.astype(dtype) / jnp.maximum(
            jnp.sum(routing_map, axis=1, keepdims=True), 1.0
        )

        devices = np.asarray(jax.devices()[:device_count]).reshape(*mesh_shape)
        mesh = Mesh(devices, mesh_axes)

        dp_axis = mesh_resource.dp_resource
        sharded_pspec = PartitionSpec(dp_axis, None)

        # Compute num_out_tokens as concrete value
        # Global num_out_tokens is passed to token_dispatch (partition function divides it)
        global_num_out_tokens = num_tokens * topk

        with mesh:
            inp_sharding = NamedSharding(mesh, sharded_pspec)
            routing_sharding = NamedSharding(mesh, sharded_pspec)
            merging_sharding = NamedSharding(mesh, sharded_pspec)

            inp_sharded = jax.device_put(inp, inp_sharding)
            routing_sharded = jax.device_put(routing_map, routing_sharding)
            merging_sharded = jax.device_put(uniform_merging_probs, merging_sharding)

            # ================================================================
            # Forward pass test
            # ================================================================
            @jax.jit
            def roundtrip(x, rm, mprobs):
                dispatched, _, rid_map, _, _ = token_dispatch(x, rm, global_num_out_tokens)
                return token_combine(dispatched, rid_map, mprobs)

            roundtrip_out = roundtrip(inp_sharded, routing_sharded, merging_sharded)

            # Should recover original input
            assert_allclose(jax.device_get(roundtrip_out), jax.device_get(inp_sharded), dtype=dtype)

            # ================================================================
            # Backward pass test (gradients)
            # ================================================================
            def roundtrip_loss(x, rm, mprobs):
                dispatched, _, rid_map, _, _ = token_dispatch(x, rm, global_num_out_tokens)
                combined = token_combine(dispatched, rid_map, mprobs)
                return jnp.sum(combined**2)

            # With uniform merging probs, roundtrip is identity, so gradient should be 2*x
            grad_fn = jax.jit(jax.grad(roundtrip_loss, argnums=0))
            computed_grad = grad_fn(inp_sharded, routing_sharded, merging_sharded)

            expected_grad = 2.0 * inp_sharded

            assert_allclose(
                jax.device_get(computed_grad), jax.device_get(expected_grad), dtype=dtype
            )

    @pytest.mark.parametrize("device_count,mesh_shape,mesh_axes,mesh_resource", generate_configs())
    @pytest_parametrize_wrapper(
        "num_tokens,num_experts,hidden_size,topk,align_size",
        DISPATCH_COMBINE_PADDING_CASES,
    )
    @pytest_parametrize_wrapper("dtype", DTYPES)
    @pytest_parametrize_wrapper("use_shardy", [False, True])
    def test_local_token_dispatch_with_padding(
        self,
        device_count,
        mesh_shape,
        mesh_axes,
        mesh_resource,
        num_tokens,
        num_experts,
        hidden_size,
        topk,
        align_size,
        dtype,
        use_shardy,
    ):
        """
        Test token_dispatch with padding using sharded inputs.

        Tests both forward pass (output values) and backward pass (gradients).
        """
        jax.config.update("jax_use_shardy_partitioner", use_shardy)
        key = jax.random.PRNGKey(42)

        # Generate global inputs
        key, inp_key, prob_key = jax.random.split(key, 3)
        inp = jax.random.uniform(
            inp_key, (num_tokens, hidden_size), dtype=dtype, minval=-1.0, maxval=1.0
        )
        routing_map = self.generate_routing_map(num_tokens, num_experts, topk, key)
        probs = jax.random.uniform(
            prob_key, (num_tokens, num_experts), dtype=dtype, minval=0.1, maxval=1.0
        )

        devices = np.asarray(jax.devices()[:device_count]).reshape(*mesh_shape)
        mesh = Mesh(devices, mesh_axes)

        dp_axis = mesh_resource.dp_resource
        sharded_pspec = PartitionSpec(dp_axis, None)
        num_dp_devices = mesh.shape[dp_axis] if dp_axis else 1

        # For padding + sharding, we need to account for per-shard padding overhead.
        # Each shard needs E*(A-1) extra space for worst-case padding.
        # Compute global_num_out_tokens such that global / num_dp >= local_worst.
        global_num_out_tokens = self.compute_padded_output_size(
            num_tokens, num_experts, topk, align_size, num_dp_devices
        )

        with mesh:
            inp_sharding = NamedSharding(mesh, sharded_pspec)
            routing_sharding = NamedSharding(mesh, sharded_pspec)
            probs_sharding = NamedSharding(mesh, sharded_pspec)

            inp_sharded = jax.device_put(inp, inp_sharding)
            routing_sharded = jax.device_put(routing_map, routing_sharding)
            probs_sharded = jax.device_put(probs, probs_sharding)

            # ================================================================
            # Forward pass test
            # ================================================================
            @jax.jit
            def dispatch_with_padding(x, rm, p):
                out, perm_probs, rid_map, pad_offsets, _ = token_dispatch(
                    x, rm, global_num_out_tokens, probs=p, align_size=align_size
                )
                return out, perm_probs, rid_map, pad_offsets

            out, perm_probs, rid_map, pad_offsets = dispatch_with_padding(
                inp_sharded, routing_sharded, probs_sharded
            )

            # Sanity checks
            out_np = jax.device_get(out)
            perm_probs_np = jax.device_get(perm_probs)
            assert not np.any(np.isnan(out_np)), "Output contains NaN"
            assert not np.any(np.isnan(perm_probs_np)), "Permuted probs contain NaN"
            assert np.all(perm_probs_np >= 0), "Permuted probs contain negative values"

            # ================================================================
            # Backward pass test (gradients)
            # ================================================================
            def loss_with_padding(x, rm, p):
                out, perm_probs, _, _, _ = token_dispatch(
                    x, rm, global_num_out_tokens, probs=p, align_size=align_size
                )
                return jnp.sum(out**2) + jnp.sum(perm_probs**2)

            grad_fn = jax.jit(jax.grad(loss_with_padding, argnums=(0, 2)))
            inp_grad, probs_grad = grad_fn(inp_sharded, routing_sharded, probs_sharded)

            # Gradients should not contain NaN
            assert not np.any(np.isnan(jax.device_get(inp_grad))), "Input gradient contains NaN"
            assert not np.any(np.isnan(jax.device_get(probs_grad))), "Probs gradient contains NaN"

    @pytest.mark.parametrize("device_count,mesh_shape,mesh_axes,mesh_resource", generate_configs())
    @pytest_parametrize_wrapper(
        "num_tokens,num_experts,hidden_size,topk,align_size",
        DISPATCH_COMBINE_PADDING_CASES,
    )
    @pytest_parametrize_wrapper("dtype", DTYPES)
    @pytest_parametrize_wrapper("use_shardy", [False, True])
    def test_local_roundtrip_with_padding(
        self,
        device_count,
        mesh_shape,
        mesh_axes,
        mesh_resource,
        num_tokens,
        num_experts,
        hidden_size,
        topk,
        align_size,
        dtype,
        use_shardy,
    ):
        """
        Test roundtrip with padding/alignment using sharded inputs.

        With uniform merging probs, should recover original input.
        Tests both forward pass and backward pass.
        """
        jax.config.update("jax_use_shardy_partitioner", use_shardy)
        key = jax.random.PRNGKey(42)

        # Generate inputs
        key, inp_key = jax.random.split(key, 2)
        inp = jax.random.uniform(
            inp_key, (num_tokens, hidden_size), dtype=dtype, minval=-1.0, maxval=1.0
        )
        routing_map = self.generate_routing_map(num_tokens, num_experts, topk, key)

        # Uniform merging probs
        uniform_merging_probs = routing_map.astype(dtype) / jnp.maximum(
            jnp.sum(routing_map, axis=1, keepdims=True), 1.0
        )

        devices = np.asarray(jax.devices()[:device_count]).reshape(*mesh_shape)
        mesh = Mesh(devices, mesh_axes)

        dp_axis = mesh_resource.dp_resource
        sharded_pspec = PartitionSpec(dp_axis, None)
        num_dp_devices = mesh.shape[dp_axis] if dp_axis else 1

        # For padding + sharding, we need to account for per-shard padding overhead.
        # Each shard needs E*(A-1) extra space for worst-case padding.
        # Compute global_num_out_tokens such that global / num_dp >= local_worst.
        global_num_out_tokens = self.compute_padded_output_size(
            num_tokens, num_experts, topk, align_size, num_dp_devices
        )

        with mesh:
            inp_sharding = NamedSharding(mesh, sharded_pspec)
            routing_sharding = NamedSharding(mesh, sharded_pspec)
            merging_sharding = NamedSharding(mesh, sharded_pspec)

            inp_sharded = jax.device_put(inp, inp_sharding)
            routing_sharded = jax.device_put(routing_map, routing_sharding)
            merging_sharded = jax.device_put(uniform_merging_probs, merging_sharding)

            # ================================================================
            # Forward pass test
            # ================================================================
            @jax.jit
            def roundtrip_with_padding(x, rm, mprobs):
                dispatched, _, rid_map, pad_offsets, _ = token_dispatch(
                    x, rm, global_num_out_tokens, align_size=align_size
                )
                return token_combine(dispatched, rid_map, mprobs, pad_offsets)

            roundtrip_out = roundtrip_with_padding(inp_sharded, routing_sharded, merging_sharded)

            # Should recover original input
            assert_allclose(jax.device_get(roundtrip_out), jax.device_get(inp_sharded), dtype=dtype)

            # ================================================================
            # Backward pass test (gradients)
            # ================================================================
            def roundtrip_loss_with_padding(x, rm, mprobs):
                dispatched, _, rid_map, pad_offsets, _ = token_dispatch(
                    x, rm, global_num_out_tokens, align_size=align_size
                )
                combined = token_combine(dispatched, rid_map, mprobs, pad_offsets)
                return jnp.sum(combined**2)

            # With uniform merging probs, roundtrip is identity, so gradient should be 2*x
            grad_fn = jax.jit(jax.grad(roundtrip_loss_with_padding, argnums=0))
            computed_grad = grad_fn(inp_sharded, routing_sharded, merging_sharded)

            expected_grad = 2.0 * inp_sharded

            assert_allclose(
                jax.device_get(computed_grad), jax.device_get(expected_grad), dtype=dtype
            )
