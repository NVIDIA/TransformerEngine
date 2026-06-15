# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Native JAX/Flax MoE baseline used by ``moe.rst``.

This file intentionally contains the lower-level reference mechanics so the
tutorial can focus on model-level code. It does not import TransformerEngine:
the router, expert-parallel ragged all-to-all, local ragged chunk reorder, and
ragged expert matmuls are implemented with JAX and Flax only.
"""

import inspect
from functools import partial
from typing import Any, Callable, Optional, Tuple

import jax
import jax.numpy as jnp
from flax import linen as nn
from jax.sharding import PartitionSpec as P


def _forward_a2a_params(
    all_tokens_per_expert: jnp.ndarray,
    shard_id: jnp.ndarray,
    num_ep: int,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Build ``ragged_all_to_all`` offsets/sizes for dispatch."""
    num_experts = all_tokens_per_expert.shape[1]
    experts_per_shard = num_experts // num_ep

    local_tokens_per_expert = jax.lax.dynamic_slice(
        all_tokens_per_expert,
        start_indices=(shard_id, 0),
        slice_sizes=(1, num_experts),
    ).squeeze(0)
    local_by_destination = local_tokens_per_expert.reshape(num_ep, experts_per_shard)
    send_sizes = jnp.sum(local_by_destination, axis=1).astype(jnp.int32)
    input_offsets = jnp.concatenate([jnp.array([0], dtype=jnp.int32), jnp.cumsum(send_sizes)[:-1]])

    local_expert_start = shard_id * experts_per_shard
    local_expert_columns = jax.lax.dynamic_slice(
        all_tokens_per_expert,
        start_indices=(0, local_expert_start),
        slice_sizes=(num_ep, experts_per_shard),
    )
    recv_sizes = jnp.sum(local_expert_columns, axis=1).astype(jnp.int32)

    sends_to_destination = jnp.sum(
        all_tokens_per_expert.reshape(num_ep, num_ep, experts_per_shard),
        axis=2,
    ).astype(jnp.int32)
    cumulative = jnp.cumsum(
        jnp.concatenate(
            [jnp.zeros((1, num_ep), dtype=jnp.int32), sends_to_destination],
            axis=0,
        ),
        axis=0,
    )
    output_offsets = jax.lax.dynamic_slice(
        cumulative,
        start_indices=(shard_id, 0),
        slice_sizes=(1, num_ep),
    ).squeeze(0)

    return input_offsets, send_sizes, output_offsets, recv_sizes


def _reverse_a2a_params(
    all_tokens_per_expert: jnp.ndarray,
    shard_id: jnp.ndarray,
    num_ep: int,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Build ``ragged_all_to_all`` offsets/sizes for combine."""
    num_experts = all_tokens_per_expert.shape[1]
    experts_per_shard = num_experts // num_ep
    local_expert_start = shard_id * experts_per_shard

    local_expert_columns = jax.lax.dynamic_slice(
        all_tokens_per_expert,
        start_indices=(0, local_expert_start),
        slice_sizes=(num_ep, experts_per_shard),
    )
    send_sizes = jnp.sum(local_expert_columns, axis=1).astype(jnp.int32)
    input_offsets = jnp.concatenate([jnp.array([0], dtype=jnp.int32), jnp.cumsum(send_sizes)[:-1]])

    local_tokens_per_expert = jax.lax.dynamic_slice(
        all_tokens_per_expert,
        start_indices=(shard_id, 0),
        slice_sizes=(1, num_experts),
    ).squeeze(0)
    local_by_destination = local_tokens_per_expert.reshape(num_ep, experts_per_shard)
    recv_sizes = jnp.sum(local_by_destination, axis=1).astype(jnp.int32)

    forward_sends_to = jnp.sum(
        all_tokens_per_expert.reshape(num_ep, num_ep, experts_per_shard),
        axis=2,
    ).astype(jnp.int32)
    reverse_sends_to = jnp.transpose(forward_sends_to)
    cumulative = jnp.cumsum(
        jnp.concatenate(
            [jnp.zeros((1, num_ep), dtype=jnp.int32), reverse_sends_to],
            axis=0,
        ),
        axis=0,
    )
    output_offsets = jax.lax.dynamic_slice(
        cumulative,
        start_indices=(shard_id, 0),
        slice_sizes=(1, num_ep),
    ).squeeze(0)

    return input_offsets, send_sizes, output_offsets, recv_sizes


def _reorder_ragged_chunks(
    x: jnp.ndarray,
    chunk_sizes: jnp.ndarray,
    source_order: jnp.ndarray,
    target_order: jnp.ndarray,
) -> jnp.ndarray:
    """Reorder a fixed-size ragged buffer from one chunk order to another."""
    source_sizes = chunk_sizes[source_order]
    source_starts = jnp.concatenate(
        [jnp.array([0], dtype=jnp.int32), jnp.cumsum(source_sizes)[:-1]]
    )
    source_ends = source_starts + source_sizes

    target_sizes = chunk_sizes[target_order]
    target_starts_by_position = jnp.concatenate(
        [jnp.array([0], dtype=jnp.int32), jnp.cumsum(target_sizes)[:-1]]
    )
    target_position_by_chunk = jnp.argsort(target_order)
    target_start_by_chunk = target_starts_by_position[target_position_by_chunk]

    rows = jnp.arange(x.shape[0], dtype=jnp.int32)
    in_source_chunk = (rows[:, None] >= source_starts[None, :]) & (
        rows[:, None] < source_ends[None, :]
    )
    valid = jnp.any(in_source_chunk, axis=1)
    source_position = jnp.argmax(in_source_chunk, axis=1)
    chunk_id = source_order[source_position]
    row_in_chunk = rows - source_starts[source_position]
    target_rows = target_start_by_chunk[chunk_id] + row_in_chunk
    target_rows = jnp.where(valid, target_rows, 0)

    updates = jnp.where(valid[:, None], x, jnp.zeros_like(x))
    return jnp.zeros_like(x).at[target_rows].add(updates)


def _route_tokens(
    x_2d: jnp.ndarray,
    gate_kernel: jnp.ndarray,
    num_experts_per_tok: int,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Softmax top-k router matching the tutorial's default TE path."""
    logits = x_2d.astype(jnp.float32) @ gate_kernel.astype(jnp.float32)
    probs = jax.nn.softmax(logits, axis=-1)
    weights, experts = jax.lax.top_k(probs, num_experts_per_tok)
    weights = weights / jnp.sum(weights, axis=-1, keepdims=True)
    return experts.astype(jnp.int32), weights.astype(x_2d.dtype)


def _native_moe_local(
    captured: dict,
    *,
    ep_axis: str,
    num_experts: int,
    num_experts_per_tok: int,
    recv_buffer_rows: int,
    dtype: jnp.dtype,
) -> jnp.ndarray:
    """One shard of the native EP MoE forward pass."""
    x = captured["x"]
    gate_kernel = captured["gate_kernel"]
    wi_0 = captured["wi_0"]
    wi_1 = captured["wi_1"]
    wo = captured["wo"]

    batch, sequence, hidden = x.shape
    tokens = batch * sequence
    x_2d = x.reshape(tokens, hidden)

    selected_experts, routing_weights = _route_tokens(x_2d, gate_kernel, num_experts_per_tok)
    flat_experts = selected_experts.reshape(-1)
    flat_token_ids = jnp.repeat(jnp.arange(tokens, dtype=jnp.int32), num_experts_per_tok)
    flat_weights = routing_weights.reshape(-1)

    sort_order = jnp.argsort(flat_experts, stable=True)
    sorted_experts = flat_experts[sort_order]
    sorted_x = x_2d[flat_token_ids][sort_order]
    tokens_per_expert = jnp.bincount(
        sorted_experts,
        length=num_experts,
        minlength=num_experts,
    ).astype(jnp.int32)

    shard_id = jax.lax.axis_index(ep_axis)
    num_ep = jax.lax.psum(1, ep_axis)
    experts_per_shard = num_experts // num_ep

    all_tokens_per_expert = jax.lax.all_gather(
        tokens_per_expert[None, :],
        axis_name=ep_axis,
        axis=0,
        tiled=True,
    )

    in_off, send_sz, out_off, recv_sz = _forward_a2a_params(all_tokens_per_expert, shard_id, num_ep)
    x_recv = jax.lax.ragged_all_to_all(
        sorted_x,
        jnp.zeros((recv_buffer_rows, hidden), dtype=sorted_x.dtype),
        in_off,
        send_sz,
        out_off,
        recv_sz,
        axis_name=ep_axis,
    )

    local_expert_start = shard_id * experts_per_shard
    local_counts_by_source = jax.lax.dynamic_slice(
        all_tokens_per_expert,
        start_indices=(0, local_expert_start),
        slice_sizes=(num_ep, experts_per_shard),
    ).astype(jnp.int32)
    local_chunk_sizes = local_counts_by_source.reshape(-1)
    source_major_order = jnp.arange(num_ep * experts_per_shard, dtype=jnp.int32)
    expert_major_order = source_major_order.reshape(num_ep, experts_per_shard).T.reshape(-1)
    local_group_sizes = jnp.sum(local_counts_by_source, axis=0).astype(jnp.int32)

    x_expert_major = _reorder_ragged_chunks(
        x_recv,
        local_chunk_sizes,
        source_major_order,
        expert_major_order,
    )
    wi_combined = jnp.concatenate([wi_0, wi_1], axis=-1)
    hidden_combined = jax.lax.ragged_dot(x_expert_major, wi_combined, local_group_sizes)
    hidden_0, hidden_1 = jnp.split(hidden_combined, 2, axis=-1)
    activated = jax.nn.silu(hidden_0) * hidden_1
    expert_output = jax.lax.ragged_dot(activated, wo, local_group_sizes).astype(dtype)

    source_major_output = _reorder_ragged_chunks(
        expert_output,
        local_chunk_sizes,
        expert_major_order,
        source_major_order,
    )
    in_off, send_sz, out_off, recv_sz = _reverse_a2a_params(all_tokens_per_expert, shard_id, num_ep)
    returned = jax.lax.ragged_all_to_all(
        source_major_output,
        jnp.zeros_like(sorted_x),
        in_off,
        send_sz,
        out_off,
        recv_sz,
        axis_name=ep_axis,
    )

    unsorted = jnp.zeros_like(returned).at[sort_order].set(returned)
    token_outputs = unsorted.reshape(tokens, num_experts_per_tok, hidden)
    weighted = token_outputs * flat_weights.reshape(tokens, num_experts_per_tok, 1)
    return jnp.sum(weighted, axis=1).reshape(batch, sequence, hidden).astype(dtype)


def native_moe_ep(
    x: jnp.ndarray,
    gate_kernel: jnp.ndarray,
    wi_0: jnp.ndarray,
    wi_1: jnp.ndarray,
    wo: jnp.ndarray,
    *,
    mesh: Any,
    ep_axis: str,
    data_parallelism_axes: Tuple[str, ...],
    num_experts: int,
    num_experts_per_tok: int,
    dtype: jnp.dtype,
) -> jnp.ndarray:
    """Run the native BF16 EP MoE baseline on an active JAX mesh."""
    if num_experts % mesh.shape[ep_axis] != 0:
        raise ValueError(
            f"num_experts={num_experts} must be divisible by EP size={mesh.shape[ep_axis]}"
        )

    if data_parallelism_axes:
        batch_axis = (ep_axis, *data_parallelism_axes)
    else:
        batch_axis = ep_axis

    dp_size = 1
    for axis in data_parallelism_axes:
        dp_size *= mesh.shape[axis]

    batch, sequence, _ = x.shape
    required_batch_multiple = mesh.shape[ep_axis] * dp_size
    if batch % required_batch_multiple != 0:
        raise ValueError(f"batch={batch} must be divisible by ep*dp={required_batch_multiple}")

    recv_buffer_rows = (batch // dp_size) * sequence * num_experts_per_tok
    captured = {
        "x": x,
        "gate_kernel": gate_kernel,
        "wi_0": wi_0,
        "wi_1": wi_1,
        "wo": wo,
    }
    in_specs = (
        {
            "x": P(batch_axis, None, None),
            "gate_kernel": P(),
            "wi_0": P(ep_axis, None, None),
            "wi_1": P(ep_axis, None, None),
            "wo": P(ep_axis, None, None),
        },
    )

    body = partial(
        _native_moe_local,
        ep_axis=ep_axis,
        num_experts=num_experts,
        num_experts_per_tok=num_experts_per_tok,
        recv_buffer_rows=recv_buffer_rows,
        dtype=dtype,
    )
    shard_map_kwargs = {
        "mesh": mesh,
        "in_specs": in_specs,
        "out_specs": P(batch_axis, None, None),
    }
    shard_map_params = inspect.signature(jax.shard_map).parameters
    if "check_rep" in shard_map_params:
        shard_map_kwargs["check_rep"] = False
    elif "check_vma" in shard_map_params:
        shard_map_kwargs["check_vma"] = False

    return jax.shard_map(body, **shard_map_kwargs)(captured)


class NativeMoEBlock(nn.Module):
    """Native JAX/Flax BF16 EP MoE block used as the tutorial baseline."""

    mesh: Any
    num_experts: int = 8
    num_experts_per_tok: int = 2
    intermediate_size: int = 2048
    ep_axis: str = "ep"
    data_parallelism_axes: Tuple[str, ...] = ("fsdp",)
    dtype: jnp.dtype = jnp.bfloat16
    kernel_init: Optional[Callable] = None

    def __post_init__(self):
        if self.kernel_init is None:
            object.__setattr__(
                self,
                "kernel_init",
                nn.initializers.variance_scaling(
                    1.0,
                    "fan_in",
                    "truncated_normal",
                    dtype=self.dtype,
                ),
            )
        super().__post_init__()

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        hidden = x.shape[-1]
        gate_kernel = self.param(
            "gate_kernel",
            self.kernel_init,
            (hidden, self.num_experts),
            self.dtype,
        )
        wi_0 = self.param(
            "wi_0",
            self.kernel_init,
            (self.num_experts, hidden, self.intermediate_size),
            self.dtype,
        )
        wi_1 = self.param(
            "wi_1",
            self.kernel_init,
            (self.num_experts, hidden, self.intermediate_size),
            self.dtype,
        )
        wo = self.param(
            "wo",
            self.kernel_init,
            (self.num_experts, self.intermediate_size, hidden),
            self.dtype,
        )
        return native_moe_ep(
            x,
            gate_kernel,
            wi_0,
            wi_1,
            wo,
            mesh=self.mesh,
            ep_axis=self.ep_axis,
            data_parallelism_axes=self.data_parallelism_axes,
            num_experts=self.num_experts,
            num_experts_per_tok=self.num_experts_per_tok,
            dtype=self.dtype,
        )
