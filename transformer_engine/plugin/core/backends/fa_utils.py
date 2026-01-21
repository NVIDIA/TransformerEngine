# Copyright (c) 2025, BAAI. All rights reserved.
#
# See LICENSE for license information.

"""Common utilities for Flash Attention backends with Context Parallelism support."""

from typing import Any, Tuple

import torch
import torch.distributed as dist


class AllGatherFunc(torch.autograd.Function):
    """Autograd function for all-gather along sequence dimension with proper backward."""

    @staticmethod
    def forward(ctx, input_tensor: torch.Tensor, cp_group: Any, seq_dim: int) -> torch.Tensor:
        world_size = dist.get_world_size(cp_group)
        gathered_list = [torch.empty_like(input_tensor) for _ in range(world_size)]
        dist.all_gather(gathered_list, input_tensor, group=cp_group)
        ctx.cp_group = cp_group
        ctx.world_size = world_size
        ctx.seq_dim = seq_dim
        return torch.cat(gathered_list, dim=seq_dim)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None, None]:
        # Split the gradient and reduce_scatter
        grad_chunks = torch.chunk(grad_output, ctx.world_size, dim=ctx.seq_dim)
        local_grad = torch.zeros_like(grad_chunks[0])
        grad_list = [chunk.contiguous() for chunk in grad_chunks]
        dist.reduce_scatter(local_grad, grad_list, group=ctx.cp_group)
        return local_grad, None, None


def all_gather_along_seq(
    tensor: torch.Tensor,
    cp_group: Any,
    seq_dim: int = 2,
) -> torch.Tensor:
    """All-gather tensor along sequence dimension across CP group.

    Args:
        tensor: Input tensor to gather.
        cp_group: Context parallelism process group.
        seq_dim: Sequence dimension (default: 2 for BHSD format).

    Returns:
        Gathered tensor with sequence dimension scaled by CP world size.
    """
    world_size = dist.get_world_size(cp_group)
    if world_size == 1:
        return tensor

    tensor = tensor.contiguous()
    return AllGatherFunc.apply(tensor, cp_group, seq_dim)


def reduce_scatter_along_seq(
    tensor: torch.Tensor,
    cp_group: Any,
    seq_dim: int = 2,
) -> torch.Tensor:
    """Reduce-scatter tensor along sequence dimension across CP group.

    Args:
        tensor: Input tensor to reduce-scatter.
        cp_group: Context parallelism process group.
        seq_dim: Sequence dimension (default: 2 for BHSD format).

    Returns:
        Reduced tensor with sequence dimension divided by CP world size.
    """
    world_size = dist.get_world_size(cp_group)
    if world_size == 1:
        return tensor

    tensor = tensor.contiguous()
    seq_len = tensor.shape[seq_dim]
    chunk_size = seq_len // world_size

    output = torch.empty(
        *tensor.shape[:seq_dim], chunk_size, *tensor.shape[seq_dim + 1:],
        dtype=tensor.dtype, device=tensor.device
    )

    dist.reduce_scatter_tensor(output, tensor, group=cp_group)
    return output


def create_cp_causal_mask(
    local_seq_len_q: int,
    full_seq_len_kv: int,
    cp_rank: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Create causal mask for context parallelism.

    In CP mode, each rank processes a different chunk of the query sequence,
    so the causal mask needs to account for global positions.

    Args:
        local_seq_len_q: Local query sequence length (per rank).
        full_seq_len_kv: Full key/value sequence length (after all-gather).
        cp_rank: Current rank in CP group.
        device: Device to create mask on.
        dtype: Data type for mask.

    Returns:
        Causal mask tensor of shape [local_seq_len_q, full_seq_len_kv].
    """
    # Calculate global query position offset
    q_start = cp_rank * local_seq_len_q

    # Create position indices
    q_indices = torch.arange(local_seq_len_q, device=device, dtype=torch.long).unsqueeze(1) + q_start
    kv_indices = torch.arange(full_seq_len_kv, device=device, dtype=torch.long).unsqueeze(0)

    # Create causal mask: mask out positions where kv_idx > q_idx
    causal_mask = torch.zeros(local_seq_len_q, full_seq_len_kv, dtype=dtype, device=device)
    causal_mask.masked_fill_(kv_indices > q_indices, float('-inf'))

    return causal_mask


def create_cp_window_mask(
    local_seq_len_q: int,
    full_seq_len_kv: int,
    cp_rank: int,
    window_size: Tuple[int, int],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Create sliding window mask for context parallelism.

    Args:
        local_seq_len_q: Local query sequence length (per rank).
        full_seq_len_kv: Full key/value sequence length (after all-gather).
        cp_rank: Current rank in CP group.
        window_size: Tuple of (left_window, right_window). -1 means no limit.
        device: Device to create mask on.
        dtype: Data type for mask.

    Returns:
        Window mask tensor of shape [local_seq_len_q, full_seq_len_kv].
    """
    left_window, right_window = window_size

    # Calculate global query position offset
    q_start = cp_rank * local_seq_len_q

    # Create position indices
    q_indices = torch.arange(local_seq_len_q, device=device, dtype=torch.long).unsqueeze(1) + q_start
    kv_indices = torch.arange(full_seq_len_kv, device=device, dtype=torch.long).unsqueeze(0)

    # Create window mask
    window_mask = torch.zeros(local_seq_len_q, full_seq_len_kv, dtype=dtype, device=device)

    if left_window >= 0:
        window_mask.masked_fill_(kv_indices < q_indices - left_window, float('-inf'))
    if right_window >= 0:
        window_mask.masked_fill_(kv_indices > q_indices + right_window, float('-inf'))

    return window_mask


def get_cp_info(cp_group: Any) -> Tuple[int, int, bool]:
    """Get context parallelism information from process group.

    Args:
        cp_group: Context parallelism process group.

    Returns:
        Tuple of (cp_size, cp_rank, use_cp).
    """
    if cp_group is None:
        return 1, 0, False

    cp_size = dist.get_world_size(cp_group)
    cp_rank = dist.get_rank(cp_group)
    use_cp = cp_size > 1

    return cp_size, cp_rank, use_cp
