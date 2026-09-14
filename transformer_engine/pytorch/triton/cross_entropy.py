# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""PyTorch wrapper functions for Cross Entropy Triton kernels."""

from typing import Optional, Union
import torch
import torch.distributed as dist

import triton

from transformer_engine.common.triton.cross_entropy import (
    cross_entropy_forward_kernel,
    cross_entropy_tp_pre_kernel,
    cross_entropy_tp_post_kernel,
    cross_entropy_backward_kernel,
)

# The optimal maximum block size depends on your hardware, your kernel, and your dtype
MAX_FUSED_SIZE = 65536 // 2


def cross_entropy_forward(
    _input: torch.Tensor,
    target: torch.Tensor,
    label_smoothing: float,
    reduce_loss: bool,
    dist_process_group: Union[dist.ProcessGroup, None],
    ignore_idx: int,
    overwrite_input: bool,
):
    """Forward implementation that saves the input and compact softmax statistics."""

    B, SQ, V = _input.shape
    n_rows = B * SQ
    BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(V))

    target = target.contiguous()
    saved_input = (
        _input
        if overwrite_input
        else torch.empty(_input.shape, dtype=_input.dtype, device=_input.device)
    )
    loss_1d = torch.empty(n_rows, dtype=torch.float32, device=_input.device)
    stats = torch.empty((n_rows, 2), dtype=torch.float32, device=_input.device)
    n_non_ignore = torch.zeros(1, dtype=torch.int64, device=_input.device) if reduce_loss else None

    rank = 0 if dist_process_group is None else dist.get_rank(dist_process_group)
    world_size = 1 if dist_process_group is None else dist.get_world_size(dist_process_group)

    if world_size == 1:
        cross_entropy_forward_kernel[(n_rows,)](
            X_ptr=_input,
            X_stride_0=_input.stride(0),
            X_stride_1=_input.stride(1),
            X_stride_2=_input.stride(2),
            saved_input_ptr=saved_input,
            Y_ptr=target,
            loss_ptr=loss_1d,
            stats_ptr=stats,
            n_non_ignore=n_non_ignore if n_non_ignore is not None else stats,
            n_cols=V,
            n_rows_1=SQ,
            ignore_idx=ignore_idx,
            label_smoothing=label_smoothing,
            COUNT_NON_IGNORE=reduce_loss,
            COPY_INPUT=not overwrite_input,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=32,
        )
    else:
        local_data = torch.empty((n_rows, 4), dtype=torch.float32, device=_input.device)
        cross_entropy_tp_pre_kernel[(n_rows,)](
            X_ptr=_input,
            X_stride_0=_input.stride(0),
            X_stride_1=_input.stride(1),
            X_stride_2=_input.stride(2),
            saved_input_ptr=saved_input,
            Y_ptr=target,
            local_data_ptr=local_data,
            n_non_ignore=n_non_ignore if n_non_ignore is not None else stats,
            rank=rank,
            n_cols=V,
            n_rows_1=SQ,
            ignore_idx=ignore_idx,
            COUNT_NON_IGNORE=reduce_loss,
            COMPUTE_X_SUM=label_smoothing > 0,
            COPY_INPUT=not overwrite_input,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=32,
        )
        gathered_data = torch.empty(
            (world_size * n_rows, 4), dtype=torch.float32, device=_input.device
        )
        dist.all_gather_into_tensor(gathered_data, local_data, group=dist_process_group)
        cross_entropy_tp_post_kernel[(n_rows,)](
            gathered_data_ptr=gathered_data,
            Y_ptr=target,
            loss_ptr=loss_1d,
            stats_ptr=stats,
            world_size=world_size,
            n_rows=n_rows,
            n_cols=V,
            ignore_idx=ignore_idx,
            label_smoothing=label_smoothing,
            num_warps=1,
        )

    loss = loss_1d.reshape(B, SQ)
    if reduce_loss:
        n_non_ignore.clamp_(min=1)
        loss = loss_1d.sum() / n_non_ignore

    return loss, saved_input, stats, target, n_non_ignore, rank, world_size


def cross_entropy_backward(
    saved_input: torch.Tensor,
    stats: torch.Tensor,
    target: torch.Tensor,
    n_non_ignore: Optional[torch.Tensor],
    grad_output: torch.Tensor,
    label_smoothing: float,
    reduce_loss: bool,
    rank: int,
    world_size: int,
    ignore_idx: int,
):
    """Reconstruct the derivative in FP32 and overwrite the saved input buffer."""

    B, SQ, V = saved_input.shape
    n_rows = B * SQ
    BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(V))
    grad_output = grad_output.contiguous()

    cross_entropy_backward_kernel[(n_rows,)](
        saved_input_ptr=saved_input,
        Y_ptr=target,
        stats_ptr=stats,
        n_non_ignore_ptr=n_non_ignore if n_non_ignore is not None else stats,
        grad_output_ptr=grad_output,
        grad_output_stride=0 if reduce_loss else 1,
        rank=rank,
        world_size=world_size,
        n_cols=V,
        ignore_idx=ignore_idx,
        reduce_loss=reduce_loss,
        label_smoothing=label_smoothing,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=32,
    )
    return saved_input
