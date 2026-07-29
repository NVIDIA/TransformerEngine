# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""PyTorch wrapper functions for Cross Entropy Triton kernels."""

from typing import Union
from functools import reduce
from operator import mul

import torch
import torch.distributed as dist

import triton

from transformer_engine.common.triton.cross_entropy import (
    online_softmax_kernel,
    cross_entropy_kernel,
    element_mul_kernel,
    cross_entropy_recompute_forward_kernel,
    cross_entropy_recompute_tp_pre_kernel,
    cross_entropy_recompute_tp_post_kernel,
    cross_entropy_recompute_backward_kernel,
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
):
    """Forward implementation of Cross Entropy kernel"""

    B, SQ, V = _input.shape
    n_rows = B * SQ

    assert reduce(mul, list(target.size())) == (B * SQ), "Each token needs a target token ID."

    BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(V))

    # unreduced loss
    loss_1d = torch.zeros(n_rows, dtype=torch.float32, device=_input.device)

    # tensor to hold this rank's m/d/X_y values
    m_d_X_y = torch.zeros(n_rows * 3, dtype=torch.float32, device=_input.device)

    n_non_ignore = torch.zeros(1, dtype=torch.int64, device=_input.device)

    # ensure _input and target are contiguous in the last dimension
    if _input.stride(-1) != 1 or _input.stride(-2) != _input.shape[-1]:
        _input = _input.contiguous()
    if target.stride(-1) != 1:
        target = target.contiguous()

    # Store the input gradient in FP32 so it is not quantized before backward.
    grad_input = torch.empty_like(_input, dtype=torch.float32)

    rank = 0 if dist_process_group is None else dist.get_rank(dist_process_group)

    online_softmax_kernel[(n_rows,)](
        X_ptr=_input,
        X_stride=_input.stride(-2),
        Y_ptr=target,
        Y_stride=target.stride(-1),  # always 1
        m_d_X_y_ptr=m_d_X_y,
        m_d_X_y_stride=m_d_X_y.stride(-1),
        rank=rank,
        n_cols=V,
        ignore_idx=ignore_idx,
        n_non_ignore=n_non_ignore,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=32,
    )

    n_non_ignore = torch.clamp(n_non_ignore, min=1)

    world_size = 1 if dist_process_group is None else dist.get_world_size(dist_process_group)

    if world_size > 1:
        m_d_X_y_gathered = torch.zeros(
            n_rows * 3 * world_size, dtype=torch.float32, device=_input.device
        )
        dist.all_gather_into_tensor(m_d_X_y_gathered, m_d_X_y, group=dist_process_group)
    else:
        m_d_X_y_gathered = m_d_X_y

    cross_entropy_kernel[(n_rows,)](
        X_ptr=_input,
        X_stride=_input.stride(-2),
        grad_input_ptr=grad_input,
        grad_input_stride=grad_input.stride(-2),
        Y_ptr=target,
        Y_stride=target.stride(-1),
        loss_ptr=loss_1d,
        loss_stride=loss_1d.stride(-1),
        m_d_X_y_ptr=m_d_X_y_gathered,
        m_d_X_y_stride=m_d_X_y_gathered.stride(-1),
        rank=rank,
        world_size=world_size,
        ignore_idx=ignore_idx,
        n_cols=V,
        n_rows=n_rows,
        n_non_ignore=n_non_ignore,
        reduce_loss=reduce_loss,
        label_smoothing=label_smoothing,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=32,
    )

    loss = (
        torch.reshape(loss_1d, (B, SQ)) if not reduce_loss else (torch.sum(loss_1d) / n_non_ignore)
    )

    return loss, grad_input


def cross_entropy_backward(
    grad_input: torch.Tensor, grad_output: torch.Tensor, is_cg_capturable: bool = False
):
    """Backward implementation of cross entropy loss kernel"""

    # If cross entropy is the last layer, grad_output is 1.0. Skip the mul to save time
    # Only check torch.equal when not in CUDA graph capturable mode
    if not is_cg_capturable and torch.equal(
        grad_output, torch.tensor(1.0, device=grad_output.device)
    ):
        pass
    else:
        B, SQ, V = grad_input.shape
        n_rows = B * SQ
        BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(V))

        element_mul_kernel[(n_rows,)](
            grad_input,
            grad_input.stride(-2),
            grad_output.contiguous(),
            1 if grad_output.numel() > 1 else 0,
            V,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=32,
        )

    return grad_input


def cross_entropy_recompute_forward(
    _input: torch.Tensor,
    target: torch.Tensor,
    label_smoothing: float,
    reduce_loss: bool,
    dist_process_group: Union[dist.ProcessGroup, None],
    ignore_idx: int,
    overwrite_input: bool,
):
    """Forward implementation that saves logits and compact softmax statistics."""

    B, SQ, V = _input.shape
    n_rows = B * SQ
    BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(V))

    target = target.contiguous().reshape(-1)
    logits = (
        _input
        if overwrite_input
        else torch.empty(_input.shape, dtype=_input.dtype, device=_input.device)
    )
    loss_1d = torch.empty(n_rows, dtype=torch.float32, device=_input.device)
    stats = torch.empty((n_rows, 2), dtype=torch.float32, device=_input.device)
    n_non_ignore = torch.zeros(1, dtype=torch.int64, device=_input.device)

    rank = 0 if dist_process_group is None else dist.get_rank(dist_process_group)
    world_size = 1 if dist_process_group is None else dist.get_world_size(dist_process_group)

    if world_size == 1:
        cross_entropy_recompute_forward_kernel[(n_rows,)](
            X_ptr=_input,
            X_stride_0=_input.stride(0),
            X_stride_1=_input.stride(1),
            X_stride_2=_input.stride(2),
            logits_ptr=logits,
            Y_ptr=target,
            loss_ptr=loss_1d,
            stats_ptr=stats,
            n_non_ignore=n_non_ignore,
            n_cols=V,
            n_rows_1=SQ,
            ignore_idx=ignore_idx,
            label_smoothing=label_smoothing,
            COPY_LOGITS=not overwrite_input,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=32,
        )
    else:
        local_data = torch.empty((n_rows, 4), dtype=torch.float32, device=_input.device)
        cross_entropy_recompute_tp_pre_kernel[(n_rows,)](
            X_ptr=_input,
            X_stride_0=_input.stride(0),
            X_stride_1=_input.stride(1),
            X_stride_2=_input.stride(2),
            logits_ptr=logits,
            Y_ptr=target,
            local_data_ptr=local_data,
            n_non_ignore=n_non_ignore,
            rank=rank,
            n_cols=V,
            n_rows_1=SQ,
            ignore_idx=ignore_idx,
            COPY_LOGITS=not overwrite_input,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=32,
        )
        gathered_data = torch.empty(
            (world_size * n_rows, 4), dtype=torch.float32, device=_input.device
        )
        dist.all_gather_into_tensor(gathered_data, local_data, group=dist_process_group)
        cross_entropy_recompute_tp_post_kernel[(n_rows,)](
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

    n_non_ignore.clamp_(min=1)
    loss = loss_1d.reshape(B, SQ)
    if reduce_loss:
        loss = loss_1d.sum() / n_non_ignore

    return loss, logits, stats, target, n_non_ignore


def cross_entropy_recompute_backward(
    logits: torch.Tensor,
    stats: torch.Tensor,
    target: torch.Tensor,
    n_non_ignore: torch.Tensor,
    grad_output: torch.Tensor,
    label_smoothing: float,
    reduce_loss: bool,
    dist_process_group: Union[dist.ProcessGroup, None],
    ignore_idx: int,
    is_cg_capturable: bool = False,
):
    """Reconstruct the derivative in FP32 and overwrite the saved logits buffer."""

    del is_cg_capturable
    B, SQ, V = logits.shape
    n_rows = B * SQ
    BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(V))
    rank = 0 if dist_process_group is None else dist.get_rank(dist_process_group)
    world_size = 1 if dist_process_group is None else dist.get_world_size(dist_process_group)
    grad_output = grad_output.contiguous()

    cross_entropy_recompute_backward_kernel[(n_rows,)](
        logits_ptr=logits,
        Y_ptr=target,
        stats_ptr=stats,
        n_non_ignore_ptr=n_non_ignore,
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
    return logits
