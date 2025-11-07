# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

    # ensure _input and target are contiguous in the last dimension
    if _input.stride(-1) != 1:
        _input = _input.contiguous()
    if target.stride(-1) != 1:
        target = target.contiguous()

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
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=32,
    )

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
        n_non_ignore=n_rows,
        reduce_loss=reduce_loss,
        label_smoothing=label_smoothing,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=32,
    )

    loss = torch.reshape(loss_1d, (B, SQ)) if not reduce_loss else (torch.sum(loss_1d) / n_rows)

    return loss, _input


def cross_entropy_backward(
    _input: torch.Tensor, grad_output: torch.Tensor, is_cg_capturable: bool = False
):
    """Backward implementation of cross entropy loss kernel"""

    # If cross entropy is the last layer, grad_output is 1.0. Skip the mul to save time
    # Only check torch.equal when not in CUDA graph capturable mode
    if not is_cg_capturable and torch.equal(
        grad_output, torch.tensor(1.0, device=grad_output.device)
    ):
        pass
    else:
        B, SQ, V = _input.shape
        n_rows = B * SQ
        BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(V))

        element_mul_kernel[(n_rows,)](
            _input,
            _input.stride(-2),
            grad_output,
            1 if grad_output.numel() > 1 else 0,
            V,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=32,
        )

    return _input
