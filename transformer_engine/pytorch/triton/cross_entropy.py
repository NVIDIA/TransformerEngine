# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Efficient Cross Entropy kernels written with OpenAI Triton."""

from typing import Union
from functools import reduce
from operator import mul

import torch
import torch.distributed as dist

import triton
import triton.language as tl


@triton.jit
def online_softmax_kernel(
    X_ptr,
    X_stride,
    Y_ptr,
    Y_stride,
    m_d_X_y_ptr,
    m_d_X_y_stride,
    rank,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    """
    This kernel computes the m/d components on this TP rank for the online softmax.

    Parameters:
    X_ptr: Pointer to input tensor.
    X_stride (int): The stride of the input tensor.
    Y_ptr: Pointer to target tensor.
    Y_stride (int): The stride of the target tensor.
    m_d_X_y_ptr: Pointer to m/d/X_y tensor.
    m_d_X_y_stride (int): The stride of the m/d/X_y tensor.
    rank (int): The rank of this device in the TP group.
    n_cols (int): The number of columns in the input tensor.
    BLOCK_SIZE (int): The block size for Triton operations.
    """

    program_id = tl.program_id(0).to(tl.int64)

    # locate the start index
    X_ptr += program_id * X_stride

    # Load Y_ptr
    Y_ptr += program_id * Y_stride
    y = tl.load(Y_ptr)

    vocab_start_idx = rank * n_cols
    vocab_end_idx = (rank + 1) * n_cols
    if y >= vocab_start_idx:
        if y < vocab_end_idx:
            X_y = tl.load(X_ptr + y - vocab_start_idx).to(tl.float32)
        else:
            X_y = float("-inf")
    else:
        X_y = float("-inf")

    m_d_X_y_ptr += program_id * m_d_X_y_stride * 3

    # 3. [Online softmax] first pass: find max + sum
    m = float("-inf")  # m is the max value. use the notation from the paper
    d = 0.0  # d is the sum. use the notation from the paper

    for i in range(0, n_cols, BLOCK_SIZE):
        X_offsets = i + tl.arange(0, BLOCK_SIZE)
        X_block = tl.load(X_ptr + X_offsets, mask=X_offsets < n_cols, other=float("-inf")).to(
            tl.float32
        )
        block_max = tl.max(X_block)
        m_new = tl.maximum(m, block_max)
        d = d * tl.exp(m - m_new) + tl.sum(tl.exp(X_block - m_new))
        m = m_new

    tl.store(m_d_X_y_ptr, m)
    tl.store(m_d_X_y_ptr + m_d_X_y_stride, d)
    tl.store(m_d_X_y_ptr + (2 * m_d_X_y_stride), X_y)


@triton.jit
def cross_entropy_kernel(
    X_ptr,
    X_stride,
    Y_ptr,
    Y_stride,
    loss_ptr,
    loss_stride,
    m_d_X_y_ptr,
    m_d_X_y_stride,
    rank,
    world_size,
    ignore_idx,
    n_cols,
    n_non_ignore,
    reduce_loss: tl.constexpr,
    label_smoothing: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    This kernel computes both cross entropy loss and the gradient of the input.

    Parameters:
    X_ptr: Pointer to input tensor.
    X_stride (int): The stride of the input tensor.
    Y_ptr: Pointer to target tensor.
    Y_stride (int): The stride of the target tensor.
    loss_ptr: Pointer to tensor to store the loss.
    loss_stride (int): The stride of the loss tensor.
    m_d_X_y_ptr: Pointer to m/d/X_y tensor.
    m_d_X_y_stride: The stride of m/d/X_y tensor.
    rank (int): The rank of this device in the TP group.
    world_size (int): The size of world involved in this distributed loss calculation.
    ignore_idx (int): Tokens to be ignored for loss and gradient calculation.
    n_cols (int): The number of columns in the input tensor.
    n_non_ignore (int): The number of non-ignored elements in the batch.
    label_smoothing (float): The amount of smoothing when computing the loss, where 0.0 means no smoothing.
    BLOCK_SIZE (int): The block size for Triton operations.
    """

    program_id = tl.program_id(0).to(tl.int64)

    # locate the start index
    X_ptr += program_id * X_stride

    # Load Y_ptr
    Y_ptr += program_id * Y_stride
    y = tl.load(Y_ptr)

    if y == ignore_idx:
        # set all X_ptr as 0
        for i in range(0, n_cols, BLOCK_SIZE):
            X_offsets = i + tl.arange(0, BLOCK_SIZE)
            tl.store(X_ptr + X_offsets, 0.0, mask=X_offsets < n_cols)
        return

    loss_ptr += program_id * loss_stride
    m_d_X_y_ptr += program_id * 3 * m_d_X_y_stride

    # Need to reduce the m/d/X_y values from other TP ranks
    m = tl.load(m_d_X_y_ptr)
    d = tl.load(m_d_X_y_ptr + m_d_X_y_stride)
    ori_X_y = tl.load(m_d_X_y_ptr + (2 * m_d_X_y_stride))

    for i in range(1, world_size):
        offset = i * 3 * n_non_ignore * m_d_X_y_stride
        access_ptr = m_d_X_y_ptr + offset
        m_new = tl.load(access_ptr)
        d_new = tl.load(access_ptr + m_d_X_y_stride)
        X_y_new = tl.load(access_ptr + (2 * m_d_X_y_stride))

        d = d * tl.exp(m - tl.maximum(m, m_new)) + d_new * tl.exp(m_new - tl.maximum(m, m_new))
        m = tl.maximum(m, m_new)
        ori_X_y = tl.maximum(ori_X_y, X_y_new)

    # Label smoothing is a general case of normal cross entropy
    scaled_x_sum = 0.0
    eps = label_smoothing / (n_cols * world_size)

    # 4. [Online softmax] second pass: calculate the gradients
    # dx_y = (softmax(x_y) - 1) / N
    # dx_i = softmax(x_i) / N, i != y
    # N is the number of non ignored elements in the batch
    # For label smoothing:
    # dx_i = (softmax(x_y) - label_smoothing / V) / N, V = n_cols, i != y
    # dx_y = (softmax(x_y) - label_smoothing / V - (1 - label_smoothing)) / N
    #      = dx_i - (1 - label_smoothing) / N
    for i in range(0, n_cols, BLOCK_SIZE):
        X_offsets = i + tl.arange(0, BLOCK_SIZE)
        X_block = tl.load(X_ptr + X_offsets, mask=X_offsets < n_cols, other=float("-inf"))
        grad_dtype = X_block.dtype
        X_block = X_block.to(tl.float32)
        if label_smoothing > 0:
            # scale X beforehand to avoid overflow
            scaled_x_sum += tl.sum(tl.where(X_offsets < n_cols, -eps * X_block, 0.0))
        # Scale gradients based on reduction mode
        # For reduce_loss=True: PyTorch will scale by 1/n_rows, so we need to scale by n_rows/n_non_ignore
        # For reduce_loss=False: No additional scaling from PyTorch, so we don't scale here
        if reduce_loss:
            X_block = (tl.exp(X_block - m) / d - eps) / (n_non_ignore)
        else:
            X_block = tl.exp(X_block - m) / d - eps
        tl.store(X_ptr + X_offsets, X_block.to(grad_dtype), mask=X_offsets < n_cols)

    # We need tl.debug_barrier() to ensure the new result of X_ptr is written
    tl.debug_barrier()

    # 5. Calculate the loss

    # loss = log (softmax(X_y)) = log ((e ^ (X_y - max(X)) / sum(e ^ (X - max(X))))
    #      = (X_y - max(X)) - log(sum(e ^ (X - max(X))))
    loss = -(ori_X_y - m - tl.log(d))

    # Orginal loss = H(q, p),  with label smoothing regularization = H(q', p) and (label_smoothing / V) = eps
    # H(q', p) = (1 - label_smoothing) * H(q, p) + label_smoothing * H(u, p)
    #          = (1 - label_smoothing) * H(q, p) + eps * sum(logsoftmax(x_i))
    # By using m (global max of xi) and d (sum of e^(xi-m)), we can simplify as:
    #          = (1 - label_smoothing) * H(q, p) + (-sum(x_i * eps) + label_smoothing * (m + logd))
    # Refer to H(q', p) in section 7 of the paper: https://arxiv.org/pdf/1512.00567
    if label_smoothing > 0:
        smooth_loss = scaled_x_sum + label_smoothing * (m + tl.log(d))
        loss = loss * (1 - label_smoothing) + smooth_loss

    # 6. Specially handle the i==y case where `dx_y = (softmax(x_y) - (1 - label_smoothing) / N`
    vocab_start_idx = rank * n_cols
    vocab_end_idx = (rank + 1) * n_cols
    if y >= vocab_start_idx:
        if y < vocab_end_idx:
            X_y = tl.load(X_ptr + y - vocab_start_idx)
            # Apply the same conditional scaling logic for the target token
            if reduce_loss:
                X_y += -(1 - label_smoothing) / (n_non_ignore)
            else:
                X_y += -(1 - label_smoothing)
            tl.store(X_ptr + y - vocab_start_idx, X_y)

    tl.store(loss_ptr, loss)


# The optimal maximum block size depends on your hardware, your kernel, and your dtype
MAX_FUSED_SIZE = 65536 // 2


@triton.jit
def element_mul_kernel(
    X_ptr,
    X_stride,
    grad_output_ptr,
    grad_output_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    """
    This function multiplies each element of the tensor pointed by X_ptr with the value pointed by grad_output_ptr.
    The multiplication is performed in-place on the tensor pointed by X_ptr.

    Parameters:
    X_ptr: Pointer to the input tensor.
    X_stride (int): The stride of the input tensor.
    grad_output_ptr: Pointer to the gradient output value.
    n_cols (int): The number of columns in the input tensor.
    BLOCK_SIZE (int): The block size for Triton operations.
    """

    # Get the program ID and convert it to int64 to avoid overflow
    program_id = tl.program_id(0).to(tl.int64)

    # Locate the start index
    X_ptr += program_id * X_stride

    # Load the gradient output value
    grad_output_ptr += program_id * grad_output_stride
    grad_output = tl.load(grad_output_ptr)

    # Perform the element-wise multiplication
    for i in range(0, n_cols, BLOCK_SIZE):
        X_offsets = i + tl.arange(0, BLOCK_SIZE)
        X_block = tl.load(X_ptr + X_offsets, mask=X_offsets < n_cols)
        tl.store(X_ptr + X_offsets, X_block * grad_output, mask=X_offsets < n_cols)


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
