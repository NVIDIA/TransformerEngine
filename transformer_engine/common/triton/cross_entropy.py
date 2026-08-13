# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Efficient Cross Entropy kernels written with OpenAI Triton."""

import triton
import triton.language as tl


@triton.jit
def cross_entropy_forward_kernel(
    X_ptr,
    X_stride_0,
    X_stride_1,
    X_stride_2,
    saved_input_ptr,
    Y_ptr,
    loss_ptr,
    stats_ptr,
    n_non_ignore,
    n_cols,
    n_rows_1,
    ignore_idx,
    label_smoothing: tl.constexpr,
    COUNT_NON_IGNORE: tl.constexpr,
    COPY_INPUT: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Compute single-rank loss/statistics and optionally preserve the input."""

    row = tl.program_id(0).to(tl.int64)
    row_0 = row // n_rows_1
    row_1 = row - row_0 * n_rows_1
    X_ptr += row_0 * X_stride_0 + row_1 * X_stride_1
    saved_input_ptr += row * n_cols

    y = tl.load(Y_ptr + row)
    if COUNT_NON_IGNORE:
        if y != ignore_idx:
            tl.atomic_add(n_non_ignore, 1)

    m = float("-inf")
    d = 0.0
    x_sum = 0.0
    for i in range(0, n_cols, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols
        x = tl.load(X_ptr + offsets * X_stride_2, mask=mask, other=float("-inf"))
        if COPY_INPUT:
            tl.store(saved_input_ptr + offsets, x, mask=mask)
        x = x.to(tl.float32)
        block_max = tl.max(x)
        m_new = tl.maximum(m, block_max)
        d = d * tl.exp(m - m_new) + tl.sum(tl.exp(x - m_new))
        m = m_new
        if label_smoothing > 0:
            x_sum += tl.sum(tl.where(mask, x, 0.0))

    tl.store(stats_ptr + row * 2, m)
    tl.store(stats_ptr + row * 2 + 1, d)

    if y == ignore_idx:
        tl.store(loss_ptr + row, 0.0)
        return

    x_y = float("-inf")
    if y >= 0:
        if y < n_cols:
            x_y = tl.load(X_ptr + y * X_stride_2).to(tl.float32)

    loss = -(x_y - m - tl.log(d))
    if label_smoothing > 0:
        eps = label_smoothing / n_cols
        smooth_loss = -eps * x_sum + label_smoothing * (m + tl.log(d))
        loss = loss * (1 - label_smoothing) + smooth_loss
    tl.store(loss_ptr + row, loss)


@triton.jit
def cross_entropy_tp_pre_kernel(
    X_ptr,
    X_stride_0,
    X_stride_1,
    X_stride_2,
    saved_input_ptr,
    Y_ptr,
    local_data_ptr,
    n_non_ignore,
    rank,
    n_cols,
    n_rows_1,
    ignore_idx,
    COUNT_NON_IGNORE: tl.constexpr,
    COPY_INPUT: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Compute the local statistics needed by tensor-parallel cross entropy."""

    row = tl.program_id(0).to(tl.int64)
    row_0 = row // n_rows_1
    row_1 = row - row_0 * n_rows_1
    X_ptr += row_0 * X_stride_0 + row_1 * X_stride_1
    saved_input_ptr += row * n_cols

    y = tl.load(Y_ptr + row)
    if COUNT_NON_IGNORE:
        if y != ignore_idx:
            tl.atomic_add(n_non_ignore, 1)

    vocab_start = rank * n_cols
    x_y = float("-inf")
    if y >= vocab_start:
        if y < vocab_start + n_cols:
            x_y = tl.load(X_ptr + (y - vocab_start) * X_stride_2).to(tl.float32)

    m = float("-inf")
    d = 0.0
    x_sum = 0.0
    for i in range(0, n_cols, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols
        x = tl.load(X_ptr + offsets * X_stride_2, mask=mask, other=float("-inf"))
        if COPY_INPUT:
            tl.store(saved_input_ptr + offsets, x, mask=mask)
        x = x.to(tl.float32)
        block_max = tl.max(x)
        m_new = tl.maximum(m, block_max)
        d = d * tl.exp(m - m_new) + tl.sum(tl.exp(x - m_new))
        m = m_new
        x_sum += tl.sum(tl.where(mask, x, 0.0))

    local_data_ptr += row * 4
    tl.store(local_data_ptr, m)
    tl.store(local_data_ptr + 1, d)
    tl.store(local_data_ptr + 2, x_y)
    tl.store(local_data_ptr + 3, x_sum)


@triton.jit
def cross_entropy_tp_post_kernel(
    gathered_data_ptr,
    Y_ptr,
    loss_ptr,
    stats_ptr,
    world_size,
    n_rows,
    n_cols,
    ignore_idx,
    label_smoothing: tl.constexpr,
):
    """Combine tensor-parallel statistics and compute loss/global statistics."""

    row = tl.program_id(0).to(tl.int64)
    data_ptr = gathered_data_ptr + row * 4
    m = tl.load(data_ptr)
    d = tl.load(data_ptr + 1)
    x_y = tl.load(data_ptr + 2)
    x_sum = tl.load(data_ptr + 3)

    for rank_idx in range(1, world_size):
        rank_data_ptr = data_ptr + rank_idx * n_rows * 4
        m_new = tl.load(rank_data_ptr)
        d_new = tl.load(rank_data_ptr + 1)
        global_m = tl.maximum(m, m_new)
        d = d * tl.exp(m - global_m) + d_new * tl.exp(m_new - global_m)
        m = global_m
        x_y = tl.maximum(x_y, tl.load(rank_data_ptr + 2))
        x_sum += tl.load(rank_data_ptr + 3)

    tl.store(stats_ptr + row * 2, m)
    tl.store(stats_ptr + row * 2 + 1, d)

    y = tl.load(Y_ptr + row)
    if y == ignore_idx:
        tl.store(loss_ptr + row, 0.0)
        return

    loss = -(x_y - m - tl.log(d))
    if label_smoothing > 0:
        vocab_size = n_cols * world_size
        eps = label_smoothing / vocab_size
        smooth_loss = -eps * x_sum + label_smoothing * (m + tl.log(d))
        loss = loss * (1 - label_smoothing) + smooth_loss
    tl.store(loss_ptr + row, loss)


@triton.jit
def cross_entropy_backward_kernel(
    saved_input_ptr,
    Y_ptr,
    stats_ptr,
    n_non_ignore_ptr,
    grad_output_ptr,
    grad_output_stride,
    rank,
    world_size,
    n_cols,
    ignore_idx,
    reduce_loss: tl.constexpr,
    label_smoothing: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Reconstruct the FP32 derivative and store it in the saved input buffer."""

    row = tl.program_id(0).to(tl.int64)
    saved_input_ptr += row * n_cols
    y = tl.load(Y_ptr + row)

    if y == ignore_idx:
        for i in range(0, n_cols, BLOCK_SIZE):
            offsets = i + tl.arange(0, BLOCK_SIZE)
            tl.store(saved_input_ptr + offsets, 0.0, mask=offsets < n_cols)
        return

    m = tl.load(stats_ptr + row * 2)
    d = tl.load(stats_ptr + row * 2 + 1)
    grad_output = tl.load(grad_output_ptr + row * grad_output_stride).to(tl.float32)
    if reduce_loss:
        grad_output /= tl.load(n_non_ignore_ptr)

    eps = label_smoothing / (n_cols * world_size)
    vocab_start = rank * n_cols
    target_col = y - vocab_start
    target_is_local = (y >= vocab_start) & (y < vocab_start + n_cols)

    for i in range(0, n_cols, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols
        x = tl.load(saved_input_ptr + offsets, mask=mask, other=float("-inf")).to(tl.float32)
        grad = tl.exp(x - m) / d - eps
        is_target = target_is_local & (offsets == target_col)
        grad -= tl.where(is_target, 1 - label_smoothing, 0.0)
        tl.store(saved_input_ptr + offsets, grad * grad_output, mask=mask)
