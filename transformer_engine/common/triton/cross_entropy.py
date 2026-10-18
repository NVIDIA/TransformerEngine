# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Efficient Cross Entropy kernels written with OpenAI Triton."""

import triton
import triton.language as tl


@triton.jit
def _is_nan(x):
    """NaN test that works on Triton tensors and scalars: NaN is the only value unequal to itself."""
    return x != x  # pylint: disable=comparison-with-itself


@triton.jit
def _fold_block_argmax(x, i, offsets, n_cols, run_max, run_idx, run_nan):
    """Fold one block of a row into a running argmax with torch.argmax's semantics: NaN counts
    as the maximum, and the first maximum (or the first NaN) wins.

    tl.max(..., return_indices=True) is not NaN-safe (its comparisons are false for NaN, so a
    block with a NaN can give a pair that is neither the maximum nor the NaN). NaN is therefore
    found separately and the maximum is taken over the other values. run_max is kept apart from
    the loss's running maximum, which is unchanged."""
    is_nan = _is_nan(x)
    block_has_nan = tl.max(is_nan.to(tl.int32)) > 0
    first_nan = tl.min(tl.where(is_nan, offsets, n_cols))
    block_max, block_idx = tl.max(
        tl.where(is_nan, float("-inf"), x),
        axis=0,
        return_indices=True,
        return_indices_tie_break_left=True,
    )
    if run_nan == 0:
        if block_has_nan:
            run_idx = first_nan
            run_nan = 1
        elif block_max > run_max:
            # Only a strictly larger maximum replaces an earlier one: first index wins.
            run_idx = i + block_idx
            run_max = block_max
    return run_max, run_idx, run_nan


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
    argmax_ptr,
    n_non_ignore,
    n_cols,
    n_rows_1,
    ignore_idx,
    label_smoothing: tl.constexpr,
    COUNT_NON_IGNORE: tl.constexpr,
    COPY_INPUT: tl.constexpr,
    COMPUTE_ARGMAX: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Compute single-rank loss/statistics and optionally preserve the input and find the
    argmax of each row."""

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
    argmax = 0
    argmax_max = float("-inf")
    argmax_nan = 0
    for i in range(0, n_cols, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols
        x = tl.load(X_ptr + offsets * X_stride_2, mask=mask, other=float("-inf"))
        if COPY_INPUT:
            tl.store(saved_input_ptr + offsets, x, mask=mask)
        x = x.to(tl.float32)
        if COMPUTE_ARGMAX:
            argmax_max, argmax, argmax_nan = _fold_block_argmax(
                x, i, offsets, n_cols, argmax_max, argmax, argmax_nan
            )
        block_max = tl.max(x)
        m_new = tl.maximum(m, block_max)
        d = d * tl.exp(m - m_new) + tl.sum(tl.exp(x - m_new))
        m = m_new
        if label_smoothing > 0:
            x_sum += tl.sum(tl.where(mask, x, 0.0))

    tl.store(stats_ptr + row * 2, m)
    tl.store(stats_ptr + row * 2 + 1, d)
    if COMPUTE_ARGMAX:
        tl.store(argmax_ptr + row, argmax.to(tl.int64))

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
    COMPUTE_X_SUM: tl.constexpr,
    COPY_INPUT: tl.constexpr,
    COMPUTE_ARGMAX: tl.constexpr,
    LOCAL_DATA_SIZE: tl.constexpr,
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
    argmax = 0
    argmax_max = float("-inf")
    argmax_nan = 0
    for i in range(0, n_cols, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols
        x = tl.load(X_ptr + offsets * X_stride_2, mask=mask, other=float("-inf"))
        if COPY_INPUT:
            tl.store(saved_input_ptr + offsets, x, mask=mask)
        x = x.to(tl.float32)
        if COMPUTE_ARGMAX:
            argmax_max, argmax, argmax_nan = _fold_block_argmax(
                x, i, offsets, n_cols, argmax_max, argmax, argmax_nan
            )
        block_max = tl.max(x)
        m_new = tl.maximum(m, block_max)
        d = d * tl.exp(m - m_new) + tl.sum(tl.exp(x - m_new))
        m = m_new
        if COMPUTE_X_SUM:
            x_sum += tl.sum(tl.where(mask, x, 0.0))

    local_data_ptr += row * LOCAL_DATA_SIZE
    tl.store(local_data_ptr, m)
    tl.store(local_data_ptr + 1, d)
    tl.store(local_data_ptr + 2, x_y)
    if COMPUTE_X_SUM:
        tl.store(local_data_ptr + 3, x_sum)
    if COMPUTE_ARGMAX:
        # The local index travels with the FP32 statistics as raw bits, followed by the local
        # maximum it belongs to, NaN if the shard holds a NaN.
        tl.store(local_data_ptr + 4, argmax.to(tl.float32, bitcast=True))
        tl.store(local_data_ptr + 5, tl.where(argmax_nan != 0, float("nan"), argmax_max))


@triton.jit
def cross_entropy_tp_post_kernel(
    gathered_data_ptr,
    Y_ptr,
    loss_ptr,
    stats_ptr,
    argmax_ptr,
    world_size,
    n_rows,
    n_cols,
    ignore_idx,
    label_smoothing: tl.constexpr,
    COMPUTE_ARGMAX: tl.constexpr,
    LOCAL_DATA_SIZE: tl.constexpr,
):
    """Combine tensor-parallel statistics and compute loss/global statistics."""

    row = tl.program_id(0).to(tl.int64)
    data_ptr = gathered_data_ptr + row * LOCAL_DATA_SIZE
    m = tl.load(data_ptr)
    d = tl.load(data_ptr + 1)
    x_y = tl.load(data_ptr + 2)
    x_sum = 0.0
    if label_smoothing > 0:
        x_sum = tl.load(data_ptr + 3)
    argmax = 0
    argmax_max = 0.0
    if COMPUTE_ARGMAX:
        argmax = tl.load(data_ptr + 4).to(tl.int32, bitcast=True).to(tl.int64)
        argmax_max = tl.load(data_ptr + 5)

    for rank_idx in range(1, world_size):
        rank_data_ptr = data_ptr + rank_idx * n_rows * LOCAL_DATA_SIZE
        m_new = tl.load(rank_data_ptr)
        d_new = tl.load(rank_data_ptr + 1)
        if COMPUTE_ARGMAX:
            # Ranks hold consecutive vocabulary shards, so the first rank with the largest
            # maximum (or with a NaN, which counts as the maximum) holds the global argmax.
            rank_argmax_max = tl.load(rank_data_ptr + 5)
            if not _is_nan(argmax_max):
                if _is_nan(rank_argmax_max) or (rank_argmax_max > argmax_max):
                    local_argmax = tl.load(rank_data_ptr + 4).to(tl.int32, bitcast=True)
                    argmax = rank_idx.to(tl.int64) * n_cols + local_argmax.to(tl.int64)
                    argmax_max = rank_argmax_max
        global_m = tl.maximum(m, m_new)
        d = d * tl.exp(m - global_m) + d_new * tl.exp(m_new - global_m)
        m = global_m
        x_y = tl.maximum(x_y, tl.load(rank_data_ptr + 2))
        if label_smoothing > 0:
            x_sum += tl.load(rank_data_ptr + 3)

    tl.store(stats_ptr + row * 2, m)
    tl.store(stats_ptr + row * 2 + 1, d)
    if COMPUTE_ARGMAX:
        tl.store(argmax_ptr + row, argmax)

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
