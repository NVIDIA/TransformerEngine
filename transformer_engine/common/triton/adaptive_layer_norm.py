# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Triton kernels for adaptive layer normalization."""

import triton
import triton.language as tl


@triton.jit
def _adaptive_layernorm_fwd_kernel(
    x_ptr,
    scale_ptr,
    shift_ptr,
    y_ptr,
    mean_ptr,
    rstd_ptr,
    BATCH_SIZE: tl.constexpr,
    BATCH_STRIDE: tl.constexpr,
    HIDDEN_SIZE: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Normalize one token and apply its sample's scale and shift."""
    # Large diffusion activations can exceed 2**31 elements.
    row = tl.program_id(0).to(tl.int64)
    sample = (row // BATCH_STRIDE) % BATCH_SIZE
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < HIDDEN_SIZE
    x = tl.load(x_ptr + row * HIDDEN_SIZE + cols, mask, other=0).to(tl.float32)
    mean = tl.sum(x, axis=0) / HIDDEN_SIZE
    centered = tl.where(mask, x - mean, 0.0)
    variance = tl.sum(centered * centered, axis=0) / HIDDEN_SIZE
    rstd = tl.rsqrt(variance + EPS)
    scale = tl.load(scale_ptr + sample * HIDDEN_SIZE + cols, mask, other=0).to(tl.float32)
    shift = tl.load(shift_ptr + sample * HIDDEN_SIZE + cols, mask, other=0).to(tl.float32)
    output = centered * rstd * (1.0 + scale) + shift
    tl.store(y_ptr + row * HIDDEN_SIZE + cols, output, mask)
    tl.store(mean_ptr + row, mean)
    tl.store(rstd_ptr + row, rstd)


@triton.jit
def _adaptive_layernorm_dx_kernel(
    dy_ptr,
    x_ptr,
    scale_ptr,
    mean_ptr,
    rstd_ptr,
    dx_ptr,
    BATCH_SIZE: tl.constexpr,
    BATCH_STRIDE: tl.constexpr,
    HIDDEN_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Compute the input gradient independently for each token."""
    # Large diffusion activations can exceed 2**31 elements.
    row = tl.program_id(0).to(tl.int64)
    sample = (row // BATCH_STRIDE) % BATCH_SIZE
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < HIDDEN_SIZE
    x = tl.load(x_ptr + row * HIDDEN_SIZE + cols, mask, other=0).to(tl.float32)
    dy = tl.load(dy_ptr + row * HIDDEN_SIZE + cols, mask, other=0).to(tl.float32)
    scale = tl.load(scale_ptr + sample * HIDDEN_SIZE + cols, mask, other=0).to(tl.float32)
    mean = tl.load(mean_ptr + row)
    rstd = tl.load(rstd_ptr + row)
    normalized = tl.where(mask, (x - mean) * rstd, 0.0)
    grad_normalized = dy * (1.0 + scale)
    grad_mean = tl.sum(grad_normalized, axis=0) / HIDDEN_SIZE
    grad_projection = tl.sum(grad_normalized * normalized, axis=0) / HIDDEN_SIZE
    dx = (grad_normalized - grad_mean - normalized * grad_projection) * rstd
    tl.store(dx_ptr + row * HIDDEN_SIZE + cols, dx, mask)


@triton.jit
def _adaptive_layernorm_condition_grads_kernel(
    dy_ptr,
    x_ptr,
    mean_ptr,
    rstd_ptr,
    dscale_ptr,
    dshift_ptr,
    SEQUENCE_LENGTH: tl.constexpr,
    BATCH_SIZE: tl.constexpr,
    BATCH_STRIDE: tl.constexpr,
    HIDDEN_SIZE: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    ROWS_PER_SPLIT: tl.constexpr,
    COMPUTE_DSCALE: tl.constexpr,
    COMPUTE_DSHIFT: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_COLS: tl.constexpr,
):
    """Compute disjoint sequence partials without atomic additions."""
    sample = tl.program_id(0).to(tl.int64)
    split = tl.program_id(1)
    col_block = tl.program_id(2)
    cols = col_block * BLOCK_COLS + tl.arange(0, BLOCK_COLS)
    col_mask = cols < HIDDEN_SIZE
    row_offsets = tl.arange(0, BLOCK_ROWS)
    scale_acc = tl.zeros((BLOCK_COLS,), dtype=tl.float32)
    shift_acc = tl.zeros((BLOCK_COLS,), dtype=tl.float32)
    split_start = split.to(tl.int64) * ROWS_PER_SPLIT
    split_end = tl.minimum(split_start + ROWS_PER_SPLIT, SEQUENCE_LENGTH)
    for row_start in range(0, tl.cdiv(ROWS_PER_SPLIT, BLOCK_ROWS)):
        rows = split_start + row_start * BLOCK_ROWS + row_offsets
        row_mask = rows < split_end
        mask = row_mask[:, None] & col_mask[None, :]
        input_rows = (
            (rows // BATCH_STRIDE) * (BATCH_SIZE * BATCH_STRIDE)
            + sample * BATCH_STRIDE
            + rows % BATCH_STRIDE
        )
        offsets = input_rows[:, None] * HIDDEN_SIZE + cols[None, :]
        dy = tl.load(dy_ptr + offsets, mask, other=0).to(tl.float32)
        if COMPUTE_DSCALE:
            x = tl.load(x_ptr + offsets, mask, other=0).to(tl.float32)
            mean = tl.load(mean_ptr + input_rows, row_mask, other=0)
            rstd = tl.load(rstd_ptr + input_rows, row_mask, other=0)
            normalized = (x - mean[:, None]) * rstd[:, None]
            scale_acc += tl.sum(dy * normalized, axis=0)
        if COMPUTE_DSHIFT:
            shift_acc += tl.sum(dy, axis=0)
    offsets = (sample * NUM_SPLITS + split) * HIDDEN_SIZE + cols
    if COMPUTE_DSCALE:
        tl.store(dscale_ptr + offsets, scale_acc, col_mask)
    if COMPUTE_DSHIFT:
        tl.store(dshift_ptr + offsets, shift_acc, col_mask)


@triton.jit
def _adaptive_layernorm_reduce_condition_grads_kernel(
    partial_dscale_ptr,
    partial_dshift_ptr,
    dscale_ptr,
    dshift_ptr,
    HIDDEN_SIZE: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    COMPUTE_DSCALE: tl.constexpr,
    COMPUTE_DSHIFT: tl.constexpr,
    BLOCK_SPLITS: tl.constexpr,
    BLOCK_COLS: tl.constexpr,
):
    """Combine sequence partials in a fixed reduction order."""
    sample = tl.program_id(0).to(tl.int64)
    cols = tl.program_id(1) * BLOCK_COLS + tl.arange(0, BLOCK_COLS)
    splits = tl.arange(0, BLOCK_SPLITS)
    mask = (splits[:, None] < NUM_SPLITS) & (cols[None, :] < HIDDEN_SIZE)
    offsets = (sample * NUM_SPLITS + splits[:, None]) * HIDDEN_SIZE + cols[None, :]
    output_offsets = sample * HIDDEN_SIZE + cols
    if COMPUTE_DSCALE:
        partial = tl.load(partial_dscale_ptr + offsets, mask, other=0)
        tl.store(dscale_ptr + output_offsets, tl.sum(partial, axis=0), cols < HIDDEN_SIZE)
    if COMPUTE_DSHIFT:
        partial = tl.load(partial_dshift_ptr + offsets, mask, other=0)
        tl.store(dshift_ptr + output_offsets, tl.sum(partial, axis=0), cols < HIDDEN_SIZE)
