# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""PyTorch launchers for adaptive layer normalization Triton kernels."""

from __future__ import annotations

import math
from typing import Optional

import torch
import triton

from transformer_engine.common.triton.adaptive_layer_norm import (
    _adaptive_layernorm_fwd_kernel,
    _adaptive_layernorm_dx_kernel,
    _adaptive_layernorm_condition_grads_kernel,
    _adaptive_layernorm_reduce_condition_grads_kernel,
)


def _check_input(x: torch.Tensor, batch_dim: int) -> tuple[int, int, int, int]:
    """Obtain batch size, tokens per sample, hidden size, and batch stride."""
    if x.device.type != "cuda":
        raise ValueError("Adaptive layer normalization requires CUDA tensors.")
    if x.dtype not in (torch.float32, torch.float16, torch.bfloat16):
        raise TypeError(f"Unsupported input dtype for adaptive layer normalization: {x.dtype}.")
    if x.ndim < 2:
        raise ValueError("Input must have shape (batch, ..., hidden_size).")
    if not 0 <= batch_dim < x.ndim - 1:
        raise ValueError("batch_dim must identify a non-normalized input dimension.")
    batch_size, hidden_size = x.shape[batch_dim], x.shape[-1]
    if not 1 <= hidden_size <= 16384:
        raise ValueError("Adaptive layer normalization supports hidden sizes from 1 to 16384.")
    sequence_length = math.prod(size for dim, size in enumerate(x.shape[:-1]) if dim != batch_dim)
    batch_stride = math.prod(x.shape[batch_dim + 1 : -1])
    return batch_size, sequence_length, hidden_size, batch_stride


def _check_condition(
    condition: torch.Tensor,
    x: torch.Tensor,
    name: str,
    batch_dim: int,
) -> None:
    """Check a per-sample condition, including an optional singleton sequence shape."""
    compact_shape = (x.shape[batch_dim], x.shape[-1])
    expanded_shape = tuple(
        size if dim in (batch_dim, x.ndim - 1) else 1 for dim, size in enumerate(x.shape)
    )
    if tuple(condition.shape) not in (compact_shape, expanded_shape):
        raise ValueError(
            f"{name} must have shape {compact_shape} or {expanded_shape}, "
            f"got {tuple(condition.shape)}."
        )
    if condition.device != x.device:
        raise ValueError(f"{name} must be on the same CUDA device as the input.")
    if condition.dtype not in (torch.float32, torch.float16, torch.bfloat16):
        raise TypeError(f"Unsupported {name} dtype: {condition.dtype}.")


def adaptive_layernorm_fwd(
    x: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    eps: float,
    *,
    batch_dim: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply adaptive layer normalization and return output, mean, and reciprocal stddev.

    The last input dimension is normalized; batch_dim identifies the batch.
    Conditions have shape (batch, hidden_size), or match the input rank with
    singleton dimensions except for batch_dim and the last dimension.
    Normalization and modulation use float32 arithmetic and the output has the
    input dtype. Mean and reciprocal stddev are flattened float32
    tensors, with one value per input row.

    These launchers do not register an autograd formula; callers must use
    adaptive_layernorm_bwd to propagate gradients.
    """
    batch_size, sequence_length, hidden_size, batch_stride = _check_input(x, batch_dim)
    _check_condition(scale, x, "scale", batch_dim)
    _check_condition(shift, x, "shift", batch_dim)
    if not math.isfinite(eps) or eps < 0:
        raise ValueError("eps must be finite and non-negative.")
    x = x.contiguous()
    scale = scale.contiguous()
    shift = shift.contiguous()
    output = torch.empty(x.shape, device=x.device, dtype=x.dtype)
    mean = torch.empty(batch_size * sequence_length, device=x.device, dtype=torch.float32)
    rstd = torch.empty_like(mean)
    if batch_size * sequence_length == 0:
        return output, mean, rstd
    block_size = triton.next_power_of_2(hidden_size)
    num_warps = min(8, max(1, block_size // 256))
    with torch.cuda.device(x.device):
        _adaptive_layernorm_fwd_kernel[(batch_size * sequence_length,)](
            x,
            scale,
            shift,
            output,
            mean,
            rstd,
            BATCH_SIZE=batch_size,
            BATCH_STRIDE=batch_stride,
            HIDDEN_SIZE=hidden_size,
            EPS=eps,
            BLOCK_SIZE=block_size,
            num_warps=num_warps,
        )
    return output, mean, rstd


def adaptive_layernorm_bwd(
    dy: torch.Tensor,
    x: torch.Tensor,
    scale: torch.Tensor,
    mean: torch.Tensor,
    rstd: torch.Tensor,
    *,
    shift_shape: tuple[int, ...],
    shift_dtype: torch.dtype,
    compute_dscale: bool = True,
    compute_dshift: bool = True,
    batch_dim: int = 0,
) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Compute input and optional condition gradients with deterministic reductions.

    batch_dim must match the forward call. shift_shape and shift_dtype describe
    the forward shift input. Its values are not needed for the backward pass.
    Each condition gradient has the shape and dtype of its corresponding input. Split sequence reductions use a
    bounded float32 workspace and never use atomic additions.
    """
    batch_size, sequence_length, hidden_size, batch_stride = _check_input(x, batch_dim)
    _check_condition(scale, x, "scale", batch_dim)
    if dy.shape != x.shape or dy.device != x.device:
        raise ValueError("Output gradient must have the input shape and device.")
    if dy.dtype not in (torch.float32, torch.float16, torch.bfloat16):
        raise TypeError(f"Unsupported output gradient dtype: {dy.dtype}.")
    expected_stats_shape = (batch_size * sequence_length,)
    for name, stats in (("mean", mean), ("rstd", rstd)):
        if (
            tuple(stats.shape) != expected_stats_shape
            or stats.dtype != torch.float32
            or stats.device != x.device
            or not stats.is_contiguous()
        ):
            raise ValueError(f"{name} must be a contiguous float32 row-statistics tensor.")
    compact_shape = (batch_size, hidden_size)
    expanded_shape = tuple(
        size if dim in (batch_dim, x.ndim - 1) else 1 for dim, size in enumerate(x.shape)
    )
    if tuple(shift_shape) not in (compact_shape, expanded_shape):
        raise ValueError("Shift gradient shape must describe a per-sample condition.")
    if shift_dtype not in (torch.float32, torch.float16, torch.bfloat16):
        raise TypeError(f"Unsupported shift gradient dtype: {shift_dtype}.")
    x = x.contiguous()
    dy = dy.contiguous()
    scale = scale.contiguous()
    dx = torch.empty(x.shape, device=x.device, dtype=x.dtype)
    dscale = (
        torch.empty(scale.shape, device=x.device, dtype=scale.dtype) if compute_dscale else None
    )
    dshift = (
        torch.empty(shift_shape, device=x.device, dtype=shift_dtype) if compute_dshift else None
    )
    if batch_size * sequence_length == 0:
        if dscale is not None:
            dscale.zero_()
        if dshift is not None:
            dshift.zero_()
        return dx, dscale, dshift

    block_size = triton.next_power_of_2(hidden_size)
    num_warps = min(8, max(1, block_size // 256))
    with torch.cuda.device(x.device):
        _adaptive_layernorm_dx_kernel[(batch_size * sequence_length,)](
            dy,
            x,
            scale,
            mean,
            rstd,
            dx,
            BATCH_SIZE=batch_size,
            BATCH_STRIDE=batch_stride,
            HIDDEN_SIZE=hidden_size,
            BLOCK_SIZE=block_size,
            num_warps=num_warps,
            # H=1 has identically zero dX. Contracting its equal terms into
            # separate FMAs can leave a residual amplified by reciprocal stddev.
            enable_fp_fusion=hidden_size != 1,
        )
        if compute_dscale or compute_dshift:
            num_splits = min(32, triton.cdiv(sequence_length, 128))
            rows_per_split = triton.cdiv(sequence_length, num_splits)
            # With one split the partial kernel stores directly in the outputs.
            # Unused pointers alias dy and are removed by constexpr specialization.
            partial_shape = (batch_size, num_splits, hidden_size)
            partial_dscale = dy
            partial_dshift = dy
            if compute_dscale:
                partial_dscale = (
                    dscale
                    if num_splits == 1
                    else torch.empty(partial_shape, device=x.device, dtype=torch.float32)
                )
            if compute_dshift:
                partial_dshift = (
                    dshift
                    if num_splits == 1
                    else torch.empty(partial_shape, device=x.device, dtype=torch.float32)
                )
            _adaptive_layernorm_condition_grads_kernel[
                (batch_size, num_splits, triton.cdiv(hidden_size, 128))
            ](
                dy,
                x,
                mean,
                rstd,
                partial_dscale,
                partial_dshift,
                SEQUENCE_LENGTH=sequence_length,
                BATCH_SIZE=batch_size,
                BATCH_STRIDE=batch_stride,
                HIDDEN_SIZE=hidden_size,
                NUM_SPLITS=num_splits,
                ROWS_PER_SPLIT=rows_per_split,
                COMPUTE_DSCALE=compute_dscale,
                COMPUTE_DSHIFT=compute_dshift,
                BLOCK_ROWS=32,
                BLOCK_COLS=128,
                num_warps=4,
            )
            if num_splits > 1:
                _adaptive_layernorm_reduce_condition_grads_kernel[
                    (batch_size, triton.cdiv(hidden_size, 128))
                ](
                    partial_dscale,
                    partial_dshift,
                    dscale if dscale is not None else dy,
                    dshift if dshift is not None else dy,
                    HIDDEN_SIZE=hidden_size,
                    NUM_SPLITS=num_splits,
                    COMPUTE_DSCALE=compute_dscale,
                    COMPUTE_DSHIFT=compute_dshift,
                    BLOCK_SPLITS=triton.next_power_of_2(num_splits),
                    BLOCK_COLS=128,
                    num_warps=4,
                )
    return dx, dscale, dshift
