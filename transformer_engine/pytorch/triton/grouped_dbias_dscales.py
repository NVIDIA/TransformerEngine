# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""PyTorch wrapper for the fused grouped dbias + dscales Triton kernel."""

from typing import Optional, Tuple

import torch
import triton

from transformer_engine.common.triton.grouped_dbias_dscales import (
    _grouped_dbias_dscales_kernel,
)


def _compute_grouped_dbias_dscales(
    dy: torch.Tensor,
    scales: torch.Tensor,
    bias: torch.Tensor,
    split_sizes: torch.Tensor,
    offsets: torch.Tensor,
    dbias: Optional[torch.Tensor] = None,
    dscales: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute dbias and dscales via a single fused Triton kernel.

    Computes the following, where token *i* belongs to group *g(i)*:

        dbias[g, j]  += sum_{i in group g} dy[i, j] * scales[i]
        dscales[i]   += sum_j dy[i, j] * bias[g(i), j]

    Both outputs use fp32 atomic adds, so pre-populated tensors are
    accumulated into (useful for fusing with upstream gradients).

    Args:
        dy: (total_tokens, hidden) -- FC2 output grad.
        scales: (total_tokens,) float32 -- per-token routing scales.
        bias: (num_groups, hidden) -- per-group FC2 biases.
        split_sizes: (num_groups,) int64 -- tokens per group.
        offsets: (num_groups+1,) int64 -- cumulative row offsets
            ``[0, s0, s0+s1, ..., total_tokens]``.
        dbias: optional (num_groups, hidden) float32 -- if provided,
            the kernel accumulates into this tensor; otherwise a
            zero tensor is allocated.
        dscales: optional (total_tokens,) float32 -- if provided,
            the kernel accumulates into this tensor; otherwise a
            zero tensor is allocated.

    Returns:
        dbias: (num_groups, hidden) -- same dtype as dy if freshly
            allocated, or the input tensor (fp32) if provided.
        dscales: (total_tokens,) float32
    """
    num_groups = bias.shape[0]
    hidden = dy.shape[1]
    total_tokens = dy.shape[0]

    alloc_dbias = dbias is None
    if dbias is None:
        dbias = torch.zeros(num_groups, hidden, dtype=torch.float32, device=dy.device)
    if dscales is None:
        dscales = torch.zeros(total_tokens, dtype=torch.float32, device=dy.device)

    BLOCK_M = 128
    BLOCK_H = 128
    N_ROW_SPLITS = 4

    grid = (
        num_groups,
        N_ROW_SPLITS,
        triton.cdiv(hidden, BLOCK_H),
    )

    _grouped_dbias_dscales_kernel[grid](
        dy, scales, bias,
        dbias, dscales,
        offsets,
        hidden,
        N_ROW_SPLITS=N_ROW_SPLITS,
        BLOCK_M=BLOCK_M,
        BLOCK_H=BLOCK_H,
        num_warps=4,
        num_stages=2,
    )

    if alloc_dbias:
        return dbias.to(dy.dtype), dscales
    return dbias, dscales
