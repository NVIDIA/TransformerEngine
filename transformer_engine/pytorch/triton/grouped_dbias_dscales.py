# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""PyTorch wrappers for the fused grouped-dbias (+optional dscales) Triton kernel."""

import os
from typing import Optional, Tuple

import torch
import triton

from transformer_engine.common.triton.grouped_dbias_dscales import _grouped_dbias_kernel


def _is_deterministic_mode() -> bool:
    """Return True if TE is currently requesting deterministic execution."""
    return not bool(int(os.getenv("NVTE_ALLOW_NONDETERMINISTIC_ALGO", "1")))


def _launch_grouped_dbias(
    dy: torch.Tensor,
    offsets: torch.Tensor,
    dbias: torch.Tensor,
    scales: Optional[torch.Tensor],
    bias: Optional[torch.Tensor],
    dscales: Optional[torch.Tensor],
) -> None:
    """Launch the unified grouped-dbias kernel.

    If ``scales`` / ``bias`` / ``dscales`` are all None, runs the dbias-only
    specialization; otherwise runs the fused dbias+dscales specialization
    (all three must be provided together).
    """
    if _is_deterministic_mode():
        raise RuntimeError(
            "grouped_dbias Triton kernel uses non-deterministic atomic adds "
            "and cannot be used when deterministic execution is requested "
            "(NVTE_ALLOW_NONDETERMINISTIC_ALGO=0). "
            "Disable determinism or use a deterministic fallback."
        )

    BLOCK_M = 128
    BLOCK_H = 128
    N_ROW_SPLITS = 4
    has_scales = scales is not None
    assert (
        has_scales == (bias is not None) == (dscales is not None)
    ), "_launch_grouped_dbias: scales, bias and dscales must be provided together"

    hidden = dy.shape[1]
    num_groups = dbias.shape[0]

    # Triton requires real pointers; reuse dy as a harmless dummy when unused.
    scales_arg = scales if has_scales else dy
    bias_arg = bias if has_scales else dy
    dscales_arg = dscales if has_scales else dy

    grid = (num_groups, N_ROW_SPLITS, triton.cdiv(hidden, BLOCK_H))
    _grouped_dbias_kernel[grid](
        dy,
        dbias,
        offsets,
        scales_arg,
        bias_arg,
        dscales_arg,
        hidden,
        HAS_SCALES=has_scales,
        N_ROW_SPLITS=N_ROW_SPLITS,
        BLOCK_M=BLOCK_M,
        BLOCK_H=BLOCK_H,
        num_warps=4,
        num_stages=2,
    )


def compute_grouped_dbias_dscales(
    dy: torch.Tensor,
    scales: torch.Tensor,
    bias: torch.Tensor,
    offsets: torch.Tensor,
    dbias: Optional[torch.Tensor] = None,
    dscales: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute scaled grouped dbias and dscales via a single fused Triton kernel.

    For tokens i in group g(i)::

        dbias[g, j]  += sum_{i in g} dy[i, j] * scales[i]
        dscales[i]   += sum_j dy[i, j] * bias[g(i), j]

    Both outputs use fp32 atomic adds, so pre-populated tensors are
    accumulated into (useful for fusing with upstream gradients).

    Args:
        dy: (total_tokens, hidden) -- FC2 output grad.
        scales: (total_tokens,) float32 -- per-token routing scales.
        bias: (num_groups, hidden) -- per-group FC2 biases.
        offsets: (num_groups+1,) int64 -- cumulative row offsets
            ``[0, s0, s0+s1, ..., total_tokens]``.
        dbias: optional (num_groups, hidden) float32 -- accumulated into
            if provided, otherwise a fresh zero tensor is allocated.
        dscales: optional (total_tokens,) float32 -- accumulated into
            if provided, otherwise a fresh zero tensor is allocated.

    Returns:
        dbias: (num_groups, hidden) float32
        dscales: (total_tokens,) float32
    """
    num_groups = bias.shape[0]
    hidden = dy.shape[1]
    total_tokens = dy.shape[0]

    if dbias is None:
        dbias = torch.zeros(num_groups, hidden, dtype=torch.float32, device=dy.device)
    else:
        assert (
            dbias.dtype == torch.float32
        ), f"compute_grouped_dbias_dscales: dbias must be float32, got {dbias.dtype}"
    if dscales is None:
        dscales = torch.zeros(total_tokens, dtype=torch.float32, device=dy.device)
    else:
        assert (
            dscales.dtype == torch.float32
        ), f"compute_grouped_dbias_dscales: dscales must be float32, got {dscales.dtype}"

    _launch_grouped_dbias(dy, offsets, dbias, scales, bias, dscales)
    return dbias, dscales


def compute_grouped_dbias(
    dy: torch.Tensor,
    offsets: torch.Tensor,
    num_groups: int,
    dbias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute grouped dbias = per-group sum of dy via the fused Triton kernel.

    For tokens i in group g::

        dbias[g, j] += sum_{i in g} dy[i, j]

    Args:
        dy: (total_tokens, hidden) -- output grad.
        offsets: (num_groups+1,) int64 -- cumulative row offsets
            ``[0, s0, s0+s1, ..., total_tokens]``.
        num_groups: number of groups (``offsets`` has ``num_groups + 1``
            entries).
        dbias: optional (num_groups, hidden) float32 -- accumulated into
            if provided, otherwise a fresh zero tensor is allocated.

    Returns:
        dbias: (num_groups, hidden) float32
    """
    hidden = dy.shape[1]

    if dbias is None:
        dbias = torch.zeros(num_groups, hidden, dtype=torch.float32, device=dy.device)
    else:
        assert (
            dbias.dtype == torch.float32
        ), f"compute_grouped_dbias: dbias must be float32, got {dbias.dtype}"

    _launch_grouped_dbias(dy, offsets, dbias, scales=None, bias=None, dscales=None)
    return dbias
