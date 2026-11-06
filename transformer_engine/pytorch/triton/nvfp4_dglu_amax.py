# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""PyTorch wrapper for the dGLU-amax to NVFP4-amax Triton kernel."""

from typing import Tuple

import torch
import triton

from transformer_engine.common.triton.nvfp4_dglu_amax import _nvfp4_amax_from_dglu_amax_kernel

# Bound on the post-RHT amax relative to the input amax. The 16-point Hadamard transform
# is normalized by 1/4, so every output satisfies |y| <= sum(|x|) / 4 <= 4 * max(|x|).
NVFP4_RHT_AMAX_BOUND = 4.0


def nvfp4_amax_from_dglu_amax(
    dglu_amax: torch.Tensor, num_groups: int, with_rht: bool
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-group NVFP4 rowwise and columnwise amax of a dGLU output from its epilogue amax.

    Args:
        dglu_amax: FP32 tensor with ``2 * num_groups`` elements, laid out as
            ``(num_groups, 2, ...)``: the amax of each GLU half, ``-inf`` for empty groups.
            The amax is reduced over FP32 values before they are rounded to the BF16 output.
        num_groups: number of groups.
        with_rht: whether the columnwise quantization applies the random Hadamard transform.

    Returns:
        ``(rowwise_amax, columnwise_amax)``, FP32 tensors of ``num_groups`` elements. The
        rowwise amax equals the amax of the BF16 output exactly. With RHT, the columnwise
        amax is the upper bound ``NVFP4_RHT_AMAX_BOUND * rowwise_amax`` on the post-RHT
        amax instead of its exact value; without RHT it equals the rowwise amax.
    """
    if dglu_amax.numel() != 2 * num_groups:
        raise ValueError(
            f"Expected {2 * num_groups} dGLU amaxes for {num_groups} groups, "
            f"got {dglu_amax.numel()}."
        )
    if dglu_amax.dtype != torch.float32:
        raise ValueError(f"Expected an FP32 dGLU amax, got {dglu_amax.dtype}.")
    rowwise_amax = torch.empty(num_groups, dtype=torch.float32, device=dglu_amax.device)
    columnwise_amax = torch.empty_like(rowwise_amax)
    _nvfp4_amax_from_dglu_amax_kernel[(1,)](
        dglu_amax.contiguous(),
        rowwise_amax,
        columnwise_amax,
        num_groups,
        NVFP4_RHT_AMAX_BOUND if with_rht else 1.0,
        BLOCK=max(16, triton.next_power_of_2(num_groups)),
    )
    return rowwise_amax, columnwise_amax
