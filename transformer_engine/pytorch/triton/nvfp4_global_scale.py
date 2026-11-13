# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""PyTorch wrapper for the fused NVFP4 global-scale Triton kernel."""

from typing import Optional, Tuple, Union

import torch
import triton

from transformer_engine.common.triton.nvfp4_global_scale import (
    NVFP4_GLOBAL_SCALE_PAIR,
    NVFP4_GLOBAL_SCALE_PRODUCT,
    NVFP4_GLOBAL_SCALE_SQRT_PRODUCT,
    _nvfp4_global_scale_pair_kernel,
)

# One program covers every group; larger group counts take the eager path.
_MAX_GROUPS = 4096

_MODES = {
    None: NVFP4_GLOBAL_SCALE_PAIR,
    "product": NVFP4_GLOBAL_SCALE_PRODUCT,
    "sqrt_product": NVFP4_GLOBAL_SCALE_SQRT_PRODUCT,
}


def nvfp4_global_scale_pair(
    amax_a: torch.Tensor,
    amax_b: torch.Tensor,
    denom_a: float,
    denom_b: float,
    combine: Optional[str] = None,
) -> Optional[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]]:
    """Per-group NVFP4 global scales ``amax / denom`` of two operands in one launch.

    Equivalent, bit for bit, to ``amax_a.float() / denom_a`` and ``amax_b.float() / denom_b``
    followed by the optional combination. ``amax_a`` and ``amax_b`` hold one FP32 amax per
    group; with ``combine``, either may instead hold a single amax shared by all groups.

    Args:
        amax_a, amax_b: per-group (or single) FP32 amaxes on the same CUDA device.
        denom_a, denom_b: the operands' ``6 * scale_max`` denominators.
        combine: ``None`` returns both scales, ``"product"`` their product and
            ``"sqrt_product"`` the square root of the product.

    Returns:
        The scale tensor(s), or ``None`` when the inputs are outside what the kernel
        handles (non-FP32 amaxes, mismatched group counts, a broadcast amax without
        ``combine``, more than 4096 groups); the caller then computes the scales eagerly.
    """
    if amax_a.dtype != torch.float32 or amax_b.dtype != torch.float32:
        return None
    if amax_a.device != amax_b.device or not amax_a.is_cuda:
        return None
    num_a, num_b = amax_a.numel(), amax_b.numel()
    n = max(num_a, num_b)
    if n == 0 or n > _MAX_GROUPS or num_a not in (1, n) or num_b not in (1, n):
        return None
    mode = _MODES[combine]
    if mode == NVFP4_GLOBAL_SCALE_PAIR and num_a != num_b:
        # Keep each scale's shape: only the combined modes broadcast.
        return None
    out_a = torch.empty(n, dtype=torch.float32, device=amax_a.device)
    out_b = torch.empty_like(out_a) if mode == NVFP4_GLOBAL_SCALE_PAIR else out_a
    _nvfp4_global_scale_pair_kernel[(1,)](
        amax_a.contiguous(),
        amax_b.contiguous(),
        out_a,
        out_b,
        n,
        num_a == 1 and n > 1,
        num_b == 1 and n > 1,
        float(denom_a),
        float(denom_b),
        MODE=mode,
        BLOCK=max(16, triton.next_power_of_2(n)),
    )
    if mode == NVFP4_GLOBAL_SCALE_PAIR:
        return out_a, out_b
    return out_a
