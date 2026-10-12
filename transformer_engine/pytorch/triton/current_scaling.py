# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""PyTorch wrapper for the two-phase FP8 current-scaling quantization Triton kernel."""

from typing import Dict, Tuple

import torch
import triton

from transformer_engine.common.triton.current_scaling import current_scaling_quantize_kernel

_BLOCK = 8192
_NUM_WARPS = 8
_CTAS_PER_SM = 4
_FP8_MAX = {torch.float8_e4m3fn: 448.0, torch.float8_e5m2: 57344.0}

# One [amax, counter, scale, pad] workspace per (device, stream): the last CTA of every launch
# re-zeroes it, but launches on different streams must not share one.
_WORKSPACES: Dict[Tuple[int, int], torch.Tensor] = {}
_NUM_SMS: Dict[int, int] = {}


def _workspace(device: torch.device) -> torch.Tensor:
    key = (device.index, torch.cuda.current_stream(device).cuda_stream)
    ws = _WORKSPACES.get(key)
    if ws is None:
        ws = _WORKSPACES[key] = torch.zeros(4, dtype=torch.float32, device=device)
    return ws


def current_scaling_quantize(
    inp: torch.Tensor,
    out: torch.Tensor,
    scale_inv: torch.Tensor,
    *,
    eps: float,
    force_pow_2_scales: bool,
) -> None:
    """Per-tensor current-scaling FP8 quantization of a contiguous tensor.

    ``out`` is a float8_e4m3fn/e5m2 tensor of the same number of elements and ``scale_inv`` a
    1-element fp32 tensor; both are written. Bitwise equal to TE's amax + compute-scale + cast.
    """
    n = inp.numel()
    device = inp.device
    if device.index not in _NUM_SMS:
        _NUM_SMS[device.index] = torch.cuda.get_device_properties(device).multi_processor_count
    ws = _workspace(device)
    n_blk = triton.cdiv(n, _BLOCK)
    common = dict(
        BLOCK=_BLOCK,
        FP8MAX=_FP8_MAX[out.dtype],
        POW2=bool(force_pow_2_scales),
        num_warps=_NUM_WARPS,
    )
    current_scaling_quantize_kernel[(min(n_blk, _NUM_SMS[device.index] * _CTAS_PER_SM),)](
        inp, out, ws, scale_inv, n, eps, CAST=False, **common
    )
    current_scaling_quantize_kernel[(n_blk,)](inp, out, ws, scale_inv, n, eps, CAST=True, **common)
