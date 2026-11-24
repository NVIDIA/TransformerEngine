# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""PyTorch wrappers for the fused RoPE + FP8 current-scaling Q/K/V quantization kernels."""

from typing import Dict, Tuple

import torch
import triton

from transformer_engine.common.triton.rope_fp8 import (
    rope_qkv_quantize_kernel,
    rope_trig_table_kernel,
)
from .current_scaling import _FP8_MAX, _CTAS_PER_SM, _NUM_SMS, _workspace

_AMAX_WARPS = 2
_CAST_WARPS = 4
_AMAX_STAGES = 3

# (cos, sin) per device, valid for one freqs tensor object at one version. The entry holds the
# tensor, so its address cannot be reused while the entry is alive.
_TRIG_CACHE: Dict[int, Tuple[torch.Tensor, int, torch.Tensor, torch.Tensor]] = {}


def rope_trig_tables(freqs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """cos/sin (fp32, freqs' shape) of contiguous fp32 RoPE angles, bitwise equal to TE's
    sincosf. All layers of a step share one rotary embedding tensor, so this runs once per step."""
    ent = _TRIG_CACHE.get(freqs.device.index)
    if ent is not None and ent[0] is freqs and ent[1] == freqs._version:
        return ent[2], ent[3]
    cos = torch.empty_like(freqs)
    sin = torch.empty_like(freqs)
    n = freqs.numel()
    rope_trig_table_kernel[(triton.cdiv(n, 1024),)](freqs, cos, sin, n, BLOCK=1024, num_warps=4)
    _TRIG_CACHE[freqs.device.index] = (freqs, freqs._version, cos, sin)
    return cos, sin


def rope_qkv_quantize(
    mixed_qkv: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    out: torch.Tensor,
    scale_inv: torch.Tensor,
    num_q_per_group: int,
    *,
    eps: float,
    force_pow_2_scales: bool,
) -> None:
    """Apply RoPE to Q/K of ``mixed_qkv`` [s, b, g, (hq + 2) * d] (bf16) and quantize Q, K, V
    with one current-scaling scale into ``out`` (flat float8, packed [Q | K | V] with Q as
    (s, b, g * hq, d) and K, V as (s, b, g, d)). ``scale_inv`` is a 1-element fp32 tensor."""
    s, b, g, width = mixed_qkv.shape
    d = cos.shape[-1]
    n_tok = s * b
    assert width == (num_q_per_group + 2) * d, "mixed_qkv last dim must be (hq + 2) * d"
    assert out.numel() == n_tok * g * width, "out must hold the packed [Q | K | V] payload"
    assert mixed_qkv.numel() < 2**31, "tensor too large for 32-bit indexing"
    device = mixed_qkv.device
    if device.index not in _NUM_SMS:
        _NUM_SMS[device.index] = torch.cuda.get_device_properties(device).multi_processor_count
    ws = _workspace(device)
    common = dict(
        B=b,
        G=g,
        HQ=num_q_per_group,
        D=d,
        FP8MAX=_FP8_MAX[out.dtype],
        POW2=bool(force_pow_2_scales),
    )
    rope_qkv_quantize_kernel[(min(n_tok, _NUM_SMS[device.index] * _CTAS_PER_SM),)](
        mixed_qkv,
        cos,
        sin,
        out,
        ws,
        scale_inv,
        n_tok,
        eps,
        CAST=False,
        NUM_STAGES=_AMAX_STAGES,
        num_warps=_AMAX_WARPS,
        **common,
    )
    rope_qkv_quantize_kernel[(n_tok,)](
        mixed_qkv,
        cos,
        sin,
        out,
        ws,
        scale_inv,
        n_tok,
        eps,
        CAST=True,
        num_warps=_CAST_WARPS,
        **common,
    )
