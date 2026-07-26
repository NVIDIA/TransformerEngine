# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Pure-PyTorch reference for MXFP4 weight fake-quantization.

Bit-identical specification of transformer_engine.pytorch.mxfp4_qat
(tex.mxfp4_fake_quantize): per 1x32 block, scale = 2^clamp(ceil(log2(amax/6)),
-126, 125) derived from the amax bit pattern, RTNE onto the E2M1 grid,
saturate at 6, multiply back. Non-finite values NaN-poison their whole block;
the sign of -0 is preserved; the scale floor/cap keep every grid value exactly
representable in bf16/fp32.
"""
import torch

_E2M1_MAX = 6.0
_MXFP4_BLOCK = 32


def round_to_e2m1_grid(y: torch.Tensor) -> torch.Tensor:
    """RTNE onto the E2M1 magnitude grid {0, .5, 1, 1.5, 2, 3, 4, 6}; input in [0, 6]."""
    fine = torch.round(y * 2.0) * 0.5
    mid = torch.round(y)
    coarse = torch.round(y * 0.5) * 2.0
    return torch.where(y <= 2.0, fine, torch.where(y <= 4.0, mid, coarse))


def mxfp4_fake_quantize_reference(weight: torch.Tensor) -> torch.Tensor:
    """Numerical oracle for the mxfp4_fake_quantize kernel (composite torch ops)."""
    rows, cols = weight.shape
    w32 = weight.contiguous().to(torch.float32).view(rows, cols // _MXFP4_BLOCK, _MXFP4_BLOCK)
    amax = w32.abs().amax(dim=-1, keepdim=True)
    nonfinite = ~torch.isfinite(amax)

    bits = amax.view(torch.int32)
    exp_field = bits >> 23
    mantissa = bits & 0x7FFFFF
    exp = exp_field - 129 + (mantissa > 0x400000).to(torch.int32)
    exp = torch.where(exp_field > 0, exp, torch.full_like(exp, -126))
    exp = exp.clamp(min=-126, max=125)
    scale = torch.ldexp(torch.ones_like(amax), exp)
    scale = torch.where(amax > 0, scale, torch.ones_like(scale))

    y = (w32 / scale).abs().clamp(max=_E2M1_MAX)
    q = round_to_e2m1_grid(torch.where(nonfinite, torch.zeros_like(y), y))
    q = torch.copysign(q, w32)
    out = q * scale
    out = torch.where(nonfinite.expand_as(out), torch.full_like(out, float("nan")), out)
    return out.view(rows, cols).to(weight.dtype)
