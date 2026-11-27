# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Reference MXFP4 quantization in plain PyTorch."""

from typing import Tuple

import torch

BLOCK_SIZE = 32
E2M1_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def float_to_e8m0_round_up(x: torch.Tensor) -> torch.Tensor:
    """Biased E8M0 exponent of the smallest power of two >= x (x >= 0, normal or zero)."""
    bits = x.to(torch.float32).view(torch.int32)
    exponent = (bits >> 23) & 0xFF
    mantissa = bits & 0x7FFFFF
    biased = exponent + (mantissa != 0).to(torch.int32)
    return biased.clamp(max=254).to(torch.uint8)


def e8m0_to_float(biased: torch.Tensor) -> torch.Tensor:
    """Decode biased E8M0 exponents."""
    return torch.ldexp(torch.ones_like(biased, dtype=torch.float32), biased.to(torch.int32) - 127)


def e8m0_reciprocal(biased: torch.Tensor) -> torch.Tensor:
    """2^(127 - biased), as computed by the quantization kernels."""
    return torch.ldexp(torch.ones_like(biased, dtype=torch.float32), 127 - biased.to(torch.int32))


def cast_to_e2m1_codes(x: torch.Tensor) -> torch.Tensor:
    """Round-to-nearest-even, saturating cast to 4-bit E2M1 codes (sign in bit 3)."""
    grid = torch.tensor(E2M1_VALUES, dtype=torch.float32, device=x.device)
    mag = x.abs().clamp(max=6.0)
    hi = torch.searchsorted(grid, mag, right=False).clamp(1, len(E2M1_VALUES) - 1)
    lo = hi - 1
    d_lo = mag - grid[lo]
    d_hi = grid[hi] - mag
    pick_hi = (d_hi < d_lo) | ((d_hi == d_lo) & (hi % 2 == 0))
    pick_hi = pick_hi | (mag == grid[hi])
    idx = torch.where(pick_hi, hi, lo)
    idx = torch.where(mag == 0, torch.zeros_like(idx), idx)
    sign = torch.signbit(x) & (idx != 0)
    return (idx | (sign.to(idx.dtype) << 3)).to(torch.uint8)


def e2m1_codes_to_float(codes: torch.Tensor) -> torch.Tensor:
    """Decode 4-bit E2M1 codes."""
    grid = torch.tensor(E2M1_VALUES, dtype=torch.float32, device=codes.device)
    mag = grid[(codes & 0x7).long()]
    return torch.where((codes & 0x8) != 0, -mag, mag)


def pack_e2m1(codes: torch.Tensor) -> torch.Tensor:
    """Pack pairs of codes along the last dimension, low nibble first."""
    return (codes[..., 0::2] | (codes[..., 1::2] << 4)).to(torch.uint8)


def unpack_e2m1(packed: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`pack_e2m1`."""
    lo = packed & 0xF
    hi = packed >> 4
    return torch.stack((lo, hi), dim=-1).reshape(*packed.shape[:-1], 2 * packed.shape[-1])


def quantize_mxfp4_rowwise(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize blocks of 32 along the last dim; returns (codes [M, K], scales [M, K/32])."""
    x = x.to(torch.float32)
    rows, cols = x.shape
    blocks = x.view(rows, cols // BLOCK_SIZE, BLOCK_SIZE)
    amax = blocks.abs().amax(dim=-1)
    scales = float_to_e8m0_round_up(amax * torch.tensor(1.0 / 6.0, dtype=torch.float32))
    scaled = blocks * e8m0_reciprocal(scales).unsqueeze(-1)
    codes = cast_to_e2m1_codes(scaled).view(rows, cols)
    return codes, scales


def quantize_mxfp4_columnwise(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize blocks of 32 along the first dim; returns (codes [M, K], scales [M/32, K])."""
    codes_t, scales_t = quantize_mxfp4_rowwise(x.t().contiguous())
    return codes_t.t().contiguous(), scales_t.t().contiguous()


def dequantize_mxfp4_rowwise(codes: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """Dequantize row-wise MXFP4 codes and compact scales."""
    values = e2m1_codes_to_float(codes)
    return values * e8m0_to_float(scales).repeat_interleave(BLOCK_SIZE, dim=1)


def dequantize_mxfp4_columnwise(codes: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """Dequantize column-wise MXFP4 codes and compact scales."""
    values = e2m1_codes_to_float(codes)
    return values * e8m0_to_float(scales).repeat_interleave(BLOCK_SIZE, dim=0)
