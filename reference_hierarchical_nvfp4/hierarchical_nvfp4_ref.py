# PyTorch twin of `hierarchical_nvfp4_ref_numpy.py`.
#
# Aligned with TE NVFP4 (core_nvfp4.cuh, quantize_nvfp4.cuh) for:
#   S_enc, S_dec = cast_f32(block_amax * S_enc / 6) as fp8e4m3, block_scale_inv = S_enc / f32(S_dec),
#   then FP4 from x * bsi. **S_enc** uses **1x64 local amax** instead of per-tensor global amax.
#
# Standalone. CPU twin: hierarchical_nvfp4_ref_numpy.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch

import numpy as np

from .fp8_e4m3_utils_np import TINY, compute_S_dec_f32_before_cast_te
from .fp8_e4m3_utils_np import compute_S_enc_from_amax_1x64_like_te as _s_enc_from_amax_cpu

try:
    from .fp8_e4m3_utils_np import e4m3_u8_to_f32 as _e4m3_u8_f32_np
    from .fp8_e4m3_utils_np import f32_to_e4m3_u8 as _f32_e4m3_u8_np
except ImportError:
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from fp8_e4m3_utils_np import e4m3_u8_to_f32 as _e4m3_u8_f32_np
    from fp8_e4m3_utils_np import f32_to_e4m3_u8 as _f32_e4m3_u8_np

COARSE = 64
FINE = 16


def fp4_e2m1_grid_torch(device, dtype) -> torch.Tensor:
    vals = (
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
        3.0,
        4.0,
        6.0,
        -0.0,
        -0.5,
        -1.0,
        -1.5,
        -2.0,
        -3.0,
        -4.0,
        -6.0,
    )
    return torch.tensor(vals, device=device, dtype=dtype)


def _round_to_nearest_fp4(x: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
    diff = (x.unsqueeze(-1) - grid).abs()
    return grid[diff.argmin(dim=-1)]


def _pack_fp4_along_k(x_fp4: torch.Tensor) -> torch.Tensor:
    m, k = x_fp4.shape
    grid = fp4_e2m1_grid_torch(x_fp4.device, torch.float32)
    diff = (x_fp4.unsqueeze(-1) - grid).abs()
    nibble = diff.argmin(dim=-1).to(torch.uint8)
    n_pairs = (k + 1) // 2
    out = torch.zeros(m, n_pairs, dtype=torch.uint8, device=x_fp4.device)
    for p in range(k // 2):
        j = 2 * p
        out[:, p] = (nibble[:, j] & 0x0F) | ((nibble[:, j + 1] & 0x0F) << 4)
    if k % 2 == 1:
        out[:, -1] = nibble[:, -1] & 0x0F
    return out


def _unpack_fp4_along_k(data_u8: torch.Tensor, k: int) -> torch.Tensor:
    m = data_u8.size(0)
    grid = fp4_e2m1_grid_torch(data_u8.device, torch.float32)
    out = torch.empty(m, k, device=data_u8.device, dtype=grid.dtype)
    p = 0
    for j in range(0, k - 1, 2):
        b = data_u8[:, p]
        p += 1
        out[:, j] = grid[(b & 0x0F).to(torch.long)]
        out[:, j + 1] = grid[((b >> 4) & 0x0F).to(torch.long)]
    if k % 2 == 1:
        b = data_u8[:, p]
        out[:, -1] = grid[(b & 0x0F).to(torch.long)]
    return out


def _pack_fp4_along_m(x_fp4: torch.Tensor) -> torch.Tensor:
    m, k = x_fp4.shape
    grid = fp4_e2m1_grid_torch(x_fp4.device, torch.float32)
    diff = (x_fp4.unsqueeze(-1) - grid).abs()
    nibble = diff.argmin(dim=-1).to(torch.uint8)
    n_pairs = (m + 1) // 2
    out = torch.zeros(n_pairs, k, dtype=torch.uint8, device=x_fp4.device)
    for p in range(m // 2):
        r = 2 * p
        out[p, :] = (nibble[r, :] & 0x0F) | ((nibble[r + 1, :] & 0x0F) << 4)
    if m % 2 == 1:
        out[-1, :] = nibble[-1, :] & 0x0F
    return out


def _unpack_fp4_along_m(data_u8: torch.Tensor, m: int, k: int) -> torch.Tensor:
    grid = fp4_e2m1_grid_torch(data_u8.device, torch.float32)
    out = torch.empty(m, k, device=data_u8.device, dtype=grid.dtype)
    p = 0
    for r in range(0, m - 1, 2):
        b = data_u8[p, :]
        p += 1
        out[r, :] = grid[(b & 0x0F).to(torch.long)]
        out[r + 1, :] = grid[((b >> 4) & 0x0F).to(torch.long)]
    if m % 2 == 1:
        b = data_u8[p, :]
        out[-1, :] = grid[(b & 0x0F).to(torch.long)]
    return out


@dataclass
class HierarchicalNVFP4Rowwise:
    m: int
    k: int
    data_u8: torch.Tensor
    S_enc: torch.Tensor
    S_dec_u8: torch.Tensor
    amax_64: torch.Tensor


@dataclass
class HierarchicalNVFP4Colwise:
    m: int
    k: int
    data_u8: torch.Tensor
    S_enc: torch.Tensor
    S_dec_u8: torch.Tensor
    amax_64: torch.Tensor


def _amax_64_k(x: torch.Tensor) -> torch.Tensor:
    m, k = int(x.size(0)), int(x.size(1))
    n64 = (k + COARSE - 1) // COARSE
    a = x.new_empty(m, n64)
    for t64 in range(n64):
        lo, hi = t64 * COARSE, min((t64 + 1) * COARSE, k)
        a[:, t64] = x[:, lo:hi].abs().max(dim=1).values
    return a


def quantize_rowwise_1x64_1x16(x: torch.Tensor, eps: float = 1e-12) -> HierarchicalNVFP4Rowwise:
    assert x.dim() == 2
    m, k = int(x.size(0)), int(x.size(1))
    device, dtype = x.device, x.dtype
    x = x.to(torch.float32)
    n16 = (k + FINE - 1) // FINE
    amax_64 = _amax_64_k(x)
    n64 = amax_64.size(1)
    S_enc = torch.empty(m, n64, device=device, dtype=torch.float32)
    for ri in range(m):
        for t64 in range(n64):
            S_enc[ri, t64] = _s_enc_from_amax_cpu(float(amax_64[ri, t64].item()))
    S_dec_u8 = torch.empty(m, n16, device=device, dtype=torch.uint8)
    w = x.clone()
    grid = fp4_e2m1_grid_torch(device, torch.float32)
    for row in range(m):
        t16b = 0
        while t16b * FINE < k:
            lo, hi = t16b * FINE, min((t16b + 1) * FINE, k)
            t64 = lo // COARSE
            s_e = S_enc[row, t64]
            segx = x[row, lo:hi]
            bamax = float(segx.abs().max().item())
            if bamax < eps:
                bamax = float(eps)
            raw = compute_S_dec_f32_before_cast_te(bamax, float(s_e.item()))
            u = int(_f32_e4m3_u8_np(np.array([raw], dtype=np.float32).reshape(1))[0])
            S_dec_u8[row, t16b] = u
            s_dec_f = max(float(_e4m3_u8_f32_np(np.array([u], dtype=np.uint8))[0]), TINY)
            bsi = float(s_e.item()) / s_dec_f
            w[row, lo:hi] = segx * bsi
            t16b += 1
    q = _round_to_nearest_fp4(w, grid)
    return HierarchicalNVFP4Rowwise(m, k, _pack_fp4_along_k(q), S_enc, S_dec_u8, amax_64)


def dequantize_rowwise(p: HierarchicalNVFP4Rowwise) -> torch.Tensor:
    m, k, device = p.m, p.k, p.data_u8.device
    q = _unpack_fp4_along_k(p.data_u8, k)
    j16 = (torch.arange(k, device=device) // FINE).long()
    j64 = (torch.arange(k, device=device) // COARSE).long()
    sdec = torch.from_numpy(_e4m3_u8_f32_np(p.S_dec_u8[:, j16].cpu().numpy().astype(np.uint8))).to(
        device=device, dtype=torch.float32
    )
    senc = p.S_enc[:, j64]
    sdec = torch.clamp(sdec, min=TINY)
    return (q * (sdec / senc)).to(torch.float32)


def _amax_64_m(x: torch.Tensor) -> torch.Tensor:
    m, k = int(x.size(0)), int(x.size(1))
    n64 = (m + COARSE - 1) // COARSE
    a = x.new_empty(n64, k)
    for t64 in range(n64):
        lo, hi = t64 * COARSE, min((t64 + 1) * COARSE, m)
        a[t64, :] = x[lo:hi, :].abs().max(dim=0).values
    return a


def quantize_columnwise_1x64_1x16(x: torch.Tensor, eps: float = 1e-12) -> HierarchicalNVFP4Colwise:
    assert x.dim() == 2
    m, k = int(x.size(0)), int(x.size(1))
    device = x.device
    x = x.to(torch.float32)
    n16 = (m + FINE - 1) // FINE
    amax_64 = _amax_64_m(x)
    n64 = amax_64.size(0)
    S_enc = torch.empty(n64, k, device=device, dtype=torch.float32)
    for t64 in range(n64):
        for col in range(k):
            S_enc[t64, col] = _s_enc_from_amax_cpu(float(amax_64[t64, col].item()))
    S_dec_u8 = torch.empty(n16, k, device=device, dtype=torch.uint8)
    w = x.clone()
    grid = fp4_e2m1_grid_torch(device, torch.float32)
    for col in range(k):
        t16b = 0
        while t16b * FINE < m:
            lo, hi = t16b * FINE, min((t16b + 1) * FINE, m)
            t64 = lo // COARSE
            s_e = S_enc[t64, col]
            segx = x[lo:hi, col]
            bamax = float(segx.abs().max().item())
            if bamax < eps:
                bamax = float(eps)
            raw = compute_S_dec_f32_before_cast_te(bamax, float(s_e.item()))
            u = int(_f32_e4m3_u8_np(np.array([raw], dtype=np.float32).reshape(1))[0])
            S_dec_u8[t16b, col] = u
            s_dec_f = max(float(_e4m3_u8_f32_np(np.array([u], dtype=np.uint8))[0]), TINY)
            bsi = float(s_e.item()) / s_dec_f
            w[lo:hi, col] = segx * bsi
            t16b += 1
    q = _round_to_nearest_fp4(w, grid)
    return HierarchicalNVFP4Colwise(m, k, _pack_fp4_along_m(q), S_enc, S_dec_u8, amax_64)


def dequantize_colwise(p: HierarchicalNVFP4Colwise) -> torch.Tensor:
    m, k, device = p.m, p.k, p.data_u8.device
    q = _unpack_fp4_along_m(p.data_u8, m, k)
    r16 = (torch.arange(m, device=device) // FINE).long()
    r64 = (torch.arange(m, device=device) // COARSE).long()
    sdec = torch.from_numpy(_e4m3_u8_f32_np(p.S_dec_u8[r16, :].cpu().numpy().astype(np.uint8))).to(
        device=device, dtype=torch.float32
    )
    senc = p.S_enc[r64, :]
    sdec = torch.clamp(sdec, min=TINY)
    return (q * (sdec / senc)).to(torch.float32)


def reference_matmul_tn(
    a_rows: HierarchicalNVFP4Rowwise,
    b_cols: HierarchicalNVFP4Colwise,
) -> torch.Tensor:
    return dequantize_rowwise(a_rows) @ dequantize_colwise(b_cols).T


def roundtrip_error(x: torch.Tensor, mode: str) -> Tuple[torch.Tensor, torch.Tensor]:
    if mode == "rowwise":
        p = quantize_rowwise_1x64_1x16(x)
        y = dequantize_rowwise(p)
    elif mode == "colwise":
        p = quantize_columnwise_1x64_1x16(x)
        y = dequantize_colwise(p)
    else:
        raise ValueError("mode is rowwise or colwise")
    e = (x.to(torch.float32) - y).abs().max()
    return y, e


if __name__ == "__main__":
    torch.manual_seed(0)
    m, n, kdim = 4, 5, 128
    a = torch.randn(m, kdim)
    b = torch.randn(n, kdim)
    pa, pb = quantize_rowwise_1x64_1x16(a), quantize_columnwise_1x64_1x16(b)
    y = reference_matmul_tn(pa, pb)
    y_ref = a @ b.T
    err = (y - y_ref).abs().max()
    print("matmul ref max abs err (via dequant):", float(err))
    _, e = roundtrip_error(a, "rowwise")
    print("rowwise roundtrip max abs err:", float(e))
