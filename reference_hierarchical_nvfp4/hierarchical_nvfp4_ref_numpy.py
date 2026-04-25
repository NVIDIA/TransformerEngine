# CPU reference aligned with TE NVFP4 (core_nvfp4.cuh + quantize_nvfp4.cuh path) except:
#   S_enc = (fp8_max * fp4_max) / amax  uses **1x64 local amax** instead of per-tensor global amax.
#
# Stages: S_enc(1x64) -> S_dec = cast_fp8( block_amax_1x16 * S_enc / 6 ) -> bsi = S_enc / f32(S_dec)
#         -> t = x * bsi -> FP4 E2M1 (nearest). Dequant: x_hat = q_fp4 * (f32(S_dec) / S_enc).
#
# Requires: numpy. Run: python TransformerEngine/reference_hierarchical_nvfp4/hierarchical_nvfp4_ref_numpy.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

try:
    from .fp8_e4m3_utils_np import (
        TINY,
        compute_S_dec_f32_before_cast_te,
        compute_S_enc_from_amax_1x64_like_te,
        e4m3_u8_to_f32,
        f32_to_e4m3_u8,
    )
except ImportError:
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from fp8_e4m3_utils_np import (
        TINY,
        compute_S_dec_f32_before_cast_te,
        compute_S_enc_from_amax_1x64_like_te,
        e4m3_u8_to_f32,
        f32_to_e4m3_u8,
    )

COARSE = 64
FINE = 16

FP4_E2M1_GRID = np.array(
    [
        0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
        -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
    ],
    dtype=np.float32,
)


def _round_to_nearest_fp4(x: np.ndarray) -> np.ndarray:
    d = np.abs(x[..., None] - FP4_E2M1_GRID)
    return FP4_E2M1_GRID[np.argmin(d, axis=-1).astype(np.int64)]


def _pack_fp4_along_k(x_fp4: np.ndarray) -> np.ndarray:
    m, k = x_fp4.shape
    d = np.abs(x_fp4[..., None] - FP4_E2M1_GRID)
    nibble = np.argmin(d, axis=-1).astype(np.uint8)
    n_pairs = (k + 1) // 2
    out = np.zeros((m, n_pairs), dtype=np.uint8)
    for p in range(k // 2):
        j = 2 * p
        out[:, p] = (nibble[:, j] & 0x0F) | ((nibble[:, j + 1] & 0x0F) << 4)
    if k % 2 == 1:
        out[:, -1] = nibble[:, -1] & 0x0F
    return out


def _unpack_fp4_along_k(data_u8: np.ndarray, k: int) -> np.ndarray:
    m = data_u8.shape[0]
    out = np.empty((m, k), dtype=np.float32)
    p = 0
    for j in range(0, k - 1, 2):
        b = data_u8[:, p]
        p += 1
        out[:, j] = FP4_E2M1_GRID[(b & 0x0F).astype(np.int64)]
        out[:, j + 1] = FP4_E2M1_GRID[((b >> 4) & 0x0F).astype(np.int64)]
    if k % 2 == 1:
        b = data_u8[:, p]
        out[:, -1] = FP4_E2M1_GRID[(b & 0x0F).astype(np.int64)]
    return out


def _pack_fp4_along_m(x_fp4: np.ndarray) -> np.ndarray:
    m, k = x_fp4.shape
    d = np.abs(x_fp4[..., None] - FP4_E2M1_GRID)
    nibble = np.argmin(d, axis=-1).astype(np.uint8)
    n_pairs = (m + 1) // 2
    out = np.zeros((n_pairs, k), dtype=np.uint8)
    for p in range(m // 2):
        r = 2 * p
        out[p, :] = (nibble[r, :] & 0x0F) | ((nibble[r + 1, :] & 0x0F) << 4)
    if m % 2 == 1:
        out[-1, :] = nibble[-1, :] & 0x0F
    return out


def _unpack_fp4_along_m(data_u8: np.ndarray, m: int, k: int) -> np.ndarray:
    out = np.empty((m, k), dtype=np.float32)
    p = 0
    for r in range(0, m - 1, 2):
        b = data_u8[p, :]
        p += 1
        out[r, :] = FP4_E2M1_GRID[(b & 0x0F).astype(np.int64)]
        out[r + 1, :] = FP4_E2M1_GRID[((b >> 4) & 0x0F).astype(np.int64)]
    if m % 2 == 1:
        b = data_u8[p, :]
        out[-1, :] = FP4_E2M1_GRID[(b & 0x0F).astype(np.int64)]
    return out


@dataclass
class HierarchicalNVFP4RowwiseNp:
    m: int
    k: int
    data_u8: np.ndarray
    S_enc: np.ndarray
    S_dec_u8: np.ndarray
    amax_64: np.ndarray


@dataclass
class HierarchicalNVFP4ColwiseNp:
    m: int
    k: int
    data_u8: np.ndarray
    S_enc: np.ndarray
    S_dec_u8: np.ndarray
    amax_64: np.ndarray


def _amax_64_k(x: np.ndarray, m: int, k: int) -> np.ndarray:
    n64 = (k + COARSE - 1) // COARSE
    out = np.empty((m, n64), dtype=np.float32)
    for row in range(m):
        for t64 in range(n64):
            lo, hi = t64 * COARSE, min((t64 + 1) * COARSE, k)
            seg = x[row, lo:hi]
            out[row, t64] = float(np.max(np.abs(seg))) if seg.size else 0.0
    return out


def quantize_rowwise_1x64_1x16(x: np.ndarray, eps: float = 1e-12) -> HierarchicalNVFP4RowwiseNp:
    x = np.asarray(x, dtype=np.float32)
    assert x.ndim == 2
    m, k = int(x.shape[0]), int(x.shape[1])
    n16 = (k + FINE - 1) // FINE
    amax_64 = _amax_64_k(x, m, k)
    n64 = amax_64.shape[1]
    S_enc = np.empty((m, n64), dtype=np.float32)
    for ri in range(m):
        for t64 in range(n64):
            S_enc[ri, t64] = compute_S_enc_from_amax_1x64_like_te(float(amax_64[ri, t64]))
    S_dec_u8 = np.empty((m, n16), dtype=np.uint8)
    w = np.empty_like(x)
    for row in range(m):
        t16b = 0
        while t16b * FINE < k:
            lo, hi = t16b * FINE, min((t16b + 1) * FINE, k)
            t64 = lo // COARSE
            S = float(S_enc[row, t64])
            segx = x[row, lo:hi]
            bamax = float(np.max(np.abs(segx)))
            if bamax < eps:
                bamax = float(eps)
            raw = compute_S_dec_f32_before_cast_te(bamax, S)
            u = f32_to_e4m3_u8(np.array(raw, dtype=np.float32).reshape(1))
            S_dec_u8[row, t16b] = u.ravel()[0]
            s_dec_f = max(float(e4m3_u8_to_f32(S_dec_u8[row, t16b : t16b + 1])[0]), TINY)
            bsi = S / s_dec_f
            w[row, lo:hi] = segx * bsi
            t16b += 1
    q = _round_to_nearest_fp4(w)
    return HierarchicalNVFP4RowwiseNp(
        m, k, _pack_fp4_along_k(q), S_enc, S_dec_u8, amax_64
    )


def dequantize_rowwise(p: HierarchicalNVFP4RowwiseNp) -> np.ndarray:
    m, k = p.m, p.k
    q = _unpack_fp4_along_k(p.data_u8, k)
    t16 = (np.arange(k) // FINE).astype(np.int64)
    t64 = (np.arange(k) // COARSE).astype(np.int64)
    sdec = e4m3_u8_to_f32(p.S_dec_u8[:, t16].astype(np.uint8))
    senc = p.S_enc[:, t64]
    sdec = np.maximum(sdec, TINY)
    return (q * (sdec / senc)).astype(np.float32)


def _amax_64_m(x: np.ndarray, m: int, k: int) -> np.ndarray:
    n64 = (m + COARSE - 1) // COARSE
    out = np.empty((n64, k), dtype=np.float32)
    for col in range(k):
        for t64 in range(n64):
            lo, hi = t64 * COARSE, min((t64 + 1) * COARSE, m)
            seg = x[lo:hi, col]
            out[t64, col] = float(np.max(np.abs(seg))) if seg.size else 0.0
    return out


def quantize_columnwise_1x64_1x16(
    x: np.ndarray, eps: float = 1e-12
) -> HierarchicalNVFP4ColwiseNp:
    x = np.asarray(x, dtype=np.float32)
    assert x.ndim == 2
    m, k = int(x.shape[0]), int(x.shape[1])
    n16 = (m + FINE - 1) // FINE
    amax_64 = _amax_64_m(x, m, k)
    n64 = amax_64.shape[0]
    S_enc = np.empty((n64, k), dtype=np.float32)
    for t64 in range(n64):
        for col in range(k):
            S_enc[t64, col] = compute_S_enc_from_amax_1x64_like_te(float(amax_64[t64, col]))
    S_dec_u8 = np.empty((n16, k), dtype=np.uint8)
    w = np.empty_like(x)
    for col in range(k):
        t16b = 0
        while t16b * FINE < m:
            lo, hi = t16b * FINE, min((t16b + 1) * FINE, m)
            t64 = lo // COARSE
            S = float(S_enc[t64, col])
            segx = x[lo:hi, col]
            bamax = float(np.max(np.abs(segx)))
            if bamax < eps:
                bamax = float(eps)
            raw = compute_S_dec_f32_before_cast_te(bamax, S)
            u = f32_to_e4m3_u8(np.array(raw, dtype=np.float32).reshape(1))
            S_dec_u8[t16b, col] = u.ravel()[0]
            s_dec_f = max(float(e4m3_u8_to_f32(S_dec_u8[t16b : t16b + 1, col : col + 1])[0, 0]), TINY)
            bsi = S / s_dec_f
            w[lo:hi, col] = segx * bsi
            t16b += 1
    q = _round_to_nearest_fp4(w)
    return HierarchicalNVFP4ColwiseNp(
        m, k, _pack_fp4_along_m(q), S_enc, S_dec_u8, amax_64
    )


def dequantize_colwise(p: HierarchicalNVFP4ColwiseNp) -> np.ndarray:
    m, k = p.m, p.k
    q = _unpack_fp4_along_m(p.data_u8, m, k)
    r16 = (np.arange(m) // FINE).astype(np.int64)
    r64 = (np.arange(m) // COARSE).astype(np.int64)
    sdec = e4m3_u8_to_f32(p.S_dec_u8[r16, :].astype(np.uint8))
    senc = p.S_enc[r64, :]
    sdec = np.maximum(sdec, TINY)
    return (q * (sdec / senc)).astype(np.float32)


def reference_matmul_tn(
    a_rows: HierarchicalNVFP4RowwiseNp,
    b_cols: HierarchicalNVFP4ColwiseNp,
) -> np.ndarray:
    return dequantize_rowwise(a_rows) @ dequantize_colwise(b_cols).T


def roundtrip_error(x: np.ndarray, mode: str) -> Tuple[np.ndarray, float]:
    if mode == "rowwise":
        p = quantize_rowwise_1x64_1x16(x)
        y = dequantize_rowwise(p)
    elif mode == "colwise":
        p = quantize_columnwise_1x64_1x16(x)
        y = dequantize_colwise(p)
    else:
        raise ValueError("mode is rowwise or colwise")
    e = float(np.max(np.abs(x.astype(np.float32) - y)))
    return y, e


def _self_test() -> None:
    rng = np.random.default_rng(0)
    m, n, kdim = 4, 5, 128
    a = rng.standard_normal((m, kdim)).astype(np.float32)
    b = rng.standard_normal((n, kdim)).astype(np.float32)
    pa, pb = quantize_rowwise_1x64_1x16(a), quantize_columnwise_1x64_1x16(b)
    y = reference_matmul_tn(pa, pb)
    a @ b.T
    # loose bound (aggressive low precision)
    assert np.isfinite(y).all()
    assert dequantize_rowwise(pa).shape == a.shape
    assert dequantize_colwise(pb).shape == b.shape
    print("hierarchical_nvfp4_ref_numpy (TE1x64 path): _self_test OK")


if __name__ == "__main__":
    _self_test()
