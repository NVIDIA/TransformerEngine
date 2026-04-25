# Compare: TE-style NVFP4 (single S_enc from *global* amax) vs
# reference 1x64 (S_enc from max|x| in each 1x64 K-window = per-row for K=64).
# Matrix shape (M, K) = (64, 64), rowwise along K, 1x16 blocks * 4 per row.
#
# Run (no torch): python3 reference_hierarchical_nvfp4/compare_64x64_global_vs_1x64.py

from __future__ import annotations

import os
import sys

import numpy as np

# Load sibling ref modules (avoid package __init__ -> torch)
_REF_DIR = os.path.dirname(os.path.abspath(__file__))
if _REF_DIR not in sys.path:
    sys.path.insert(0, _REF_DIR)

from fp8_e4m3_utils_np import (
    TINY,
    compute_S_dec_f32_before_cast_te,
    compute_S_enc_from_amax_1x64_like_te,
    e4m3_u8_to_f32,
    f32_to_e4m3_u8,
)
from hierarchical_nvfp4_ref_numpy import (
    FINE,
    FP4_E2M1_GRID,
    dequantize_rowwise,
    quantize_rowwise_1x64_1x16,
)

M = 64
K = 64


def _round_nearest_fp4(x: np.ndarray) -> np.ndarray:
    d = np.abs(x[..., None] - FP4_E2M1_GRID)
    return FP4_E2M1_GRID[np.argmin(d, axis=-1).astype(np.int64)]


def te_nvfp4_rowwise_global_senc(
    x: np.ndarray, eps: float = 1e-12
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    """
    Same math as ref but S_enc = 2688 / global_amax (single float for all blocks).
    Returns: x_recon, global_amax, w_pre_fp4, S_dec_u8  (M, n16) per-block fp8
    """
    x = np.asarray(x, np.float32)
    m, k = x.shape
    g_amax = float(np.max(np.abs(x)))
    S_g = compute_S_enc_from_amax_1x64_like_te(g_amax)
    n16 = (k + FINE - 1) // FINE
    w = np.empty_like(x)
    s_dec = np.empty((m, n16), dtype=np.uint8)
    for r in range(m):
        t16b = 0
        while t16b * FINE < k:
            lo, hi = t16b * FINE, min((t16b + 1) * FINE, k)
            segx = x[r, lo:hi]
            bamax = float(np.max(np.abs(segx)))
            if bamax < eps:
                bamax = float(eps)
            raw = compute_S_dec_f32_before_cast_te(bamax, S_g)
            u = f32_to_e4m3_u8(np.array([raw], dtype=np.float32).reshape(1))
            s_dec[r, t16b] = u.reshape(-1)[0]
            s_d = max(float(e4m3_u8_to_f32(s_dec[r : r + 1, t16b : t16b + 1]).reshape(-1)[0]), TINY)
            bsi = S_g / s_d
            w[r, lo:hi] = segx * bsi
            t16b += 1
    q = _round_nearest_fp4(w)
    t16g = (np.arange(k) // FINE).astype(np.int64)
    sde = e4m3_u8_to_f32(s_dec[:, t16g].astype(np.uint8))
    sde = np.maximum(sde, TINY)
    x_recon = q * (sde / S_g)
    return x_recon.astype(np.float32), g_amax, w, s_dec


def main() -> None:
    rng = np.random.default_rng(2026)
    # "Real" data: not uniform — heavy-tailed + one row scaled up for local/global gap
    x = rng.standard_normal((M, K)).astype(np.float32)
    x *= 0.35
    x[7, :] *= 4.0
    x[0:8, 12:20] += 0.4

    x_ref1 = quantize_rowwise_1x64_1x16(x)
    recon_1x64 = dequantize_rowwise(x_ref1)
    recon_global, g_amax, _, _ = te_nvfp4_rowwise_global_senc(x)

    d = np.abs(recon_1x64 - x)
    d2 = np.abs(recon_global - x)
    dg = np.abs(recon_1x64 - recon_global)

    print("=== 64x64 数值对比 (rowwise, K=4×16) ===")
    print("global_amax =", g_amax)
    print("S_enc 现网(全局) = 2688 / global_amax =", compute_S_enc_from_amax_1x64_like_te(g_amax))
    print("---")
    print("量化再反归一 vs 原张量: max abs err  [1x64 S_enc 参考] :", float(np.max(d)))
    print("量化再反归一 vs 原张量: max abs err  [全局 S_enc]       :", float(np.max(d2)))
    print("RMS 误差 vs 原张量 [1x64]:", float(np.sqrt(np.mean(d**2))))
    print("RMS 误差 vs 原张量 [全局]:", float(np.sqrt(np.mean(d2**2))))
    print("---")
    print("两种重建之间的 max abs 差 |recon_1x64 - recon_global| :", float(np.max(dg)))
    print("RMS( recon_1x64 - recon_global )                   :", float(np.sqrt(np.mean(dg**2))))
    fn = float(np.linalg.norm(x, "fro"))
    if fn > 0:
        print("||x||_F =", fn)
        print("Fro 相对: ||recon_1x64 - recon_global||_F / ||x||_F =", float(np.linalg.norm(dg, "fro") / fn))


if __name__ == "__main__":
    main()
