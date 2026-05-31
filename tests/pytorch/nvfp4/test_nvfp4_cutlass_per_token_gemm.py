# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Numerical tests for tex.nvfp4_cutlass_per_token_gemm (fwd + bwd).

Layer 3 baseline = REAL-SHIP prod NVFP4 (NVFP4BlockScaling defaults):
  * input quantizer (X): RHT on cols, no SR, 1D
  * weight quantizer (W): no RHT, no SR, 2D (16x16 block scale)
  * grad quantizer (dY): RHT on cols, SR on cast, 1D

per-token (cf) keeps the bare path: no RHT, no SR, plain K1+K2 quant.
The cf/pten ratio therefore reflects the actual ship-decision delta
(per-token's per-row outer scale + reduced launch chain vs prod's
RHT+SR outlier resistance + bias correction).

FWD (3 layers): (1) GEMM-only -- fused EVT vs cuBLAS-LT per-token.
(2) E2E single-path SNR -- per-token quant + fused EVT vs BF16 fp32 GT.
(3) E2E side-by-side SNR (per-token vs prod per-tensor, both vs BF16 GT).

BWD (2 layers, mirror): Layer 2 -- per-token dgrad/wgrad SNR vs fp32
GT. Layer 3 -- per-token vs real-ship prod (general_gemm layout='NN'
for dgrad, layout='NT' for wgrad).

M, N, K % 256 == 0 (kernel + per-token quant alignment).
"""

from __future__ import annotations

from typing import Dict, Tuple

import pytest
import torch

# Must import transformer_engine first to dlopen libtransformer_engine.so.
import transformer_engine.pytorch as te  # noqa: F401
import transformer_engine_torch as tex  # type: ignore
from transformer_engine.pytorch import NVFP4Quantizer


def _has_sm100() -> bool:
    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability()
    return major >= 10


_GATED_SM100 = pytest.mark.skipif(
    not _has_sm100(),
    reason="CUTLASS NVFP4 fused per-token GEMM requires SM100 (Blackwell).",
)

_GATED_HAS_KERNEL = pytest.mark.skipif(
    not hasattr(tex, "nvfp4_cutlass_per_token_gemm"),
    reason="tex.nvfp4_cutlass_per_token_gemm not built into this binary.",
)


def _quantize_per_token(
    x: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-token quantize (rows, K) bf16 -> (q_FP4_packed (rows, K/2),
    sf_FP8e4m3 (rows, K/16), row_amax fp32 (rows,)).
    """
    assert x.dim() == 2 and x.dtype == torch.bfloat16
    rows, K = x.shape
    q_row = torch.empty((rows, K // 2), dtype=torch.uint8, device=x.device)
    s_row = torch.empty((rows, K // 16), dtype=torch.uint8, device=x.device)
    a_row = torch.empty((rows,), dtype=torch.float32, device=x.device)
    q_col = torch.empty(0, dtype=torch.uint8, device=x.device)
    s_col = torch.empty(0, dtype=torch.uint8, device=x.device)
    a_col = torch.empty(0, dtype=torch.float32, device=x.device)
    tex.nvfp4_per_token_quantize(
        x,
        q_row,
        s_row,
        a_row,
        q_col,
        s_col,
        a_col,
        rowwise=True,
        columnwise=False,
        with_rht=False,
        random_sign_mask_t=int(0xACE1),
        with_swizzle=False,
    )
    return q_row, s_row, a_row


def _ref_pertoken_gemm_via_cublaslt(
    a_q: torch.Tensor,
    b_q: torch.Tensor,
    a_sf: torch.Tensor,
    b_sf: torch.Tensor,
    alpha_a: torch.Tensor,
    alpha_b: torch.Tensor,
    M: int,
    N: int,
    K: int,
) -> torch.Tensor:
    """Reference: tex.nvfp4_per_token_gemm = cuBLAS-LT NVFP4 GEMM + standalone
    post-scale. Already correctness-tested in test_nvfp4_per_token.py.
    """
    workspace = torch.empty(33_554_432, dtype=torch.uint8, device=a_q.device)
    d = torch.empty((M, N), dtype=torch.bfloat16, device=a_q.device)
    tex.nvfp4_per_token_gemm(
        a_q,
        b_q,
        a_sf.reshape(-1),
        b_sf.reshape(-1),
        alpha_a,
        alpha_b,
        d,
        workspace,
        M,
        N,
        K,
        1.0,
        0.0,
        a_sf_swizzled=False,
        b_sf_swizzled=False,
    )
    return d


def _run_fused(
    a_q: torch.Tensor,
    b_q: torch.Tensor,
    a_sf: torch.Tensor,
    b_sf: torch.Tensor,
    alpha_a: torch.Tensor,
    alpha_b: torch.Tensor,
    M: int,
    N: int,
    K: int,
) -> torch.Tensor:
    d = torch.empty((M, N), dtype=torch.bfloat16, device=a_q.device)
    tex.nvfp4_cutlass_per_token_gemm(
        a_q,
        b_q,
        a_sf.reshape(-1),
        b_sf.reshape(-1),
        alpha_a,
        alpha_b,
        d,
        M,
        N,
        K,
        a_sf_swizzled=False,
        b_sf_swizzled=False,
    )
    return d


# Shapes obey the kernel contract (M, N, K all multiples of 256).
_SHAPES = [
    (256, 256, 256),  # smallest legal shape
    (512, 256, 256),
    (256, 512, 256),
    (256, 256, 512),
    (512, 1024, 768),  # not power-of-2 K
    (1024, 1024, 1024),
]


@_GATED_SM100
@_GATED_HAS_KERNEL
@pytest.mark.parametrize("M,N,K", _SHAPES)
def test_fused_matches_cublaslt_per_token(M: int, N: int, K: int) -> None:
    """Fused CUTLASS per-token == cuBLAS LT per-token (within bf16 + reduction-order tolerance)."""
    device = torch.device("cuda")
    torch.manual_seed(0xACE1)

    a = (torch.randn((M, K), dtype=torch.bfloat16, device=device) * 0.5).contiguous()
    b = (torch.randn((N, K), dtype=torch.bfloat16, device=device) * 0.5).contiguous()

    # Per-token quantize for both operands.
    a_q, a_sf, a_row_amax = _quantize_per_token(a)
    b_q, b_sf, b_row_amax = _quantize_per_token(b)

    # The two paths share quantizer outputs (a_q, b_q, a_sf, b_sf) and amaxes,
    # so the difference is purely in the GEMM kernel and the order of the
    # per-row * per-col fold (epilogue vs separate kernel).
    d_ref = _ref_pertoken_gemm_via_cublaslt(
        a_q,
        b_q,
        a_sf,
        b_sf,
        a_row_amax,
        b_row_amax,
        M,
        N,
        K,
    )
    d_fused = _run_fused(
        a_q,
        b_q,
        a_sf,
        b_sf,
        a_row_amax,
        b_row_amax,
        M,
        N,
        K,
    )

    # Float32 view for comparison; bf16 ULP is 2^-7 = 7.8e-3 relative.
    ref_f32 = d_ref.float()
    out_f32 = d_fused.float()

    # Diagnostic statistics for failure mode debugging.
    abs_diff = (out_f32 - ref_f32).abs()
    rel_diff = abs_diff / (ref_f32.abs().clamp_min(1e-6))
    max_abs = abs_diff.max().item()
    mean_abs = abs_diff.mean().item()
    max_rel = rel_diff.max().item()
    mean_rel = rel_diff.mean().item()

    # Relative tolerance 2e-2 leaves ~2.5x headroom over the bf16 ULP floor.
    torch.testing.assert_close(out_f32, ref_f32, rtol=2e-2, atol=2e-2)

    print(
        f"  M={M:>5} N={N:>5} K={K:>5}: "
        f"max_abs={max_abs:.3e} mean_abs={mean_abs:.3e} "
        f"max_rel={max_rel:.3e} mean_rel={mean_rel:.3e}"
    )


# NVFP4 spec outer-dequant baked into the fused EVT; cuBLAS-LT auto-folds the
# same factor via its amax slot. Mirror of nvfp4_cutlass_gemm.cu.
NVFP4_DEQUANT_K = 1.0 / (6.0 * 6.0 * 448.0 * 448.0)  # = 1 / 2688^2 ~= 1.38e-7


@_GATED_SM100
@_GATED_HAS_KERNEL
def test_fused_alpha_unity_matches_scalar_gemm_with_baked_const() -> None:
    """With alpha=1 the EVT collapses to D = bf16(NVFP4_DEQUANT_K * acc) and
    must match nvfp4_cutlass_gemm(alpha=NVFP4_DEQUANT_K) BIT-FOR-BIT (same
    mainloop reduction; the *1.0f multiplies are exact in fp32)."""
    M, N, K = 256, 256, 256
    device = torch.device("cuda")
    torch.manual_seed(0xACE2)

    a = (torch.randn((M, K), dtype=torch.bfloat16, device=device) * 0.5).contiguous()
    b = (torch.randn((N, K), dtype=torch.bfloat16, device=device) * 0.5).contiguous()
    a_q, a_sf, _ = _quantize_per_token(a)
    b_q, b_sf, _ = _quantize_per_token(b)

    alpha_a = torch.ones((M,), dtype=torch.float32, device=device)
    alpha_b = torch.ones((N,), dtype=torch.float32, device=device)

    d_fused = _run_fused(a_q, b_q, a_sf, b_sf, alpha_a, alpha_b, M, N, K)

    d_scalar = torch.empty((M, N), dtype=torch.bfloat16, device=device)
    tex.nvfp4_cutlass_gemm(
        a_q,
        b_q,
        a_sf.reshape(-1),
        b_sf.reshape(-1),
        d_scalar,
        M,
        N,
        K,
        NVFP4_DEQUANT_K,
        0.0,
        a_sf_swizzled=False,
        b_sf_swizzled=False,
    )

    # Exact match (any deviation = EVT bug).
    torch.testing.assert_close(
        d_fused.float(),
        d_scalar.float(),
        rtol=0.0,
        atol=0.0,
        msg=(
            "Fused EVT with unity alpha + baked 1/2688^2 must match "
            "nvfp4_cutlass_gemm(alpha=1/2688^2) bit-exact."
        ),
    )


def _bf16_gemm_ground_truth(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """D = bf16(A.float() @ B.float().T). fp32 intermediates so bf16 mainloop
    noise doesn't mix with NVFP4 quant noise in the assert.
    """
    return (a.float() @ b.float().T).bfloat16()


def _make_input_quantizer() -> NVFP4Quantizer:
    """Prod fwd INPUT quantizer (A operand). Matches NVFP4BlockScaling defaults:
    fp4_quant_fwd_inp = QParams(RHT=True, SR=False, 2D=False) in recipe/__init__.py.
    """
    return NVFP4Quantizer(
        fp4_dtype=tex.DType.kFloat4E2M1,
        rowwise=True,
        columnwise=True,
        with_amax_reduction=False,
        amax_reduction_group=None,
        with_rht=True,
        with_post_rht_amax=True,
        with_2d_quantization=False,
        stochastic_rounding=False,
        with_random_sign_mask=True,
    )


def _make_weight_quantizer() -> NVFP4Quantizer:
    """Prod fwd WEIGHT quantizer (B operand). Matches NVFP4BlockScaling defaults:
    fp4_quant_fwd_weight = QParams(RHT=False, SR=False, 2D=True). Weight does
    not need RHT (static, infrequent quant) but uses 2D block scaling.
    """
    return NVFP4Quantizer(
        fp4_dtype=tex.DType.kFloat4E2M1,
        rowwise=True,
        columnwise=True,
        with_amax_reduction=False,
        amax_reduction_group=None,
        with_rht=False,
        with_post_rht_amax=False,
        with_2d_quantization=True,
        stochastic_rounding=False,
        with_random_sign_mask=True,
    )


def _pten_e2e(a: torch.Tensor, b: torch.Tensor, M: int, N: int, K: int) -> torch.Tensor:
    """Prod per-tensor E2E fwd via general_gemm (the real production dispatch
    used by nn.Linear fwd). A through input quantizer (RHT, 1D), B through
    weight quantizer (no RHT, 2D); general_gemm internally swizzles SF for
    both 1D and 2D NVFP4 formats, so this is byte-aligned with
    NVFP4BlockScaling fwd recipe. D = bf16 (M, N).
    """
    from transformer_engine.pytorch.cpp_extensions import general_gemm

    input_q = _make_input_quantizer()
    weight_q = _make_weight_quantizer()
    dst_a = input_q.make_empty(a.shape, dtype=torch.bfloat16, device=a.device)
    dst_b = weight_q.make_empty(b.shape, dtype=torch.bfloat16, device=b.device)
    tex.quantize(a, input_q, dst_a, None)
    tex.quantize(b, weight_q, dst_b, None)
    # general_gemm(weight, input, ...) mirrors linear.py forward: A=weight (N,K),
    # B=input (M,K), default layout="TN", output (M, N).
    del M, N, K  # shapes already encoded in dst_a / dst_b
    d, _, _, _ = general_gemm(dst_b, dst_a, out_dtype=torch.bfloat16)
    return d


def _snr_stats(out: torch.Tensor, ref: torch.Tensor) -> Dict[str, float]:
    """SNR-style stats vs ref. ``rel_l2 = ||out-ref||_2 / ||ref||_2`` is the
    primary metric (robust to near-zero ref entries, unlike pointwise rel).
    ``mean_ratio = mean(|diff|) / mean(|ref|)`` is the secondary metric.
    Pointwise max_rel kept for diagnostics but NOT used in assertions
    (a single near-zero ref entry trivially blows it up to 1e6+).
    """
    out_f32 = out.float()
    ref_f32 = ref.float()
    abs_diff = (out_f32 - ref_f32).abs()
    return {
        "max_abs": abs_diff.max().item(),
        "mean_abs": abs_diff.mean().item(),
        "ref_mean_abs": ref_f32.abs().mean().item(),
        "rel_l2": (abs_diff.norm() / ref_f32.norm().clamp_min(1e-12)).item(),
        "mean_ratio": (abs_diff.mean() / ref_f32.abs().mean().clamp_min(1e-12)).item(),
    }


# Training + small-batch inference shapes (M, N, K % 256 == 0). Kept compact
# so the test stays under a couple seconds per shape on a single GB200 GPU.
_E2E_FWD_SHAPES = [
    (256, 256, 256),
    (512, 512, 512),
    (1024, 1024, 1024),
    (1024, 4096, 4096),
    (2048, 2048, 2048),
]


@_GATED_SM100
@_GATED_HAS_KERNEL
@pytest.mark.parametrize("M,N,K", _E2E_FWD_SHAPES)
def test_e2e_fwd_per_token_vs_bf16_ground_truth(M: int, N: int, K: int) -> None:
    """E2E forward (per-token quant + fused EVT GEMM) vs BF16 fp32 ground truth.
    Uses ``rel_l2 = ||out-ref||_2 / ||ref||_2`` (robust to near-zero refs)
    rather than pointwise rtol+atol; K-small shapes routinely have a few
    output entries with |ref| << atol where pointwise rtol+atol breaks down.
    Single SNR floor for the FULL ship path (quant + GEMM), not just the GEMM.
    """
    device = torch.device("cuda")
    torch.manual_seed(0xACE3)

    a = (torch.randn((M, K), dtype=torch.bfloat16, device=device) * 0.5).contiguous()
    b = (torch.randn((N, K), dtype=torch.bfloat16, device=device) * 0.5).contiguous()

    a_q, a_sf, a_row_amax = _quantize_per_token(a)
    b_q, b_sf, b_row_amax = _quantize_per_token(b)

    d_fused = _run_fused(a_q, b_q, a_sf, b_sf, a_row_amax, b_row_amax, M, N, K)
    d_ref = _bf16_gemm_ground_truth(a, b)

    snr = _snr_stats(d_fused, d_ref)
    print(
        f"  M={M:>5} N={N:>5} K={K:>5}: "
        f"max_abs={snr['max_abs']:.3e} mean_abs={snr['mean_abs']:.3e} "
        f"rel_l2={snr['rel_l2']:.4f} mean_ratio={snr['mean_ratio']:.4f}"
    )

    # Per-token NVFP4 + fused EVT: rel_l2 is ~0.05-0.15 (per-row outer scale
    # gives near-bf16 SNR for K>=512; K=256 lands around ~0.10-0.15 due to
    # smaller-K accumulator window). 0.30 is a generous ship-path ceiling
    # well above measured noise, well below the ~1.0 plumbing-break signature.
    assert snr["rel_l2"] < 0.30, (
        f"per-token + fused EVT rel_l2={snr['rel_l2']:.4f} exceeds 0.30 "
        "-- accuracy regression in quant or GEMM kernel."
    )


@_GATED_SM100
def test_prod_fwd_weight_2d_quant_plumbing() -> None:
    """Sanity: NVFP4BlockScaling fwd pipeline (input 1D+RHT, weight 2D no-RHT)
    via general_gemm plumbs through without errors AND its SNR vs BF16 GT is
    within NVFP4 spec. Standalone plumbing check before the wider Layer 3 SNR
    comparison; fails loudly when general_gemm / quantizer plumbing breaks.
    """
    M, N, K = 512, 512, 512
    device = torch.device("cuda")
    torch.manual_seed(0xACE5)

    a = (torch.randn((M, K), dtype=torch.bfloat16, device=device) * 0.5).contiguous()
    b = (torch.randn((N, K), dtype=torch.bfloat16, device=device) * 0.5).contiguous()

    d_pten = _pten_e2e(a, b, M, N, K)
    d_gt = _bf16_gemm_ground_truth(a, b)
    snr = _snr_stats(d_pten, d_gt)

    print(
        f"\n  prod fwd 2D-weight plumbing  M={M} N={N} K={K}: "
        f"max_abs={snr['max_abs']:.3e} mean_abs={snr['mean_abs']:.3e} "
        f"rel_l2={snr['rel_l2']:.3e} mean_ratio={snr['mean_ratio']:.3e}"
    )

    # Per-tensor NVFP4 at K=512: rel_l2 lands ~0.13-0.18 (single per-tensor
    # outer scale forces wide FP4 dynamic range). 0.30 is a generous plumbing
    # ceiling: well above the ~0.18 fluctuation window, well below the ~1.0
    # signature of "output is unrelated to input".
    assert snr["rel_l2"] < 0.30, (
        "prod fwd (input 1D+RHT, weight 2D no-RHT) rel_l2="
        f"{snr['rel_l2']:.4f} exceeds 0.30 -- pipeline-level bug "
        "(general_gemm dispatch, NVFP4Quantizer plumbing, or 2D SF swizzle)."
    )


# Layer 3 floors. ``rel_l2 = ||out-ref||_2 / ||ref||_2`` is the metric.
# Per-token at K>=512 typically lands at 5-10%; per-tensor at K=512 lands
# 13-18% (single outer scale, coarse FP4 dynamic range). 0.30 is a generous
# upper bound that catches plumbing breakage but tolerates the per-tensor
# small-K hit; both paths must stay under it.
_LAYER3_REL_L2_HARD_FLOOR = 0.30


# ============================================================================
# Backward (dgrad / wgrad) helpers
# ============================================================================


def _quantize_per_token_dual(
    x: torch.Tensor,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,  # rowwise: q, sf, amax (M-vec)
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,  # columnwise: q, sf, amax (K-vec)
]:
    """Per-token dual-direction quantize. For input (M, K) bf16:

    rowwise:    q (M, K/2), sf (M, K/16), amax (M,)
    columnwise: q (K, M/2), sf (K, M/16), amax (K,)   <-- note (K, M) layout

    The columnwise output is the (per-column-amax-quantized) transpose of x
    in raw memory: row k of the columnwise buffer holds the FP4 of x[:, k].
    """
    assert x.dim() == 2 and x.dtype == torch.bfloat16
    M, K = x.shape
    q_row = torch.empty((M, K // 2), dtype=torch.uint8, device=x.device)
    s_row = torch.empty((M, K // 16), dtype=torch.uint8, device=x.device)
    a_row = torch.empty((M,), dtype=torch.float32, device=x.device)
    q_col = torch.empty((K, M // 2), dtype=torch.uint8, device=x.device)
    s_col = torch.empty((K, M // 16), dtype=torch.uint8, device=x.device)
    a_col = torch.empty((K,), dtype=torch.float32, device=x.device)
    tex.nvfp4_per_token_quantize(
        x,
        q_row,
        s_row,
        a_row,
        q_col,
        s_col,
        a_col,
        rowwise=True,
        columnwise=True,
        with_rht=False,
        random_sign_mask_t=int(0xACE1),
        with_swizzle=False,
    )
    return q_row, s_row, a_row, q_col, s_col, a_col


def _run_fused_dgrad(
    dy_q_row: torch.Tensor,
    dy_sf_row: torch.Tensor,
    dy_amax_row: torch.Tensor,
    w_q_col: torch.Tensor,
    w_sf_col: torch.Tensor,
    w_amax_col: torch.Tensor,
    M: int,
    N: int,
    K: int,
) -> torch.Tensor:
    """dgrad: dX = dY @ W via fused EVT. dY is rowwise quant of (M, N);
    W contributes its columnwise quant (K rows of N FP4 elts = W.T quant
    in raw memory). Kernel sees (m_kernel=M, n_kernel=K, k_kernel=N).

    alpha_a = dY's per-row M-vec amax (output M-axis);
    alpha_b = W's per-column K-vec amax (output K-axis).
    """
    d = torch.empty((M, K), dtype=torch.bfloat16, device=dy_q_row.device)
    tex.nvfp4_cutlass_per_token_gemm(
        dy_q_row,
        w_q_col,
        dy_sf_row.reshape(-1),
        w_sf_col.reshape(-1),
        dy_amax_row,
        w_amax_col,
        d,
        M,
        K,
        N,
        a_sf_swizzled=False,
        b_sf_swizzled=False,
    )
    return d


def _run_fused_wgrad(
    dy_q_col: torch.Tensor,
    dy_sf_col: torch.Tensor,
    dy_amax_col: torch.Tensor,
    x_q_col: torch.Tensor,
    x_sf_col: torch.Tensor,
    x_amax_col: torch.Tensor,
    M: int,
    N: int,
    K: int,
) -> torch.Tensor:
    """wgrad: dW = dY^T @ X via fused EVT. Both operands feed their
    columnwise quant: dY columnwise = (N, M/2) raw FP4 (= dY.T rowwise);
    X columnwise = (K, M/2) raw FP4 (= X.T rowwise). Kernel sees
    (m_kernel=N, n_kernel=K, k_kernel=M).

    alpha_a = dY's per-column N-vec amax (output N-axis);
    alpha_b = X's per-column K-vec amax (output K-axis).
    """
    d = torch.empty((N, K), dtype=torch.bfloat16, device=dy_q_col.device)
    tex.nvfp4_cutlass_per_token_gemm(
        dy_q_col,
        x_q_col,
        dy_sf_col.reshape(-1),
        x_sf_col.reshape(-1),
        dy_amax_col,
        x_amax_col,
        d,
        N,
        K,
        M,
        a_sf_swizzled=False,
        b_sf_swizzled=False,
    )
    return d


def _bf16_dgrad_ground_truth(dy: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """dX = bf16(dY.float() @ W.float()). dy: (M, N), w: (N, K); out: (M, K).
    fp32 intermediates so bf16 mainloop noise doesn't mix with quant noise.
    """
    return (dy.float() @ w.float()).bfloat16()


def _bf16_wgrad_ground_truth(dy: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """dW = bf16(dY.float().T @ X.float()). dy: (M, N), x: (M, K); out: (N, K)."""
    return (dy.float().T @ x.float()).bfloat16()


def _make_grad_quantizer() -> NVFP4Quantizer:
    """Prod bwd GRAD_OUTPUT quantizer. Matches NVFP4BlockScaling defaults:
    fp4_quant_bwd_grad = QParams(RHT=True, SR=True, 2D=False) in
    recipe/__init__.py. RHT lives on the columnwise side (consumed by
    wgrad GEMM); SR is applied during the FP4 round step on grad to avoid
    systematic bias.
    """
    return NVFP4Quantizer(
        fp4_dtype=tex.DType.kFloat4E2M1,
        rowwise=True,
        columnwise=True,
        with_amax_reduction=False,
        amax_reduction_group=None,
        with_rht=True,
        with_post_rht_amax=True,
        with_2d_quantization=False,
        stochastic_rounding=True,
        with_random_sign_mask=True,
    )


def _pten_dgrad(
    dy: torch.Tensor,
    w: torch.Tensor,
    M: int,
    N: int,
    K: int,
) -> torch.Tensor:
    """Prod per-tensor dgrad via general_gemm (linear.py:986). A=W
    (columnwise), B=dY (rowwise), layout='NN' -> output dX shape (M, K).
    Real-ship prod baseline: W with weight quantizer (2D, no RHT/SR);
    dY with grad quantizer (RHT, SR). dgrad consumes dY rowwise (no RHT
    applied to rowwise) and W columnwise; real prod runs exactly this
    pipeline.
    """
    from transformer_engine.pytorch.cpp_extensions import general_gemm

    weight_q = _make_weight_quantizer()
    grad_q = _make_grad_quantizer()
    dst_w = weight_q.make_empty(w.shape, dtype=torch.bfloat16, device=w.device)
    dst_dy = grad_q.make_empty(dy.shape, dtype=torch.bfloat16, device=dy.device)
    tex.quantize(w, weight_q, dst_w, None)
    tex.quantize(dy, grad_q, dst_dy, None)
    del M, N, K
    d, _, _, _ = general_gemm(
        dst_w,
        dst_dy,
        layout="NN",
        grad=True,
        out_dtype=torch.bfloat16,
    )
    return d


def _pten_wgrad(
    dy: torch.Tensor,
    x: torch.Tensor,
    M: int,
    N: int,
    K: int,
) -> torch.Tensor:
    """Prod per-tensor wgrad via general_gemm (linear.py:1159). A=X
    (columnwise), B=dY (columnwise), layout='NT' -> output dW shape
    (N, K). Real-ship prod baseline: X with input quantizer (RHT on
    cols, no SR); dY with grad quantizer (RHT on cols, SR on cast).
    Matching RHT on both columnwise operands cancels in the GEMM
    (H^T H = I) so this is a valid wgrad with reduced FP4 noise.
    """
    from transformer_engine.pytorch.cpp_extensions import general_gemm

    input_q = _make_input_quantizer()
    grad_q = _make_grad_quantizer()
    dst_x = input_q.make_empty(x.shape, dtype=torch.bfloat16, device=x.device)
    dst_dy = grad_q.make_empty(dy.shape, dtype=torch.bfloat16, device=dy.device)
    tex.quantize(x, input_q, dst_x, None)
    tex.quantize(dy, grad_q, dst_dy, None)
    del M, N, K
    d, _, _, _ = general_gemm(
        dst_x,
        dst_dy,
        layout="NT",
        grad=True,
        out_dtype=torch.bfloat16,
    )
    return d


# Bwd shapes: same alignment as fwd (M, N, K % 256 == 0).
_E2E_BWD_SHAPES = [
    (256, 256, 256),
    (512, 512, 512),
    (1024, 1024, 1024),
    (1024, 4096, 4096),
    (2048, 2048, 2048),
]


@_GATED_SM100
@_GATED_HAS_KERNEL
@pytest.mark.parametrize("M,N,K", _E2E_FWD_SHAPES)
def test_e2e_fwd_per_token_vs_per_tensor_snr(M: int, N: int, K: int) -> None:
    """Layer 3 (ship decision basis): side-by-side SNR vs BF16 ground truth.
    Both paths quantize the SAME bf16 input and compare to fp32 GT;
    rel_l2(cf) / rel_l2(pten) < 1.0 = per-token wins on accuracy (per-row
    outer scale > per-tensor outer scale + RHT). Hard assert: both rel_l2
    below the hard floor. Soft (printed): cf/pten ratio.
    """
    device = torch.device("cuda")
    torch.manual_seed(0xACE4)

    a = (torch.randn((M, K), dtype=torch.bfloat16, device=device) * 0.5).contiguous()
    b = (torch.randn((N, K), dtype=torch.bfloat16, device=device) * 0.5).contiguous()

    d_gt = _bf16_gemm_ground_truth(a, b)

    # Path A: prod per-tensor (NVFP4Quantizer recipe defaults + general_gemm).
    d_pten = _pten_e2e(a, b, M, N, K)
    snr_pten = _snr_stats(d_pten, d_gt)

    # Path B: per-token ship target (no RHT + fused-EVT CUTLASS GEMM).
    a_q, a_sf, a_amax = _quantize_per_token(a)
    b_q, b_sf, b_amax = _quantize_per_token(b)
    d_cf = _run_fused(a_q, b_q, a_sf, b_sf, a_amax, b_amax, M, N, K)
    snr_cf = _snr_stats(d_cf, d_gt)

    ratio = snr_cf["rel_l2"] / max(snr_pten["rel_l2"], 1e-12)

    print(
        f"\n  M={M:>5} N={N:>5} K={K:>5}:\n"
        f"    pten:    rel_l2={snr_pten['rel_l2']:.4f}  mean_ratio={snr_pten['mean_ratio']:.4f}"
        f"  mean_abs={snr_pten['mean_abs']:.3e}\n"
        f"    cf:      rel_l2={snr_cf['rel_l2']:.4f}    mean_ratio={snr_cf['mean_ratio']:.4f}"
        f"    mean_abs={snr_cf['mean_abs']:.3e}\n"
        f"    cf/pten: rel_l2={ratio:.3f}x  "
        f"({'per-token wins' if ratio < 1.0 else 'per-tensor wins or tied'})"
    )

    assert snr_pten["rel_l2"] < _LAYER3_REL_L2_HARD_FLOOR, (
        f"per-tensor rel_l2={snr_pten['rel_l2']:.4f} exceeds NVFP4 hard floor "
        f"{_LAYER3_REL_L2_HARD_FLOOR}"
    )
    assert snr_cf["rel_l2"] < _LAYER3_REL_L2_HARD_FLOOR, (
        f"per-token rel_l2={snr_cf['rel_l2']:.4f} exceeds NVFP4 hard floor "
        f"{_LAYER3_REL_L2_HARD_FLOOR}"
    )


@_GATED_SM100
@_GATED_HAS_KERNEL
@pytest.mark.parametrize("M,N,K", _E2E_BWD_SHAPES)
def test_e2e_bwd_per_token_vs_bf16_ground_truth(M: int, N: int, K: int) -> None:
    """E2E backward (per-token quant + fused EVT dgrad/wgrad) vs BF16 fp32
    ground truth. dY (M, N), W (N, K), X (M, K). Two GEMM checks
    side-by-side: dgrad shape (M, K), wgrad shape (N, K). rel_l2 floor 0.30
    (same as fwd Layer 2; per-token NVFP4 noise is K-invariant ~0.13 by
    construction, see fwd analysis).
    """
    device = torch.device("cuda")
    torch.manual_seed(0xACE6)

    dy = (torch.randn((M, N), dtype=torch.bfloat16, device=device) * 0.5).contiguous()
    w = (torch.randn((N, K), dtype=torch.bfloat16, device=device) * 0.5).contiguous()
    x = (torch.randn((M, K), dtype=torch.bfloat16, device=device) * 0.5).contiguous()

    # dgrad: dX = dY @ W; needs dY rowwise + W columnwise.
    dy_qr, dy_sr, dy_ar, _, _, _ = _quantize_per_token_dual(dy)
    _, _, _, w_qc, w_sc, w_ac = _quantize_per_token_dual(w)
    d_dgrad = _run_fused_dgrad(dy_qr, dy_sr, dy_ar, w_qc, w_sc, w_ac, M, N, K)
    snr_dgrad = _snr_stats(d_dgrad, _bf16_dgrad_ground_truth(dy, w))

    # wgrad: dW = dY^T @ X; needs dY columnwise + X columnwise.
    _, _, _, dy_qc, dy_sc, dy_ac = _quantize_per_token_dual(dy)
    _, _, _, x_qc, x_sc, x_ac = _quantize_per_token_dual(x)
    d_wgrad = _run_fused_wgrad(dy_qc, dy_sc, dy_ac, x_qc, x_sc, x_ac, M, N, K)
    snr_wgrad = _snr_stats(d_wgrad, _bf16_wgrad_ground_truth(dy, x))

    print(
        f"\n  M={M:>5} N={N:>5} K={K:>5}:\n"
        f"    dgrad: rel_l2={snr_dgrad['rel_l2']:.4f}  "
        f"mean_ratio={snr_dgrad['mean_ratio']:.4f}  "
        f"max_abs={snr_dgrad['max_abs']:.3e}\n"
        f"    wgrad: rel_l2={snr_wgrad['rel_l2']:.4f}  "
        f"mean_ratio={snr_wgrad['mean_ratio']:.4f}  "
        f"max_abs={snr_wgrad['max_abs']:.3e}"
    )

    assert snr_dgrad["rel_l2"] < 0.30, (
        f"per-token bwd dgrad rel_l2={snr_dgrad['rel_l2']:.4f} exceeds 0.30 "
        "-- accuracy regression in dgrad path."
    )
    assert snr_wgrad["rel_l2"] < 0.30, (
        f"per-token bwd wgrad rel_l2={snr_wgrad['rel_l2']:.4f} exceeds 0.30 "
        "-- accuracy regression in wgrad path."
    )


@_GATED_SM100
@_GATED_HAS_KERNEL
@pytest.mark.parametrize("M,N,K", _E2E_BWD_SHAPES)
def test_e2e_bwd_per_token_vs_per_tensor_snr(M: int, N: int, K: int) -> None:
    """Layer 3 bwd: per-token (no RHT/SR) vs REAL-SHIP prod (RHT+SR+2D
    per NVFP4BlockScaling defaults) side-by-side vs BF16 fp32 GT.
    cf/pten ratio < 1.0 = per-token wins on accuracy even WITHOUT
    RHT/SR/2D help (per-row outer scale alone beats prod's combined
    bag of tricks). Real ship-decision data: prod has every advantage
    enabled; per-token only has its core per-row scale axis.
    """
    device = torch.device("cuda")
    torch.manual_seed(0xACE7)

    dy = (torch.randn((M, N), dtype=torch.bfloat16, device=device) * 0.5).contiguous()
    w = (torch.randn((N, K), dtype=torch.bfloat16, device=device) * 0.5).contiguous()
    x = (torch.randn((M, K), dtype=torch.bfloat16, device=device) * 0.5).contiguous()

    d_dgrad_gt = _bf16_dgrad_ground_truth(dy, w)
    d_wgrad_gt = _bf16_wgrad_ground_truth(dy, x)

    # Path A: prod per-tensor real-ship (general_gemm + NN/NT layout +
    # NVFP4BlockScaling defaults: input RHT, weight 2D, grad RHT+SR).
    d_pten_dgrad = _pten_dgrad(dy, w, M, N, K)
    d_pten_wgrad = _pten_wgrad(dy, x, M, N, K)
    snr_pten_dgrad = _snr_stats(d_pten_dgrad, d_dgrad_gt)
    snr_pten_wgrad = _snr_stats(d_pten_wgrad, d_wgrad_gt)

    # Path B: per-token (fused EVT, no RHT, no SR).
    dy_qr, dy_sr, dy_ar, dy_qc, dy_sc, dy_ac = _quantize_per_token_dual(dy)
    _, _, _, w_qc, w_sc, w_ac = _quantize_per_token_dual(w)
    _, _, _, x_qc, x_sc, x_ac = _quantize_per_token_dual(x)
    d_cf_dgrad = _run_fused_dgrad(dy_qr, dy_sr, dy_ar, w_qc, w_sc, w_ac, M, N, K)
    d_cf_wgrad = _run_fused_wgrad(dy_qc, dy_sc, dy_ac, x_qc, x_sc, x_ac, M, N, K)
    snr_cf_dgrad = _snr_stats(d_cf_dgrad, d_dgrad_gt)
    snr_cf_wgrad = _snr_stats(d_cf_wgrad, d_wgrad_gt)

    ratio_dgrad = snr_cf_dgrad["rel_l2"] / max(snr_pten_dgrad["rel_l2"], 1e-12)
    ratio_wgrad = snr_cf_wgrad["rel_l2"] / max(snr_pten_wgrad["rel_l2"], 1e-12)

    print(
        f"\n  M={M:>5} N={N:>5} K={K:>5}:\n"
        f"    dgrad pten: rel_l2={snr_pten_dgrad['rel_l2']:.4f}  "
        f"cf: rel_l2={snr_cf_dgrad['rel_l2']:.4f}  "
        f"cf/pten={ratio_dgrad:.3f}x "
        f"({'per-token wins' if ratio_dgrad < 1.0 else 'per-tensor wins or tied'})\n"
        f"    wgrad pten: rel_l2={snr_pten_wgrad['rel_l2']:.4f}  "
        f"cf: rel_l2={snr_cf_wgrad['rel_l2']:.4f}  "
        f"cf/pten={ratio_wgrad:.3f}x "
        f"({'per-token wins' if ratio_wgrad < 1.0 else 'per-tensor wins or tied'})"
    )

    assert (
        snr_pten_dgrad["rel_l2"] < _LAYER3_REL_L2_HARD_FLOOR
    ), f"per-tensor dgrad rel_l2={snr_pten_dgrad['rel_l2']:.4f} > floor."
    assert (
        snr_cf_dgrad["rel_l2"] < _LAYER3_REL_L2_HARD_FLOOR
    ), f"per-token dgrad rel_l2={snr_cf_dgrad['rel_l2']:.4f} > floor."
    assert (
        snr_pten_wgrad["rel_l2"] < _LAYER3_REL_L2_HARD_FLOOR
    ), f"per-tensor wgrad rel_l2={snr_pten_wgrad['rel_l2']:.4f} > floor."
    assert (
        snr_cf_wgrad["rel_l2"] < _LAYER3_REL_L2_HARD_FLOOR
    ), f"per-token wgrad rel_l2={snr_cf_wgrad['rel_l2']:.4f} > floor."


if __name__ == "__main__":
    if not _has_sm100():
        print("SKIP: not SM100")
    elif not hasattr(tex, "nvfp4_cutlass_per_token_gemm"):
        print("SKIP: kernel not built")
    else:
        for shape in _SHAPES:
            test_fused_matches_cublaslt_per_token(*shape)
        test_fused_alpha_unity_matches_scalar_gemm_with_baked_const()
        test_prod_fwd_weight_2d_quant_plumbing()
        for shape in _E2E_FWD_SHAPES:
            test_e2e_fwd_per_token_vs_bf16_ground_truth(*shape)
        for shape in _E2E_FWD_SHAPES:
            test_e2e_fwd_per_token_vs_per_tensor_snr(*shape)
        for shape in _E2E_BWD_SHAPES:
            test_e2e_bwd_per_token_vs_bf16_ground_truth(*shape)
        for shape in _E2E_BWD_SHAPES:
            test_e2e_bwd_per_token_vs_per_tensor_snr(*shape)
        print("All tests passed.")
