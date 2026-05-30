# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Numerical tests for tex.nvfp4_cutlass_per_token_gemm (fused EVT) vs the
cuBLAS-LT per-token reference (GEMM + standalone post_scale). M, N, K multiples
of 256; rtol=2e-2 ~ 2.5x bf16 ULP for fp32 reduction-order noise."""

from __future__ import annotations

from typing import Tuple

import pytest
import torch

# Must import transformer_engine first to dlopen libtransformer_engine.so.
import transformer_engine.pytorch as te  # noqa: F401
import transformer_engine_torch as tex  # type: ignore


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


if __name__ == "__main__":
    if not _has_sm100():
        print("SKIP: not SM100")
    elif not hasattr(tex, "nvfp4_cutlass_per_token_gemm"):
        print("SKIP: kernel not built")
    else:
        for shape in _SHAPES:
            test_fused_matches_cublaslt_per_token(*shape)
        test_fused_alpha_unity_matches_scalar_gemm_with_baked_const()
        print("All tests passed.")
