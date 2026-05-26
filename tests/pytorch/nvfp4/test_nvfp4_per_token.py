# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Correctness tests for NVFP4 per-token cast + cuBLAS LT NVFP4 GEMM.

Covers byte-equal kernel-vs-reference quantize parity, K1/K2 split-vs-composite
parity, dequant + fp32 reference, and a cuBLAS LT NVFP4 GEMM smoke. Requires
bf16 input, M % 128 == 0, K % 128 == 0; GEMM tests gated by SM100.
"""

from __future__ import annotations

import pytest
import torch

# Must import transformer_engine first to dlopen libtransformer_engine.so so
# transformer_engine_torch.so can resolve typeinfo / vtable symbols at load time.
import transformer_engine.pytorch as te  # noqa: F401
import transformer_engine_torch as tex  # type: ignore  # noqa: F401

from transformer_engine.pytorch.custom_recipes.gemm_nvfp4_per_token import (
    dequantize_nvfp4_per_token,
    nvfp4_per_token_gemm,
    nvfp4_per_token_gemm_dequant,
)
from transformer_engine.pytorch.custom_recipes.quantization_nvfp4_per_token import (
    BLOCK_K,
    NVFP4QuantizerPerTokenRef,
    nvfp4_per_token_amax,
    nvfp4_per_token_encode,
    nvfp4_per_token_quantize,
)


def _has_sm100() -> bool:
    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability()
    return major >= 10


_GATED_SM100 = pytest.mark.skipif(
    not _has_sm100(),
    reason="NVFP4 per-token GEMM via cuBLAS LT requires SM100 (Blackwell).",
)

_GATED_FP4 = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="NVFP4 per-token cast requires CUDA.",
)


# (1) Quantize parity: kernel vs Python reference.

# Shapes obey the kernel contract (M % 128 == 0, K % 128 == 0).
_QUANT_SHAPES = [
    (128, 128),  # smallest legal shape
    (128, 256),  # K > inner SF window of single chunk
    (256, 128),  # M > inner SF window of single chunk
    (256, 512),
    (512, 1024),
]


def _unpack_fp4_byte_pairs(x: torch.Tensor) -> torch.Tensor:
    """Unpack two FP4 values per byte into one uint8 nibble per element."""
    repeated = x.repeat_interleave(2, dim=1)
    repeated[:, 0::2] &= 0x0F
    repeated[:, 1::2] >>= 4
    return repeated


@_GATED_FP4
@pytest.mark.parametrize("M,N", _QUANT_SHAPES)
@pytest.mark.parametrize("rowwise,columnwise", [(True, False), (False, True), (True, True)])
def test_per_token_quantize_byte_exact(M: int, N: int, rowwise: bool, columnwise: bool) -> None:
    """Composite per-token output is byte-equal to the Python reference."""
    torch.manual_seed(0xBEEF * (M + 17) + (N + 3))
    device = torch.device("cuda")
    x = torch.randn((M, N), dtype=torch.bfloat16, device=device) * 4.0
    # Outliers so the per-row outer is exercised.
    if M >= 4:
        x[0, :] *= 8.0
        x[-1, :] *= 0.125

    ref = NVFP4QuantizerPerTokenRef(rowwise=rowwise, columnwise=columnwise).quantize(x)
    sut = nvfp4_per_token_quantize(x, rowwise=rowwise, columnwise=columnwise)

    if rowwise:
        qx_sut = _unpack_fp4_byte_pairs(sut.data.view(torch.uint8))
        qx_ref = _unpack_fp4_byte_pairs(ref.data.view(torch.uint8))
        torch.testing.assert_close(qx_sut, qx_ref, atol=0.0, rtol=0.0)
        torch.testing.assert_close(
            sut.scale.view(torch.uint8),
            ref.scale.view(torch.uint8),
            atol=0.0,
            rtol=0.0,
        )
        torch.testing.assert_close(sut.row_amax, ref.row_amax, atol=0.0, rtol=0.0)

    if columnwise:
        qxt_sut = _unpack_fp4_byte_pairs(sut.columnwise_data.view(torch.uint8))
        qxt_ref = _unpack_fp4_byte_pairs(ref.columnwise_data.view(torch.uint8))
        torch.testing.assert_close(qxt_sut, qxt_ref, atol=0.0, rtol=0.0)
        torch.testing.assert_close(
            sut.columnwise_scale.view(torch.uint8),
            ref.columnwise_scale.view(torch.uint8),
            atol=0.0,
            rtol=0.0,
        )
        torch.testing.assert_close(sut.col_amax, ref.col_amax, atol=0.0, rtol=0.0)


# (2) Split-kernel parity: K1 then K2 == composite K1+K2.


@_GATED_FP4
@pytest.mark.parametrize("M,N", _QUANT_SHAPES)
@pytest.mark.parametrize("rowwise,columnwise", [(True, False), (False, True), (True, True)])
def test_per_token_split_byte_equal(
    M: int,
    N: int,
    rowwise: bool,
    columnwise: bool,
) -> None:
    """K1 (amax) then K2 (encode) byte-equals the composite K1+K2."""
    torch.manual_seed(0xC0FFEE * (M + 7) + (N + 11))
    device = torch.device("cuda")
    x = torch.randn((M, N), dtype=torch.bfloat16, device=device) * 4.0
    if M >= 4:
        x[0, :] *= 8.0
        x[-1, :] *= 0.125

    composite = nvfp4_per_token_quantize(x, rowwise=rowwise, columnwise=columnwise)

    row_amax, col_amax = nvfp4_per_token_amax(
        x,
        rowwise=rowwise,
        columnwise=columnwise,
    )
    split = nvfp4_per_token_encode(
        x,
        row_amax=row_amax,
        col_amax=col_amax,
        rowwise=rowwise,
        columnwise=columnwise,
    )

    if rowwise:
        torch.testing.assert_close(split.row_amax, composite.row_amax, atol=0.0, rtol=0.0)
        torch.testing.assert_close(
            split.data.view(torch.uint8),
            composite.data.view(torch.uint8),
            atol=0.0,
            rtol=0.0,
        )
        torch.testing.assert_close(
            split.scale.view(torch.uint8),
            composite.scale.view(torch.uint8),
            atol=0.0,
            rtol=0.0,
        )
    if columnwise:
        torch.testing.assert_close(split.col_amax, composite.col_amax, atol=0.0, rtol=0.0)
        torch.testing.assert_close(
            split.columnwise_data.view(torch.uint8),
            composite.columnwise_data.view(torch.uint8),
            atol=0.0,
            rtol=0.0,
        )
        torch.testing.assert_close(
            split.columnwise_scale.view(torch.uint8),
            composite.columnwise_scale.view(torch.uint8),
            atol=0.0,
            rtol=0.0,
        )


# (2b) Input-validation rejections.


@_GATED_FP4
def test_per_token_validation_rejects_fp32() -> None:
    """Per-token must ``ValueError`` on non-bf16 input (no fallback path)."""
    device = torch.device("cuda")
    x = torch.randn((128, 128), dtype=torch.float32, device=device)
    with pytest.raises(ValueError, match="bf16"):
        nvfp4_per_token_quantize(x, rowwise=True, columnwise=False)


@_GATED_FP4
def test_per_token_validation_rejects_unaligned() -> None:
    """Per-token must ``ValueError`` on M or K not 128-aligned."""
    device = torch.device("cuda")
    x = torch.randn((128, 64), dtype=torch.bfloat16, device=device)
    with pytest.raises(ValueError, match="K % 128"):
        nvfp4_per_token_quantize(x, rowwise=True, columnwise=False)

    x2 = torch.randn((64, 128), dtype=torch.bfloat16, device=device)
    with pytest.raises(ValueError, match="M % 128"):
        nvfp4_per_token_quantize(x2, rowwise=True, columnwise=False)


# (3) Dequant + fp32 reference matmul sanity (pure-Python, no kernel).


@_GATED_FP4
@pytest.mark.parametrize("M,N", [(32, 64), (64, 256)])
def test_per_token_dequant_roundtrip_close(M: int, N: int) -> None:
    """``dequantize(quantize(x)) ~ x`` at FP4 quantization precision."""
    torch.manual_seed(0x1234)
    device = torch.device("cuda")
    x = torch.randn((M, N), dtype=torch.float32, device=device)

    ref = NVFP4QuantizerPerTokenRef(rowwise=True).quantize(x)
    y = dequantize_nvfp4_per_token(ref.data, ref.scale, ref.row_amax)

    # Loose bound: catches dequant-formula bugs, not quantization quality.
    rel = (y - x).abs() / x.abs().clamp(min=1e-6)
    assert rel.mean().item() < 0.5, f"mean rel error {rel.mean().item():.3g} > 0.5"


# (4) Production GEMM: cuBLAS LT NVFP4 + post-scale composite.
# Shapes need M, N % 128 == 0 and K % 16 == 0 for cuBLAS LT NVFP4.
_GEMM_SHAPES = [
    (128, 128, 128),  # smallest legal shape
    (128, 128, 256),  # exercise K > inner SF window
    (256, 128, 256),  # non-square (M != N)
    (256, 256, 256),  # square mid-size
]


def _three_pronged_bf16_close(
    d_test: torch.Tensor,
    d_ref: torch.Tensor,
    *,
    label: str,
    rel_l2_floor: float = 2e-2,
    bad_count_ratio: float = 1e-2,
    atol: float = 1e-1,
    bad_rtol: float = 5e-2,
) -> None:
    """Dequant-vs-SUT closeness for random GEMM outputs.

    Three-pronged: energy-weighted rel_l2 (primary), torch.allclose-style
    n_bad_mixed (localised faults), max_abs (NaN-like blow-up sanity).
    """
    finite_mask = torch.isfinite(d_test) & torch.isfinite(d_ref)
    d_t = d_test.float()[finite_mask]
    d_r = d_ref.float()[finite_mask]
    diff = (d_t - d_r).abs()
    n = d_t.numel()

    diff_l2 = float(diff.norm().item())
    ref_l2 = float(d_r.norm().item())
    rel_l2 = diff_l2 / (ref_l2 + 1e-30)

    n_bad_mixed = int((diff > atol + bad_rtol * d_r.abs()).sum().item())

    max_abs = float(diff.max().item()) if n else float("nan")
    mean_ref_abs = float(d_r.abs().mean().item()) if n else float("nan")
    max_abs_bound = atol + bad_rtol * mean_ref_abs

    rel = diff / d_r.abs().clamp(min=1e-30)
    mean_rel = float(rel.mean().item()) if n else float("nan")
    max_rel = float(rel.max().item()) if n else float("nan")

    diag = (
        f"[{label}] N_finite={n}/{int(finite_mask.numel())} "
        f"rel_l2={rel_l2:.3g} max_abs={max_abs:.3g} n_bad_mixed={n_bad_mixed} "
        f"mean_|d_ref|={mean_ref_abs:.3g} "
        f"(diag: mean_rel={mean_rel:.3g} max_rel={max_rel:.3g} "
        "— mean_rel/max_rel are NOT asserted; see helper docstring)"
    )
    print(diag)

    bad_count_abs_floor = max(8, int(bad_count_ratio * n))
    assert rel_l2 <= rel_l2_floor, (
        f"{diag} -> rel_l2 > {rel_l2_floor} (energy-weighted global "
        "relative error too high — possible structural bug)"
    )
    assert n_bad_mixed <= bad_count_abs_floor, (
        f"{diag} -> n_bad_mixed > {bad_count_abs_floor} "
        f"(|diff| > atol={atol} + rtol={bad_rtol} * |d_r| for too "
        "many elements — possible localised broken row/col)"
    )
    assert max_abs <= max_abs_bound, (
        f"{diag} -> max_abs > {max_abs_bound:.3g} = atol + "
        "bad_rtol * mean_|d_ref| (worst element is way outside the "
        "noise envelope — possible NaN-like blow-up)"
    )


@_GATED_SM100
@pytest.mark.parametrize("M,N,K", _GEMM_SHAPES)
def test_per_token_gemm_close_to_bf16(M: int, N: int, K: int) -> None:
    """End-to-end per_token_gemm is structurally close to BF16 GEMM.

    Uses cos_sim + magnitude-ratio (direction + magnitude) instead of
    per-element mean_rel, which is pathological on random GEMM outputs.
    """
    torch.manual_seed(0xACE * M + K)
    device = torch.device("cuda")
    a = torch.randn((M, K), dtype=torch.bfloat16, device=device)
    b = torch.randn((N, K), dtype=torch.bfloat16, device=device)

    a_q = nvfp4_per_token_quantize(a, rowwise=True)
    b_q = nvfp4_per_token_quantize(b, rowwise=True)

    d_sut = nvfp4_per_token_gemm(
        a_q.data,
        a_q.scale,
        a_q.row_amax,
        b_q.data,
        b_q.scale,
        b_q.row_amax,
    )

    d_ref = (a.float() @ b.float().t()).to(torch.bfloat16)

    d_sut_f = d_sut.float().flatten()
    d_ref_f = d_ref.float().flatten()

    sut_norm = d_sut_f.norm()
    ref_norm = d_ref_f.norm()
    cos_sim = float((d_sut_f @ d_ref_f) / (sut_norm * ref_norm + 1e-30))
    mag_ratio = float(sut_norm / (ref_norm + 1e-30))

    # cos_sim >= 0.95 catches operand swap; mag in [0.7, 1.3] catches
    # missing/duplicated scale or wrong alpha-by-constant.
    cos_sim_floor = 0.95
    mag_lo, mag_hi = 0.7, 1.3

    diag = (
        f"[per_token({M}x{N}x{K})] cos_sim={cos_sim:.4f} "
        f"mag_ratio={mag_ratio:.4f} "
        f"||d_sut||={float(sut_norm):.4g} ||d_ref||={float(ref_norm):.4g}"
    )
    assert cos_sim >= cos_sim_floor, (
        f"{diag} -> cos_sim < {cos_sim_floor} (structural mismatch; "
        "likely wrong operand swap, missing scale, or indexing bug)"
    )
    assert mag_lo <= mag_ratio <= mag_hi, (
        f"{diag} -> mag_ratio not in [{mag_lo}, {mag_hi}] "
        "(systematic magnitude error; check alpha/post-scale)"
    )


@_GATED_SM100
@pytest.mark.parametrize("M,N,K", _GEMM_SHAPES)
def test_per_token_gemm_close_to_dequant_ref(M: int, N: int, K: int) -> None:
    """End-to-end per_token_gemm close to dequant + fp32 matmul (TF32 envelope)."""
    torch.manual_seed(0xDEAD * (M + 7) + (N + 1) * K)
    device = torch.device("cuda")
    a = torch.randn((M, K), dtype=torch.bfloat16, device=device) * 0.5
    b = torch.randn((N, K), dtype=torch.bfloat16, device=device) * 0.5

    a_q = nvfp4_per_token_quantize(a, rowwise=True)
    b_q = nvfp4_per_token_quantize(b, rowwise=True)

    d_sut = nvfp4_per_token_gemm(
        a_q.data,
        a_q.scale,
        a_q.row_amax,
        b_q.data,
        b_q.scale,
        b_q.row_amax,
    ).float()

    d_ref = nvfp4_per_token_gemm_dequant(
        a_q.data,
        a_q.scale,
        a_q.row_amax,
        b_q.data,
        b_q.scale,
        b_q.row_amax,
        out_dtype=torch.float32,
    )

    _three_pronged_bf16_close(
        d_sut,
        d_ref,
        label=f"vs_dequant({M}x{N}x{K})",
        # Empirical rel_l2 ~5e-3..1.5e-2 on random N(0, 0.5), K=128-256.
        rel_l2_floor=2e-2,
        atol=1e-1,
        bad_rtol=5e-2,
        bad_count_ratio=1e-2,
    )


@_GATED_SM100
def test_per_token_gemm_rejects_beta_nonzero() -> None:
    """beta != 0 raises until residual handling is added."""
    device = torch.device("cuda")
    M, N, K = 128, 128, 128
    a = torch.randn((M, K), dtype=torch.bfloat16, device=device)
    b = torch.randn((N, K), dtype=torch.bfloat16, device=device)
    a_q = nvfp4_per_token_quantize(a, rowwise=True)
    b_q = nvfp4_per_token_quantize(b, rowwise=True)

    with pytest.raises(ValueError, match=r"beta != 0"):
        nvfp4_per_token_gemm(
            a_q.data,
            a_q.scale,
            a_q.row_amax,
            b_q.data,
            b_q.scale,
            b_q.row_amax,
            beta=1.0,
        )
