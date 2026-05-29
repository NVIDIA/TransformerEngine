# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Correctness tests for NVFP4 per-token cast + cuBLAS LT NVFP4 GEMM.

Covers byte-equal kernel-vs-reference quantize parity, K1/K2 split-vs-composite
parity, dequant + fp32 reference, optional RHT (K1 amax + K2 cast), and a
cuBLAS LT NVFP4 GEMM smoke. Requires bf16 input, M % 128 == 0, K % 128 == 0;
GEMM and RHT tests gated by SM100.
"""

from __future__ import annotations

from typing import Tuple

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


# =============================================================================
# (5) RHT correctness: K1 amax + K2 cast with optional col-wise RHT.
# Opt-in via with_rht=True + random_sign_mask_t=<u16>; row direction never
# sees RHT. with_rht=False is byte-equal to the pre-RHT path.
# =============================================================================

_RHT_SHAPES = [
    (128, 128),
    (256, 256),
    (128, 1024),  # K > single 64x64 sub-tile along col
    (1024, 128),  # M > single 64x64 sub-tile along row
    (512, 512),
]


def _walsh_hadamard_16(device: torch.device) -> torch.Tensor:
    """16x16 Sylvester / Walsh-Hadamard matrix, +/-1 entries (unnormalized)."""
    H = torch.tensor([[1.0]], dtype=torch.float32, device=device)
    for _ in range(4):
        top = torch.cat([H, H], dim=1)
        bot = torch.cat([H, -H], dim=1)
        H = torch.cat([top, bot], dim=0)
    return H


def _sign_diag_16(mask: int, device: torch.device) -> torch.Tensor:
    """16-elt +/-1 vector; s_i = -1 iff bit i of `mask` is set."""
    bits = torch.tensor(
        [1 - 2 * ((mask >> i) & 1) for i in range(16)],
        dtype=torch.float32,
        device=device,
    )
    return bits


def _reference_col_amax_rht(x_bf16: torch.Tensor, mask: int) -> torch.Tensor:
    """PyTorch reference for the per-token col-wise RHT amax: max over
    16-row blocks of |H * D * x_block| / 4. FHT may permute element order
    but |y|.max() is permutation-invariant.
    """
    M, K = x_bf16.shape
    assert M % 16 == 0, "Test setup error: M must be a multiple of 16."
    H = _walsh_hadamard_16(x_bf16.device)
    sign = _sign_diag_16(mask, x_bf16.device)
    x = x_bf16.to(torch.float32)
    blocks = x.reshape(M // 16, 16, K)
    masked = blocks * sign.view(1, 16, 1)
    rotated = torch.einsum("ij,bjk->bik", H, masked)
    return (rotated.abs() / 4.0).reshape(-1, K).amax(dim=0)


def _reference_amax_raw(x_bf16: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Raw per-row + per-col absolute max (no RHT, bf16 -> fp32 first)."""
    x = x_bf16.to(torch.float32)
    return x.abs().amax(dim=1), x.abs().amax(dim=0)


def _allocate_per_token_buffers(M: int, K: int, device: torch.device):
    """Match the layout that ``tex.nvfp4_per_token_quantize`` writes."""
    return {
        "q_row": torch.empty((M, K // 2), dtype=torch.uint8, device=device),
        "s_row": torch.empty((M, K // BLOCK_K), dtype=torch.uint8, device=device),
        "ra": torch.empty((M,), dtype=torch.float32, device=device),
        "q_col": torch.empty((K, M // 2), dtype=torch.uint8, device=device),
        "s_col": torch.empty((K, M // BLOCK_K), dtype=torch.uint8, device=device),
        "ca": torch.empty((K,), dtype=torch.float32, device=device),
    }


def _dequant_fp4_with_outer_amax(
    q_packed: torch.Tensor,  # (R, C // 2) uint8 packed FP4
    s_dec: torch.Tensor,  # (R, C // 16) e4m3 held as uint8
    outer_amax: torch.Tensor,  # (R,) fp32
) -> torch.Tensor:
    """Decode a rowwise FP4 tensor back to fp32 using the kernel's own
    arithmetic: x_hat = qcode * s_dec_e4m3 * (6 / S_enc_row),
    S_enc_row = (448 * 6) / max(outer_amax, 1e-12).
    """
    R, half_C = q_packed.shape
    C = half_C * 2
    s_dec_f = s_dec.view(torch.float8_e4m3fn).to(torch.float32)

    lo = (q_packed & 0x0F).to(torch.int8)
    hi = ((q_packed >> 4) & 0x0F).to(torch.int8)
    interleaved = torch.stack([lo, hi], dim=-1).reshape(R, C)
    # NVFP4 E2M1 LUT (sign-magnitude): 0000..0111 map to {0, 0.5, 1, 1.5,
    # 2, 3, 4, 6}; 1000..1111 are the negatives.
    fp4_lut = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
        dtype=torch.float32,
        device=q_packed.device,
    )
    fp4_val = fp4_lut[interleaved.to(torch.int64)]

    fp8_max = 448.0
    fp4_max = 6.0
    safe_amax = torch.clamp(outer_amax, min=1e-12)
    S_enc_row = (fp8_max * fp4_max) / safe_amax
    inv_S = (1.0 / S_enc_row).unsqueeze(1)

    block_scale_inv = s_dec_f * inv_S
    block_scale_inv = block_scale_inv.repeat_interleave(BLOCK_K, dim=1)

    return fp4_val * block_scale_inv


# ----- (5a) K1 RHT: standalone amax kernel ----------------------------------


@_GATED_SM100
@pytest.mark.parametrize("M,K", _RHT_SHAPES)
def test_per_token_k1_with_rht_false_equals_raw_amax(M: int, K: int) -> None:
    """Regression: with_rht=False reproduces raw bf16->fp32 amax along each axis."""
    torch.manual_seed(0xABCD * (M + 1) + K)
    device = torch.device("cuda")
    x = torch.randn((M, K), dtype=torch.bfloat16, device=device)

    row_amax = torch.empty((M,), dtype=torch.float32, device=device)
    col_amax = torch.empty((K,), dtype=torch.float32, device=device)

    tex.nvfp4_per_token_amax(
        x,
        row_amax,
        col_amax,
        True,
        True,
        with_rht=False,
        random_sign_mask_t=0,
    )

    ref_row, ref_col = _reference_amax_raw(x)
    torch.testing.assert_close(
        row_amax, ref_row, rtol=0.0, atol=0.0, msg=f"row_amax mismatch at ({M}, {K})"
    )
    torch.testing.assert_close(
        col_amax, ref_col, rtol=0.0, atol=0.0, msg=f"col_amax mismatch at ({M}, {K})"
    )


@_GATED_SM100
@pytest.mark.parametrize("M,K", _RHT_SHAPES)
@pytest.mark.parametrize("mask", [0x0000, 0xACE1, 0xFFFF, 0x5A5A])
def test_per_token_k1_with_rht_matches_reference(
    M: int,
    K: int,
    mask: int,
) -> None:
    """with_rht=True col_amax matches max|H*D*x_block|/4; rowwise stays raw."""
    torch.manual_seed(0xDEAD * (M + 7) + (K + 3) + mask)
    device = torch.device("cuda")
    x = torch.randn((M, K), dtype=torch.bfloat16, device=device)

    row_amax = torch.empty((M,), dtype=torch.float32, device=device)
    col_amax = torch.empty((K,), dtype=torch.float32, device=device)

    tex.nvfp4_per_token_amax(
        x,
        row_amax,
        col_amax,
        True,
        True,
        with_rht=True,
        random_sign_mask_t=mask,
    )

    ref_row, _ = _reference_amax_raw(x)
    torch.testing.assert_close(
        row_amax,
        ref_row,
        rtol=0.0,
        atol=0.0,
        msg=f"row_amax mismatch at ({M}, {K}, mask=0x{mask:04X})",
    )

    # Col tolerance accounts for bf16->fp32 promotion noise + butterfly
    # summation order vs. einsum reduction order.
    ref_col = _reference_col_amax_rht(x, mask)
    torch.testing.assert_close(
        col_amax,
        ref_col,
        rtol=2e-3,
        atol=1e-4,
        msg=f"col_amax (RHT) mismatch at ({M}, {K}, mask=0x{mask:04X})",
    )


@_GATED_SM100
@pytest.mark.parametrize("M,K", [(128, 128), (256, 512)])
def test_per_token_k1_with_rht_zero_mask_is_hadamard_only(M: int, K: int) -> None:
    """mask=0 -> D=I; col_amax equals bare Hadamard amax max|H*x_block|/4."""
    torch.manual_seed(0xC0DE * (M + 11) + K)
    device = torch.device("cuda")
    x = torch.randn((M, K), dtype=torch.bfloat16, device=device)

    row_amax = torch.empty((M,), dtype=torch.float32, device=device)
    col_amax = torch.empty((K,), dtype=torch.float32, device=device)

    tex.nvfp4_per_token_amax(
        x,
        row_amax,
        col_amax,
        True,
        True,
        with_rht=True,
        random_sign_mask_t=0,
    )

    H = _walsh_hadamard_16(device)
    x_fp32 = x.to(torch.float32)
    blocks = x_fp32.reshape(M // 16, 16, K)
    rotated = torch.einsum("ij,bjk->bik", H, blocks)
    ref_col = (rotated.abs() / 4.0).reshape(-1, K).amax(dim=0)

    torch.testing.assert_close(
        col_amax,
        ref_col,
        rtol=2e-3,
        atol=1e-4,
        msg=f"col_amax (RHT, mask=0) mismatch at ({M}, {K})",
    )


# ----- (5b) K2 + composite RHT: encode kernel and composite quantize --------


@_GATED_SM100
@pytest.mark.parametrize("M,K", _RHT_SHAPES)
def test_per_token_composite_with_rht_false_byte_equal(M: int, K: int) -> None:
    """Regression: with_rht=False composite byte-equals the default (no-kwargs) path."""
    torch.manual_seed(0xCAFE * (M + 1) + K)
    device = torch.device("cuda")
    x = torch.randn((M, K), dtype=torch.bfloat16, device=device)

    bufs_default = _allocate_per_token_buffers(M, K, device)
    bufs_explicit = _allocate_per_token_buffers(M, K, device)

    tex.nvfp4_per_token_quantize(
        x,
        bufs_default["q_row"],
        bufs_default["s_row"],
        bufs_default["ra"],
        bufs_default["q_col"],
        bufs_default["s_col"],
        bufs_default["ca"],
        True,
        True,
    )
    tex.nvfp4_per_token_quantize(
        x,
        bufs_explicit["q_row"],
        bufs_explicit["s_row"],
        bufs_explicit["ra"],
        bufs_explicit["q_col"],
        bufs_explicit["s_col"],
        bufs_explicit["ca"],
        True,
        True,
        with_rht=False,
        random_sign_mask_t=0xACE1,
    )

    for k in ("q_row", "s_row", "ra", "q_col", "s_col", "ca"):
        assert torch.equal(
            bufs_default[k], bufs_explicit[k]
        ), f"with_rht=False not byte-equal to default path on `{k}` at ({M}, {K})"


@_GATED_SM100
@pytest.mark.parametrize("M,K", _RHT_SHAPES)
def test_per_token_composite_rowwise_unchanged_under_rht(M: int, K: int) -> None:
    """Rowwise FP4 + inner SF + row amax byte-equal across with_rht=False / True."""
    torch.manual_seed(0xBEEF * (M + 3) + K)
    device = torch.device("cuda")
    x = torch.randn((M, K), dtype=torch.bfloat16, device=device)

    bufs_no_rht = _allocate_per_token_buffers(M, K, device)
    bufs_with_rht = _allocate_per_token_buffers(M, K, device)

    tex.nvfp4_per_token_quantize(
        x,
        bufs_no_rht["q_row"],
        bufs_no_rht["s_row"],
        bufs_no_rht["ra"],
        bufs_no_rht["q_col"],
        bufs_no_rht["s_col"],
        bufs_no_rht["ca"],
        True,
        True,
        with_rht=False,
        random_sign_mask_t=0,
    )
    tex.nvfp4_per_token_quantize(
        x,
        bufs_with_rht["q_row"],
        bufs_with_rht["s_row"],
        bufs_with_rht["ra"],
        bufs_with_rht["q_col"],
        bufs_with_rht["s_col"],
        bufs_with_rht["ca"],
        True,
        True,
        with_rht=True,
        random_sign_mask_t=0xACE1,
    )

    for k in ("q_row", "s_row", "ra"):
        assert torch.equal(bufs_no_rht[k], bufs_with_rht[k]), (
            f"rowwise output differs between with_rht=False/True on `{k}` "
            f"at ({M}, {K}) -- rowwise should never see RHT."
        )


@_GATED_SM100
@pytest.mark.parametrize("M,K", [(128, 128), (256, 512), (512, 512)])
@pytest.mark.parametrize("mask", [0x0000, 0xACE1, 0xFFFF])
def test_per_token_composite_with_rht_col_dequant_matches_reference(
    M: int,
    K: int,
    mask: int,
) -> None:
    """Dequant'd col FP4 (with_rht=True) ~ H*D*x_block/sqrt(16); checks
    column-aggregate median + p99 relative error (FP4's 16-code grain and
    butterfly permutation make element-wise comparison too loose).
    """
    torch.manual_seed(0xFEED * (M + 5) + K + mask)
    device = torch.device("cuda")
    # Scale down so most blocks land in non-saturating FP4 (else we measure
    # clamping noise, not RHT).
    x = torch.randn((M, K), dtype=torch.bfloat16, device=device) * 0.5

    bufs = _allocate_per_token_buffers(M, K, device)
    tex.nvfp4_per_token_quantize(
        x,
        bufs["q_row"],
        bufs["s_row"],
        bufs["ra"],
        bufs["q_col"],
        bufs["s_col"],
        bufs["ca"],
        True,
        True,
        with_rht=True,
        random_sign_mask_t=mask,
    )

    H = _walsh_hadamard_16(device)
    sign = _sign_diag_16(mask, device)
    x_fp32 = x.to(torch.float32)
    blocks = x_fp32.reshape(M // 16, 16, K)
    masked = blocks * sign.view(1, 16, 1)
    rotated = torch.einsum("ij,bjk->bik", H, masked)  # (M/16, 16, K)
    y_ref = rotated.reshape(M, K) / 4.0  # (M, K)
    y_ref_col_view = y_ref.transpose(0, 1).contiguous()  # (K, M)

    y_kernel = _dequant_fp4_with_outer_amax(
        bufs["q_col"],
        bufs["s_col"],
        bufs["ca"],
    )  # (K, M)

    diff = (y_kernel - y_ref_col_view).abs()
    col_outer = bufs["ca"].unsqueeze(1).clamp(min=1e-6)
    rel = diff / col_outer
    p99 = torch.quantile(rel.flatten(), 0.99).item()
    median = rel.median().item()
    assert median < 0.1, (
        f"median per-element relative error too large: {median:.4f} > 0.1 "
        f"at ({M}, {K}, mask=0x{mask:04X})"
    )
    assert (
        p99 < 0.5
    ), f"p99 per-element relative error too large: {p99:.4f} > 0.5 at ({M}, {K}, mask=0x{mask:04X})"


@_GATED_SM100
@pytest.mark.parametrize("M,K", [(128, 128), (256, 256)])
def test_per_token_composite_with_rht_col_amax_matches_k1(
    M: int,
    K: int,
) -> None:
    """Composite col_amax byte-equals standalone K1 amax with the same mask."""
    torch.manual_seed(0xDADA * (M + 13) + K)
    device = torch.device("cuda")
    x = torch.randn((M, K), dtype=torch.bfloat16, device=device)
    mask = 0xACE1

    bufs = _allocate_per_token_buffers(M, K, device)
    tex.nvfp4_per_token_quantize(
        x,
        bufs["q_row"],
        bufs["s_row"],
        bufs["ra"],
        bufs["q_col"],
        bufs["s_col"],
        bufs["ca"],
        True,
        True,
        with_rht=True,
        random_sign_mask_t=mask,
    )

    ra_k1 = torch.empty((M,), dtype=torch.float32, device=device)
    ca_k1 = torch.empty((K,), dtype=torch.float32, device=device)
    tex.nvfp4_per_token_amax(
        x,
        ra_k1,
        ca_k1,
        True,
        True,
        with_rht=True,
        random_sign_mask_t=mask,
    )

    torch.testing.assert_close(
        bufs["ca"], ca_k1, rtol=0.0, atol=0.0, msg=f"composite ca != K1-only ca at ({M}, {K})"
    )
    torch.testing.assert_close(
        bufs["ra"], ra_k1, rtol=0.0, atol=0.0, msg=f"composite ra != K1-only ra at ({M}, {K})"
    )


# =============================================================================
# (6) Fused-swizzle correctness: K2 with_swizzle=True emits rowwise SF in
# cuBLAS LT layout. Tests cover byte-equal vs Python reference, other-outputs
# identical to with_swizzle=False, and GEMM fast-path numerical equivalence.
# =============================================================================

_SWIZZLE_SHAPES = [
    (128, 128),
    (256, 256),
    (512, 512),
    (256, 1024),
    (1024, 256),
]


def _swizzle_sf_reference(sf_m_major: torch.Tensor) -> torch.Tensor:
    """Reference M-major (M, K_SF) e4m3 -> cuBLAS LT swizzled flat bytes
    (128Mx4K tile, 16-byte slot = 4 M-stripes x 4 K-bytes stripe-major)."""
    M, K_SF = sf_m_major.shape
    assert M % 128 == 0
    assert K_SF % 4 == 0
    device = sf_m_major.device
    sf_u8 = sf_m_major.contiguous().view(torch.uint8)
    out = torch.empty(M * K_SF, dtype=torch.uint8, device=device)

    m_idx = torch.arange(M, device=device, dtype=torch.int64).view(M, 1).expand(M, K_SF)
    k_idx = torch.arange(K_SF, device=device, dtype=torch.int64).view(1, K_SF).expand(M, K_SF)
    m_tile = m_idx // 128
    k_tile = k_idx // 4
    out_idx = (
        m_tile * 128 * K_SF
        + k_tile * 512
        + (m_idx % 32) * 16
        + ((m_idx % 128) // 32) * 4
        + (k_idx % 4)
    )
    out[out_idx.reshape(-1)] = sf_u8.reshape(-1)
    return out


@_GATED_SM100
@pytest.mark.parametrize("M,K", _SWIZZLE_SHAPES)
def test_per_token_with_swizzle_sf_byte_equal_to_reference(M: int, K: int) -> None:
    """Fused-swizzle rowwise scale_inv matches the Python byte-permutation
    reference of the M-major SF (covers both rowwise-only and rowwise+colwise).
    """
    device = torch.device("cuda")
    torch.manual_seed(0)
    x = torch.randn((M, K), dtype=torch.bfloat16, device=device)

    out_plain = nvfp4_per_token_quantize(x, rowwise=True, columnwise=True, with_swizzle=False)
    out_swz = nvfp4_per_token_quantize(x, rowwise=True, columnwise=True, with_swizzle=True)

    ref_swz_sf = _swizzle_sf_reference(out_plain.scale.view(torch.uint8))
    got_swz_sf = out_swz.scale.view(torch.uint8).reshape(-1)

    torch.testing.assert_close(
        got_swz_sf,
        ref_swz_sf,
        rtol=0,
        atol=0,
        msg=f"fused-swizzle rowwise SF mismatch at ({M}, {K})",
    )


@_GATED_SM100
@pytest.mark.parametrize("M,K", _SWIZZLE_SHAPES)
def test_per_token_with_swizzle_other_outputs_unchanged(M: int, K: int) -> None:
    """Only the rowwise scale_inv layout differs: FP4 data, row_amax, colwise
    data / scale_inv / col_amax must be byte-identical between with_swizzle
    True and False.
    """
    device = torch.device("cuda")
    torch.manual_seed(0)
    x = torch.randn((M, K), dtype=torch.bfloat16, device=device)

    out_plain = nvfp4_per_token_quantize(x, rowwise=True, columnwise=True, with_swizzle=False)
    out_swz = nvfp4_per_token_quantize(x, rowwise=True, columnwise=True, with_swizzle=True)

    torch.testing.assert_close(out_swz.data, out_plain.data, rtol=0, atol=0)
    torch.testing.assert_close(out_swz.row_amax, out_plain.row_amax, rtol=0, atol=0)
    torch.testing.assert_close(out_swz.columnwise_data, out_plain.columnwise_data, rtol=0, atol=0)
    torch.testing.assert_close(out_swz.columnwise_scale, out_plain.columnwise_scale, rtol=0, atol=0)
    torch.testing.assert_close(out_swz.col_amax, out_plain.col_amax, rtol=0, atol=0)


@_GATED_SM100
@pytest.mark.parametrize("M,K", [(256, 256), (512, 1024), (1024, 512)])
def test_per_token_gemm_with_fused_swizzle_matches_unswizzled(M: int, K: int) -> None:
    """E2E GEMM two paths: (A) with_swizzle=False + ext swizzle (sf_swizzled=False)
    vs (B) with_swizzle=True + sf_swizzled=True. Same SF bytes to cuBLAS LT,
    so C outputs must be byte-equal."""
    device = torch.device("cuda")
    torch.manual_seed(0)
    N = M  # square; GEMM is TN with M, N free
    A = torch.randn((M, K), dtype=torch.bfloat16, device=device)
    B = torch.randn((N, K), dtype=torch.bfloat16, device=device)

    a_plain = nvfp4_per_token_quantize(A, rowwise=True, columnwise=False, with_swizzle=False)
    b_plain = nvfp4_per_token_quantize(B, rowwise=True, columnwise=False, with_swizzle=False)
    c_unswz = nvfp4_per_token_gemm(
        a_plain.data,
        a_plain.scale,
        a_plain.row_amax,
        b_plain.data,
        b_plain.scale,
        b_plain.row_amax,
    )

    a_swz = nvfp4_per_token_quantize(A, rowwise=True, columnwise=False, with_swizzle=True)
    b_swz = nvfp4_per_token_quantize(B, rowwise=True, columnwise=False, with_swizzle=True)
    c_swz = nvfp4_per_token_gemm(
        a_swz.data,
        a_swz.scale,
        a_swz.row_amax,
        b_swz.data,
        b_swz.scale,
        b_swz.row_amax,
        a_sf_swizzled=True,
        b_sf_swizzled=True,
    )

    torch.testing.assert_close(
        c_swz,
        c_unswz,
        rtol=0,
        atol=0,
        msg=f"fused-swizzle GEMM output != unswizzled-input GEMM at ({M}, {K})",
    )
