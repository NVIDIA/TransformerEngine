# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Tests for per-token NVFP4 quantization kernel (tex.quantize_nvfp4_pertoken).

These tests validate the CUDA kernel in quantize_pertoken_nvfp4.cuh, which
performs per-row amax reduction and NVFP4 quantization in a single kernel.

Tests require SM100+ (Blackwell) for FP4 hardware support.
"""

import math
import pytest
import torch

import transformer_engine.pytorch as te
import transformer_engine_torch as tex

# Check hardware support
_, reason_for_no_nvfp4 = te.is_nvfp4_available(return_reason=True)
nvfp4_available = te.is_nvfp4_available()

pytestmark = pytest.mark.skipif(not nvfp4_available, reason=reason_for_no_nvfp4)

FP4_MAX = 6.0
FP8_E4M3_MAX = 448.0

# FP4 E2M1 look-up table: 4-bit index -> float value
# Lower nibble = first element, upper nibble = second element
_FP4_E2M1_LUT = [
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
]


def unpack_fp4(packed: torch.Tensor) -> torch.Tensor:
    """Unpack uint8 packed FP4 data to two columns per byte.

    Each byte contains 2 FP4 values: lower nibble = first, upper nibble = second.
    Returns a uint8 tensor with 2x the columns.
    """
    repeated = packed.repeat_interleave(2, dim=1)
    repeated[:, 0::2] = repeated[:, 0::2] & 0x0F  # Lower 4 bits
    repeated[:, 1::2] = repeated[:, 1::2] >> 4  # Upper 4 bits
    return repeated


def fp4_to_fp32(unpacked: torch.Tensor) -> torch.Tensor:
    """Convert unpacked FP4 indices to float32 values using E2M1 LUT."""
    lut = torch.tensor(_FP4_E2M1_LUT, dtype=torch.float32, device=unpacked.device)
    return lut[unpacked.long()]


def dequantize_pertoken_fp4(
    data: torch.Tensor, scales: torch.Tensor, per_token_scales: torch.Tensor
) -> torch.Tensor:
    """Dequantize per-token NVFP4: result = fp4_val * block_scale * per_token_scale.

    Args:
        data: (M, K/2) uint8 packed FP4
        scales: (M, K/16) uint8 block scales (FP8 E4M3)
        per_token_scales: (M,) FP32 per-token global scales

    Returns:
        (M, K) float32 dequantized tensor
    """
    num_rows = data.shape[0]
    num_cols = data.shape[1] * 2  # 2 FP4 values per byte

    # Unpack FP4 -> float32
    fp4_vals = fp4_to_fp32(unpack_fp4(data))  # (M, K)

    # Expand block scales: each scale covers 16 elements
    block_scales_f32 = scales.view(torch.float8_e4m3fn).float()  # (M, K/16)
    block_scales_expanded = block_scales_f32.repeat_interleave(16, dim=1)  # (M, K)
    block_scales_expanded = block_scales_expanded[:, :num_cols]

    # Expand per-token scales: one per row
    token_scales_expanded = per_token_scales.unsqueeze(1)  # (M, 1)

    return fp4_vals * block_scales_expanded * token_scales_expanded


def _has_pertoken_kernel():
    """Check if the per-token kernel binding is available."""
    return hasattr(tex, "quantize_nvfp4_pertoken")


# ---------------------------------------------------------------------------
#  Reference implementation
# ---------------------------------------------------------------------------


def nvfp4_pertoken_quantize_ref(input_tensor: torch.Tensor):
    """Pure PyTorch reference for per-token NVFP4 quantization.

    Reproduces the exact logic of quantize_pertoken_nvfp4_kernel:
      Pass 1: per-row amax → S_enc → per_token_scale
      Pass 2: per-block(16) amax → S_dec_b (E4M3) → scale + quantize to FP4

    Returns:
        data: (M, K/2) uint8 packed FP4
        scales: (M, K/16) uint8 (FP8 E4M3 block scales)
        per_token_scales: (M,) FP32
    """
    from transformer_engine.pytorch.custom_recipes.quantization_nvfp4 import cast_to_fp4x2

    assert input_tensor.dim() == 2
    num_rows, num_cols = input_tensor.shape
    assert num_cols % 16 == 0

    x = input_tensor.float()

    # --- Pass 1: Per-row amax → S_enc → per_token_scale ---
    row_amax = x.abs().amax(dim=1)  # (M,)

    # compute_global_encode_scaling_factor_FP4: S_enc = fp8_max * fp4_max / amax
    S_enc = FP8_E4M3_MAX * FP4_MAX / row_amax
    S_enc = torch.clamp(S_enc, max=torch.finfo(torch.float32).max)
    S_enc = torch.where((row_amax == 0) | (S_enc == 0), torch.ones_like(S_enc), S_enc)

    per_token_scales = 1.0 / S_enc  # global_scale = 1 / S_enc
    per_token_scales = torch.where(
        row_amax == 0, torch.ones_like(per_token_scales), per_token_scales
    )

    # --- Pass 2: Per-block quantization ---
    num_blocks = num_cols // 16
    x_blocks = x.view(num_rows, num_blocks, 16)  # (M, K/16, 16)

    # Per-block amax
    block_amax = x_blocks.abs().amax(dim=-1)  # (M, K/16)

    # compute_decoding_scaling_factor: S_dec_b = block_amax * S_enc / fp4_max
    # Then cast to FP8 E4M3
    S_enc_expanded = S_enc.unsqueeze(1)  # (M, 1)
    S_dec_b = block_amax * S_enc_expanded / FP4_MAX
    S_dec_b = torch.clamp(S_dec_b, max=FP8_E4M3_MAX)
    S_dec_b_fp8 = S_dec_b.to(torch.float8_e4m3fn)
    S_dec_b_f = S_dec_b_fp8.float()

    # Block encode scale = S_enc / S_dec_b_f (inverse for quantization)
    block_encode_scale = torch.where(
        S_dec_b_f != 0,
        S_enc_expanded / S_dec_b_f,
        torch.zeros_like(S_dec_b_f),
    )  # (M, K/16)

    # Scale input and clamp to FP4 range [-6, 6]
    block_encode_expanded = block_encode_scale.unsqueeze(-1)  # (M, K/16, 1)
    scaled_x = x_blocks * block_encode_expanded  # (M, K/16, 16)
    scaled_x = scaled_x.reshape(num_rows, num_cols)
    clamped_x = torch.clamp(scaled_x, -FP4_MAX, FP4_MAX)

    # Pack to FP4 using TE's reference cast_to_fp4x2
    data = cast_to_fp4x2(clamped_x)

    # Block scales as uint8 (FP8 E4M3 raw bytes)
    scales = S_dec_b_fp8.view(torch.uint8)

    return data, scales, per_token_scales


# ---------------------------------------------------------------------------
#  Tests
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _has_pertoken_kernel(), reason="tex.quantize_nvfp4_pertoken not available")
class TestQuantizeNvfp4Pertoken:
    """Test suite for per-token NVFP4 quantization kernel."""

    @pytest.mark.parametrize(
        "num_rows,num_cols",
        [
            (1, 16),
            (1, 256),
            (4, 256),
            (32, 256),
            (64, 4096),
            (128, 4096),
            (256, 4096),
            (512, 14336),
        ],
        ids=lambda x: f"{x}",
    )
    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
    def test_output_shapes(self, num_rows, num_cols, dtype):
        """Verify output tensor shapes are correct."""
        x = torch.randn(num_rows, num_cols, dtype=dtype, device="cuda")
        data, scales, per_token_scales = tex.quantize_nvfp4_pertoken(x)

        assert data.shape == (num_rows, num_cols // 2), f"data shape: {data.shape}"
        assert scales.shape == (num_rows, num_cols // 16), f"scales shape: {scales.shape}"
        assert per_token_scales.shape == (
            num_rows,
        ), f"per_token_scales shape: {per_token_scales.shape}"
        assert data.dtype == torch.uint8
        assert scales.dtype == torch.uint8
        assert per_token_scales.dtype == torch.float32

    @pytest.mark.parametrize(
        "num_rows,num_cols",
        [
            (1, 256),
            (32, 256),
            (64, 4096),
            (256, 4096),
        ],
    )
    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
    def test_per_token_scales_match_reference(self, num_rows, num_cols, dtype):
        """Verify per-token scales match pure PyTorch reference."""
        x = torch.randn(num_rows, num_cols, dtype=dtype, device="cuda")
        _, _, per_token_scales = tex.quantize_nvfp4_pertoken(x)

        _, _, ref_scales = nvfp4_pertoken_quantize_ref(x)

        torch.testing.assert_close(
            per_token_scales,
            ref_scales.to(device="cuda"),
            atol=1e-5,
            rtol=1e-3,
            msg="Per-token scales should match reference",
        )

    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
    def test_zero_input(self, dtype):
        """Zero input: S_enc = 1.0 (fallback), so global_scale = 1/1 = 1.0."""
        x = torch.zeros(16, 256, dtype=dtype, device="cuda")
        _, _, per_token_scales = tex.quantize_nvfp4_pertoken(x)

        # When amax=0, compute_global_encode_scaling_factor_FP4 returns 1.0
        # so global_scale = 1/S_enc = 1/1 = 1.0
        assert (
            per_token_scales == 1.0
        ).all(), f"Zero input should give global_scale=1.0 (S_enc fallback), got {per_token_scales}"

    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
    def test_uniform_rows_same_scale(self, dtype):
        """Rows with the same magnitude should produce the same per-token scale."""
        num_rows = 8
        num_cols = 256
        x = torch.randn(1, num_cols, dtype=dtype, device="cuda").expand(num_rows, -1).contiguous()
        _, _, per_token_scales = tex.quantize_nvfp4_pertoken(x)

        # All rows identical → all scales identical
        assert torch.allclose(
            per_token_scales, per_token_scales[0].expand_as(per_token_scales)
        ), "Identical rows should produce identical per-token scales"

    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
    def test_different_rows_different_scales(self, dtype):
        """Rows with different magnitudes should produce different per-token scales."""
        num_cols = 256
        # Row 0: small values, Row 1: large values
        x = torch.zeros(2, num_cols, dtype=dtype, device="cuda")
        x[0] = torch.randn(num_cols, dtype=dtype, device="cuda") * 0.01
        x[1] = torch.randn(num_cols, dtype=dtype, device="cuda") * 100.0
        _, _, per_token_scales = tex.quantize_nvfp4_pertoken(x)

        # Scale for large row should be much larger
        assert per_token_scales[1] > per_token_scales[0] * 10, (
            f"Large row scale ({per_token_scales[1].item():.6f}) should be >> "
            f"small row scale ({per_token_scales[0].item():.6f})"
        )

    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
    def test_scale_formula(self, dtype):
        """Verify scale = row_amax / (fp8_max * fp4_max)."""
        num_rows = 4
        num_cols = 256
        x = torch.randn(num_rows, num_cols, dtype=dtype, device="cuda")
        _, _, per_token_scales = tex.quantize_nvfp4_pertoken(x)

        # Compute expected scales
        row_amax = x.float().abs().amax(dim=1)
        expected_scales = row_amax / (FP8_E4M3_MAX * FP4_MAX)

        torch.testing.assert_close(
            per_token_scales,
            expected_scales,
            atol=1e-5,
            rtol=1e-3,
            msg="Scale should equal row_amax / (fp8_max * fp4_max)",
        )

    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
    def test_block_scales_are_valid_fp8(self, dtype):
        """Block scales should be valid FP8 E4M3 values (non-NaN, non-Inf)."""
        x = torch.randn(32, 4096, dtype=dtype, device="cuda")
        _, scales, _ = tex.quantize_nvfp4_pertoken(x)

        # Reinterpret uint8 as FP8 E4M3 and check for validity
        scales_f32 = scales.to(torch.float8_e4m3fn).float()
        assert not torch.isnan(scales_f32).any(), "Block scales contain NaN"
        assert not torch.isinf(scales_f32).any(), "Block scales contain Inf"
        assert (scales_f32 >= 0).all(), "Block scales should be non-negative"

    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
    def test_packed_fp4_data_shape(self, dtype):
        """Packed FP4 output should have exactly half the columns (2 elements per byte)."""
        for num_cols in [16, 32, 256, 4096]:
            x = torch.randn(4, num_cols, dtype=dtype, device="cuda")
            data, _, _ = tex.quantize_nvfp4_pertoken(x)
            assert data.shape[1] == num_cols // 2

    @pytest.mark.parametrize(
        "num_rows,num_cols",
        [
            (4, 256),
            (32, 256),
            (64, 4096),
        ],
    )
    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
    def test_dequantized_data_close_to_input(self, num_rows, num_cols, dtype):
        """Dequantized FP4 data should be close to the original input.

        Quantize -> dequantize round-trip should preserve values within FP4 precision.
        FP4 E2M1 has ~1 bit mantissa, so expect ~25% relative error for non-tiny values.
        """
        torch.manual_seed(42)
        x = torch.randn(num_rows, num_cols, dtype=dtype, device="cuda")
        data, scales, per_token_scales = tex.quantize_nvfp4_pertoken(x)

        dequant = dequantize_pertoken_fp4(data, scales, per_token_scales)

        # Compare against original (allow FP4 quantization error)
        x_f32 = x.float()
        nonzero = x_f32.abs() > 0.1  # skip very small values where relative error is meaningless
        if nonzero.any():
            rel_error = ((dequant[nonzero] - x_f32[nonzero]).abs() / x_f32[nonzero].abs()).mean()
            assert rel_error < 0.5, (
                f"Mean relative error {rel_error:.3f} too high for FP4 round-trip "
                f"(shape={num_rows}x{num_cols}, dtype={dtype})"
            )

    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
    def test_fp4_values_in_valid_range(self, dtype):
        """Unpacked FP4 indices should be in [0, 15] (valid 4-bit range)."""
        x = torch.randn(16, 256, dtype=dtype, device="cuda")
        data, _, _ = tex.quantize_nvfp4_pertoken(x)

        unpacked = unpack_fp4(data)
        assert (unpacked >= 0).all() and (
            unpacked <= 15
        ).all(), f"FP4 indices out of range: min={unpacked.min()}, max={unpacked.max()}"

    def test_input_validation_not_2d(self):
        """Should reject non-2D input."""
        x = torch.randn(2, 3, 256, dtype=torch.bfloat16, device="cuda")
        with pytest.raises(RuntimeError):
            tex.quantize_nvfp4_pertoken(x)

    def test_input_validation_not_multiple_of_16(self):
        """Should reject num_cols not divisible by 16."""
        x = torch.randn(4, 100, dtype=torch.bfloat16, device="cuda")
        with pytest.raises(RuntimeError):
            tex.quantize_nvfp4_pertoken(x)

    def test_input_validation_wrong_dtype(self):
        """Should reject non-BF16/FP16 input."""
        x = torch.randn(4, 256, dtype=torch.float32, device="cuda")
        with pytest.raises(RuntimeError):
            tex.quantize_nvfp4_pertoken(x)

    # -----------------------------------------------------------------------
    #  Exact byte-match tests (following test_nvfp4_quantize_exact.py pattern)
    # -----------------------------------------------------------------------

    @pytest.mark.parametrize(
        "M, N",
        [
            (4, 256),
            (16, 256),
            (32, 1024),
            (128, 4096),
        ],
    )
    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
    def test_fp4_data_exact_match(self, M, N, dtype):
        """FP4 packed data must exactly match Python reference (byte-for-byte)."""
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
        x = torch.randn(M, N, dtype=dtype, device="cuda")

        data, scales, pts = tex.quantize_nvfp4_pertoken(x)
        ref_data, ref_scales, ref_pts = nvfp4_pertoken_quantize_ref(x)

        # Unpack both to 4-bit indices for comparison
        kernel_unpacked = unpack_fp4(data)
        ref_unpacked = unpack_fp4(ref_data.to(device="cuda"))

        torch.testing.assert_close(
            kernel_unpacked,
            ref_unpacked,
            atol=0.0,
            rtol=0.0,
            msg=f"FP4 data mismatch for shape ({M}, {N}), dtype={dtype}",
        )

    @pytest.mark.parametrize(
        "M, N",
        [
            (4, 256),
            (16, 256),
            (32, 1024),
            (128, 4096),
        ],
    )
    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
    def test_block_scales_exact_match(self, M, N, dtype):
        """Block scales must exactly match Python reference (byte-for-byte)."""
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
        x = torch.randn(M, N, dtype=dtype, device="cuda")

        _, scales, _ = tex.quantize_nvfp4_pertoken(x)
        _, ref_scales, _ = nvfp4_pertoken_quantize_ref(x)

        torch.testing.assert_close(
            scales,
            ref_scales.to(device="cuda"),
            atol=0.0,
            rtol=0.0,
            msg=f"Block scales mismatch for shape ({M}, {N}), dtype={dtype}",
        )

    @pytest.mark.parametrize(
        "M, N",
        [
            (4, 256),
            (16, 256),
            (32, 1024),
            (128, 4096),
        ],
    )
    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
    def test_per_token_scales_exact_match(self, M, N, dtype):
        """Per-token scales must exactly match Python reference."""
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
        x = torch.randn(M, N, dtype=dtype, device="cuda")

        _, _, pts = tex.quantize_nvfp4_pertoken(x)
        _, _, ref_pts = nvfp4_pertoken_quantize_ref(x)

        torch.testing.assert_close(
            pts,
            ref_pts.to(device="cuda"),
            atol=0.0,
            rtol=0.0,
            msg=f"Per-token scales mismatch for shape ({M}, {N}), dtype={dtype}",
        )


# ---------------------------------------------------------------------------
#  Standalone test (can run without tex binding for reference validation)
# ---------------------------------------------------------------------------


class TestPertokenScaleReference:
    """Test the pure PyTorch reference implementation (no CUDA kernel needed)."""

    def test_reference_basic(self):
        """Basic reference test on CPU."""
        x = torch.tensor([[1.0, 2.0, 3.0, 4.0] * 4], dtype=torch.float32)
        _, _, pts = nvfp4_pertoken_quantize_ref(x)
        expected = torch.tensor([4.0 / (FP8_E4M3_MAX * FP4_MAX)])
        torch.testing.assert_close(pts, expected)

    def test_reference_multi_row(self):
        """Multi-row reference test."""
        x = torch.zeros(3, 16, dtype=torch.float32)
        x[0] = 1.0
        x[1] = 10.0
        x[2] = 0.1
        _, _, pts = nvfp4_pertoken_quantize_ref(x)

        assert pts[1] > pts[0] > pts[2]
        torch.testing.assert_close(pts[0], torch.tensor(1.0 / (FP8_E4M3_MAX * FP4_MAX)))
        torch.testing.assert_close(pts[1], torch.tensor(10.0 / (FP8_E4M3_MAX * FP4_MAX)))

    def test_reference_zero_row(self):
        """Zero row: S_enc=1.0 fallback, so global_scale=1.0."""
        x = torch.zeros(2, 16, dtype=torch.float32)
        x[0] = 5.0
        _, _, pts = nvfp4_pertoken_quantize_ref(x)
        assert pts[1] == 1.0
