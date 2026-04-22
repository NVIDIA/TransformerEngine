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


def _has_pertoken_kernel():
    """Check if the per-token kernel binding is available."""
    return hasattr(tex, "quantize_nvfp4_pertoken")


# ---------------------------------------------------------------------------
#  Reference implementation
# ---------------------------------------------------------------------------


def nvfp4_pertoken_quantize_ref(input_tensor: torch.Tensor):
    """Pure PyTorch reference for per-token NVFP4 quantization.

    Returns:
        per_token_scales: (num_rows,) FP32 tensor
            global_scale[row] = row_amax / (fp8_max * fp4_max)
    """
    assert input_tensor.dim() == 2
    num_rows, num_cols = input_tensor.shape
    assert num_cols % 16 == 0

    input_f32 = input_tensor.float()

    # Per-row amax
    row_amax = input_f32.abs().amax(dim=1)  # (num_rows,)

    # S_enc = fp8_max * fp4_max / row_amax
    # global_scale = 1 / S_enc = row_amax / (fp8_max * fp4_max)
    # When amax=0, S_enc=1.0 (fallback), so global_scale=1.0
    per_token_scales = row_amax / (FP8_E4M3_MAX * FP4_MAX)
    per_token_scales = torch.where(
        row_amax == 0, torch.ones_like(per_token_scales), per_token_scales
    )

    return per_token_scales


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

        ref_scales = nvfp4_pertoken_quantize_ref(x)

        torch.testing.assert_close(
            per_token_scales,
            ref_scales,
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


# ---------------------------------------------------------------------------
#  Standalone test (can run without tex binding for reference validation)
# ---------------------------------------------------------------------------


class TestPertokenScaleReference:
    """Test the pure PyTorch reference implementation (no CUDA kernel needed)."""

    def test_reference_basic(self):
        """Basic reference test on CPU."""
        x = torch.tensor([[1.0, 2.0, 3.0, 4.0] * 4], dtype=torch.float32)
        scales = nvfp4_pertoken_quantize_ref(x)
        expected = torch.tensor([4.0 / (FP8_E4M3_MAX * FP4_MAX)])
        torch.testing.assert_close(scales, expected)

    def test_reference_multi_row(self):
        """Multi-row reference test."""
        x = torch.zeros(3, 16, dtype=torch.float32)
        x[0] = 1.0
        x[1] = 10.0
        x[2] = 0.1
        scales = nvfp4_pertoken_quantize_ref(x)

        assert scales[1] > scales[0] > scales[2]
        torch.testing.assert_close(scales[0], torch.tensor(1.0 / (FP8_E4M3_MAX * FP4_MAX)))
        torch.testing.assert_close(scales[1], torch.tensor(10.0 / (FP8_E4M3_MAX * FP4_MAX)))

    def test_reference_zero_row(self):
        """Zero row: S_enc=1.0 fallback, so global_scale=1.0."""
        x = torch.zeros(2, 16, dtype=torch.float32)
        x[0] = 5.0
        scales = nvfp4_pertoken_quantize_ref(x)
        assert scales[1] == 1.0
