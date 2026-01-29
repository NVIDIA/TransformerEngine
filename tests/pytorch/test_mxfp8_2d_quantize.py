# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""
Unit tests for MXFP8 2D block scaling quantization.
MXFP8 2D scaling: 32x32 blocks share a single scaling factor, rowwise and colwise scales are identical.
"""

import pytest
import torch

import transformer_engine.pytorch as te
import transformer_engine_torch as tex
from transformer_engine.pytorch import MXFP8Quantizer
from transformer_engine.common.recipe import MXFP8BlockScaling, QParams


mxfp8_available, reason_for_no_mxfp8 = te.is_mxfp8_available(return_reason=True)

# MXFP8 constants
MXFP8_BLOCK_SIZE = 32
FP8_E4M3_MAX = 448.0


def float_to_e8m0(amax: torch.Tensor) -> torch.Tensor:
    """
    Convert absolute maximum values to E8M0 biased exponent (scale inverse).
    
    This mimics the GPU implementation in ptx::float_to_e8m0:
    1. Compute val = amax / FP8_MAX (same as amax * max_norm_rcp)
    2. Extract the biased exponent from the IEEE754 FP32 representation
    3. Round up if there's any mantissa (ceil behavior)
    
    E8M0 format: 8-bit unsigned integer representing 2^(value - 127)
    """
    # Compute val = amax / FP8_MAX (same as GPU: amax * max_norm_rcp)
    val = amax.to(torch.float32) / FP8_E4M3_MAX
    
    # Reinterpret float32 bits as int32
    val_u32 = val.view(torch.int32)
    
    # Extract biased exponent (bits 30:23) - GPU does: (val_u32 >> 23) and truncates to uint8
    exponent = ((val_u32 >> 23) & 0xFF).to(torch.int32)
    
    # Extract mantissa (bits 22:0)
    mantissa = val_u32 & 0x7FFFFF
    
    # Round up condition from GPU:
    # if ((mantissa > 0 && exponent != 0xFE) && !(exponent == 0 && mantissa <= 0x400000))
    round_up = (mantissa > 0) & (exponent != 254) & ~((exponent == 0) & (mantissa <= 0x400000))
    exponent = exponent + round_up.to(torch.int32)
    
    # Handle special cases (GPU handles these before the main logic)
    # val == 0 -> return 0
    exponent = torch.where(val == 0, torch.zeros_like(exponent), exponent)
    
    return exponent.to(torch.uint8)


def e8m0_to_scale_inv(e8m0: torch.Tensor) -> torch.Tensor:
    """Convert E8M0 biased exponent back to scale inverse (float)."""
    return torch.pow(2.0, e8m0.to(torch.float32) - 127)


def quantize_mxfp8_2d_reference(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Reference implementation of MXFP8 2D block scaling quantization.
    
    For 2D scaling, each 32x32 block shares a single E8M0 scale factor.
    
    Args:
        x: Input tensor of shape (M, N), assumes M and N are multiples of 32
    
    Returns:
        qx_rowwise: Quantized data in row-major order
        scale_rowwise: E8M0 scale inverses for rowwise (shape: M x ceil(N/32))
        qx_colwise: Quantized data in column-major order  
        scale_colwise: E8M0 scale inverses for colwise (shape: ceil(M/32) x N)
    """
    M, N = x.shape
    device = x.device
    dtype = x.dtype
    
    # Pad to multiples of 32 if needed
    pad_M = (MXFP8_BLOCK_SIZE - M % MXFP8_BLOCK_SIZE) % MXFP8_BLOCK_SIZE
    pad_N = (MXFP8_BLOCK_SIZE - N % MXFP8_BLOCK_SIZE) % MXFP8_BLOCK_SIZE
    if pad_M > 0 or pad_N > 0:
        x = torch.nn.functional.pad(x, (0, pad_N, 0, pad_M), mode='constant', value=0.0)
    
    M_padded, N_padded = x.shape
    num_block_rows = M_padded // MXFP8_BLOCK_SIZE
    num_block_cols = N_padded // MXFP8_BLOCK_SIZE
    
    # Reshape to expose 32x32 blocks
    x_blocks = x.view(
        num_block_rows, MXFP8_BLOCK_SIZE,
        num_block_cols, MXFP8_BLOCK_SIZE
    ).permute(0, 2, 1, 3)  # (num_block_rows, num_block_cols, 32, 32)
    
    # Compute amax for each 32x32 block
    block_amax = torch.amax(torch.abs(x_blocks.to(torch.float32)), dim=(-1, -2))  # (num_block_rows, num_block_cols)
    
    # Convert to E8M0 scale inverse
    block_scale_e8m0 = float_to_e8m0(block_amax)  # (num_block_rows, num_block_cols)
    block_scale_inv = e8m0_to_scale_inv(block_scale_e8m0)  # (num_block_rows, num_block_cols)
    
    # Expand scale to match input dimensions for quantization
    # For rowwise: each row in a block uses the same scale, scale shape is (M, num_block_cols)
    scale_rowwise = block_scale_e8m0.repeat_interleave(MXFP8_BLOCK_SIZE, dim=0)  # (M_padded, num_block_cols)
    
    # For colwise: each column in a block uses the same scale, scale shape is (num_block_rows, N)
    scale_colwise = block_scale_e8m0.repeat_interleave(MXFP8_BLOCK_SIZE, dim=1)  # (num_block_rows, N_padded)
    
    # Compute scale inverse for quantization (broadcast over 32x32 blocks)
    scale_inv_expanded = block_scale_inv.unsqueeze(-1).unsqueeze(-1)  # (num_block_rows, num_block_cols, 1, 1)
    scale_inv_expanded = scale_inv_expanded.expand(-1, -1, MXFP8_BLOCK_SIZE, MXFP8_BLOCK_SIZE)
    
    # Quantize: x_quantized = round(x / scale_inv) clamped to FP8 range
    x_blocks_float = x_blocks.to(torch.float32)
    x_scaled = x_blocks_float / scale_inv_expanded

    # Convert to FP8 (using PyTorch's float8_e4m3fn)
    x_quantized = x_scaled.to(torch.float8_e4m3fn)
    
    # Reshape back to original layout
    # Rowwise: (M_padded, N_padded)
    qx_rowwise = x_quantized.permute(0, 2, 1, 3).reshape(M_padded, N_padded)
    
    # Colwise: same data but transposed for column-major access
    qx_colwise = x_quantized.permute(0, 2, 1, 3).reshape(M_padded, N_padded)
    
    # Remove padding from outputs
    qx_rowwise = qx_rowwise[:M, :N]
    qx_colwise = qx_colwise[:M, :N]
    scale_rowwise = scale_rowwise[:M, :]
    scale_colwise = scale_colwise[:, :N]
    
    return qx_rowwise, scale_rowwise, qx_colwise, scale_colwise


def check_mxfp8_2d_quantization_versus_reference(
    x_dtype: torch.dtype,
    M: int,
    N: int,
    use_cpp_allocator: bool,
) -> None:
    """
    Test MXFP8 2D quantization against CPU reference implementation.
    
    Verifies:
    1. scales match reference
    2. 32x32 blocks share the same scale
    3. rowwise and colwise quantized data match reference
    """
    fp8_dtype = tex.DType.kFloat8E4M3

    device = "cuda"
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # Create input tensor
    x = torch.randn((M, N), dtype=x_dtype, device=device)

    # GPU Quantization using MXFP8Quantizer with 2D scaling
    quantizer = MXFP8Quantizer(
        fp8_dtype=fp8_dtype,
        rowwise=True,
        columnwise=True,
        with_2d_quantization=True,
    )

    if use_cpp_allocator:
        x_mxfp8 = quantizer(x)
    else:
        x_mxfp8 = quantizer.make_empty(
            (M, N), dtype=x_dtype, device=device, requires_grad=False
        )
        x_mxfp8 = quantizer.update_quantized(x, x_mxfp8)

    # Extract GPU results
    assert x_mxfp8._rowwise_data is not None
    assert x_mxfp8._columnwise_data is not None
    assert x_mxfp8._rowwise_scale_inv is not None
    assert x_mxfp8._columnwise_scale_inv is not None

    gpu_qx_rowwise = x_mxfp8._rowwise_data
    gpu_scale_rowwise = x_mxfp8._rowwise_scale_inv
    gpu_qx_colwise = x_mxfp8._columnwise_data
    gpu_scale_colwise = x_mxfp8._columnwise_scale_inv

    # Reference Quantization
    ref_qx_rowwise, ref_scale_rowwise, ref_qx_colwise, ref_scale_colwise = \
        quantize_mxfp8_2d_reference(x)

    num_block_rows = (M + MXFP8_BLOCK_SIZE - 1) // MXFP8_BLOCK_SIZE
    num_block_cols = (N + MXFP8_BLOCK_SIZE - 1) // MXFP8_BLOCK_SIZE
    
    # GPU scales may have padding, compare valid portion
    gpu_scale_rowwise_valid = gpu_scale_rowwise[:M, :num_block_cols]
    gpu_scale_colwise_valid = gpu_scale_colwise[:num_block_rows, :N]

    # 1. Verify scales match reference
    torch.testing.assert_close(
        gpu_scale_rowwise_valid,
        ref_scale_rowwise,
        atol=0, rtol=0,
    )
    
    # 2. Verify 32x32 blocks share the same scale
    for bi in range(num_block_rows):
        for bj in range(num_block_cols):
            row_start = bi * MXFP8_BLOCK_SIZE
            row_end = min((bi + 1) * MXFP8_BLOCK_SIZE, M)
            col_start = bj * MXFP8_BLOCK_SIZE
            col_end = min((bj + 1) * MXFP8_BLOCK_SIZE, N)

            # All rows in block should have same scale for this column block
            block_rowwise_scales = gpu_scale_rowwise[row_start:row_end, bj]
            assert torch.all(block_rowwise_scales == block_rowwise_scales[0]), (
                f"2D mode: Block ({bi},{bj}) rowwise scales should be identical"
            )

            # All columns in block should have same scale for this row block
            block_colwise_scales = gpu_scale_colwise[bi, col_start:col_end]
            assert torch.all(block_colwise_scales == block_colwise_scales[0]), (
                f"2D mode: Block ({bi},{bj}) colwise scales should be identical"
            )

            # Rowwise and colwise scales should match
            assert block_rowwise_scales[0] == block_colwise_scales[0], (
                f"2D mode: Block ({bi},{bj}) rowwise and colwise scales should be equal, "
                f"got rowwise={block_rowwise_scales[0]}, colwise={block_colwise_scales[0]}"
            )

    # 3. Verify rowwise and colwise quantized data match reference
    # Convert FP8 tensors to uint8 for bitwise comparison
    gpu_qx_rowwise_uint8 = gpu_qx_rowwise.view(torch.uint8)[:M, :N]
    gpu_qx_colwise_uint8 = gpu_qx_colwise.view(torch.uint8)[:M, :N]
    ref_qx_rowwise_uint8 = ref_qx_rowwise.view(torch.uint8)
    
    torch.testing.assert_close(
        gpu_qx_rowwise_uint8,
        ref_qx_rowwise_uint8,
        atol=0, rtol=0,
    )

    torch.testing.assert_close(
        gpu_qx_colwise_uint8,
        ref_qx_rowwise_uint8,
        atol=0, rtol=0,
    )


@pytest.mark.skipif(not mxfp8_available, reason=reason_for_no_mxfp8)
@pytest.mark.parametrize(
    "M, N",
    [
        # Full tile cases (multiples of 32)
        (64, 64),
        (128, 128),
        (256, 256),
        (256, 1024),
        (1024, 256),
        # Padding required cases
        (256, 288),
        (320, 320),
        (352, 256),
        # Larger sizes
        (2048, 2048),
        (1024, 2048),
        (2048, 1024),
    ],
)
@pytest.mark.parametrize("x_dtype", [torch.float32, torch.bfloat16], ids=str)
@pytest.mark.parametrize(
    "use_cpp_allocator", [True, False], ids=["cpp_allocator", "python_allocator"]
)
def test_mxfp8_2d_quantization_versus_reference(
    M: int,
    N: int,
    x_dtype: torch.dtype,
    use_cpp_allocator: bool,
) -> None:
    """Test MXFP8 2D quantization against reference implementation."""
    check_mxfp8_2d_quantization_versus_reference(
        x_dtype=x_dtype,
        M=M,
        N=N,
        use_cpp_allocator=use_cpp_allocator,
    )


# ============================================================================
# Recipe Configuration Tests
# ============================================================================

class TestMXFP8BlockScalingRecipe:
    """Tests for MXFP8BlockScaling recipe configuration."""

    @pytest.mark.skipif(not mxfp8_available, reason=reason_for_no_mxfp8)
    def test_default_recipe_has_qparams(self):
        """Test that default MXFP8BlockScaling has QParams attributes."""
        mxfp8_recipe = MXFP8BlockScaling()
        
        # Verify QParams attributes exist
        assert hasattr(mxfp8_recipe, 'fp8_quant_fwd_inp')
        assert hasattr(mxfp8_recipe, 'fp8_quant_fwd_weight')
        assert hasattr(mxfp8_recipe, 'fp8_quant_bwd_grad')
        
        # Verify they are QParams instances
        assert isinstance(mxfp8_recipe.fp8_quant_fwd_inp, QParams)
        assert isinstance(mxfp8_recipe.fp8_quant_fwd_weight, QParams)
        assert isinstance(mxfp8_recipe.fp8_quant_bwd_grad, QParams)

    @pytest.mark.skipif(not mxfp8_available, reason=reason_for_no_mxfp8)
    def test_default_2d_quantization_disabled(self):
        """Test that 2D quantization is disabled by default."""
        mxfp8_recipe = MXFP8BlockScaling()
        
        # By default, 2D quantization should be disabled
        assert mxfp8_recipe.enable_2d_quantization is False
        
        # QParams should reflect this
        assert mxfp8_recipe.fp8_quant_fwd_inp.mxfp8_2d_quantization is False
        assert mxfp8_recipe.fp8_quant_fwd_weight.mxfp8_2d_quantization is False
        assert mxfp8_recipe.fp8_quant_bwd_grad.mxfp8_2d_quantization is False

    @pytest.mark.skipif(not mxfp8_available, reason=reason_for_no_mxfp8)
    def test_2d_quantization_enabled_only_for_weight(self):
        """Test that when 2D quantization is enabled, it only applies to weight."""
        # Create recipe with 2D quantization enabled
        mxfp8_recipe = MXFP8BlockScaling(enable_2d_quantization=True)
        
        # enable_2d_quantization should be True
        assert mxfp8_recipe.enable_2d_quantization is True
        
        # Only weight should have 2D quantization enabled
        assert mxfp8_recipe.fp8_quant_fwd_inp.mxfp8_2d_quantization is False
        assert mxfp8_recipe.fp8_quant_fwd_weight.mxfp8_2d_quantization is True
        assert mxfp8_recipe.fp8_quant_bwd_grad.mxfp8_2d_quantization is False

    @pytest.mark.skipif(not mxfp8_available, reason=reason_for_no_mxfp8)
    def test_qparams_default_values(self):
        """Test that QParams have correct default values for MXFP8."""
        mxfp8_recipe = MXFP8BlockScaling()
        
        # Check default values for all QParams
        for qparams in [
            mxfp8_recipe.fp8_quant_fwd_inp,
            mxfp8_recipe.fp8_quant_fwd_weight,
            mxfp8_recipe.fp8_quant_bwd_grad,
        ]:
            # These should use defaults for MXFP8
            assert qparams.power_2_scale is False  # MXFP8 uses E8M0, inherently power of 2
            assert qparams.amax_epsilon == 0.0
            assert qparams.random_hadamard_transform is False
            assert qparams.stochastic_rounding is False
            assert qparams.fp4_2d_quantization is False  # Not applicable to MXFP8
            assert qparams.mxfp8_2d_quantization is False  # Default is False

    @pytest.mark.skipif(not mxfp8_available, reason=reason_for_no_mxfp8)
    def test_recipe_repr_includes_2d_quantization(self):
        """Test that recipe __repr__ includes 2D quantization status."""
        mxfp8_recipe_disabled = MXFP8BlockScaling(enable_2d_quantization=False)
        mxfp8_recipe_enabled = MXFP8BlockScaling(enable_2d_quantization=True)
        
        repr_disabled = repr(mxfp8_recipe_disabled)
        repr_enabled = repr(mxfp8_recipe_enabled)
        
        assert "enable_2d_quantization=False" in repr_disabled
        assert "enable_2d_quantization=True" in repr_enabled


@pytest.mark.skipif(not mxfp8_available, reason=reason_for_no_mxfp8)
def test_mxfp8_quantizer_respects_2d_flag():
    """Test that MXFP8Quantizer correctly uses the 2D quantization flag from recipe."""
    # Test with 2D disabled
    quantizer_1d = MXFP8Quantizer(
        fp8_dtype=tex.DType.kFloat8E4M3,
        rowwise=True,
        columnwise=True,
        with_2d_quantization=False,
    )
    assert quantizer_1d.with_2d_quantization is False
    
    # Test with 2D enabled
    quantizer_2d = MXFP8Quantizer(
        fp8_dtype=tex.DType.kFloat8E4M3,
        rowwise=True,
        columnwise=True,
        with_2d_quantization=True,
    )
    assert quantizer_2d.with_2d_quantization is True


@pytest.mark.skipif(not mxfp8_available, reason=reason_for_no_mxfp8)
def test_mxfp8_recipe_state_creates_correct_quantizers():
    """Test that MXFP8BlockScalingRecipeState creates quantizers with correct 2D settings."""
    from transformer_engine.pytorch.quantization import MXFP8BlockScalingRecipeState
    
    # Test with 2D disabled
    recipe_1d = MXFP8BlockScaling(enable_2d_quantization=False)
    state_fwd_1d = MXFP8BlockScalingRecipeState(
        recipe=recipe_1d,
        mode="forward",
        num_quantizers=3,  # input, weight, output
    )
    quantizers_1d = state_fwd_1d.make_quantizers()
    
    # All quantizers should have 2D disabled
    for idx, q in enumerate(quantizers_1d):
        assert q.with_2d_quantization is False, f"Quantizer {idx} should have 2D disabled"
    
    # Test with 2D enabled
    recipe_2d = MXFP8BlockScaling(enable_2d_quantization=True)
    state_fwd_2d = MXFP8BlockScalingRecipeState(
        recipe=recipe_2d,
        mode="forward",
        num_quantizers=3,
    )
    quantizers_2d = state_fwd_2d.make_quantizers()
    
    # Only weight (idx % 3 == 1) should have 2D enabled
    for idx, q in enumerate(quantizers_2d):
        if idx % 3 == 1:  # weight
            assert q.with_2d_quantization is True, f"Weight quantizer {idx} should have 2D enabled"
        else:  # input or output
            assert q.with_2d_quantization is False, f"Non-weight quantizer {idx} should have 2D disabled"
