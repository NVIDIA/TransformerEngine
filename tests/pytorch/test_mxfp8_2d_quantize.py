# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Tests for MXFP8 2D quantization."""

import pytest
import torch

import transformer_engine.pytorch as te
import transformer_engine_torch as tex
from transformer_engine.common.recipe import MXFP8BlockScaling
from transformer_engine.pytorch import MXFP8Quantizer
from transformer_engine.pytorch.quantization import (
    MXFP8BlockScalingRecipeState,
    QuantizerRole,
)


mxfp8_available, reason_for_no_mxfp8 = te.is_mxfp8_available(return_reason=True)
MXFP8_BLOCK_SIZE = 32
FP8_E4M3_MAX = 448.0
MXFP8_TEST_SHAPES = [
    (64, 64),
    (128, 128),
    (256, 256),
    (256, 1024),
    (1024, 256),
    (256, 288),
    (320, 320),
    (352, 256),
    (2048, 2048),
    (1024, 2048),
    (2048, 1024),
]
MXFP8_TEST_DTYPES = [torch.float32, torch.bfloat16]


def _valid_rowwise_scale(scale: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
    """Return the logical, unpadded rowwise scale region."""
    return scale[:rows, : (cols + MXFP8_BLOCK_SIZE - 1) // MXFP8_BLOCK_SIZE]


def _valid_columnwise_scale(scale: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
    """Return the logical, unpadded columnwise scale region."""
    return scale[: (rows + MXFP8_BLOCK_SIZE - 1) // MXFP8_BLOCK_SIZE, :cols]


def _float_to_e8m0(amax: torch.Tensor) -> torch.Tensor:
    """Convert amax values to E8M0 scale bytes with the same ceil policy as TE."""
    val = (amax.to(torch.float32) / FP8_E4M3_MAX).contiguous()
    val_u32 = val.view(torch.int32)
    exponent = ((val_u32 >> 23) & 0xFF).to(torch.int32)
    mantissa = val_u32 & 0x7FFFFF

    round_up = (mantissa > 0) & (exponent != 254) & ~((exponent == 0) & (mantissa <= 0x400000))
    exponent = exponent + round_up.to(torch.int32)
    exponent = torch.where(val == 0, torch.zeros_like(exponent), exponent)

    return exponent.to(torch.uint8)


def _e8m0_to_scale_inv(e8m0: torch.Tensor) -> torch.Tensor:
    """Convert E8M0 scale bytes back to scale-inverse values."""
    return torch.pow(2.0, e8m0.to(torch.float32) - 127)


def _mxfp8_2d_quantize_reference(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reference MXFP8 2D quantization using one scale per 32x32 block."""
    rows, cols = x.shape
    assert rows % MXFP8_BLOCK_SIZE == 0
    assert cols % MXFP8_BLOCK_SIZE == 0

    block_rows = rows // MXFP8_BLOCK_SIZE
    block_cols = cols // MXFP8_BLOCK_SIZE
    x_blocks = x.view(
        block_rows,
        MXFP8_BLOCK_SIZE,
        block_cols,
        MXFP8_BLOCK_SIZE,
    ).permute(0, 2, 1, 3)

    block_amax = torch.amax(torch.abs(x_blocks.to(torch.float32)), dim=(-1, -2))
    block_scale_e8m0 = _float_to_e8m0(block_amax)
    block_scale_inv = _e8m0_to_scale_inv(block_scale_e8m0)

    x_scaled = x_blocks.to(torch.float32) / block_scale_inv[:, :, None, None]
    x_quantized = x_scaled.to(torch.float8_e4m3fn)
    rowwise_data = x_quantized.permute(0, 2, 1, 3).reshape(rows, cols)
    rowwise_scale = block_scale_e8m0.repeat_interleave(MXFP8_BLOCK_SIZE, dim=0)
    columnwise_scale = block_scale_e8m0.repeat_interleave(MXFP8_BLOCK_SIZE, dim=1)

    return rowwise_data, rowwise_scale, columnwise_scale


def _quantize(
    quantizer: MXFP8Quantizer,
    x: torch.Tensor,
    use_preallocated_output: bool,
) -> torch.Tensor:
    """Quantize with either the C++ allocator or an explicitly preallocated output."""
    if not use_preallocated_output:
        return quantizer(x)

    out = quantizer.make_empty(
        x.shape,
        dtype=x.dtype,
        device=x.device,
        requires_grad=False,
    )
    return quantizer.update_quantized(x, out)


def _assert_rowwise_scales_are_2d(scales: torch.Tensor, rows: int, cols: int) -> None:
    """Check that each 32x32 block uses one rowwise scale for all rows."""
    valid = _valid_rowwise_scale(scales, rows, cols)
    block_rows = (rows + MXFP8_BLOCK_SIZE - 1) // MXFP8_BLOCK_SIZE
    block_cols = valid.shape[1]
    for block_row in range(block_rows):
        row_start = block_row * MXFP8_BLOCK_SIZE
        row_end = min(row_start + MXFP8_BLOCK_SIZE, rows)
        for block_col in range(block_cols):
            block_scales = valid[row_start:row_end, block_col]
            torch.testing.assert_close(
                block_scales,
                block_scales[0].expand_as(block_scales),
                atol=0,
                rtol=0,
            )


def _assert_bidirectional_scales_are_2d(
    rowwise_scales: torch.Tensor,
    columnwise_scales: torch.Tensor,
    rows: int,
    cols: int,
) -> None:
    """Check that rowwise and columnwise metadata agree per 32x32 block."""
    rowwise_valid = _valid_rowwise_scale(rowwise_scales, rows, cols)
    columnwise_valid = _valid_columnwise_scale(columnwise_scales, rows, cols)
    block_rows = columnwise_valid.shape[0]
    block_cols = rowwise_valid.shape[1]

    for block_row in range(block_rows):
        row_start = block_row * MXFP8_BLOCK_SIZE
        row_end = min(row_start + MXFP8_BLOCK_SIZE, rows)
        for block_col in range(block_cols):
            col_start = block_col * MXFP8_BLOCK_SIZE
            col_end = min(col_start + MXFP8_BLOCK_SIZE, cols)
            rowwise_block_scales = rowwise_valid[row_start:row_end, block_col]
            columnwise_block_scales = columnwise_valid[block_row, col_start:col_end]
            torch.testing.assert_close(
                rowwise_block_scales,
                rowwise_block_scales[0].expand_as(rowwise_block_scales),
                atol=0,
                rtol=0,
            )
            torch.testing.assert_close(
                columnwise_block_scales,
                columnwise_block_scales[0].expand_as(columnwise_block_scales),
                atol=0,
                rtol=0,
            )
            torch.testing.assert_close(
                rowwise_block_scales[0],
                columnwise_block_scales[0],
                atol=0,
                rtol=0,
            )


@pytest.mark.skipif(not mxfp8_available, reason=reason_for_no_mxfp8)
@pytest.mark.parametrize("columnwise", [False, True], ids=["rowwise_only", "bidirectional"])
def test_mxfp8_2d_quantize_scales_match_known_block_amax(columnwise: bool) -> None:
    """Check exact 2D MXFP8 scale bytes on a hand-built 2x2 block matrix.

    The input has one known amax per 32x32 block. Each amax is chosen so the
    expected E8M0 scale byte is known exactly, which makes this test independent
    of the random-input comparisons below. Rowwise-only mode should emit only
    rowwise tensors, while bidirectional mode should emit matching rowwise and
    columnwise scale metadata for the same 2D blocks.
    """
    rows, cols = 64, 64
    x = torch.zeros((rows, cols), dtype=torch.float32, device="cuda")
    block_exponents = torch.tensor([[-2, -1], [0, 1]], device="cuda")
    expected_block_scales = (block_exponents + 127).to(torch.uint8)

    for block_row in range(block_exponents.shape[0]):
        for block_col in range(block_exponents.shape[1]):
            row_start = block_row * MXFP8_BLOCK_SIZE
            col_start = block_col * MXFP8_BLOCK_SIZE
            amax = 448.0 * (2.0 ** int(block_exponents[block_row, block_col].item()))
            x[
                row_start : row_start + MXFP8_BLOCK_SIZE,
                col_start : col_start + MXFP8_BLOCK_SIZE,
            ] = (
                amax * 0.5
            )
            x[row_start, col_start] = amax

    quantizer = MXFP8Quantizer(
        fp8_dtype=tex.DType.kFloat8E4M3,
        rowwise=True,
        columnwise=columnwise,
        with_2d_quantization=True,
    )
    out = quantizer(x)

    expected_rowwise_scales = expected_block_scales.repeat_interleave(MXFP8_BLOCK_SIZE, dim=0)
    torch.testing.assert_close(
        _valid_rowwise_scale(out._rowwise_scale_inv, rows, cols),
        expected_rowwise_scales,
        atol=0,
        rtol=0,
    )

    if columnwise:
        expected_columnwise_scales = expected_block_scales.repeat_interleave(
            MXFP8_BLOCK_SIZE, dim=1
        )
        torch.testing.assert_close(
            _valid_columnwise_scale(out._columnwise_scale_inv, rows, cols),
            expected_columnwise_scales,
            atol=0,
            rtol=0,
        )
    else:
        assert out._columnwise_data is None
        assert out._columnwise_scale_inv is None


@pytest.mark.skipif(not mxfp8_available, reason=reason_for_no_mxfp8)
@pytest.mark.parametrize("shape", MXFP8_TEST_SHAPES)
@pytest.mark.parametrize("dtype", MXFP8_TEST_DTYPES, ids=str)
@pytest.mark.parametrize("columnwise", [False, True], ids=["rowwise_only", "bidirectional"])
@pytest.mark.parametrize(
    "use_preallocated_output",
    [False, True],
    ids=["cpp_allocator", "preallocated_output"],
)
def test_mxfp8_2d_quantize_matches_torch_reference(
    shape: tuple[int, int],
    dtype: torch.dtype,
    columnwise: bool,
    use_preallocated_output: bool,
) -> None:
    """Compare random-input MXFP8 2D data and scales against a PyTorch reference."""
    rows, cols = shape
    torch.manual_seed(9012)
    x = torch.randn(shape, dtype=dtype, device="cuda")

    quantizer = MXFP8Quantizer(
        fp8_dtype=tex.DType.kFloat8E4M3,
        rowwise=True,
        columnwise=columnwise,
        with_2d_quantization=True,
    )
    out = _quantize(quantizer, x, use_preallocated_output)

    ref_data, ref_rowwise_scale, ref_columnwise_scale = _mxfp8_2d_quantize_reference(x)
    ref_data_uint8 = ref_data.view(torch.uint8)

    assert out._rowwise_data is not None
    assert out._rowwise_scale_inv is not None
    torch.testing.assert_close(
        out._rowwise_data.view(torch.uint8),
        ref_data_uint8,
        atol=0,
        rtol=0,
    )
    torch.testing.assert_close(
        _valid_rowwise_scale(out._rowwise_scale_inv, rows, cols),
        ref_rowwise_scale,
        atol=0,
        rtol=0,
    )

    if columnwise:
        assert out._columnwise_data is not None
        assert out._columnwise_scale_inv is not None
        torch.testing.assert_close(
            out._columnwise_data.view(torch.uint8),
            ref_data_uint8,
            atol=0,
            rtol=0,
        )
        torch.testing.assert_close(
            _valid_columnwise_scale(out._columnwise_scale_inv, rows, cols),
            ref_columnwise_scale,
            atol=0,
            rtol=0,
        )
    else:
        assert out._columnwise_data is None
        assert out._columnwise_scale_inv is None


@pytest.mark.skipif(not mxfp8_available, reason=reason_for_no_mxfp8)
@pytest.mark.parametrize("shape", MXFP8_TEST_SHAPES)
@pytest.mark.parametrize("dtype", MXFP8_TEST_DTYPES, ids=str)
@pytest.mark.parametrize(
    "use_preallocated_output",
    [False, True],
    ids=["cpp_allocator", "preallocated_output"],
)
def test_mxfp8_2d_quantize_rowwise_only_matches_bidirectional(
    shape: tuple[int, int],
    dtype: torch.dtype,
    use_preallocated_output: bool,
) -> None:
    """2D MXFP8 must support inference-style rowwise-only weight quantization."""
    rows, cols = shape
    torch.manual_seed(1234)
    x = torch.randn(shape, dtype=dtype, device="cuda")

    rowwise_only_quantizer = MXFP8Quantizer(
        fp8_dtype=tex.DType.kFloat8E4M3,
        rowwise=True,
        columnwise=False,
        with_2d_quantization=True,
    )
    bidirectional_quantizer = MXFP8Quantizer(
        fp8_dtype=tex.DType.kFloat8E4M3,
        rowwise=True,
        columnwise=True,
        with_2d_quantization=True,
    )

    rowwise_only = _quantize(rowwise_only_quantizer, x, use_preallocated_output)
    bidirectional = _quantize(bidirectional_quantizer, x, use_preallocated_output)

    assert rowwise_only._rowwise_data is not None
    assert rowwise_only._rowwise_scale_inv is not None
    assert rowwise_only._columnwise_data is None
    assert rowwise_only._columnwise_scale_inv is None

    torch.testing.assert_close(
        rowwise_only._rowwise_data.view(torch.uint8),
        bidirectional._rowwise_data.view(torch.uint8),
        atol=0,
        rtol=0,
    )
    torch.testing.assert_close(
        _valid_rowwise_scale(rowwise_only._rowwise_scale_inv, rows, cols),
        _valid_rowwise_scale(bidirectional._rowwise_scale_inv, rows, cols),
        atol=0,
        rtol=0,
    )
    _assert_rowwise_scales_are_2d(rowwise_only._rowwise_scale_inv, rows, cols)


@pytest.mark.skipif(not mxfp8_available, reason=reason_for_no_mxfp8)
@pytest.mark.parametrize("shape", MXFP8_TEST_SHAPES)
@pytest.mark.parametrize("dtype", MXFP8_TEST_DTYPES, ids=str)
@pytest.mark.parametrize(
    "use_preallocated_output",
    [False, True],
    ids=["cpp_allocator", "preallocated_output"],
)
def test_mxfp8_2d_quantize_bidirectional_scales_match(
    shape: tuple[int, int],
    dtype: torch.dtype,
    use_preallocated_output: bool,
) -> None:
    """Rowwise and columnwise scale metadata should encode the same 32x32 block scales."""
    rows, cols = shape
    torch.manual_seed(5678)
    x = torch.randn(shape, dtype=dtype, device="cuda")

    quantizer = MXFP8Quantizer(
        fp8_dtype=tex.DType.kFloat8E4M3,
        rowwise=True,
        columnwise=True,
        with_2d_quantization=True,
    )
    out = _quantize(quantizer, x, use_preallocated_output)

    assert out._rowwise_scale_inv is not None
    assert out._columnwise_scale_inv is not None
    _assert_bidirectional_scales_are_2d(
        out._rowwise_scale_inv,
        out._columnwise_scale_inv,
        rows,
        cols,
    )


def test_mxfp8_recipe_default_2d_quantization_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """MXFP8 2D quantization is opt-in."""
    monkeypatch.setenv("NVTE_MXFP8_ENABLE_2D_QUANTIZATION", "0")
    mxfp8_recipe = MXFP8BlockScaling()
    assert mxfp8_recipe.enable_2d_quantization is False

    state = MXFP8BlockScalingRecipeState(
        recipe=mxfp8_recipe,
        mode="forward",
        num_quantizers=3,
        roles=[
            QuantizerRole(module_type="linear", tensor_type="input"),
            QuantizerRole(module_type="linear", tensor_type="weight"),
            QuantizerRole(module_type="linear", tensor_type="output"),
        ],
    )
    assert [q.with_2d_quantization for q in state.make_quantizers()] == [
        False,
        False,
        False,
    ]


def test_mxfp8_recipe_state_uses_2d_only_for_forward_weights() -> None:
    """Only forward weight quantizers should inherit MXFP8 2D quantization."""
    recipe = MXFP8BlockScaling(enable_2d_quantization=True)
    roles = [
        QuantizerRole(module_type="linear", tensor_type="input"),
        QuantizerRole(module_type="linear", tensor_type="weight"),
        QuantizerRole(module_type="linear", tensor_type="output"),
    ]
    state = MXFP8BlockScalingRecipeState(
        recipe=recipe, mode="forward", num_quantizers=3, roles=roles
    )
    quantizers = state.make_quantizers()

    assert [q.with_2d_quantization for q in quantizers] == [False, True, False]

    backward_state = MXFP8BlockScalingRecipeState(
        recipe=recipe,
        mode="backward",
        num_quantizers=2,
        roles=[
            QuantizerRole(module_type="linear", tensor_type="grad_output"),
            QuantizerRole(module_type="linear", tensor_type="grad_input"),
        ],
    )
    assert [q.with_2d_quantization for q in backward_state.make_quantizers()] == [
        False,
        False,
    ]


def test_mxfp8_recipe_state_2d_requires_explicit_weight_role() -> None:
    """MXFP8 2D should not enable itself for unknown positional slots."""
    recipe = MXFP8BlockScaling(enable_2d_quantization=True)
    state = MXFP8BlockScalingRecipeState(recipe=recipe, mode="forward", num_quantizers=3)
    assert [q.with_2d_quantization for q in state.make_quantizers()] == [
        False,
        False,
        False,
    ]


def test_mxfp8_recipe_state_2d_ignores_non_linear_roles() -> None:
    """MXFP8 2D is limited to Linear-style weight quantizers."""
    recipe = MXFP8BlockScaling(enable_2d_quantization=True)
    roles = [
        QuantizerRole(module_type="dpa", tensor_type="qkv"),
        QuantizerRole(module_type="dpa", tensor_type="weight"),
        QuantizerRole(module_type="", tensor_type="weight"),
    ]
    state = MXFP8BlockScalingRecipeState(
        recipe=recipe,
        mode="forward",
        num_quantizers=len(roles),
        roles=roles,
    )
    assert [q.with_2d_quantization for q in state.make_quantizers()] == [
        False,
        False,
        False,
    ]


def test_mxfp8_quantizer_copy_preserves_2d_flag() -> None:
    """MXFP8Quantizer.copy should preserve the 2D quantization setting."""
    quantizer = MXFP8Quantizer(
        fp8_dtype=tex.DType.kFloat8E4M3,
        rowwise=True,
        columnwise=False,
        with_2d_quantization=True,
    )
    copied = quantizer.copy()
    assert copied.rowwise_usage is True
    assert copied.columnwise_usage is False
    assert copied.with_2d_quantization is True
