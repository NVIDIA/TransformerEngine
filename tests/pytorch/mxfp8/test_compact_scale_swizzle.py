# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# See LICENSE for license information.

"""Scale swizzling directly pads compact MXFP8 scale buffers."""

import pytest
import torch
from mxfp8_utils import swizzle_mxfp8_scale

import transformer_engine_torch as tex
import transformer_engine.pytorch as te
from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer, MXFP8Tensor


def _round_up(value, multiple):
    return (value + multiple - 1) // multiple * multiple


def _scale_shape(rows, cols, rowwise, compact):
    if rowwise:
        return (
            rows if compact else _round_up(rows, 128),
            _round_up((cols + 31) // 32, 4),
        )
    return (
        (rows + 31) // 32 if compact else _round_up((rows + 31) // 32, 4),
        _round_up(cols, 128),
    )


def _make_tensor_and_reference(shape, rowwise):
    rows, cols = shape
    compact_shape = _scale_shape(rows, cols, rowwise, compact=True)
    logical_shape = (rows, (cols + 31) // 32) if rowwise else ((rows + 31) // 32, cols)
    scales = torch.zeros(compact_shape, dtype=torch.uint8, device="cuda")
    scales[: logical_shape[0], : logical_shape[1]] = torch.randint(
        1, 256, logical_shape, dtype=torch.uint8, device="cuda"
    )

    padded_rows, padded_cols = _round_up(rows, 128), _round_up(cols, 128)
    padded_shape = _scale_shape(padded_rows, padded_cols, rowwise, compact=False)
    padded = torch.zeros(padded_shape, dtype=torch.uint8, device="cuda")
    padded[: compact_shape[0], : compact_shape[1]].copy_(scales)
    reference = swizzle_mxfp8_scale(padded_rows, padded_cols, padded, columnwise=not rowwise)

    tensor = MXFP8Tensor(
        shape=shape,
        dtype=torch.bfloat16,
        device="cuda",
        rowwise_data=torch.empty(shape, dtype=torch.uint8, device="cuda") if rowwise else None,
        columnwise_data=None if rowwise else torch.empty(shape, dtype=torch.uint8, device="cuda"),
        rowwise_scale_inv=scales if rowwise else None,
        columnwise_scale_inv=None if rowwise else scales,
        fp8_dtype=tex.DType.kFloat8E4M3,
        quantizer=MXFP8Quantizer(tex.DType.kFloat8E4M3, rowwise=rowwise, columnwise=not rowwise),
        with_gemm_swizzled_scales=False,
    )
    return tensor, reference


@pytest.mark.skipif(not te.is_mxfp8_available(), reason="MXFP8 requires Blackwell")
@pytest.mark.parametrize("shape", [(32, 32), (160, 96), (288, 224), (4096, 4128)])
@pytest.mark.parametrize("rowwise", [False, True])
def test_compact_scale_swizzle(shape, rowwise):
    """Single-tensor swizzle pads compact scales and zero-fills the missing outer extent."""
    tensor, reference = _make_tensor_and_reference(shape, rowwise)
    tex.swizzle_scales_for_gemm_(tensor)
    actual = tensor._rowwise_scale_inv if rowwise else tensor._columnwise_scale_inv
    torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    assert tensor._with_gemm_swizzled_scales


@pytest.mark.skipif(not te.is_mxfp8_available(), reason="MXFP8 requires Blackwell")
@pytest.mark.parametrize("rowwise", [False, True])
def test_multi_tensor_compact_scale_swizzle(rowwise):
    """Multi-tensor swizzle handles compact and already outer-aligned inputs in one batch."""
    tensors_and_references = [
        _make_tensor_and_reference(shape, rowwise) for shape in ((160, 96), (256, 128), (288, 224))
    ]
    tensors = [item[0] for item in tensors_and_references]
    tex.multi_tensor_swizzle_scales_for_gemm_(tensors, rowwise, not rowwise)
    for tensor, reference in tensors_and_references:
        actual = tensor._rowwise_scale_inv if rowwise else tensor._columnwise_scale_inv
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)
