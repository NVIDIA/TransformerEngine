# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# See LICENSE for license information.

"""Compact MXFP8 scales are padded as part of GEMM preparation."""

import pytest
import torch
from mxfp8_utils import swizzle_mxfp8_scale

import transformer_engine.pytorch as te
import transformer_engine_torch as tex
from transformer_engine.pytorch.cpp_extensions import general_gemm, general_grouped_gemm
from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer, MXFP8Tensor


def _copy_with_scale_layout(reference, compact):
    """Copy independently quantized planes, isolating swizzle from quantization."""
    rows, cols = reference.shape
    row_scales = reference._rowwise_scale_inv
    col_scales = reference._columnwise_scale_inv
    if compact:
        row_scales = row_scales[:rows, : cols // 32]
        col_scales = col_scales[: rows // 32, :cols]
    return MXFP8Tensor(
        shape=reference.shape,
        dtype=reference.dtype,
        device=reference.device,
        rowwise_data=reference._rowwise_data.clone(),
        columnwise_data=reference._columnwise_data.clone(),
        rowwise_scale_inv=row_scales.clone().contiguous(),
        columnwise_scale_inv=col_scales.clone().contiguous(),
        fp8_dtype=tex.DType.kFloat8E4M3,
        quantizer=MXFP8Quantizer(tex.DType.kFloat8E4M3),
        with_gemm_swizzled_scales=False,
    )


def _snapshot(tensor):
    return [
        (plane.data_ptr(), plane.clone())
        for plane in (
            tensor._rowwise_data,
            tensor._columnwise_data,
            tensor._rowwise_scale_inv,
            tensor._columnwise_scale_inv,
        )
    ]


def _check_unchanged(tensor, snapshot):
    assert not tensor._with_gemm_swizzled_scales
    for plane, (pointer, before) in zip(
        (
            tensor._rowwise_data,
            tensor._columnwise_data,
            tensor._rowwise_scale_inv,
            tensor._columnwise_scale_inv,
        ),
        snapshot,
    ):
        assert plane.data_ptr() == pointer
        torch.testing.assert_close(plane, before, rtol=0, atol=0)


@pytest.mark.skipif(not te.is_mxfp8_available(), reason="MXFP8 GEMM requires Blackwell")
@pytest.mark.parametrize("layout", ["TN", "NN", "NT"])
@pytest.mark.parametrize("grouped", [False, True])
@pytest.mark.parametrize("storage", ["compact", "padded", "mixed"])
def test_compact_scales_gemm(layout, grouped, storage):
    torch.manual_seed(1234)
    quantizer = MXFP8Quantizer(tex.DType.kFloat8E4M3)
    operands_a, operands_b, expected = [], [], []
    references_a, references_b = [], []
    for i, shape in enumerate([(64, 64), (96, 160), (256, 128)]):
        rows, cols = shape
        b_shape = (96, cols) if layout == "TN" else (96, rows) if layout == "NN" else (rows, 96)
        references = [quantizer(torch.randn(s, device="cuda")) for s in (shape, b_shape)]
        references_a.append(references[0])
        references_b.append(references[1])
        operands_a.append(_copy_with_scale_layout(references[0], storage != "padded"))
        operands_b.append(
            _copy_with_scale_layout(
                references[1], storage == "compact" or (storage == "mixed" and i % 2 == 0)
            )
        )
        expected.append(general_gemm(*references, out_dtype=torch.bfloat16, layout=layout)[0])
    operands = operands_a + operands_b
    snapshots = [_snapshot(tensor) for tensor in operands]
    if grouped:
        actual = [torch.empty_like(tensor) for tensor in expected]
        general_grouped_gemm(
            references_a,
            references_b,
            expected,
            [None] * len(expected),
            torch.bfloat16,
            layout=layout,
        )
        general_grouped_gemm(
            operands_a, operands_b, actual, [None] * len(actual), torch.bfloat16, layout=layout
        )
    else:
        actual = [
            general_gemm(a, b, out_dtype=torch.bfloat16, layout=layout)[0]
            for a, b in zip(operands_a, operands_b)
        ]
    for result, reference in zip(actual, expected):
        torch.testing.assert_close(result, reference, rtol=0, atol=0)
    for tensor, snapshot in zip(operands, snapshots):
        _check_unchanged(tensor, snapshot)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Scale swizzling requires CUDA")
@pytest.mark.parametrize("rowwise", [False, True])
@pytest.mark.parametrize("grouped", [False, True])
@pytest.mark.parametrize("per_tile", [False, True])
def test_compact_scale_swizzle_bytes(rowwise, grouped, per_tile, monkeypatch):
    monkeypatch.setenv("NVTE_MXFP8_PER_TILE_SWIZZLE", str(int(per_tile)))
    shapes = [(32, 32), (64, 64), (96, 160), (128, 128), (160, 96), (4096, 4128)]
    if grouped:
        shapes = shapes * 12  # Cross the compact kernel's batch size.
    tensors, expected, inputs = [], [], []
    for i, (rows, cols) in enumerate(shapes):
        padded_rows, padded_cols = (rows + 127) // 128 * 128, (cols + 127) // 128 * 128
        scale_shape = (rows, cols // 32) if rowwise else (rows // 32, cols)
        padded_shape = (
            (padded_rows, padded_cols // 32) if rowwise else (padded_rows // 32, padded_cols)
        )
        scales = torch.randint(0, 256, scale_shape, dtype=torch.uint8, device="cuda")
        padded = torch.zeros(padded_shape, dtype=torch.uint8, device="cuda")
        padded[: scale_shape[0], : scale_shape[1]].copy_(scales)
        expected.append(
            swizzle_mxfp8_scale(padded_rows, padded_cols, padded, columnwise=not rowwise)
        )
        if grouped and i % 3 == 0:
            scales = padded  # Exercise mixed compact/padded batches.
        inputs.append((scales, scales.clone()))
        data = torch.empty((rows, cols), dtype=torch.uint8, device="cuda")
        tensors.append(
            MXFP8Tensor(
                shape=(rows, cols),
                dtype=torch.bfloat16,
                device="cuda",
                rowwise_data=data if rowwise else None,
                columnwise_data=None if rowwise else data,
                rowwise_scale_inv=scales if rowwise else None,
                columnwise_scale_inv=None if rowwise else scales,
                fp8_dtype=tex.DType.kFloat8E4M3,
                quantizer=MXFP8Quantizer(
                    tex.DType.kFloat8E4M3, rowwise=rowwise, columnwise=not rowwise
                ),
                with_gemm_swizzled_scales=False,
            )
        )
    if grouped:
        tex.multi_tensor_swizzle_scales_for_gemm_(tensors, rowwise, not rowwise)
    else:
        for tensor in tensors:
            tex.swizzle_scales_for_gemm_(tensor)
    for tensor, reference, (scales, before) in zip(tensors, expected, inputs):
        actual = tensor._rowwise_scale_inv if rowwise else tensor._columnwise_scale_inv
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)
        torch.testing.assert_close(scales, before, rtol=0, atol=0)
        assert tensor._with_gemm_swizzled_scales
