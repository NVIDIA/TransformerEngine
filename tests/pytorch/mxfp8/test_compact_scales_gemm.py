# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# See LICENSE for license information.

"""GEMM preparation accepts compact unswizzled MXFP8 storage."""

import pytest
import torch
from mxfp8_utils import swizzle_mxfp8_scale

import transformer_engine.pytorch as te
import transformer_engine_torch as tex
from transformer_engine.pytorch.cpp_extensions import general_gemm
from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer, MXFP8Tensor


@pytest.mark.skipif(not te.is_mxfp8_available(), reason="MXFP8 requires Blackwell")
@pytest.mark.parametrize("shape", [(64, 64), (96, 160), (256, 128)])
@pytest.mark.parametrize("layout", ["TN", "NN", "NT"])
@pytest.mark.parametrize("compact", [False, True])
def test_compact_scales_gemm(shape, layout, compact):
    quantizer = MXFP8Quantizer(tex.DType.kFloat8E4M3)
    torch.manual_seed(1234)
    m, n = shape
    # general_gemm uses column-major operand conventions.
    b_shape = (96, n) if layout == "TN" else (96, m) if layout == "NN" else (m, 96)
    inputs = [torch.randn(s, device="cuda", dtype=torch.float32) for s in (shape, b_shape)]
    references = [quantizer(x) for x in inputs]
    operands = []
    snapshots = []
    for x, reference in zip(inputs, references):
        rows, cols = x.shape
        row_shape = (
            (rows, cols // 32) if compact else ((rows + 127) // 128 * 128, (cols + 127) // 128 * 4)
        )
        col_shape = (
            (rows // 32, cols) if compact else ((rows + 127) // 128 * 4, (cols + 127) // 128 * 128)
        )
        tensor = MXFP8Tensor(
            shape=x.shape,
            dtype=torch.bfloat16,
            device=x.device,
            rowwise_data=torch.empty_like(x, dtype=torch.uint8),
            columnwise_data=torch.empty_like(x, dtype=torch.uint8),
            rowwise_scale_inv=torch.empty(row_shape, device=x.device, dtype=torch.uint8),
            columnwise_scale_inv=torch.empty(col_shape, device=x.device, dtype=torch.uint8),
            fp8_dtype=tex.DType.kFloat8E4M3,
            quantizer=quantizer,
            with_gemm_swizzled_scales=False,
        )
        tensor._rowwise_data.copy_(reference._rowwise_data)
        tensor._columnwise_data.copy_(reference._columnwise_data)
        tensor._rowwise_scale_inv.copy_(
            reference._rowwise_scale_inv[: row_shape[0], : row_shape[1]]
        )
        tensor._columnwise_scale_inv.copy_(
            reference._columnwise_scale_inv[: col_shape[0], : col_shape[1]]
        )
        operands.append(tensor)
        snapshots.append(
            [
                (s.data_ptr(), s.clone())
                for s in (tensor._rowwise_scale_inv, tensor._columnwise_scale_inv)
            ]
        )
    expected = general_gemm(*references, out_dtype=torch.bfloat16, layout=layout)[0]
    actual = general_gemm(*operands, out_dtype=torch.bfloat16, layout=layout)[0]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for tensor, snapshot in zip(operands, snapshots):
        assert not tensor._with_gemm_swizzled_scales
        for scale, (pointer, before) in zip(
            (tensor._rowwise_scale_inv, tensor._columnwise_scale_inv), snapshot
        ):
            assert scale.data_ptr() == pointer
            torch.testing.assert_close(scale, before, rtol=0, atol=0)


@pytest.mark.skipif(not te.is_mxfp8_available(), reason="MXFP8 requires Blackwell")
@pytest.mark.parametrize(
    "shape", [(32, 32), (64, 64), (96, 160), (128, 128), (160, 96), (4096, 4128)]
)
@pytest.mark.parametrize("rowwise", [False, True])
def test_compact_scale_swizzle_bytes(shape, rowwise):
    m, n = shape
    scale_shape = (m, n // 32) if rowwise else (m // 32, n)
    scales = torch.randint(0, 256, scale_shape, dtype=torch.uint8, device="cuda")
    padded_m, padded_n = (m + 127) // 128 * 128, (n + 127) // 128 * 128
    padded_shape = (padded_m, padded_n // 32) if rowwise else (padded_m // 32, padded_n)
    padded = torch.zeros(padded_shape, dtype=torch.uint8, device="cuda")
    padded[: scale_shape[0], : scale_shape[1]].copy_(scales)
    expected = swizzle_mxfp8_scale(padded_m, padded_n, padded, columnwise=not rowwise)
    tensor = MXFP8Tensor(
        shape=shape,
        dtype=torch.bfloat16,
        device="cuda",
        rowwise_data=(torch.empty(shape, dtype=torch.uint8, device="cuda") if rowwise else None),
        columnwise_data=(None if rowwise else torch.empty(shape, dtype=torch.uint8, device="cuda")),
        rowwise_scale_inv=scales if rowwise else None,
        columnwise_scale_inv=None if rowwise else scales,
        fp8_dtype=tex.DType.kFloat8E4M3,
        quantizer=MXFP8Quantizer(tex.DType.kFloat8E4M3, rowwise=rowwise, columnwise=not rowwise),
        with_gemm_swizzled_scales=False,
    )
    tex.swizzle_scales_for_gemm_(tensor)
    actual = tensor._rowwise_scale_inv if rowwise else tensor._columnwise_scale_inv
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert tensor._with_gemm_swizzled_scales
