# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import pytest
import torch

import transformer_engine.pytorch as te
from transformer_engine.pytorch.tensor.mxfp4_tensor import MXFP4Quantizer, MXFP4Tensor
from transformer_engine.pytorch.tensor.storage.mxfp4_tensor_storage import MXFP4TensorStorage

from mxfp4_utils import (
    BLOCK_SIZE,
    dequantize_mxfp4_columnwise,
    dequantize_mxfp4_rowwise,
    quantize_mxfp4_columnwise,
    quantize_mxfp4_rowwise,
    unpack_e2m1,
)

mxfp8_available, reason_for_no_mxfp8 = te.is_mxfp8_available(return_reason=True)
pytestmark = pytest.mark.skipif(not mxfp8_available, reason=reason_for_no_mxfp8)


def _normalize_negative_zero(codes: torch.Tensor) -> torch.Tensor:
    return torch.where(codes == 0x8, torch.zeros_like(codes), codes)


def _make_input(shape, dtype, seed=0):
    torch.manual_seed(seed)
    x = torch.randn(shape, dtype=torch.float32, device="cuda")
    row_scale = torch.logspace(-3, 3, shape[0], device="cuda").unsqueeze(1)
    col_scale = torch.logspace(-2, 2, shape[1], device="cuda").unsqueeze(0)
    x = x * row_scale * col_scale
    x[0, :BLOCK_SIZE] = 0
    return x.to(dtype)


@pytest.mark.parametrize("shape", [(32, 32), (128, 256), (96, 160), (256, 1024), (1024, 4096)])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32, torch.float16])
@pytest.mark.parametrize("swizzled", [False, True])
def test_mxfp4_quantize_matches_reference(shape, dtype, swizzled):
    x = _make_input(shape, dtype)
    quantizer = MXFP4Quantizer(rowwise=True, columnwise=True)
    quantizer.optimize_for_gemm = swizzled
    qx = quantizer(x)
    assert isinstance(qx, MXFP4Tensor)
    assert qx.shape == x.shape
    assert qx._rowwise_data.shape == (shape[0], shape[1] // 2)
    assert qx._columnwise_data.shape == (shape[0], shape[1] // 2)
    assert qx._with_gemm_swizzled_scales == swizzled

    ref_row_codes, ref_row_scales = quantize_mxfp4_rowwise(x)
    ref_col_codes, ref_col_scales = quantize_mxfp4_columnwise(x)

    row_codes = unpack_e2m1(qx._rowwise_data)
    col_codes = unpack_e2m1(qx._columnwise_data)
    torch.testing.assert_close(_normalize_negative_zero(row_codes), ref_row_codes, rtol=0, atol=0)
    torch.testing.assert_close(_normalize_negative_zero(col_codes), ref_col_codes, rtol=0, atol=0)

    if not swizzled:
        rows, cols = shape
        row_scales = qx._rowwise_scale_inv[:rows, : cols // BLOCK_SIZE]
        col_scales = qx._columnwise_scale_inv[: rows // BLOCK_SIZE, :cols]
        torch.testing.assert_close(row_scales, ref_row_scales, rtol=0, atol=0)
        torch.testing.assert_close(col_scales, ref_col_scales, rtol=0, atol=0)

    ref_row = dequantize_mxfp4_rowwise(ref_row_codes, ref_row_scales)
    ref_col = dequantize_mxfp4_columnwise(ref_col_codes, ref_col_scales)
    torch.testing.assert_close(qx.dequantize(dtype=torch.float32), ref_row, rtol=0, atol=0)

    col_quantizer = MXFP4Quantizer(rowwise=False, columnwise=True)
    col_quantizer.optimize_for_gemm = swizzled
    qx_col = col_quantizer(x)
    assert qx_col._rowwise_data is None
    torch.testing.assert_close(qx_col.dequantize(dtype=torch.float32), ref_col, rtol=0, atol=0)


@pytest.mark.parametrize("shape", [(64, 96), (8, 4, 128)])
def test_mxfp4_quantize_rowwise_only_and_internal(shape):
    x = torch.randn(shape, dtype=torch.bfloat16, device="cuda")
    quantizer = MXFP4Quantizer(rowwise=True, columnwise=False)
    quantizer.internal = True
    qx = quantizer(x)
    assert type(qx) is MXFP4TensorStorage
    assert qx._columnwise_data is None
    assert qx.size() == x.shape
    x2d = x.reshape(-1, shape[-1])
    ref_codes, ref_scales = quantize_mxfp4_rowwise(x2d)
    ref = dequantize_mxfp4_rowwise(ref_codes, ref_scales).view(shape)
    torch.testing.assert_close(qx.dequantize(dtype=torch.float32), ref, rtol=0, atol=0)


def test_mxfp4_quantize_in_place_update():
    x = torch.randn((128, 256), dtype=torch.bfloat16, device="cuda")
    y = torch.randn((128, 256), dtype=torch.bfloat16, device="cuda")
    quantizer = MXFP4Quantizer()
    qx = quantizer(x)
    qx.quantize_(y)
    ref_codes, ref_scales = quantize_mxfp4_rowwise(y)
    ref = dequantize_mxfp4_rowwise(ref_codes, ref_scales)
    torch.testing.assert_close(qx.dequantize(dtype=torch.float32), ref, rtol=0, atol=0)

    noop = torch.ones(1, dtype=torch.float32, device="cuda")
    quantizer.update_quantized(x, qx, noop_flag=noop)
    torch.testing.assert_close(qx.dequantize(dtype=torch.float32), ref, rtol=0, atol=0)


@pytest.mark.parametrize("swizzled", [False, True])
def test_mxfp4_tensor_view(swizzled):
    x = _make_input((1024, 128), torch.bfloat16)
    quantizer = MXFP4Quantizer()
    quantizer.optimize_for_gemm = swizzled
    qx = quantizer(x).requires_grad_()
    ref = qx.dequantize(dtype=torch.float32).detach()

    for qy in (qx.view((32, 32, 128)), qx.view(32, -1, 128), qx.reshape(32, 32, 128)):
        assert isinstance(qy, MXFP4Tensor)
        assert qy.shape == (32, 32, 128)
        assert qy._with_gemm_swizzled_scales == swizzled
        torch.testing.assert_close(
            qy.dequantize(dtype=torch.float32), ref.view(32, 32, 128), rtol=0, atol=0
        )

    qx.view(-1, 128).dequantize(dtype=torch.float32).sum().backward()
    assert qx.grad is not None
    assert qx.grad.shape == qx.shape

    with pytest.raises(RuntimeError):
        qx.view(2048, 64)


@pytest.mark.parametrize("src_swizzled", [False, True])
@pytest.mark.parametrize("dst_swizzled", [False, True])
def test_mxfp4_tensor_copy_between_scale_layouts(src_swizzled, dst_swizzled):
    src_quantizer = MXFP4Quantizer()
    src_quantizer.optimize_for_gemm = src_swizzled
    dst_quantizer = MXFP4Quantizer()
    dst_quantizer.optimize_for_gemm = dst_swizzled
    src = src_quantizer(_make_input((256, 512), torch.bfloat16, seed=0))
    dst = dst_quantizer(_make_input((256, 512), torch.bfloat16, seed=1))

    dst.copy_(src)
    assert dst._with_gemm_swizzled_scales == dst_swizzled
    torch.testing.assert_close(
        dst.dequantize(dtype=torch.float32), src.dequantize(dtype=torch.float32), rtol=0, atol=0
    )


def test_mxfp4_quantize_rejects_misaligned_shape():
    quantizer = MXFP4Quantizer()
    with pytest.raises(RuntimeError):
        quantizer(torch.randn((64, 48), dtype=torch.bfloat16, device="cuda"))
