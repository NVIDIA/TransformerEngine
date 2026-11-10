# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import pytest
import torch

import transformer_engine.pytorch as te
import transformer_engine_torch as tex
from transformer_engine.pytorch.cpp_extensions import general_gemm
from transformer_engine.pytorch.tensor.mxfp4_tensor import MXFP4Quantizer
from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer

mxfp8_available, reason_for_no_mxfp8 = te.is_mxfp8_available(return_reason=True)
pytestmark = pytest.mark.skipif(not mxfp8_available, reason=reason_for_no_mxfp8)


def _quantize(quantizer, x, swizzled):
    quantizer.optimize_for_gemm = swizzled
    return quantizer(x)


def _mxfp8_quantizer(rowwise=True, columnwise=False):
    return MXFP8Quantizer(tex.DType.kFloat8E4M3, rowwise=rowwise, columnwise=columnwise)


# (tokens, in_features, out_features)
SHAPES = [
    (128, 128, 128),
    (256, 512, 384),
    (96, 256, 128),
    (1056, 1024, 2048),
    (2048, 4096, 1024),
]


@pytest.mark.parametrize("tokens,in_features,out_features", SHAPES)
@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("swizzled", [True, False])
@pytest.mark.parametrize("use_bias", [False, True])
def test_mxfp4_mxfp8_fprop_gemm(tokens, in_features, out_features, out_dtype, swizzled, use_bias):
    torch.manual_seed(0)
    x = torch.randn((tokens, in_features), dtype=torch.bfloat16, device="cuda")
    w = torch.randn((out_features, in_features), dtype=torch.bfloat16, device="cuda")
    bias = torch.randn(out_features, dtype=torch.bfloat16, device="cuda") if use_bias else None

    qx = _quantize(_mxfp8_quantizer(), x, swizzled)
    qw = _quantize(MXFP4Quantizer(rowwise=True, columnwise=False), w, swizzled)

    out, *_ = general_gemm(qw, qx, out_dtype=out_dtype, layout="TN", bias=bias)

    ref = qx.dequantize(dtype=torch.float32) @ qw.dequantize(dtype=torch.float32).t()
    if use_bias:
        ref = ref + bias.float()
    if out_dtype == torch.float32:
        torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-3)
    else:
        torch.testing.assert_close(out, ref.to(out_dtype), rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("tokens,in_features,out_features", SHAPES)
@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("swizzled", [True, False])
def test_mxfp4_mxfp8_dgrad_gemm(tokens, in_features, out_features, out_dtype, swizzled):
    torch.manual_seed(1)
    dy = torch.randn((tokens, out_features), dtype=torch.bfloat16, device="cuda")
    w = torch.randn((out_features, in_features), dtype=torch.bfloat16, device="cuda")

    qdy = _quantize(_mxfp8_quantizer(), dy, swizzled)
    qw = _quantize(MXFP4Quantizer(rowwise=False, columnwise=True), w, swizzled)

    dx, *_ = general_gemm(qw, qdy, out_dtype=out_dtype, layout="NN", grad=True)

    ref = qdy.dequantize(dtype=torch.float32) @ qw.dequantize(dtype=torch.float32)
    if out_dtype == torch.float32:
        torch.testing.assert_close(dx, ref, rtol=1e-5, atol=1e-3)
    else:
        torch.testing.assert_close(dx, ref.to(out_dtype), rtol=1e-2, atol=1e-2)


def test_mxfp4_mxfp8_gemm_accumulate():
    torch.manual_seed(2)
    x = torch.randn((256, 512), dtype=torch.bfloat16, device="cuda")
    w = torch.randn((256, 512), dtype=torch.bfloat16, device="cuda")
    qx = _quantize(_mxfp8_quantizer(), x, True)
    qw = _quantize(MXFP4Quantizer(rowwise=True, columnwise=False), w, True)
    out = torch.randn((256, 256), dtype=torch.float32, device="cuda")
    ref = out + qx.dequantize(dtype=torch.float32) @ qw.dequantize(dtype=torch.float32).t()
    general_gemm(qw, qx, out_dtype=torch.float32, layout="TN", out=out, accumulate=True)
    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-3)


@pytest.mark.parametrize(
    "in_features,out_features,layout",
    [(160, 128, "TN"), (160, 128, "NN")],
)
def test_mxfp4_mxfp8_gemm_rejects_misaligned_dims(in_features, out_features, layout):
    x_features = out_features if layout == "NN" else in_features
    x = torch.randn((128, x_features), dtype=torch.bfloat16, device="cuda")
    w = torch.randn((out_features, in_features), dtype=torch.bfloat16, device="cuda")
    qx = _quantize(_mxfp8_quantizer(), x, True)
    qw = _quantize(MXFP4Quantizer(), w, True)
    with pytest.raises(RuntimeError, match="multiple of 128"):
        general_gemm(qw, qx, out_dtype=torch.bfloat16, layout=layout)
