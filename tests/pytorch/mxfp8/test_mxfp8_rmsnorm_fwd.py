# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""RMSNorm forward with MXFP8 output from Transformer Engine's fused kernel."""

import os

import pytest
import torch

import transformer_engine.pytorch as te
import transformer_engine_torch as tex
from transformer_engine.pytorch import MXFP8Quantizer
from transformer_engine.pytorch.constants import TE_DType

from mxfp8_utils import swizzle_mxfp8_scale

recipe_available, reason_for_no_recipe = te.is_mxfp8_available(return_reason=True)


@pytest.mark.skipif(not recipe_available, reason=reason_for_no_recipe)
@pytest.mark.skipif(
    os.getenv("NVTE_NORM_FWD_USE_CUDNN", "0") == "1",
    reason="Tests Transformer Engine's fused kernel, not cuDNN",
)
@pytest.mark.parametrize("shape", [(128, 128), (256, 1024), (384, 7168)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("fp8_dtype", [tex.DType.kFloat8E4M3, tex.DType.kFloat8E5M2])
@pytest.mark.parametrize("zero_centered_gamma", [False, True])
@pytest.mark.parametrize("optimize_for_gemm", [False, True])
def test_rmsnorm_fwd_mxfp8(shape, dtype, fp8_dtype, zero_centered_gamma, optimize_for_gemm):
    """The fused output is the MXFP8 quantization of the FP32 normalized values."""
    torch.manual_seed(1234)
    rows, cols = shape
    eps = 1e-5
    x = torch.randn(rows, cols, dtype=dtype, device="cuda")
    weight = torch.randn(cols, dtype=dtype, device="cuda")

    quantizer = MXFP8Quantizer(fp8_dtype)
    quantizer.optimize_for_gemm = optimize_for_gemm
    out, _, rsigma = tex.rmsnorm_fwd(
        x, weight, eps, None, quantizer, TE_DType[dtype], 0, zero_centered_gamma
    )
    assert out._with_gemm_swizzled_scales == optimize_for_gemm

    torch.testing.assert_close(
        rsigma, torch.rsqrt(x.float().square().mean(dim=1) + eps), atol=0, rtol=1e-5
    )

    # Reference: the MXFP8 quantizer on the FP32 normalized values, with the kernel's rsigma.
    gamma = weight.float() + 1 if zero_centered_gamma else weight.float()
    y = (x.float() * rsigma.unsqueeze(1)) * gamma
    ref_quantizer = MXFP8Quantizer(fp8_dtype)
    ref_quantizer.optimize_for_gemm = False
    ref = ref_quantizer(y)

    for attr in ("_rowwise_data", "_columnwise_data"):
        torch.testing.assert_close(
            getattr(out, attr).view(torch.uint8),
            getattr(ref, attr).view(torch.uint8),
            atol=0,
            rtol=0,
            msg=attr,
        )
    for attr, columnwise in (("_rowwise_scale_inv", False), ("_columnwise_scale_inv", True)):
        expected = getattr(ref, attr)
        if optimize_for_gemm:
            expected = swizzle_mxfp8_scale(rows, cols, expected, columnwise=columnwise)
        torch.testing.assert_close(getattr(out, attr), expected, atol=0, rtol=0, msg=attr)


@pytest.mark.skipif(not recipe_available, reason=reason_for_no_recipe)
@pytest.mark.skipif(
    os.getenv("NVTE_NORM_FWD_USE_CUDNN", "0") == "1",
    reason="Tests Transformer Engine's fused kernel, not cuDNN",
)
@pytest.mark.parametrize("optimize_for_gemm", [False, True])
def test_rmsnorm_fwd_mxfp8_sm_margin(optimize_for_gemm):
    """With all SMs but one reserved, the kernels run over chunks of rows with the same output."""
    torch.manual_seed(1234)
    x = torch.randn(1024, 7168, dtype=torch.bfloat16, device="cuda")
    weight = torch.randn(7168, dtype=torch.bfloat16, device="cuda")
    sm_count = torch.cuda.get_device_properties(x.device).multi_processor_count

    outputs = []
    for sm_margin in (0, sm_count - 1):
        quantizer = MXFP8Quantizer(tex.DType.kFloat8E4M3)
        quantizer.optimize_for_gemm = optimize_for_gemm
        outputs.append(
            tex.rmsnorm_fwd(x, weight, 1e-5, None, quantizer, TE_DType[x.dtype], sm_margin, False)
        )
    (out, _, rsigma), (out_margin, _, rsigma_margin) = outputs
    torch.testing.assert_close(rsigma_margin, rsigma, atol=0, rtol=0)
    for attr in (
        "_rowwise_data",
        "_columnwise_data",
        "_rowwise_scale_inv",
        "_columnwise_scale_inv",
    ):
        torch.testing.assert_close(
            getattr(out_margin, attr), getattr(out, attr), atol=0, rtol=0, msg=attr
        )


@pytest.mark.skipif(not recipe_available, reason=reason_for_no_recipe)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_rmsnorm_fwd_mxfp8_2d_quantization(dtype):
    """The fused kernels quantize 1D blocks, so with 2D quantization the forward normalizes and
    then quantizes 32x32 blocks."""
    torch.manual_seed(1234)
    x = torch.randn(256, 1024, dtype=dtype, device="cuda")
    weight = torch.randn(1024, dtype=dtype, device="cuda")

    quantizer = MXFP8Quantizer(tex.DType.kFloat8E4M3, with_2d_quantization=True)
    out, _, _ = tex.rmsnorm_fwd(x, weight, 1e-5, None, quantizer, TE_DType[dtype], 0, False)

    y, _, _ = tex.rmsnorm_fwd(x, weight, 1e-5, None, None, TE_DType[dtype], 0, False)
    ref = MXFP8Quantizer(tex.DType.kFloat8E4M3, with_2d_quantization=True)(y)
    for attr in (
        "_rowwise_data",
        "_columnwise_data",
        "_rowwise_scale_inv",
        "_columnwise_scale_inv",
    ):
        torch.testing.assert_close(
            getattr(out, attr).view(torch.uint8),
            getattr(ref, attr).view(torch.uint8),
            atol=0,
            rtol=0,
            msg=attr,
        )
