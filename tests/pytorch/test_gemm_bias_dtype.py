# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Tests for non-FP8/FP4 general_gemm bias dtype vs output dtype contract.

cuBLASLt assumes bias dtype equals D when CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE is
unset. TE only sets that attribute on the FP8/FP4 path, so mismatched non-FP8
bias/out dtypes previously caused silent reinterpretation and OOB reads.
See NVIDIA/TransformerEngine#3562 and Megatron-LM#6000.
"""

import pytest
import torch

from transformer_engine.pytorch.cpp_extensions import general_gemm
from transformer_engine.pytorch import (
    is_fp8_available,
    Float8CurrentScalingQuantizer,
    DType,
)

fp8_available, reason_for_no_fp8 = is_fp8_available(return_reason=True)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize("cast_bias", [False, True])
def test_general_gemm_non_fp8_bias_dtype_must_match_out(cast_bias):
    """BF16 bias + FP32 out must fail; casting bias to FP32 must match addmm."""
    E, H, T = 128, 256, 64
    weight = torch.randn(E, H, dtype=torch.bfloat16, device="cuda")
    inp = torch.randn(T, H, dtype=torch.bfloat16, device="cuda")
    bias_bf16 = torch.randn(E, dtype=torch.bfloat16, device="cuda")
    bias = bias_bf16.float() if cast_bias else bias_bf16
    ref = torch.addmm(bias_bf16.float(), inp.float(), weight.float().t())
    if cast_bias:
        out = general_gemm(weight, inp, torch.float32, layout="TN", bias=bias)[0]
        assert torch.allclose(out, ref, atol=1e-3, rtol=1e-3)
    else:
        with pytest.raises(RuntimeError, match="bias dtype must match output dtype"):
            general_gemm(weight, inp, torch.float32, layout="TN", bias=bias)


@pytest.mark.parametrize(
    "dtype",
    [torch.float32, torch.bfloat16, torch.float16],
)
def test_general_gemm_matched_bias_and_out_dtype(dtype):
    """Matched non-FP8 bias and out dtypes remain valid and match addmm."""
    E, H, T = 64, 128, 32
    weight = torch.randn(E, H, dtype=dtype, device="cuda")
    inp = torch.randn(T, H, dtype=dtype, device="cuda")
    bias = torch.randn(E, dtype=dtype, device="cuda")
    out = general_gemm(weight, inp, dtype, layout="TN", bias=bias)[0]
    ref = torch.addmm(bias.float(), inp.float(), weight.float().t()).to(dtype)
    assert out.dtype == dtype
    assert torch.allclose(out.float(), ref.float(), atol=1e-2, rtol=1e-2)


def test_general_gemm_no_bias_fp32_out():
    """No-bias path is unaffected by the non-FP8 bias dtype guard."""
    E, H, T = 64, 128, 32
    weight = torch.randn(E, H, dtype=torch.bfloat16, device="cuda")
    inp = torch.randn(T, H, dtype=torch.bfloat16, device="cuda")
    out = general_gemm(weight, inp, torch.float32, layout="TN", bias=None)[0]
    ref = torch.matmul(inp.float(), weight.float().t())
    assert out.dtype == torch.float32
    assert torch.allclose(out, ref, atol=1e-3, rtol=1e-3)


def test_general_gemm_oob_style_packed_bias_raises():
    """Issue 3562 packing smoke: BF16 bias next to a buffer of ones must raise.

    Before the guard, cuBLAS reinterpreted BF16 bytes as FP32 and read into the
    neighboring ones, corrupting the second half of the bias epilogue.
    """
    E, H, T = 128, 256, 64
    weight = torch.randn(E, H, dtype=torch.bfloat16, device="cuda")
    inp = torch.randn(T, H, dtype=torch.bfloat16, device="cuda")
    # Pack BF16 bias then FP32 ones in one allocation (issue repro shape).
    packed = torch.empty(E * 2 + E, dtype=torch.bfloat16, device="cuda")
    bias_bf16 = packed[:E]
    bias_bf16.copy_(torch.randn(E, dtype=torch.bfloat16, device="cuda"))
    ones_view = packed[E:].view(torch.float32)[:E]
    ones_view.fill_(1.0)
    with pytest.raises(RuntimeError, match="bias dtype must match output dtype"):
        general_gemm(weight, inp, torch.float32, layout="TN", bias=bias_bf16)


@pytest.mark.skipif(not fp8_available, reason=reason_for_no_fp8)
def test_fp8_gemm_bf16_bias_still_allowed():
    """FP8 path may keep BF16 bias with BF16 D (BIAS_DATA_TYPE is set there)."""
    E, H, T = 64, 128, 32
    weight_hp = torch.randn(E, H, dtype=torch.bfloat16, device="cuda")
    inp_hp = torch.randn(T, H, dtype=torch.bfloat16, device="cuda")
    bias = torch.randn(E, dtype=torch.bfloat16, device="cuda")
    quantizer = Float8CurrentScalingQuantizer(fp8_dtype=DType.kFloat8E4M3, device="cuda")
    weight = quantizer(weight_hp)
    inp = quantizer(inp_hp)
    out = general_gemm(weight, inp, torch.bfloat16, layout="TN", bias=bias)[0]
    assert out.dtype == torch.bfloat16
    assert out.shape == (T, E)
