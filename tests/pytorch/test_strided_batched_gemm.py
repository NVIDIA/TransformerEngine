# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Tests for experimental strided batched GEMM."""

from __future__ import annotations

import math
from typing import Optional

import pytest
import torch

import transformer_engine.pytorch as te
from transformer_engine.pytorch import (
    Float8BlockQuantizer,
    Float8CurrentScalingQuantizer,
    Float8Quantizer,
    MXFP8Quantizer,
    NVFP4Quantizer,
    is_bf16_available,
)
from transformer_engine.pytorch.cpp_extensions import strided_batched_gemm
import transformer_engine_torch as tex

from utils import dtype_tols

fp8_available, reason_for_no_fp8 = te.is_fp8_available(return_reason=True)
mxfp8_available, reason_for_no_mxfp8 = te.is_mxfp8_available(return_reason=True)
nvfp4_available, reason_for_no_nvfp4 = te.is_nvfp4_available(return_reason=True)
fp8_block_scaling_available, reason_for_no_fp8_block_scaling = te.is_fp8_block_scaling_available(
    return_reason=True
)

# Quantization recipes
_quantization_list: list[Optional[str]] = [None]
if fp8_available:
    _quantization_list.extend(("fp8_delayed_scaling", "fp8_current_scaling"))
if mxfp8_available and tex.get_cublasLt_version() >= 120800:
    _quantization_list.append("mxfp8")
if nvfp4_available:
    _quantization_list.append("nvfp4")
if fp8_block_scaling_available:
    _quantization_list.append("fp8_block_scaling")


def _quantize(
    tensor: torch.Tensor,
    quantization: Optional[str],
    *,
    batch_dim: int,
    batch_size: int,
    rowwise: bool,
) -> torch.Tensor:
    """Quantize a GEMM operand

    MXFP8 data is quantized with only the usage needed by the GEMM.
    With batch_dim=-2, column-wise MXFP8 data is quantized as a 2D
    [R, G*C] view so that scaling blocks don't mix batch entries.

    """
    if quantization is None:
        return tensor
    if quantization == "mxfp8":
        quantizer = MXFP8Quantizer(
            fp8_dtype=tex.DType.kFloat8E4M3,
            rowwise=rowwise,
            columnwise=not rowwise,
        )
        quantizer.internal = True
        quantizer.optimize_for_gemm = False
        if not rowwise and batch_dim == -2:
            tensor = tensor.view(-1, batch_size * tensor.size(-1))
        return quantizer(tensor)
    if quantization == "fp8_delayed_scaling":
        quantizer = Float8Quantizer(
            scale=torch.ones(1, dtype=torch.float32, device=tensor.device),
            amax=torch.zeros(1, dtype=torch.float32, device=tensor.device),
            fp8_dtype=tex.DType.kFloat8E4M3,
        )
    elif quantization == "fp8_current_scaling":
        quantizer = Float8CurrentScalingQuantizer(
            fp8_dtype=tex.DType.kFloat8E4M3,
            device=tensor.device,
        )
    elif quantization == "fp8_block_scaling":
        quantizer = Float8BlockQuantizer(
            fp8_dtype=tex.DType.kFloat8E4M3,
            rowwise=True,
            columnwise=False,
            block_scaling_dim=1,
        )
    elif quantization == "nvfp4":
        quantizer = NVFP4Quantizer(
            with_rht=False,
            with_post_rht_amax=False,
            with_2d_quantization=False,
            stochastic_rounding=False,
            with_random_sign_mask=False,
        )
    else:
        raise ValueError(f"Unsupported quantization scheme ({quantization})")
    quantizer.set_usage(rowwise=True, columnwise=False)
    return quantizer(tensor)


def _to_reference(tensor: torch.Tensor, shape: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
    """FP64 CPU tensor with the operand values used by the GEMM"""
    if isinstance(tensor, te.QuantizedTensorStorage):
        tensor = tensor.dequantize(dtype=dtype)
    return tensor.reshape(shape).to(dtype=torch.float64, device="cpu")


def _batch_first(tensor: torch.Tensor, batch_dim: int) -> torch.Tensor:
    """Permute 3D tensor to [G, R, C]"""
    return tensor if batch_dim == 0 else tensor.transpose(0, 1)


def _batched_shape(batch_size: int, rows: int, cols: int, batch_dim: int) -> tuple[int, ...]:
    return (batch_size, rows, cols) if batch_dim == 0 else (rows, batch_size, cols)


@pytest.mark.parametrize("layout", ("TN", "NN", "NT"))
@pytest.mark.parametrize("batch_dim_a", (0, -2))
@pytest.mark.parametrize("batch_dim_b", (0, -2))
@pytest.mark.parametrize("batch_dim_out", (0, -2))
@pytest.mark.parametrize("quantization", _quantization_list)
def test_strided_batched_gemm(
    *,
    layout: str,
    batch_dim_a: int,
    batch_dim_b: int,
    batch_dim_out: int,
    quantization: Optional[str],
    accumulate: bool = False,
    batch_size: int = 3,
    m: int = 256,
    n: int = 128,
    k: int = 384,
    dtype: torch.dtype = torch.bfloat16,
) -> None:
    """Strided batched GEMM matches a reference GEMM for each batch entry

    Following the convention of general_gemm, matrices are row-major
    and out[i] = op(B[i]) @ op(A[i]), where op transposes if the
    corresponding entry in layout is "T".

    """
    if not is_bf16_available():
        pytest.skip("BF16 is not available")
    torch.manual_seed(1234)
    transa, transb = layout[0] == "T", layout[1] == "T"

    # Operand shapes
    a_rows, a_cols = (m, k) if transa else (k, m)
    b_rows, b_cols = (k, n) if transb else (n, k)
    a_shape = _batched_shape(batch_size, a_rows, a_cols, batch_dim_a)
    b_shape = _batched_shape(batch_size, b_rows, b_cols, batch_dim_b)
    out_shape = _batched_shape(batch_size, n, m, batch_dim_out)

    # Operands, quantized with scaling along the reduction dim
    a = torch.randn(a_shape, dtype=dtype, device="cuda")
    b = torch.randn(b_shape, dtype=dtype, device="cuda")
    out = torch.randn(out_shape, dtype=dtype, device="cuda")
    out_init = out.clone()
    a = _quantize(a, quantization, batch_dim=batch_dim_a, batch_size=batch_size, rowwise=transa)
    b = _quantize(b, quantization, batch_dim=batch_dim_b, batch_size=batch_size, rowwise=not transb)

    # Reference implementation
    a_ref = _batch_first(_to_reference(a, a_shape, dtype), batch_dim_a)
    b_ref = _batch_first(_to_reference(b, b_shape, dtype), batch_dim_b)
    if transa:
        a_ref = a_ref.transpose(1, 2)
    if transb:
        b_ref = b_ref.transpose(1, 2)
    out_ref = _batch_first(torch.bmm(b_ref, a_ref), batch_dim_out)
    if accumulate:
        out_ref = out_ref + _to_reference(out_init, out_shape, dtype)

    # Strided batched GEMM
    strided_batched_gemm(
        a,
        b,
        out,
        batch_dim_a=batch_dim_a,
        batch_dim_b=batch_dim_b,
        batch_dim_out=batch_dim_out,
        layout=layout,
        accumulate=accumulate,
    )

    # Operands are exactly representable, so error is from output
    # rounding and FP32 accumulation over k terms
    tols = dtype_tols(dtype)
    tols["atol"] = 1e-3
    if accumulate:
        # GEMM result may cancel with the accumulated values, so the
        # error scales with their magnitude (~sqrt(k)) instead
        tols["atol"] = tols["rtol"] * math.sqrt(k) / 4
    torch.testing.assert_close(out.cpu().double(), out_ref, **tols)


@pytest.mark.parametrize("quantization", _quantization_list)
def test_strided_batched_gemm_accumulate(quantization: Optional[str]) -> None:
    """Strided batched GEMM accumulates into the output"""
    test_strided_batched_gemm(
        layout="TN",
        batch_dim_a=0,
        batch_dim_b=-2,
        batch_dim_out=-2,
        quantization=quantization,
        accumulate=True,
    )
