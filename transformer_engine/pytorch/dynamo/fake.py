# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Data-free descriptions of quantized operation outputs."""

from __future__ import annotations
import math
import os
from typing import Optional

import torch
import transformer_engine_torch as tex

from .tensor_spec import TensorSpec
from ..quantized_tensor import Quantizer
from ..tensor.mxfp8_tensor import MXFP8Quantizer
from ..tensor.nvfp4_tensor import NVFP4Quantizer
from ..utils import get_device_compute_capability


@torch.compiler.assume_constant_result
def _nvfp4_gemm_swizzled_scales_alignment(
    optimize_for_gemm: bool,
    with_rht: bool,
    with_2d_quantization: bool,
    row_scaled_nvfp4: bool,
    nvfp4_use_4over6: bool,
    sm_arch: int,
    disable_rht_cast_fusion: bool,
) -> Optional[tuple[int, int]]:
    return tex.nvfp4_gemm_swizzled_scales_alignment(
        optimize_for_gemm,
        with_rht,
        with_2d_quantization,
        row_scaled_nvfp4,
        nvfp4_use_4over6,
        sm_arch,
        disable_rht_cast_fusion,
    )


def _make_fused_quantization_output_spec(
    shape: tuple[int, ...],
    dtype: torch.dtype,
    device: Optional[torch.device],
    quantizer: Optional[Quantizer],
) -> TensorSpec:
    """Describe a native output without a separate post-quantize swizzle."""
    spec = TensorSpec(shape=tuple(shape), dtype=dtype, device=device, quantizer=quantizer)
    quantizer = spec.quantizer
    if not isinstance(quantizer, NVFP4Quantizer):
        return spec
    major, minor = get_device_compute_capability(device)
    # Direct environ access lets Dynamo guard changes to the flag.
    disable_rht_cast_fusion = bool(
        int(os.environ.get("NVTE_NVFP4_DISABLE_RHT_CAST_FUSION", "0") or "0")
    )
    alignment = _nvfp4_gemm_swizzled_scales_alignment(
        quantizer.optimize_for_gemm,
        quantizer.with_rht,
        quantizer.with_2d_quantization,
        quantizer.row_scaled_nvfp4,
        quantizer.nvfp4_use_4over6,
        major * 10 + minor,
        disable_rht_cast_fusion,
    )
    # Keep divisibility checks visible to Dynamo without specializing the dimensions.
    rows, cols = math.prod(shape[:-1]), shape[-1]
    if alignment is None:
        spec.with_gemm_swizzled_scales = False
    else:
        row_alignment, col_alignment = alignment
        spec.with_gemm_swizzled_scales = bool(
            rows % row_alignment == 0 and cols % col_alignment == 0
        )
    return spec


def fake_bgrad(
    grad_output: TensorSpec,
    quantizer: Optional[Quantizer] = None,
) -> tuple[TensorSpec, TensorSpec]:
    """Describe bias reduction and optional gradient quantization, in native order."""
    db = TensorSpec(
        shape=(grad_output.shape[-1],), dtype=grad_output.dtype, device=grad_output.device
    )
    dx = (
        grad_output
        if quantizer is None
        else _make_fused_quantization_output_spec(
            grad_output.shape, grad_output.dtype, grad_output.device, quantizer
        )
    )
    return db, dx


def fake_activation(
    input_: TensorSpec,
    quantizer: Optional[Quantizer] = None,
    *,
    gated: bool = False,
    dtype: Optional[torch.dtype] = None,
    fused_quantization: bool = True,
) -> TensorSpec:
    """Describe an activation output, including optional fused quantization."""
    shape = (*input_.shape[:-1], input_.shape[-1] // 2 if gated else input_.shape[-1])
    dtype = dtype or input_.dtype
    if fused_quantization:
        return _make_fused_quantization_output_spec(shape, dtype, input_.device, quantizer)
    return TensorSpec(shape=shape, dtype=dtype, device=input_.device, quantizer=quantizer)


def fake_norm(
    input_: TensorSpec,
    quantizer: Optional[Quantizer] = None,
    *,
    normalization: str = "LayerNorm",
    dtype: Optional[torch.dtype] = None,
) -> tuple[TensorSpec, Optional[TensorSpec], TensorSpec]:
    """Describe newly allocated norm output, mean and reciprocal standard deviation."""
    if normalization not in ("LayerNorm", "RMSNorm"):
        raise ValueError(f"Unsupported normalization: {normalization}")
    out = _make_fused_quantization_output_spec(
        input_.shape, dtype or input_.dtype, input_.device, quantizer
    )
    rows, cols = math.prod(input_.shape[:-1]), input_.shape[-1]
    if (
        isinstance(quantizer, MXFP8Quantizer)
        and int(os.getenv("NVTE_NORM_FWD_USE_CUDNN", "0") or "0")
        and rows % 128 == 0
        and cols % 128 == 0
    ):
        if normalization == "LayerNorm" or not quantizer.with_2d_quantization:
            out.with_gemm_swizzled_scales = False
    mean = (
        TensorSpec(shape=(rows,), dtype=torch.float32, device=input_.device)
        if normalization == "LayerNorm"
        else None
    )
    rsigma = TensorSpec(shape=(rows,), dtype=torch.float32, device=input_.device)
    return out, mean, rsigma
