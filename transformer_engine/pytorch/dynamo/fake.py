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
    # Keep shape predicates visible to Dynamo so only alignment, not batch size, is guarded.
    rows, cols = math.prod(shape[:-1]), shape[-1]
    spec.with_gemm_swizzled_scales = alignment is not None and bool(
        rows % alignment[0] == 0 and cols % alignment[1] == 0
    )
    return spec


def fake_bgrad(
    grad_output: TensorSpec,
    quantizer: Optional[Quantizer] = None,
    *,
    dtype: Optional[torch.dtype] = None,
) -> tuple[TensorSpec, TensorSpec]:
    """Describe bias reduction and optional gradient quantization, in native order."""
    dtype = dtype or grad_output.dtype
    db = TensorSpec(shape=(grad_output.shape[-1],), dtype=dtype, device=grad_output.device)
    dx = (
        grad_output
        if quantizer is None and dtype == grad_output.dtype
        else _make_fused_quantization_output_spec(
            grad_output.shape, dtype, grad_output.device, quantizer
        )
    )
    return db, dx
