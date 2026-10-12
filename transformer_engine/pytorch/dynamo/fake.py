# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Data-free descriptions of quantized operation outputs."""

from __future__ import annotations
import math
import os
from typing import Optional

import torch

from .tensor_spec import TensorSpec
from ..quantized_tensor import Quantizer
from ..tensor.nvfp4_tensor import NVFP4Quantizer
from ..utils import get_device_compute_capability


def _make_fused_quantization_output_spec(
    shape: tuple[int, ...],
    dtype: torch.dtype,
    device: Optional[torch.device],
    quantizer: Optional[Quantizer],
) -> TensorSpec:
    """Describe a native output without a separate post-quantize swizzle."""
    spec = TensorSpec(shape=tuple(shape), dtype=dtype, device=device, quantizer=quantizer)
    quantizer = spec.quantizer
    if not isinstance(quantizer, NVFP4Quantizer) or not quantizer.optimize_for_gemm:
        return spec
    rows, cols = math.prod(shape[:-1]), shape[-1]
    if not (10, 0) <= get_device_compute_capability(device) <= (11, 0):
        spec.with_gemm_swizzled_scales = False
    elif quantizer.with_rht:
        spec.with_gemm_swizzled_scales = bool(
            not int(os.getenv("NVTE_NVFP4_DISABLE_RHT_CAST_FUSION", "0") or "0")
            and rows % 64 == 0
            and cols % 128 == 0
        )
    else:
        spec.with_gemm_swizzled_scales = bool(
            quantizer.with_2d_quantization
            and not quantizer.row_scaled_nvfp4
            and not quantizer.nvfp4_use_4over6
            and rows % 128 == 0
            and cols % 128 == 0
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
