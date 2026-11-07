# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Data-free descriptions of quantized operation outputs."""

from __future__ import annotations
import math
from typing import Optional

import torch

from .tensor_spec import TensorSpec
from ..quantized_tensor import Quantizer


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
        else TensorSpec(
            shape=grad_output.shape,
            dtype=grad_output.dtype,
            device=grad_output.device,
            quantizer=quantizer,
        )
    )
    return db, dx


def fake_activation(
    input_: TensorSpec,
    quantizer: Optional[Quantizer] = None,
    *,
    gated: bool = False,
    dtype: Optional[torch.dtype] = None,
) -> TensorSpec:
    """Describe an optionally quantized activation output."""
    shape = (*input_.shape[:-1], input_.shape[-1] // 2 if gated else input_.shape[-1])
    return TensorSpec(
        shape=shape, dtype=dtype or input_.dtype, device=input_.device, quantizer=quantizer
    )


def fake_norm(
    input_: TensorSpec,
    quantizer: Optional[Quantizer] = None,
    *,
    normalization: str = "LayerNorm",
    dtype: Optional[torch.dtype] = None,
) -> tuple[TensorSpec, Optional[TensorSpec], TensorSpec]:
    """Describe norm output, mean and reciprocal standard deviation."""
    if normalization not in ("LayerNorm", "RMSNorm"):
        raise ValueError(f"Unsupported normalization: {normalization}")
    out = TensorSpec(
        shape=input_.shape, dtype=dtype or input_.dtype, device=input_.device, quantizer=quantizer
    )
    rows = math.prod(input_.shape[:-1])
    mean = (
        TensorSpec(shape=(rows,), dtype=torch.float32, device=input_.device)
        if normalization == "LayerNorm"
        else None
    )
    rsigma = TensorSpec(shape=(rows,), dtype=torch.float32, device=input_.device)
    return out, mean, rsigma
