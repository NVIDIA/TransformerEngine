# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Data-free descriptions of quantized operation outputs."""

from __future__ import annotations
from typing import Optional

import torch

from .tensor_spec import TensorSpec
from ..quantized_tensor import Quantizer


def fake_activation(
    input_: TensorSpec,
    quantizer: Optional[Quantizer] = None,
    *,
    gated: bool = False,
    dtype: Optional[torch.dtype] = None,
) -> TensorSpec:
    """Describe an activation output, including optional quantization."""
    shape = (*input_.shape[:-1], input_.shape[-1] // 2 if gated else input_.shape[-1])
    dtype = dtype or input_.dtype
    return TensorSpec(shape=shape, dtype=dtype, device=input_.device, quantizer=quantizer)


def fake_activation_backward(
    input_: TensorSpec,
    quantizer: Optional[Quantizer] = None,
    *,
    dtype: Optional[torch.dtype] = None,
) -> TensorSpec:
    """Describe the gradient of the original, pre-activation input."""
    return TensorSpec(
        shape=input_.shape,
        dtype=dtype or input_.dtype,
        device=input_.device,
        quantizer=quantizer,
    )


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
        else TensorSpec(
            shape=grad_output.shape,
            dtype=dtype,
            device=grad_output.device,
            quantizer=quantizer,
        )
    )
    return db, dx
