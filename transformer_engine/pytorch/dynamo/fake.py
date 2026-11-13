# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Data-free descriptions of quantized operation outputs."""

from __future__ import annotations
from typing import Optional

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
