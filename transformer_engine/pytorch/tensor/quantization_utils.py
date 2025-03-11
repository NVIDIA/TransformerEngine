# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Utility functions related to quantized tensors."""

from typing import Any
from .quantized_tensor import QuantizedTensor
from ._internal.float8_tensor_base import Float8TensorBase
from ._internal.mxfp8_tensor_base import MXFP8TensorBase

def is_quantized_tensor(t: Any) -> bool:
    """Check if the input is a quantized tensor
    (either internal or external)"""
    return isinstance(t, (QuantizedTensor,
                          Float8TensorBase,
                          MXFP8TensorBase))
