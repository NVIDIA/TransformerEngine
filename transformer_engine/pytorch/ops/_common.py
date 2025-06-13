# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Helper functions used in fusible operations."""

from __future__ import annotations
from typing import Any, Iterable, Optional

import torch

from transformer_engine_torch import FP8TensorMeta
from .. import torch_version
from ..fp8 import FP8GlobalStateManager
from ..tensor.float8_tensor import Float8Tensor
from ..utils import (
    canonicalize_device,
    canonicalize_dtype,
    devices_match,
)


def is_float8_tensor(tensor: Any) -> bool:
    """Check if object is a `Float8Tensor`"""
    return isinstance(tensor, Float8Tensor)


def convert_tensor(
    tensor: torch.Tensor | Float8Tensor,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor | Float8Tensor:
    """Convert tensor attributes, keeping same data if possible"""

    # Default kwargs
    if dtype is None:
        dtype = tensor.dtype

    if tensor.is_contiguous():
        if dtype == tensor.dtype:
            return tensor
        if is_float8_tensor(tensor):
            out = Float8Tensor.make_like(tensor, dtype=dtype)
            out.data = tensor._data
            return out
        else:
            return tensor.to(dtype=dtype)
    else:
        if is_float8_tensor(tensor):
            out = Float8Tensor.make_like(tensor, dtype=dtype)
            out.data = tensor._data.contiguous()
            return out
        else:
            return tensor.to(dtype).contiguous()


def reshape(
    tensor: torch.Tensor | Float8Tensor,
    shape: Iterable[int],
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor | Float8Tensor:
    """Reshape tensor, keeping same data if possible"""
    tensor = convert_tensor(tensor, dtype=dtype)
    return tensor.reshape(*shape)


def maybe_autocast_dtype(
    *,
    device_type: str = "cuda",
    default_dtype: Optional[torch.dtype] = None,
) -> torch.dtype:
    """Get autocast dtype if enabled"""

    if torch_version() >= (2, 4, 3):
        if torch.is_autocast_enabled(device_type):
            return torch.get_autocast_dtype(device_type)
    else:
        if torch.is_autocast_enabled():
            return torch.get_autocast_gpu_dtype()
    return canonicalize_dtype(default_dtype)


def get_fp8_meta_from_fp8_tensor(tensor: Float8Tensor) -> tuple[FP8TensorMeta, int]:
    """Get FP8TensorMeta object and index corresponding to Float8Tensor

    Constructs FP8TensorMeta if needed.

    """

    # Check if tensor already has FP8 metadata
    if tensor._fp8_meta is not None:
        key = FP8GlobalStateManager.get_meta_tensor_key(
            forward=tensor._fp8_meta_forward,
        )
        return tensor._fp8_meta[key], tensor._fp8_meta_index

    # Create FP8TensorMeta class
    fp8_meta = FP8TensorMeta()
    fp8_meta.scale = tensor._scale_inv.reciprocal()
    fp8_meta.amax_history = torch.empty(1, 1, dtype=torch.float32, device=tensor.device)
    fp8_meta.scale_inv = tensor._scale_inv
    return fp8_meta, 0
