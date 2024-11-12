# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Helper functions used in fusible operations."""

from __future__ import annotations
from typing import Any, Iterable, Optional

import torch

from transformer_engine_torch import FP8TensorMeta
from ..fp8 import FP8GlobalStateManager
from ..tensor import Float8Tensor
from ..utils import (
    canonicalize_device,  # pylint: disable=unused-import
    canonicalize_dtype,  # pylint: disable=unused-import
    devices_match,  # pylint: disable=unused-import
)


def is_float8_tensor(tensor: Any) -> bool:
    """Check if object is a `Float8Tensor`"""
    return isinstance(tensor, Float8Tensor)


def convert_tensor(
    tensor: torch.Tensor | Float8Tensor,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    memory_format: torch.memory_format = torch.preserve_format,
) -> torch.Tensor | Float8Tensor:
    """Convert tensor attributes, keeping same data if possible"""

    # Default kwargs
    if device is None:
        device = tensor.device
    device = canonicalize_device(device)
    if dtype is None:
        dtype = tensor.dtype
    dtype = canonicalize_dtype(dtype)

    # Make sure output is detached from autograd graph
    tensor = tensor.detach()

    # Return immediately if tensor already has desired attributes
    if devices_match(device, tensor.device) and dtype == tensor.dtype:
        if memory_format == torch.preserve_format or tensor.is_contiguous(
            memory_format=memory_format
        ):
            return tensor

    # Convert FP8 tensor
    if is_float8_tensor(tensor):
        data = tensor._data
        if not devices_match(device, data.device):
            data = data.to(device=device)
        if memory_format != torch.preserve_format and not data.is_contiguous(
            memory_format=memory_format
        ):
            # Note: torch.Tensor.to ignores memory_format kwarg (see
            # https://github.com/pytorch/pytorch/issues/132020).
            data = data.contiguous(memory_format=memory_format)
        return Float8Tensor.make_like(
            tensor,
            data=data,
            fp8_attrs=tensor._fp8_attrs,
            dtype=dtype,
        )

    # Convert standard PyTorch tensor
    tensor = tensor.to(device=device, dtype=dtype)
    if memory_format != torch.preserve_format and not tensor.is_contiguous(
        memory_format=memory_format
    ):
        # Note: torch.Tensor.to ignores memory_format kwarg (see
        # https://github.com/pytorch/pytorch/issues/132020).
        tensor = tensor.contiguous(memory_format=memory_format)
    return tensor


def reshape(
    tensor: torch.Tensor | Float8Tensor,
    shape: Iterable[int],
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor | Float8Tensor:
    """Reshape tensor, keeping same data if possible

    If the input is a Float8Tensor, this function attempts to preserve
    the cached transpose if available and valid. If a cached transpose
    is present, it is interpreted as the transpose of a 2D matrix
    where the width matches the innermost tensor dimension.

    """

    # Make sure tensor is in expected format
    tensor = convert_tensor(
        tensor,
        device=device,
        dtype=dtype,
        memory_format=torch.contiguous_format,
    )

    # Return immediately if tensor already has desired shape
    shape = list(shape)
    if len(shape) == tensor.dim():
        if sum(1 for d in shape if d == -1) > 1:
            raise ValueError(
                "Attempted to reshape tensor with "
                f"shape={tuple(tensor.size())} into shape={tuple(shape)}"
            )
        if all(d1 == d2 for d1, d2 in zip(shape, tensor.size()) if d1 != -1):
            return tensor

    # Reshape FP8 tensor
    # Note: Preserve cached transpose if possible
    if is_float8_tensor(tensor):
        out = Float8Tensor.make_like(
            tensor,
            data=tensor._data.view(shape),
            fp8_attrs=tensor._fp8_attrs,
        )
        return out

    # Reshape standard PyTorch tensor
    return tensor.view(shape)


def maybe_autocast_dtype(
    *,
    device_type: str = "cuda",
    default_dtype: Optional[torch.dtype] = None,
) -> torch.dtype:
    """Get autocast dtype if enabled"""
    if torch.is_autocast_enabled(device_type):
        return torch.get_autocast_dtype(device_type)
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
