# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Helper functions used in fusible operations."""

from __future__ import annotations
from typing import Any, Iterable, Optional

import torch

from transformer_engine.pytorch.float8_tensor import Float8Tensor


def canonicalize_device(device: Optional[torch.device | str]) -> torch.device:
    """Canonicalize PyTorch device

    If `None`, then returns the default CUDA device.

    """
    if device is None:
        # Use default CUDA device
        device = torch.get_default_device()
        if device.type != "cuda":
            device = torch.device("cuda", torch.cuda.current_device())
    elif not isinstance(device, torch.device):
        device = torch.device(device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    return device


def canonicalize_dtype(dtype: Optional[torch.dtype]) -> torch.dtype:
    """Canonicalize PyTorch datatype

    If `None`, then returns the default PyTorch datatype.

    """
    if dtype is None:
        # Use default dtype
        dtype = torch.get_default_dtype()
    return dtype


def devices_match(device1: torch.device, device2: torch.device) -> bool:
    """Whether two devices are the same"""
    device1 = torch.device(device1)
    device2 = torch.device(device2)
    if device1.type != device2.type:
        return False
    if device1.type == "cuda":
        index1 = device1.index
        index2 = device2.index
        if index1 is None:
            index1 = torch.cuda.current_device()
        if index2 is None:
            index2 = torch.cuda.current_device()
        return index1 == index2
    return device1 == device2


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
        data = tensor._data.to(device=device, memory_format=memory_format)
        return Float8Tensor.make_like(
            tensor,
            data=data,
            fp8_attrs=tensor._fp8_attrs,
            dtype=dtype,
        )

    # Convert standard PyTorch tensor
    return tensor.to(device=device, dtype=dtype, memory_format=memory_format)


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
