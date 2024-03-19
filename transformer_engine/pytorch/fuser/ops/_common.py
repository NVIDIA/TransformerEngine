# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

from __future__ import annotations

import torch

from transformer_engine.pytorch.cpp_extensions import (
    fp8_cast_transpose_fused,
)
from transformer_engine.pytorch.float8_tensor import Float8Tensor
from transformer_engine.pytorch.fp8 import FP8GlobalStateManager

def canonicalize_device(device: Optional[torch.device | str]) -> torch.device:
    """Canonicalize PyTorch device

    If `None`, then returns the default CUDA device.

    """
    if device is None:
        # Use default CUDA device
        device = torch.get_default_device()
        if device.type != "cuda":
            device = torch.device("cuda")
    elif not isinstance(device, torch.device):
        device = torch.device(device)
    return device

def canonicalize_dtype(dtype: Optional[torch.dtype]) -> torch.dtype:
    """Canonicalize PyTorch datatype

    If `None`, then returns the default PyTorch datatype.

    """
    if dtype is None:
        # Use default dtype
        dtype = torch.get_default_dtype()
    return dtype

def is_float8_tensor(tensor: Any) -> bool:
    """Check if object is a `Float8Tensor`"""
    return isinstance(tensor, Float8Tensor)

@torch.no_grad()
def convert_tensor(
    tensor: torch.Tensor | Float8Tensor,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    memory_format: torch.memory_format = torch.preserve_format,
) -> torch.Tensor | Float8Tensor:
    """Convert tensor attributes, keeping same data if possible

    Supports Float8Tensor.

    """

    # Default kwargs
    if device is None:
        device = tensor.device
    device = canonicalize_device(device)
    if dtype is None:
        dtype = tensor.dtype
    dtype = canonicalize_dtype(dtype)

    # Return immediately if tensor already has desired attributes
    if device == tensor.device and dtype == tensor.dtype:
        if (
            memory_format == torch.preserve_format
            or tensor.is_contiguous(memory_format=memory_format)
        ):
            return tensor

    # Convert FP8 tensor
    if is_float8_tensor(tensor):
        data = tensor._data.to(device=device, memory_format=memory_format)
        return Float8Tensor.make_like(tensor, data=data, dtype=dtype)

    # Convert standard PyTorch tensor
    return tensor.to(device=device, dtype=dtype, memory_format=memory_format)

@torch.no_grad()
def fp8_cast_transpose(
    tensor: torch.Tensor,
    fp8_meta: Dict[str, Any],
    fp8_meta_forward: bool,
    fp8_meta_index: int,
    fp8_dtype: tex.DType,
) -> Float8Tensor:
    """Fused FP8 cast and transpose"""

    # Check tensor
    device = tensor.device
    if device.type != "cuda":
        device = canonicalize_device(None)
    dtype = tensor.dtype
    if dtype not in (torch.float32, torch.float16, torch.bfloat16):
        dtype = torch.float32
    tensor = tensor.to(
        device=device,
        dtype=dtype,
        memory_format=torch.contiguous_format,
    )

    # Compute FP8 cast and FP8 transpose
    fp8_meta_key = FP8GlobalStateManager.get_meta_tensor_key(
        forward=fp8_meta_forward,
    )
    data, data_t = fp8_cast_transpose_fused(
        tensor,
        fp8_meta[fp8_meta_key],
        fp8_meta_index,
        fp8_dtype,
    )

    # Construct FP8 tensor and populate transpose cache
    out = Float8Tensor(
        data=data,
        fp8_meta=fp8_meta,
        fp8_meta_forward=fp8_meta_forward,
        fp8_meta_index=fp8_meta_index,
        fp8_dtype=fp8_dtype,
    )
    out._transpose = Float8Tensor.make_like(out, data=data_t)
    return out
