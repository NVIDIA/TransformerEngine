from __future__ import annotations

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

def convert_tensor(
    tensor: torch.Tensor | Float8Tensor,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    memory_format: torch.memory_format = torch.preserve_format,
):
    """Convert tensor attributes, keeping same data if possible

    Supports Float8Tensor.

    """

    # Default kwargs
    if device is None:
        device = tensor.device
    if dtype is None:
        dtype = tensor.dtype

    # Convert FP8 tensor
    if is_float8_tensor(tensor):
        data = tensor._data.to(device=device, memory_format=memory_format)
        return Float8Tensor.make_like(tensor, data=data, dtype=dtype)

    # Convert standard PyTorch tensor
    return tensor.to(device=device, dtype=dtype, memory_format=memory_format)
