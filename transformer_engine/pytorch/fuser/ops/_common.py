from __future__ import annotations

import torch

def canonicalize_device(device: Optional[torch.device | str]) -> torch.device:
    """Canonicalize PyTorch device"""
    if device is None:
        # Use default CUDA device
        device = torch.get_default_device()
        if device.type != "cuda":
            device = torch.device("cuda")
    elif not isinstance(device, torch.device):
        device = torch.device(device)
    return device

def canonicalize_dtype(dtype: Optional[torch.dtype]) -> torch.dtype:
    """Canonicalize PyTorch dtype"""
    if dtype is None:
        # Use default dtype
        dtype = torch.get_default_dtype()
    return dtype
