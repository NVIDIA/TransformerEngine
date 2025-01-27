# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

from __future__ import annotations

import torch

import transformer_engine
import transformer_engine.pytorch as te
import transformer_engine_torch as tex


def str_to_dtype(dtype: str | torch.dtype) -> torch.dtype:
    """Convert type name to PyTorch dtype"""
    if isinstance(dtype, torch.dtype):
        return dtype
    name = str(dtype).strip().lower()
    if name.startswith("torch."):
        name = name.replace("torch.", "", 1)
    if name.startswith("fp"):
        name = name.replace("fp", "float", 1)
    dtype = dict(
        float32=torch.float32,
        float=torch.float32,
        float64=torch.float64,
        double=torch.float64,
        float16=torch.float16,
        half=torch.float16,
        bfloat16=torch.bfloat16,
        bf16=torch.bfloat16,
        float8_e4m3fn=torch.float8_e4m3fn,
        float8_e4m3=torch.float8_e4m3fn,
        float8e4m3=torch.float8_e4m3fn,
        float8=torch.float8_e4m3fn,
        float8_e5m2=torch.float8_e5m2,
        float8e5m2=torch.float8_e5m2,
        uint8=torch.uint8,
        byte=torch.uint8,
        int8=torch.int8,
        char=torch.int8,
        int16=torch.int16,
        short=torch.int16,
        int32=torch.int32,
        int=torch.int32,
        int64=torch.int64,
        long=torch.int64,
        bool=torch.bool,
    )[name]
    return dtype


def dtype_tols(dtype: torch.dtype | tex.DType) -> dict[str, float]:
    """Estimated numerical error for a datatype

    Based on tolerances for torch.testing.assert_close.

    """

    # Transformer Engine dtypes
    if isinstance(dtype, tex.DType):
        dtype = {
            tex.DType.kByte: torch.uint8,
            tex.DType.kInt32: torch.int32,
            tex.DType.kFloat32: torch.float32,
            tex.DType.kFloat16: torch.half,
            tex.DType.kBFloat16: torch.bfloat16,
            tex.DType.kFloat8E4M3: torch.float8_e4m3fn,
            tex.DType.kFloat8E5M2: torch.float8_e5m2,
        }[dtype]

    # PyTorch dtypes
    if dtype == torch.float16:
        return dict(rtol=1e-3, atol=1e-5)
    if dtype == torch.bfloat16:
        return dict(rtol=1.6e-2, atol=1e-5)
    if dtype == torch.float32:
        return dict(rtol=1.3e-6, atol=1e-5)
    if dtype == torch.float64:
        return dict(rtol=1e-7, atol=1e-7)
    if dtype == torch.float8_e4m3fn:
        return dict(rtol=0.125, atol=0.0675)  # epsilon = 0.0625
    if dtype == torch.float8_e5m2:
        return dict(rtol=0.25, atol=0.125)  # epsilon = 0.152
    raise ValueError(f"Unsupported dtype ({dtype})")
