# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""PyTorch wrapper for the round-through-dtype Triton kernel."""

import torch
import triton

from transformer_engine.common.triton.round_through_dtype import _round_through_dtype_kernel

_BLOCK = 1024


def round_through_dtype(x: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Round an FP32 tensor to ``dtype`` (BF16 or FP16) and return it in FP32.

    Bitwise equal to ``x.to(dtype).to(torch.float32)`` for every non-NaN input
    (NaN stays NaN), but runs as a single kernel and never materializes the
    16-bit intermediate. Returns a new contiguous tensor with the shape of ``x``.
    """
    if x.dtype != torch.float32:
        raise ValueError(f"round_through_dtype expects an FP32 tensor, got {x.dtype}")
    if dtype not in (torch.bfloat16, torch.float16):
        raise ValueError(f"round_through_dtype supports BF16 and FP16, got {dtype}")
    if not x.is_cuda:
        return x.to(dtype).to(torch.float32)
    x = x.contiguous()
    y = torch.empty_like(x)
    n = x.numel()
    if n > 0:
        _round_through_dtype_kernel[(triton.cdiv(n, _BLOCK),)](
            x,
            y,
            n,
            ROUND_TO_BF16=dtype == torch.bfloat16,
            BLOCK=_BLOCK,
        )
    return y
