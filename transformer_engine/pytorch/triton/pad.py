# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""PyTorch wrapper functions for padding Triton kernels."""

import torch
import triton

from transformer_engine.common.triton.pad import zero_pad_kernel


def pad_columnwise_scale_inv(inp: torch.Tensor) -> torch.Tensor:
    """Pads a tensor assuming it's a columnwise scaling inverse."""

    assert inp.ndim == 2
    dim0, dim1 = inp.shape

    pad_x = (128 - dim0 % 128) % 128
    pad_y = (4 - dim1 % 4) % 4
    out_x = dim0 + pad_x
    out_y = dim1 + pad_y
    out = torch.empty((out_x, out_y), device=inp.device, dtype=inp.dtype)

    in_s0, in_s1 = inp.stride()
    out_s0, out_s1 = out.stride()

    BLOCK_M, BLOCK_N = 128, 128
    grid = (triton.cdiv(out_x, BLOCK_M), triton.cdiv(out_y, BLOCK_N))

    zero_pad_kernel[grid](
        inp,
        out,
        dim0,
        dim1,
        out_x,
        out_y,
        in_s0,
        in_s1,
        out_s0,
        out_s1,
    )
    return out
