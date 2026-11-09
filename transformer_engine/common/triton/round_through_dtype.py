# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Triton kernel that rounds an FP32 tensor to a 16-bit float format, keeping FP32 storage."""

import triton
import triton.language as tl


@triton.jit
def _round_through_dtype_kernel(
    x_ptr,
    y_ptr,
    n,
    ROUND_TO_BF16: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """y = float32(round_to_nearest_even(x, BF16 or FP16)) in one pass.

    Equivalent to ``x.to(torch.bfloat16).to(torch.float32)`` (or FP16), without
    materializing the 16-bit intermediate.
    """
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    if ROUND_TO_BF16:
        y = x.to(tl.bfloat16).to(tl.float32)
    else:
        y = x.to(tl.float16).to(tl.float32)
    tl.store(y_ptr + offs, y, mask=mask)
