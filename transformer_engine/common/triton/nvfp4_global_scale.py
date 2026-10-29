# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Triton kernel for the per-group NVFP4 global scales of a GEMM operand pair."""

import triton
import triton.language as tl

# Output modes of _nvfp4_global_scale_pair_kernel.
NVFP4_GLOBAL_SCALE_PAIR = 0  # out_a = amax_a / denom_a, out_b = amax_b / denom_b
NVFP4_GLOBAL_SCALE_PRODUCT = 1  # out_a = (amax_a / denom_a) * (amax_b / denom_b)
NVFP4_GLOBAL_SCALE_SQRT_PRODUCT = 2  # out_a = sqrt of the product above


@triton.jit
def _nvfp4_global_scale_pair_kernel(
    amax_a_ptr,
    amax_b_ptr,
    out_a_ptr,
    out_b_ptr,
    n,
    a_broadcast,
    b_broadcast,
    denom_a,
    denom_b,
    MODE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Per-group NVFP4 global scales of two operands, and optionally their product.

    One program handles all ``n`` groups. ``a_broadcast`` / ``b_broadcast`` mark an operand
    with a single amax shared by every group. The arithmetic matches the eager PyTorch ops
    bit for bit: dividing a CUDA tensor by a Python scalar multiplies by the FP32 reciprocal,
    and the product and square root are IEEE round-to-nearest.
    """
    idx = tl.arange(0, BLOCK)
    mask = idx < n
    amax_a = tl.load(amax_a_ptr + tl.where(a_broadcast, 0, idx), mask=mask, other=0.0)
    amax_b = tl.load(amax_b_ptr + tl.where(b_broadcast, 0, idx), mask=mask, other=0.0)
    one = tl.full([BLOCK], 1.0, tl.float32)
    rcp_a = tl.math.div_rn(one, tl.full([BLOCK], denom_a, tl.float32))
    rcp_b = tl.math.div_rn(one, tl.full([BLOCK], denom_b, tl.float32))
    scale_a = amax_a.to(tl.float32) * rcp_a
    scale_b = amax_b.to(tl.float32) * rcp_b
    if MODE == 0:
        tl.store(out_a_ptr + idx, scale_a, mask=mask)
        tl.store(out_b_ptr + idx, scale_b, mask=mask)
    else:
        product = scale_a * scale_b
        if MODE == 2:
            product = tl.math.sqrt_rn(product)
        tl.store(out_a_ptr + idx, product, mask=mask)
