# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Triton kernel turning a dGLU epilogue amax into NVFP4 quantization amaxes."""

import triton
import triton.language as tl


@triton.jit
def _nvfp4_amax_from_dglu_amax_kernel(
    dglu_amax_ptr,
    rowwise_amax_ptr,
    columnwise_amax_ptr,
    n,
    columnwise_factor,
    BLOCK: tl.constexpr,
):
    """Per-group rowwise and columnwise amax from the per-GLU-half dGLU amax.

    ``dglu_amax`` holds two amaxes per group (one per GLU half), ``-inf`` for empty groups.
    The rowwise amax is their maximum, clamped at zero and rounded to BF16 so that it equals
    the amax of the BF16 dGLU output exactly (round to nearest is monotone). The columnwise
    amax is the rowwise amax times ``columnwise_factor``.
    """
    idx = tl.arange(0, BLOCK)
    mask = idx < n
    a0 = tl.load(dglu_amax_ptr + 2 * idx, mask=mask, other=0.0)
    a1 = tl.load(dglu_amax_ptr + 2 * idx + 1, mask=mask, other=0.0)
    amax = tl.maximum(tl.maximum(a0, a1), 0.0)
    amax = amax.to(tl.bfloat16).to(tl.float32)
    tl.store(rowwise_amax_ptr + idx, amax, mask=mask)
    tl.store(columnwise_amax_ptr + idx, amax * columnwise_factor, mask=mask)
