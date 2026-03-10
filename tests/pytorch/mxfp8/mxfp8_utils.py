# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import torch
import math


# Calculate the shape of the scaling tensor for MXFP8 1D blockwise quantization without padding
def get_mxfp8_scale_shape_no_padding(shape, columnwise):
    M, K = 1, 1
    M = math.prod(shape[:-1])
    K = shape[-1]

    if columnwise:
        outer = M // 32
        inner = K
        return (outer, inner)
    # rowwise
    outer = M
    inner = K // 32
    return (outer, inner)


def _rowwise_swizzle_mxfp8_scale(input_M, input_N, scale: torch.Tensor) -> torch.Tensor:
    assert scale.dim() == 2
    assert input_M == scale.shape[0]
    assert input_N // 32 == scale.shape[1]

    x = scale.view(input_M // 128, 4, 32, input_N // 128, 4)
    x = x.permute(0, 3, 2, 1, 4)
    x = x.contiguous()
    # View back as original 2D shape
    x = x.view(input_M, input_N // 32)
    return x


def _columnwise_swizzle_mxfp8_scale(input_M, input_N, scale: torch.Tensor) -> torch.Tensor:
    assert scale.dim() == 2
    assert input_M // 32 == scale.shape[0]
    assert input_N == scale.shape[1]

    x = scale.view(input_M // 128, 4, input_N // 128, 4, 32)
    x = x.permute(2, 0, 4, 3, 1)
    x = x.contiguous()

    # alternative way: transpose the scale and do rowwise swizzle with M, N swapped
    x1 = _rowwise_swizzle_mxfp8_scale(input_N, input_M, scale.transpose(0, 1).contiguous())
    torch.testing.assert_close(
        x.view(-1), x1.view(-1), atol=0.0, rtol=0.0, msg="columnwise swizzle sanity check failed"
    )

    # View back as original 2D shape
    x = x.view(input_M // 32, input_N)
    return x


def swizzle_mxfp8_scale(input_M, input_N, scale: torch.Tensor, columnwise: bool) -> torch.Tensor:
    if not columnwise:
        return _rowwise_swizzle_mxfp8_scale(input_M, input_N, scale)
    else:
        return _columnwise_swizzle_mxfp8_scale(input_M, input_N, scale)
