# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.


import transformer_engine.pytorch as te
from transformer_engine.pytorch import MXFP8Quantizer
from transformer_engine.pytorch.tensor.storage.mxfp8_tensor_storage import MXFP8TensorStorage

import pytest
import torch
import random
import math

from typing import Tuple

from mxfp8_utils import swizzle_mxfp8_scale, get_mxfp8_scale_shape_no_padding

recipe_available, reason_for_no_recipe = te.is_mxfp8_available(return_reason=True)


def unpack_quantized_tensor(
    quantized_tensor: MXFP8TensorStorage,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    qx, sx, qx_t, sx_t = None, None, None, None
    if quantized_tensor._rowwise_data is not None:
        qx = quantized_tensor._rowwise_data.view(dtype=torch.uint8)
    if quantized_tensor._rowwise_scale_inv is not None:
        sx = quantized_tensor._rowwise_scale_inv
    if quantized_tensor._columnwise_data is not None:
        qx_t = quantized_tensor._columnwise_data.view(dtype=torch.uint8)
    if quantized_tensor._columnwise_scale_inv is not None:
        sx_t = quantized_tensor._columnwise_scale_inv
    return qx, sx, qx_t, sx_t


def check_mxfp8_quantize_swizzle_fusion(
    x_dtype: torch.dtype,
    M: int,
    N: int,
    return_rowwise: bool,
    return_transpose: bool,
) -> None:

    te_dtype = te.DType.kFloat8E4M3

    # Setup device and random seed
    device = "cuda"
    seed = 0
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # Input
    x = torch.randn((M, N), dtype=x_dtype, device=device)

    # Quantize
    quantizer = MXFP8Quantizer(
        fp8_dtype=te_dtype,
        rowwise=return_rowwise,
        columnwise=return_transpose,
    )

    quantizer_swizzle_fusion = quantizer.copy()
    quantizer_swizzle_fusion.optimize_for_gemm = True

    x_qx_swf, x_sx_swf, x_qx_t_swf, x_sx_t_swf = unpack_quantized_tensor(
        quantizer_swizzle_fusion(x)
    )
    x_qx_ref, x_sx_ref, x_qx_t_ref, x_sx_t_ref = unpack_quantized_tensor(quantizer(x))

    if return_rowwise:
        torch.testing.assert_close(x_qx_swf, x_qx_ref, atol=0.0, rtol=0.0)
        valid_scale_shape = get_mxfp8_scale_shape_no_padding(x.shape, False)
        assert valid_scale_shape == x_sx_swf.shape, (
            "The scale shape is not correctly aligned, this test assumes no padding is needed for"
            " scaling factors"
        )
        x_sx_ref_swizzled = swizzle_mxfp8_scale(M, N, x_sx_ref, columnwise=False)
        torch.testing.assert_close(x_sx_swf, x_sx_ref_swizzled, atol=0.0, rtol=0.0)

    if return_transpose:
        torch.testing.assert_close(x_qx_t_swf, x_qx_t_ref, atol=0.0, rtol=0.0)
        valid_scale_shape = get_mxfp8_scale_shape_no_padding(x.shape, True)
        assert valid_scale_shape == x_sx_t_swf.shape, (
            "The scale shape is not correctly aligned, this test assumes no padding is needed for"
            " scaling factors"
        )
        x_sx_t_ref_swizzled = swizzle_mxfp8_scale(M, N, x_sx_t_ref, columnwise=True)
        torch.testing.assert_close(x_sx_t_swf, x_sx_t_ref_swizzled, atol=0.0, rtol=0.0)


@pytest.mark.skipif(not recipe_available, reason=reason_for_no_recipe)
@pytest.mark.parametrize(
    "M, N",
    [
        # full tile cases
        (1024, 256),
        # larger sizes
        (8192, 1024),
        (16384, 8192),
        (16384, 16384),
    ],
)
@pytest.mark.parametrize("x_dtype", [torch.bfloat16], ids=str)
@pytest.mark.parametrize("quantize_mode", ["rowwise_only", "both_directions", "columnwise_only"])
def test_mxfp8_quantize_swizzle_fusion(
    x_dtype: torch.dtype,
    M: int,
    N: int,
    quantize_mode: str,
) -> None:

    if quantize_mode == "rowwise_only":
        return_rowwise = True
        return_transpose = False
    elif quantize_mode == "both_directions":
        return_rowwise = True
        return_transpose = True
    elif quantize_mode == "columnwise_only":
        return_rowwise = False
        return_transpose = True
    else:
        raise ValueError(f"Invalid quantize mode: {quantize_mode}")

    check_mxfp8_quantize_swizzle_fusion(
        x_dtype=x_dtype,
        M=M,
        N=N,
        return_rowwise=return_rowwise,
        return_transpose=return_transpose,
    )


@pytest.mark.skipif(not recipe_available, reason=reason_for_no_recipe)
@pytest.mark.parametrize("M, N", [(96, 160), (4096, 576), (4096, 2112)])
def test_mxfp8_bidirectional_swizzled_row_scale_padding(M: int, N: int) -> None:
    """The specialized bidirectional kernel must not overwrite padded row scales."""
    x = torch.randn((M, N), dtype=torch.bfloat16, device="cuda")
    quantizer = MXFP8Quantizer(
        fp8_dtype=te.DType.kFloat8E4M3,
        rowwise=True,
        columnwise=True,
    )
    quantizer.optimize_for_gemm = True
    scale = quantizer(x)._rowwise_scale_inv.view(torch.uint8)

    scale_rows = torch.arange(M, device=scale.device, dtype=torch.int64).view(-1, 1)
    scale_cols = torch.arange(N // 32, device=scale.device, dtype=torch.int64).view(1, -1)
    num_tiles_x = math.ceil(N / 128)
    scale_indices = (
        ((scale_rows // 128) * num_tiles_x + scale_cols // 4) * (128 * 4)
        + (scale_rows % 32) * 16
        + ((scale_rows % 128) // 32) * 4
        + scale_cols % 4
    )
    valid_mask = torch.zeros(scale.numel(), dtype=torch.bool, device=scale.device)
    valid_mask[scale_indices.view(-1)] = True

    torch.testing.assert_close(
        scale.view(-1)[~valid_mask],
        torch.zeros_like(scale.view(-1)[~valid_mask]),
        atol=0,
        rtol=0,
    )
