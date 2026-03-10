# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.


import transformer_engine.pytorch as te
import transformer_engine_torch as tex
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
    return_identity: bool,
    return_transpose: bool,
) -> None:

    te_dtype = tex.DType.kFloat8E4M3

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
        rowwise=return_identity,
        columnwise=return_transpose,
    )

    quantizer_swizzle_fusion = quantizer.copy()
    quantizer_swizzle_fusion.optimize_for_gemm = True

    x_qx_swf, x_sx_swf, x_qx_t_swf, x_sx_t_swf = unpack_quantized_tensor(
        quantizer_swizzle_fusion(x)
    )
    x_qx_ref, x_sx_ref, x_qx_t_ref, x_sx_t_ref = unpack_quantized_tensor(quantizer(x))

    if return_identity:
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
@pytest.mark.parametrize(
    "quantize_mode", ["quantize", "quantize_transpose", "quantize_colwise_only"]
)
def test_mxfp8_quantize_swizzle_fusion(
    x_dtype: torch.dtype,
    M: int,
    N: int,
    quantize_mode: str,
) -> None:

    if quantize_mode == "quantize":
        return_identity = True
        return_transpose = False
    elif quantize_mode == "quantize_transpose":
        return_identity = True
        return_transpose = True
    elif quantize_mode == "quantize_colwise_only":
        return_identity = False
        return_transpose = True
    else:
        raise ValueError(f"Invalid quantize mode: {quantize_mode}")

    check_mxfp8_quantize_swizzle_fusion(
        x_dtype=x_dtype,
        M=M,
        N=N,
        return_identity=return_identity,
        return_transpose=return_transpose,
    )
