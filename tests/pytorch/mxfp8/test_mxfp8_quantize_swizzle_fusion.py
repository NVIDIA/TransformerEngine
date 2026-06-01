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


def poison_mxfp8_scale_padding_with_nan(
    scale: torch.Tensor,
    valid_shape: Tuple[int, int],
) -> torch.Tensor:
    """Fill only the padded part of a compact MXFP8 scale tensor with NaNs."""
    scale = scale.clone()
    scale_e8m0 = scale.view(dtype=torch.float8_e8m0fnu)
    scale_e8m0[valid_shape[0] :, :] = float("nan")
    scale_e8m0[:, valid_shape[1] :] = float("nan")
    return scale


def zero_mxfp8_scale_padding(
    scale: torch.Tensor,
    valid_shape: Tuple[int, int],
) -> torch.Tensor:
    """Zero only the padded part of a compact MXFP8 scale tensor."""
    scale = scale.clone()
    scale_uint8 = scale.view(dtype=torch.uint8)
    scale_uint8[valid_shape[0] :, :] = 0
    scale_uint8[:, valid_shape[1] :] = 0
    return scale


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
@pytest.mark.parametrize("columnwise", [False, True])
def test_mxfp8_pointer_swizzle_uses_unpadded_data_shape(columnwise: bool) -> None:
    # This shape has valid MXFP8 blocks, but its scale tensor needs padding:
    # rowwise scale is padded from (160, 3) to (256, 4), columnwise from
    # (5, 96) to (8, 128).
    M, N = 160, 96
    x = torch.randn((M, N), dtype=torch.bfloat16, device="cuda")
    quantizer = MXFP8Quantizer(
        fp8_dtype=tex.DType.kFloat8E4M3,
        rowwise=not columnwise,
        columnwise=columnwise,
    )
    quantized = quantizer(x)
    valid_scale_shape = get_mxfp8_scale_shape_no_padding(x.shape, columnwise)

    if columnwise:
        scale = quantized._columnwise_scale_inv
        transform_type = "uniform_mxfp8_columnwise_swizzle"
        inferred_padded_shape = (scale.shape[0] * 32, scale.shape[1])
    else:
        scale = quantized._rowwise_scale_inv
        transform_type = "uniform_mxfp8_rowwise_swizzle"
        inferred_padded_shape = (scale.shape[0], scale.shape[1] * 32)

    # Poison padded scale values with E8M0 NaNs. If swizzle reconstructs the
    # data shape from the padded scale shape, these values are considered real
    # and survive.
    scale = poison_mxfp8_scale_padding_with_nan(scale, valid_scale_shape)

    # With the actual data shape, swizzle should treat the poisoned bytes as
    # padding. Build the expected output by zeroing that padding before the
    # Python reference swizzle.
    expected_compact_scale = zero_mxfp8_scale_padding(scale, valid_scale_shape)
    padded_M, padded_N = inferred_padded_shape
    expected = swizzle_mxfp8_scale(
        padded_M,
        padded_N,
        expected_compact_scale,
        columnwise=columnwise,
    )

    _, swizzled_buffer = tex.transform_and_copy_data_ptrs_to_device(
        transform_type,
        [scale],
        x.device,
        x.shape,
    )
    torch.cuda.synchronize()
    actual = swizzled_buffer[: scale.numel()].view(dtype=expected.dtype).view_as(expected)

    torch.testing.assert_close(actual, expected, atol=0.0, rtol=0.0)
