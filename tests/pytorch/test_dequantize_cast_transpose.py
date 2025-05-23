# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import pytest
import torch
import transformer_engine as te
import transformer_engine_torch as tex

from transformer_engine.pytorch.constants import TE_DType
from transformer_engine.pytorch.fp8 import FP8GlobalStateManager
from transformer_engine.pytorch.tensor.float8_blockwise_tensor import (
    Float8BlockQuantizer,
    Float8BlockwiseQTensor,
)

recipe_available, reason_for_no_recipe = FP8GlobalStateManager.is_fp8_block_scaling_available()


@pytest.mark.skipif(not recipe_available, reason=reason_for_no_recipe)
@pytest.mark.parametrize("x_dtype", [torch.bfloat16], ids=str)
@pytest.mark.parametrize(
    "M, N",
    [
        (128, 128),
        (144, 144),
        (148, 148),
    ],
)
@pytest.mark.parametrize("quant_dtype", [torch.float8_e4m3fn], ids=str)
@pytest.mark.parametrize("eps", [1e-6], ids=["eps_1e-6"])
def test_dequantize_cast_transpose(
    x_dtype: torch.dtype,
    M: int,
    N: int,
    quant_dtype: torch.dtype,
    eps: float,
):
    quantizer = Float8BlockQuantizer(
        fp8_dtype=TE_DType[quant_dtype],
        rowwise=True,
        columnwise=True,
        amax_epsilon=eps,
        force_pow_2_scales=True,
        block_scaling_dim=1,
    )
    # Setup device and random seed
    device = "cuda"
    seed = 0
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # Input
    x_base = torch.randn((M, N), dtype=x_dtype, device=device)
    x_ref = x_base.clone()
    x_fp8_base = quantizer.make_empty(x_base.shape, dtype=x_base.dtype, device=x_base.device)
    x_fp8_ref = quantizer.make_empty(x_ref.shape, dtype=x_ref.dtype, device=x_ref.device)

    # Quantize to fp8
    quantizer.update_quantized(x_base, x_fp8_base)
    quantizer.update_quantized(x_ref, x_fp8_ref)
    torch.testing.assert_close(
        x_fp8_base._rowwise_data, x_fp8_ref._rowwise_data, atol=0.0, rtol=0.0
    )
    torch.testing.assert_close(
        x_fp8_base._rowwise_scale_inv, x_fp8_ref._rowwise_scale_inv, atol=0.0, rtol=0.0
    )
    torch.testing.assert_close(
        x_fp8_base._columnwise_data, x_fp8_ref._columnwise_data, atol=0.0, rtol=0.0
    )
    torch.testing.assert_close(
        x_fp8_base._columnwise_scale_inv, x_fp8_ref._columnwise_scale_inv, atol=0.0, rtol=0.0
    )

    # Naive implementation
    def dequantize_cast_transpose_naive(x, quantizer):
        x_dequantized = x.dequantize()
        x_transposed = x_dequantized.transpose(0, 1).contiguous()
        x_casted = quantizer.make_empty(
            x_transposed.shape, dtype=x_transposed.dtype, device=x_transposed.device
        )
        quantizer.update_quantized(x_transposed, x_casted)
        return x_casted._rowwise_data, x_casted._rowwise_scale_inv

    res_base_data, res_base_scale_inv = dequantize_cast_transpose_naive(x_fp8_base, quantizer)
    # Fused implementation
    res_ref = tex.fp8_blockwise_transpose(x_fp8_ref, quantizer)
    res_ref_data, res_ref_scale_inv = res_ref._columnwise_data, res_ref._columnwise_scale_inv

    # Check results
    torch.testing.assert_close(res_base_scale_inv, res_ref_scale_inv, atol=0.0, rtol=0.0)
    torch.testing.assert_close(res_base_data, res_ref_data, atol=0.0, rtol=0.0)


# from typing import Dict
# # Numerical tolerances with FP8 types
# _tols: Dict[tex.DType, Dict[str, float]] = {
#     tex.DType.kFloat8E4M3: dict(rtol=0.125, atol=0.08),
#     tex.DType.kFloat8E5M2: dict(rtol=0.25, atol=0.125),
# }

# @pytest.mark.skipif(not recipe_available, reason=reason_for_no_recipe)
# @pytest.mark.parametrize("x_dtype", [torch.bfloat16], ids=str)
# @pytest.mark.parametrize(
#     "M, N",
#     [
#         (128, 128),
#         (144, 144),
#         (148, 148),
#     ],
# )
# @pytest.mark.parametrize("quant_dtype", [torch.float8_e4m3fn], ids=str)
# @pytest.mark.parametrize("eps", [1e-6], ids=["eps_1e-6"])
# def test_e2e_dequantize_cast_transpose(
#     x_dtype: torch.dtype,
#     M: int,
#     N: int,
#     quant_dtype: torch.dtype,
#     eps: float,
# ):
#     quantizer = Float8BlockQuantizer(
#         fp8_dtype=TE_DType[quant_dtype],
#         rowwise=True,
#         columnwise=True,
#         amax_epsilon=eps,
#         force_pow_2_scales=True,
#         block_scaling_dim=1,
#     )
#     # Setup device and random seed
#     device = "cuda"
#     seed = 0
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     # Input
#     x_base = torch.randn((M, N), dtype=x_dtype, device=device)
#     x_ref = x_base.clone()
#     x_fp8_base = quantizer.make_empty(x_base.shape, dtype=x_base.dtype, device=x_base.device)
#     x_fp8_ref = quantizer.make_empty(x_ref.shape, dtype=x_ref.dtype, device=x_ref.device)

#     # Naive implementation
#     quantizer.update_quantized(x_base, x_fp8_base)
#     x_fp8_base.update_usage(rowwise_usage=True, columnwise_usage=True)
#     res_base_data, res_base_scale_inv = x_fp8_base._columnwise_data, x_fp8_base._columnwise_scale_inv

#     # Fused implementation
#     quantizer.update_quantized(x_ref, x_fp8_ref)
#     res_ref = tex.fp8_blockwise_transpose(x_fp8_ref, quantizer)
#     res_ref_data, res_ref_scale_inv = res_ref._columnwise_data, res_ref._columnwise_scale_inv

#     # Check results
#     torch.testing.assert_close(res_base_scale_inv, res_ref_scale_inv, **_tols[TE_DType[quant_dtype]])
#     torch.testing.assert_close(res_base_data, res_ref_data, **_tols[TE_DType[quant_dtype]])
