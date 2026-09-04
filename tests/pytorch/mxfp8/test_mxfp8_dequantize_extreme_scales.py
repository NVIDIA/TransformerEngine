# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import pytest
import torch

import transformer_engine.pytorch as te
from transformer_engine.pytorch import MXFP8Quantizer

recipe_available, reason_for_no_recipe = te.is_mxfp8_available(return_reason=True)


@pytest.mark.skipif(not recipe_available, reason=reason_for_no_recipe)
def test_dequantize_extreme_e8m0_scale_codes() -> None:
    """UE8M0 scale code 0 is 2^-127 and code 255 is NaN, not 0.0 and +Inf."""
    quantizer = MXFP8Quantizer(fp8_dtype=te.DType.kFloat8E4M3, columnwise=False)
    x = torch.randn(32, 64, dtype=torch.bfloat16, device="cuda")
    qx = quantizer(x)
    data = qx._rowwise_data.view(torch.uint8)
    scales = qx._rowwise_scale_inv.view(torch.uint8)
    data[0, :32] = 56
    scales[0, 0] = 0
    data[1, :32] = 56
    scales[1, 0] = 255
    y = qx.dequantize(dtype=torch.float32)
    assert (y[0, :32] == 2.0**-127).all()
    assert torch.isnan(y[1, :32]).all()
