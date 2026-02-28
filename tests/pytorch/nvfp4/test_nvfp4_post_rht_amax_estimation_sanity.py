# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import pytest
import torch

import transformer_engine.pytorch as te
from transformer_engine.pytorch import NVFP4Quantizer


recipe_available, reason_for_no_recipe = te.is_nvfp4_available(return_reason=True)


@pytest.mark.skipif(not recipe_available, reason=reason_for_no_recipe)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for NVFP4 quantization")
def test_nvfp4_post_rht_amax_estimation_sanity() -> None:
    """Sanity: when using post-RHT amax estimation, columnwise amax is scaled pre-RHT amax."""

    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    # Shape must satisfy NVFP4 constraints and RHT kernel constraints.
    # rows % 64 == 0 and cols % 128 == 0 triggers the fast RHT-cast fusion path.
    M, N = 128, 128
    x = torch.randn((M, N), device="cuda", dtype=torch.bfloat16)

    scale = 2.0
    q = NVFP4Quantizer(
        rowwise=True,
        columnwise=True,
        with_rht=True,
        # Estimation path requires post-RHT amax kernel disabled.
        with_post_rht_amax=False,
        amax_estimation_scale=scale,
        stochastic_rounding=False,
    )

    y = q(x)
    assert y._amax_rowwise is not None
    assert y._amax_columnwise is not None

    amax_pre = torch.max(torch.abs(x)).to(torch.float32).view(1)
    torch.testing.assert_close(y._amax_rowwise, amax_pre, atol=0.0, rtol=0.0)
    torch.testing.assert_close(y._amax_columnwise, amax_pre * scale, atol=0.0, rtol=0.0)
