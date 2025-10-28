# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

from typing import Optional

import pytest
import torch
import warnings

import transformer_engine.common.recipe
import transformer_engine.pytorch as te
from transformer_engine.pytorch import (
    Float8BlockQuantizer,
    MXFP8Quantizer,
    Float8Quantizer,
    NVFP4Quantizer,
    quantized_model_init,
    SelectiveLayerNormMLP,
)

import transformer_engine_torch as tex
from transformer_engine.pytorch.quantization import (
    FP8GlobalStateManager,
    _amax_and_scale_update,
)
import transformer_engine.pytorch.ops as te_ops
from transformer_engine.common.recipe import DelayedScaling, Float8BlockScaling, MXFP8BlockScaling

# Check if FP8 is supported
fp8_available, reason_for_no_fp8 = te.is_fp8_available(return_reason=True)
mxfp8_available, reason_for_no_mxfp8 = te.is_mxfp8_available(return_reason=True)
fp8_block_scaling_available, reason_for_no_fp8_block_scaling = te.is_fp8_block_scaling_available(
    return_reason=True
)
fp4_available, reason_for_no_fp4 = te.is_nvfp4_available(return_reason=True)


# FP8 per tensor delayed scaling
@pytest.mark.skipif(not fp8_available, reason=reason_for_no_fp8)
class TestFP8Recipe:

    @staticmethod
    def setup_class(cls) -> None:
        # Configure RNG
        seed = 1234
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

    @pytest.mark.parametrize(
        "module_class",
        [SelectiveLayerNormMLP],
    )
    def test_quantizer_update(self, module_class):
        in_features = 32
        out_features = 32
        batch_size = 32

        recipe = DelayedScaling(amax_history_len=1024)
        with quantized_model_init(recipe=recipe):
            module = module_class(in_features, out_features).cuda()

        x = torch.randn(batch_size, in_features, device="cuda")
        recipe = DelayedScaling(amax_history_len=1)
        with te.autocast(enabled=True, recipe=recipe):
            warn_msg = "Quantizer is being updated, this may affect model behavior"
            with pytest.warns(UserWarning, match=warn_msg):
                y = module(x)
