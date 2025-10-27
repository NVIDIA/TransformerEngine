# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import pytest
import torch

from transformer_engine.pytorch.tensor.float8_blockwise_tensor import Float8BlockQuantizer
from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer
from transformer_engine.pytorch.fp8 import (
    FP8GlobalStateManager,
    fp8_model_init,
)
from transformer_engine.pytorch import SelectiveLayerNormMLP
from transformer_engine.pytorch.distributed import fp8_autocast
from transformer_engine.common.recipe import DelayedScaling

# Check if FP8 is supported
fp8_available, reason_for_no_fp8 = FP8GlobalStateManager.is_fp8_available()
mxfp8_available, reason_for_no_mxfp8 = FP8GlobalStateManager.is_mxfp8_available()
fp8_block_scaling_available, reason_for_no_fp8_block_scaling = (
    FP8GlobalStateManager.is_fp8_block_scaling_available()
)


# FP8 per tensor delayed scaling
@pytest.mark.skipif(not fp8_available, reason=reason_for_no_fp8)
class TestFP8Recipe:

    @staticmethod
    def setup_class(cls) -> None:
        # Configure RNG
        seed = 1234
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

    @pytest.mark.parametrize("module_class", [SelectiveLayerNormMLP])
    def test_quantizer_update(self, module_class):
        in_features = 32
        out_features = 32
        batch_size = 32

        recipe = DelayedScaling(amax_history_len=1024)
        with fp8_model_init(recipe=recipe):
            module = module_class(in_features, out_features).cuda()

        x = torch.randn(batch_size, in_features, device="cuda")
        recipe = DelayedScaling(amax_history_len=1)
        with fp8_autocast(enabled=True, fp8_recipe=recipe):
            warn_msg = "Quantizer is being updated, this may affect model behavior"
            with pytest.warns(UserWarning, match=warn_msg):
                y = module(x)
