# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import pytest
import torch
from transformer_engine.pytorch.fp8 import fp8_autocast, FP8GlobalStateManager, fp8_model_init
from transformer_engine.pytorch.utils import is_bf16_compatible
from transformer_engine.pytorch import Linear


# Only run FP8 tests on H100.
fp8_available, reason_for_no_fp8 = FP8GlobalStateManager.is_fp8_available()

class ModelConfig:
    def __init__(self, hidden_size, eps, num_attention_heads, embed, num_layers, seq_len):
        self.hidden_size = hidden_size
        self.eps = eps
        self.num_attention_heads = num_attention_heads
        self.embed = embed
        self.num_layers = num_layers
        self.seq_len = seq_len


model_configs = {
    "126m": ModelConfig(768, 1e-5, 12, 64, 12, 2048),
}

param_types = [torch.float32, torch.float16]
if is_bf16_compatible():  # bf16 requires sm_80 or higher
    param_types.append(torch.bfloat16)

batch_sizes_with_zero = [0, 1, 2]
all_boolean = [True, False]

@pytest.mark.parametrize("dtype", param_types)
@pytest.mark.parametrize("bs", batch_sizes_with_zero)
@pytest.mark.parametrize("model", model_configs.keys())
@pytest.mark.parametrize("fp8", all_boolean)
@pytest.mark.parametrize("fp8_model_params", all_boolean)
@pytest.mark.parametrize("use_bias", all_boolean)
def test_linear_layer(dtype, bs, model, fp8, fp8_model_params, use_bias):
    config = model_configs[model]
    with fp8_model_init(enabled=fp8 and fp8_model_params):
        te_linear = (
            Linear(
                config.hidden_size,
                4 * config.hidden_size,
                bias=use_bias,
                params_dtype=dtype
            )
            .cuda()
        )

    inp_hidden_states = torch.randn(
        bs*config.seq_len, config.hidden_size, dtype=dtype, requires_grad=True
    ).cuda()
    with fp8_autocast(enabled=fp8):
        out = te_linear(inp_hidden_states) 
    loss = out.sum()
    loss.backward()
