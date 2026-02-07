# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import torch

# Requires Ada (SM89) or Hopper (SM90), different results on Blackwell+
cc = torch.cuda.get_device_capability()
assert cc[0] == 8 and cc[1] >= 9 or cc[0] == 9, "This example requires SM89 (Ada) or SM90 (Hopper)"

# START_FUSED_LAYERS

import torch
import transformer_engine.pytorch as te
from transformer_engine.common.recipe import DelayedScaling

# Example 1: Separate LayerNorm and Linear layers
layer_norm = te.LayerNorm(1024)
linear = te.Linear(1024, 1024)

inp = torch.randn(32, 128, 1024, dtype=torch.bfloat16, device="cuda")

# Two separate operations: LayerNorm produces FP32, then Linear quantizes it
normalized = layer_norm(inp)
output_separate = linear(normalized)

# Example 2: Fused LayerNormLinear layer
fused_layer = te.LayerNormLinear(1024, 1024, params_dtype=torch.bfloat16)

# Single operation: LayerNorm output is directly quantized
recipe = DelayedScaling()
with te.autocast(enabled=True, recipe=recipe):
    output_fused = fused_layer(inp)

# The fused layer is more efficient as it avoids redundant quantization

# END_FUSED_LAYERS
