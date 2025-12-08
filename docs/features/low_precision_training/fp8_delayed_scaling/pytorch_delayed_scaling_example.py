# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import torch

# Requires Ada (SM89) or newer for FP8 support
assert torch.cuda.get_device_capability()[0] >= 9 or (
    torch.cuda.get_device_capability()[0] == 8 and torch.cuda.get_device_capability()[1] >= 9
), "This example requires SM89 (Ada) or newer"

# START_DELAYED_SCALING_EXAMPLE

import torch
import transformer_engine.pytorch as te
from transformer_engine.common.recipe import DelayedScaling

# Create FP8 Delayed Scaling recipe
recipe = DelayedScaling(
    margin=0,  # Margin for scaling factor computation (default: 0)
    amax_history_len=1024,  # Length of amax history window (default: 1024)
    amax_compute_algo="max",  # How to compute amax from history (default: "max")
)

# Create a linear layer
layer = te.Linear(1024, 1024)
optimizer = torch.optim.AdamW(layer.parameters(), lr=1e-4)

# Training with FP8 Delayed Scaling
inp = torch.randn(32, 128, 1024, dtype=torch.bfloat16, device="cuda")

with te.autocast(enabled=True, recipe=recipe):
    output = layer(inp)
    loss = output.sum()

loss.backward()
optimizer.step()

# END_DELAYED_SCALING_EXAMPLE
