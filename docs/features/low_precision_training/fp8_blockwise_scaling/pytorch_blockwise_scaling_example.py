# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import torch

# Check for Hopper or newer GPU
major, minor = torch.cuda.get_device_capability()
assert major >= 9, f"FP8 Blockwise Scaling requires SM90 (Hopper) or later, got SM{major}{minor}"

# START_BLOCKWISE_SCALING_EXAMPLE

import torch
import transformer_engine.pytorch as te
from transformer_engine.common.recipe import Float8BlockScaling

# Create FP8 Blockwise Scaling recipe
recipe = Float8BlockScaling(
    fp8_format=te.common.recipe.Format.E4M3,  # E4M3 or HYBRID (default: E4M3)
    x_block_scaling_dim=1,  # 1D scaling for activations (default: 1)
    w_block_scaling_dim=2,  # 2D scaling for weights (default: 2)
    grad_block_scaling_dim=1,  # 1D scaling for gradients (default: 1)
)

# Create a linear layer with bfloat16 parameters
layer = te.Linear(1024, 1024, params_dtype=torch.bfloat16)

# Forward and backward pass
inp = torch.randn(32, 128, 1024, dtype=torch.bfloat16, device="cuda")

with te.autocast(enabled=True, recipe=recipe):
    output = layer(inp)
    loss = output.sum()

loss.backward()

# END_BLOCKWISE_SCALING_EXAMPLE
