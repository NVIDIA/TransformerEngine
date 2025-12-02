# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

# START_BLOCKWISE_SCALING_EXAMPLE

import torch
import transformer_engine.pytorch as te
from transformer_engine.common.recipe import Float8BlockScaling

# Create FP8 Blockwise Scaling recipe
recipe = Float8BlockScaling(
    fp8_format=te.common.recipe.Format.E4M3,  # FP8 format (default: E4M3)
    x_block_scaling_dim=1,  # 1D scaling for activations (default: 1)
    w_block_scaling_dim=2,  # 2D scaling for weights (default: 2)
    grad_block_scaling_dim=1,  # 1D scaling for gradients (default: 1)
)

# Create a linear layer
layer = te.Linear(1024, 1024)
optimizer = torch.optim.AdamW(layer.parameters(), lr=1e-4)

# Training with FP8 Blockwise Scaling
inp = torch.randn(32, 128, 1024, dtype=torch.bfloat16, device="cuda")

with te.fp8_autocast(enabled=True, fp8_recipe=recipe):
    output = layer(inp)
    loss = output.sum()

loss.backward()
optimizer.step()

# END_BLOCKWISE_SCALING_EXAMPLE
