# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

# START_NVFP4_EXAMPLE

import torch
import transformer_engine.pytorch as te
from transformer_engine.common.recipe import NVFP4Recipe, Format

# Define NVFP4 recipe
# Key features like 2D weight quantization and RHT can be enabled here
recipe = NVFP4Recipe(
    fp8_format=Format.E4M3,
    use_2d_weight_quantization=True,
    use_rht=True,
)

# Create a linear layer and optimizer
layer = te.Linear(1024, 1024)
optimizer = torch.optim.AdamW(layer.parameters(), lr=1e-4)

# Training step
inp = torch.randn(32, 128, 1024, dtype=torch.bfloat16, device="cuda")

with te.fp8_autocast(enabled=True, fp8_recipe=recipe):
    output = layer(inp)
    loss = output.sum()

loss.backward()
optimizer.step()

# END_NVFP4_EXAMPLE
