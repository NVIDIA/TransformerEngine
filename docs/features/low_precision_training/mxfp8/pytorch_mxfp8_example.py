# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

# START_MXFP8_EXAMPLE

import torch
import transformer_engine.pytorch as te
from transformer_engine.common.recipe import MXFP8BlockScaling, Format

# Create MXFP8 recipe
recipe = MXFP8BlockScaling(
    fp8_format=Format.E4M3,  # FP8 format (default: E4M3, E5M2 not supported)
)

# Create a linear layer
layer = te.Linear(1024, 1024)
optimizer = torch.optim.AdamW(layer.parameters(), lr=1e-4)

# Training with MXFP8
inp = torch.randn(32, 128, 1024, dtype=torch.bfloat16, device="cuda")

with te.fp8_autocast(enabled=True, fp8_recipe=recipe):
    output = layer(inp)
    loss = output.sum()

loss.backward()
optimizer.step()

# END_MXFP8_EXAMPLE
