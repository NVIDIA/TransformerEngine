# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

# START_CURRENT_SCALING_EXAMPLE

import torch
import transformer_engine.pytorch as te
from transformer_engine.common.recipe import Float8CurrentScaling, Format

# Create FP8 Current Scaling recipe
# Available formats:
#   - Format.HYBRID (default) -- E4M3 for forward pass, E5M2 for backward pass
#   - Format.E4M3 -- E4M3 for both forward and backward pass
recipe = Float8CurrentScaling(fp8_format=Format.HYBRID)

# Create a simple linear layer with bfloat16 parameters
layer = te.Linear(1024, 1024, params_dtype=torch.bfloat16)

# Forward and backward pass
inp = torch.randn(32, 128, 1024, dtype=torch.bfloat16, device="cuda")

with te.autocast(enabled=True, recipe=recipe):
    output = layer(inp)
    loss = output.sum()

loss.backward()

# END_CURRENT_SCALING_EXAMPLE
