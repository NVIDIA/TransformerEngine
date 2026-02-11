# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import torch

# Check for Blackwell or newer GPU
major, minor = torch.cuda.get_device_capability()
assert major >= 10, f"MXFP8 requires SM100 (Blackwell) or later, got SM{major}{minor}"

# START_MXFP8_EXAMPLE

import torch
import transformer_engine.pytorch as te
from transformer_engine.common.recipe import MXFP8BlockScaling, Format

# Create MXFP8 recipe
recipe = MXFP8BlockScaling(
    fp8_format=Format.E4M3,  # E4M3 (default) or HYBRID; pure E5M2 not supported
)

# Create a linear layer with bfloat16 parameters
layer = te.Linear(1024, 1024, params_dtype=torch.bfloat16)

# Forward and backward pass
inp = torch.randn(32, 128, 1024, dtype=torch.bfloat16, device="cuda")

with te.autocast(enabled=True, recipe=recipe):
    output = layer(inp)
    loss = output.sum()

loss.backward()

# END_MXFP8_EXAMPLE
