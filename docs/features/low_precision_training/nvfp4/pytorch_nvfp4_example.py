# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import torch

# Check for Blackwell or newer GPU
major, minor = torch.cuda.get_device_capability()
assert major >= 10, f"NVFP4 requires SM100 (Blackwell) or later, got SM{major}{minor}"

# START_NVFP4_EXAMPLE

import torch
import transformer_engine.pytorch as te
from transformer_engine.common.recipe import NVFP4BlockScaling

# Define NVFP4 recipe
# 2D weight quantization and RHT are enabled by default
recipe = NVFP4BlockScaling()
# To disable features, use:
#   recipe = NVFP4BlockScaling(disable_rht=True, disable_2d_quantization=True)

# Create a linear layer with bfloat16 parameters
layer = te.Linear(1024, 1024, params_dtype=torch.bfloat16)

# Forward and backward pass
inp = torch.randn(32, 128, 1024, dtype=torch.bfloat16, device="cuda")

with te.autocast(enabled=True, recipe=recipe):
    output = layer(inp)
    loss = output.sum()

loss.backward()

# END_NVFP4_EXAMPLE
