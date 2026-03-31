# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import torch

# Requires Ada (SM89) or newer for FP8 support
assert torch.cuda.get_device_capability()[0] >= 9 or (
    torch.cuda.get_device_capability()[0] == 8 and torch.cuda.get_device_capability()[1] >= 9
), "This example requires SM89 (Ada) or newer"

# START_AUTOCAST_BASIC

import torch
import transformer_engine.pytorch as te
from transformer_engine.common.recipe import DelayedScaling, Format

recipe = DelayedScaling()
layer = te.Linear(1024, 1024)
inp = torch.randn(32, 1024, dtype=torch.float32, device="cuda")

with te.autocast(enabled=True, recipe=recipe):
    output = layer(inp)

# .backward() is called outside of autocast
loss = output.sum()
loss.backward()

# END_AUTOCAST_BASIC


# START_AUTOCAST_SEQUENTIAL

encoder_recipe = DelayedScaling(fp8_format=Format.E4M3)
decoder_recipe = DelayedScaling(fp8_format=Format.HYBRID)

encoder = te.Linear(1024, 1024)
decoder = te.Linear(1024, 1024)

with te.autocast(enabled=True, recipe=encoder_recipe):
    hidden = encoder(inp)

with te.autocast(enabled=True, recipe=decoder_recipe):
    output = decoder(hidden)

# END_AUTOCAST_SEQUENTIAL


# START_AUTOCAST_NESTED

outer_recipe = DelayedScaling(fp8_format=Format.E4M3)
inner_recipe = DelayedScaling(fp8_format=Format.HYBRID)

layer1 = te.Linear(1024, 1024)
layer2 = te.Linear(1024, 1024)
layer3 = te.Linear(1024, 1024)

with te.autocast(enabled=True, recipe=outer_recipe):
    # layer1 uses outer_recipe
    x = layer1(inp)

    with te.autocast(enabled=True, recipe=inner_recipe):
        # layer2 uses inner_recipe (overrides outer)
        x = layer2(x)

    # layer3 uses outer_recipe again
    output = layer3(x)

# END_AUTOCAST_NESTED
