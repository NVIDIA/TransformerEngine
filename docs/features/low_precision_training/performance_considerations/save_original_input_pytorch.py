# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import torch

# Requires Ada (SM89) or Hopper (SM90), different results on Blackwell+
cc = torch.cuda.get_device_capability()
assert cc[0] == 8 and cc[1] >= 9 or cc[0] == 9, "This example requires SM89 (Ada) or SM90 (Hopper)"

print("# START_SAVE_ORIGINAL_INPUT")
# START_SAVE_ORIGINAL_INPUT
import torch
import transformer_engine.pytorch as te
from transformer_engine.common.recipe import Float8CurrentScaling

recipe = Float8CurrentScaling()


def residual_block(layer, inp):
    """Residual connection: input is saved for addition after linear."""
    out = layer(inp)
    return out + inp  # inp must be kept for this addition


def measure_memory(use_save_original):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    layer = te.Linear(
        1024, 1024, params_dtype=torch.bfloat16, save_original_input=use_save_original
    )
    inp = torch.randn(1024, 1024, dtype=torch.bfloat16, device="cuda", requires_grad=True)

    with te.autocast(enabled=True, recipe=recipe):
        out = residual_block(layer, inp)
    out.sum().backward()

    return torch.cuda.max_memory_allocated() / 1024**2


# Warmup runs
measure_memory(False)
measure_memory(True)

# Actual measurements
for use_save_original in [False, True]:
    peak = measure_memory(use_save_original)
    print(f"save_original_input={use_save_original}: {peak:.1f} MB")
# END_SAVE_ORIGINAL_INPUT
print("# END_SAVE_ORIGINAL_INPUT")
