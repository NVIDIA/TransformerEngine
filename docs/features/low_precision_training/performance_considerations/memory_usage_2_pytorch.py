# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import torch

# Requires Ada (SM89) or Hopper (SM90), different results on Blackwell+
cc = torch.cuda.get_device_capability()
assert cc[0] == 8 and cc[1] >= 9 or cc[0] == 9, "This example requires SM89 (Ada) or SM90 (Hopper)"

print("# START_MEMORY_USAGE_2")
import torch
import transformer_engine.pytorch as te


def measure_memory():
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    init_memory = torch.cuda.memory_allocated()
    layer = te.Linear(1024, 1024, params_dtype=torch.bfloat16)

    inp = torch.randn(1024, 1024, dtype=torch.bfloat16, device="cuda")
    with te.autocast(enabled=True):
        out = layer(inp)
    del inp  # Input is saved by model for backward, not by user script

    mem_after_forward = torch.cuda.memory_allocated() - init_memory
    return mem_after_forward


# Warmup run
measure_memory()

# Actual measurement
mem_after_forward = measure_memory()
print(f"Memory after forward pass: {mem_after_forward/1024**2:.2f} MB")
# END_MEMORY_USAGE_2
print("# END_MEMORY_USAGE_2")
