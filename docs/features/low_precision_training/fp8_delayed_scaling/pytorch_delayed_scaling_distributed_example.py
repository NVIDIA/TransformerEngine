# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

# START_AMAX_REDUCTION_EXAMPLE
import torch.distributed as dist
import transformer_engine.pytorch as te
from transformer_engine.common.recipe import DelayedScaling

# Create process group for amax reduction (e.g., all 8 GPUs)
amax_reduction_group = dist.new_group(ranks=[0, 1, 2, 3, 4, 5, 6, 7])

recipe = DelayedScaling(reduce_amax=True)

with te.autocast(recipe=recipe, amax_reduction_group=amax_reduction_group):
    output = model(inp)

# END_AMAX_REDUCTION_EXAMPLE
