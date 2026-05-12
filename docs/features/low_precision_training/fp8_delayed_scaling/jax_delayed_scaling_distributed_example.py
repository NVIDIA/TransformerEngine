# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

# START_AMAX_REDUCTION_EXAMPLE
import transformer_engine.jax as te
from transformer_engine.common.recipe import DelayedScaling

# Amax reduction scope is managed internally
recipe = DelayedScaling(reduce_amax=True)  # Must be True in JAX

with te.autocast(enabled=True, recipe=recipe):
    output = layer.apply(params, inp)

# END_AMAX_REDUCTION_EXAMPLE
