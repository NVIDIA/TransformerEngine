# Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""Methods needed for recompute."""

import os
import inspect

from paddle.distributed import fleet

from .constants import RecomputeFunctionNames
from .fp8 import get_global_fp8_state

__all__ = ['recompute', 'is_in_recompute_phase']

_DISABLE_RECOMPUTE = int(os.getenv("NVTE_DISABLE_RECOMPUTE", "0"))


def is_in_recompute_phase():
    """Inspect call stack to determine if this is called from
    backward phase. Paddle has two recompute methods:
    (1) Use RecomputeFunction. The recomputed function is called from `RecomputeFunction.backward`;
    (2) Use paddle.autograd.saved_tensors_hooks. The recompute function is called from `unpack`."""
    if _DISABLE_RECOMPUTE:
        return False
    frame = inspect.currentframe().f_back
    while frame:
        if frame.f_code.co_name in RecomputeFunctionNames:
            return True
        frame = frame.f_back
    return False


def recompute(function, *args, **kwargs):
    """
    This is a wrapper of paddle.distributed.fleet.utils.recompute. It provides necessary
    state information for fp8 layers.
    """
    assert not _DISABLE_RECOMPUTE, "Recompute is disabled. " \
        f"Got NVTE_DISABLE_RECOMPUTE={_DISABLE_RECOMPUTE}."

    global_fp8_state = get_global_fp8_state()

    try:
        global_fp8_state._fp8_recompute_enabled = True
        outputs = fleet.utils.recompute(function, *args, **kwargs)
    finally:
        global_fp8_state._fp8_recompute_enabled = False

    return outputs
