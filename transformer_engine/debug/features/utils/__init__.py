# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""
Utils for the debug features.
"""

import torch
import nvdlfw_inspect.api as debug_api

from transformer_engine.debug.pytorch.debug_state import TEDebugState


def get_reduction_params(tensor_name: str, tp_group: torch.distributed.ProcessGroup):
    """
    Returns the statistics reduction parameters for the tensor.
    """
    skip_reduction = False
    reduction_group = debug_api.get_tensor_reduction_group()
    reduce_within_microbatch = tensor_name != "weight"
    if tensor_name == "weight":
        if TEDebugState.weight_tensor_tp_group_reduce:
            reduction_group = tp_group
        else:
            skip_reduction = True
    return skip_reduction, reduction_group, reduce_within_microbatch


def next_enabled_iter(start_step, end_step, start_end_list, freq, iteration):
    """
    Determines whether the feature should be enabled at the current iteration,
    and computes the next iteration at which the feature will be enabled.

    Returns
    -------
    run_current : bool
        True if the feature should be enabled at the current iteration.
    next_iter : int
        The next iteration index at which the feature will be enabled.
    """

    run_current = False

    if start_end_list:
        intervals = sorted(start_end_list)
    else:
        start_step = 0 if start_step is None else start_step
        end = float("inf") if end_step is None else end_step
        intervals = [(start_step, end)]

    for s, e in intervals:
        if iteration % freq == 0 and s <= iteration <= e:
            run_current = True

        first = max(iteration + 1, s)
        offset = first % freq
        candidate = first if offset == 0 else first + (freq - offset)
        if candidate <= e:
            return run_current, candidate

    return run_current, None  # No next iteration found
