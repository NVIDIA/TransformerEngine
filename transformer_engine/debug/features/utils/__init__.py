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
