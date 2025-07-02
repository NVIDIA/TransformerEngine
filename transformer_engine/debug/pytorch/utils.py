# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Utils functions for the debug module."""


def next_iter_for_debug(quantizers):
    """Returns True if at least one API call is made from DebugQuantizer."""
    if any(q.get_next_debug_iter() is None for q in quantizers):
        return None
    return min(q.get_next_debug_iter() for q in quantizers)
