# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Utils functions for the debug module."""


def next_iter_when_debug_should_be_run(quantizers):
    """Returns next iteration at which the debug should be run."""
    values = [q.get_next_debug_iter() for q in quantizers]
    non_false_values = [v for v in values if v != False]
    if len(non_false_values) == 0:
        return False
    return min(non_false_values)


def any_feature_enabled(quantizers):
    """Returns True if at least one API call is made from DebugQuantizer."""
    return any(q.any_feature_enabled() for q in quantizers)
