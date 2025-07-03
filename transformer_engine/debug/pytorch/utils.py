# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Utils functions for the debug module."""


def next_iter_when_debug_should_be_run(quantizers):
    """Returns next iteration at which the debug should be run."""
    return min(q.get_next_debug_iter() for q in quantizers)


def any_feature_enabled(quantizers):
    """Returns True if at least one API call is made from DebugQuantizer."""
    return any(q.any_feature_enabled() for q in quantizers)


def _as_pair(x):
    """If x is a tuple, return x, otherwise return (x, None)"""
    if isinstance(x, tuple):
        assert len(x) == 2, "Expected a tuple of length 2"
        return x
    else:
        return x, None
