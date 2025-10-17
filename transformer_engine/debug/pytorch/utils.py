# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Utils functions for the debug module."""

from typing import Optional


def next_iter_when_debug_should_be_run(quantizers) -> Optional[int]:
    """
    Returns next iteration at which the debug should be run.
    If debug will never be run for this layer, returns None.
    """

    out = None
    for q in quantizers:
        if q.get_next_debug_iter() is not None:
            if out is None:
                out = q.get_next_debug_iter()
            else:
                out = min(out, q.get_next_debug_iter())

    return out


def any_feature_enabled(quantizers):
    """Returns True if at least one API call is made from DebugQuantizer."""
    return any(q.any_feature_enabled() for q in quantizers)
