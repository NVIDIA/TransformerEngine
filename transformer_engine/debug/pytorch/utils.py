# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Utils functions for the debug module."""


def any_feature_enabled(quantizers):
    """Returns True if at least one API call is made from DebugQuantizer."""
    return any(q.any_feature_enabled() for q in quantizers)
