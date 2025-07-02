# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""
Utils for the debug features.
"""


def next_enabled_iter(start_step, end_step, start_end_list, freq, iteration):
    """ Returns next iteration at which the feature will be enabled. """

    if start_end_list:
        intervals = sorted(start_end_list)
    else:
        if start_step is None:
            return None
        end = float("inf") if end_step is None else end_step
        intervals = [(start_step, end)]

    for s, e in intervals:
        first = max(iteration + 1, s)
        offset = first % freq
        candidate = first if offset == 0 else first + (freq - offset)
        if candidate <= e:
            return candidate