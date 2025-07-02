# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""
Utils for the debug features.
"""


def next_enabled_iter(start_step, end_step, start_end_list, freq, iteration):
    """ Returns next iteration at which the feature will be enabled. """

    run_current = False

    if start_end_list:
        intervals = sorted(start_end_list)
    else:
        start_step = 0 if start_step is None else start_step
        end = float("inf") if end_step is None else end_step
        intervals = [(start_step, end)]

    for s, e in intervals:
        if iteration % freq == 0 and iteration >= s and iteration <= e:
            run_current = True

        first = max(iteration + 1, s)
        offset = first % freq
        candidate = first if offset == 0 else first + (freq - offset)
        if candidate <= e:
            return run_current, candidate