# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""
Utility functions for ONNX export.
"""

import time
import torch


def _measure_time(f):

    time_taken = []
    num_iterations = 10
    f()  # warm-up

    for _ in range(num_iterations):
        start_time = time.time()
        f()
        torch.cuda.synchronize()
        end_time = time.time()
        time_taken.append(end_time - start_time)
    return round(sum(time_taken) / num_iterations, 3)
