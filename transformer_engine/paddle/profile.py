# Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""Utils for profiling"""

from contextlib import contextmanager

from paddle.fluid import core


@contextmanager
def nvtx_range(msg):
    """Context to insert NVTX"""
    core.nvprof_nvtx_push(msg)
    yield
    core.nvprof_nvtx_pop()
