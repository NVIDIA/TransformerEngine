# Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""Utils for profiling"""

import os
import ctypes
from contextlib import contextmanager

_cuda_home = os.environ.get('CUDA_HOME', '/usr/local/cuda')

_cudart = ctypes.CDLL(os.path.join(_cuda_home, 'lib64/libcudart.so'))


def cuda_profile_start():
    """Calls cudaProfilerStart"""
    _cudart.cudaProfilerStart()


def cuda_profile_stop():
    """Calls cudaProfilerStop"""
    _cudart.cudaProfilerStop()


_nvtx = ctypes.CDLL(os.path.join(_cuda_home, 'lib64/libnvToolsExt.so'))


def cuda_nvtx_range_push(name):
    """Calls nvtxRangePushW"""
    _nvtx.nvtxRangePushW(ctypes.c_wchar_p(name))


def cuda_nvtx_range_pop():
    """Calls nvtxRangePop"""
    _nvtx.nvtxRangePop()


@contextmanager
def nvtx_range(msg):
    """Context to insert NVTX"""
    cuda_nvtx_range_push(msg)
    yield
    cuda_nvtx_range_pop()
