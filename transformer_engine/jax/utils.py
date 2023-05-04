# Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""CUDA API helper"""
import ctypes

from cuda import cudart

_CUDA_SUCCESS = cudart.cudaError_t.cudaSuccess


def get_cublasLt_version():
    """Return cuBLASLt"""
    libcuBLASLt = ctypes.CDLL("libcublasLt.so")
    cublasLtGetVersion = libcuBLASLt.cublasLtGetVersion
    cublasLtGetVersion.restype = ctypes.c_size_t
    ver = cublasLtGetVersion()
    return ver


def get_cuda_version():
    """Return CUDA version"""
    ret, ver = cudart.cudaRuntimeGetVersion()
    assert ret == _CUDA_SUCCESS
    return ver / 1000


def get_device_compute_capability(gpu_id):
    """Return device compute capability"""
    (ret,) = cudart.cudaSetDevice(gpu_id)
    assert ret == _CUDA_SUCCESS
    flag = cudart.cudaDeviceAttr.cudaDevAttrComputeCapabilityMajor
    ret, major = cudart.cudaDeviceGetAttribute(flag, gpu_id)
    assert ret == _CUDA_SUCCESS
    flag = cudart.cudaDeviceAttr.cudaDevAttrComputeCapabilityMinor
    ret, minor = cudart.cudaDeviceGetAttribute(flag, gpu_id)
    assert ret == _CUDA_SUCCESS
    gpu_arch = major + minor / 10
    return gpu_arch
