# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""CUDA-driver locality-domain green contexts.

This avoids depending on experimental PyTorch locality-domain APIs. PyTorch is
only used to wrap the resulting ``CUstream`` handles as ``ExternalStream``.
"""

from __future__ import annotations

import ctypes
import os
from typing import List, Optional

import torch
from cuda.bindings import driver


_CU_DEVICE_ATTRIBUTE_LOCALITY_DOMAIN_COUNT = 149
_CU_DEV_SM_RESOURCE_GROUP_BACKFILL = 0x1
_CU_DEV_SM_RESOURCE_GROUP_LOCALITY_DOMAIN_ID = 0x2
_GROUP_PARAMS_SIZE = 64
_CUDEV_RESOURCE_SIZE = 144
_SM_OFFSET = 96
_libcuda: Optional[ctypes.CDLL] = None


def _check(error, operation: str) -> None:
    if isinstance(error, tuple):
        error = error[0]
    if error != driver.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"{operation} failed: {error}")


def _object_pointer(obj) -> int:
    try:
        return obj.getPtr()
    except AttributeError:
        return ctypes.addressof(obj)


def _write_u32(buffer, byte_offset: int, value: int) -> None:
    ctypes.cast(
        ctypes.addressof(buffer) + byte_offset,
        ctypes.POINTER(ctypes.c_uint32),
    )[0] = value


def _read_u32(buffer, byte_offset: int) -> int:
    return ctypes.cast(
        ctypes.addressof(buffer) + byte_offset,
        ctypes.POINTER(ctypes.c_uint32),
    )[0]


def _get_libcuda() -> ctypes.CDLL:
    global _libcuda
    if _libcuda is None:
        _libcuda = ctypes.CDLL("libcuda.so.1")
        _libcuda.cuDevSmResourceSplit.restype = ctypes.c_int
        _libcuda.cuDevSmResourceSplit.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_uint,
            ctypes.c_void_p,
        ]
    return _libcuda


def _get_device_attribute(cuda_device, attribute: int) -> int:
    try:
        enum_attribute = driver.CUdevice_attribute(attribute)
        error, value = driver.cuDeviceGetAttribute(enum_attribute, cuda_device)
        _check(error, f"cuDeviceGetAttribute({attribute})")
        return value
    except ValueError:
        value = ctypes.c_int(0)
        result = _get_libcuda().cuDeviceGetAttribute(
            ctypes.byref(value),
            ctypes.c_int(attribute),
            ctypes.c_int(int(cuda_device)),
        )
        if result != 0:
            raise RuntimeError(
                f"cuDeviceGetAttribute({attribute}) failed with driver error {result}"
            )
        return value.value


def _split_sm_resources(cuda_device, num_domains: int):
    error, device_resource = driver.cuDeviceGetDevResource(
        cuda_device,
        driver.CUdevResourceType.CU_DEV_RESOURCE_TYPE_SM,
    )
    _check(error, "cuDeviceGetDevResource(SM)")

    requested_sms = int(os.getenv("NVTE_LOCALIZATION_SMS_PER_DOMAIN", "0"))
    coscheduled_sms = int(os.getenv("NVTE_LOCALIZATION_COSCHEDULED_SMS", "2"))
    group_flags = _CU_DEV_SM_RESOURCE_GROUP_LOCALITY_DOMAIN_ID
    if requested_sms > 0:
        group_flags |= _CU_DEV_SM_RESOURCE_GROUP_BACKFILL

    result_buffer = (ctypes.c_ubyte * (_CUDEV_RESOURCE_SIZE * num_domains))()
    remainder_buffer = (ctypes.c_ubyte * _CUDEV_RESOURCE_SIZE)()
    params_buffer = (ctypes.c_ubyte * (_GROUP_PARAMS_SIZE * num_domains))()
    for domain in range(num_domains):
        offset = domain * _GROUP_PARAMS_SIZE
        _write_u32(params_buffer, offset, requested_sms)
        _write_u32(params_buffer, offset + 4, coscheduled_sms)
        _write_u32(params_buffer, offset + 8, coscheduled_sms)
        _write_u32(params_buffer, offset + 12, group_flags)
        _write_u32(params_buffer, offset + 16, domain)

    result = _get_libcuda().cuDevSmResourceSplit(
        ctypes.addressof(result_buffer),
        ctypes.c_uint(num_domains),
        ctypes.c_void_p(_object_pointer(device_resource)),
        ctypes.addressof(remainder_buffer),
        ctypes.c_uint(0),
        ctypes.addressof(params_buffer),
    )
    if result != 0:
        raise RuntimeError(
            "cuDevSmResourceSplit failed with driver error "
            f"{result} ({num_domains=}, {requested_sms=}, {coscheduled_sms=})"
        )

    resources: List[driver.CUdevResource] = []
    for domain in range(num_domains):
        resource = driver.CUdevResource()
        ctypes.memmove(
            _object_pointer(resource),
            ctypes.addressof(result_buffer) + domain * _CUDEV_RESOURCE_SIZE,
            _CUDEV_RESOURCE_SIZE,
        )
        sm_count = _read_u32(result_buffer, domain * _CUDEV_RESOURCE_SIZE + _SM_OFFSET)
        if sm_count == 0:
            raise RuntimeError(f"Locality domain {domain} has no assigned SMs")
        resources.append(resource)
    return resources


class DriverLocalityContext:
    """Own two memory-node-affine CUDA green contexts and streams."""

    def __init__(self, device_index: int) -> None:
        _check(driver.cuInit(0), "cuInit")
        error, self.cuda_device = driver.cuDeviceGet(device_index)
        _check(error, "cuDeviceGet")
        error, self.primary_context = driver.cuDevicePrimaryCtxRetain(self.cuda_device)
        _check(error, "cuDevicePrimaryCtxRetain")
        _check(driver.cuCtxSetCurrent(self.primary_context), "cuCtxSetCurrent")

        self.num_domains = _get_device_attribute(
            self.cuda_device,
            _CU_DEVICE_ATTRIBUTE_LOCALITY_DOMAIN_COUNT,
        )
        if self.num_domains != 2:
            raise RuntimeError(f"Expected exactly 2 locality domains, got {self.num_domains}")

        self.green_contexts = []
        self.cuda_streams = []
        self.streams = []
        device = torch.device("cuda", device_index)
        for domain, resource in enumerate(_split_sm_resources(self.cuda_device, self.num_domains)):
            error, descriptor = driver.cuDevResourceGenerateDesc([resource], 1)
            _check(error, f"cuDevResourceGenerateDesc(domain={domain})")
            error, green_context = driver.cuGreenCtxCreate(
                descriptor,
                self.cuda_device,
                driver.CUgreenCtxCreate_flags.CU_GREEN_CTX_DEFAULT_STREAM,
            )
            _check(error, f"cuGreenCtxCreate(domain={domain})")
            error, cuda_stream = driver.cuGreenCtxStreamCreate(
                green_context,
                driver.CUstream_flags.CU_STREAM_NON_BLOCKING,
                0,
            )
            _check(error, f"cuGreenCtxStreamCreate(domain={domain})")
            self.green_contexts.append(green_context)
            self.cuda_streams.append(cuda_stream)
            self.streams.append(torch.cuda.ExternalStream(int(cuda_stream), device=device))

        self.green_contexts = tuple(self.green_contexts)
        self.cuda_streams = tuple(self.cuda_streams)
        self.streams = tuple(self.streams)
