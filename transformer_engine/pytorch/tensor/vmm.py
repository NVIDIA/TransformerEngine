# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Minimal CUDA VMM allocator for two-domain localization experiments.

This module intentionally supports only allocations whose logical midpoint is
already aligned to the CUDA VMM granularity. That restriction keeps the
returned tensor truly contiguous: there is no hidden padding between its two
row partitions.
"""

from __future__ import annotations

import ctypes
from functools import reduce
from operator import mul
from typing import Dict, Optional, Tuple

import torch


# cuda.h (CUDA 13.4): CU_MEM_LOCATION_TYPE_DEVICE_MEMORY_NODE.
# cuda-bindings releases that predate the enum still have the right ABI layout.
_CU_MEM_LOCATION_TYPE_DEVICE_MEMORY_NODE = 6


def _cuda_object_pointer(obj) -> int:
    try:
        return obj.getPtr()
    except AttributeError:
        return ctypes.addressof(obj)


def _check_cuda(error, operation: str) -> None:
    from cuda.bindings import driver

    if isinstance(error, tuple):
        error = error[0]
    if error != driver.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"{operation} failed: {error}")


def _set_memory_node(
    prop,
    device_index: int,
    locality_domain_ordinal: int,
) -> None:
    """Set the localized location union on a CUmemAllocationProp."""
    pointer = _cuda_object_pointer(prop)
    ctypes.cast(pointer + 8, ctypes.POINTER(ctypes.c_int))[0] = (
        _CU_MEM_LOCATION_TYPE_DEVICE_MEMORY_NODE
    )
    ctypes.cast(pointer + 12, ctypes.POINTER(ctypes.c_int))[0] = (
        int(device_index) | (int(locality_domain_ordinal) << 8)
    )


class VMMRowSplitAllocator:
    """Own VMM allocations split evenly across two GPU locality domains.

    Allocations live until :meth:`close` is called. This explicit lifetime is
    suitable for persistent CUDA-graph buffers and avoids relying on Python
    finalization order for raw CUDA mappings.
    """

    def __init__(self, device: Optional[torch.device | str | int] = None) -> None:
        try:
            from cuda.bindings import driver
        except ImportError as exc:
            raise RuntimeError(
                "VMM localization requires the cuda.bindings Python package"
            ) from exc

        if device is None:
            device = torch.device("cuda", torch.cuda.current_device())
        elif isinstance(device, int):
            device = torch.device("cuda", device)
        else:
            device = torch.device(device)
        if device.type != "cuda":
            raise ValueError(f"VMM localization requires a CUDA device, got {device}")
        self.device = device
        self.device_index = (
            torch.cuda.current_device() if device.index is None else device.index
        )
        _check_cuda(driver.cuInit(0), "cuInit")
        self._driver = driver
        self._granularity: Optional[int] = None
        self._allocations: Dict[int, Tuple[int, Tuple[object, object]]] = {}

    def _allocation_properties(self, domain: Optional[int] = None):
        driver = self._driver
        prop = driver.CUmemAllocationProp()
        prop.type = driver.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
        if domain is None:
            prop.location.type = driver.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
            prop.location.id = self.device_index
        else:
            _set_memory_node(prop, self.device_index, domain)
        return prop

    @property
    def granularity(self) -> int:
        """Recommended VMM mapping granularity in bytes."""
        if self._granularity is None:
            driver = self._driver
            result = driver.cuMemGetAllocationGranularity(
                self._allocation_properties(),
                driver.CUmemAllocationGranularity_flags.CU_MEM_ALLOC_GRANULARITY_RECOMMENDED,
            )
            _check_cuda(result[0], "cuMemGetAllocationGranularity")
            self._granularity = int(result[1])
        return self._granularity

    def _access_descriptor(self):
        driver = self._driver
        descriptor = driver.CUmemAccessDesc()
        descriptor.location.type = driver.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
        descriptor.location.id = self.device_index
        descriptor.flags = driver.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE
        return descriptor

    def allocate(self, shape: Tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
        """Allocate one contiguous tensor with a domain boundary at dim-0 midpoint."""
        if not shape or shape[0] % 2 != 0:
            raise ValueError(f"Expected an even, non-empty leading dimension, got {shape}")
        element_size = torch.empty((), dtype=dtype).element_size()
        numel = reduce(mul, shape, 1)
        total_bytes = numel * element_size
        half_bytes = total_bytes // 2
        if total_bytes == 0 or total_bytes % 2 != 0:
            raise ValueError(f"Allocation size must be positive and even, got {total_bytes}")
        if half_bytes % self.granularity != 0:
            raise ValueError(
                "Each logical row partition must be VMM-granularity aligned "
                f"(half={half_bytes} bytes, granularity={self.granularity} bytes)"
            )

        driver = self._driver
        result = driver.cuMemAddressReserve(total_bytes, self.granularity, 0, 0)
        _check_cuda(result[0], "cuMemAddressReserve")
        base = int(result[1])

        handles = []
        mapped_bytes = 0
        try:
            for domain in range(2):
                result = driver.cuMemCreate(
                    half_bytes,
                    self._allocation_properties(domain),
                    0,
                )
                _check_cuda(result[0], f"cuMemCreate(domain={domain})")
                handle = result[1]
                handles.append(handle)
                _check_cuda(
                    driver.cuMemMap(
                        base + domain * half_bytes,
                        half_bytes,
                        0,
                        handle,
                        0,
                    ),
                    f"cuMemMap(domain={domain})",
                )
                mapped_bytes += half_bytes
            _check_cuda(
                driver.cuMemSetAccess(
                    base,
                    total_bytes,
                    [self._access_descriptor()],
                    1,
                ),
                "cuMemSetAccess",
            )
        except Exception:
            if mapped_bytes:
                driver.cuMemUnmap(base, mapped_bytes)
            for handle in handles:
                driver.cuMemRelease(handle)
            driver.cuMemAddressFree(base, total_bytes)
            raise

        self._allocations[base] = (total_bytes, (handles[0], handles[1]))
        storage = torch._C._construct_storage_from_data_pointer(
            base,
            torch.device("cuda", self.device_index),
            total_bytes,
        )
        strides = []
        stride = 1
        for dimension in reversed(shape):
            strides.append(stride)
            stride *= dimension
        strides.reverse()
        tensor = torch.empty(
            0,
            dtype=dtype,
            device=torch.device("cuda", self.device_index),
        )
        tensor.set_(storage, 0, shape, tuple(strides))
        # Some PyTorch versions retain alias metadata after set_(), which makes
        # custom autograd Functions reject mark_dirty() on multiple VMM-backed
        # outputs. detach() creates a root tensor wrapper without changing the
        # data pointer or VMM physical placement.
        tensor = tensor.detach()
        # Keep the allocator reachable for as long as a full tensor is alive.
        # Views are used only while their owning full tensor is retained.
        tensor._nvte_vmm_allocator = self
        return tensor

    def split(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return the two dim-0 views matching the physical VMM mappings."""
        midpoint = tensor.shape[0] // 2
        return tensor[:midpoint], tensor[midpoint:]

    def close(self) -> None:
        """Synchronize and release all mappings owned by this allocator."""
        if not self._allocations:
            return
        torch.cuda.synchronize(self.device_index)
        driver = self._driver
        for base, (total_bytes, handles) in list(self._allocations.items()):
            _check_cuda(driver.cuMemUnmap(base, total_bytes), "cuMemUnmap")
            for handle in handles:
                _check_cuda(driver.cuMemRelease(handle), "cuMemRelease")
            _check_cuda(
                driver.cuMemAddressFree(base, total_bytes),
                "cuMemAddressFree",
            )
            del self._allocations[base]
