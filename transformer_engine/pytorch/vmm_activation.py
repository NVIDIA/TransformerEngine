# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Fixed-address MUSA virtual-memory slots for graph-captured activations."""

from __future__ import annotations

from typing import Sequence

import torch
import transformer_engine_torch as tex


def vmm_driver_memory_info() -> dict[str, int]:
    """Return free and total bytes reported by the active MUSA driver context."""
    return {key: int(value) for key, value in dict(tex.vmm_driver_memory_info()).items()}


class MUSAActivationVMMAllocation:
    """Own a fixed virtual address and replaceable physical device backing."""

    def __init__(self, shape: Sequence[int], stride: Sequence[int], dtype: torch.dtype,
                 device: torch.device) -> None:
        self.shape = tuple(shape)
        self.stride = tuple(stride)
        self.dtype = dtype
        self.device = torch.device(device)
        if self.device.type != "musa":
            raise ValueError(f"MUSA VMM allocation requires a MUSA device, got {self.device}")
        if not self.shape or len(self.shape) != len(self.stride):
            raise ValueError(f"invalid shape/stride: {self.shape}/{self.stride}")
        if any(size <= 0 for size in self.shape) or any(value < 0 for value in self.stride):
            raise ValueError(f"unsupported shape/stride: {self.shape}/{self.stride}")
        element_size = torch.empty((), dtype=dtype).element_size()
        maximum_element_offset = sum((size - 1) * value for size, value in zip(self.shape, self.stride))
        self.storage_bytes = (maximum_element_offset + 1) * element_size
        device_index = self.device.index
        if device_index is None:
            device_index = torch.musa.current_device()
        self._allocation = tex.VMMActivationSlot(self.storage_bytes, device_index)
        self._tensor = self._allocation.tensor(self.shape, self.stride, self.dtype)
        self._address = self._tensor.data_ptr()
        info = self.info()
        if info["address"] != self._address:
            raise RuntimeError("VMM tensor pointer differs from its reserved virtual address")
        self.aligned_bytes = info["aligned_bytes"]
        self._closed = False

    @property
    def tensor(self) -> torch.Tensor:
        return self._tensor

    @property
    def address(self) -> int:
        return self._address

    @property
    def mapped(self) -> bool:
        return bool(self._allocation.info()["mapped"])

    def info(self) -> dict[str, int | bool]:
        return {key: value for key, value in dict(self._allocation.info()).items()}

    def unmap_and_release(self) -> None:
        self._allocation.unmap_and_release()

    def create_and_remap(self) -> None:
        self._allocation.create_and_remap()
        if self.info()["address"] != self._address or self._tensor.data_ptr() != self._address:
            raise RuntimeError("MUSA VMM allocation changed virtual address after remap")

    def close(self) -> None:
        if self._closed:
            return
        self._allocation.close()
        self._closed = True
