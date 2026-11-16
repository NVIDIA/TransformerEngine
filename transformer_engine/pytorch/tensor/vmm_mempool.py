# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# See LICENSE for license information.

"""Allocator-managed MemPool with reusable, domain-local physical VMM blocks."""

from contextlib import contextmanager

import torch


def is_vmm_mempool_pointer(device_index, pointer):
    """Recognize allocations even after the Python pool wrapper is destroyed."""
    import transformer_engine_torch as tex

    return hasattr(tex, "vmm_mempool_contains") and tex.vmm_mempool_contains(device_index, pointer)


class VMMMemPool(torch.cuda.MemPool):
    """Map reusable physical blocks into a new contiguous range per allocation.

    ``domain=-1`` places the rounded first half in LD0 and the second in LD1.
    ``domain=0`` or ``1`` backs the entire tensor in that domain. The logical
    midpoint can differ from the physical midpoint by less than ``block_bytes``.
    This requires the experimental PyTorch allocator-managed MemPool interface
    and TE built with ``NVTE_WITH_VMM_MEMPOOL=1``. Graph Tree checkpoints and
    CUDA IPC are unsupported. Ordinary static CUDA Graph capture is supported.

    PyTorch owns Storage/alias and recorded-stream tracking. The native backend
    adds allocation-stream completion events before unmapping eager allocations.
    Captured allocations keep their addresses until ``empty_cache`` is allowed
    by PyTorch after all graphs retaining this pool have been reset.
    """

    def __init__(self, device=None, *, domain=-1, block_bytes=0):
        if domain not in (-1, 0, 1):
            raise ValueError(f"Expected domain -1, 0, or 1, got {domain}")
        if block_bytes < 0:
            raise ValueError("block_bytes must be nonnegative")
        if device is None:
            device = torch.device("cuda", torch.cuda.current_device())
        elif isinstance(device, int):
            device = torch.device("cuda", device)
        else:
            device = torch.device(device)
        if device.type != "cuda":
            raise ValueError(f"VMM MemPool requires a CUDA device, got {device}")
        self.device_index = torch.cuda.current_device() if device.index is None else device.index
        self.domain = domain
        if not hasattr(torch._C, "_cuda_customManagedPoolAllocator"):
            raise RuntimeError(
                "VMM MemPool requires PyTorch with allocator-managed MemPool support"
            )
        import transformer_engine_torch as tex

        if not hasattr(tex, "VMMMemPoolBackend"):
            raise RuntimeError(
                "Rebuild TE with NVTE_WITH_VMM_MEMPOOL=1 to enable VMM MemPool support"
            )
        with torch.cuda.device(self.device_index):
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("Create VMM MemPools before CUDA Graph capture")
            self._backend = tex.VMMMemPoolBackend(self.device_index, domain, block_bytes)
            super().__init__(self._backend.allocator, allocator_managed=True)

    @contextmanager
    def use(self):
        """Route this thread's allocations to the pool on its owning device."""
        with torch.cuda.device(self.device_index), torch.cuda.use_mem_pool(self, self.device_index):
            yield self

    def contains(self, pointer):
        """Whether a pointer lies in a live logical allocation, including views."""
        return self._backend.contains(pointer)

    def stats(self):
        """Physical bytes, mapping costs, and reuse counters, separate from torch stats."""
        return self._backend.stats()

    def empty_cache(self):
        """Drain completed frees and release cached backing; keep live tensors valid.

        PyTorch rejects this operation during capture, inside ``use()``, or while
        any CUDA graph retains this pool. Call it after graph reset to retire
        captured mappings. No public method bypasses that lifetime check.
        """
        with torch.cuda.device(self.device_index):
            torch.cuda.memory.empty_allocator_cache(self)
