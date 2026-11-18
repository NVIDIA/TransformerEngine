# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# See LICENSE for license information.

"""Localized tensors using ordinary PyTorch MemPools and reusable VMM blocks."""

import math
import operator
import os
import threading

import torch


# Keep pending native pools alive even if the Python caller drops its controller
# during capture. Explicit global draining also reaches these orphaned controllers.
_CONTROLLERS = set()


def is_vmm_mempool_pointer(device_index, pointer):
    """Recognize allocations even after the Python controller is destroyed."""
    import transformer_engine_torch as tex

    return hasattr(tex, "vmm_mempool_contains") and tex.vmm_mempool_contains(device_index, pointer)


def clear_all_vmm_mem_pool_caches():
    """Drain dedicated native pools and unused physical backing on all devices."""
    import transformer_engine_torch as tex

    for controller in list(_CONTROLLERS):
        controller.empty_cache()
    if hasattr(tex, "vmm_mempool_empty_all_caches"):
        tex.vmm_mempool_empty_all_caches()


class VMMMemPool:
    """Allocate each localized tensor in a dedicated ordinary torch.cuda.MemPool.

    A controller shares physical-block caches, not mapped tensor addresses. Each
    call to allocate creates a single-use native MemPool, so native segment
    caching cannot serve a later tensor with the wrong split boundary. Only the
    backing tensor is allocated in that pool; any extra native segment tail is
    reserved virtual space with no physical backing.

    This is a controller, not a torch.cuda.MemPool subclass or a routing context
    for arbitrary torch.empty calls. Use allocate(shape, dtype). No experimental
    allocator-managed PyTorch API is required in the default dedicated mode.
    With mode="managed" (NVTE_VMM_MEMPOOL_MODE=managed), a patched PyTorch build
    routes every allocation through one shared MemPool per controller instead.
    """

    def __init__(self, device=None, *, domain=-1, block_bytes=0, mode=None):
        self.mode = os.getenv("NVTE_VMM_MEMPOOL_MODE", "dedicated") if mode is None else mode
        if self.mode not in ("dedicated", "managed"):
            raise ValueError(f"Expected dedicated or managed MemPool mode, got {self.mode!r}")
        if self.mode == "managed" and not callable(
            getattr(torch.cuda.memory, "empty_allocator_cache", None)
        ):
            raise RuntimeError(
                "Managed VMM mode requires patched PyTorch with allocator_managed=True "
                "and torch.cuda.memory.empty_allocator_cache"
            )
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
        self._lock = threading.RLock()
        self._pools = []
        if torch.cuda.memory.get_allocator_backend() != "native":
            raise RuntimeError("VMM MemPool requires PyTorch's native CUDA allocator backend")
        import transformer_engine_torch as tex

        if not hasattr(tex, "VMMAllocationTicket") or not hasattr(
            tex.VMMMemPoolBackend, "collect_completed"
        ):
            raise RuntimeError(
                "Rebuild TE with NVTE_WITH_VMM_MEMPOOL=1 for ordinary-PyTorch VMM support"
            )
        with torch.cuda.device(self.device_index):
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("Create VMM MemPool controllers before CUDA Graph capture")
            self._backend = tex.VMMMemPoolBackend(self.device_index, domain, block_bytes)
            self._managed_pool = None
            if self.mode == "managed":
                try:
                    self._managed_pool = torch.cuda.MemPool(
                        self._backend.allocator, allocator_managed=True
                    )
                except TypeError as exc:
                    raise RuntimeError(
                        "Managed VMM mode requires patched PyTorch MemPool(allocator_managed=True)"
                    ) from exc
        _CONTROLLERS.add(self)

    def allocate(self, shape, dtype):
        """Return a normal contiguous tensor whose Storage owns the native allocation."""
        shape = tuple(operator.index(dimension) for dimension in shape)
        if any(dimension < 0 for dimension in shape):
            raise ValueError(f"Negative tensor dimension in {shape}")
        numel = math.prod(shape)
        if numel == 0:
            raise ValueError("Localized allocations must contain at least one element")
        with self._lock, torch.cuda.device(self.device_index):
            if self.mode == "managed":
                with torch.cuda.use_mem_pool(self._managed_pool, self.device_index):
                    return torch.empty(
                        shape, dtype=dtype, device=torch.device("cuda", self.device_index)
                    )
            element_size = torch.empty((), dtype=dtype, device="cpu").element_size()
            self.collect()
            ticket = self._backend.ticket(numel * element_size)
            native_pool = torch.cuda.MemPool(ticket.allocator)
            # Retain the pool even when allocation fails during capture. Native
            # pool destruction/cache drains must happen outside capture.
            self._pools.append([native_pool, ticket, native_pool.id])
            with torch.cuda.use_mem_pool(native_pool, self.device_index):
                backing = torch.empty(
                    ticket.mapped_bytes,
                    dtype=torch.uint8,
                    device=torch.device("cuda", self.device_index),
                )
            ticket.watch(backing)  # C++ weak StorageImpl reference, not a tensor weakref.
            return backing.view(dtype).flatten()[:numel].view(shape).detach()

    def collect(self):
        """Return dead tensors' dedicated native segments to TE's VMM allocator.

        Views keep Storage alive. Native graph references keep pools alive.
        Native teardown may wait for recorded-stream work. During capture this
        method is a no-op; CUDA graph user objects independently pin mappings.
        Managed mode only polls TE retirements; native allocation/drain operations
        deliver completed frees without per-tensor pool collection.
        """
        with self._lock, torch.cuda.device(self.device_index):
            if torch.cuda.is_current_stream_capturing():
                return
            if self.mode == "managed":
                self._backend.collect_completed()
                return
            remaining = []
            for entry in self._pools:
                native_pool, ticket, pool_id = entry
                if native_pool is not None:
                    if ticket.storage_alive() or native_pool.use_count() > 1:
                        remaining.append(entry)
                        continue
                    entry[0] = None
                    del native_pool  # Standard MemPool destructor releases its ownership.
                if not ticket.trim(pool_id):
                    remaining.append(entry)
            self._pools = remaining
            self._backend.collect_completed()

    def contains(self, pointer):
        """Whether a pointer lies in backing still held by this allocator."""
        return self._backend.contains(pointer)

    def stats(self):
        """Physical backing, virtual reservations, and mapping/reuse counters."""
        with self._lock:
            result = self._backend.stats()
            result["dedicated_pools"] = len(self._pools)
            result["managed_pools"] = int(self.mode == "managed")
            return result

    def empty_cache(self):
        """Drain dead pools and cached handles; retain live Storage and graph mappings.

        Managed mode uses PyTorch's guarded drain, which rejects live graphs.

        A graph's user-object release callback can run asynchronously after graph
        reset. A subsequent collect/allocation/drain reclaims its mapping once
        that callback has run. Live graphs are safe even if this is called early.
        """
        with self._lock, torch.cuda.device(self.device_index):
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("Cannot drain VMM MemPools during CUDA Graph capture")
            if self.mode == "managed":
                torch.cuda.memory.empty_allocator_cache(self._managed_pool)
            else:
                self.collect()
                self._backend.empty_cache()
