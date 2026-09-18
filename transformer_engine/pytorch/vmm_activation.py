# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Fixed-address CUDA virtual-memory slots for graph-captured activations."""

from __future__ import annotations

from typing import Sequence

import torch
import transformer_engine_torch as tex


def vmm_enable_trace(enabled: bool, path: str) -> None:
    """Enable thread-safe native VMM trace collection to a JSONL path."""
    tex.vmm_enable_trace(bool(enabled), str(path))


def vmm_initialize_workers() -> None:
    """Start native VMM workers before profiler capture begins."""
    tex.vmm_initialize_workers()


def vmm_profiler_status() -> dict[str, object]:
    """Return profiler callback and native trace state."""
    return dict(tex.vmm_profiler_status())


def vmm_driver_memory_info() -> dict[str, int]:
    """Return free and total bytes reported by the active CUDA driver context."""
    return {key: int(value) for key, value in dict(tex.vmm_driver_memory_info()).items()}


class CUDAActivationVMMAllocation:
    """Own a fixed virtual address and replaceable physical device backing."""

    def __init__(
        self, shape: Sequence[int], stride: Sequence[int], dtype: torch.dtype, device: torch.device
    ) -> None:
        self.shape = tuple(shape)
        self.stride = tuple(stride)
        self.dtype = dtype
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError(f"CUDA VMM allocation requires a CUDA device, got {self.device}")
        if not self.shape or len(self.shape) != len(self.stride):
            raise ValueError(f"invalid shape/stride: {self.shape}/{self.stride}")
        if any(size <= 0 for size in self.shape) or any(value < 0 for value in self.stride):
            raise ValueError(f"unsupported shape/stride: {self.shape}/{self.stride}")
        element_size = torch.empty((), dtype=dtype).element_size()
        maximum_element_offset = sum(
            (size - 1) * value for size, value in zip(self.shape, self.stride)
        )
        self.storage_bytes = (maximum_element_offset + 1) * element_size
        device_index = self.device.index
        if device_index is None:
            device_index = torch.cuda.current_device()
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

    def set_slot_id(self, slot_id: str) -> None:
        """Attach a stable human-readable identity used by worker diagnostics."""
        self._allocation.set_slot_id(str(slot_id))

    @property
    def mapped(self) -> bool:
        """Whether the slot currently has a physical mapping.

        Drains a finished async release first so the flag reflects reality
        once the worker has done its unmap (errors from the worker surface
        here; use wait_for_async_release() to also block for in-flight ones).
        """
        if self._allocation.async_remap_done() and not self._closed:
            self.wait_for_async_remap()
        if self._allocation.async_release_done() and not self._closed:
            self._allocation.wait_for_async_release()
        return bool(self._allocation.info()["mapped"])

    def info(self) -> dict[str, int | bool]:
        return {key: value for key, value in dict(self._allocation.info()).items()}

    def unmap_and_release(self) -> None:
        self._allocation.unmap_and_release()

    def create_and_remap(self) -> None:
        self._allocation.create_and_remap()
        if self.info()["address"] != self._address or self._tensor.data_ptr() != self._address:
            raise RuntimeError("CUDA VMM allocation changed virtual address after remap")

    # ------------------------------------------------------------------
    # Asynchronous release API
    # ------------------------------------------------------------------

    def release_hook_after(self, stream: torch.Stream) -> "VMMReleaseHookContext":
        """Enqueue a stream host-func that releases backing after prior work.

        The unmap/release run on a resident worker thread once `stream` has
        completed everything enqueued before this call (the D2H burst). The
        virtual address reservation is kept; remap with `create_and_remap()`.
        """
        return self._allocation.release_hook_after(_raw_cuda_stream(stream))

    def wait_for_async_release(self) -> None:
        """Block until the in-flight async release completed; surface errors."""
        self._allocation.wait_for_async_release()

    def async_release_done(self) -> bool:
        """Whether a previously enqueued async release already finished."""
        return bool(self._allocation.async_release_done())

    def wait_for_async_remap(self) -> None:
        """Adopt a completed async remap and verify its fixed address."""
        self._allocation.wait_for_async_remap()
        if self.info()["address"] != self._address or self._tensor.data_ptr() != self._address:
            raise RuntimeError("CUDA VMM allocation changed virtual address after async remap")

    def adopt_async_remap(self) -> None:
        """Adopt a worker-completed remap without waiting for it."""
        self._allocation.adopt_async_remap()
        if self.info()["address"] != self._address or self._tensor.data_ptr() != self._address:
            raise RuntimeError("CUDA VMM allocation changed virtual address after async remap")

    def async_remap_done(self) -> bool:
        """Whether a previously enqueued asynchronous remap already finished."""
        return bool(self._allocation.async_remap_done())

    def close(self) -> None:
        if self._closed:
            return
        self._allocation.close()
        self._closed = True


def _raw_cuda_stream(stream: torch.Stream) -> int:
    """Return the raw CUDA stream handle (uintptr_t) for a torch stream."""
    raw = getattr(stream, "cuda_stream", None)
    if raw is None:
        raise ValueError(f"stream {stream!r} does not expose a cuda_stream handle")
    return int(raw)


def release_hooks_after(
    allocations: Sequence["CUDAActivationVMMAllocation"], stream: torch.Stream
) -> "VMMReleaseHookContext":
    """Batch variant: one host func releases backing for many slots at once.

    Enqueue all D2H copies on `stream` first; this adds a single host function
    after them so the DMA burst stays contiguous and the worker receives all
    raw release work in one batch.
    """
    return tex.release_hooks_after([a._allocation for a in allocations], _raw_cuda_stream(stream))


def remap_hooks_after(
    allocations: Sequence["CUDAActivationVMMAllocation"], stream: torch.Stream
) -> "VMMRemapHookContext":
    """Submit one asynchronous fixed-address remap batch on ``stream``."""
    return tex.remap_hooks_after([a._allocation for a in allocations], _raw_cuda_stream(stream))


def remap_and_copy_after(
    allocations: Sequence["CUDAActivationVMMAllocation"],
    host_tensors: Sequence[torch.Tensor],
    stream: torch.Stream,
) -> "VMMRemapHookContext":
    """Asynchronously remap slots and submit their pinned-host H2D copies."""
    return tex.remap_and_copy_after(
        [allocation._allocation for allocation in allocations],
        list(host_tensors),
        _raw_cuda_stream(stream),
    )


def remap_and_copy_slot_after(
    allocation: CUDAActivationVMMAllocation,
    host_tensor: torch.Tensor,
    stream: torch.Stream,
) -> "VMMRemapHookContext":
    """Submit exactly one slot request to the remap server."""
    return tex.remap_and_copy_slot_after(
        allocation._allocation,
        host_tensor,
        _raw_cuda_stream(stream),
    )


def remap_only_slot_after(
    allocation: CUDAActivationVMMAllocation,
    host_tensor: torch.Tensor,
    stream: torch.Stream,
) -> "VMMRemapHookContext":
    """Submit one remap-only slot request; H2D is deferred to launch_slot_h2d."""
    return tex.remap_only_slot_after(
        allocation._allocation,
        host_tensor,
        _raw_cuda_stream(stream),
    )


def launch_remap_slot_h2d(
    context: "VMMRemapHookContext", slot_index: int, stream: torch.Stream
) -> None:
    """Launch one deferred slot's H2D now, waiting for its remap if needed."""
    context.launch_slot_h2d(int(slot_index), _raw_cuda_stream(stream))


def wait_remap_copy_on_stream(context: "VMMRemapHookContext", stream: torch.Stream) -> None:
    """Wait for worker submission, then install the GPU-side H2D dependency."""
    context.wait_on_stream(_raw_cuda_stream(stream))


def enqueue_remap_copy_wait(context: "VMMRemapHookContext", stream: torch.Stream) -> None:
    """Install only the GPU-side H2D dependency without waiting on the worker."""
    context.enqueue_wait_on_stream(_raw_cuda_stream(stream))


def enqueue_remap_slot_waits(context: "VMMRemapHookContext", stream: torch.Stream) -> None:
    """Install one completion wait for each remapped slot."""
    context.enqueue_slot_waits(_raw_cuda_stream(stream))


def wait_remap_slot_on_stream(
    context: "VMMRemapHookContext", slot_index: int, stream: torch.Stream
) -> None:
    """Wait until one slot H2D is submitted, then enqueue its GPU dependency."""
    context.enqueue_slot_wait(int(slot_index), _raw_cuda_stream(stream))


def wait_until_remap_slot_submitted(context: "VMMRemapHookContext", slot_index: int) -> None:
    """Block until one slot's remap and H2D event have been submitted."""
    context.wait_until_slot_submitted(int(slot_index))
