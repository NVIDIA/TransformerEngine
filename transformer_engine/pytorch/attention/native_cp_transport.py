# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""NCCL Device API transport for context-parallel rings.

Building with ``NVTE_WITH_NCCL_DEVICE_CP=1`` requires NCCL 2.29.7 or newer
and matching compile/runtime versions. Process-wide GIN settings are left
to the application; this transport does not change NCCL environment variables.
"""

import math
import weakref
from typing import Iterable, Optional

import torch

import transformer_engine_torch as tex

_group_transports = weakref.WeakKeyDictionary()


class _Work:
    """Stream dependency compatible with ProcessGroup work handles."""

    def __init__(self, handle: int, channel: int) -> None:
        self.handle = handle
        self.channel = channel

    def wait(self) -> bool:
        """Wait until the native operation has completed."""
        if self.handle:
            tex.cp_native_transport_wait(self.handle, self.channel)
            self.handle = 0
        return True


class NativeCPTransport:
    """One symmetric arena attached to a borrowed parent NCCL communicator."""

    def __init__(self, parent_group, payload_bytes: int) -> None:
        if not hasattr(tex, "cp_native_transport_create"):
            raise RuntimeError("Transformer Engine was not built with native CP transport")
        torch.distributed.barrier(group=parent_group, device_ids=[torch.cuda.current_device()])
        backend = parent_group._get_backend(torch.device("cuda"))
        if not hasattr(backend, "_comm_ptr"):
            raise RuntimeError("ProcessGroupNCCL does not expose _comm_ptr()")

        self._parent = weakref.ref(parent_group)
        self._parent_rank = {
            global_rank: rank
            for rank, global_rank in enumerate(
                torch.distributed.get_process_group_ranks(parent_group)
            )
        }
        self.handle, self.arena = tex.cp_native_transport_create(
            int(backend._comm_ptr()), int(payload_bytes)
        )
        self.handle = int(self.handle)

    @property
    def payload_bytes(self) -> int:
        """Return the usable size of the symmetric arena in bytes."""
        return 0 if self.arena is None else self.arena.numel()

    def _view(self, offset: int, shape: Iterable[int], dtype: torch.dtype) -> torch.Tensor:
        shape = tuple(int(dim) for dim in shape)
        size = math.prod(shape) * torch.empty((), dtype=dtype).element_size()
        if offset + size > self.payload_bytes:
            raise RuntimeError(
                f"Native CP arena needs {offset + size} bytes, has {self.payload_bytes}"
            )
        return self.arena.narrow(0, offset, size).view(dtype).view(shape)

    def attention_buffer_pair(self, shape: Iterable[int], dtype: torch.dtype):
        """Return two contiguous ``[KV, dKV]`` work buffers."""
        shape = tuple(shape)
        pair_bytes = 2 * math.prod(shape) * torch.empty((), dtype=dtype).element_size()
        return self._view(0, (2, *shape), dtype), self._view(
            (pair_bytes + 255) // 256 * 256, (2, *shape), dtype
        )

    def send_recv(
        self,
        send_tensor: torch.Tensor,
        send_global_rank: Optional[int],
        recv_tensor: torch.Tensor,
        recv_global_rank: Optional[int],
        channel: int = 0,
    ) -> _Work:
        """Launch an arena exchange; a ``None`` peer disables that direction."""
        try:
            send_peer = -1 if send_global_rank is None else self._parent_rank[int(send_global_rank)]
            recv_peer = -1 if recv_global_rank is None else self._parent_rank[int(recv_global_rank)]
        except KeyError as error:
            raise ValueError(f"Peer {error.args[0]} is outside the parent group") from error
        channel = tex.cp_native_transport_send_recv(
            self.handle, send_tensor, recv_tensor, send_peer, recv_peer, int(channel)
        )
        return _Work(self.handle, int(channel))

    def exchange(
        self,
        send_tensor: torch.Tensor,
        send_global_rank: Optional[int],
        recv_tensor: torch.Tensor,
        recv_global_rank: Optional[int],
        channel: int = 3,
    ) -> None:
        """Exchange small halos using the existing arena, ordered on the caller stream.

        Input and output may be ordinary, noncontiguous tensors. Calls must be
        serialized with other arena users, just like ``all_reduce``. No additional
        persistent payload storage is allocated. A missing receive leaves its
        output unchanged, allowing callers to retain physical-boundary fills.
        """
        if send_tensor.shape != recv_tensor.shape or send_tensor.dtype != recv_tensor.dtype:
            raise ValueError("Native CP halo tensors must have matching shapes and dtypes")
        if send_tensor.device != self.arena.device or recv_tensor.device != self.arena.device:
            raise ValueError("Native CP halo tensors must be on the arena device")
        if send_tensor.numel() == 0 or (send_global_rank is None and recv_global_rank is None):
            return
        size = send_tensor.nbytes
        send = self._view(0, send_tensor.shape, send_tensor.dtype)
        recv = self._view((size + 255) // 256 * 256, recv_tensor.shape, recv_tensor.dtype)
        if send_global_rank is not None:
            send.copy_(send_tensor)
        self.send_recv(send, send_global_rank, recv, recv_global_rank, channel).wait()
        if recv_global_rank is not None:
            recv_tensor.copy_(recv)

    def all_reduce(self, tensor: torch.Tensor, group, channel: int = 2) -> torch.Tensor:
        """Ring sum over a dynamic-CP subgroup."""
        ranks = (
            group.ranks
            if hasattr(group, "ranks")
            else torch.distributed.get_process_group_ranks(group)
        )
        result = tensor.contiguous().clone()
        if len(ranks) == 1:
            return result

        size = tensor.nbytes
        send = self._view(0, tensor.shape, tensor.dtype)
        recv = self._view((size + 255) // 256 * 256, tensor.shape, tensor.dtype)
        send.copy_(tensor)
        rank = group.rank()
        dst, src = ranks[(rank + 1) % len(ranks)], ranks[(rank - 1) % len(ranks)]
        for _ in range(len(ranks) - 1):
            self.send_recv(send, dst, recv, src, channel).wait()
            result.add_(recv)
            send, recv = recv, send
        return result

    def destroy(self) -> None:
        """Collectively release the native transport and its symmetric arena."""
        if not self.handle:
            return
        parent = self._parent()
        if parent is None:
            raise RuntimeError("Parent ProcessGroup was released before its native transport")
        torch.distributed.barrier(group=parent, device_ids=[torch.cuda.current_device()])
        tex.cp_native_transport_destroy(self.handle)
        self.handle, self.arena = 0, None
        torch.distributed.barrier(group=parent, device_ids=[torch.cuda.current_device()])


def initialize_native_cp_transport(parent_group, payload_bytes: int) -> NativeCPTransport:
    """Collectively initialize one transport per parent ProcessGroup."""
    transport = get_native_cp_transport(parent_group)
    if transport is None:
        transport = NativeCPTransport(parent_group, payload_bytes)
        _group_transports[parent_group] = transport
    elif transport.payload_bytes < payload_bytes:
        raise RuntimeError(
            f"Existing native CP arena has {transport.payload_bytes} bytes; "
            f"requested {payload_bytes}"
        )
    return transport


def set_native_cp_parent_group(cp_group, parent_group) -> None:
    """Route a dynamic-CP subgroup through its parent's native transport."""
    transport = get_native_cp_transport(parent_group)
    if transport is None:
        raise RuntimeError("Native CP parent transport is not initialized")
    _group_transports[cp_group] = transport


def get_native_cp_transport(group) -> Optional[NativeCPTransport]:
    """Return the live native transport mapped to ``group``, if any."""
    transport = _group_transports.get(group)
    return transport if transport is not None and transport.handle else None


def destroy_native_cp_transport(parent_group) -> None:
    """Destroy the transport mapped to ``parent_group`` and remove its aliases."""
    transport = _group_transports.get(parent_group)
    if transport is not None:
        transport.destroy()
        for group, mapped in list(_group_transports.items()):
            if mapped is transport:
                del _group_transports[group]
