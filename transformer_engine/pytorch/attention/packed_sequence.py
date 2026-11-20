# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Host metadata and scoped scratch for packed attention."""
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from numbers import Integral
from threading import RLock
from typing import Any, Iterator, Optional, Sequence
import weakref

import torch

_PREFIXES = {}


def register_cu_seqlens(tensor: torch.Tensor, offsets: Sequence[int]) -> None:
    """Associate a prefix tensor with the host integers used to create it.

    This is a producer contract: values are not read back from the tensor.
    Call after its final write. Mutation, resizing, and tensor destruction
    invalidate the registration. No model, head count, or length is assumed.
    """
    if torch.is_tensor(offsets) or any(not isinstance(n, Integral) for n in offsets):
        raise TypeError("Offsets must be a sequence of host integers.")
    values = tuple(int(n) for n in offsets)
    if len(values) < 1 or values[0] != 0 or any(b < a for a, b in zip(values, values[1:])):
        raise ValueError("Offsets must start at zero and be nondecreasing.")
    if (
        tensor.dtype != torch.int32
        or tensor.ndim != 1
        or not tensor.is_contiguous()
        or tensor.numel() != len(values)
        or values[-1] > torch.iinfo(torch.int32).max
    ):
        raise ValueError("Expected a matching contiguous int32 prefix tensor.")
    if tensor.is_inference():
        return
    key = (tensor.device, tensor.data_ptr())

    def expired(ref):
        if _PREFIXES.get(key, (None,))[0] is ref:
            _PREFIXES.pop(key, None)

    _PREFIXES[key] = (weakref.ref(tensor, expired), tensor._version, values)


def _get_cu_seqlens(tensor: Optional[torch.Tensor]) -> Optional[tuple[int, ...]]:
    if tensor is None or tensor.dtype != torch.int32 or tensor.ndim != 1:
        return None
    if not tensor.is_contiguous() or tensor.is_inference():
        return None
    entry = _PREFIXES.get((tensor.device, tensor.data_ptr()))
    if entry is None:
        return None
    owner, version, values = entry
    owner = owner()
    if (
        owner is not None
        and owner._version == version == tensor._version
        and owner.data_ptr() == tensor.data_ptr()
        and owner.numel() == tensor.numel() == len(values)
    ):
        return values
    return None


@dataclass
class _AttentionBackendWorkspace:
    buffers: dict[Any, Any] = field(default_factory=dict)
    lock: Any = field(default_factory=RLock)
    active: bool = True


_WORKSPACE: ContextVar[Optional[_AttentionBackendWorkspace]] = ContextVar(
    "te_attention_backend_workspace", default=None
)


def _get_attention_backend_workspace() -> Optional[_AttentionBackendWorkspace]:
    return _WORKSPACE.get()


@contextmanager
def attention_backend_workspace() -> Iterator[None]:
    """Reuse optional attention scratch until this scope exits.

    Enclose forward and backward. Nested scopes share the outer owner.
    Scratch references are released on success or exception; cached compiled
    plans remain reusable. This does not enable any optional backend.
    """
    if _WORKSPACE.get() is not None:
        yield
        return
    owner = _AttentionBackendWorkspace()
    token = _WORKSPACE.set(owner)
    try:
        yield
    finally:
        with owner.lock:
            owner.active = False
            owner.buffers.clear()
        _WORKSPACE.reset(token)
