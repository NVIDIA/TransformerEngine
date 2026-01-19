# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Opaque kwargs container for torch.compile compatibility."""

from __future__ import annotations
from typing import Any

from torch._library.opaque_object import register_opaque_type


def opaque_kwargs_from_dicts(kwargs_list: list[dict[str, Any]]) -> "OpaqueKwargs":
    """Create OpaqueKwargs from a list of dictionaries.
    
    This is a module-level function instead of classmethod to be compatible
    with torch.compile (classmethods are not supported on opaque types).
    
    Args:
        kwargs_list: List of kwargs dicts, one per operation
        
    Returns:
        OpaqueKwargs instance
    """
    return OpaqueKwargs(tuple(tuple(sorted(d.items())) for d in kwargs_list))


class OpaqueKwargs:
    """Immutable container for per-op keyword arguments.
    
    Value type because kwargs are constant within a compiled graph -
    changes trigger recompilation via __eq__ guard.
    
    The internal representation uses nested tuples for hashability:
    - Outer tuple: one element per operation
    - Inner tuple: sorted (key, value) pairs for that operation's kwargs
    """
    
    __slots__ = ("_data", "__weakref__")
    
    def __init__(self, data: tuple[tuple[tuple[str, Any], ...], ...]) -> None:
        self._data = data
    
    def to_dicts(self) -> list[dict[str, Any]]:
        """Convert back to list of dictionaries.
        
        Returns:
            List of kwargs dicts, one per operation
        """
        return [dict(items) for items in self._data]
    
    def __len__(self) -> int:
        """Return number of operations."""
        return len(self._data)
    
    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Get kwargs dict for operation at index."""
        return dict(self._data[idx])
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, OpaqueKwargs):
            return NotImplemented
        return self._data == other._data
    
    def __hash__(self) -> int:
        return hash(self._data)
    
    def __fx_repr__(self) -> tuple[str, dict[str, type]]:
        """FX-evaluable representation for graph codegen."""
        return f"OpaqueKwargs({self._data!r})", {"OpaqueKwargs": OpaqueKwargs}
    
    def __repr__(self) -> str:
        return f"OpaqueKwargs({self._data!r})"


# Register as value type (immutable, can be baked into graph)
register_opaque_type(OpaqueKwargs, typ="value")
