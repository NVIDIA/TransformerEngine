# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Tensor metadata classes for torch.compile compatibility."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any

import torch
from torch._library.opaque_object import register_opaque_type


@dataclass(frozen=True)
class TensorInfo:
    """Lightweight tensor descriptor for pseudo_forward - no actual tensor data.

    This class carries tensor metadata (shape, dtype, requires_grad) without
    storing actual tensor data. Used for:
    1. Compile-time shape inference (register_fake)
    2. Backward ctx reconstruction (avoids storing ctx in opaque container)
    """

    shape: tuple[int, ...]
    dtype: torch.dtype
    requires_grad: bool = False
    extra: tuple[tuple[str, Any], ...] = ()  # Hashable op-specific metadata

    @classmethod
    def from_tensor(cls, t: torch.Tensor, **extra) -> "TensorInfo":
        """Create TensorInfo from a tensor with optional extra metadata."""
        return cls(tuple(t.shape), t.dtype, t.requires_grad, tuple(sorted(extra.items())))

    def __hash__(self) -> int:
        return hash((self.shape, self.dtype, self.requires_grad, self.extra))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TensorInfo):
            return NotImplemented
        return (
            self.shape == other.shape
            and self.dtype == other.dtype
            and self.requires_grad == other.requires_grad
            and self.extra == other.extra
        )

    def __fx_repr__(self) -> tuple[str, dict[str, type]]:
        """FX-evaluable representation for graph codegen."""
        return (
            (
                f"TensorInfo({self.shape!r}, torch.{self.dtype}, {self.requires_grad!r},"
                f" {self.extra!r})"
            ),
            {"TensorInfo": TensorInfo, "torch": torch},
        )


@dataclass
class PseudoForwardResult:
    """Result of pseudo_forward - contains ctx info and output shapes.

    This is NOT registered as an opaque type because it's only used
    internally during compile-time shape inference (via USE_REAL member)
    and backward ctx reconstruction.
    """

    output_info: TensorInfo  # shape/dtype of forward output
    tensors_to_save_info: list[TensorInfo] = field(
        default_factory=list
    )  # shapes of tensors saved for backward
    extra_outputs_info: list[TensorInfo] = field(
        default_factory=list
    )  # shapes of extra outputs (e.g., MakeExtraOutput)
    ctx_data: dict[str, Any] = field(default_factory=dict)  # non-tensor ctx attributes for backward

    # Source of each tensor in tensors_to_save:
    # -1 = new tensor (returned from forward, not an input alias)
    #  0 = input x
    #  1..num_params = params[i-1]
    #  num_params+1..num_params+num_extra_inputs = extra_inputs[i-num_params-1]
    tensor_sources: list[int] = field(default_factory=list)


# Register TensorInfo as value type (immutable, can be baked into graph)
register_opaque_type(TensorInfo, typ="value")


class TensorInfoList:
    """Container for list of TensorInfo to work with torch.compile custom ops.

    PyTorch's @torch.library.custom_op does not support list[OpaqueType]
    in function signatures. This container wraps the list so we can pass
    it through custom ops.

    Value type because TensorInfo is immutable and the list contents are constant.
    """

    __slots__ = ("_items", "__weakref__")

    def __init__(self, items: list[TensorInfo]) -> None:
        # Store as tuple for immutability and hashability
        self._items = tuple(items)

    @property
    def items(self) -> list[TensorInfo]:
        """Get the contained list of TensorInfo."""
        return list(self._items)

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int) -> TensorInfo:
        return self._items[idx]

    def __iter__(self):
        return iter(self._items)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TensorInfoList):
            return NotImplemented
        return self._items == other._items

    def __hash__(self) -> int:
        return hash(self._items)

    def __fx_repr__(self) -> tuple[str, dict[str, type]]:
        """FX-evaluable representation for graph codegen."""
        items_repr = ", ".join(
            f"TensorInfo({info.shape!r}, torch.{info.dtype}, {info.requires_grad!r},"
            f" {info.extra!r})"
            for info in self._items
        )
        return (
            f"TensorInfoList([{items_repr}])",
            {"TensorInfoList": TensorInfoList, "TensorInfo": TensorInfo, "torch": torch},
        )

    def __repr__(self) -> str:
        return f"TensorInfoList({list(self._items)!r})"


# Register TensorInfoList as value type (immutable)
register_opaque_type(TensorInfoList, typ="value")
