# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""TensorProto: a data-free description of a tensor / quantized tensor."""

from __future__ import annotations
import copy as _copy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch


def _contiguous_stride(shape: Tuple[int, ...]) -> Tuple[int, ...]:
    """Row-major (contiguous) stride for ``shape``."""
    stride: list = []
    acc = 1
    for dim in reversed(shape):
        stride.append(acc)
        acc *= dim
    return tuple(reversed(stride))


@dataclass
class TensorProto:
    """A data-free *prototype* of a tensor or quantized tensor.

    Captures ``shape`` / ``dtype`` and, for quantized tensors, the
    (value-opaque) ``quantizer`` -- enough to rebuild a tensor without holding
    storage. The common abstraction over plain ``torch.Tensor``,
    ``QuantizedTensorStorage`` and ``QuantizedTensor``, used for custom-op fake
    impls and for reassembling a quantized tensor from bare buffers.
    """

    shape: Tuple[int, ...]
    dtype: torch.dtype
    quantizer: Optional[Any] = None
    requires_grad: bool = False
    device: Optional[torch.device] = field(default=None)

    def __post_init__(self) -> None:
        # Own a private copy of the quantizer so usage changes (update_usage)
        # never touch the shared, value-opaque quantizer. The copy inherits the
        # quantizer's current row-/column-wise usage as this proto's layout.
        if self.quantizer is not None:
            q = self.quantizer
            self.quantizer = q.copy() if hasattr(q, "copy") else _copy.copy(q)

    @property
    def is_quantized(self) -> bool:
        """Whether this proto describes a quantized tensor."""
        return self.quantizer is not None

    def update_usage(
        self,
        *,
        rowwise_usage: Optional[bool] = None,
        columnwise_usage: Optional[bool] = None,
    ) -> None:
        """Mirror ``QuantizedTensor.update_usage`` on the proto's buffer layout.

        Applied to the proto's own quantizer copy, so the shared (value-opaque)
        quantizer is never mutated. No-op for plain (non-quantized) protos.
        """
        if self.quantizer is None:
            return
        self.quantizer.set_usage(rowwise=rowwise_usage, columnwise=columnwise_usage)

    def inner_names(self) -> Tuple[str, ...]:
        """Names of the flat tensor buffers backing this proto, in order.

        The real op flattens a quantized output via the storage's
        ``__tensor_flatten__`` -- i.e. ``_FLATTEN_TENSOR_BUFFERS`` order, keeping
        only the present buffers. ``_describe_buffers`` may emit the same buffers
        in a different (per-usage) order (e.g. NVFP4 groups each amax right after
        its scale), so reorder to the canonical flatten order here to keep the
        fake layout aligned with the real one slot-for-slot.
        """
        if self.quantizer is None:
            return ("data",)
        # pylint: disable=protected-access
        described = list(self.quantizer._describe_buffers(tuple(self.shape)).keys())
        storage_cls = self.quantizer._storage_metadata(self.dtype)["cls"]
        flatten_order = [attr for attr, _ in storage_cls._FLATTEN_TENSOR_BUFFERS]
        ordered = [name for name in flatten_order if name in described]
        ordered += [name for name in described if name not in flatten_order]
        return tuple(ordered)

    def create_metadata(self) -> Dict[str, Any]:
        """Data-free ``__tensor_unflatten__`` context describing this tensor."""
        if self.quantizer is None:
            return {
                "is_tensor": True,
                "is_quantized": False,
                "dtype": self.dtype,
                "requires_grad": self.requires_grad,
            }
        return self.quantizer.create_metadata(
            tuple(self.shape), dtype=self.dtype, requires_grad=self.requires_grad
        )

    def create_inner_tensors(self) -> List[torch.Tensor]:
        """Materialize the flat inner buffers (in :meth:`inner_names` order).

        Under ``register_fake`` the ``torch.empty`` calls produce ``FakeTensor``s;
        ``requires_grad`` is left default (managed by ``register_autograd``).
        """
        device = self.device if self.device is not None else torch.device("cuda")
        if self.quantizer is None:
            return [torch.empty(tuple(self.shape), dtype=self.dtype, device=device)]
        inner = self.quantizer.alloc_tensors(tuple(self.shape), device=device)
        return [inner[name] for name in self.inner_names()]

    def create_tensor(self) -> torch.Tensor:
        """Materialize an (uninitialized) tensor matching this proto (traceable).

        Quantized protos reassemble the :meth:`create_inner_tensors` buffers via
        the storage's ``__tensor_unflatten__``.
        """
        if self.quantizer is None:
            device = self.device if self.device is not None else torch.device("cuda")
            return torch.empty(
                tuple(self.shape),
                dtype=self.dtype,
                device=device,
                requires_grad=self.requires_grad,
            )
        from ..quantized_tensor import (  # pylint: disable=import-outside-toplevel
            _STORAGE_REGISTRY,
        )

        shape = tuple(self.shape)
        ctx = self.create_metadata()
        inner = dict(zip(self.inner_names(), self.create_inner_tensors()))
        storage_cls = _STORAGE_REGISTRY[ctx["cls"]]
        return storage_cls.__tensor_unflatten__(inner, ctx, shape, _contiguous_stride(shape))


def to_tensor_proto(tensor: Any) -> TensorProto:
    """Build a :class:`TensorProto` describing ``tensor``.

    Works for plain ``torch.Tensor`` and for ``QuantizedTensorStorage`` /
    ``QuantizedTensor``. A *bare* storage exposes its shape via ``.size()`` and
    its (fake) dtype via ``_dtype`` rather than ``.shape`` / ``.dtype``.
    """
    from ..quantized_tensor import (  # pylint: disable=import-outside-toplevel
        QuantizedTensorStorage,
    )

    requires_grad = bool(getattr(tensor, "requires_grad", False))
    if isinstance(tensor, QuantizedTensorStorage):
        shape = getattr(tensor, "shape", None)
        if shape is None:
            shape = tensor.size()
        dtype = getattr(tensor, "dtype", None)
        if dtype is None:
            dtype = getattr(tensor, "_dtype", None)
        return TensorProto(
            shape=tuple(shape),
            dtype=dtype,
            quantizer=getattr(tensor, "_quantizer", None),
            requires_grad=requires_grad,
            device=tensor.device,
        )
    return TensorProto(
        shape=tuple(tensor.shape),
        dtype=tensor.dtype,
        quantizer=None,
        requires_grad=requires_grad,
        device=tensor.device,
    )
