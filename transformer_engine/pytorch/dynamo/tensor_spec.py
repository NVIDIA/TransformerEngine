# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""TensorSpec: a data-free description of a tensor / quantized tensor."""

from __future__ import annotations
import copy as _copy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch._prims_common import make_contiguous_strides_for


@dataclass
class TensorSpec:
    """A data-free description of a tensor or quantized tensor.

    Captures ``shape`` / ``dtype`` and, for quantized tensors, the
    (value-opaque) ``quantizer`` -- enough to rebuild a tensor without holding
    storage. The common abstraction over plain ``torch.Tensor``,
    ``QuantizedTensorStorage`` and ``QuantizedTensor``, used for custom-op fake
    impls and for reassembling a quantized tensor from bare inner tensors.
    """

    shape: Tuple[int, ...]
    dtype: torch.dtype
    quantizer: Optional[Any] = None
    requires_grad: bool = False
    device: Optional[torch.device] = field(default=None)

    def __post_init__(self) -> None:
        # Own a private copy of the quantizer so usage changes (update_usage)
        # never touch the shared, value-opaque quantizer. The copy inherits the
        # quantizer's current row-/column-wise usage as this spec's layout.
        if self.quantizer is not None:
            q = self.quantizer
            self.quantizer = q.copy() if hasattr(q, "copy") else _copy.copy(q)

    @property
    def is_quantized(self) -> bool:
        """Whether this spec describes a quantized tensor."""
        return self.quantizer is not None

    def update_usage(
        self,
        *,
        rowwise_usage: Optional[bool] = None,
        columnwise_usage: Optional[bool] = None,
    ) -> None:
        """Mirror ``QuantizedTensor.update_usage`` on the spec's inner-tensor layout.

        Applied to the spec's own quantizer copy, so the shared (value-opaque)
        quantizer is never mutated. Raises on plain (non-quantized) specs --
        a real plain ``torch.Tensor`` has no ``update_usage`` either.
        """
        if self.quantizer is None:
            raise ValueError("update_usage called on a non-quantized TensorSpec")
        self.quantizer.set_usage(rowwise=rowwise_usage, columnwise=columnwise_usage)

    def inner_names(self) -> Tuple[str, ...]:
        """Names of the flat inner tensors backing this spec, in order.

        The real op flattens a quantized output via the storage's
        ``__tensor_flatten__`` -- i.e. ``_INNER_TENSORS`` order, keeping only the
        present inner tensors. ``inner_tensor_specs`` is contracted to emit them
        in that same order, which keeps the fake layout aligned with the real one
        slot-for-slot; the contract is checked here rather than papered over by
        reordering, so a mismatching quantizer fails loudly.
        """
        if self.quantizer is None:
            return ("data",)
        # pylint: disable=protected-access
        described = tuple(self.quantizer.inner_tensor_specs(tuple(self.shape)))
        storage_cls = self.quantizer.storage_metadata(self.dtype)["cls"]
        flatten_order = tuple(attr for attr, _ in storage_cls._INNER_TENSORS)
        expected = tuple(name for name in flatten_order if name in described)
        if described != expected:
            raise RuntimeError(
                f"{type(self.quantizer).__name__}.inner_tensor_specs returned {described}, "
                f"which does not follow {storage_cls.__name__}._INNER_TENSORS order "
                f"{flatten_order} (expected {expected}); the fake layout would not match "
                "the real one slot-for-slot."
            )
        return described

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
        """Materialize the flat inner tensors (in :meth:`inner_names` order).

        Under ``register_fake`` the ``torch.empty`` calls produce ``FakeTensor``s;
        ``requires_grad`` is left default (managed by ``register_autograd``).
        """
        device = self.device if self.device is not None else torch.device("cuda")
        if self.quantizer is None:
            return [torch.empty(tuple(self.shape), dtype=self.dtype, device=device)]
        inner = self.quantizer.alloc_tensors(tuple(self.shape), device=device)
        return [inner[name] for name in self.inner_names()]

    def assemble(self, inner_tensors: List[torch.Tensor]) -> torch.Tensor:
        """Rebuild the tensor from ready-made ``inner_tensors`` (in :meth:`inner_names`
        order). Shared by :meth:`create_tensor` (fresh ones) and the custom-op
        boundary (inner tensors arriving from an op's flat ``Tensor[]`` payload).

        Non-quantized specs are the single inner tensor as-is; quantized specs
        are reassembled into the storage/wrapper via ``__tensor_unflatten__``.
        """
        if self.quantizer is None:
            return inner_tensors[0]
        shape = tuple(self.shape)
        ctx = self.create_metadata()
        inner = dict(zip(self.inner_names(), inner_tensors))
        storage_cls = ctx["cls"]
        return storage_cls.__tensor_unflatten__(
            inner, ctx, shape, make_contiguous_strides_for(shape)
        )

    def create_tensor(self) -> torch.Tensor:
        """Materialize an (uninitialized) tensor matching this spec (traceable).

        Quantized specs reassemble freshly-allocated :meth:`create_inner_tensors`
        inner tensors via :meth:`assemble`.
        """
        if self.quantizer is None:
            device = self.device if self.device is not None else torch.device("cuda")
            return torch.empty(
                tuple(self.shape),
                dtype=self.dtype,
                device=device,
                requires_grad=self.requires_grad,
            )
        return self.assemble(self.create_inner_tensors())


def to_tensor_spec(tensor: Any) -> TensorSpec:
    """Build a :class:`TensorSpec` describing ``tensor``.

    Works for plain ``torch.Tensor`` and for ``QuantizedTensorStorage`` /
    ``QuantizedTensor``. A *bare* storage exposes its (fake) dtype via
    ``_dtype`` rather than ``.dtype``.
    """
    requires_grad = bool(getattr(tensor, "requires_grad", False))
    dtype = getattr(tensor, "dtype", None)
    if dtype is None:
        dtype = getattr(tensor, "_dtype", None)
    return TensorSpec(
        shape=tuple(tensor.shape),
        dtype=dtype,
        quantizer=getattr(tensor, "_quantizer", None),
        requires_grad=requires_grad,
        device=tensor.device,
    )
