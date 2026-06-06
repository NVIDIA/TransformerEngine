# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""torch.compile glue for Transformer Engine quantizers.

This module isolates the torch.compile-specific plumbing that turns a
*tensorless* quantizer into a torch.compile **value** opaque type:

  * :func:`register_value_opaque_quantizer` -- attaches the ``__fx_repr__`` used
    by FX codegen and registers the quantizer class with
    ``torch._library.opaque_object``. It is a no-op on PyTorch builds without
    the opaque-object API, so importing Transformer Engine never fails on older
    PyTorch -- only torch.compile specialization on the quantizer is
    unavailable there.
  * :func:`_rebuild_quantizer` -- rebuilds a quantizer constant from its value
    items inside the generated FX graph. The quantizer class is captured
    directly in the FX globals (see :func:`_quantizer_fx_repr`), so no global
    class registry is needed.

The eager value semantics (``__eq__`` / ``__hash__`` / ``_value_key`` /
``_value_fields``) live on the quantizer itself; see
:class:`transformer_engine.pytorch.quantized_tensor.Quantizer`.

See ``torch._library.opaque_object`` Note [Opaque Objects] for the contract a
value-typed opaque object must satisfy (``__eq__`` / ``__hash__`` /
``__fx_repr__``). The ``__fx_repr__`` contract -- ``(repr_string, {name: type})``
where ``repr_string`` references the names in the dict -- is exactly how
PyTorch's own value opaque types (e.g. DTensor placements) reconstruct
themselves, including across the on-disk compile cache.
"""

from __future__ import annotations
from typing import Any, Dict, Tuple

from .constants import DType


def _rebuild_quantizer(cls: type, items: Tuple[Tuple[str, Any], ...]) -> Any:
    """Rebuild a tensorless quantizer of type *cls* from its value items.

    Referenced by the ``__fx_repr__`` emitted for value-opaque quantizers; the
    generated FX code calls this to materialize the quantizer constant. The
    deprecated amax-reduction process group is never part of the value, so a
    reconstructed quantizer always starts with no stored group.
    """
    # Bypass ``__init__`` and restore the value attributes directly: the value
    # items already capture every value-defining field (including derived ones),
    # and the constructors have heterogeneous signatures / side effects.
    obj = cls.__new__(cls)
    field_names = set()
    for name, value in items:
        if name == "dtype":
            value = DType.cast(value)
        object.__setattr__(obj, name, value)
        field_names.add(name)
    # The deprecated amax-reduction process group is excluded from the value;
    # restore it as ``None`` for quantizers that still carry the fallback so
    # attribute access keeps working.
    if "with_amax_reduction" in field_names and not hasattr(obj, "amax_reduction_group"):
        object.__setattr__(obj, "amax_reduction_group", None)
    return obj


def _quantizer_fx_repr(self: Any) -> Tuple[str, Dict[str, Any]]:
    """``__fx_repr__`` for value-opaque quantizers (attached at registration).

    Returns an evaluable expression that rebuilds the quantizer via
    :func:`_rebuild_quantizer`, capturing both the helper and the quantizer
    class itself in the FX globals so codegen can resolve them with no global
    registry and no qualname collisions.
    """
    cls = type(self)
    items = self._value_key()[1]
    return (
        f"_rebuild_quantizer({cls.__name__}, {items!r})",
        {"_rebuild_quantizer": _rebuild_quantizer, cls.__name__: cls},
    )


def register_value_opaque_quantizer(cls: type) -> None:
    """Register a tensorless quantizer class as a torch.compile value opaque type.

    Attaches ``__fx_repr__`` and registers the class with
    ``torch._library.opaque_object``. Safe to call on any PyTorch build: on
    versions without the opaque-object API it only attaches ``__fx_repr__``
    (harmless), so Transformer Engine keeps importing and running in eager mode.

    The quantizer class must already provide value ``__eq__`` / ``__hash__`` and
    a non-``None`` ``_value_fields`` (see
    :class:`transformer_engine.pytorch.quantized_tensor.Quantizer`).
    """
    # ``register_opaque_type`` requires ``__fx_repr__`` to already exist on the
    # class, so attach it before registering.
    if "__fx_repr__" not in cls.__dict__:
        cls.__fx_repr__ = _quantizer_fx_repr

    try:
        from torch._library.opaque_object import (  # pylint: disable=import-outside-toplevel
            register_opaque_type,
            is_opaque_value_type,
        )
    except (ImportError, AttributeError):
        # Older PyTorch without the opaque-object API: eager value semantics
        # still work; torch.compile specialization on the quantizer does not.
        return

    if is_opaque_value_type(cls):
        return

    try:
        register_opaque_type(cls, typ="value")
    except (ImportError, AttributeError, RuntimeError, TypeError):
        # Tolerate partial / experimental opaque-object support.
        pass
