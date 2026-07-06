# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Value-opaque quantizers for torch.compile."""

from __future__ import annotations
import enum
from typing import Any, Dict, Tuple, get_type_hints

from ..constants import DType


# Qualnames of the registered quantizer classes. The set holds strings rather
# than the classes themselves so that ``is_value_opaque_quantizer`` can be
# called inside a ``torch.compile``'d function without a graph break: Dynamo
# can evaluate ``type(q).__qualname__ in <set of strings>``, but not set
# membership of a class registered as an opaque type.
_VALUE_OPAQUE_QUALNAMES: set = set()


def is_value_opaque_quantizer(quantizer: Any) -> bool:
    """Whether *quantizer*'s class is registered as a torch.compile value-opaque
    type."""
    return type(quantizer).__qualname__ in _VALUE_OPAQUE_QUALNAMES


def _rebuild_quantizer(cls: type, items: Tuple[Tuple[str, Any], ...]) -> Any:
    """Rebuild a tensorless quantizer of type *cls* from its value items.

    Referenced by the ``__fx_repr__`` emitted for value-opaque quantizers; the
    generated FX code calls this to materialize the quantizer constant.
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
    # The deprecated amax-reduction group is not a value field; initialize it to
    # None so attribute access keeps working on the rebuilt quantizer.
    if "with_amax_reduction" in field_names and not hasattr(obj, "amax_reduction_group"):
        object.__setattr__(obj, "amax_reduction_group", None)
    # Restore non-value derived state that ``__init__`` would normally build but
    # that cannot live in the value key (e.g. NVFP4's ``rht_matrix`` tensor).
    finalize = getattr(obj, "_rebuild_derived_state", None)
    if finalize is not None:
        finalize()
    return obj


def _quantizer_fx_repr(self: Any) -> Tuple[str, Dict[str, Any]]:
    """``__fx_repr__`` for value-opaque quantizers (attached at registration).

    Returns an evaluable expression that rebuilds the quantizer via
    :func:`_rebuild_quantizer`, capturing both the helper and the quantizer
    class itself in the FX globals so codegen can resolve them with no global
    registry and no qualname collisions.

    Raises ``TypeError`` (via :meth:`Quantizer._value_key`) if the quantizer
    stores a process group (e.g. a non-``None`` deprecated
    ``amax_reduction_group``): live distributed state must never be baked into
    the graph as a constant. Pass the reduction group per quantize call instead
    of storing it on the quantizer.
    """
    cls = type(self)
    items = self._value_key()[1]
    return (
        f"_rebuild_quantizer({cls.__name__}, {items!r})",
        {"_rebuild_quantizer": _rebuild_quantizer, cls.__name__: cls},
    )


def register_value_opaque_quantizer(cls: type) -> None:
    """Register a tensorless quantizer class as a torch.compile value opaque type.

    This is the opt-in point for value semantics: it derives the value fields
    from the class annotations and stores them on the class (enabling
    config-based ``__eq__`` / ``__hash__``, see
    :class:`transformer_engine.pytorch.quantized_tensor.Quantizer`), attaches
    ``__fx_repr__`` and registers the class with
    ``torch._library.opaque_object``. Safe to call on any PyTorch build: on
    versions without the opaque-object API the value semantics still apply,
    only the torch.compile specialization is skipped.

    Only plain value types (``int``/``bool``/``float``/``str`` and enums) may
    be annotated: anything else (derived tensors, process groups, containers)
    cannot be hashed into the value key or rebuilt from its repr, so it must
    be left unannotated and rebuilt in ``_rebuild_derived_state`` instead.
    This runs once per class at import time, not in any hot path, so resolving
    the annotation strings to real types is affordable.
    """
    fields = cls._annotated_fields()
    resolved = get_type_hints(cls)
    for name in fields:
        typ = resolved[name]
        if typ not in (int, bool, float, str) and not (
            isinstance(typ, type) and issubclass(typ, enum.Enum)
        ):
            raise TypeError(
                f"{cls.__name__} cannot be a torch.compile value quantizer: "
                f"annotated field {name!r} ({typ!r}) is not a plain value type "
                "(int/bool/float/str/enum). Remove the annotation and rebuild "
                "the field in ``_rebuild_derived_state`` instead."
            )
    cls._value_field_names = tuple(fields)
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

    try:
        if not is_opaque_value_type(cls):
            register_opaque_type(cls, typ="value")
    except (RuntimeError, TypeError):
        # Keep TE importable: neither the opaque-type query nor the registration
        # must crash the import, e.g. on PyTorch versions with only partial /
        # experimental opaque-object support.
        return

    _VALUE_OPAQUE_QUALNAMES.add(cls.__qualname__)
