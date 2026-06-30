# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Value-opaque quantizers for torch.compile."""

from __future__ import annotations
from typing import Any, Dict, Tuple

from ..constants import DType


# Registration marks the class with this attribute rather than recording it in a
# module-level set. It looks odd but is a deliberate workaround: the check must
# stay traceable when it runs inside a torch.compile graph -- Dynamo can bake a
# ``getattr`` on the opaque quantizer into a constant, but cannot evaluate
# ``type(q) in some_set`` (no equality/hash rules for the opaque class object),
# which would graph-break under ``fullgraph=True``.
_VALUE_OPAQUE_FLAG = "_te_compile_value_opaque"


def is_value_opaque_quantizer(quantizer: Any) -> bool:
    """Whether *quantizer*'s class is registered as a torch.compile value-opaque
    type."""
    return getattr(quantizer, _VALUE_OPAQUE_FLAG, False)


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

    try:
        if not is_opaque_value_type(cls):
            register_opaque_type(cls, typ="value")
    except (RuntimeError, TypeError):
        # Keep TE importable: neither the opaque-type query nor the registration
        # must crash the import, e.g. on PyTorch versions with only partial /
        # experimental opaque-object support.
        return

    setattr(cls, _VALUE_OPAQUE_FLAG, True)
