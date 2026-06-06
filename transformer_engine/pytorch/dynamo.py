# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""torch.compile glue for Transformer Engine quantizers.

This module isolates the torch.compile-specific plumbing that turns a
*tensorless* quantizer into a torch.compile **value** opaque type:

  * :func:`register_value_opaque_quantizer` -- attaches the ``__fx_repr__`` used
    by FX codegen and registers the quantizer class with
    ``torch._library.opaque_object``. It is a no-op (other than populating the
    local registry) on PyTorch builds without the opaque-object API, so
    importing Transformer Engine never fails on older PyTorch -- only
    torch.compile specialization on the quantizer is unavailable there.
  * :func:`_quantizer_from_value_key` -- rebuilds a quantizer constant from its
    value key inside the generated FX graph.

The eager value semantics (``__eq__`` / ``__hash__`` / ``_value_key`` /
``_value_fields``) live on the quantizer itself; see
:class:`transformer_engine.pytorch.quantized_tensor.Quantizer`.

See ``torch._library.opaque_object`` Note [Opaque Objects] for the contract a
value-typed opaque object must satisfy (``__eq__`` / ``__hash__`` /
``__fx_repr__``).
"""

from __future__ import annotations
from typing import Any, Dict, Tuple

from .constants import DType


# Maps a quantizer class qualname to the class object. A value key stores only
# the qualname, so reconstruction looks the class up here. Populated by
# ``register_value_opaque_quantizer`` at import time of each tensor module; this
# avoids importing the tensor modules into this module (which would create an
# import cycle).
_QUANTIZER_VALUE_REGISTRY: Dict[str, type] = {}


def _quantizer_from_value_key(key: Tuple[Any, ...]) -> Any:
    """Rebuild a tensorless quantizer from its value key.

    Referenced by the ``__fx_repr__`` emitted for value-opaque quantizers; the
    generated FX code calls this to materialize the quantizer constant. The
    deprecated amax-reduction process group is never part of the value, so a
    reconstructed quantizer always starts with no stored group.
    """
    qualname, items = key[0], key[1]
    cls = _QUANTIZER_VALUE_REGISTRY[qualname]
    # Bypass ``__init__`` and restore the value attributes directly: the value
    # key already captures every value-defining field (including derived ones),
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
    :func:`_quantizer_from_value_key`, together with the globals needed to
    evaluate it.
    """
    return (
        f"_quantizer_from_value_key({self._value_key()!r})",
        {"_quantizer_from_value_key": _quantizer_from_value_key},
    )


def register_value_opaque_quantizer(cls: type) -> None:
    """Register a tensorless quantizer class as a torch.compile value opaque type.

    Attaches ``__fx_repr__`` and registers the class with
    ``torch._library.opaque_object``. Safe to call on any PyTorch build: on
    versions without the opaque-object API it only records the class in the
    local registry and attaches ``__fx_repr__`` (both harmless), so Transformer
    Engine keeps importing and running in eager mode.

    The quantizer class must already provide value ``__eq__`` / ``__hash__`` and
    a non-``None`` ``_value_fields`` (see
    :class:`transformer_engine.pytorch.quantized_tensor.Quantizer`).
    """
    _QUANTIZER_VALUE_REGISTRY[cls.__qualname__] = cls

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
