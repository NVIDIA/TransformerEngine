# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""torch.compile (Dynamo) integration for TransformerEngine modules."""
from __future__ import annotations

import dataclasses
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)

import torch


__all__ = [
    "ArgObject",
    "OpaqueSimpleMetadata",
    "_te_register_custom_op",
]


# Sentinel for ``None`` entries inside the op's flat ``Tensor[]`` return.
# Used by :func:`_te_register_custom_op` to support ``None`` outputs (e.g.
# an FP8 weight workspace returned only on the cache-miss path) on a
# non-nullable schema -- ``Tensor?[]`` returns are not picked up by
# ``torch.library.register_autograd``, so the registered backward never
# attaches a ``grad_fn`` to the op's outputs.
_NONE_SENTINEL_DTYPE = torch.uint8


def _encode_none(t: Optional[torch.Tensor]) -> torch.Tensor:
    """Replace ``None`` with a 0-element uint8 sentinel tensor."""
    if t is None:
        return torch.empty(0, dtype=_NONE_SENTINEL_DTYPE)
    return t


def _decode_none(t: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    """Inverse of :func:`_encode_none`."""
    if t is None:
        return None
    if t.numel() == 0 and t.dtype == _NONE_SENTINEL_DTYPE:
        return None
    return t


# --------------------------------------------------------------------------- #
# OpaqueSimpleMetadata
# --------------------------------------------------------------------------- #

class OpaqueSimpleMetadata:
    """Opaque value-type bundle of simple Python values.

    Wraps a ``{name: value}`` dict so that many small non-Tensor arguments
    of a TE custom op can be passed as a single op input. Registered as a
    torch.compile *value* opaque type, meaning Dynamo specializes the
    traced graph on the bundle's contents: ``__eq__`` installs a guard,
    and any change to a wrapped value triggers a recompile.

    Allowed value types: primitives in :attr:`PRIMITIVE_TYPES`,
    :class:`enum.Enum`, :class:`torch.Size`, plus arbitrarily nested
    tuples/lists thereof.
    """

    # Primitive Python types we are willing to bundle into a single op
    # input. The bundle is registered as a torch.compile *value* opaque
    # type, so its contents must be hashable, comparable for equality,
    # and round-trippable through ``__fx_repr__``.
    PRIMITIVE_TYPES: Tuple[type, ...] = (
        type(None),
        bool,
        int,
        float,
        str,
        torch.dtype,
        torch.device,
    )

    @classmethod
    def _is_opaque_value(cls, value: Any) -> bool:
        """Whether ``value``'s class is registered as a value-opaque type."""
        try:
            from torch._library.opaque_object import is_opaque_value_type
        except Exception:  # pragma: no cover - older torch
            return False
        return is_opaque_value_type(type(value))

    @classmethod
    def is_simple_value(cls, value: Any) -> bool:
        """Whether ``value`` is allowed inside an instance.

        Accepts simple primitives (see :attr:`PRIMITIVE_TYPES`),
        :class:`enum.Enum`, :class:`torch.Size`, instances of any class
        registered as a torch.compile *value*-opaque type (the latter
        already supplies ``__eq__`` / ``__hash__`` / ``__fx_repr__`` as
        a registration prerequisite), and arbitrarily nested
        tuples / lists thereof.
        """
        if isinstance(value, cls.PRIMITIVE_TYPES):
            return True
        if isinstance(value, Enum):
            return True
        if isinstance(value, torch.Size):
            return True
        if cls._is_opaque_value(value):
            return True
        if isinstance(value, (list, tuple)):
            return all(cls.is_simple_value(v) for v in value)
        return False

    @classmethod
    def _to_hashable(cls, value: Any) -> Any:
        """Convert a simple value into something hashable (lists -> tuples)."""
        if isinstance(value, (list, tuple, torch.Size)):
            return tuple(cls._to_hashable(v) for v in value)
        # Opaque-value instances already supply ``__hash__`` (required
        # by registration) so they can stay as-is.
        return value

    @classmethod
    def _fmt_simple(cls, value: Any) -> str:
        """Repr for a simple value, evaluable in a context with ``torch`` globals."""
        if isinstance(value, torch.dtype):
            return f"__import__('torch').{str(value).split('.')[-1]}"
        if isinstance(value, torch.device):
            return f"__import__('torch').device({str(value)!r})"
        if isinstance(value, torch.Size):
            return f"__import__('torch').Size({list(value)!r})"
        if isinstance(value, Enum):
            return f"{type(value).__name__}.{value.name}"
        if isinstance(value, list):
            return "[" + ", ".join(cls._fmt_simple(v) for v in value) + "]"
        if isinstance(value, tuple):
            body = ", ".join(cls._fmt_simple(v) for v in value)
            return f"({body},)" if len(value) == 1 else f"({body})"
        if cls._is_opaque_value(value):
            # Opaque-value types declare their FX reconstruction via
            # ``__fx_repr__``; just splice their expression in here.
            return value.__fx_repr__()[0]
        return repr(value)

    def __init__(
        self,
        data: Optional[Dict[str, Any]] = None,
        /,
        **kwargs: Any,
    ) -> None:
        merged: Dict[str, Any] = dict(data) if data else {}
        merged.update(kwargs)
        cls = type(self)
        for k, v in merged.items():
            if not cls.is_simple_value(v):
                raise TypeError(
                    f"OpaqueSimpleMetadata field '{k}' has unsupported "
                    f"type {type(v).__name__}; only simple primitives "
                    f"({', '.join(t.__name__ for t in cls.PRIMITIVE_TYPES)}, "
                    f"Enum, torch.Size, registered torch.compile value-"
                    f"opaque types) and tuples/lists thereof are allowed."
                )
        self._data: Dict[str, Any] = merged
        self._frozen: Tuple[Tuple[str, Any], ...] = tuple(
            (k, cls._to_hashable(v)) for k, v in sorted(merged.items())
        )

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __getattr__(self, name: str) -> Any:
        # Only called when normal attribute lookup fails, so ``_data`` /
        # ``_frozen`` won't recurse here once set in ``__init__``.
        try:
            return self._data[name]
        except KeyError as e:
            raise AttributeError(name) from e

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def keys(self) -> List[str]:
        return list(self._data.keys())

    def values(self) -> List[Any]:
        return list(self._data.values())

    def items(self) -> List[Tuple[str, Any]]:
        return list(self._data.items())

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def as_dict(self) -> Dict[str, Any]:
        return dict(self._data)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, OpaqueSimpleMetadata):
            return NotImplemented
        return self._frozen == other._frozen

    def __hash__(self) -> int:
        return hash(self._frozen)

    def __fx_repr__(self) -> Tuple[str, Dict[str, Any]]:
        cls = type(self)
        items = ", ".join(
            f"{k!r}: {cls._fmt_simple(v)}" for k, v in self._data.items()
        )
        # Collect every type referenced by a nested opaque-value's
        # ``__fx_repr__`` so the FX codegen can resolve those names.
        globals_: Dict[str, Any] = {
            "OpaqueSimpleMetadata": OpaqueSimpleMetadata,
        }

        def _collect(value: Any) -> None:
            if isinstance(value, (list, tuple)):
                for v in value:
                    _collect(v)
                return
            # Skip plain Python / torch primitives up-front: they're
            # rendered as literals by ``_fmt_simple`` and need no
            # globals entry.
            if isinstance(value, cls.PRIMITIVE_TYPES):
                return
            if isinstance(value, torch.Size):
                return
            if isinstance(value, Enum):
                # ``_fmt_simple`` emits ``EnumName.MEMBER``; the Enum
                # class must be in scope when the source string is
                # later ``exec``d (e.g. by ``GraphModule.print_readable``
                # or by Inductor's runtime wrapper).
                t = type(value)
                globals_[t.__name__] = t
                return
            if cls._is_opaque_value(value):
                _, extra = value.__fx_repr__()
                globals_.update(extra)

        for v in self._data.values():
            _collect(v)
        return (f"OpaqueSimpleMetadata({{{items}}})", globals_)

    def __repr__(self) -> str:
        # ``__repr__`` is on hot diagnostic paths (Inductor error
        # formatters, FX node printers, ...) and must never raise:
        # treating any embedded value's ``repr`` failure as a soft
        # placeholder keeps those error reporters from masking the
        # actual root-cause exception with a crash inside our repr.
        parts: List[str] = []
        for k, v in self._data.items():
            try:
                v_repr = repr(v)
            except Exception as e:  # pylint: disable=broad-except
                v_repr = f"<{type(v).__name__}: repr failed: {e!s}>"
            parts.append(f"{k!r}: {v_repr}")
        return f"OpaqueSimpleMetadata({{{', '.join(parts)}}})"


# Register OpaqueSimpleMetadata as a torch.compile value-opaque type, and
# resolve the schema name of ``torch.distributed.ProcessGroup`` (registered
# upstream as a *reference* opaque type via
# ``torch.distributed.device_mesh._register_distributed_opaque_types``).
# Both are done at module import so that any TE op declared via
# ``_te_register_custom_op`` can immediately reference them in its schema.
# Older PyTorch versions without these APIs are tolerated: the eager path
# keeps working, only torch.compile tracing of TE custom ops is unavailable.
try:
    from torch._library.opaque_object import (
        get_opaque_type_name,
        register_opaque_type,
    )

    register_opaque_type(OpaqueSimpleMetadata, typ="value")
    _OPAQUE_SIMPLE_META_TYPE_NAME: Optional[str] = get_opaque_type_name(
        OpaqueSimpleMetadata
    )

    _PROCESS_GROUP_TYPE_NAME: Optional[str] = None
    try:
        from torch.distributed import ProcessGroup
        from torch.distributed.device_mesh import (
            _register_distributed_opaque_types,
        )

        _register_distributed_opaque_types()
        _PROCESS_GROUP_TYPE_NAME = get_opaque_type_name(ProcessGroup)
    except Exception:  # pragma: no cover - distributed not built / disabled
        _PROCESS_GROUP_TYPE_NAME = None
except Exception:  # pragma: no cover - older torch without opaque_object
    _OPAQUE_SIMPLE_META_TYPE_NAME = None
    _PROCESS_GROUP_TYPE_NAME = None


# --------------------------------------------------------------------------- #
# Field buckets
# --------------------------------------------------------------------------- #

# Each dataclass field of an :class:`ArgObject` is mapped to exactly one
# bucket. A bucket owns the full per-field "vocabulary" -- which schema
# slots it emits, how its packed value(s) are produced from the dataclass
# instance, and how the unpacked value is re-injected into the
# reconstructed instance. ``ArgObject`` then becomes three trivial loops
# over a list of buckets, instead of three parallel branch ladders.
#
# Five bucket kinds are used:
#
# * :class:`_TensorBucket` -- :class:`torch.Tensor` /
#   :class:`Optional[torch.Tensor] <typing.Optional>` -> one ``Tensor`` /
#   ``Tensor?`` slot.
# * :class:`_TensorListBucket` -- ``List[torch.Tensor]`` /
#   ``Tuple[torch.Tensor, ...]`` -> one ``Tensor[]`` slot. Used for
#   variable-length tensor sequences such as ``ctx.saved_tensors``.
# * :class:`_ProcessGroupBucket` -- :class:`torch.distributed.ProcessGroup`
#   (already registered upstream as a value-opaque type) -> one direct
#   slot.
# * :class:`_FlattenableBucket` -- a field whose type implements the
#   ``_flatten`` / ``_unflatten`` protocol (today: any
#   :class:`Quantizer` or :class:`Recipe` subclass) -> three slots
#   ``<name>__fmeta`` / ``<name>__fpg`` / ``<name>__ftensors``. Bases
#   are discovered via :func:`_flattenable_bases`, lazily imported to
#   avoid an import cycle.
# * :class:`_SimpleBundleBucket` -- aggregator over **all** simple-typed
#   fields of the dataclass; emits a single ``_simple_meta`` slot
#   carrying an :class:`OpaqueSimpleMetadata` bundle.
# * :class:`_UnknownBucket` -- a field whose annotation matches none of
#   the above. Emits no schema slot; pack raises if the field holds a
#   non-``None`` value, unpack restores it as ``None``.


def _strip_optional(annot: Any) -> Tuple[Any, bool]:
    """If ``annot`` is ``Optional[X]`` return ``(X, True)``; else ``(annot, False)``.

    Shared by all bucket matchers below.
    """
    if get_origin(annot) is Union:
        args = get_args(annot)
        if type(None) in args:
            non_none = [a for a in args if a is not type(None)]
            if len(non_none) == 1:
                return non_none[0], True
    return annot, False


class _Bucket:
    """Per-field handler for translating between a dataclass field and the
    flat ``{slot_name: slot_value}`` view consumed by ``torch.library``.

    Each concrete bucket owns:

    * a :meth:`try_build` classmethod that decides whether a given
      ``(name, annotation)`` pair belongs to this bucket (returning an
      instance, or ``None`` to defer to the next bucket);
    * the runtime :meth:`schema_slots` / :meth:`pack` / :meth:`unpack`
      logic for that field.

    :class:`_SimpleBundleBucket` is the exception: it aggregates many
    simple-typed fields into a single op input, so it does not implement
    ``try_build``. It exposes :meth:`matches_field` for the per-field
    membership test, and is constructed once at the end of dispatch
    with the collected names.
    """

    @classmethod
    def try_build(cls, name: str, annot: Any) -> Optional["_Bucket"]:
        """Return an instance handling ``(name, annot)``, or ``None``."""
        raise NotImplementedError

    def schema_slots(self) -> List[Tuple[str, str]]:
        """Return ``[(slot_name, schema_type_str), ...]`` for this field."""
        raise NotImplementedError

    def pack(self, owner: "ArgObject") -> List[Tuple[str, Any]]:
        """Return ``[(slot_name, value), ...]`` extracted from ``owner``."""
        raise NotImplementedError

    def unpack(self, args: Dict[str, Any], kwargs: Dict[str, Any]) -> None:
        """Read this field's slots from ``args`` and write the
        reconstructed dataclass attribute(s) into ``kwargs``."""
        raise NotImplementedError


class _TensorOrStorageBucket(_Bucket):
    """``Tensor | QuantizedTensorStorage`` -> meta / pg / Tensor[] slots.

    Plain tensors are carried as a single-element ``Tensor[]``. Quantized
    tensor wrappers and storage shells are carried through their
    ``_torch_compile_flatten`` protocol so the backward op receives the same
    structured object type that eager restoration produced.
    """

    SUFFIX_META = "__tsmeta"
    SUFFIX_PG = "__tspg"
    SUFFIX_TENSORS = "__tstensors"

    KIND_KEY = "_te_tensor_storage_kind"
    KIND_NONE = "none"
    KIND_TENSOR = "tensor"

    def __init__(self, name: str) -> None:
        if _OPAQUE_SIMPLE_META_TYPE_NAME is None or _PROCESS_GROUP_TYPE_NAME is None:
            raise RuntimeError(
                f"Tensor/storage field {name!r} requires both "
                "OpaqueSimpleMetadata and torch.distributed.ProcessGroup "
                "to be registered as torch._library opaque types; one or "
                "both are unavailable in this PyTorch build."
            )
        self.name = name

    @staticmethod
    def _is_tensor_storage_union(annot: Any) -> bool:
        origin = get_origin(annot)
        if origin is not Union:
            return False
        members = [a for a in get_args(annot) if a is not type(None)]
        if torch.Tensor not in members:
            return False
        try:
            from transformer_engine.pytorch.quantized_tensor import QuantizedTensorStorage
        except Exception:  # pragma: no cover - partial init
            return False
        return any(
            isinstance(member, type) and issubclass(member, QuantizedTensorStorage)
            for member in members
        )

    @classmethod
    def try_build(cls, name: str, annot: Any) -> Optional["_TensorOrStorageBucket"]:
        if cls._is_tensor_storage_union(annot):
            return cls(name)
        return None

    def _slot_meta(self) -> str:
        return self.name + self.SUFFIX_META

    def _slot_pg(self) -> str:
        return self.name + self.SUFFIX_PG

    def _slot_tensors(self) -> str:
        return self.name + self.SUFFIX_TENSORS

    def schema_slots(self) -> List[Tuple[str, str]]:
        return [
            (self._slot_meta(), _OPAQUE_SIMPLE_META_TYPE_NAME),
            (self._slot_pg(), _PROCESS_GROUP_TYPE_NAME + "?"),
            (self._slot_tensors(), "Tensor[]"),
        ]

    def pack(self, owner: "ArgObject") -> List[Tuple[str, Any]]:
        value = getattr(owner, self.name)
        if value is None:
            meta = OpaqueSimpleMetadata({self.KIND_KEY: self.KIND_NONE})
            pg: Any = None
            tensors: List[torch.Tensor] = []
        else:
            from transformer_engine.pytorch.quantized_tensor import QuantizedTensorStorage

            if isinstance(value, QuantizedTensorStorage):
                meta, pg, tensors = value._torch_compile_flatten()
            elif isinstance(value, torch.Tensor):
                meta = OpaqueSimpleMetadata({self.KIND_KEY: self.KIND_TENSOR})
                pg = None
                tensors = [value]
            else:
                raise TypeError(
                    f"{type(owner).__name__} field {self.name!r} expected "
                    "None, torch.Tensor, or QuantizedTensorStorage, got "
                    f"{type(value).__name__}"
                )
        return [
            (self._slot_meta(), meta),
            (self._slot_pg(), pg),
            (self._slot_tensors(), list(tensors)),
        ]

    def unpack(self, args: Dict[str, Any], kwargs: Dict[str, Any]) -> None:
        meta = args[self._slot_meta()]
        kind = meta.get(self.KIND_KEY)
        if kind == self.KIND_NONE:
            kwargs[self.name] = None
            return
        tensors = args[self._slot_tensors()]
        if kind == self.KIND_TENSOR:
            kwargs[self.name] = tensors[0]
            return

        from transformer_engine.pytorch.quantized_tensor import QuantizedTensorStorage

        kwargs[self.name] = QuantizedTensorStorage._torch_compile_unflatten(
            meta,
            args[self._slot_pg()],
            tensors,
        )


class _TensorBucket(_Bucket):
    """``Tensor`` / ``Optional[Tensor]`` -> single ``Tensor`` / ``Tensor?`` slot."""

    def __init__(self, name: str, is_optional: bool) -> None:
        self.name = name
        self.type_str = "Tensor?" if is_optional else "Tensor"

    @classmethod
    def try_build(cls, name: str, annot: Any) -> Optional["_TensorBucket"]:
        stripped, is_optional = _strip_optional(annot)
        if stripped is torch.Tensor:
            return cls(name, is_optional)
        return None

    def schema_slots(self) -> List[Tuple[str, str]]:
        return [(self.name, self.type_str)]

    def pack(self, owner: "ArgObject") -> List[Tuple[str, Any]]:
        return [(self.name, getattr(owner, self.name))]

    def unpack(self, args: Dict[str, Any], kwargs: Dict[str, Any]) -> None:
        kwargs[self.name] = args[self.name]


class _TensorListBucket(_Bucket):
    """``List[Tensor]`` / ``Tuple[Tensor, ...]`` -> single ``Tensor[]`` slot.

    Used for fields like ``LinearBwdArgs.saved_tensors`` that carry an
    arbitrary-length sequence of tensors (typically the
    ``ctx.saved_tensors`` payload restored before invoking the backward
    op). The slot itself is non-nullable, but individual ``None``
    elements are smuggled through using :func:`_encode_none` /
    :func:`_decode_none` sentinels (matching what the forward op return
    list already does). An empty sequence is valid.
    """

    def __init__(self, name: str, container: type) -> None:
        self.name = name
        # Remember the original container type so unpack returns the
        # exact same Python type the dataclass annotation declared.
        self.container = container

    @classmethod
    def try_build(cls, name: str, annot: Any) -> Optional["_TensorListBucket"]:
        stripped, _ = _strip_optional(annot)
        origin = get_origin(stripped)
        if origin is None:
            return None
        args = get_args(stripped)
        if not args:
            return None
        # ``Tuple[Tensor, ...]`` -> args = (Tensor, Ellipsis); other forms
        # like ``Tuple[Tensor, Tensor]`` or ``List[Tensor]`` only have
        # type entries.
        if origin is tuple:
            if len(args) == 2 and args[1] is Ellipsis:
                elem = args[0]
            else:
                elem = args[0] if all(a is args[0] for a in args) else None
        elif origin is list:
            elem = args[0]
        else:
            return None
        if elem is not torch.Tensor:
            return None
        return cls(name, list if origin is list else tuple)

    def schema_slots(self) -> List[Tuple[str, str]]:
        return [(self.name, "Tensor[]")]

    def pack(self, owner: "ArgObject") -> List[Tuple[str, Any]]:
        value = getattr(owner, self.name) or ()
        return [(self.name, [_encode_none(t) for t in value])]

    def unpack(self, args: Dict[str, Any], kwargs: Dict[str, Any]) -> None:
        kwargs[self.name] = self.container(_decode_none(t) for t in args[self.name])


class _ProcessGroupBucket(_Bucket):
    """``ProcessGroup`` / ``Optional[ProcessGroup]`` -> one direct opaque-ref slot.

    PG is registered upstream (in ``torch.distributed.device_mesh``) as
    a value-opaque type, so torch.library carries it without help.
    """

    def __init__(self, name: str, is_optional: bool) -> None:
        if _PROCESS_GROUP_TYPE_NAME is None:
            raise RuntimeError(
                f"ProcessGroup field {name!r} requires torch.distributed "
                "and the opaque-type registration in "
                "torch.distributed.device_mesh; neither is available in "
                "this PyTorch build."
            )
        self.name = name
        self.type_str = _PROCESS_GROUP_TYPE_NAME + ("?" if is_optional else "")

    @classmethod
    def try_build(cls, name: str, annot: Any) -> Optional["_ProcessGroupBucket"]:
        stripped, is_optional = _strip_optional(annot)
        if not isinstance(stripped, type):
            return None
        try:
            from torch.distributed import ProcessGroup
        except Exception:  # pragma: no cover - distributed not built
            return None
        if not issubclass(stripped, ProcessGroup):
            return None
        return cls(name, is_optional)

    def schema_slots(self) -> List[Tuple[str, str]]:
        return [(self.name, self.type_str)]

    def pack(self, owner: "ArgObject") -> List[Tuple[str, Any]]:
        return [(self.name, getattr(owner, self.name))]

    def unpack(self, args: Dict[str, Any], kwargs: Dict[str, Any]) -> None:
        kwargs[self.name] = args[self.name]


def _flattenable_bases() -> Tuple[type, ...]:
    """Return the list of base classes whose subclasses are routed
    through :class:`_FlattenableBucket`.

    A "flattenable" type implements the duck-typed pair

    * instance method ``_flatten() -> (OpaqueSimpleMetadata, ref, list[Tensor])``
    * classmethod ``_unflatten(meta, ref, tensors)`` (dispatches by an
      identifier stamped into ``meta``)

    Lazy import keeps ``dynamo`` importable before the modules that
    define these bases (avoid import cycles).
    """
    bases: List[type] = []
    try:
        from transformer_engine.pytorch.quantized_tensor import QuantizedTensorStorage, Quantizer

        bases.append(Quantizer)
        bases.append(QuantizedTensorStorage)
    except Exception:  # pragma: no cover - partial init
        pass
    try:
        from transformer_engine.common.recipe import Recipe

        bases.append(Recipe)
    except Exception:  # pragma: no cover - partial init
        pass
    return tuple(bases)


class _FlattenableBucket(_Bucket):
    """Three-slot expansion (``meta`` / ``ref`` / ``tensors``) for any
    field whose type implements the ``_flatten`` / ``_unflatten``
    protocol (see :func:`_flattenable_bases`). Used today for
    :class:`~transformer_engine.pytorch.quantized_tensor.Quantizer` and
    :class:`~transformer_engine.common.recipe.Recipe`.
    """

    SUFFIX_META = "__fmeta"
    SUFFIX_PG = "__fpg"
    SUFFIX_TENSORS = "__ftensors"

    # Stored under ``_qcls`` in the metadata bundle to encode ``None``
    # without making any of the three slots nullable.
    NONE_MARKER_KEY = "_qcls"
    NONE_MARKER_VAL = ""

    def __init__(self, name: str, base_cls: type) -> None:
        if _OPAQUE_SIMPLE_META_TYPE_NAME is None or _PROCESS_GROUP_TYPE_NAME is None:
            raise RuntimeError(
                f"Flattenable field {name!r} requires both "
                "OpaqueSimpleMetadata and torch.distributed.ProcessGroup "
                "to be registered as torch._library opaque types; one or "
                "both are unavailable in this PyTorch build."
            )
        self.name = name
        self.base_cls = base_cls

    @classmethod
    def try_build(cls, name: str, annot: Any) -> Optional["_FlattenableBucket"]:
        stripped, _ = _strip_optional(annot)
        if not isinstance(stripped, type):
            return None
        for base in _flattenable_bases():
            if issubclass(stripped, base):
                return cls(name, base)
        return None

    def _slot_meta(self) -> str:
        return self.name + self.SUFFIX_META

    def _slot_pg(self) -> str:
        return self.name + self.SUFFIX_PG

    def _slot_tensors(self) -> str:
        return self.name + self.SUFFIX_TENSORS

    def schema_slots(self) -> List[Tuple[str, str]]:
        return [
            (self._slot_meta(), _OPAQUE_SIMPLE_META_TYPE_NAME),
            (self._slot_pg(), _PROCESS_GROUP_TYPE_NAME + "?"),
            (self._slot_tensors(), "Tensor[]"),
        ]

    def pack(self, owner: "ArgObject") -> List[Tuple[str, Any]]:
        value = getattr(owner, self.name)
        if value is None:
            meta = OpaqueSimpleMetadata({self.NONE_MARKER_KEY: self.NONE_MARKER_VAL})
            pg: Any = None
            tensors: List[torch.Tensor] = []
        else:
            if hasattr(value, "_flatten"):
                meta, pg, tensors = value._flatten()
            else:
                meta, pg, tensors = value._torch_compile_flatten()
        return [
            (self._slot_meta(), meta),
            (self._slot_pg(), pg),
            (self._slot_tensors(), list(tensors)),
        ]

    def unpack(self, args: Dict[str, Any], kwargs: Dict[str, Any]) -> None:
        meta = args[self._slot_meta()]
        if meta.get(self.NONE_MARKER_KEY) == self.NONE_MARKER_VAL:
            kwargs[self.name] = None
            return
        if hasattr(self.base_cls, "_unflatten"):
            kwargs[self.name] = self.base_cls._unflatten(
                meta, args[self._slot_pg()], args[self._slot_tensors()]
            )
        else:
            kwargs[self.name] = self.base_cls._torch_compile_unflatten(
                meta, args[self._slot_pg()], args[self._slot_tensors()]
            )


class _SimpleBundleBucket(_Bucket):
    """Aggregator: bundles every simple-typed field of the dataclass
    into a single :class:`OpaqueSimpleMetadata` slot.

    Does not implement :meth:`try_build` because membership is decided
    per-field via :meth:`matches_field`, with construction deferred
    until all simple field names are collected.
    """

    SLOT = "_simple_meta"

    def __init__(self, names: List[str]) -> None:
        if _OPAQUE_SIMPLE_META_TYPE_NAME is None:
            raise RuntimeError(
                "OpaqueSimpleMetadata could not be registered with "
                "torch._library.opaque_object; cannot bundle simple fields "
                f"{names}. Upgrade PyTorch or drop the simple fields."
            )
        self.names = list(names)

    @classmethod
    def matches_field(cls, annot: Any) -> bool:
        """Whether ``annot`` (Optional-aware, recursive on tuple/list) is
        bundled-simple, i.e. eligible for this aggregator.

        Accepts simple primitives, :class:`enum.Enum`, :class:`torch.Size`,
        any class registered as a torch.compile *value*-opaque type, and
        nested tuples / lists thereof.
        """
        annot, _ = _strip_optional(annot)
        if annot in OpaqueSimpleMetadata.PRIMITIVE_TYPES:
            return True
        if isinstance(annot, type) and issubclass(annot, Enum):
            return True
        if annot is torch.Size:
            return True
        # Any registered value-opaque class is hashable / FX-reproducible
        # and therefore safe to embed in the OpaqueSimpleMetadata bundle.
        if isinstance(annot, type):
            try:
                from torch._library.opaque_object import is_opaque_value_type
            except Exception:  # pragma: no cover - older torch
                is_opaque_value_type = None
            if is_opaque_value_type is not None and is_opaque_value_type(annot):
                return True
        origin = get_origin(annot)
        if origin in (tuple, list):
            # Inner args may contain Ellipsis (e.g. ``Tuple[int, ...]``);
            # only require the *concrete* inner annotations to be simple.
            inner = [a for a in get_args(annot) if a is not Ellipsis]
            return bool(inner) and all(cls.matches_field(a) for a in inner)
        return False

    def schema_slots(self) -> List[Tuple[str, str]]:
        return [(self.SLOT, _OPAQUE_SIMPLE_META_TYPE_NAME)]

    def pack(self, owner: "ArgObject") -> List[Tuple[str, Any]]:
        bundle = OpaqueSimpleMetadata({n: getattr(owner, n) for n in self.names})
        return [(self.SLOT, bundle)]

    def unpack(self, args: Dict[str, Any], kwargs: Dict[str, Any]) -> None:
        if self.SLOT not in args:
            return
        meta = args[self.SLOT]
        for n in self.names:
            kwargs[n] = meta[n]


class _UnknownBucket(_Bucket):
    """Fallback for fields whose annotation no other bucket claims.
    Emits no schema slot; pack rejects non-trivial values to avoid silent
    data loss; unpack restores the field as ``None``.

    A "trivial" value is one that carries no observable information --
    ``None`` itself or a sequence of all-``None`` entries (e.g. a
    ``tensor_objects`` payload from :func:`prepare_for_saving` over a
    bag of plain bf16 tensors). Such values are dropped on the way into
    the op and reconstructed from companion fields (``saved_tensors``,
    quantizer metadata, ...) on the way out.

    Constructed directly by :meth:`ArgObject._buckets` (it has no
    annotation-based ``try_build`` -- it's the explicit "no match" case).
    """

    def __init__(self, name: str, owner_cls_name: str) -> None:
        self.name = name
        self.owner_cls_name = owner_cls_name

    @staticmethod
    def _is_trivial(value: Any) -> bool:
        if value is None:
            return True
        if isinstance(value, (list, tuple)):
            return all(v is None for v in value)
        return False

    def schema_slots(self) -> List[Tuple[str, str]]:
        return []

    def pack(self, owner: "ArgObject") -> List[Tuple[str, Any]]:
        value = getattr(owner, self.name, None)
        if not self._is_trivial(value):
            raise TypeError(
                f"{self.owner_cls_name} field {self.name!r} has a type not "
                "supported by torch.compile (not Tensor, simple, "
                "ProcessGroup, or Quantizer) and carries "
                "a non-trivial value; override "
                f"{self.owner_cls_name}.torch_compile_pack to handle it."
            )
        return []

    def unpack(self, args: Dict[str, Any], kwargs: Dict[str, Any]) -> None:
        kwargs[self.name] = None


# Buckets, in priority order, that own ``try_build`` for a single field.
_FIELD_BUCKETS: Tuple[type, ...] = (
    _TensorOrStorageBucket,
    _TensorBucket,
    _TensorListBucket,
    _ProcessGroupBucket,
    _FlattenableBucket,
)


# --------------------------------------------------------------------------- #
# ArgObject
# --------------------------------------------------------------------------- #


class ArgObject:
    """Base class for structured argument containers passed to TE custom ops.

    Subclassed by per-module forward / backward dataclasses
    (e.g. ``LinearFwdArgs``, ``LinearBwdArgs``). Provides the pack /
    unpack / schema hooks consumed by :func:`_te_register_custom_op`
    when wiring the dataclass into a ``torch.library`` schema.

    The default pack / unpack / schema implementations dispatch on
    dataclass field annotations. Each field is mapped to exactly one
    :class:`_Bucket` (see module-level docstring); the three methods
    then become trivial iterations over the bucket list.
    """

    @classmethod
    def _resolved_field_annotations(cls) -> List[Tuple[str, Any]]:
        if not dataclasses.is_dataclass(cls):
            raise TypeError(
                f"{cls.__name__} must be a @dataclass to use the default "
                f"ArgObject torch_compile_* implementations."
            )
        # ``get_type_hints`` resolves forward references and PEP 563
        # ``from __future__ import annotations`` strings.
        try:
            hints = get_type_hints(cls)
        except Exception:
            hints = {}
        return [
            (f.name, hints.get(f.name, f.type)) for f in dataclasses.fields(cls)
        ]

    @classmethod
    def _buckets(cls) -> List[_Bucket]:
        """Build the bucket list for this dataclass from field annotations.

        Dispatch order per field: try each bucket in :data:`_FIELD_BUCKETS`
        (Tensor, ProcessGroup, Quantizer); if none claims the field, route
        it to :class:`_SimpleBundleBucket` if its annotation is bundle-able,
        else to :class:`_UnknownBucket`.

        Intentionally **not** cached. Caching on ``cls`` (e.g. by writing
        ``cls.__te_buckets__``) tickles Dynamo: subsequent reads of
        ``cls.__dict__`` from a compiled function trigger
        "mappingproxy affected by dictionary mutation" graph breaks.
        Hot paths must instead capture the bucket list once at op
        registration time and pass it explicitly to :meth:`torch_compile_pack`
        / :meth:`torch_compile_unpack`.
        """
        buckets: List[_Bucket] = []
        simple_names: List[str] = []
        for name, annot in cls._resolved_field_annotations():
            built: Optional[_Bucket] = None
            for bucket_cls in _FIELD_BUCKETS:
                built = bucket_cls.try_build(name, annot)
                if built is not None:
                    break
            if built is not None:
                buckets.append(built)
            elif _SimpleBundleBucket.matches_field(annot):
                simple_names.append(name)
            else:
                buckets.append(_UnknownBucket(name, cls.__name__))
        if simple_names:
            buckets.append(_SimpleBundleBucket(simple_names))
        return buckets

    @classmethod
    def torch_compile_get_schema(cls) -> List[Tuple[str, str]]:
        """Default: derive the schema from dataclass annotations.

        See :class:`_Bucket` subclasses for the per-field-kind layout
        (Tensor, ProcessGroup, Quantizer, and the
        aggregated ``_simple_meta`` bundle of simple fields).
        """
        return [slot for b in cls._buckets() for slot in b.schema_slots()]

    def torch_compile_pack(
        self, buckets: Optional[List[_Bucket]] = None
    ) -> Dict[str, Any]:
        """Default: ask each bucket to extract its slot(s) from ``self``.

        ``buckets`` is the precomputed bucket list (from
        :meth:`_buckets`). Hot paths -- e.g. the closures created by
        :func:`_te_register_custom_op` -- must pass it to avoid recomputing
        and, critically, to keep Dynamo away from ``cls.__dict__`` while
        tracing. When ``None``, this method recomputes the buckets
        (eager-only fallback intended for ad-hoc / test usage).
        """
        if buckets is None:
            buckets = type(self)._buckets()
        out: Dict[str, Any] = {}
        for bucket in buckets:
            for name, value in bucket.pack(self):
                out[name] = value
        return out

    @classmethod
    def torch_compile_unpack(
        cls,
        args: Dict[str, Any],
        buckets: Optional[List[_Bucket]] = None,
    ) -> "ArgObject":
        """Default: ask each bucket to inject its field(s) into a fresh
        instance built via ``__new__`` (we bypass the dataclass
        ``__init__`` so unknown-typed fields can stay as ``None`` even
        when they have no default).

        ``buckets`` semantics match :meth:`torch_compile_pack`: hot paths
        pass the precomputed list, eager-only callers may omit it.
        """
        if buckets is None:
            buckets = cls._buckets()
        kwargs: Dict[str, Any] = {}
        for bucket in buckets:
            bucket.unpack(args, kwargs)
        obj = cls.__new__(cls)
        for k, v in kwargs.items():
            object.__setattr__(obj, k, v)
        return obj

    @classmethod
    def torch_compile_get_input_tensors_for_grad(cls) -> List[str]:
        """Names of forward inputs (from :meth:`torch_compile_get_schema`)
        for which the corresponding ``backward_impl`` produces gradients,
        in the exact order ``backward_impl`` returns them.

        Only meaningful on the forward arg type. Default is ``[]`` (no
        gradients, e.g. for inference-only ops). The wrapper uses this
        to pad the autograd return tuple with ``None`` for every input
        not listed here, so torch sees one slot per forward input as
        required by ``register_autograd``.
        """
        return []


def _te_register_custom_op(
    *,
    linear_impl: Callable[[Any], Any],
    linear_arg_type: type,
    setup_context: Callable[..., None],
    backward_impl: Callable[[Any], Any],
    backward_obj: type,
    backward_arg_type: type,
    num_outputs: int,
    linear_fake_impl: Optional[Callable[[Any], Any]] = None,
    backward_fake_impl: Optional[Callable[[Any], Any]] = None,
    op_namespace: str = "transformer_engine",
    op_name: str = "linear",
) -> Callable[..., Any]:
    """Register a TE module's forward + backward as a single torch custom op.

    Parameters
    ----------
    linear_impl
        Eager forward implementation. Receives a single argument of type
        ``linear_arg_type`` and must return a tuple of the form
        ``(*output_tensors, tensors_to_save, tensor_objects, ctx_attrs)``
        where:

        * ``output_tensors`` -- one or more :class:`torch.Tensor` outputs
          returned to the caller.
        * ``tensors_to_save`` -- flat list of :class:`torch.Tensor` to be
          stashed via ``ctx.save_for_backward``.
        * ``tensor_objects`` -- the metadata object produced by
          :func:`prepare_for_saving`, paired with ``tensors_to_save`` to
          let the backward reconstruct quantized / structured tensors.
        * ``ctx_attrs`` -- non-tensor state to attach to the autograd
          context, restricted to values that cannot be derived from the
          forward args inside ``setup_context``.
    linear_arg_type
        Dataclass type aggregating all forward inputs (e.g.
        :class:`LinearFwdArgs`). Used to (re)build the structured argument
        from the flat tensor / non-tensor inputs accepted by the custom op.
    setup_context
        Eager autograd ``setup_context`` analogue. Receives a freshly
        constructed ``backward_obj`` instance, the forward args, the
        forward output, and ``ctx_attrs`` produced by ``linear_impl``;
        is responsible for populating the backward-state object so that
        ``backward_impl`` can later consume it.
    backward_impl
        Eager backward implementation. Receives a single argument of type
        ``backward_arg_type`` and returns the gradient tuple.
    backward_obj
        Dataclass / class used to instantiate a fresh backward-state
        container at the end of the forward pass (typically the same as
        ``backward_arg_type``).
    backward_arg_type
        Type accepted by ``backward_impl``. May differ from ``backward_obj``
        if the backward op needs a wrapped / opaque view of the state.
    num_outputs
        Number of user-facing tensor outputs returned by ``linear_impl``.
        The op concatenates ``[*output_tensors, *tensors_to_save]`` into
        a single ``Tensor[]`` return; the wrapper uses ``num_outputs`` to
        split the two halves on the way back out.

        The list of forward inputs that receive gradients is declared on
        the forward arg type itself, via
        :meth:`ArgObject.torch_compile_get_input_tensors_for_grad`.
        ``backward_impl`` must return its gradients in that exact order.
    linear_fake_impl
        Optional fake (shape inference) counterpart of ``linear_impl``,
        registered via ``torch.library.register_fake``. Returns the same
        tuple shape as ``linear_impl`` -- ``(*output_tensors,
        tensors_to_save, tensor_objects, ctx_attrs)`` -- but every
        ``torch.Tensor`` is a fake tensor (allocated via
        ``quantizer.make_empty`` or ``torch.empty``) carrying only the
        correct shape / dtype / device, with no real storage or
        computation. ``tensor_objects`` and ``ctx_attrs`` must be
        structurally identical to those produced by ``linear_impl`` so
        that ``setup_context`` and ``backward_impl`` see the same
        non-tensor state in eager and traced modes.
    backward_fake_impl
        Optional fake counterpart of ``backward_impl``. Returns the same
        gradient tuple as ``backward_impl``, with fake tensors in place
        of the real gradients.
    op_namespace, op_name
        Library namespace / op name used when registering with
        ``torch.library``.

    Returns
    -------
    Callable
        A function ``forward_fn(linear_arg_type_instance)`` that dispatches
        through the registered custom op, returning the user-facing
        outputs (single tensor if ``num_outputs == 1``, otherwise a
        tuple). Use under ``torch.compiler.is_compiling()`` as a drop-in
        for ``Function.apply``.
    """

    fwd_qualname = f"{op_namespace}::{op_name}"
    bwd_op_name = f"{op_name}_backward"
    bwd_qualname = f"{op_namespace}::{bwd_op_name}"

    # Precompute the bucket list for both arg types and capture them in
    # the closures below. Critical for the compiled path: re-deriving
    # buckets at call time would force ``ArgObject._buckets`` to read
    # ``cls.__dict__`` from inside a Dynamo-traced function, which
    # triggers a "mappingproxy affected by dictionary mutation" graph
    # break under ``fullgraph=True``.
    fwd_buckets: List[_Bucket] = linear_arg_type._buckets()
    bwd_buckets: List[_Bucket] = backward_arg_type._buckets()

    def _build_schema(buckets: List[_Bucket]) -> Tuple[str, List[str]]:
        spec = [slot for b in buckets for slot in b.schema_slots()]
        names = [name for name, _ in spec]
        schema_str = "(" + ", ".join(f"{type_str} {name}" for name, type_str in spec) + ")"
        return schema_str, names

    fwd_schema_args, fwd_arg_names = _build_schema(fwd_buckets)
    bwd_schema_args, bwd_arg_names = _build_schema(bwd_buckets)

    # ``torch.library.register_autograd`` requires the backward to return
    # one grad slot per forward input, with the same Python tree
    # structure as the input itself: a ``Tensor[]`` slot must get back a
    # ``list``, not a bare ``None``. Precompute the per-slot "no-grad"
    # value so the autograd return matches.
    fwd_slot_defaults: List[Any] = []
    for bucket in fwd_buckets:
        for _, type_str in bucket.schema_slots():
            fwd_slot_defaults.append([] if type_str.endswith("[]") else None)

    # Validate ``input_tensors_for_grad`` references real forward inputs
    # and precompute the positions where backward grads land in the
    # autograd return tuple. Some logical fields (e.g. Tensor-or-storage
    # fields) expand to a ``Tensor[]`` slot; their gradient must be returned
    # as a list matching that input tree.
    input_tensors_for_grad = linear_arg_type.torch_compile_get_input_tensors_for_grad()
    fwd_grad_targets: Dict[str, Tuple[int, bool]] = {}
    slot_offset = 0
    for bucket in fwd_buckets:
        slots = bucket.schema_slots()
        if isinstance(bucket, _TensorBucket):
            fwd_grad_targets[bucket.name] = (slot_offset, False)
        elif isinstance(bucket, _TensorListBucket):
            fwd_grad_targets[bucket.name] = (slot_offset, True)
        elif isinstance(bucket, _TensorOrStorageBucket):
            for i, (slot_name, _) in enumerate(slots):
                if slot_name == bucket._slot_tensors():
                    fwd_grad_targets[bucket.name] = (slot_offset + i, True)
                    break
        slot_offset += len(slots)
    unknown_grad_names = [n for n in input_tensors_for_grad if n not in fwd_grad_targets]
    if unknown_grad_names:
        raise ValueError(
            f"{linear_arg_type.__name__}.torch_compile_get_input_tensors_for_grad() "
            f"contains names not present in "
            f"{linear_arg_type.__name__}.torch_compile_get_schema(): "
            f"{unknown_grad_names}"
        )
    grad_targets = [fwd_grad_targets[n] for n in input_tensors_for_grad]
    num_grad_inputs = len(input_tensors_for_grad)

    lib = torch.library.Library(op_namespace, "FRAGMENT")
    # Forward op concatenates user outputs and tensors_to_save into a
    # single ``Tensor[]`` return so that autograd's ``setup_context`` can
    # stash the saved-for-backward tensors without re-running the eager
    # impl. The schema is non-nullable (``Tensor[]``, not ``Tensor?[]``)
    # because ``torch.library.register_autograd`` does not propagate
    # ``grad_fn`` to a nullable list output. ``None`` entries on either
    # side are smuggled through via :func:`_encode_none` /
    # :func:`_decode_none` sentinels (see below).
    lib.define(f"{op_name}{fwd_schema_args} -> Tensor[]")
    lib.define(f"{bwd_op_name}{bwd_schema_args} -> Tensor[]")

    def _outputs_for_setup(outputs: List[torch.Tensor]) -> Any:
        return outputs[0] if num_outputs == 1 else tuple(outputs)

    def _prepare_for_saving(tensors: Any) -> Tuple[List[Optional[torch.Tensor]], Any]:
        from transformer_engine.pytorch.quantized_tensor import prepare_for_saving

        return prepare_for_saving(*(tensors or ()))

    def _restore_from_saved(tensor_objects: Any, saved_tensors: List[Any]) -> Any:
        from transformer_engine.pytorch.quantized_tensor import restore_from_saved

        return restore_from_saved(tensor_objects, saved_tensors)

    def _fwd_impl(*flat: Any) -> List[torch.Tensor]:
        kwargs = dict(zip(fwd_arg_names, flat))
        obj = linear_arg_type.torch_compile_unpack(kwargs, fwd_buckets)
        result = linear_impl(obj)
        outputs = list(result[:num_outputs])
        tensors_to_save, _ = _prepare_for_saving(result[num_outputs])
        return [_encode_none(t) for t in outputs + tensors_to_save]

    lib.impl(op_name, _fwd_impl, "CompositeExplicitAutograd")

    if linear_fake_impl is not None:

        def _fwd_fake(*flat: Any) -> List[torch.Tensor]:
            kwargs = dict(zip(fwd_arg_names, flat))
            obj = linear_arg_type.torch_compile_unpack(kwargs, fwd_buckets)
            result = linear_fake_impl(obj)
            outputs = list(result[:num_outputs])
            tensors_to_save, _ = _prepare_for_saving(result[num_outputs])
            return [_encode_none(t) for t in outputs + tensors_to_save]

        torch.library.register_fake(fwd_qualname, _fwd_fake, lib=lib)

    def _check_bwd_len(grads):
        if len(grads) != num_grad_inputs:
            raise RuntimeError(
                f"{op_namespace}::{bwd_op_name} expected backward_impl to "
                f"return {num_grad_inputs} grads (one per "
                f"input_tensors_for_grad entry), got {len(grads)}"
            )

    def _bwd_impl(*flat: Any) -> List[torch.Tensor]:
        kwargs = dict(zip(bwd_arg_names, flat))
        obj = backward_arg_type.torch_compile_unpack(kwargs, bwd_buckets)
        grads = list(backward_impl(obj))
        _check_bwd_len(grads)
        return [_encode_none(g) for g in grads]

    lib.impl(bwd_op_name, _bwd_impl, "CompositeExplicitAutograd")

    if backward_fake_impl is not None:

        def _bwd_fake(*flat: Any) -> List[torch.Tensor]:
            kwargs = dict(zip(bwd_arg_names, flat))
            obj = backward_arg_type.torch_compile_unpack(kwargs, bwd_buckets)
            grads = list(backward_fake_impl(obj))
            _check_bwd_len(grads)
            return [_encode_none(g) for g in grads]

        torch.library.register_fake(bwd_qualname, _bwd_fake, lib=lib)

    # Re-run fake (or real) impl in setup_context to recover
    # tensor_objects / ctx_attrs, which are not part of the op's return.
    fake_for_setup = linear_fake_impl if linear_fake_impl is not None else linear_impl

    def _setup_context(ctx, inputs, output):
        ctx._te_fwd_tensor_list_lengths = {
            i: len(value) for i, value in enumerate(inputs) if isinstance(value, list)
        }
        kwargs = dict(zip(fwd_arg_names, inputs))
        fwd_obj = linear_arg_type.torch_compile_unpack(kwargs, fwd_buckets)
        fake_result = fake_for_setup(fwd_obj)
        _, tensor_objects = _prepare_for_saving(fake_result[num_outputs])
        ctx_attrs = fake_result[num_outputs + 2]

        # Split op output: first num_outputs are user-facing tensors,
        # the rest are tensors_to_save. ``output`` is a flat ``Tensor[]``
        # with our None-sentinels in place; decode here so downstream
        # eager code sees the original ``None``\ s.
        user_outputs = [_decode_none(t) for t in output[:num_outputs]]
        op_saved_tensors = [_decode_none(t) for t in output[num_outputs:]]
        tensors_to_save_from_forward = _restore_from_saved(
            tensor_objects,
            op_saved_tensors,
        )

        bwd_obj = backward_obj()
        tensors_to_save_from_setup = setup_context(
            bwd_obj,
            fwd_obj,
            _outputs_for_setup(user_outputs),
            ctx_attrs,
            tensors_to_save_from_forward,
        )
        tensors_to_save, tensor_objects = _prepare_for_saving(tensors_to_save_from_setup)
        ctx.tensor_objects = tensor_objects
        ctx.save_for_backward(*tensors_to_save)
        ctx.bwd_obj = bwd_obj

    def _autograd_backward(ctx, *grad_outputs):
        bwd_obj = ctx.bwd_obj
        if hasattr(bwd_obj, "setup_saved_tensors"):
            bwd_obj.setup_saved_tensors(ctx.saved_tensors, ctx.tensor_objects)
        ctx.tensor_objects = None
        # The forward op returns a single ``Tensor[]`` (concatenation of
        # user outputs and saved tensors), so ``grad_outputs`` is a
        # 1-tuple containing the per-element grad list. Only the first
        # ``num_outputs`` of those correspond to user-facing outputs;
        # ``grad_output`` for the backward is the grad of the primary
        # output.
        per_output_grads = grad_outputs[0]
        bwd_obj.grad_output = _decode_none(per_output_grads[0])
        kwargs = backward_arg_type.torch_compile_pack(bwd_obj, bwd_buckets)
        bwd_args_flat = [kwargs[name] for name in bwd_arg_names]
        bwd_op = getattr(getattr(torch.ops, op_namespace), bwd_op_name)
        grads = [_decode_none(g) for g in bwd_op(*bwd_args_flat)]
        # ``register_autograd`` requires one grad slot per forward input
        # with the same tree structure as the input (a ``Tensor[]`` slot
        # must get back a list, never a bare ``None``). Start from the
        # precomputed per-slot defaults and overlay the produced grads
        # at the positions declared by ``input_tensors_for_grad``.
        out: List[Any] = list(fwd_slot_defaults)
        tensor_list_lengths = getattr(ctx, "_te_fwd_tensor_list_lengths", {})
        for (pos, as_list), g in zip(grad_targets, grads):
            if as_list:
                length = tensor_list_lengths.get(pos, 1)
                out[pos] = [g] + [None] * (length - 1)
            else:
                out[pos] = g
        return tuple(out)

    torch.library.register_autograd(
        fwd_qualname,
        _autograd_backward,
        setup_context=_setup_context,
        lib=lib,
    )

    fwd_op = getattr(getattr(torch.ops, op_namespace), op_name)

    def forward_fn(fwd_args):
        # Bind ``lib`` here so its registrations (impl / register_fake /
        # register_autograd) outlive ``_te_register_custom_op`` even if
        # all other references to it are dropped: ``torch.library`` uses
        # the ``Library`` instance lifetime for all attached registrations.
        _ = lib  # noqa: F841 -- closure-captured for lifetime only
        kwargs = linear_arg_type.torch_compile_pack(fwd_args, fwd_buckets)
        flat = [kwargs[name] for name in fwd_arg_names]
        result = fwd_op(*flat)
        outputs = [_decode_none(t) for t in result[:num_outputs]]
        if num_outputs == 1:
            return outputs[0]
        return tuple(outputs)

    return forward_fn
