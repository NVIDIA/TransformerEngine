# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""torch.compile custom-op framework for Transformer Engine.

Registers TE modules' eager forward/backward as ``torch.library`` custom ops so
``torch.compile(fullgraph=True)`` traces them as single graph nodes.
``register_custom_op`` is the entry point; ``module/linear.py`` is the first user.

A TE forward/backward implementation takes one dataclass argument
(``fwd_arg_type`` / ``bwd_arg_type``, e.g. ``LinearFwdArgs``) whose fields mix
tensors, quantized tensors, quantizers, process groups and plain Python values.

A ``torch.library`` custom op is narrower: it only accepts flat schema slots
(tensors plus opaque objects) and returns a flat ``Tensor[]``.

Bridging the two takes three parts (below): per-field *adapters* map the args
dataclass onto the op's input slots; *fake impls* on data-free specs give the
output geometry and reassemble the op's flat return; and a *two-tier op* lets a
quantized-tensor subclass be an op input.

Field <-> slot mapping. This mapping turns each field of the args dataclass into
the op's flat input slots, in a way that suits the field's type. A field's type
annotation selects the one ``_Adapter`` that handles it; that adapter declares the
slot(s) the field needs, packs the field's value into them on the way into the op,
and unpacks it back on the way out. The kinds -- and how each represents its field
as op inputs:

  * ``_TensorAdapter`` -- a plain ``Tensor`` / ``Optional[Tensor]``: one tensor
    slot.
  * ``_TensorOrQuantizedAdapter`` -- a field that may be a plain tensor, a bare
    quantized storage, or ``None``: three slots (the tensor, its flat inner
    buffers, and a ``__kind__`` tag) so a quantized tensor crosses as its buffers.
  * ``_QuantizerAdapter`` -- a quantizer, baked into the graph as a value-opaque
    constant.
  * ``_ProcessGroupAdapter`` -- a ProcessGroup, carried as its c10d registry
    name and re-resolved inside the op.
  * ``_SimpleBundleAdapter`` -- every remaining simple value (scalars, enums,
    sizes, nested collections of them), gathered into one ``OpaqueValueBundle``
    slot.
  * ``_UnsupportedAdapter`` -- fallback for a field no adapter can encode; allowed
    only when its value is trivial (``None`` / all-``None``) at call time.

What runs where. Each op registers a data-free fake (``register_fake``) so it
traces under ``torch.compile`` without allocating. ``register_custom_op`` returns
``forward_fn`` -- the drop-in for the eager ``autograd.Function.apply``. A forward
call through it:

  * runs the fake ``fwd_fake_impl`` on ``TensorSpec`` descriptors (data-free; see
    ``tensor_spec.py``) to get the outputs' geometry in pure Python;
  * calls the *forward op* -- which runs the real ``fwd_impl`` -- for a flat
    ``Tensor[]`` payload;
  * rebuilds the structured user outputs from that payload, sliced and reassembled
    per the fake's output descriptors (``_unflatten_values``;
    ``_flatten_value`` is the pack-side inverse).

Autograd, registered on the op, drives backward:

  * ``setup_context`` (run when the forward is taped) re-runs ``fwd_fake_impl`` for
    the saved-tensor descriptors and a ``ctx_attrs`` dict, reassembles the saved
    tensors from the op's flat output, then calls the user ``setup_context`` to
    fill the backward args from forward state + ``ctx_attrs`` (e.g. saved-tensor
    aliases) and return the tensors to persist;
  * on ``backward()`` the backward args container's optional ``setup_saved_tensors``
    hook restores those saved tensors, then the *backward op* runs the real
    ``bwd_impl`` and returns the flat grads (``bwd_fake_impl`` is its
    data-free fake).

Two-tier op (``base`` + ``wrapper``), so a ``QuantizedTensor`` subclass can be an
op *input*. The ``<op>_base`` op carries the real schema + autograd; a custom op
can't take a tensor-subclass input directly, so the ``<op>`` wrapper intercepts
those via ``register_torch_dispatch`` and flattens each into the base op's slots
(``_flatten_subclass_into_slots``) before forwarding. An empty subclass list makes
the wrapper a pass-through (plain / bf16 calls go straight through).
"""

from __future__ import annotations
import dataclasses
import math
import types as _types  # aliased: torch_dispatch rules take a ``types`` param
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)

import torch

from torch._prims_common import make_contiguous_strides_for

from .tensor_spec import TensorSpec, to_tensor_spec
from ..quantized_tensor import (
    QuantizedTensor,
    QuantizedTensorStorage,
    Quantizer,
    _quantized_tensor_passthrough_ops,
    prepare_for_saving,
)
from ..utils import record_compile_disabled

_TE_OP_NAMESPACE = "transformer_engine_compile"

# Annotation for an op arg field that may hold a plain tensor, a quantized
# tensor subclass or a *bare* ``QuantizedTensorStorage`` (the internal-quantizer
# optimization). Matched exactly by ``_TensorOrQuantizedAdapter``.
TensorOrQuantized = Union[torch.Tensor, QuantizedTensorStorage]


# ``None`` entries in an op's flat ``Tensor[]`` return are smuggled through a
# 0-element sentinel tensor: a non-nullable ``Tensor[]`` schema is required for
# ``register_autograd`` to attach a ``grad_fn`` to the outputs. The sentinel is
# recognized by (numel == 0, dtype), so its dtype must be one no real payload
# tensor can have -- complex is never a TE payload (quantized data / scales are
# uint8 / fp32, outputs are float), while e.g. a uint8 sentinel would collide
# with a genuinely empty FP8 data buffer (empty batch).
_NONE_SENTINEL_DTYPE = torch.complex32


def _encode_none(t: Optional[torch.Tensor]) -> torch.Tensor:
    """Replace ``None`` with a 0-element sentinel tensor."""
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
# OpaqueValueBundle: bundle of simple / value-opaque Python values
# --------------------------------------------------------------------------- #


class OpaqueValueBundle:
    """Opaque value-type bundle of simple Python values.

    Wraps a ``{name: value}`` dict so many small non-Tensor args pass through a
    single custom-op input; registered as a torch.compile *value* opaque type
    (Dynamo specializes the graph on its contents). Allowed values: primitives
    in :attr:`PRIMITIVE_TYPES` (incl. ``torch.Size``), ``enum.Enum``, classes,
    any registered value-opaque type (e.g. TE quantizers), plus nested tuples /
    lists / dicts thereof (so a bundle can carry a ``__tensor_flatten__``
    context verbatim -- including its ``cls`` entry).
    """

    PRIMITIVE_TYPES: Tuple[type, ...] = (
        type(None),
        bool,
        int,
        float,
        str,
        torch.dtype,
        torch.device,
        torch.Size,
    )

    @classmethod
    def is_simple_value(cls, value: Any) -> bool:
        """Whether ``value`` may be stored inside an instance (recursive)."""
        if isinstance(value, cls.PRIMITIVE_TYPES):
            return True
        if isinstance(value, Enum):
            return True
        if isinstance(value, type):
            return True
        if _is_opaque_value_type is not None and _is_opaque_value_type(type(value)):
            return True
        if isinstance(value, dict):
            return all(isinstance(k, str) and cls.is_simple_value(v) for k, v in value.items())
        if isinstance(value, (list, tuple)):
            return all(cls.is_simple_value(v) for v in value)
        return False

    @classmethod
    def _to_hashable(cls, value: Any) -> Any:
        # Tag with the concrete type so e.g. [1] / (1,) / Size([1]) or True / 1
        # stay distinct under __eq__ / __hash__ (graph guards compare bundles).
        if isinstance(value, dict):
            return ("dict", tuple(sorted((k, cls._to_hashable(v)) for k, v in value.items())))
        if isinstance(value, (list, tuple)):  # incl. torch.Size
            return (type(value).__name__, tuple(cls._to_hashable(v) for v in value))
        return (type(value).__name__, value)

    @classmethod
    def _fmt_simple(cls, value: Any) -> str:
        """Repr for a value, evaluable in a context with ``torch`` globals."""
        if isinstance(value, torch.dtype):
            return f"__import__('torch').{str(value).split('.')[-1]}"
        if isinstance(value, torch.device):
            return f"__import__('torch').device({str(value)!r})"
        if isinstance(value, torch.Size):
            return f"__import__('torch').Size({list(value)!r})"
        # Enum before primitives: IntEnum is also ``int`` but must render as
        # ``EnumName.MEMBER`` (the Enum class is added to globals by ``_collect``).
        if isinstance(value, Enum):
            return f"{type(value).__name__}.{value.name}"
        # Class objects (e.g. the flatten context's ``cls``) render by name; the
        # class itself is added to globals by ``_collect``.
        if isinstance(value, type):
            return value.__name__
        if isinstance(value, dict):
            body = ", ".join(f"{k!r}: {cls._fmt_simple(v)}" for k, v in value.items())
            return f"{{{body}}}"
        if isinstance(value, list):
            return "[" + ", ".join(cls._fmt_simple(v) for v in value) + "]"
        if isinstance(value, tuple):
            body = ", ".join(cls._fmt_simple(v) for v in value)
            return f"({body},)" if len(value) == 1 else f"({body})"
        if _is_opaque_value_type(type(value)):
            return value.__fx_repr__()[0]
        # repr(float('inf')) is 'inf', which is not an evaluable literal.
        if isinstance(value, float) and not math.isfinite(value):
            return f"float({str(value)!r})"
        return repr(value)

    def __init__(self, data: Optional[Dict[str, Any]] = None) -> None:
        data = dict(data) if data else {}
        for k, v in data.items():
            if not OpaqueValueBundle.is_simple_value(v):
                raise TypeError(
                    f"OpaqueValueBundle field '{k}' has unsupported type "
                    f"{type(v).__name__}; only simple primitives, Enum, "
                    "torch.Size, registered value-opaque types and nested "
                    "tuples / lists / dicts thereof are allowed."
                )
        self._data: Dict[str, Any] = data
        self._frozen: Tuple[Tuple[str, Any], ...] = tuple(
            (k, OpaqueValueBundle._to_hashable(v)) for k, v in sorted(data.items())
        )
        # Precomputed: Dynamo guards hash bundles on every compiled call.
        self._hash: int = hash(self._frozen)

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __getattr__(self, name: str) -> Any:
        # Underscored names raise cleanly: copy/pickle probe dunders on a clone
        # created without __init__, where reading ``self._data`` would recurse.
        if name.startswith("_"):
            raise AttributeError(name)
        try:
            return self._data[name]
        except KeyError as e:
            raise AttributeError(name) from e

    def get(self, key: str, default: Any = None) -> Any:
        """Return ``self._data.get(key, default)``."""
        return self._data.get(key, default)

    def as_dict(self) -> Dict[str, Any]:
        """Return a shallow copy of the stored mapping."""
        return dict(self._data)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, OpaqueValueBundle):
            return NotImplemented
        return self._frozen == other._frozen

    def __hash__(self) -> int:
        return self._hash

    def __fx_repr__(self) -> Tuple[str, Dict[str, Any]]:
        items = ", ".join(
            f"{k!r}: {OpaqueValueBundle._fmt_simple(v)}" for k, v in self._data.items()
        )
        globals_: Dict[str, Any] = {"OpaqueValueBundle": OpaqueValueBundle}

        def _collect(value: Any) -> None:
            if isinstance(value, dict):
                for v in value.values():
                    _collect(v)
                return
            if isinstance(value, (list, tuple)):
                for v in value:
                    _collect(v)
                return
            if isinstance(value, Enum):
                globals_[type(value).__name__] = type(value)
                return
            if isinstance(value, type):
                globals_[value.__name__] = value
                return
            if isinstance(value, OpaqueValueBundle.PRIMITIVE_TYPES):
                return
            if _is_opaque_value_type(type(value)):
                _, extra = value.__fx_repr__()
                globals_.update(extra)

        for v in self._data.values():
            _collect(v)
        return (f"OpaqueValueBundle({{{items}}})", globals_)


try:
    from torch._library.opaque_object import (
        get_opaque_type_name,
        is_opaque_value_type as _is_opaque_value_type,
        register_opaque_type,
    )

    register_opaque_type(OpaqueValueBundle, typ="value")
    _OPAQUE_VALUE_BUNDLE_TYPE_NAME: Optional[str] = get_opaque_type_name(OpaqueValueBundle)
# Older torch without opaque_object support.
except Exception as e:  # pylint: disable=broad-exception-caught  # pragma: no cover
    record_compile_disabled(
        f"could not register OpaqueValueBundle as an opaque type ({e}); use a newer PyTorch build"
    )
    _is_opaque_value_type = None
    _OPAQUE_VALUE_BUNDLE_TYPE_NAME = None

try:
    from torch._C._distributed_c10d import ProcessGroup as _PROCESS_GROUP_TYPE
    from torch._C._distributed_c10d import _resolve_process_group
except ImportError:  # pragma: no cover
    _PROCESS_GROUP_TYPE = None
    _resolve_process_group = None


# --------------------------------------------------------------------------- #
# Storage flatten / unflatten (value-opaque quantizer; no ProcessGroup)
# --------------------------------------------------------------------------- #


def _storage_flatten(
    value: Any, extra_meta: Optional[Dict[str, Any]] = None
) -> Tuple["OpaqueValueBundle", List[torch.Tensor]]:
    """Split a ``QuantizedTensor`` / bare storage into ``(meta, Tensor[])``.

    The flatten context (embedding the value-opaque quantizer) plus inner names
    and -- for a wrapper subclass -- the outer geometry are stashed in the bundle
    so :func:`_storage_unflatten` can rebuild without PyTorch's ``outer_size``.
    ``extra_meta`` is merged in before the bundle is built (so its ``_frozen``
    hash key stays consistent) -- used to tag the tensor-or-quantized slot ``__kind__``.
    """
    inner_names, ctx = value.__tensor_flatten__()
    meta = dict(ctx)
    meta["_inner_names"] = list(inner_names)
    if isinstance(value, torch.Tensor):
        meta["_outer_shape"] = torch.Size(value.shape)
    if extra_meta:
        meta.update(extra_meta)
    tensors = [getattr(value, name) for name in inner_names]
    return OpaqueValueBundle(meta), tensors


def _storage_unflatten(meta: Any, tensors: List[torch.Tensor]) -> Any:
    """Inverse of :func:`_storage_flatten`."""
    meta_dict = meta.as_dict() if isinstance(meta, OpaqueValueBundle) else dict(meta)
    inner_names = meta_dict["_inner_names"]
    inner = dict(zip(inner_names, tensors))
    outer_shape = meta_dict.get("_outer_shape")
    stride = make_contiguous_strides_for(tuple(outer_shape)) if outer_shape is not None else None
    return QuantizedTensorStorage.__tensor_unflatten__(inner, meta_dict, outer_shape, stride)


# --------------------------------------------------------------------------- #
# Field adapters: dataclass field <-> flat torch.library slot(s)
# --------------------------------------------------------------------------- #


def _is_union(annot: Any) -> bool:
    """True for both ``typing.Union[...]`` / ``Optional[...]`` and PEP 604 ``X | Y``.

    ``get_origin`` returns ``typing.Union`` for the former but ``types.UnionType``
    for the latter, so the two syntaxes must be checked separately.
    """
    origin = get_origin(annot)
    return origin is Union or origin is _types.UnionType


def _strip_optional(annot: Any) -> Tuple[Any, bool]:
    """If ``annot`` is ``Optional[X]`` return ``(X, True)``; else ``(annot, False)``."""
    if _is_union(annot):
        args = get_args(annot)
        if type(None) in args:
            non_none = [a for a in args if a is not type(None)]
            if len(non_none) == 1:
                return non_none[0], True
    return annot, False


class _Adapter:
    """Maps one (or, for the aggregating adapter, several) dataclass field(s)
    to/from a contiguous run of custom-op schema *slots*.

    A custom op only takes flat, simply-typed arguments, but a TE op takes a
    single ``@dataclass`` of mixed fields. Each adapter knows how to translate
    its kind of field both ways. ``try_build`` and ``schema_slots`` run once at
    registration (to build the op's schema); ``to_slots`` and ``from_slots`` run
    on each call and must agree on the slot layout that ``schema_slots`` declares.
    """

    @classmethod
    def try_build(cls, name: str, annot: Any) -> Optional["_Adapter"]:
        """Decide whether this adapter type handles the field ``name`` given its
        type annotation ``annot``; return a configured adapter if so, else
        ``None`` so the next candidate is tried.

        Called once per field at registration, iterating :data:`_FIELD_ADAPTERS`
        (adapters are mutually exclusive on annotations, so the order is not a
        ranking).
        """
        raise NotImplementedError

    def schema_slots(self) -> List[Tuple[str, str]]:
        """Declare the schema slots this field occupies, each as a
        ``(slot_name, schema_type)`` pair (e.g. ``("bias", "Tensor?")``).

        Concatenated across all adapters to form the op's schema string.
        """
        raise NotImplementedError

    def to_slots(self, owner: Any) -> Dict[str, Any]:
        """Read this field from the dataclass ``owner`` and produce the concrete
        value for each of its schema slots, as a ``{slot_name: value}`` dict.

        Composite values are flattened to fit the (tensor-only) slots: e.g. a
        quantized tensor is split into its plain inner buffers plus a metadata
        bundle. Inverse of :meth:`from_slots`.
        """
        raise NotImplementedError

    def from_slots(self, args: Dict[str, Any], kwargs: Dict[str, Any]) -> None:
        """Read this field's slots back from the op arguments ``args`` and write
        the reconstructed field value into ``kwargs`` (rebuilding any flattened
        composite). The filled ``kwargs`` are then used to rebuild the original
        dataclass for the eager implementation. Inverse of :meth:`to_slots`.
        """
        raise NotImplementedError

    def grad_slot(self) -> Optional[int]:
        """Index (within this adapter's :meth:`schema_slots`) of the slot that
        carries a gradient, or ``None`` if the field is not differentiable.

        Used to map ``input_tensors_for_grad`` names onto backward grad-output
        positions. Non-tensor adapters (quantizers, metadata) return ``None``.
        """
        return None


class _TensorOrQuantizedKind(Enum):
    """What a tensor-or-quantized slot group carries, tagged in its ``__meta``."""

    NONE = "none"
    TENSOR = "tensor"
    STORAGE = "storage"


class _TensorOrQuantizedAdapter(_Adapter):
    """``Tensor | QuantizedTensorStorage | None`` (also subclass tensor) field.

    Three slots regardless of value: ``<name>`` (``Tensor?`` -- plain / subclass
    tensor passes through, ``None`` for bare storage), ``<name>__tensors``
    (``Tensor[]`` flat inner tensors when flattened), ``<name>__meta``
    (``OpaqueValueBundle`` flatten metadata + a ``__kind__`` marker). A ``None``
    field is tagged ``_TensorOrQuantizedKind.NONE`` with the other two slots empty.
    """

    KIND_KEY = "__kind__"

    def __init__(self, name: str) -> None:
        self.name = name

    def tensor_slot(self) -> str:
        """Primary slot name for a plain / subclass tensor."""
        return self.name

    def inner_slot(self) -> str:
        """Flat inner-tensor slot name."""
        return self.name + "__tensors"

    def meta_slot(self) -> str:
        """Flatten-metadata slot name."""
        return self.name + "__meta"

    def schema_slots(self) -> List[Tuple[str, str]]:
        return [
            (self.tensor_slot(), "Tensor?"),
            (self.inner_slot(), "Tensor[]"),
            (self.meta_slot(), _OPAQUE_VALUE_BUNDLE_TYPE_NAME),
        ]

    # Matched by exact member set, so a bare quantized annotation or an
    # accidental extra union member is rejected rather than silently taken as a
    # tensor-or-quantized field.
    _MEMBERS = frozenset(get_args(TensorOrQuantized))

    @classmethod
    def _is_tensor_storage_union(cls, annot: Any) -> bool:
        if not _is_union(annot):
            return False
        members = frozenset(a for a in get_args(annot) if a is not type(None))
        return members == cls._MEMBERS

    @classmethod
    def try_build(cls, name: str, annot: Any) -> Optional["_TensorOrQuantizedAdapter"]:
        if cls._is_tensor_storage_union(annot):
            return cls(name)
        return None

    def to_slots(self, owner: Any) -> Dict[str, Any]:
        value = getattr(owner, self.name)
        if value is None:
            return {
                self.tensor_slot(): None,
                self.inner_slot(): [],
                self.meta_slot(): OpaqueValueBundle({self.KIND_KEY: _TensorOrQuantizedKind.NONE}),
            }
        if isinstance(value, torch.Tensor):
            # Plain tensor *and* subclass (e.g. Float8Tensor) pass through the
            # ``Tensor?`` slot; subclass flattening (if any) is done by the
            # wrapper op's ``register_torch_dispatch`` rule.
            return {
                self.tensor_slot(): value,
                self.inner_slot(): [],
                self.meta_slot(): OpaqueValueBundle({self.KIND_KEY: _TensorOrQuantizedKind.TENSOR}),
            }
        if isinstance(value, QuantizedTensorStorage):
            meta, tensors = _storage_flatten(value, {self.KIND_KEY: _TensorOrQuantizedKind.STORAGE})
            return {
                self.tensor_slot(): None,
                self.inner_slot(): tensors,
                self.meta_slot(): meta,
            }
        raise TypeError(
            f"field {self.name!r} expected None, torch.Tensor, or "
            f"QuantizedTensorStorage, got {type(value).__name__}"
        )

    def from_slots(self, args: Dict[str, Any], kwargs: Dict[str, Any]) -> None:
        meta = args[self.meta_slot()]
        kind = meta.get(self.KIND_KEY)
        if kind == _TensorOrQuantizedKind.NONE:
            kwargs[self.name] = None
        elif kind == _TensorOrQuantizedKind.TENSOR:
            kwargs[self.name] = args[self.tensor_slot()]
        else:
            kwargs[self.name] = _storage_unflatten(meta, args[self.inner_slot()])

    def grad_slot(self) -> Optional[int]:
        # Gradient flows to the plain / subclass tensor slot (``tensor_slot()``,
        # the first of the three).
        return 0


class _TensorAdapter(_Adapter):
    """``Tensor`` / ``Optional[Tensor]`` -> single ``Tensor`` / ``Tensor?`` slot."""

    def __init__(self, name: str, is_optional: bool) -> None:
        self.name = name
        self.type_str = "Tensor?" if is_optional else "Tensor"

    @classmethod
    def try_build(cls, name: str, annot: Any) -> Optional["_TensorAdapter"]:
        stripped, is_optional = _strip_optional(annot)
        if stripped is torch.Tensor:
            return cls(name, is_optional)
        return None

    def schema_slots(self) -> List[Tuple[str, str]]:
        return [(self.name, self.type_str)]

    def to_slots(self, owner: Any) -> Dict[str, Any]:
        return {self.name: getattr(owner, self.name)}

    def from_slots(self, args: Dict[str, Any], kwargs: Dict[str, Any]) -> None:
        kwargs[self.name] = args[self.name]

    def grad_slot(self) -> Optional[int]:
        return 0


class _QuantizerAdapter(_Adapter):
    """``Quantizer`` / ``Optional[Quantizer]`` -> one own ``OpaqueValueBundle`` slot.

    Each quantizer gets its own dedicated slot. The field is annotated with the
    base ``Quantizer`` (not itself a registered opaque type), so the simple
    bundle would not claim it.
    """

    QUANTIZER_KEY = "q"

    def __init__(self, name: str) -> None:
        self.name = name

    def meta_slot(self) -> str:
        """Opaque quantizer metadata slot name."""
        return self.name + "__q"

    @classmethod
    def try_build(cls, name: str, annot: Any) -> Optional["_QuantizerAdapter"]:
        stripped, _ = _strip_optional(annot)
        if isinstance(stripped, type) and issubclass(stripped, Quantizer):
            return cls(name)
        return None

    def schema_slots(self) -> List[Tuple[str, str]]:
        return [(self.meta_slot(), _OPAQUE_VALUE_BUNDLE_TYPE_NAME)]

    def to_slots(self, owner: Any) -> Dict[str, Any]:
        return {
            self.meta_slot(): OpaqueValueBundle({self.QUANTIZER_KEY: getattr(owner, self.name)})
        }

    def from_slots(self, args: Dict[str, Any], kwargs: Dict[str, Any]) -> None:
        kwargs[self.name] = args[self.meta_slot()][self.QUANTIZER_KEY]


class _ProcessGroupAdapter(_Adapter):
    """``ProcessGroup`` -> its c10d registry name in one ``OpaqueValueBundle`` slot.

    Mirrors traceable functional collectives: the graph carries the group's
    *name* (a plain string, so guards and the FX cache key are trivial) and the
    live group is re-resolved from the registry inside the op, in the same
    process -- ``from_slots(to_slots(pg))`` returns the very group the caller
    passed. Groups created outside the c10d registry fail the resolve loudly.
    """

    NAME_KEY = "group_name"

    def __init__(self, name: str) -> None:
        self.name = name

    def meta_slot(self) -> str:
        """Group-name slot name."""
        return self.name + "__pg"

    @classmethod
    def try_build(cls, name: str, annot: Any) -> Optional["_ProcessGroupAdapter"]:
        if _PROCESS_GROUP_TYPE is None:
            return None
        stripped, _ = _strip_optional(annot)
        if stripped is _PROCESS_GROUP_TYPE:
            return cls(name)
        return None

    def schema_slots(self) -> List[Tuple[str, str]]:
        return [(self.meta_slot(), _OPAQUE_VALUE_BUNDLE_TYPE_NAME)]

    def to_slots(self, owner: Any) -> Dict[str, Any]:
        pg = getattr(owner, self.name)
        name = None if pg is None else pg.group_name
        return {self.meta_slot(): OpaqueValueBundle({self.NAME_KEY: name})}

    def from_slots(self, args: Dict[str, Any], kwargs: Dict[str, Any]) -> None:
        name = args[self.meta_slot()][self.NAME_KEY]
        kwargs[self.name] = None if name is None else _resolve_process_group(name)


class _SimpleBundleAdapter(_Adapter):
    """Aggregates every simple-typed field into a single OpaqueValueBundle.

    Unlike the per-field adapters, at most one of these exists per op (none if the
    dataclass has no simple-typed fields): it owns the single shared
    ``_simple_meta`` slot, and ``_get_adapters`` builds it once from all
    simple-typed field names collected across the dataclass.
    """

    META_SLOT = "_simple_meta"

    def __init__(self, names: List[str]) -> None:
        self.names = list(names)

    @classmethod
    def matches_field(cls, annot: Any) -> bool:
        """Whether ``annot`` (Optional-aware, recursive) is bundle-simple."""
        annot, _ = _strip_optional(annot)
        if annot in OpaqueValueBundle.PRIMITIVE_TYPES:
            return True
        if isinstance(annot, type) and issubclass(annot, Enum):
            return True
        if (
            isinstance(annot, type)
            and _is_opaque_value_type is not None
            and _is_opaque_value_type(annot)
        ):
            return True
        if get_origin(annot) in (tuple, list):
            inner = [a for a in get_args(annot) if a is not Ellipsis]
            return bool(inner) and all(cls.matches_field(a) for a in inner)
        return False

    def schema_slots(self) -> List[Tuple[str, str]]:
        return [(self.META_SLOT, _OPAQUE_VALUE_BUNDLE_TYPE_NAME)]

    def to_slots(self, owner: Any) -> Dict[str, Any]:
        return {self.META_SLOT: OpaqueValueBundle({n: getattr(owner, n) for n in self.names})}

    def from_slots(self, args: Dict[str, Any], kwargs: Dict[str, Any]) -> None:
        if self.META_SLOT not in args:
            return
        meta = args[self.META_SLOT]
        for n in self.names:
            kwargs[n] = meta[n]


class _UnsupportedAdapter(_Adapter):
    """Fallback for fields whose type no other adapter can encode.

    Such a field cannot cross the op boundary, so it emits no slot and is
    tolerated only when its runtime value carries nothing: ``to_slots`` accepts
    ``None`` / an all-``None`` sequence (e.g. an unset ``Optional[Any]`` field,
    or an empty list, on the compiled path) and ``from_slots`` restores it as
    ``None``. A non-trivial value means the config is genuinely unsupported
    under torch.compile, and ``to_slots`` raises.

    The check must run at call time (not in ``_get_adapters``): the annotation
    alone -- e.g. ``Optional[Any]`` -- is valid when the value is ``None``, so
    only the runtime value can decide.
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

    def to_slots(self, owner: Any) -> Dict[str, Any]:
        value = getattr(owner, self.name, None)
        if not self._is_trivial(value):
            raise TypeError(
                f"{self.owner_cls_name} field {self.name!r} has a type not "
                "supported by torch.compile (not Tensor, simple, Quantizer, or "
                "ProcessGroup) and carries a "
                "non-trivial value; add a matching adapter in custom_op.py to handle it."
            )
        return {}

    def from_slots(self, args: Dict[str, Any], kwargs: Dict[str, Any]) -> None:
        kwargs[self.name] = None


# Adapter candidates for a single field, tried via ``try_build``. Mutually
# exclusive on annotations, so the order is not a ranking.
_FIELD_ADAPTERS: Tuple[type, ...] = (
    _TensorOrQuantizedAdapter,
    _TensorAdapter,
    _ProcessGroupAdapter,
    _QuantizerAdapter,
)


def _resolved_field_annotations(cls: type) -> List[Tuple[str, Any]]:
    """Return ``[(field_name, resolved_type), ...]`` for a dataclass."""
    if not dataclasses.is_dataclass(cls):
        raise TypeError(f"{cls.__name__} must be a @dataclass to be a TE op arg container.")
    try:
        hints = get_type_hints(cls)
    except Exception:  # pylint: disable=broad-exception-caught
        hints = {}
    return [(f.name, hints.get(f.name, f.type)) for f in dataclasses.fields(cls)]


def _get_adapters(cls: type) -> List[_Adapter]:
    """Build the adapter list for a dataclass from its field annotations."""
    if _OPAQUE_VALUE_BUNDLE_TYPE_NAME is None:
        raise RuntimeError(
            f"{cls.__name__} cannot be turned into a TE custom op: OpaqueValueBundle "
            "is not registered as a torch._library value-opaque type (PyTorch build "
            "without opaque-object support)."
        )
    adapters: List[_Adapter] = []
    simple_names: List[str] = []
    for name, annot in _resolved_field_annotations(cls):
        built: Optional[_Adapter] = None
        for adapter_cls in _FIELD_ADAPTERS:
            built = adapter_cls.try_build(name, annot)
            if built is not None:
                break
        if built is not None:
            adapters.append(built)
        elif _SimpleBundleAdapter.matches_field(annot):
            simple_names.append(name)
        else:
            adapters.append(_UnsupportedAdapter(name, cls.__name__))
    if simple_names:
        adapters.append(_SimpleBundleAdapter(simple_names))
    return adapters


def _tensor_field_names(adapters: List[_Adapter]) -> List[str]:
    """Names of fields carrying tensors (for building the spec view)."""
    return [b.name for b in adapters if isinstance(b, (_TensorAdapter, _TensorOrQuantizedAdapter))]


def _build_schema(adapters: List[_Adapter]) -> Tuple[str, List[str]]:
    """Return ``(schema_arg_str, slot_names)`` for an adapter list."""
    spec = [slot for b in adapters for slot in b.schema_slots()]
    names = [name for name, _ in spec]
    schema_str = "(" + ", ".join(f"{type_str} {name}" for name, type_str in spec) + ")"
    return schema_str, names


def _args_to_slots(obj: Any, adapters: List[_Adapter]) -> Dict[str, Any]:
    """Build the op's flat ``{slot_name: value}`` argument dict from an args
    dataclass ``obj`` (e.g. ``LinearFwdArgs``), by collecting every adapter's
    packed slot(s). Inverse of :func:`_args_from_slots`.
    """
    out: Dict[str, Any] = {}
    for adapter in adapters:
        out.update(adapter.to_slots(obj))
    return out


def _args_from_slots(cls: type, args: Dict[str, Any], adapters: List[_Adapter]) -> Any:
    """Rebuild a fresh args dataclass ``cls`` (e.g. ``LinearFwdArgs``) from the
    op's flat slot ``args`` dict, by letting every adapter restore its field(s).
    Inverse of :func:`_args_to_slots`.
    """
    kwargs: Dict[str, Any] = {}
    for adapter in adapters:
        adapter.from_slots(args, kwargs)
    obj = cls.__new__(cls)
    for k, v in kwargs.items():
        object.__setattr__(obj, k, v)
    return obj


def _spec_view(obj: Any, tensor_field_names: Sequence[str]) -> Any:
    """Copy of dataclass ``obj`` with each tensor field replaced by a :class:`TensorSpec`.

    Only tensor fields have a ``TensorSpec`` equivalent, so quantizer / scalar
    fields are simply carried over unchanged; the fake impl works purely on
    geometry. Built with :func:`dataclasses.replace` (the only such construction
    Dynamo can trace).
    """
    overrides: Dict[str, Any] = {}
    for name in tensor_field_names:
        value = getattr(obj, name, None)
        if value is not None and not isinstance(value, TensorSpec):
            overrides[name] = to_tensor_spec(value)
    if not overrides:
        return obj
    return dataclasses.replace(obj, **overrides)


# --------------------------------------------------------------------------- #
# Op outputs <-> flat ``Tensor[]`` payload: this is how an op returns / saves
# quantized tensors (and wrapper subclasses). Outputs are flattened to their
# inner buffers on the way out and rebuilt via ``__tensor_unflatten__`` on the
# way back; on the fake side a TensorSpec supplies the geometry.
# --------------------------------------------------------------------------- #


def _spec_slot_count(spec: Optional[TensorSpec]) -> int:
    """Flat ``Tensor[]`` slots the value for ``spec`` occupies."""
    if spec is None:
        return 1
    return len(spec.inner_names())


def _unflatten_values(
    specs: Sequence[Optional[TensorSpec]],
    flat: Sequence[Optional[torch.Tensor]],
    cursor: int = 0,
) -> Tuple[List[Any], int]:
    """Rebuild one group of values from an op's flat return, starting at ``cursor``.

    Returns the values and the new cursor, so consecutive groups (user outputs,
    then saved tensors) can walk the same payload.
    """
    values: List[Any] = []
    for spec in specs:
        n = _spec_slot_count(spec)
        chunk = [_decode_none(t) for t in flat[cursor : cursor + n]]
        cursor += n
        # ``spec is None`` is the op-boundary sentinel for an absent output.
        values.append(spec.assemble(chunk) if spec is not None else None)
    return values, cursor


def _flatten_value(
    value: Optional[Union[torch.Tensor, QuantizedTensorStorage, TensorSpec]],
) -> List[torch.Tensor]:
    """Return the flat ``Tensor[]`` slots that represent one op output ``value``.

    Inverse of :func:`_unflatten_values`; the slot count matches
    :func:`_spec_slot_count`.
    """
    if value is None:
        return [_encode_none(None)]
    if isinstance(value, TensorSpec):
        return [_encode_none(t) for t in value.create_inner_tensors()]
    if hasattr(value, "__tensor_flatten__"):
        inner_names, _ = value.__tensor_flatten__()
        return [_encode_none(getattr(value, n)) for n in inner_names]
    if isinstance(value, torch.Tensor):
        return [_encode_none(value)]
    raise TypeError(
        f"unsupported value type {type(value).__name__}; expected None / "
        "torch.Tensor / tensor subclass / bare storage / TensorSpec."
    )


# Trailing slots in every fwd-impl return: ``tensors_to_save, ctx_attrs``.
# User-output count is ``len(result) - this``.
_FWD_TRAILING_SLOTS = 2


def _check_fwd_result(result: Any) -> None:
    """Validate a fwd-impl return against the
    ``(*user_outputs, tensors_to_save, ctx_attrs)`` contract, with a clear
    message for op authors (user-output *types* are checked later, by
    :func:`_flatten_value`).

    Only called on the fake path (:func:`_unpack_fwd_fake_result`), which runs at
    trace/compile time -- so this is a compile-time check with no per-call cost.
    The real impl must return the same shape as the fake, so validating the fake
    covers both.
    """
    if not isinstance(result, tuple) or len(result) < _FWD_TRAILING_SLOTS:
        raise TypeError(
            f"fwd impl must return a tuple of >= {_FWD_TRAILING_SLOTS} elements "
            "(*user_outputs, tensors_to_save, ctx_attrs); "
            f"got {type(result).__name__}"
        )
    tensors_to_save, ctx_attrs = result[-2], result[-1]
    if tensors_to_save is not None and not isinstance(tensors_to_save, (list, tuple)):
        raise TypeError("fwd impl 'tensors_to_save' slot must be a list/tuple or None")
    if ctx_attrs is not None and not isinstance(ctx_attrs, dict):
        raise TypeError("fwd impl 'ctx_attrs' slot must be a dict or None")


def _pack_fwd_result(result: Any) -> List[torch.Tensor]:
    """Pack a fwd-impl return tuple into the op's ``Tensor[]`` payload.

    User outputs first, then saved-for-backward tensors in declaration order.
    """
    num_outputs = len(result) - _FWD_TRAILING_SLOTS
    flat: List[torch.Tensor] = []
    for value in result[:num_outputs]:
        flat.extend(_flatten_value(value))
    saved = result[num_outputs]
    if saved is not None:
        for value in saved:
            flat.extend(_flatten_value(value))
    return flat


def _pack_bwd_result(grads: Any, num_grad_inputs: int, op_qualname: str) -> List[torch.Tensor]:
    """Pack a backward-impl return tuple into the op's ``Tensor[]`` payload.

    Each grad occupies exactly one slot (validated against ``num_grad_inputs``);
    a :class:`TensorSpec` grad is materialized into a single tensor.
    """
    grads = list(grads)
    if len(grads) != num_grad_inputs:
        raise RuntimeError(
            f"{op_qualname} expected bwd_impl to return {num_grad_inputs} grads "
            f"(one per input_tensors_for_grad entry), got {len(grads)}"
        )
    out: List[torch.Tensor] = []
    for g in grads:
        if isinstance(g, TensorSpec):
            out.append(_encode_none(g.create_tensor()))
        else:
            out.append(_encode_none(g))
    return out


def _unpack_fwd_fake_result(
    result: Tuple[Any, ...],
) -> Tuple[List[Any], List[Any], Dict[str, Any]]:
    """Slice a fwd fake-impl return into ``(user_fakes, saved_fakes, ctx_attrs)``."""
    _check_fwd_result(result)
    num_outputs = len(result) - _FWD_TRAILING_SLOTS
    saved = result[num_outputs]
    ctx_attrs = result[num_outputs + 1]
    user_fakes = list(result[:num_outputs])
    saved_fakes = list(saved) if saved is not None else []
    ctx_attrs = dict(ctx_attrs) if ctx_attrs else {}
    return user_fakes, saved_fakes, ctx_attrs


# --------------------------------------------------------------------------- #
# Op registration
# --------------------------------------------------------------------------- #


def _resolve_grad_targets(
    fwd_adapters: List[_Adapter],
    input_tensors_for_grad: List[str],
) -> Tuple[int, List[int]]:
    """Validate ``input_tensors_for_grad`` and resolve the grad-output layout.

    ``fwd_adapters`` already encode the arg dataclass's fields (they are built
    from it), so the type itself is not needed here.

    Returns ``(slot_count, grad_targets)``: the total number of input schema
    slots and, for each requested input name, the schema-slot index its gradient
    maps to.
    """
    name_to_slot: Dict[str, int] = {}
    slot_offset = 0
    for adapter in fwd_adapters:
        slots = adapter.schema_slots()
        grad_slot = adapter.grad_slot()
        if grad_slot is not None:
            name_to_slot[adapter.name] = slot_offset + grad_slot
        slot_offset += len(slots)

    non_differentiable = [n for n in input_tensors_for_grad if n not in name_to_slot]
    if non_differentiable:
        raise ValueError(
            f"input_tensors_for_grad contains non-differentiable fields: {non_differentiable}"
        )
    grad_targets = [name_to_slot[n] for n in input_tensors_for_grad]
    return slot_offset, grad_targets


def _register_base_op(
    *,
    op_name: str,
    schema_str: str,
    arg_type: type,
    arg_names: List[str],
    adapters: List[_Adapter],
    tensor_field_names: List[str],
    impl: Callable[[Any], Any],
    fake_impl: Callable[[Any], Any],
    pack_result: Callable[[Any], List[torch.Tensor]],
) -> Any:
    """Define the op via ``torch.library.custom_op`` with the real ``impl`` + the
    ``fake_impl`` (spec), returning the ``CustomOpDef``.

    The real kernel rebuilds the dataclass and runs ``impl``; the fake kernel
    runs the spec fake impl on the :func:`_spec_view`. Both go through
    ``pack_result``.
    """

    def _impl(*flat: Any) -> List[torch.Tensor]:
        kwargs = dict(zip(arg_names, flat))
        obj = _args_from_slots(arg_type, kwargs, adapters)
        return pack_result(impl(obj))

    def _fake(*flat: Any) -> List[torch.Tensor]:
        kwargs = dict(zip(arg_names, flat))
        obj = _args_from_slots(arg_type, kwargs, adapters)
        spec_obj = _spec_view(obj, tensor_field_names)
        return pack_result(fake_impl(spec_obj))

    op = torch.library.custom_op(
        f"{_TE_OP_NAMESPACE}::{op_name}", _impl, mutates_args=(), schema=schema_str
    )
    op.register_fake(_fake)
    return op


def _register_autograd_for_op(
    *,
    fwd_op: Any,
    bwd_op: Any,
    fwd_arg_type: type,
    fwd_arg_names: List[str],
    fwd_adapters: List[_Adapter],
    fwd_tensor_field_names: List[str],
    bwd_arg_names: List[str],
    bwd_adapters: List[_Adapter],
    slot_count: int,
    grad_targets: List[int],
    setup_context_user: Callable[..., Any],
    bwd_arg_type: type,
    fwd_fake_impl: Callable[[Any], Tuple[Any, ...]],
) -> None:
    """Wire ``register_autograd`` on a forward op so its backward calls ``bwd_op``.

    ``setup_context`` re-runs the spec fwd fake impl to recover output / saved
    templates, reassembles each flat output chunk, and hands the saved tuple +
    ``ctx_attrs`` to the module's ``setup_context``.
    """

    def _setup_context(ctx, inputs, output):
        ctx.fwd_tensor_list_lengths = {
            i: len(value) for i, value in enumerate(inputs) if isinstance(value, list)
        }
        kwargs = dict(zip(fwd_arg_names, inputs))
        fwd_obj = _args_from_slots(fwd_arg_type, kwargs, fwd_adapters)
        spec_obj = _spec_view(fwd_obj, fwd_tensor_field_names)

        user_fakes, saved_fakes, ctx_attrs = _unpack_fwd_fake_result(fwd_fake_impl(spec_obj))

        user_outputs, cursor = _unflatten_values(user_fakes, output)
        saved_list, _ = _unflatten_values(saved_fakes, output, cursor)

        bwd_obj = bwd_arg_type()
        tensors_to_save_from_setup = setup_context_user(
            bwd_obj,
            fwd_obj,
            user_outputs[0] if len(user_outputs) == 1 else tuple(user_outputs),
            ctx_attrs,
            tuple(saved_list),
        )
        tensors_to_save, tensor_objects = prepare_for_saving(*(tensors_to_save_from_setup or ()))
        ctx.tensor_objects = tensor_objects
        ctx.save_for_backward(*tensors_to_save)
        ctx.backward_objects = bwd_obj

    def _autograd_backward(ctx, *grad_outputs):
        bwd_obj = ctx.backward_objects
        if hasattr(bwd_obj, "setup_saved_tensors"):
            bwd_obj.setup_saved_tensors(ctx)
        ctx.tensor_objects = None
        flat_grads = grad_outputs[0]
        bwd_obj.grad_output = _decode_none(flat_grads[0])
        kwargs = _args_to_slots(bwd_obj, bwd_adapters)
        bwd_args_flat = [kwargs[name] for name in bwd_arg_names]
        grads = [_decode_none(g) for g in bwd_op(*bwd_args_flat)]
        ctx.backward_objects = None
        # One grad per input schema slot: default None, but a ``Tensor[]`` slot
        # (always recorded in ``fwd_tensor_list_lengths``) needs a
        # list-shaped no-grad of matching length.
        out: List[Any] = [None] * slot_count
        for pos, length in ctx.fwd_tensor_list_lengths.items():
            out[pos] = [None] * length
        for pos, g in zip(grad_targets, grads):
            out[pos] = g
        return tuple(out)

    fwd_op.register_autograd(_autograd_backward, setup_context=_setup_context)


def _tensor_or_quantized_offsets(adapters: List[_Adapter]) -> List[int]:
    """Start index of each ``_TensorOrQuantizedAdapter`` group in the flat args."""
    offsets: List[int] = []
    pos = 0
    for adapter in adapters:
        if isinstance(adapter, _TensorOrQuantizedAdapter):
            offsets.append(pos)
        pos += len(adapter.schema_slots())
    return offsets


def _flatten_subclass_into_slots(
    new_args: List[Any], slot_offsets: List[int], subclass: type
) -> None:
    """Rewrite each tensor-or-quantized-adapter group whose ``Tensor?`` slot holds an
    instance of ``subclass`` into the storage layout (3 slots: name / tensors / meta).
    """
    for offset in slot_offsets:
        val = new_args[offset]
        if not isinstance(val, subclass):
            continue
        meta, tensors = _storage_flatten(
            val, {_TensorOrQuantizedAdapter.KIND_KEY: _TensorOrQuantizedKind.STORAGE}
        )
        new_args[offset] = None
        new_args[offset + 1] = tensors
        new_args[offset + 2] = meta


def _make_slot_forwarder(
    base_op: Any, slot_offsets: Sequence[int], subclasses: Sequence[type]
) -> Callable[[Sequence[Any]], List[torch.Tensor]]:
    """Return ``call(args)`` forwarding to ``base_op``, first flattening any
    ``subclasses`` instance sitting in the tensor-or-quantized slot groups at
    ``slot_offsets``.

    A ``torch.library`` op cannot take a tensor subclass directly, so the wrapper
    op body and its ``register_torch_dispatch`` rules all funnel through this one
    path -- see the two-tier op note in the module docstring. With no slots or no
    subclasses to flatten it is a plain pass-through.
    """
    enabled = bool(slot_offsets) and bool(subclasses)

    def call(args: Sequence[Any]) -> List[torch.Tensor]:
        if not enabled:
            return base_op(*args)
        new_args = list(args)
        for sub in subclasses:
            _flatten_subclass_into_slots(new_args, slot_offsets, sub)
        return base_op(*new_args)

    return call


def _make_dispatch_rule(
    forward: Callable[[Sequence[Any]], List[torch.Tensor]],
) -> Callable[..., Any]:
    """Adapt a slot forwarder to the ``register_torch_dispatch`` signature."""

    def _rule(mode, func, types, args, kwargs):
        del mode, func, types, kwargs
        return forward(args)

    return _rule


def _register_wrapper_op(
    *,
    wrapper_op_name: str,
    schema_str: str,
    base_op: Any,
    slot_offsets: Sequence[int] = (),
    subclasses: Sequence[type] = (),
) -> Any:
    """Define the wrapper op via ``torch.library.custom_op``: forward to the base
    op through :func:`_make_slot_forwarder`. Returns the ``CustomOpDef``.
    """
    forward = _make_slot_forwarder(base_op, slot_offsets, subclasses)

    def _forward(*flat: Any) -> List[torch.Tensor]:
        return forward(flat)

    op_def = torch.library.custom_op(
        f"{_TE_OP_NAMESPACE}::{wrapper_op_name}", _forward, mutates_args=(), schema=schema_str
    )
    op_def.register_fake(_forward)
    return op_def


def _all_quantized_tensor_subclasses() -> List[type]:
    """Return every imported ``QuantizedTensor`` wrapper subclass."""
    import transformer_engine.pytorch.tensor  # noqa: F401  pylint: disable=import-outside-toplevel,unused-import

    found: List[type] = []
    stack = list(QuantizedTensor.__subclasses__())
    while stack:
        cls = stack.pop()
        if cls not in found:
            found.append(cls)
            stack.extend(cls.__subclasses__())
    return found


def register_custom_op(
    *,
    op_name: str,
    input_tensors_for_grad: List[str],
    fwd_arg_type: type,
    fwd_impl: Callable[[Any], Any],
    setup_context: Callable[..., Any],
    bwd_arg_type: type,
    bwd_impl: Callable[[Any], Any],
    fwd_fake_impl: Callable[[Any], Tuple[Any, ...]],
    bwd_fake_impl: Callable[[Any], Tuple[Any, ...]],
) -> Optional[Callable[..., Any]]:
    """Register a TE module's forward + backward as torch custom ops.

    Returns ``forward_fn(fwd_arg_type_instance)`` -- a drop-in for
    ``Function.apply`` under ``torch.compiler.is_compiling()`` that dispatches
    through the op and returns the user-facing outputs.

    ``fwd_arg_type`` and ``bwd_arg_type`` are ``@dataclass``es whose *field
    annotations* define the op schema (see the module docstring for the
    field <-> slot mapping). The caller builds a ``fwd_arg_type`` instance and
    passes it to ``forward_fn``. ``input_tensors_for_grad`` lists the
    ``fwd_arg_type`` fields that receive gradients and fixes the backward grad
    order. ``bwd_arg_type`` is also instantiated by the framework
    (``bwd_arg_type()``), so it must be constructible with no arguments.

    Callable contracts:

    * ``fwd_impl(fwd_args) -> (*user_outputs, tensors_to_save, ctx_attrs)`` -- the
      real forward. ``user_outputs``: op outputs (tensor / quantized / ``None``);
      ``tensors_to_save``: list/tuple (or ``None``) of tensors for backward;
      ``ctx_attrs``: dict (or ``None``) of plain metadata for ``setup_context``.
      The trailing two slots are fixed (``_FWD_TRAILING_SLOTS``); everything
      before them is a user output.
    * ``fwd_fake_impl(fwd_args)`` -- data-free traceable twin of ``fwd_impl``:
      same return shape, but tensor outputs are :class:`TensorSpec`. Must match
      ``fwd_impl``'s shape (checked at compile time by ``_check_fwd_result``).
    * ``setup_context(bwd_obj, fwd_args, user_outputs, ctx_attrs, saved)
      -> tensors_to_save`` -- populate ``bwd_obj`` from forward state; return the
      tensors to persist across the boundary.
    * ``bwd_impl(bwd_args) -> grads`` -- exactly one grad per
      ``input_tensors_for_grad`` entry, in that order (``None`` for a
      non-differentiable input).
    * ``bwd_fake_impl(bwd_args)`` -- data-free twin of ``bwd_impl`` returning
      :class:`TensorSpec` grads.
    * ``bwd_arg_type.setup_saved_tensors(ctx)`` -- optional hook; skipped if
      absent.

    How the backward container is populated: ``setup_context`` fills the
    ``bwd_arg_type`` instance's non-tensor fields (quantizers, config) from
    forward state and returns the tensors to persist; the framework saves them
    via ``ctx.save_for_backward``. Before ``bwd_impl`` runs, the framework
    restores them into the container's tensor fields through the
    ``setup_saved_tensors`` hook and sets ``grad_output`` directly, so
    ``bwd_impl`` receives a fully-populated ``bwd_arg_type``.

    Registration touches experimental ``torch.library`` / opaque-object APIs
    that may be missing on older PyTorch. If it fails, this warns once and
    returns ``None`` instead of raising, so callers can fall back to eager under
    ``torch.compile`` (a graph break) rather than breaking import.
    """
    try:
        return _register_custom_op_impl(
            op_name=op_name,
            input_tensors_for_grad=input_tensors_for_grad,
            fwd_arg_type=fwd_arg_type,
            fwd_impl=fwd_impl,
            setup_context=setup_context,
            bwd_arg_type=bwd_arg_type,
            bwd_impl=bwd_impl,
            fwd_fake_impl=fwd_fake_impl,
            bwd_fake_impl=bwd_fake_impl,
        )
    except (ImportError, AttributeError, RuntimeError, TypeError) as e:
        record_compile_disabled(
            f"could not register the custom op '{op_name}' ({type(e).__name__}: {e})"
        )
        return None


def _register_custom_op_impl(
    *,
    op_name: str,
    input_tensors_for_grad: List[str],
    fwd_arg_type: type,
    fwd_impl: Callable[[Any], Any],
    setup_context: Callable[..., Any],
    bwd_arg_type: type,
    bwd_impl: Callable[[Any], Any],
    fwd_fake_impl: Callable[[Any], Tuple[Any, ...]],
    bwd_fake_impl: Callable[[Any], Tuple[Any, ...]],
) -> Callable[..., Any]:
    """Body of :func:`register_custom_op`; see it for semantics."""
    # Existence check at the API boundary: every ``input_tensors_for_grad`` name
    # must be an actual field of ``fwd_arg_type`` (differentiability -- whether
    # that field can carry a gradient -- is checked later in
    # :func:`_resolve_grad_targets`).
    fwd_field_names = {f.name for f in dataclasses.fields(fwd_arg_type)}
    missing = [n for n in input_tensors_for_grad if n not in fwd_field_names]
    if missing:
        raise ValueError(f"input_tensors_for_grad names not in {fwd_arg_type.__name__}: {missing}")

    wrapper_fwd_name = op_name
    wrapper_bwd_name = f"{op_name}_backward"
    base_fwd_name = f"{op_name}_base"
    base_bwd_name = f"{wrapper_bwd_name}_base"
    subclass_list = _all_quantized_tensor_subclasses()

    fwd_adapters = _get_adapters(fwd_arg_type)
    bwd_adapters = _get_adapters(bwd_arg_type)
    fwd_tensor_field_names = _tensor_field_names(fwd_adapters)
    bwd_tensor_field_names = _tensor_field_names(bwd_adapters)

    fwd_schema_args, fwd_arg_names = _build_schema(fwd_adapters)
    bwd_schema_args, bwd_arg_names = _build_schema(bwd_adapters)

    num_grad_inputs = len(input_tensors_for_grad)
    slot_count, grad_targets = _resolve_grad_targets(fwd_adapters, input_tensors_for_grad)

    fwd_schema = f"{fwd_schema_args} -> Tensor[]"
    bwd_schema = f"{bwd_schema_args} -> Tensor[]"

    base_bwd_qualname = f"{_TE_OP_NAMESPACE}::{base_bwd_name}"

    base_fwd_def = _register_base_op(
        op_name=base_fwd_name,
        schema_str=fwd_schema,
        arg_type=fwd_arg_type,
        arg_names=fwd_arg_names,
        adapters=fwd_adapters,
        tensor_field_names=fwd_tensor_field_names,
        impl=fwd_impl,
        fake_impl=fwd_fake_impl,
        pack_result=_pack_fwd_result,
    )
    _register_base_op(
        op_name=base_bwd_name,
        schema_str=bwd_schema,
        arg_type=bwd_arg_type,
        arg_names=bwd_arg_names,
        adapters=bwd_adapters,
        tensor_field_names=bwd_tensor_field_names,
        impl=bwd_impl,
        fake_impl=bwd_fake_impl,
        pack_result=lambda g: _pack_bwd_result(g, num_grad_inputs, base_bwd_qualname),
    )

    base_fwd_op = getattr(getattr(torch.ops, _TE_OP_NAMESPACE), base_fwd_name)
    base_bwd_op = getattr(getattr(torch.ops, _TE_OP_NAMESPACE), base_bwd_name)

    fwd_slot_offsets = _tensor_or_quantized_offsets(fwd_adapters)
    bwd_slot_offsets = _tensor_or_quantized_offsets(bwd_adapters)

    wrapper_fwd_def = _register_wrapper_op(
        wrapper_op_name=wrapper_fwd_name,
        schema_str=fwd_schema,
        base_op=base_fwd_op,
        slot_offsets=fwd_slot_offsets,
        subclasses=subclass_list,
    )
    # Pass-through: a subclass input reaches the base op through the dispatch
    # rule below, never through the wrapper body.
    wrapper_bwd_def = _register_wrapper_op(
        wrapper_op_name=wrapper_bwd_name, schema_str=bwd_schema, base_op=base_bwd_op
    )

    autograd_common = {
        "fwd_arg_type": fwd_arg_type,
        "fwd_arg_names": fwd_arg_names,
        "fwd_adapters": fwd_adapters,
        "fwd_tensor_field_names": fwd_tensor_field_names,
        "bwd_arg_names": bwd_arg_names,
        "bwd_adapters": bwd_adapters,
        "slot_count": slot_count,
        "grad_targets": grad_targets,
        "setup_context_user": setup_context,
        "bwd_arg_type": bwd_arg_type,
        "fwd_fake_impl": fwd_fake_impl,
    }
    wrapper_fwd_op = getattr(getattr(torch.ops, _TE_OP_NAMESPACE), wrapper_fwd_name)
    wrapper_bwd_op = getattr(getattr(torch.ops, _TE_OP_NAMESPACE), wrapper_bwd_name)

    _register_autograd_for_op(fwd_op=base_fwd_def, bwd_op=base_bwd_op, **autograd_common)
    _register_autograd_for_op(fwd_op=wrapper_fwd_def, bwd_op=wrapper_bwd_op, **autograd_common)

    _fwd_rule = _make_dispatch_rule(
        _make_slot_forwarder(base_fwd_op, fwd_slot_offsets, subclass_list)
    )
    _bwd_rule = _make_dispatch_rule(
        _make_slot_forwarder(base_bwd_op, bwd_slot_offsets, subclass_list)
    )

    for sub in subclass_list:
        wrapper_fwd_def.register_torch_dispatch(sub, _fwd_rule)
        wrapper_bwd_def.register_torch_dispatch(sub, _bwd_rule)

    _quantized_tensor_passthrough_ops.add(wrapper_fwd_op.default)
    _quantized_tensor_passthrough_ops.add(wrapper_bwd_op.default)
    _quantized_tensor_passthrough_ops.add(base_fwd_op.default)
    _quantized_tensor_passthrough_ops.add(base_bwd_op.default)

    def forward_fn(fwd_args):
        spec_obj = _spec_view(fwd_args, fwd_tensor_field_names)
        user_fakes, _saved_fakes, _ctx_attrs = _unpack_fwd_fake_result(fwd_fake_impl(spec_obj))
        kwargs = _args_to_slots(fwd_args, fwd_adapters)
        flat_in = [kwargs[name] for name in fwd_arg_names]
        result = wrapper_fwd_op(*flat_in)

        outputs, _ = _unflatten_values(user_fakes, result)
        if len(outputs) == 1:
            return outputs[0]
        return tuple(outputs)

    return forward_fn
