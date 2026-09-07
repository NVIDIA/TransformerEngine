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

Bridging the two takes three parts (below): a parsed per-op *arg plan* maps the
args dataclass onto the op's input slots; a per-trace *output plan*, parsed from
the data-free fake impl's result, maps the logical outputs / saved tensors /
grads onto ranges of the flat return; and a *two-tier op* lets a
quantized-tensor subclass be an op input.

Field <-> slot mapping. ``_parse_arg_type`` parses the dataclass's field
annotations once, at registration, into an immutable ``_ArgPlan``: per field a
``_FieldPlan`` (its ``_FieldKind`` plus the schema slots it occupies), and the
derived layout -- schema string, slot order, gradient placement and
tensor-or-quantized offsets. ``_ArgPlan.pack`` / ``unpack`` interpret the plan
on each call. The kinds -- and how each represents its field as op inputs:

  * ``TENSOR`` -- a plain ``Tensor`` / ``Optional[Tensor]``: one tensor slot.
  * ``TENSOR_OR_QUANTIZED`` -- a field that may be a plain tensor, a bare
    quantized storage, or ``None``: three slots (the tensor, its flat inner
    buffers, and a ``__kind__`` tag) so a quantized tensor crosses as its buffers.
  * ``SIMPLE`` -- every remaining simple value (scalars, enums, sizes,
    quantizers -- value-opaque constants baked into the graph -- and nested
    collections of them), gathered into one shared ``OpaqueValueBundle`` slot.
  * ``PROCESS_GROUP`` -- rides in the shared bundle too, as its c10d registry
    name, re-resolved inside the op.
  * ``UNSUPPORTED`` -- a field no kind can encode; emits no slot and is allowed
    only when its value is trivial (``None`` / all-``None``) at call time.

What runs where. Each op registers a data-free fake (``register_fake``) so it
traces under ``torch.compile`` without allocating. ``register_custom_op`` returns
``forward_fn`` -- the drop-in for the eager ``autograd.Function.apply``. A forward
call through it:

  * runs the fake ``fwd_fake_impl`` on ``TensorSpec`` descriptors (data-free; see
    ``tensor_spec.py``) and parses its result into an ``_OutputPlan`` -- the
    outputs' geometry and their ranges in the flat payload, in pure Python;
  * calls the *forward op* -- which runs the real ``fwd_impl`` -- for a flat
    ``Tensor[]`` payload;
  * rebuilds the structured user outputs from that payload per the plan
    (``_OutputPlan.user_outputs``; ``_flatten_value`` is the pack-side inverse).

Autograd, registered on the op, drives backward:

  * ``setup_context`` (run when the forward is taped) re-runs ``fwd_fake_impl``,
    parses the ``_OutputPlan``, reassembles the saved tensors from the op's flat
    output, then calls the user ``setup_context`` to fill the backward args from
    forward state + ``ctx_attrs`` (e.g. saved-tensor aliases) and return the
    tensors to persist; the plan's output ranges are stashed on ``ctx``;
  * on ``backward()`` the incoming flat grads are sliced per user output from the
    stashed plan (a ``grad_outputs`` field on the backward args receives the
    whole tuple; otherwise ``grad_output`` receives the first output's grad),
    the container's optional ``setup_saved_tensors`` hook restores the saved
    tensors, then the *backward op* runs the real ``bwd_impl`` and returns the
    flat grads (``bwd_fake_impl`` is its data-free fake).

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
import os
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


def _storage_unflatten(meta: "OpaqueValueBundle", tensors: List[torch.Tensor]) -> Any:
    """Inverse of :func:`_storage_flatten`."""
    meta_dict = meta.as_dict()
    inner_names = meta_dict["_inner_names"]
    inner = dict(zip(inner_names, tensors))
    outer_shape = meta_dict.get("_outer_shape")
    stride = make_contiguous_strides_for(tuple(outer_shape)) if outer_shape is not None else None
    return QuantizedTensorStorage.__tensor_unflatten__(inner, meta_dict, outer_shape, stride)


# --------------------------------------------------------------------------- #
# Arg plans: dataclass annotations are parsed once, at registration, into an
# immutable per-op plan (schema string, slot order, gradient placement,
# tensor-or-quantized offsets); packing / unpacking interpret that plan on
# each call.
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


class _FieldKind(Enum):
    """How one dataclass field crosses the custom-op boundary."""

    TENSOR = "tensor"  # one ``Tensor`` / ``Tensor?`` slot
    TENSOR_OR_QUANTIZED = "tensor_or_quantized"  # 3 slots: tensor / inner / meta
    PROCESS_GROUP = "process_group"  # c10d group name inside the shared bundle
    SIMPLE = "simple"  # value carried verbatim inside the shared bundle
    UNSUPPORTED = "unsupported"  # no slots; only a trivial value may cross


class _TensorOrQuantizedKind(Enum):
    """What a tensor-or-quantized slot group carries, tagged in its ``__meta``."""

    NONE = "none"
    TENSOR = "tensor"
    STORAGE = "storage"


_TQ_KIND_KEY = "__kind__"
_SIMPLE_META_SLOT = "_simple_meta"

# Matched by exact member set, so a bare quantized annotation or an accidental
# extra union member is rejected rather than silently taken as tensor-or-quantized.
_TQ_MEMBERS = frozenset(get_args(TensorOrQuantized))


@dataclasses.dataclass(frozen=True)
class _SlotSpec:
    """One schema slot: its name and torch.library type string."""

    name: str
    type_str: str


@dataclasses.dataclass(frozen=True)
class _FieldPlan:
    """Parsed record for one dataclass field.

    ``slots`` are the schema slots the field occupies (empty for the kinds that
    ride in the shared simple bundle, or cross nothing).
    """

    name: str
    kind: _FieldKind
    slots: Tuple[_SlotSpec, ...]


def _is_tensor_storage_union(annot: Any) -> bool:
    """Whether ``annot`` is exactly the tensor-or-quantized union."""
    if not _is_union(annot):
        return False
    members = frozenset(a for a in get_args(annot) if a is not type(None))
    return members == _TQ_MEMBERS


def _is_process_group_annot(annot: Any) -> bool:
    """Whether the field annotation is (Optional) ProcessGroup."""
    if _PROCESS_GROUP_TYPE is None:
        return False
    stripped, _ = _strip_optional(annot)
    return stripped is _PROCESS_GROUP_TYPE


def _is_simple_annot(annot: Any) -> bool:
    """Whether ``annot`` (Optional-aware, recursive) is bundle-simple."""
    annot, _ = _strip_optional(annot)
    if annot in OpaqueValueBundle.PRIMITIVE_TYPES:
        return True
    if isinstance(annot, type) and issubclass(annot, Enum):
        return True
    # Quantizers are value-opaque constants; the abstract ``Quantizer``
    # annotation itself is not a registered opaque type, so match by base.
    if isinstance(annot, type) and issubclass(annot, Quantizer):
        return True
    if (
        isinstance(annot, type)
        and _is_opaque_value_type is not None
        and _is_opaque_value_type(annot)
    ):
        return True
    if get_origin(annot) in (tuple, list):
        inner = [a for a in get_args(annot) if a is not Ellipsis]
        return bool(inner) and all(_is_simple_annot(a) for a in inner)
    return False


def _parse_field(name: str, annot: Any) -> _FieldPlan:
    """Parse one field's annotation into its :class:`_FieldPlan`."""
    if _is_tensor_storage_union(annot):
        slots = (
            _SlotSpec(name, "Tensor?"),
            _SlotSpec(name + "__tensors", "Tensor[]"),
            _SlotSpec(name + "__meta", _OPAQUE_VALUE_BUNDLE_TYPE_NAME),
        )
        return _FieldPlan(name, _FieldKind.TENSOR_OR_QUANTIZED, slots)
    stripped, is_optional = _strip_optional(annot)
    if stripped is torch.Tensor:
        slot = _SlotSpec(name, "Tensor?" if is_optional else "Tensor")
        return _FieldPlan(name, _FieldKind.TENSOR, (slot,))
    # A union mixing tensor types with anything else is a malformed signature
    # (e.g. a bare quantized-storage Optional, or Tensor | int): reject it at
    # registration instead of silently degrading to an unsupported field.
    if _is_union(annot):
        members = [a for a in get_args(annot) if a is not type(None)]
        if any(
            isinstance(m, type) and issubclass(m, (torch.Tensor, QuantizedTensorStorage))
            for m in members
        ):
            raise TypeError(
                f"field {name!r}: union {annot!r} is not a supported tensor "
                "signature; use Tensor, Optional[Tensor], or TensorOrQuantized."
            )
    if _is_process_group_annot(annot):
        return _FieldPlan(name, _FieldKind.PROCESS_GROUP, ())
    if _is_simple_annot(annot):
        return _FieldPlan(name, _FieldKind.SIMPLE, ())
    return _FieldPlan(name, _FieldKind.UNSUPPORTED, ())


def _is_trivial(value: Any) -> bool:
    """Whether an unsupported field's runtime value carries nothing."""
    if value is None:
        return True
    if isinstance(value, (list, tuple)):
        return all(v is None for v in value)
    return False


def _pack_tensor_or_quantized(field: _FieldPlan, value: Any, slots: Dict[str, Any]) -> None:
    """Fill a tensor-or-quantized field's three slots from its runtime value."""
    tensor_slot, inner_slot, meta_slot = (s.name for s in field.slots)
    if value is None:
        slots[tensor_slot] = None
        slots[inner_slot] = []
        slots[meta_slot] = OpaqueValueBundle({_TQ_KIND_KEY: _TensorOrQuantizedKind.NONE})
    elif isinstance(value, torch.Tensor):
        # Plain tensor *and* subclass (e.g. Float8Tensor) pass through the
        # ``Tensor?`` slot; subclass flattening (if any) is done by the
        # wrapper op's ``register_torch_dispatch`` rule.
        slots[tensor_slot] = value
        slots[inner_slot] = []
        slots[meta_slot] = OpaqueValueBundle({_TQ_KIND_KEY: _TensorOrQuantizedKind.TENSOR})
    elif isinstance(value, QuantizedTensorStorage):
        meta, tensors = _storage_flatten(value, {_TQ_KIND_KEY: _TensorOrQuantizedKind.STORAGE})
        slots[tensor_slot] = None
        slots[inner_slot] = tensors
        slots[meta_slot] = meta
    else:
        raise TypeError(
            f"field {field.name!r} expected None, torch.Tensor, or "
            f"QuantizedTensorStorage, got {type(value).__name__}"
        )


def _unpack_tensor_or_quantized(field: _FieldPlan, slots: Dict[str, Any]) -> Any:
    """Inverse of :func:`_pack_tensor_or_quantized`."""
    tensor_slot, inner_slot, meta_slot = (s.name for s in field.slots)
    meta = slots[meta_slot]
    kind = meta[_TQ_KIND_KEY]
    if kind == _TensorOrQuantizedKind.NONE:
        return None
    if kind == _TensorOrQuantizedKind.TENSOR:
        return slots[tensor_slot]
    return _storage_unflatten(meta, slots[inner_slot])


@dataclasses.dataclass(frozen=True)
class _ArgPlan:
    """The parsed plan for one args dataclass: the single source of truth for
    the schema string, slot order, packing / unpacking, gradient placement and
    the tensor-or-quantized slot offsets used for subclass flattening (the
    latter two derived from ``fields`` on demand -- registration-time only).

    Built once per registration by :func:`_parse_arg_type`; :meth:`pack` and
    :meth:`unpack` interpret it on each call. ProcessGroup fields ride in the
    shared bundle as their c10d registry *name* -- mirroring traceable
    functional collectives -- and the op re-resolves the very group the caller
    passed; groups created outside the c10d registry fail the resolve loudly.
    """

    arg_type: type
    fields: Tuple[_FieldPlan, ...]
    slot_names: Tuple[str, ...]
    schema_str: str

    def tensor_field_names(self) -> Tuple[str, ...]:
        """Names of the tensor-valued fields (the ones :func:`_spec_view` converts)."""
        return tuple(
            f.name
            for f in self.fields
            if f.kind in (_FieldKind.TENSOR, _FieldKind.TENSOR_OR_QUANTIZED)
        )

    def tensor_or_quantized_offsets(self) -> List[int]:
        """Start offset of each tensor-or-quantized slot group.

        Derived from ``fields`` on demand -- used once per registration, for
        the subclass-flattening dispatch wiring.
        """
        offsets: List[int] = []
        offset = 0
        for field in self.fields:
            if field.kind is _FieldKind.TENSOR_OR_QUANTIZED:
                offsets.append(offset)
            offset += len(field.slots)
        return offsets

    def resolve_grad_targets(self, input_tensors_for_grad: Sequence[str]) -> List[int]:
        """Absolute schema-slot index receiving each requested field's gradient.

        Derived from ``fields`` on demand -- called once per registration, so
        the plan doesn't cache the mapping.
        """
        index: Dict[str, int] = {}
        offset = 0
        for field in self.fields:
            if field.kind in (_FieldKind.TENSOR, _FieldKind.TENSOR_OR_QUANTIZED):
                # The gradient flows to the group's first slot -- for
                # tensor-or-quantized that is the ``Tensor?`` slot, the one
                # autograd sees the (subclass) tensor in.
                index[field.name] = offset
            offset += len(field.slots)
        non_differentiable = [n for n in input_tensors_for_grad if n not in index]
        if non_differentiable:
            raise ValueError(
                f"input_tensors_for_grad contains non-differentiable fields: {non_differentiable}"
            )
        return [index[n] for n in input_tensors_for_grad]

    def pack(self, obj: Any) -> Dict[str, Any]:
        """Flatten an ``arg_type`` instance into the op's ``{slot: value}`` dict.

        Inverse of :meth:`unpack`.
        """
        slots: Dict[str, Any] = {}
        simple: Dict[str, Any] = {}
        for field in self.fields:
            value = getattr(obj, field.name, None)
            match field.kind:
                case _FieldKind.TENSOR:
                    slots[field.slots[0].name] = value
                case _FieldKind.TENSOR_OR_QUANTIZED:
                    _pack_tensor_or_quantized(field, value, slots)
                case _FieldKind.PROCESS_GROUP:
                    simple[field.name] = None if value is None else value.group_name
                case _FieldKind.SIMPLE:
                    simple[field.name] = value
                case _FieldKind.UNSUPPORTED:
                    # Annotation alone (e.g. Optional[Any]) can't decide; only
                    # the runtime value can, so the check runs at pack time.
                    if not _is_trivial(value):
                        raise TypeError(
                            f"{self.arg_type.__name__} field {field.name!r} has a type not "
                            "supported by torch.compile (not Tensor, simple, Quantizer, or "
                            "ProcessGroup) and carries a "
                            "non-trivial value; add a matching field kind in custom_op.py "
                            "to handle it."
                        )
        # Non-empty exactly when the dataclass has SIMPLE / PROCESS_GROUP
        # fields, i.e. when the schema ends with the shared bundle slot.
        if simple:
            slots[_SIMPLE_META_SLOT] = OpaqueValueBundle(simple)
        return slots

    def unpack(self, slots: Dict[str, Any]) -> Any:
        """Rebuild a fresh ``arg_type`` instance from the op's flat slot dict.

        Inverse of :meth:`pack`.
        """
        kwargs: Dict[str, Any] = {}
        bundle = slots.get(_SIMPLE_META_SLOT)
        for field in self.fields:
            match field.kind:
                case _FieldKind.TENSOR:
                    kwargs[field.name] = slots[field.slots[0].name]
                case _FieldKind.TENSOR_OR_QUANTIZED:
                    kwargs[field.name] = _unpack_tensor_or_quantized(field, slots)
                case _FieldKind.PROCESS_GROUP:
                    if bundle is not None:
                        name = bundle[field.name]
                        kwargs[field.name] = None if name is None else _resolve_process_group(name)
                case _FieldKind.SIMPLE:
                    if bundle is not None:
                        kwargs[field.name] = bundle[field.name]
                case _FieldKind.UNSUPPORTED:
                    kwargs[field.name] = None
        obj = self.arg_type.__new__(self.arg_type)  # pylint: disable=no-value-for-parameter
        for k, v in kwargs.items():
            object.__setattr__(obj, k, v)
        return obj


def _resolved_field_annotations(cls: type) -> List[Tuple[str, Any]]:
    """Return ``[(field_name, resolved_type), ...]`` for a dataclass."""
    if not dataclasses.is_dataclass(cls):
        raise TypeError(f"{cls.__name__} must be a @dataclass to be a TE op arg container.")
    try:
        hints = get_type_hints(cls)
    except Exception:  # pylint: disable=broad-exception-caught
        hints = {}
    return [(f.name, hints.get(f.name, f.type)) for f in dataclasses.fields(cls)]


def _parse_arg_type(cls: type) -> _ArgPlan:
    """Parse an args ``@dataclass`` into its immutable :class:`_ArgPlan`.

    The layout pass assigns absolute slot positions (the shared simple bundle,
    if any field needs it, takes the last slot) and validates the result:
    duplicate slot names are rejected here, before any op is registered.
    """
    if _OPAQUE_VALUE_BUNDLE_TYPE_NAME is None:
        raise RuntimeError(
            f"{cls.__name__} cannot be turned into a TE custom op: OpaqueValueBundle "
            "is not registered as a torch._library value-opaque type (PyTorch build "
            "without opaque-object support)."
        )
    fields = tuple(_parse_field(name, annot) for name, annot in _resolved_field_annotations(cls))

    slot_specs: List[_SlotSpec] = []
    for field in fields:
        slot_specs.extend(field.slots)
    if any(f.kind in (_FieldKind.SIMPLE, _FieldKind.PROCESS_GROUP) for f in fields):
        slot_specs.append(_SlotSpec(_SIMPLE_META_SLOT, _OPAQUE_VALUE_BUNDLE_TYPE_NAME))

    slot_names = tuple(s.name for s in slot_specs)
    if len(set(slot_names)) != len(slot_names):
        dupes = sorted(n for n in set(slot_names) if slot_names.count(n) > 1)
        raise ValueError(f"{cls.__name__}: duplicate schema slot names: {dupes}")
    schema_str = "(" + ", ".join(f"{s.type_str} {s.name}" for s in slot_specs) + ")"

    return _ArgPlan(
        arg_type=cls,
        fields=fields,
        slot_names=slot_names,
        schema_str=schema_str,
    )


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


def _flatten_value(
    value: Optional[Union[torch.Tensor, QuantizedTensorStorage, TensorSpec]],
) -> List[torch.Tensor]:
    """Return the flat ``Tensor[]`` slots that represent one op output ``value``.

    Pack-side inverse of :meth:`_OutputPlan.user_outputs`; the slot count
    matches :func:`_spec_slot_count`.
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

    Only called on the fake path (:meth:`_OutputPlan.parse`), which runs at
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


@dataclasses.dataclass(frozen=True)
class _OutputPlan:
    """Per-trace layout of an op's flat ``Tensor[]`` return.

    Parsed from a fwd fake-impl result: the logical user outputs and the
    saved-for-backward tensors, each with its range in the flat payload. The
    single source of truth for rebuilding forward outputs and saved tensors and
    for slicing backward grad_outputs per user output. Per-trace rather than
    per-registration because a quantized output's inner-tensor count is only
    known from the fake result.
    """

    user_specs: Tuple[Optional[TensorSpec], ...]
    saved_specs: Tuple[Optional[TensorSpec], ...]
    ctx_attrs: Dict[str, Any]
    user_ranges: Tuple[Tuple[int, int], ...]
    saved_start: int

    @classmethod
    def parse(cls, result: Tuple[Any, ...]) -> "_OutputPlan":
        """Slice a fwd fake-impl return into the plan (validating the contract)."""
        _check_fwd_result(result)
        num_outputs = len(result) - _FWD_TRAILING_SLOTS
        user_specs = tuple(result[:num_outputs])
        saved = result[num_outputs]
        ctx_attrs = result[num_outputs + 1]
        cursor = 0
        user_ranges: List[Tuple[int, int]] = []
        for spec in user_specs:
            n = _spec_slot_count(spec)
            user_ranges.append((cursor, cursor + n))
            cursor += n
        return cls(
            user_specs=user_specs,
            saved_specs=tuple(saved) if saved is not None else (),
            ctx_attrs=dict(ctx_attrs) if ctx_attrs else {},
            user_ranges=tuple(user_ranges),
            saved_start=cursor,
        )

    @staticmethod
    def _assemble(
        spec: Optional[TensorSpec], flat: Sequence[Optional[torch.Tensor]], start: int, stop: int
    ) -> Any:
        chunk = [_decode_none(t) for t in flat[start:stop]]
        # ``spec is None`` is the op-boundary sentinel for an absent output.
        return spec.assemble(chunk) if spec is not None else None

    def user_outputs(self, flat: Sequence[Optional[torch.Tensor]]) -> List[Any]:
        """Rebuild the structured user outputs from the op's flat return."""
        return [
            self._assemble(spec, flat, start, stop)
            for spec, (start, stop) in zip(self.user_specs, self.user_ranges)
        ]

    def saved_tensors(self, flat: Sequence[Optional[torch.Tensor]]) -> List[Any]:
        """Rebuild the saved-for-backward tensors from the op's flat return."""
        values: List[Any] = []
        cursor = self.saved_start
        for spec in self.saved_specs:
            n = _spec_slot_count(spec)
            values.append(self._assemble(spec, flat, cursor, cursor + n))
            cursor += n
        return values


def _slice_user_grads(
    user_ranges: Tuple[Tuple[int, int], ...], flat_grads: Sequence[Optional[torch.Tensor]]
) -> List[Any]:
    """Gradient of each user output, sliced from the op's flat grad list.

    A single-slot output yields its tensor grad; a flattened quantized output
    yields the tuple of its inner-buffer grads. Takes the bare ranges (not the
    whole :class:`_OutputPlan`) so ``setup_context`` only has to stash those on
    ``ctx`` -- keeping the specs (and the quantizers they reference) off the
    autograd tape.
    """
    grads: List[Any] = []
    for start, stop in user_ranges:
        chunk = [_decode_none(g) for g in flat_grads[start:stop]]
        grads.append(chunk[0] if stop - start == 1 else tuple(chunk))
    return grads


# --------------------------------------------------------------------------- #
# Op registration
# --------------------------------------------------------------------------- #


def _mark_effectful(op_def: Any) -> None:
    """Protect the op from dead-code elimination.

    The real impls launch collectives and mutate persistent state (Userbuffers,
    amax history), so an op whose outputs are unused must still run -- a DCE'd
    collective desynchronizes ranks. Default ``fx`` mode registers the op in
    FX's side-effect registry; ``token`` additionally threads an ordered effect
    token through the graph (stronger -- also blocks reordering -- but
    incompatible with cudagraph trees on current torch); ``0`` disables.
    Controlled by ``NVTE_COMPILE_OP_SIDE_EFFECTS``.
    """
    mode = os.getenv("NVTE_COMPILE_OP_SIDE_EFFECTS", "fx")
    if mode == "fx":
        torch.fx.node.has_side_effect(op_def._opoverload)
    elif mode == "token":
        try:
            # pylint: disable=import-outside-toplevel
            from torch._higher_order_ops.effects import _register_effectful_op
            from torch._library.effects import EffectType
        except ImportError:
            return
        _register_effectful_op(op_def, EffectType.ORDERED)


def _register_base_op(
    *,
    op_name: str,
    schema_str: str,
    plan: _ArgPlan,
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
        obj = plan.unpack(dict(zip(plan.slot_names, flat)))
        return pack_result(impl(obj))

    def _fake(*flat: Any) -> List[torch.Tensor]:
        obj = plan.unpack(dict(zip(plan.slot_names, flat)))
        spec_obj = _spec_view(obj, plan.tensor_field_names())
        return pack_result(fake_impl(spec_obj))

    op = torch.library.custom_op(
        f"{_TE_OP_NAMESPACE}::{op_name}", _impl, mutates_args=(), schema=schema_str
    )
    op.register_fake(_fake)
    _mark_effectful(op)
    return op


def _register_autograd_for_op(
    *,
    fwd_op: Any,
    bwd_op: Any,
    fwd_plan: _ArgPlan,
    bwd_plan: _ArgPlan,
    grad_targets: List[int],
    setup_context_user: Callable[..., Any],
    fwd_fake_impl: Callable[[Any], Tuple[Any, ...]],
) -> None:
    """Wire ``register_autograd`` on a forward op so its backward calls ``bwd_op``.

    ``setup_context`` re-runs the spec fwd fake impl to parse the
    :class:`_OutputPlan`, reassembles the outputs / saved tensors from it, hands
    the saved tuple + ``ctx_attrs`` to the module's ``setup_context`` and stashes
    the plan on ``ctx`` so backward can slice its grads per user output.
    """
    bwd_takes_grad_tuple = any(f.name == "grad_outputs" for f in bwd_plan.fields)

    def _setup_context(ctx, inputs, output):
        ctx.fwd_tensor_list_lengths = {
            i: len(value) for i, value in enumerate(inputs) if isinstance(value, list)
        }
        fwd_obj = fwd_plan.unpack(dict(zip(fwd_plan.slot_names, inputs)))
        spec_obj = _spec_view(fwd_obj, fwd_plan.tensor_field_names())

        out_plan = _OutputPlan.parse(fwd_fake_impl(spec_obj))
        user_outputs = out_plan.user_outputs(output)
        saved_list = out_plan.saved_tensors(output)

        bwd_obj = bwd_plan.arg_type()
        tensors_to_save_from_setup = setup_context_user(
            bwd_obj,
            fwd_obj,
            user_outputs[0] if len(user_outputs) == 1 else tuple(user_outputs),
            out_plan.ctx_attrs,
            tuple(saved_list),
        )
        tensors_to_save, tensor_objects = prepare_for_saving(*(tensors_to_save_from_setup or ()))
        ctx.tensor_objects = tensor_objects
        ctx.save_for_backward(*tensors_to_save)
        ctx.backward_objects = bwd_obj
        ctx.output_ranges = out_plan.user_ranges
        # Input shapes for the grad slots (SymInt-safe on ctx): a bwd impl may
        # rederive shapes lossily (e.g. rank-1 inputs come back rank-2), so the
        # returned grads are viewed back to the true input shapes below.
        ctx.grad_input_shapes = {
            pos: inputs[pos].shape for pos in grad_targets if isinstance(inputs[pos], torch.Tensor)
        }

    def _autograd_backward(ctx, *grad_outputs):
        bwd_obj = ctx.backward_objects
        if hasattr(bwd_obj, "setup_saved_tensors"):
            bwd_obj.setup_saved_tensors(ctx)
        ctx.tensor_objects = None
        user_grads = _slice_user_grads(ctx.output_ranges, grad_outputs[0])
        ctx.output_ranges = None
        if bwd_takes_grad_tuple:
            bwd_obj.grad_outputs = tuple(user_grads)
        else:
            bwd_obj.grad_output = user_grads[0]
        kwargs = bwd_plan.pack(bwd_obj)
        bwd_args_flat = [kwargs[name] for name in bwd_plan.slot_names]
        grads = [_decode_none(g) for g in bwd_op(*bwd_args_flat)]
        ctx.backward_objects = None
        # One grad per input schema slot: default None, but a ``Tensor[]`` slot
        # (always recorded in ``fwd_tensor_list_lengths``) needs a
        # list-shaped no-grad of matching length.
        out: List[Any] = [None] * len(fwd_plan.slot_names)
        for pos, length in ctx.fwd_tensor_list_lengths.items():
            out[pos] = [None] * length
        for pos, g in zip(grad_targets, grads):
            if g is not None:
                shape = ctx.grad_input_shapes.get(pos)
                if shape is not None and g.shape != shape:
                    g = g.view(shape)
            out[pos] = g
        ctx.grad_input_shapes = None
        return tuple(out)

    fwd_op.register_autograd(_autograd_backward, setup_context=_setup_context)


def _flatten_subclass_into_slots(
    new_args: List[Any], slot_offsets: Sequence[int], subclass: type
) -> None:
    """Rewrite each tensor-or-quantized slot group whose ``Tensor?`` slot holds an
    instance of ``subclass`` into the storage layout (3 slots: name / tensors / meta).
    """
    for offset in slot_offsets:
        val = new_args[offset]
        if not isinstance(val, subclass):
            continue
        meta, tensors = _storage_flatten(val, {_TQ_KIND_KEY: _TensorOrQuantizedKind.STORAGE})
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
    _mark_effectful(op_def)
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
    ``setup_saved_tensors`` hook and sets the incoming gradient directly --
    into a ``grad_outputs`` field (tuple, one grad per user output) if
    ``bwd_arg_type`` declares one, else into ``grad_output`` (the first user
    output's grad) -- so ``bwd_impl`` receives a fully-populated
    ``bwd_arg_type``.

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
    # that field can carry a gradient -- is checked later, in
    # :meth:`_ArgPlan.resolve_grad_targets`).
    fwd_field_names = {f.name for f in dataclasses.fields(fwd_arg_type)}
    missing = [n for n in input_tensors_for_grad if n not in fwd_field_names]
    if missing:
        raise ValueError(f"input_tensors_for_grad names not in {fwd_arg_type.__name__}: {missing}")

    wrapper_fwd_name = op_name
    wrapper_bwd_name = f"{op_name}_backward"
    base_fwd_name = f"{op_name}_base"
    base_bwd_name = f"{wrapper_bwd_name}_base"
    subclass_list = _all_quantized_tensor_subclasses()

    fwd_plan = _parse_arg_type(fwd_arg_type)
    bwd_plan = _parse_arg_type(bwd_arg_type)

    num_grad_inputs = len(input_tensors_for_grad)
    grad_targets = fwd_plan.resolve_grad_targets(input_tensors_for_grad)

    fwd_schema = f"{fwd_plan.schema_str} -> Tensor[]"
    bwd_schema = f"{bwd_plan.schema_str} -> Tensor[]"

    base_bwd_qualname = f"{_TE_OP_NAMESPACE}::{base_bwd_name}"

    base_fwd_def = _register_base_op(
        op_name=base_fwd_name,
        schema_str=fwd_schema,
        plan=fwd_plan,
        impl=fwd_impl,
        fake_impl=fwd_fake_impl,
        pack_result=_pack_fwd_result,
    )
    _register_base_op(
        op_name=base_bwd_name,
        schema_str=bwd_schema,
        plan=bwd_plan,
        impl=bwd_impl,
        fake_impl=bwd_fake_impl,
        pack_result=lambda g: _pack_bwd_result(g, num_grad_inputs, base_bwd_qualname),
    )

    base_fwd_op = getattr(getattr(torch.ops, _TE_OP_NAMESPACE), base_fwd_name)
    base_bwd_op = getattr(getattr(torch.ops, _TE_OP_NAMESPACE), base_bwd_name)

    fwd_slot_offsets = fwd_plan.tensor_or_quantized_offsets()
    bwd_slot_offsets = bwd_plan.tensor_or_quantized_offsets()

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
        "fwd_plan": fwd_plan,
        "bwd_plan": bwd_plan,
        "grad_targets": grad_targets,
        "setup_context_user": setup_context,
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
        spec_obj = _spec_view(fwd_args, fwd_plan.tensor_field_names())
        out_plan = _OutputPlan.parse(fwd_fake_impl(spec_obj))
        kwargs = fwd_plan.pack(fwd_args)
        flat_in = [kwargs[name] for name in fwd_plan.slot_names]
        result = wrapper_fwd_op(*flat_in)

        outputs = out_plan.user_outputs(result)
        if len(outputs) == 1:
            return outputs[0]
        return tuple(outputs)

    return forward_fn
