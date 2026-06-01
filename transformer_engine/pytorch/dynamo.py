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
    Sequence,
    Tuple,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)

import torch


__all__ = [
    "OpaqueSimpleMetadata",
    "_te_register_custom_op",
]


_TE_OP_NAMESPACE = "transformer_engine_compile"
_TE_LIB = torch.library.Library(_TE_OP_NAMESPACE, "FRAGMENT")


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
# Output layout helpers
# --------------------------------------------------------------------------- #
#
# A user output of a TE custom op can be one of:
#   * ``None``                              -> 1 sentinel slot.
#   * plain :class:`torch.Tensor`           -> 1 slot.
#   * wrapper-subclass tensor with
#     ``__tensor_flatten__`` (e.g.
#     :class:`Float8Tensor`)                -> ``len(inner_names)`` slots.
#   * pure-Python class with
#     ``_torch_compile_flatten`` (e.g.
#     :class:`Float8TensorStorage`)         -> ``len(tensors)`` slots.
#
# At op-execution time, :func:`_format_fwd_result` splits each output via
# its flatten protocol and concatenates the inner plain tensors into the
# op's ``Tensor[]`` return.
#
# At call-site time (in :func:`forward_fn` / ``setup_context``), the layout
# for each output is read straight off the forward ``fake_impl``'s fake
# values, which double as reassembly templates (:func:`_template_slot_count`
# / :func:`_template_reassemble`).


def _contiguous_stride(shape: Sequence[int]) -> Tuple[int, ...]:
    """Row-major contiguous stride for ``shape``.

    Used to fill in the ``stride`` field expected by
    ``__tensor_unflatten__`` when rebuilding a wrapper subclass from a
    fake template (:func:`_template_reassemble`).
    """
    stride: List[int] = [1] * len(shape)
    for i in range(len(shape) - 2, -1, -1):
        stride[i] = stride[i + 1] * int(shape[i + 1])
    return tuple(stride)


# --------------------------------------------------------------------------- #
# Reassembly: rebuild user-facing objects from the op's flat ``Tensor[]``.
# --------------------------------------------------------------------------- #
#
# The forward ``fake_impl`` returns the op's outputs / saved tensors as fake
# values (``make_fake_empty`` wrappers / ``make_empty`` storages /
# ``torch.empty`` plains / aliased forward args / ``None``). Each fake value is
# itself a complete reassembly *template*: it says how many flat slots the real
# value occupies and how to rebuild it. :func:`_flatten_value_into` packs a
# value into slots; the two helpers below are its inverse (slot count +
# rebuild), reading straight off the fake template -- no separate descriptor
# object is materialised.


def _template_slot_count(template: Any, *, aliased: bool = False) -> int:
    """Flat ``Tensor[]`` slots the real value for ``template`` occupies.

    ``aliased`` arg / ``None`` -> 1 (an :func:`_encode_none` sentinel); a
    plain tensor -> 1; a ``make_fake_empty`` subclass -> ``len(inner_names)``
    (from its stamped plan); a storage -> ``len(_torch_compile_flatten())``.
    """
    if aliased or template is None:
        return 1
    if isinstance(template, torch.Tensor):
        plan = getattr(template, _TE_COMPILE_UNFLATTEN_PLAN, None)
        if plan is not None:
            inner_names, _ = plan
            return len(inner_names)
        return 1
    flatten = getattr(template, "_torch_compile_flatten", None)
    if flatten is not None:
        _, _, tensors = flatten()
        return len(tensors)
    raise TypeError(
        f"fake_impl produced an unsupported value of type {type(template).__name__}; "
        "expected None / torch.Tensor (plain or make_fake_empty subclass) / "
        "a storage exposing _torch_compile_flatten()."
    )


def _template_reassemble(
    template: Any,
    chunk: List[Any],
    *,
    with_autograd: bool = False,
    aliased: bool = False,
) -> Any:
    """Rebuild the user-facing value for ``template`` from real slots ``chunk``.

    Inverse of :func:`_flatten_value_into`, driven by the fake template: an
    ``aliased`` arg / ``None`` -> ``None`` (aliases are resolved by the
    caller's ``setup_context`` from the alias name); a plain tensor ->
    ``chunk[0]``; a ``make_fake_empty`` subclass -> ``__tensor_unflatten__``
    (routed through :class:`_ToSubclassFn` when ``with_autograd`` so the wrap
    stays on the autograd graph); a storage -> ``_torch_compile_do_unflatten``.
    """
    if aliased or template is None:
        return None
    if isinstance(template, torch.Tensor):
        plan = getattr(template, _TE_COMPILE_UNFLATTEN_PLAN, None)
        if plan is not None:
            inner_names, meta = plan
            shape = tuple(template.shape)
            stride = _contiguous_stride(shape)
            if with_autograd:
                result = _ToSubclassFn.apply(
                    type(template), inner_names, meta, shape, stride, *chunk
                )
            else:
                inner_dict = dict(zip(inner_names, chunk))
                result = type(template).__tensor_unflatten__(inner_dict, meta, shape, stride)
            # ``__tensor_unflatten__`` rebuilds with ``quantizer=None`` (the
            # snapshot can't carry a live ``ProcessGroup``); restore the live
            # quantizer the fake template stashed so the output keeps its
            # amax-reduction group.
            quantizer = getattr(template, "_te_compile_quantizer", None)
            if quantizer is not None:
                result._quantizer = quantizer
            return result
        return chunk[0]
    flatten = getattr(template, "_torch_compile_flatten", None)
    if flatten is not None:
        meta, pg, _ = flatten()
        real_tensors = [t for t in chunk if t is not None]
        return type(template)._torch_compile_do_unflatten(meta, pg, real_tensors)
    raise TypeError(
        f"fake_impl produced an unsupported value of type {type(template).__name__}; "
        "expected None / torch.Tensor (plain or make_fake_empty subclass) / "
        "a storage exposing _torch_compile_flatten()."
    )


def _split_fwd_fake_result(
    result: Tuple[Any, ...],
) -> Tuple[List[Any], List[Any], Dict[str, Any]]:
    """Slice a forward ``fake_impl`` return into ``(user_fakes, saved_fakes, ctx_attrs)``.

    ``result`` has the eager-impl tuple shape ``(*user_outputs,
    tensors_to_save, tensor_objects, ctx_attrs)``; the fake values double as
    reassembly templates for :func:`_template_slot_count` /
    :func:`_template_reassemble`.
    """
    num_outputs = len(result) - _FWD_TRAILING_SLOTS
    saved = result[num_outputs]
    ctx_attrs = result[num_outputs + 2]
    user_fakes = list(result[:num_outputs])
    saved_fakes = list(saved) if saved is not None else []
    ctx_attrs = dict(ctx_attrs) if ctx_attrs else {}
    return user_fakes, saved_fakes, ctx_attrs


# --------------------------------------------------------------------------- #
# ``fake_impl`` consumers.
#
# A module describes its forward op outputs directly as a ``fwd_fake_impl``
# that returns the same ``(*user_outputs, tensors_to_save, tensor_objects,
# ctx_attrs)`` tuple as the eager ``fwd_impl``, but built out of *fake*
# values:
#   * ``quantizer.make_fake_empty(...)`` -- Dynamo-safe quantized wrapper.
#   * ``quantizer.make_empty(...)``      -- quantized storage.
#   * ``torch.empty(...)``               -- plain tensor.
#   * the actual forward-arg tensor      -- an aliased saved slot.
#   * ``None``                           -- absent output / saved slot.
# These fake values are the single source of truth for the op's layout:
#   * ``forward_fn`` / ``setup_context`` reassemble the real flat ``Tensor[]``
#     using the fakes as templates (:func:`_template_slot_count` /
#     :func:`_template_reassemble`), resolving aliased saved slots via
#     :func:`_alias_name_for`.
#   * :func:`_fwd_register_fake_from_fake_impl` wires the same callable as the
#     op's ``register_fake`` (aliased saved slots nulled so the fake flat
#     ``Tensor[]`` layout matches the eager impl, which writes ``None`` for
#     aliases).
# The backward ``bwd_fake_impl`` is used directly as the backward
# ``register_fake`` -- backward grads never round-trip through the op
# payload, so no reassembly is needed.
# --------------------------------------------------------------------------- #

# Attribute stamped on ``make_fake_empty`` outputs carrying the
# ``(inner_names, meta)`` plan needed to rebuild the subclass via
# ``__tensor_unflatten__``. The adapter reads it back (as a Dynamo
# constant) instead of calling ``value.__tensor_flatten__()`` in-trace:
# a tensor method returning non-tensors graph-breaks under fullgraph,
# whereas a plain attribute read is inlined.
_TE_COMPILE_UNFLATTEN_PLAN = "_te_compile_unflatten_plan"


def _fwd_arg_alias_pairs(
    fwd_obj: Any, field_names: Sequence[str]
) -> List[Tuple[torch.Tensor, str]]:
    """Collect ``(tensor field value, field name)`` for a fwd-arg object.

    ``field_names`` is precomputed outside the trace (reading
    ``dataclasses.fields`` in-trace would graph-break on the class
    ``mappingproxy``); attribute access by name is inlined. Used to
    detect saved slots that alias a forward arg by identity (``is``).
    """
    pairs: List[Tuple[torch.Tensor, str]] = []
    for name in field_names:
        value = getattr(fwd_obj, name, None)
        if isinstance(value, torch.Tensor):
            pairs.append((value, name))
    return pairs


def _alias_name_for(value: Any, pairs: List[Tuple[torch.Tensor, str]]) -> Optional[str]:
    """Return the forward-arg name ``value`` aliases (by ``is``), else ``None``."""
    for tensor, name in pairs:
        if value is tensor:
            return name
    return None


def _fwd_register_fake_from_fake_impl(
    fwd_fake_impl: Callable[[Any], Tuple[Any, ...]],
    field_names: Sequence[str],
) -> Callable[[Any], Tuple[Any, ...]]:
    """Adapt a forward ``fake_impl`` into a ``register_fake`` kernel.

    The user's ``fake_impl`` returns the *actual* forward-arg tensor for
    aliased saved slots; the eager impl instead writes ``None`` there
    (the value rides along as a ctx alias, not through the op payload).
    Aliased saved slots are nulled here so the fake flat ``Tensor[]``
    layout stays identical to the eager impl.
    """

    def fwd_fake(fwd_obj: Any) -> Tuple[Any, ...]:
        result = fwd_fake_impl(fwd_obj)
        num_outputs = len(result) - _FWD_TRAILING_SLOTS
        user_outputs = result[:num_outputs]
        saved = result[num_outputs]
        if saved is None:
            tensors_to_save: Any = None
        else:
            pairs = _fwd_arg_alias_pairs(fwd_obj, field_names)
            tensors_to_save = tuple(
                None if _alias_name_for(v, pairs) is not None else v for v in saved
            )
        return (*user_outputs, tensors_to_save, None, None)

    return fwd_fake


class _ToSubclassFn(torch.autograd.Function):
    """Construct a wrapper-subclass tensor from its inner plain tensors,
    preserving autograd flow through ``__tensor_unflatten__``.

    Non-tensor args (``cls``, ``inner_names``, ``meta``, ``outer_shape``,
    ``outer_stride``) are static constants. Dynamo / AOT capture them as
    constants on the autograd.Function node; the variadic ``inner_tensors``
    are real / fake graph tensors emitted by the underlying custom op.
    """

    @staticmethod
    def forward(ctx, cls, inner_names, meta, outer_shape, outer_stride, *inner_tensors):
        """Reassemble ``cls`` from ``inner_tensors`` via ``__tensor_unflatten__``."""
        ctx.inner_names = inner_names
        ctx.num_inner = len(inner_tensors)
        inner_dict = dict(zip(inner_names, inner_tensors))
        return cls.__tensor_unflatten__(inner_dict, meta, outer_shape, outer_stride)

    @staticmethod
    def backward(ctx, grad_output):
        """Route ``grad_output`` back to its per-inner-name slots."""
        # Under AOTAutograd, ``grad_output`` typically arrives flattened
        # via the subclass machinery; under eager it may be the subclass
        # itself. Both paths support ``__tensor_flatten__``-driven routing.
        if grad_output is not None and hasattr(grad_output, "__tensor_flatten__"):
            names_in_grad, _ = grad_output.__tensor_flatten__()
            grad_by_name = {n: getattr(grad_output, n) for n in names_in_grad}
            grads = tuple(grad_by_name.get(n) for n in ctx.inner_names)
        else:
            # Fallback: route the lone grad to the first inner slot; the
            # remaining slots (typically derived quantities like scale)
            # get ``None``.
            grads = (grad_output,) + (None,) * (ctx.num_inner - 1)
        # First five returns correspond to the five leading non-tensor args
        # to ``forward`` (``cls``, ``inner_names``, ``meta``, ``shape``,
        # ``stride``); none of them carries a gradient.
        return (None, None, None, None, None) + grads


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
        return _is_opaque_value_type(type(value))

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

    def __init__(self, data: Optional[Dict[str, Any]] = None) -> None:
        data = dict(data) if data else {}
        cls = type(self)
        for k, v in data.items():
            if not cls.is_simple_value(v):
                raise TypeError(
                    f"OpaqueSimpleMetadata field '{k}' has unsupported "
                    f"type {type(v).__name__}; only simple primitives "
                    f"({', '.join(t.__name__ for t in cls.PRIMITIVE_TYPES)}, "
                    "Enum, torch.Size, registered torch.compile value-"
                    "opaque types) and tuples/lists thereof are allowed."
                )
        self._data: Dict[str, Any] = data
        self._frozen: Tuple[Tuple[str, Any], ...] = tuple(
            (k, cls._to_hashable(v)) for k, v in sorted(data.items())
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
        items = ", ".join(f"{k!r}: {cls._fmt_simple(v)}" for k, v in self._data.items())
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
        is_opaque_value_type as _is_opaque_value_type,
        register_opaque_type,
    )

    register_opaque_type(OpaqueSimpleMetadata, typ="value")
    _OPAQUE_SIMPLE_META_TYPE_NAME: Optional[str] = get_opaque_type_name(OpaqueSimpleMetadata)

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
    _is_opaque_value_type = None
    _OPAQUE_SIMPLE_META_TYPE_NAME = None
    _PROCESS_GROUP_TYPE_NAME = None


# --------------------------------------------------------------------------- #
# Field buckets
# --------------------------------------------------------------------------- #
#
# Each dataclass field is mapped to exactly one bucket that owns its
# schema slots and the pack/unpack logic between the dataclass attribute
# and the flat ``torch.library`` view. Concrete bucket types are defined
# below; the per-class docstrings describe what each one matches.


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

    def pack(self, owner: Any) -> List[Tuple[str, Any]]:
        """Return ``[(slot_name, value), ...]`` extracted from ``owner``."""
        raise NotImplementedError

    def unpack(self, args: Dict[str, Any], kwargs: Dict[str, Any]) -> None:
        """Read this field's slots from ``args`` and write the
        reconstructed dataclass attribute(s) into ``kwargs``."""
        raise NotImplementedError


class _MetaPGTensorsBucket(_Bucket):
    """Shared three-slot bucket emitting ``<name>__meta`` /
    ``<name>__pg`` / ``<name>__tensors``.

    Used by every field whose value must be carried as the triple
    ``(OpaqueSimpleMetadata, ProcessGroup?, Tensor[])`` -- today this
    covers ``Tensor | QuantizedTensorStorage`` unions (see
    :class:`_UniversalTensorBucket`) and ``Quantizer`` instances
    (see :class:`_FlattenableBucket`). Concrete subclasses
    implement :meth:`_pack_value` / :meth:`_unpack_value` for their
    flatten/unflatten protocol; the rest of the bucket contract is
    identical and lives here.
    """

    SUFFIX_META = "__meta"
    SUFFIX_PG = "__pg"
    SUFFIX_TENSORS = "__tensors"

    def __init__(self, name: str) -> None:
        if _OPAQUE_SIMPLE_META_TYPE_NAME is None or _PROCESS_GROUP_TYPE_NAME is None:
            raise RuntimeError(
                f"Field {name!r} requires both OpaqueSimpleMetadata and "
                "torch.distributed.ProcessGroup to be registered as "
                "torch._library opaque types; one or both are "
                "unavailable in this PyTorch build."
            )
        self.name = name

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

    def pack(self, owner: Any) -> List[Tuple[str, Any]]:
        value = getattr(owner, self.name)
        meta, pg, tensors = self._pack_value(value)
        return [
            (self._slot_meta(), meta),
            (self._slot_pg(), pg),
            (self._slot_tensors(), list(tensors)),
        ]

    def unpack(self, args: Dict[str, Any], kwargs: Dict[str, Any]) -> None:
        kwargs[self.name] = self._unpack_value(
            args[self._slot_meta()],
            args[self._slot_pg()],
            args[self._slot_tensors()],
        )

    def _pack_value(self, value: Any) -> Tuple[Any, Any, List[torch.Tensor]]:
        """Flatten one field value into ``(meta, pg, tensors)``."""
        raise NotImplementedError

    def _unpack_value(self, meta: Any, pg: Any, tensors: List[torch.Tensor]) -> Any:
        """Inverse of :meth:`_pack_value`."""
        raise NotImplementedError


class _UniversalTensorBucket(_Bucket):
    """``Tensor | QuantizedTensorStorage`` (also subclass-tensor) field.

    Emits four schema slots per field, regardless of the runtime value:

    * ``<name>`` (``Tensor?``)        -- plain tensor / subclass tensor
                                         (e.g. :class:`Float8Tensor`)
                                         passes through here untouched.
                                         ``None`` for the storage path.
    * ``<name>__tensors`` (``Tensor[]``) -- flat inner tensors when the
                                         value was carried through a
                                         flatten protocol (storage at
                                         pack-time, or a subclass that
                                         was dispatched into flat form
                                         by ``register_torch_dispatch``
                                         on the outer op).
    * ``<name>__pg`` (``ProcessGroup?``) -- distributed handle attached
                                         to the flatten metadata, if
                                         any.
    * ``<name>__meta`` (``OpaqueSimpleMetadata``) -- everything else:
                                         the storage / subclass meta
                                         dict, plus a ``__kind__``
                                         marker telling the unpacker
                                         which slot to look at:
                                         ``"none"``, ``"tensor"``, or
                                         ``"storage"`` (the latter
                                         covers both storage and any
                                         already-flattened subclass).

    Storage values are flattened at ``_pack`` time (callsite). Plain
    tensors -- including subclass instances -- are passed unchanged
    through ``<name>``; under ``torch.compile`` an outer-op
    ``register_torch_dispatch`` rule turns each registered subclass
    into the storage layout *between* outer and inner op so the
    autograd graph stays attached to the user-facing wrapper.
    """

    SUFFIX_TENSORS = "__tensors"
    SUFFIX_PG = "__pg"
    SUFFIX_META = "__meta"

    KIND_KEY = "__kind__"
    KIND_NONE = "none"
    KIND_TENSOR = "tensor"
    KIND_STORAGE = "storage"

    def __init__(self, name: str) -> None:
        if _OPAQUE_SIMPLE_META_TYPE_NAME is None or _PROCESS_GROUP_TYPE_NAME is None:
            raise RuntimeError(
                f"Field {name!r} requires both OpaqueSimpleMetadata and "
                "torch.distributed.ProcessGroup to be registered as "
                "torch._library opaque types; one or both are "
                "unavailable in this PyTorch build."
            )
        self.name = name

    def slot_name(self) -> str:
        return self.name

    def slot_tensors(self) -> str:
        return self.name + self.SUFFIX_TENSORS

    def slot_pg(self) -> str:
        return self.name + self.SUFFIX_PG

    def slot_meta(self) -> str:
        return self.name + self.SUFFIX_META

    def schema_slots(self) -> List[Tuple[str, str]]:
        return [
            (self.slot_name(), "Tensor?"),
            (self.slot_tensors(), "Tensor[]"),
            (self.slot_pg(), _PROCESS_GROUP_TYPE_NAME + "?"),
            (self.slot_meta(), _OPAQUE_SIMPLE_META_TYPE_NAME),
        ]

    @staticmethod
    def _is_tensor_storage_union(annot: Any) -> bool:
        origin = get_origin(annot)
        if origin is not Union:
            return False
        members = [a for a in get_args(annot) if a is not type(None)]
        if torch.Tensor not in members:
            return False
        qts = _quantized_tensor_storage_cls()
        if qts is None:
            return False
        return any(isinstance(member, type) and issubclass(member, qts) for member in members)

    @classmethod
    def try_build(cls, name: str, annot: Any) -> Optional["_UniversalTensorBucket"]:
        if cls._is_tensor_storage_union(annot):
            return cls(name)
        return None

    def pack(self, owner: Any) -> List[Tuple[str, Any]]:
        value = getattr(owner, self.name)
        if value is None:
            return [
                (self.slot_name(), None),
                (self.slot_tensors(), []),
                (self.slot_pg(), None),
                (self.slot_meta(), OpaqueSimpleMetadata({self.KIND_KEY: self.KIND_NONE})),
            ]
        # Plain ``torch.Tensor`` *and* any subclass (e.g. ``Float8Tensor``)
        # hit this branch first -- the wrapper is forwarded untouched
        # through the ``Tensor?`` slot so the autograd graph stays
        # attached to the user-facing tensor object. Subclass-specific
        # flattening (if any) happens later inside the outer op's
        # ``register_torch_dispatch`` rule.
        if isinstance(value, torch.Tensor):
            return [
                (self.slot_name(), value),
                (self.slot_tensors(), []),
                (self.slot_pg(), None),
                (self.slot_meta(), OpaqueSimpleMetadata({self.KIND_KEY: self.KIND_TENSOR})),
            ]
        qts = _quantized_tensor_storage_cls()
        if qts is not None and isinstance(value, qts):
            meta, pg, tensors = value._torch_compile_flatten()
            # Stamp the storage-flatten meta with our kind marker so the
            # unpacker can route by ``__kind__`` alone.
            meta._data[self.KIND_KEY] = self.KIND_STORAGE
            return [
                (self.slot_name(), None),
                (self.slot_tensors(), list(tensors)),
                (self.slot_pg(), pg),
                (self.slot_meta(), meta),
            ]
        raise TypeError(
            f"field {self.name!r} expected None, torch.Tensor, or "
            f"QuantizedTensorStorage, got {type(value).__name__}"
        )

    def unpack(self, args: Dict[str, Any], kwargs: Dict[str, Any]) -> None:
        meta = args[self.slot_meta()]
        kind = meta.get(self.KIND_KEY)
        if kind == self.KIND_NONE:
            kwargs[self.name] = None
            return
        if kind == self.KIND_TENSOR:
            kwargs[self.name] = args[self.slot_name()]
            return
        qts = _quantized_tensor_storage_cls()
        kwargs[self.name] = qts._torch_compile_unflatten(
            meta, args[self.slot_pg()], args[self.slot_tensors()]
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

    def pack(self, owner: Any) -> List[Tuple[str, Any]]:
        return [(self.name, getattr(owner, self.name))]

    def unpack(self, args: Dict[str, Any], kwargs: Dict[str, Any]) -> None:
        kwargs[self.name] = args[self.name]


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

    def pack(self, owner: Any) -> List[Tuple[str, Any]]:
        return [(self.name, getattr(owner, self.name))]

    def unpack(self, args: Dict[str, Any], kwargs: Dict[str, Any]) -> None:
        kwargs[self.name] = args[self.name]


# Cached resolutions of TE types that ``dynamo`` references lazily to
# avoid import cycles (they live in modules that themselves import this
# one). Each ``_*_cls`` getter resolves its target once and reuses the
# result on every subsequent call; the values are kept module-level
# rather than baked into bucket instances so the cache survives across
# different dataclass registrations.
_QTS_REF: Optional[type] = None
_QUANTIZER_REF: Optional[type] = None


def _quantized_tensor_storage_cls() -> Optional[type]:
    """Lazy-resolve :class:`QuantizedTensorStorage`; ``None`` if unavailable."""
    global _QTS_REF
    if _QTS_REF is None:
        try:
            from transformer_engine.pytorch.quantized_tensor import (
                QuantizedTensorStorage,
            )

            _QTS_REF = QuantizedTensorStorage
        except Exception:  # pragma: no cover - partial init
            return None
    return _QTS_REF


def _quantizer_cls() -> Optional[type]:
    """Lazy-resolve :class:`Quantizer`; ``None`` if unavailable."""
    global _QUANTIZER_REF
    if _QUANTIZER_REF is None:
        try:
            from transformer_engine.pytorch.quantized_tensor import Quantizer

            _QUANTIZER_REF = Quantizer
        except Exception:  # pragma: no cover - partial init
            return None
    return _QUANTIZER_REF


def _flattenable_bases() -> Tuple[type, ...]:
    """Return the list of base classes whose subclasses are routed
    through :class:`_FlattenableBucket`.

    A "flattenable" type implements the duck-typed pair

    * instance method ``_flatten() -> (OpaqueSimpleMetadata, ref, list[Tensor])``
    * classmethod ``_unflatten(meta, ref, tensors)`` (dispatches by an
      identifier stamped into ``meta``).
    """
    return tuple(
        cls for cls in (_quantizer_cls(), _quantized_tensor_storage_cls()) if cls is not None
    )


class _FlattenableBucket(_MetaPGTensorsBucket):
    """Field whose type implements the ``_flatten`` / ``_unflatten``
    protocol (see :func:`_flattenable_bases`). Used today for
    :class:`~transformer_engine.pytorch.quantized_tensor.Quantizer` and
    :class:`~transformer_engine.pytorch.quantized_tensor.QuantizedTensorStorage`.
    """

    # Stored under ``_qcls`` in the metadata bundle to encode ``None``
    # without making any of the three slots nullable.
    NONE_MARKER_KEY = "_qcls"
    NONE_MARKER_VAL = ""

    def __init__(self, name: str, base_cls: type) -> None:
        super().__init__(name)
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

    def _pack_value(self, value: Any) -> Tuple[Any, Any, List[torch.Tensor]]:
        if value is None:
            return (
                OpaqueSimpleMetadata({self.NONE_MARKER_KEY: self.NONE_MARKER_VAL}),
                None,
                [],
            )
        if hasattr(value, "_flatten"):
            return value._flatten()
        return value._torch_compile_flatten()

    def _unpack_value(self, meta: Any, pg: Any, tensors: List[torch.Tensor]) -> Any:
        if meta.get(self.NONE_MARKER_KEY) == self.NONE_MARKER_VAL:
            return None
        if hasattr(self.base_cls, "_unflatten"):
            return self.base_cls._unflatten(meta, pg, tensors)
        return self.base_cls._torch_compile_unflatten(meta, pg, tensors)


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
        if isinstance(annot, type) and _is_opaque_value_type(annot):
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

    def pack(self, owner: Any) -> List[Tuple[str, Any]]:
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

    Constructed directly by :func:`_get_buckets` (it has no
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

    def pack(self, owner: Any) -> List[Tuple[str, Any]]:
        value = getattr(owner, self.name, None)
        if not self._is_trivial(value):
            raise TypeError(
                f"{self.owner_cls_name} field {self.name!r} has a type not "
                "supported by torch.compile (not Tensor, simple, "
                "ProcessGroup, or Quantizer) and carries a non-trivial "
                "value; add a matching bucket in dynamo.py to handle it."
            )
        return []

    def unpack(self, args: Dict[str, Any], kwargs: Dict[str, Any]) -> None:
        kwargs[self.name] = None


# Buckets, in priority order, that own ``try_build`` for a single field.
_FIELD_BUCKETS: Tuple[type, ...] = (
    _UniversalTensorBucket,
    _TensorBucket,
    _ProcessGroupBucket,
    _FlattenableBucket,
)


# --------------------------------------------------------------------------- #
# Dataclass <-> torch.library plumbing
# --------------------------------------------------------------------------- #
#
# The helpers below translate a plain ``@dataclass`` argument container
# into the flat ``{slot_name: slot_value}`` view ``torch.library`` works
# with. Each dataclass field is dispatched (by annotation) to one
# :class:`_Bucket`; schema / pack / unpack are then loops over that list.


def _resolved_field_annotations(cls: type) -> List[Tuple[str, Any]]:
    """Return ``[(field_name, resolved_type), ...]`` for a dataclass."""
    if not dataclasses.is_dataclass(cls):
        raise TypeError(
            f"{cls.__name__} must be a @dataclass to be used as a TE custom-op argument container."
        )
    # ``get_type_hints`` resolves forward references and PEP 563
    # ``from __future__ import annotations`` strings.
    try:
        hints = get_type_hints(cls)
    except Exception:
        hints = {}
    return [(f.name, hints.get(f.name, f.type)) for f in dataclasses.fields(cls)]


def _get_buckets(cls: type) -> List[_Bucket]:
    """Build the bucket list for a dataclass from its field annotations.

    Dispatch order per field: try each bucket in :data:`_FIELD_BUCKETS`
    (Tensor, ProcessGroup, Quantizer); if none claims the field, route
    it to :class:`_SimpleBundleBucket` if its annotation is bundle-able,
    else to :class:`_UnknownBucket`.

    Intentionally **not** cached on ``cls``. Caching there (e.g. by
    writing ``cls.__te_buckets__``) tickles Dynamo: subsequent reads of
    ``cls.__dict__`` from a compiled function trigger
    "mappingproxy affected by dictionary mutation" graph breaks. Hot
    paths must instead capture the bucket list once at op registration
    time and pass it explicitly to :func:`_pack` / :func:`_unpack`.
    """
    buckets: List[_Bucket] = []
    simple_names: List[str] = []
    for name, annot in _resolved_field_annotations(cls):
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


def _build_schema(buckets: List[_Bucket]) -> Tuple[str, List[str]]:
    """Return ``(schema_str, slot_names)`` for a precomputed bucket list.

    ``schema_str`` is the parenthesised argument list (e.g.
    ``"(Tensor x, Tensor? y)"``) that ``torch.library.Library.define``
    appends to the op name; ``slot_names`` is the ordered list of slot
    keys produced by :func:`_pack`, used to flatten/unflatten the
    keyword dict into the positional call.
    """
    spec = [slot for b in buckets for slot in b.schema_slots()]
    names = [name for name, _ in spec]
    schema_str = "(" + ", ".join(f"{type_str} {name}" for name, type_str in spec) + ")"
    return schema_str, names


def _pack(obj: Any, buckets: List[_Bucket]) -> Dict[str, Any]:
    """Ask each bucket to extract its slot(s) from ``obj``.

    ``buckets`` is the precomputed bucket list (from :func:`_get_buckets`).
    Hot paths -- e.g. the closures created by
    :func:`_te_register_custom_op` -- must pass the precomputed list to
    avoid recomputing and, critically, to keep Dynamo away from
    ``cls.__dict__`` while tracing.
    """
    out: Dict[str, Any] = {}
    for bucket in buckets:
        for name, value in bucket.pack(obj):
            out[name] = value
    return out


def _unpack(cls: type, args: Dict[str, Any], buckets: List[_Bucket]) -> Any:
    """Ask each bucket to inject its field(s) into a fresh instance.

    The instance is built via ``cls.__new__(cls)`` (we bypass any
    dataclass ``__init__`` so unknown-typed fields can stay as ``None``
    even when they have no default). ``buckets`` semantics match
    :func:`_pack`.
    """
    kwargs: Dict[str, Any] = {}
    for bucket in buckets:
        bucket.unpack(args, kwargs)
    obj = cls.__new__(cls)
    for k, v in kwargs.items():
        object.__setattr__(obj, k, v)
    return obj


# --------------------------------------------------------------------------- #
# Op registration helpers
# --------------------------------------------------------------------------- #
#
# Per-step building blocks (schema, kernel wrapping, autograd bridge,
# dispatcher) used by :func:`_te_register_custom_op` to turn user-supplied
# eager kernels + dataclass arg types into a ``torch.library`` custom op.


def _prepare_for_saving(tensors: Any) -> Tuple[List[Optional[torch.Tensor]], Any]:
    """Lazy wrapper around :func:`quantized_tensor.prepare_for_saving`.

    Used only to flatten the user's setup-context return into a
    ``(flat_tensors, tensor_objects)`` pair stashed on ``ctx`` for the
    backward; the forward output and saved-tensor restoration on the
    compile-path go through :func:`_template_reassemble` instead. Lazy-imports
    avoid the dynamo<->quantized_tensor circular import that
    ``transformer_engine.pytorch`` would otherwise trigger at module
    import time.
    """
    from transformer_engine.pytorch.quantized_tensor import prepare_for_saving

    return prepare_for_saving(*(tensors or ()))


# --------------------------------------------------------------------------- #
# Forward-result packing
# --------------------------------------------------------------------------- #
#
# The op schema is fixed at ``-> Tensor[]``. To return non-tensor
# values (subclass wrappers, ``QuantizedTensorStorage``, ``None``...),
# :func:`_format_fwd_result` runs each user output through its
# flatten protocol and concatenates the inner tensors into the flat
# return; saved-for-backward tensors follow in declaration order.


def _flatten_value_into(flat: List[torch.Tensor], value: Any) -> None:
    """Append the ``Tensor[]`` slots produced by ``value`` to ``flat``.

    The inverse of :func:`_template_reassemble`; the slot counts match
    :func:`_template_slot_count`:

    * ``None``           -> 1 sentinel slot (via :func:`_encode_none`).
    * plain Tensor       -> 1 slot.
    * tensor subclass with ``__tensor_flatten__`` -> ``len(inner_names)``
      slots, in the order declared by the class.
    * storage with ``_torch_compile_flatten`` -> ``len(tensors)`` slots.
    """
    if value is None:
        flat.append(_encode_none(None))
        return
    if isinstance(value, torch.Tensor):
        if type(value) is not torch.Tensor and hasattr(value, "__tensor_flatten__"):
            inner_names, _ = value.__tensor_flatten__()
            flat.extend(_encode_none(getattr(value, n)) for n in inner_names)
        else:
            flat.append(_encode_none(value))
        return
    if hasattr(value, "_torch_compile_flatten"):
        _, _, tensors = value._torch_compile_flatten()
        flat.extend(_encode_none(t) for t in tensors)
        return
    raise TypeError(
        f"unsupported value type {type(value).__name__}; expected "
        "None / torch.Tensor / tensor subclass with __tensor_flatten__ / "
        "class with _torch_compile_flatten."
    )


# Trailing slots in every ``fwd_impl`` return tuple:
# ``tensors_to_save, tensor_objects, ctx_attrs``. User-output count
# is ``len(result) - _FWD_TRAILING_SLOTS``.
_FWD_TRAILING_SLOTS = 3


def _format_fwd_result(result: Any) -> List[torch.Tensor]:
    """Pack a fwd-impl return tuple into the op's ``Tensor[]`` payload.

    User outputs come first, then the saved-for-backward tensors in
    declaration order. Both groups go through the same per-value
    :func:`_flatten_value_into` dispatch -- the slot layout produced
    here must match exactly what :func:`_template_slot_count` reports
    for the corresponding fake template, since the call-site reassembly in
    :func:`forward_fn` / :func:`_setup_context` slices this flat list
    back into user-facing objects using those per-template counts.

    ``None`` entries on either side are smuggled through
    :func:`_encode_none` so the schema stays non-nullable and
    ``register_autograd`` still attaches a ``grad_fn`` to the op's
    outputs.

    The split point between user outputs and saved tensors is
    inferred from the impl's return shape:
    ``(*user_outputs, tensors_to_save, tensor_objects, ctx_attrs)``
    -- the last three slots are the standard ``fwd_impl`` tail.
    """
    num_outputs = len(result) - _FWD_TRAILING_SLOTS
    flat: List[torch.Tensor] = []
    for value in result[:num_outputs]:
        _flatten_value_into(flat, value)
    saved = result[num_outputs] or ()
    for value in saved:
        _flatten_value_into(flat, value)
    return flat


def _format_bwd_result(grads: Any, num_grad_inputs: int, op_qualname: str) -> List[torch.Tensor]:
    """Pack a backward-impl return tuple into the op's ``Tensor[]`` payload.

    Validates that the user kernel returned exactly one grad per
    ``input_tensors_for_grad`` entry; raises with the op's qualified
    name on mismatch.
    """
    grads = list(grads)
    if len(grads) != num_grad_inputs:
        raise RuntimeError(
            f"{op_qualname} expected backward_impl to return "
            f"{num_grad_inputs} grads (one per input_tensors_for_grad "
            f"entry), got {len(grads)}"
        )
    return [_encode_none(g) for g in grads]


def _resolve_grad_targets(
    fwd_buckets: List[_Bucket],
    fwd_arg_type: type,
    input_tensors_for_grad: List[str],
) -> Tuple[List[Any], List[Tuple[int, bool]]]:
    """Validate ``input_tensors_for_grad`` and resolve grad-output layout.

    Returns ``(fwd_slot_defaults, grad_targets)`` where:

    * ``fwd_slot_defaults`` is the per-slot "no-grad" template the
      autograd return tuple starts from -- ``[]`` for ``Tensor[]``
      slots, ``None`` otherwise. ``register_autograd`` requires one
      grad slot per forward input with matching tree structure (a
      ``Tensor[]`` slot must get back a list, not bare ``None``).
    * ``grad_targets`` is the ``[(slot_index, as_list), ...]`` mapping
      for each name in ``input_tensors_for_grad``, in the same order;
      ``as_list`` is ``True`` for ``Tensor[]``-shaped slots so the
      caller wraps the single grad into a length-matched list.
    """
    fwd_slot_defaults: List[Any] = []
    for bucket in fwd_buckets:
        for _, type_str in bucket.schema_slots():
            fwd_slot_defaults.append([] if type_str.endswith("[]") else None)

    fwd_grad_targets: Dict[str, Tuple[int, bool]] = {}
    slot_offset = 0
    for bucket in fwd_buckets:
        slots = bucket.schema_slots()
        if isinstance(bucket, _TensorBucket):
            fwd_grad_targets[bucket.name] = (slot_offset, False)
        elif isinstance(bucket, _UniversalTensorBucket):
            # Grad routes to the ``Tensor?`` slot -- the wrapper /
            # plain-tensor passthrough -- so the gradient flows back
            # to the user-facing object (e.g. an ``nn.Parameter``
            # wrapped as ``Float8Tensor``). In the storage path the
            # ``Tensor?`` slot is ``None`` and the kernel does not
            # request a grad for it.
            for i, (slot_name, _) in enumerate(slots):
                if slot_name == bucket.slot_name():
                    fwd_grad_targets[bucket.name] = (slot_offset + i, False)
                    break
        slot_offset += len(slots)

    unknown = [n for n in input_tensors_for_grad if n not in fwd_grad_targets]
    if unknown:
        raise ValueError(
            "input_tensors_for_grad contains names not present in "
            f"{fwd_arg_type.__name__} schema: {unknown}"
        )
    grad_targets = [fwd_grad_targets[n] for n in input_tensors_for_grad]
    return fwd_slot_defaults, grad_targets


def _register_kernel(
    *,
    op_name: str,
    op_qualname: str,
    arg_type: type,
    arg_names: List[str],
    buckets: List[_Bucket],
    impl: Callable[[Any], Any],
    fake_impl: Callable[[Any], Any],
    format_result: Callable[[Any], List[torch.Tensor]],
) -> None:
    """Wire ``impl`` + ``fake_impl`` into :data:`_TE_LIB` under ``op_name``.

    The wrapper unpacks the flat positional args using
    ``arg_names`` / ``buckets``, calls the user kernel with the rebuilt
    dataclass instance, and packs the result through ``format_result``
    (which encodes ``None``s into the op's ``Tensor[]`` return slot).
    """

    def _eager(*flat: Any) -> List[torch.Tensor]:
        kwargs = dict(zip(arg_names, flat))
        obj = _unpack(arg_type, kwargs, buckets)
        return format_result(impl(obj))

    def _fake(*flat: Any) -> List[torch.Tensor]:
        kwargs = dict(zip(arg_names, flat))
        obj = _unpack(arg_type, kwargs, buckets)
        return format_result(fake_impl(obj))

    _TE_LIB.impl(op_name, _eager, "CompositeExplicitAutograd")
    torch.library.register_fake(op_qualname, _fake, lib=_TE_LIB)


def _collect_universal_slot_offsets(buckets: List[_Bucket]) -> List[int]:
    """Return the start index of each :class:`_UniversalTensorBucket`
    group inside the flat positional arg list of a registered op.

    The four schema slots emitted by a universal bucket are always
    contiguous (``name``, ``__tensors``, ``__pg``, ``__meta``); knowing
    the offset of the first slot lets a subclass dispatch rule rewrite
    all four slots in place at trace / eager time without re-deriving
    the bucket list.
    """
    offsets: List[int] = []
    pos = 0
    for bucket in buckets:
        if isinstance(bucket, _UniversalTensorBucket):
            offsets.append(pos)
        pos += len(bucket.schema_slots())
    return offsets


def _flatten_subclass_into_slots(
    new_args: List[Any], slot_offsets: List[int], subclass: type
) -> None:
    """Rewrite each ``_UniversalTensorBucket`` group whose ``Tensor?``
    slot holds an instance of ``subclass`` into the storage layout.

    Used as the body of a ``register_torch_dispatch`` rule on the outer
    fwd / bwd op: a subclass passed through the user-facing op is
    flattened in place (via ``_torch_compile_flatten``) so that the
    inner op only ever sees plain tensors plus the storage-flatten
    metadata. The wrapper's autograd identity remains attached to the
    inner tensors via the wrapper-subclass machinery, so gradients
    still flow back to the user-facing tensor.
    """
    for offset in slot_offsets:
        val = new_args[offset]
        if val is None or not isinstance(val, subclass):
            continue
        meta, pg, tensors = val._torch_compile_flatten()
        meta._data[_UniversalTensorBucket.KIND_KEY] = _UniversalTensorBucket.KIND_STORAGE
        new_args[offset] = None
        new_args[offset + 1] = list(tensors)
        new_args[offset + 2] = pg
        new_args[offset + 3] = meta


def _register_autograd_for_op(
    *,
    fwd_op_name: str,
    bwd_op_name: str,
    fwd_arg_type: type,
    fwd_arg_names: List[str],
    fwd_buckets: List[_Bucket],
    bwd_arg_names: List[str],
    bwd_buckets: List[_Bucket],
    fwd_slot_defaults: List[Any],
    grad_targets: List[Tuple[int, bool]],
    setup_context_user: Callable[..., None],
    backward_obj_type: type,
    fwd_fake_impl: Callable[[Any], Tuple[Any, ...]],
    fwd_field_names: Sequence[str],
) -> None:
    """Wire ``register_autograd`` on a forward op so its backward calls
    ``bwd_op_name``.

    Both the inner and outer tiers of a two-tier op share an identical
    autograd bridge (the wrapper logic only cares about op *names*), so
    this helper is called once per tier; the actual kernel
    registration is handled separately (by :func:`_register_kernel`
    for the inner tier and :func:`_register_outer_forwarder` for the
    outer tier).

    The op's ``Tensor[]`` return holds the flat layout produced by
    :func:`_format_fwd_result` -- one chunk per user output / saved
    tensor. ``setup_context`` re-runs ``fwd_fake_impl`` to recover the
    fake output / saved templates, then reassembles each chunk via
    :func:`_template_reassemble`. Saved slots that alias a forward arg
    (the fake returns the actual arg) are detected by identity and
    surfaced to the user's ``setup_context`` via
    ``ctx_attrs["saved_tensor_aliases"]``.
    """
    fwd_qualname = f"{_TE_OP_NAMESPACE}::{fwd_op_name}"

    def _setup_context(ctx, inputs, output):
        ctx._te_fwd_tensor_list_lengths = {
            i: len(value) for i, value in enumerate(inputs) if isinstance(value, list)
        }
        kwargs = dict(zip(fwd_arg_names, inputs))
        fwd_obj = _unpack(fwd_arg_type, kwargs, fwd_buckets)

        user_fakes, saved_fakes, ctx_attrs = _split_fwd_fake_result(fwd_fake_impl(fwd_obj))
        pairs = _fwd_arg_alias_pairs(fwd_obj, fwd_field_names)
        saved_aliases = tuple(_alias_name_for(t, pairs) for t in saved_fakes)
        ctx_attrs = dict(ctx_attrs)
        ctx_attrs["saved_tensor_aliases"] = saved_aliases

        cursor = 0
        user_outputs: List[Any] = []
        for template in user_fakes:
            n = _template_slot_count(template)
            chunk = [_decode_none(t) for t in output[cursor : cursor + n]]
            cursor += n
            user_outputs.append(_template_reassemble(template, chunk))

        tensors_to_save_from_forward_list: List[Any] = []
        for template, alias in zip(saved_fakes, saved_aliases):
            aliased = alias is not None
            n = _template_slot_count(template, aliased=aliased)
            chunk = [_decode_none(t) for t in output[cursor : cursor + n]]
            cursor += n
            tensors_to_save_from_forward_list.append(
                _template_reassemble(template, chunk, aliased=aliased)
            )
        tensors_to_save_from_forward = tuple(tensors_to_save_from_forward_list)

        bwd_obj = backward_obj_type()
        tensors_to_save_from_setup = setup_context_user(
            bwd_obj,
            fwd_obj,
            user_outputs[0] if len(user_fakes) == 1 else tuple(user_outputs),
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
            bwd_obj.setup_saved_tensors(ctx)
        ctx.tensor_objects = None
        per_output_grads = grad_outputs[0]
        bwd_obj.grad_output = _decode_none(per_output_grads[0])
        kwargs = _pack(bwd_obj, bwd_buckets)
        bwd_args_flat = [kwargs[name] for name in bwd_arg_names]
        bwd_op = getattr(getattr(torch.ops, _TE_OP_NAMESPACE), bwd_op_name)
        grads = [_decode_none(g) for g in bwd_op(*bwd_args_flat)]
        out: List[Any] = list(fwd_slot_defaults)
        tensor_list_lengths = getattr(ctx, "_te_fwd_tensor_list_lengths", {})
        # Pad every ``Tensor[]`` slot with ``None`` entries matching the
        # corresponding forward input length. AOT's pytree check on the
        # backward return rejects an empty list where the forward input
        # was a non-empty list -- the list structure must match
        # element-for-element. Grad-target slots below overwrite the
        # first entry with the actual gradient.
        for pos, length in tensor_list_lengths.items():
            if isinstance(out[pos], list):
                out[pos] = [None] * length
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
        lib=_TE_LIB,
    )


def _register_outer_forwarder(
    *,
    outer_op_name: str,
    inner_op_name: str,
    buckets: Optional[List[_Bucket]] = None,
    subclass_list: Optional[List[type]] = None,
) -> None:
    """Register the outer op's default kernel + fake.

    Both kernel and fake forward to the inner op, optionally with an
    in-place input flatten step for any registered subclass arg (so the
    inner op's plain-tensor schema is satisfied). Outputs travel
    untouched in their flat ``Tensor[]`` shape -- the user-facing
    wrapping back into subclasses / storage happens in
    :func:`forward_fn` via :class:`_ToSubclassFn`.
    """
    inner_op = getattr(getattr(torch.ops, _TE_OP_NAMESPACE), inner_op_name)

    input_flatten_enabled = bool(subclass_list) and buckets is not None

    if input_flatten_enabled:
        slot_offsets = _collect_universal_slot_offsets(buckets)

        def _flatten_all(new_args: List[Any]) -> None:
            for sub in subclass_list:
                _flatten_subclass_into_slots(new_args, slot_offsets, sub)

        def _outer_kernel(*flat: Any) -> List[torch.Tensor]:
            new_args = list(flat)
            _flatten_all(new_args)
            return inner_op(*new_args)

        def _outer_fake(*flat: Any) -> List[torch.Tensor]:
            new_args = list(flat)
            _flatten_all(new_args)
            return inner_op(*new_args)

    else:

        def _outer_kernel(*flat: Any) -> List[torch.Tensor]:
            return inner_op(*flat)

        def _outer_fake(*flat: Any) -> List[torch.Tensor]:
            return inner_op(*flat)

    _TE_LIB.impl(outer_op_name, _outer_kernel, "CompositeExplicitAutograd")
    torch.library.register_fake(f"{_TE_OP_NAMESPACE}::{outer_op_name}", _outer_fake, lib=_TE_LIB)


def _all_quantized_tensor_subclasses() -> List[type]:
    """Return every imported ``QuantizedTensor`` wrapper subclass.

    Imports the ``transformer_engine.pytorch.tensor`` package as a side
    effect so that all concrete wrapper subclasses (``Float8Tensor``,
    ``MXFP8Tensor``, ``Float8BlockwiseQTensor``, ``NVFP4Tensor``) get
    registered with Python's subclass tracker before we walk
    ``QuantizedTensor.__subclasses__()`` recursively. The lazy import
    keeps ``dynamo.py`` itself free of top-level ``tensor`` imports
    (which would form a cycle through the in-function ``dynamo``
    imports inside the tensor modules), while still giving every
    custom op the full subclass set at registration time.
    """
    import transformer_engine.pytorch.tensor  # noqa: F401 -- side-effect: registers subclasses
    from transformer_engine.pytorch.quantized_tensor import QuantizedTensor

    seen: List[type] = []
    stack: List[type] = list(QuantizedTensor.__subclasses__())
    while stack:
        cls = stack.pop()
        if cls in seen:
            continue
        seen.append(cls)
        stack.extend(cls.__subclasses__())
    return seen


def _te_register_custom_op(
    *,
    op_name: str,
    input_tensors_for_grad: List[str],
    fwd_arg_type: type,
    fwd_impl: Callable[[Any], Any],
    setup_context: Callable[..., None],
    backward_arg_type: type,
    backward_obj: type,
    backward_impl: Callable[[Any], Any],
    fwd_fake_impl: Callable[[Any], Tuple[Any, ...]],
    bwd_fake_impl: Callable[[Any], Tuple[Any, ...]],
) -> Callable[..., Any]:
    """Register a TE module's forward + backward as a single torch custom op.

    Parameters
    ----------
    op_name
        Op name used when registering with ``torch.library``. The
        namespace is fixed at module level (:data:`_TE_OP_NAMESPACE`).
    input_tensors_for_grad
        Names of forward-arg-type fields for which ``backward_impl``
        returns gradients, in the same order. The wrapper uses this to
        pad the autograd return tuple with ``None`` for every input not
        listed here, so torch sees one grad slot per forward input as
        required by ``register_autograd``.
    fwd_arg_type
        Dataclass type aggregating all forward inputs (e.g.
        ``LinearFwdArgs``). Used to (re)build the structured argument
        from the flat tensor / non-tensor inputs accepted by the custom op.
    fwd_impl
        Eager forward implementation. Receives a single argument of type
        ``fwd_arg_type`` and must return a tuple of the form
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
    setup_context
        Eager autograd ``setup_context`` analogue. Receives a freshly
        constructed ``backward_obj`` instance, the forward args, the
        forward output, and ``ctx_attrs`` produced by ``fwd_impl``;
        is responsible for populating the backward-state object so that
        ``backward_impl`` can later consume it.
    backward_arg_type
        Type accepted by ``backward_impl``. May differ from ``backward_obj``
        if the backward op needs a wrapped / opaque view of the state.
    backward_obj
        Dataclass / class used to instantiate a fresh backward-state
        container at the end of the forward pass (typically the same as
        ``backward_arg_type``).
    backward_impl
        Eager backward implementation. Receives a single argument of type
        ``backward_arg_type`` and returns the gradient tuple.
    fwd_fake_impl
        Forward fake implementation: ``fn(fwd_obj) -> (*user_outputs,
        tensors_to_save, tensor_objects, ctx_attrs)`` -- the same tuple
        shape as ``fwd_impl``, but built from *fake* values instead of
        running the real kernel. Each slot is one of:

        * ``quantizer.make_fake_empty(shape, dtype, device)`` -- a
          Dynamo-safe quantized wrapper (assembled via
          ``__tensor_unflatten__`` with a snapshot-free meta).
        * ``quantizer.make_empty(shape, dtype, device)`` -- a quantized
          storage (e.g. an FP8 weight workspace).
        * ``torch.empty(shape, dtype, device)`` -- a plain tensor.
        * the actual forward-arg tensor -- for a saved slot that aliases
          a forward input (detected by identity).
        * ``None`` -- an absent output / saved slot.

        This single callable drives both consumers: ``forward_fn`` /
        ``setup_context`` use its fake values directly as reassembly
        templates (:func:`_template_slot_count` /
        :func:`_template_reassemble`), and
        :func:`_fwd_register_fake_from_fake_impl` wires it (with aliased
        saved slots nulled) as the op's
        :func:`torch.library.register_fake`. The whole callable must be
        Dynamo-traceable under ``fullgraph=True``.
    bwd_fake_impl
        Backward fake implementation: ``fn(bwd_obj) -> grad_tuple``, one
        fake grad per gradient output in the same order as
        ``backward_impl``'s return tuple (``None`` for missing grads,
        ``torch.empty`` for plain, ``quantizer.make_empty`` for
        quantized). Wired directly as the backward op's
        ``register_fake`` -- backward grads never round-trip through the
        op payload, so no layout adapter is needed.

    Returns
    -------
    Callable
        A function ``forward_fn(fwd_arg_type_instance)`` that dispatches
        through the registered custom op, returning the user-facing
        outputs (single tensor if the impl produced exactly one
        user-facing output, otherwise a tuple). Use under
        ``torch.compiler.is_compiling()`` as a drop-in for
        ``Function.apply``.
    """

    outer_fwd_name = op_name
    outer_bwd_name = f"{op_name}_backward"
    # Auto-discover every imported ``QuantizedTensor`` wrapper subclass
    # so callers never have to enumerate them. Each subclass gets a
    # ``register_torch_dispatch`` rule on the outer op (see below) and
    # is flattened into plain tensors before the inner op runs.
    subclass_list = _all_quantized_tensor_subclasses()

    # Precompute the bucket list once per arg type and capture it in
    # the registered closures. Re-deriving the bucket list inside a
    # compiled call would force :func:`_get_buckets` to read
    # ``cls.__dict__`` from inside a Dynamo-traced function, which
    # triggers a "mappingproxy affected by dictionary mutation" graph
    # break under ``fullgraph=True``.
    fwd_buckets: List[_Bucket] = _get_buckets(fwd_arg_type)
    bwd_buckets: List[_Bucket] = _get_buckets(backward_arg_type)

    fwd_schema_args, fwd_arg_names = _build_schema(fwd_buckets)
    bwd_schema_args, bwd_arg_names = _build_schema(bwd_buckets)

    num_grad_inputs = len(input_tensors_for_grad)
    fwd_slot_defaults, grad_targets = _resolve_grad_targets(
        fwd_buckets, fwd_arg_type, input_tensors_for_grad
    )

    # Two-tier layout when at least one ``QuantizedTensor`` subclass is
    # imported (the common case -- ``_all_quantized_tensor_subclasses``
    # discovers them automatically):
    #   inner = ``{op_name}_base`` -- real impl, sees only plain tensors
    #           and the storage-flatten metadata.
    #   outer = ``{op_name}`` -- user-facing op that either falls through
    #           to the inner op (plain-tensor path) or is rewritten by a
    #           ``register_torch_dispatch`` rule (subclass path) into a
    #           call to the inner op with subclass tensors flattened in
    #           place. Both tiers carry their own ``register_autograd``
    #           bridge.
    # Single-tier fallback: if no ``QuantizedTensor`` subclasses have
    # been imported (e.g. minimal embedded build) only the outer pair
    # is defined and it owns the real impl directly.
    inner_fwd_name = f"{op_name}_base" if subclass_list else outer_fwd_name
    inner_bwd_name = f"{outer_bwd_name}_base" if subclass_list else outer_bwd_name

    # Forward op concatenates user outputs and tensors_to_save into a
    # single ``Tensor[]`` return so that autograd's ``setup_context`` can
    # stash the saved-for-backward tensors without re-running the eager
    # impl. The schema is non-nullable (``Tensor[]``, not ``Tensor?[]``)
    # because ``torch.library.register_autograd`` does not propagate
    # ``grad_fn`` to a nullable list output. ``None`` entries on either
    # side are smuggled through via :func:`_encode_none` /
    # :func:`_decode_none` sentinels.
    _TE_LIB.define(f"{inner_fwd_name}{fwd_schema_args} -> Tensor[]")
    _TE_LIB.define(f"{inner_bwd_name}{bwd_schema_args} -> Tensor[]")
    if subclass_list:
        # Outer fwd / outer bwd are user-facing entry points. The
        # outer fwd is the target of ``register_torch_dispatch`` for
        # the forward subclass path; outer bwd is the target for the
        # backward subclass path. Both forward to the corresponding
        # inner op when no rule matches (plain-tensor / pure-storage
        # path).
        _TE_LIB.define(f"{outer_fwd_name}{fwd_schema_args} -> Tensor[]")
        _TE_LIB.define(f"{outer_bwd_name}{bwd_schema_args} -> Tensor[]")

    # Inner pair owns the real implementation. The fwd & bwd kernels
    # are registered directly against the user-supplied impls; the
    # autograd bridge below wires the inner fwd op's backward to call
    # the inner bwd op.
    inner_fwd_qualname = f"{_TE_OP_NAMESPACE}::{inner_fwd_name}"
    inner_bwd_qualname = f"{_TE_OP_NAMESPACE}::{inner_bwd_name}"

    # The module supplies its output layout as a forward ``fake_impl``
    # (fake values in the eager-impl tuple shape). ``forward_fn`` /
    # ``setup_context`` consume it directly as reassembly templates; the
    # forward ``register_fake`` kernel wraps it to null aliased saved slots
    # (so the fake flat ``Tensor[]`` matches the eager impl). The backward
    # ``fake_impl`` is the backward ``register_fake`` directly. ``field
    # names`` are precomputed here (reading ``dataclasses.fields`` in-trace
    # would graph-break) for the alias-by-identity detection.
    fwd_field_names = [f.name for f in dataclasses.fields(fwd_arg_type)]
    fwd_register_fake = _fwd_register_fake_from_fake_impl(fwd_fake_impl, fwd_field_names)

    _register_kernel(
        op_name=inner_fwd_name,
        op_qualname=inner_fwd_qualname,
        arg_type=fwd_arg_type,
        arg_names=fwd_arg_names,
        buckets=fwd_buckets,
        impl=fwd_impl,
        fake_impl=fwd_register_fake,
        format_result=_format_fwd_result,
    )
    _register_kernel(
        op_name=inner_bwd_name,
        op_qualname=inner_bwd_qualname,
        arg_type=backward_arg_type,
        arg_names=bwd_arg_names,
        buckets=bwd_buckets,
        impl=backward_impl,
        fake_impl=bwd_fake_impl,
        format_result=lambda g: _format_bwd_result(g, num_grad_inputs, inner_bwd_qualname),
    )
    _register_autograd_for_op(
        fwd_op_name=inner_fwd_name,
        bwd_op_name=inner_bwd_name,
        fwd_arg_type=fwd_arg_type,
        fwd_arg_names=fwd_arg_names,
        fwd_buckets=fwd_buckets,
        bwd_arg_names=bwd_arg_names,
        bwd_buckets=bwd_buckets,
        fwd_slot_defaults=fwd_slot_defaults,
        grad_targets=grad_targets,
        setup_context_user=setup_context,
        backward_obj_type=backward_obj,
        fwd_fake_impl=fwd_fake_impl,
        fwd_field_names=fwd_field_names,
    )

    if subclass_list:
        # Outer tier (thin shell): default kernels forward to inner
        # plus a ``register_torch_dispatch`` rule per subclass that
        # flattens the wrapper in place before forwarding. Carries
        # its own autograd bridge so the user-facing subclass tensor
        # (e.g. a ``Float8Tensor`` parameter) stays on the autograd
        # graph and receives a ``.grad``.
        _register_outer_forwarder(
            outer_op_name=outer_fwd_name,
            inner_op_name=inner_fwd_name,
            buckets=fwd_buckets,
            subclass_list=list(subclass_list),
        )
        _register_outer_forwarder(
            outer_op_name=outer_bwd_name,
            inner_op_name=inner_bwd_name,
        )
        _register_autograd_for_op(
            fwd_op_name=outer_fwd_name,
            bwd_op_name=outer_bwd_name,
            fwd_arg_type=fwd_arg_type,
            fwd_arg_names=fwd_arg_names,
            fwd_buckets=fwd_buckets,
            bwd_arg_names=bwd_arg_names,
            bwd_buckets=bwd_buckets,
            fwd_slot_defaults=fwd_slot_defaults,
            grad_targets=grad_targets,
            setup_context_user=setup_context,
            backward_obj_type=backward_obj,
            fwd_fake_impl=fwd_fake_impl,
            fwd_field_names=fwd_field_names,
        )

        fwd_slot_offsets = _collect_universal_slot_offsets(fwd_buckets)
        bwd_slot_offsets = _collect_universal_slot_offsets(bwd_buckets)
        inner_fwd_op = getattr(getattr(torch.ops, _TE_OP_NAMESPACE), inner_fwd_name)
        inner_bwd_op = getattr(getattr(torch.ops, _TE_OP_NAMESPACE), inner_bwd_name)
        outer_fwd_op = getattr(getattr(torch.ops, _TE_OP_NAMESPACE), outer_fwd_name)
        outer_bwd_op = getattr(getattr(torch.ops, _TE_OP_NAMESPACE), outer_bwd_name)
        outer_fwd_qualname = f"{_TE_OP_NAMESPACE}::{outer_fwd_name}"
        outer_bwd_qualname = f"{_TE_OP_NAMESPACE}::{outer_bwd_name}"

        def _flatten_all_subclasses(new_args: List[Any], slot_offsets: List[int]) -> None:
            for sub in subclass_list:
                _flatten_subclass_into_slots(new_args, slot_offsets, sub)

        def _fwd_rule(mode, func, types, args, kwargs):
            del mode, func, types, kwargs
            new_args = list(args)
            _flatten_all_subclasses(new_args, fwd_slot_offsets)
            return inner_fwd_op(*new_args)

        def _bwd_rule(mode, func, types, args, kwargs):
            del mode, func, types, kwargs
            new_args = list(args)
            _flatten_all_subclasses(new_args, bwd_slot_offsets)
            return inner_bwd_op(*new_args)

        # Per-subclass dispatch rule: any registered subclass arg
        # passed to the outer op (e.g. Dynamo lifting a
        # ``Float8Tensor`` weight into the FX graph) is flattened
        # into its storage layout before forwarding to the inner op,
        # which only ever sees plain tensors.
        for sub in subclass_list:
            torch.library.register_torch_dispatch(outer_fwd_qualname, sub, _fwd_rule, lib=_TE_LIB)
            torch.library.register_torch_dispatch(outer_bwd_qualname, sub, _bwd_rule, lib=_TE_LIB)

        # ``QuantizedTensor.__torch_dispatch__`` falls back to
        # dequantizing all subclass args for any op it does not
        # recognise, which would defeat our
        # ``register_torch_dispatch`` rules and would also crash on
        # FakeTensors (``tex.dequantize`` needs ``data_ptr``). Mark
        # every op we register through this helper -- both tiers and
        # both directions -- as passthroughs so QuantizedTensor
        # delegates straight to ``super().__torch_dispatch__``.
        from transformer_engine.pytorch.quantized_tensor import (
            _quantized_tensor_passthrough_ops,
        )

        _quantized_tensor_passthrough_ops.add(outer_fwd_op.default)
        _quantized_tensor_passthrough_ops.add(outer_bwd_op.default)
        _quantized_tensor_passthrough_ops.add(inner_fwd_op.default)
        _quantized_tensor_passthrough_ops.add(inner_bwd_op.default)

    fwd_op = getattr(getattr(torch.ops, _TE_OP_NAMESPACE), outer_fwd_name)

    def forward_fn(fwd_args):
        user_fakes, _saved_fakes, _ctx_attrs = _split_fwd_fake_result(fwd_fake_impl(fwd_args))
        kwargs = _pack(fwd_args, fwd_buckets)
        flat_in = [kwargs[name] for name in fwd_arg_names]
        result = fwd_op(*flat_in)

        # Slice the flat result using the fake outputs as templates. Subclass
        # templates route through :class:`_ToSubclassFn` to keep the wrap on
        # the autograd graph; plain tensors / storage classes are
        # reconstructed directly. User outputs never alias a forward arg.
        cursor = 0
        outputs: List[Any] = []
        for template in user_fakes:
            n = _template_slot_count(template)
            chunk = [_decode_none(t) for t in result[cursor : cursor + n]]
            cursor += n
            outputs.append(_template_reassemble(template, chunk, with_autograd=True))

        if len(outputs) == 1:
            return outputs[0]
        return tuple(outputs)

    return forward_fn
