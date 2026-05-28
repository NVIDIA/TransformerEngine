# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Pure Python base classes for quantization."""

from __future__ import annotations
from typing import Optional, Tuple, Iterable, Any, Dict, List, Union
import abc
import warnings
import math

import torch
from torch.utils._pytree import tree_map

from transformer_engine.common.recipe import Recipe
from transformer_engine.pytorch.tensor._quantization_helpers import (
    _QuantizeFunc,
    _IdentityFunc,
    _stride_from_shape,
)


# Maps a Quantizer subclass's ``__qualname__`` to the class object. Populated
# lazily via :meth:`Quantizer.__init_subclass__` and consumed by
# :meth:`Quantizer._unflatten` to dispatch reconstruction to the right
# subclass when a TE custom op is unpacked under ``torch.compile``.
_QUANTIZER_REGISTRY: Dict[str, type] = {}


def _quantizer_subclass_snapshot(
    quantizer: Optional["Quantizer"],
) -> Optional[Tuple[Tuple[str, Any], ...]]:
    """Return a Dynamo-guard-stable snapshot of a quantizer, or ``None``.

    Used by tensor subclasses (e.g. :class:`Float8Tensor`) to embed a
    tensor-free, comparable representation of their live
    :class:`Quantizer` in the ``meta`` dict returned from
    ``__tensor_flatten__``. PyTorch's tensor-subclass metadata guard
    diff-checks that dict via ``dict.__eq__`` on every entry into the
    compiled region, so values that resolve to elementwise tensor
    comparison or identity-only equality (live ``torch.Tensor``
    objects, ``ProcessGroup``, the live quantizer instance itself)
    cannot appear there.

    The snapshot is a sorted tuple of ``(key, value)`` pairs derived
    from ``quantizer._flatten()`` whenever the quantizer's state is
    fully expressible without tensors (an empty trailing tensor list
    in the ``_flatten`` triplet). Quantizers carrying tensors in their
    state (e.g. :class:`Float8Quantizer`'s ``scale`` / ``amax``) and
    quantizers that don't implement ``_flatten`` produce ``None``;
    in that case the subclass's ``__tensor_unflatten__`` will
    rebuild the wrapper with ``quantizer=None`` and any code that
    needs the live quantizer must source it from the bucket-level
    opaque metadata flowing through the inner custom op.
    """
    if quantizer is None:
        return None
    try:
        meta, _pg, tensors = quantizer._flatten()
    except NotImplementedError:
        return None
    if tensors:
        return None
    if hasattr(meta, "_data"):
        meta_dict = meta._data
    elif isinstance(meta, dict):
        meta_dict = meta
    else:
        return None
    return tuple(sorted(meta_dict.items(), key=lambda kv: kv[0]))


def _quantizer_from_subclass_snapshot(
    snapshot: Optional[Tuple[Tuple[str, Any], ...]],
) -> Optional["Quantizer"]:
    """Inverse of :func:`_quantizer_subclass_snapshot`.

    Rebuilds the quantizer from the qualname stored in the snapshot's
    ``"_qcls"`` entry, dispatching via :func:`Quantizer._unflatten`
    (and so via the right subclass's ``_do_unflatten``). The
    reconstructed quantizer's process-group reference is always
    ``None`` -- live ``ProcessGroup`` objects cannot survive the
    snapshot round trip; callers that need a real process group
    obtain it via the bucket-level opaque metadata instead.
    """
    if snapshot is None:
        return None
    meta_dict = dict(snapshot)
    return Quantizer._unflatten(meta_dict, None, [])

# Same idea for lightweight QuantizedTensorStorage shells. Populated via
# :meth:`QuantizedTensorStorage.__init_subclass__` and consumed by
# :meth:`QuantizedTensorStorage._torch_compile_unflatten`.
_STORAGE_REGISTRY: Dict[str, type] = {}


# Custom ops that should pass through __torch_dispatch__ without unwrapping
# QuantizedTensor subclasses (e.g. Float8Tensor). Register ops here that
# handle quantized tensors internally.
_quantized_tensor_passthrough_ops: set = set()


class QuantizedTensorStorage:
    r"""Base class for all TensorStorage classes.

    This class (and its subclasses) are optimization for when
    the full QuantizedTensor is not needed (when it is fully
    contained inside torch.autograd function and not visible to
    PyTorch's autograd).

    When creating a new tensor type X one should create both
    XTensorStorage class inheriting from QuantizedTensorStorage and
    XTensor inheriting from XTensorStorage and QuantizedTensor.
    XTensorStorage should contain all data members needed to
    implement the functionality of the tensor, while
    XTensor should only implement the functionality needed
    to behave like regular torch.Tensor (like __torch_dispatch__)."""

    _dtype: torch.dtype
    _quantizer: Optional[Quantizer]

    # ------------------------------------------------------------------ #
    # Declarative schema for the unified flatten / unflatten machinery   #
    # (consumed by both the storage ``_torch_compile_flatten`` protocol  #
    # and ``QuantizedTensor``'s PyTorch ``__tensor_flatten__`` helper).  #
    # ------------------------------------------------------------------ #

    # Names of optional tensor attributes on the instance, in canonical
    # order. Each name must be an attribute on ``self`` and must be
    # accepted as a kwarg by ``cls(**kwargs)`` (potentially after
    # remapping through :attr:`_FLATTEN_CTOR_KWARG`).
    _FLATTEN_TENSOR_ATTRS: Tuple[str, ...] = ()

    # Maps each entry in :attr:`_FLATTEN_TENSOR_ATTRS` to one of
    # ``"rowwise"`` / ``"columnwise"`` / ``"always"``. Consumed by
    # :meth:`Quantizer.create_storage_metadata` to translate a live
    # quantizer's ``rowwise_usage`` / ``columnwise_usage`` flags into
    # per-attribute presence (``has_*``) flags at output-spec time.
    # Unmapped attributes default to ``"always"``.
    _FLATTEN_TENSOR_USAGE: Dict[str, str] = {}

    # Names of value-stable scalar / enum attributes needed to round-trip
    # the instance. Same naming / kwarg conventions as
    # :attr:`_FLATTEN_TENSOR_ATTRS`.
    _FLATTEN_META_ATTRS: Tuple[str, ...] = ()

    # Map from attribute name to constructor kwarg name, used when they
    # differ (e.g. ``_data`` -> ``data``). Identity by default.
    _FLATTEN_CTOR_KWARG: Dict[str, str] = {}

    @classmethod
    def _flatten_meta_overrides(cls, meta: Dict[str, Any]) -> Dict[str, Any]:
        """Hook for last-mile meta value massaging before unflatten dispatches
        to ``cls(**kwargs)``. Default: no-op.

        Used today by :class:`Float8Tensor` to bridge :class:`FP8DType`
        (carried by the subclass output spec) back to the native
        ``tex.DType`` accepted by pybind-bound kernels.
        """
        return meta

    def update_usage(
        self,
        rowwise_usage: Optional[bool] = None,
        columnwise_usage: Optional[bool] = None,
    ):
        r"""
        Generate or remove quantized data based on provided usage.

        Parameters
        ----------
        rowwise_usage : Optional[bool[, default = None
                        Whether to create or keep the data needed for using the tensor
                        in rowwise fashion (e.g. as B argument in TN GEMM). Leaving it as `None`
                        preserves the original value in the tensor.
        columnwise_usage : Optional[bool], default = None
                           Whether to create or keep the data needed for using the tensor
                           in columnwise fashion (e.g. as A argument in TN GEMM). Leaving it as
                           `None` preserves the original value in the tensor.

        """
        raise NotImplementedError(
            f"{self.__class__.__name__} class does not implement update_usage function"
        )

    def get_usages(self) -> Dict[str, bool]:
        """Get the usage of the tensor"""
        raise NotImplementedError(
            f"{self.__class__.__name__} class does not implement get_usages function"
        )

    def prepare_for_saving(
        self,
    ) -> Tuple[list[Optional[torch.Tensor]], QuantizedTensorStorage]:
        """Prepare the tensor base for saving for backward"""
        raise NotImplementedError(
            f"{self.__class__.__name__} class does not implement prepare_for_saving function"
        )

    def restore_from_saved(
        self, tensors: list[Optional[torch.Tensor]]
    ) -> list[Optional[torch.Tensor]]:
        """Restore the tensor base data from the saved tensors list"""
        raise NotImplementedError(
            f"{self.__class__.__name__} class does not implement restore_from_saved function"
        )

    def _get_quantizer(self) -> Quantizer:
        """Get builder for quantized tensor

        Quantizer can be used for in-place operations.

        """
        if self._quantizer is not None:
            return self._quantizer
        return self._build_default_quantizer()

    def _build_default_quantizer(self) -> Quantizer:
        """Build default quantizer for the tensor"""
        raise ValueError(
            f"{self.__class__.__name__} has no quantizer "
            "and no default quantizer is available defined in the subclass."
        )

    def quantize_(
        self, tensor: torch.Tensor, *, noop_flag: Optional[torch.Tensor] = None
    ) -> QuantizedTensor:
        """Quantize tensor in-place"""
        self._get_quantizer().update_quantized(tensor, self, noop_flag=noop_flag)
        return self

    def update_quantizer(self, quantizer: Quantizer):
        """Update quantizer for the tensor"""
        if self._quantizer is None:
            raise RuntimeError("To be updated, quantizer must be set")
        if self._quantizer is not quantizer:
            warnings.warn("Quantizer is being updated, this may affect model behavior")
            self._quantizer = quantizer

    def copy_from_storage(self, src: QuantizedTensorStorage) -> None:
        """Copy data from another QuantizedTensorStorage."""
        raise NotImplementedError(
            f"{self.__class__.__name__} class does not implement copy_from_storage function"
        )

    # ------------------------------------------------------------------ #
    # torch.compile flatten / unflatten protocol
    # ------------------------------------------------------------------ #

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        _STORAGE_REGISTRY[cls.__qualname__] = cls

    def __eq__(self, other: object) -> bool:
        return self is other

    def __hash__(self) -> int:
        return id(self)

    @classmethod
    def _flatten_ctor_kw(cls, attr_name: str) -> str:
        """Return the constructor kwarg name corresponding to ``attr_name``.

        Identity unless overridden via :attr:`_FLATTEN_CTOR_KWARG`.
        """
        return cls._FLATTEN_CTOR_KWARG.get(attr_name, attr_name)

    @staticmethod
    def _flatten_presence_key(attr_name: str) -> str:
        """Return the ``has_*`` meta key indicating whether ``attr_name`` is
        present in the flattened payload. Derived from the attribute name
        (with the leading underscore stripped) so the static metadata
        constructors in ``float8_tensor.py`` etc. don't need to know about
        :attr:`_FLATTEN_CTOR_KWARG` remapping.
        """
        return f"has_{attr_name.lstrip('_')}"

    def _torch_compile_flatten(
        self,
    ) -> Tuple[Any, Optional["torch.distributed.ProcessGroup"], List[torch.Tensor]]:
        """Pack storage state into the ``(meta, pg, tensors)`` triplet
        consumed by :mod:`transformer_engine.pytorch.dynamo`.

        Generic implementation driven by :attr:`_FLATTEN_TENSOR_ATTRS`,
        :attr:`_FLATTEN_META_ATTRS`, and :attr:`_FLATTEN_CTOR_KWARG`.
        Quantizer-with-tensors (e.g. :class:`Float8Quantizer`'s
        ``scale`` / ``amax``) is round-tripped via
        :meth:`Quantizer._flatten`; quantizer tensors trail the
        storage's own tensors in the flat list.
        """
        from transformer_engine.pytorch.dynamo import (  # pylint: disable=import-outside-toplevel
            OpaqueSimpleMetadata,
        )

        tensors: List[torch.Tensor] = []
        meta_dict: Dict[str, Any] = {"_qstorage_cls": type(self).__qualname__}
        # Tensor-wrapper fields are only relevant when ``self`` is a live
        # ``torch.Tensor`` (e.g. ``Float8Tensor`` rewritten in-place to a
        # storage payload by ``_rewrite_subclass_to_storage``); a bare
        # storage shell has no outer shape / requires_grad / device.
        if isinstance(self, torch.Tensor):
            meta_dict.update(
                {
                    "is_tensor": True,
                    "shape": torch.Size(self.shape),
                    "requires_grad": self.requires_grad,
                    "device": self.device,
                }
            )
        for attr in self._FLATTEN_META_ATTRS:
            meta_dict[self._flatten_ctor_kw(attr)] = getattr(self, attr)
        for attr in self._FLATTEN_TENSOR_ATTRS:
            tensor = getattr(self, attr)
            present = tensor is not None
            meta_dict[self._flatten_presence_key(attr)] = present
            if present:
                tensors.append(tensor)
        quantizer_meta = None
        process_group = None
        if self._quantizer is not None:
            quantizer_meta, process_group, q_tensors = self._quantizer._flatten()
            tensors.extend(q_tensors)
        meta_dict["quantizer_meta"] = quantizer_meta
        return OpaqueSimpleMetadata(meta_dict), process_group, tensors

    @classmethod
    def _torch_compile_do_unflatten(
        cls,
        meta: Any,
        process_group: Optional["torch.distributed.ProcessGroup"],
        tensors: List[torch.Tensor],
    ) -> "QuantizedTensorStorage":
        """Reconstruct ``cls`` from a triplet produced by
        :meth:`_torch_compile_flatten`. Generic; driven by the same
        ``_FLATTEN_*`` declarations.
        """
        meta = cls._flatten_meta_overrides(meta)
        tensor_iter = iter(tensors)
        kwargs: Dict[str, Any] = {}
        for attr in cls._FLATTEN_TENSOR_ATTRS:
            kw = cls._flatten_ctor_kw(attr)
            kwargs[kw] = next(tensor_iter) if meta[cls._flatten_presence_key(attr)] else None
        quantizer = None
        if meta["quantizer_meta"] is not None:
            quantizer = Quantizer._unflatten(
                meta["quantizer_meta"], process_group, list(tensor_iter)
            )
        for attr in cls._FLATTEN_META_ATTRS:
            kw = cls._flatten_ctor_kw(attr)
            kwargs[kw] = meta[kw]
        kwargs["quantizer"] = quantizer
        if meta.get("is_tensor", False):
            kwargs.update(
                {
                    "shape": meta["shape"],
                    "dtype": kwargs["fake_dtype"],
                    "requires_grad": meta["requires_grad"],
                    "device": meta["device"],
                }
            )
        return cls(**kwargs)

    @classmethod
    def _torch_compile_unflatten(
        cls,
        meta: Any,
        process_group: Optional["torch.distributed.ProcessGroup"],
        tensors: List[torch.Tensor],
    ) -> "QuantizedTensorStorage":
        """Dispatch to the right storage subclass based on metadata."""
        storage_cls = meta["_qstorage_cls"]
        target = _STORAGE_REGISTRY.get(storage_cls)
        if target is None:
            raise ValueError(
                f"No QuantizedTensorStorage subclass registered under "
                f"qualname {storage_cls!r}; known: {sorted(_STORAGE_REGISTRY)}"
            )
        return target._torch_compile_do_unflatten(meta, process_group, tensors)



TensorOrQuantized = Union[torch.Tensor, QuantizedTensorStorage]


def prepare_for_saving(
    *tensors: Union[torch.Tensor, QuantizedTensorStorage],
) -> Tuple[
    list[Optional[Union[torch.Tensor, torch.nn.Parameter]]],
    list[Optional[QuantizedTensorStorage]],
]:
    """Prepare tensors for saving. Needed because save_for_backward accepts only
    torch.Tensor/torch.nn.Parameter types, while we want to be able to save
    the internal TensorStorage types too."""

    tensor_list, tensor_objects_list = [], []
    for tensor in tensors:
        if tensor is None or isinstance(tensor, torch.Tensor):
            tensor_list.append(tensor)
            tensor_objects_list.append(None)
        else:
            t, t_obj = tensor.prepare_for_saving()
            tensor_list.extend(t)
            tensor_objects_list.append(t_obj)

    return tensor_list, tensor_objects_list


def restore_from_saved(
    tensors: list[Optional[Union[torch.Tensor, QuantizedTensorStorage]]],
    saved_tensors: list[Optional[Union[torch.Tensor, torch.nn.Parameter]]],
    return_saved_tensors: bool = False,
) -> (
    list[Optional[torch.Tensor | QuantizedTensorStorage]]
    | tuple[
        list[Optional[torch.Tensor | QuantizedTensorStorage]],
        list[Optional[torch.Tensor]],
    ]
):
    """Recombine the tensor data and metadata during backward pass.
    Note: please use `restore_from_func_ctx` instead if you are restoring tensors from a function context to make sure tensor_objects is detached and its memory can be freed
    """
    tensor_objects = []
    for tensor in tensors:
        if tensor is None or isinstance(tensor, torch.Tensor):
            tensor_objects.append(saved_tensors[0])
            saved_tensors = saved_tensors[1:]
        else:
            saved_tensors = tensor.restore_from_saved(saved_tensors)
            tensor_objects.append(tensor)

    if return_saved_tensors:
        return tensor_objects, saved_tensors
    return tensor_objects


def restore_from_func_ctx(ctx: torch.autograd.function.FunctionCtx, return_saved_tensors=False) -> (
    list[Optional[torch.Tensor | QuantizedTensorStorage]]
    | tuple[
        list[Optional[torch.Tensor | QuantizedTensorStorage]],
        list[Optional[torch.Tensor]],
    ]
):
    """Recombine the tensor data and metadata during backward pass and delete tensor objects attached to function context."""
    if not hasattr(ctx, "tensor_objects") or ctx.tensor_objects is None:
        raise AttributeError("ctx must have .tensor_objects to restore saved tensors")
    out = restore_from_saved(
        ctx.tensor_objects, ctx.saved_tensors, return_saved_tensors=return_saved_tensors
    )
    # Delete the references to tensor objects once they've been consumed by the `restore_from_saved` method to construct back the actual tensors.
    ctx.tensor_objects = None
    return out


class Quantizer(abc.ABC):
    """Builder class for quantized tensors.

    This class is typically used to convert a high-precision tensor
    (e.g. in FP32 or BF16) into a quantized tensor (e.g. in FP8).

    """

    """Whether to construct quantized tensors with "row-wise usage"

    Hand-wave explanation: Consider the matrix multiplication C = A *
    B^T (used in linear forward). Tensor Cores prefer "TN GEMMs" (in
    Fortran-style column-major order), so A and B should be in
    row-major order.

    """
    rowwise_usage: bool

    # The :class:`QuantizedTensorStorage` subclass produced by this
    # quantizer's quantize / make_empty path. Consumed by
    # :meth:`create_storage_metadata` to declare a ``("storage", ...)``
    # output payload that round-trips through the generic
    # :meth:`QuantizedTensorStorage._torch_compile_do_unflatten`.
    _storage_cls: type["QuantizedTensorStorage"]

    """Whether to construct quantized tensors with "column-wise usage"

    Hand-wave explanation: Consider the matrix multiplication C = A^T
    * B (used in linear backward wgrad). Tensor Cores prefer "TN
    GEMMs" (in Fortran-style column-major order), so A and B should be
    in column-major order.

    """
    columnwise_usage: bool

    """Whether to instantiates tensor for purely internal usage

    Internal tensors are storage classes with minimal logic. They have
    less overhead than PyTorch tensor sub-classes, but are not
    compatible with PyTorch's autograd infrastructure nor PyTorch
    operations.

    """
    internal: bool

    """Whether to solely optimize for matrix multiplication

    The resulting quantized tensors are not guaranteed to support any
    operation other than matrix multiplication. Use with care since
    this is likely to break communication, checkpointing, and many
    other features.

    """
    optimize_for_gemm: bool

    def __init__(self, *, rowwise: bool, columnwise: bool) -> None:
        self.rowwise_usage = rowwise
        self.columnwise_usage = columnwise
        self.internal = False
        self.optimize_for_gemm = False

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"rowwise_usage={self.rowwise_usage}, "
            f"columnwise_usage={self.columnwise_usage}, "
            f"internal={self.internal}, "
            ")"
        )

    def update_quantized(
        self,
        src: torch.Tensor,
        dst: QuantizedTensor,
        *,
        noop_flag: Optional[torch.Tensor] = None,
    ) -> QuantizedTensor:
        """Quantize tensor in-place"""
        raise NotImplementedError(
            f"{self.__class__.__name__} class does not implement update_quantized"
        )

    def quantize(
        self,
        tensor: torch.Tensor,
        *,
        out: Optional[QuantizedTensor] = None,
        dtype: Optional[torch.dtype] = None,  # pylint: disable=unused-argument # used by override
    ) -> QuantizedTensor:
        """Quantize tensor"""
        if out is not None:
            return self.update_quantized(tensor, out)
        if (not self.internal) and torch.is_grad_enabled():
            return _QuantizeFunc.apply(tensor, self.quantize_impl)
        return _QuantizeFunc.forward(None, tensor, self.quantize_impl)

    def quantize_impl(self, tensor: torch.Tensor) -> QuantizedTensor:
        """Quantize tensor implementation"""
        raise NotImplementedError(
            f"{self.__class__.__name__} class does not implement quantize_impl function"
        )

    def multi_quantize(self, list_of_tensors):
        """Quantize multiple tensors"""
        list_of_output_tensors = []
        for tensor in list_of_tensors:
            list_of_output_tensors.append(self.quantize(tensor))
        return list_of_output_tensors

    def __call__(self, tensor: torch.Tensor) -> QuantizedTensor:
        """Quantize tensor"""
        return self.quantize(tensor)

    def make_empty(
        self,
        shape: Iterable[int],
        *,
        dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None,
    ) -> QuantizedTensor:
        """Construct quantized tensor with uninitialized data"""
        raise NotImplementedError(
            f"{self.__class__.__name__} class does not implement make_empty function, "
            "required for construction of unintialized quantized tensor"
        )

    def calibrate(self, tensor: torch.Tensor) -> None:
        """Calibrate quantizer state

        Updates quantization state as if quantizing a tensor, but
        without actually performing the quantization.

        """

    def set_usage(
        self, *, rowwise: Optional[bool] = None, columnwise: Optional[bool] = None
    ) -> None:
        """Set how the quantized tensor is expected to be used

        See documentation for `rowwise_usage` and `columnwise_usage`
        variables.

        """
        if rowwise is not None:
            self.rowwise_usage = rowwise
        if columnwise is not None:
            self.columnwise_usage = columnwise

    def onnx_quantize(self, tensor: torch.Tensor) -> QuantizedTensor:
        """Symbolic function for ONNX export"""
        raise NotImplementedError(
            f"{self.__class__.__name__} class does not implement onnx_quantize"
        )

    def onnx_dequantize(self, tensor) -> torch.Tensor:
        """Symbolic function for ONNX export"""
        raise NotImplementedError(
            f"{self.__class__.__name__} class does not implement onnx_dequantize"
        )

    def _get_compatible_recipe(self) -> Union[type[Recipe], None]:
        """Returns recipe class that is compatible with this quantizer"""
        raise NotImplementedError(
            f"{self.__class__.__name__} class does not implement _get_compatible_recipe"
        )

    def supports_only_rowwise_all_gather(self) -> bool:
        """Returns True if the quantizer supports only rowwise all-gather"""
        return False

    def is_quantizable(self, inp: torch.Tensor) -> bool:  # pylint: disable=unused-argument
        """Whether tensor supports quantized all-gather

        Consider a less misleading function name.

        """
        return True

    def get_usages(self) -> Dict[str, bool]:
        """Get the usage of the quantizer"""
        return {
            "rowwise": self.rowwise_usage,
            "columnwise": self.columnwise_usage,
        }

    # ------------------------------------------------------------------ #
    # torch.compile flatten / unflatten protocol
    # ------------------------------------------------------------------ #

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        # Auto-register every Quantizer subclass so ``_unflatten`` can
        # dispatch back to it by ``__qualname__``.
        _QUANTIZER_REGISTRY[cls.__qualname__] = cls

    # ---- Declarative schema for the generic :meth:`_flatten` / ---- #
    # ---- :meth:`_do_unflatten` implementations below.            ---- #

    # ``__init__`` kwarg name for ``self.dtype`` (e.g. ``"fp8_dtype"``,
    # ``"fp4_dtype"``).
    _DTYPE_INIT_KWARG: str = "fp8_dtype"

    # Scalar attribute names (besides ``dtype`` / ``rowwise_usage`` /
    # ``columnwise_usage``) threaded through ``__init__``. The kwarg name
    # is assumed to match the attribute name.
    _INIT_META_ATTRS: Tuple[str, ...] = ()

    # Scalar attribute names (besides ``internal`` / ``optimize_for_gemm``)
    # set on the instance after ``__init__``.
    _POST_INIT_META_ATTRS: Tuple[str, ...] = ()

    # Tensor attribute names threaded through ``__init__``, in flatten
    # order.
    _INIT_TENSOR_ATTRS: Tuple[str, ...] = ()

    # Tensor attribute names set on the instance after ``__init__``.
    _POST_INIT_TENSOR_ATTRS: Tuple[str, ...] = ()

    # Attribute name on ``self`` holding the (optional) ``ProcessGroup``,
    # or ``None`` if the quantizer has no PG.
    _PG_ATTR: Optional[str] = None
    # ``__init__`` kwarg name to thread the PG through. ``None`` means
    # set ``_PG_ATTR`` directly after ``__init__``.
    _PG_INIT_KWARG: Optional[str] = None

    # Hardcoded ``__init__`` kwargs not derived from meta (e.g.
    # ``device=torch.device("cuda")`` for ``Float8CurrentScalingQuantizer``).
    _FIXED_INIT_KWARGS: Dict[str, Any] = {}

    def _flatten(
        self,
    ) -> Tuple[Any, Optional["torch.distributed.ProcessGroup"], List[torch.Tensor]]:
        """Pack this quantizer's state into the
        ``(meta, process_group, tensors)`` triplet expected by the
        flattenable bucket in :mod:`transformer_engine.pytorch.dynamo`.

        Generic implementation driven by the declarative schema attrs above.
        Subclasses only declare which scalars / tensors go through
        ``__init__`` vs. are set post-init; the base class round-trips
        ``dtype`` / ``rowwise_usage`` / ``columnwise_usage`` and
        ``internal`` / ``optimize_for_gemm`` on every quantizer.
        """
        from .dynamo import OpaqueSimpleMetadata  # pylint: disable=import-outside-toplevel

        cls = type(self)
        meta_dict: Dict[str, Any] = {
            "_qcls": cls.__qualname__,
            "dtype": self.dtype,
            "rowwise_usage": self.rowwise_usage,
            "columnwise_usage": self.columnwise_usage,
            "internal": self.internal,
            "optimize_for_gemm": self.optimize_for_gemm,
        }
        for attr in (*cls._INIT_META_ATTRS, *cls._POST_INIT_META_ATTRS):
            meta_dict[attr] = getattr(self, attr)
        tensors = [
            getattr(self, attr)
            for attr in (*cls._INIT_TENSOR_ATTRS, *cls._POST_INIT_TENSOR_ATTRS)
        ]
        pg = getattr(self, cls._PG_ATTR) if cls._PG_ATTR else None
        return OpaqueSimpleMetadata(meta_dict), pg, tensors

    @classmethod
    def _do_unflatten(
        cls,
        meta: Any,
        process_group: Optional["torch.distributed.ProcessGroup"],
        tensors: List[torch.Tensor],
    ) -> "Quantizer":
        """Reconstruct an instance of ``cls`` from the triplet returned by a
        previous :meth:`_flatten` on the same subclass. Generic; driven
        by the declarative schema attrs.
        """
        init_kwargs: Dict[str, Any] = {
            cls._DTYPE_INIT_KWARG: meta["dtype"],
            "rowwise": meta["rowwise_usage"],
            "columnwise": meta["columnwise_usage"],
        }
        for attr in cls._INIT_META_ATTRS:
            init_kwargs[attr] = meta[attr]
        if cls._PG_INIT_KWARG is not None:
            init_kwargs[cls._PG_INIT_KWARG] = process_group
        init_kwargs.update(cls._FIXED_INIT_KWARGS)
        tensor_iter = iter(tensors)
        for attr in cls._INIT_TENSOR_ATTRS:
            init_kwargs[attr] = next(tensor_iter)
        q = cls(**init_kwargs)
        q.internal = meta["internal"]
        q.optimize_for_gemm = meta["optimize_for_gemm"]
        for attr in cls._POST_INIT_META_ATTRS:
            setattr(q, attr, meta[attr])
        for attr in cls._POST_INIT_TENSOR_ATTRS:
            setattr(q, attr, next(tensor_iter))
        if cls._PG_ATTR is not None and cls._PG_INIT_KWARG is None:
            setattr(q, cls._PG_ATTR, process_group)
        return q

    @classmethod
    def _unflatten(
        cls,
        meta: Any,
        process_group: Optional["torch.distributed.ProcessGroup"],
        tensors: List[torch.Tensor],
    ) -> "Quantizer":
        """Dispatch to the right subclass's :meth:`_do_unflatten` based on
        the ``"_qcls"`` qualname stored in ``meta``.
        """
        qcls = meta["_qcls"]
        target = _QUANTIZER_REGISTRY.get(qcls)
        if target is None:
            raise ValueError(
                f"No Quantizer subclass registered under qualname {qcls!r}; "
                f"known: {sorted(_QUANTIZER_REGISTRY)}"
            )
        return target._do_unflatten(meta, process_group, tensors)

    def _storage_scalars(self) -> Dict[str, Any]:
        """Per-quantizer scalar fields for the storage's ``_FLATTEN_META_ATTRS``.

        Keys are constructor kwarg names (matching the values of
        :attr:`QuantizedTensorStorage._FLATTEN_CTOR_KWARG`). ``fake_dtype``
        is supplied separately by :meth:`create_storage_metadata`; subclasses
        only need to return their quantizer-specific scalars (e.g.
        ``fp8_dtype``, ``with_gemm_swizzled_scales``).
        """
        raise NotImplementedError(
            f"{type(self).__name__} class does not implement _storage_scalars; "
            "required for torch.compile output specs that emit a "
            "QuantizedTensorStorage."
        )

    def create_storage_metadata(
        self,
        *,
        shape: Iterable[int],
        fake_dtype: torch.dtype,
        device: Optional[torch.device] = None,
    ) -> Tuple[type["QuantizedTensorStorage"], Any, Optional[Any], int]:
        """Return ``(cls, meta, process_group, tensor_count)`` describing
        the ``("storage", ...)`` payload of a Dynamo output spec.

        The Dynamo layer hands the trailing
        ``(meta, process_group, tensors[: tensor_count])`` triple to
        :meth:`QuantizedTensorStorage._torch_compile_do_unflatten` to
        reconstruct the freshly-quantized storage on the consumer side.

        Driven entirely by the storage's ``_FLATTEN_*`` schema plus a
        per-quantizer :meth:`_storage_scalars` hook; ``has_*`` flags are
        derived from ``rowwise_usage`` / ``columnwise_usage`` and the
        storage's :attr:`QuantizedTensorStorage._FLATTEN_TENSOR_USAGE`
        map. Quantizers with tensor state (e.g. :class:`Float8Quantizer`'s
        ``scale`` / ``amax``) append those tensors after the storage's own
        slots; :meth:`Quantizer._flatten` provides both the count and the
        ``quantizer_meta`` payload needed to rebuild the quantizer.
        """
        from .dynamo import OpaqueSimpleMetadata  # pylint: disable=import-outside-toplevel

        if device is None:
            device = torch.device("cuda")
        del device, shape  # storage-only path: no outer tensor view
        storage_cls = type(self)._storage_cls
        usage_flag = {
            "rowwise": self.rowwise_usage,
            "columnwise": self.columnwise_usage,
            "always": True,
        }
        has_flags: Dict[str, bool] = {}
        tensor_count = 0
        for attr in storage_cls._FLATTEN_TENSOR_ATTRS:
            usage = storage_cls._FLATTEN_TENSOR_USAGE.get(attr, "always")
            flag = usage_flag[usage]
            has_flags[storage_cls._flatten_presence_key(attr)] = flag
            if flag:
                tensor_count += 1
        quantizer_meta, _, quantizer_tensors = self._flatten()
        tensor_count += len(quantizer_tensors)
        scalars = self._storage_scalars()
        scalars["fake_dtype"] = fake_dtype
        meta = OpaqueSimpleMetadata(
            {
                "_qstorage_cls": storage_cls.__qualname__,
                **scalars,
                **has_flags,
                "quantizer_meta": quantizer_meta,
            }
        )
        return storage_cls, meta, None, tensor_count


class QuantizedTensor(torch.Tensor):
    """Abstract base class for tensor with quantized data

    This is a proxy class with the interface of a standard PyTorch
    tensor, but with data that has been encoded with some quantization
    scheme. Derived classes should implement the quantization scheme
    by overriding the `quantize_` and `dequantize` functions.

    """

    def __new__(
        cls,
        shape: Iterable[int],
        dtype: torch.dtype,
        *,
        fake_dtype: Optional[torch.dtype] = None,
        requires_grad: bool = False,
        device: Optional[torch.device] = None,
        stride: Optional[Iterable[int]] = None,
    ):
        if fake_dtype is not None and fake_dtype != dtype:
            raise ValueError(f"fake_dtype ({fake_dtype}) does not match dtype ({dtype})")
        # For stride, We are assuming only contiguous tensors
        # Calculate stride from shape if not provided. When creating this object from
        # C++ code, we provide the stride computed from shape in C++ to avoid the
        # PyobjectVectorCall overhead of calling _stride_from_shape from C++ to Python.
        stride = _stride_from_shape(shape) if stride is None else stride
        instance = torch.Tensor._make_wrapper_subclass(
            cls,
            shape,
            strides=stride,
            storage_offset=0,
            dtype=dtype,
            layout=torch.strided,
            requires_grad=requires_grad,
            device=torch.cuda.current_device() if device is None else device,
        )
        instance._requires_grad = requires_grad
        instance._dtype = dtype
        return instance

    @property
    def dtype(self) -> torch.dtype:
        """
        Return the high precision data type of the tensor
        Attribute access of custom tensors goes through an
        expensive Pyobject lookup. Since dtype for a tensor is never
        change after creation, we cache it in a member variable and return
        """
        # Lazy initialization for tensors created via alternate paths
        if not hasattr(self, "_dtype"):
            # pylint: disable=unnecessary-dunder-call
            self._dtype = torch._C.TensorBase.dtype.__get__(self, type(self))
        return self._dtype

    @dtype.setter
    def dtype(self, value: torch.dtype) -> None:
        """Set dtype property"""
        self._dtype = value

    @property
    def requires_grad(self) -> bool:
        """
        Return whether or not the tensor requires gradient.
        Attribute access of custom tensors goes through an
        expensive Pyobject lookup. Since requires_grad is set during
        initialization and may be updated, we cache it in a member variable.
        """
        # Fallback to parent if not cached yet
        if not hasattr(self, "_requires_grad"):
            # pylint: disable=unnecessary-dunder-call
            self._requires_grad = torch._C.TensorBase.requires_grad.__get__(self, type(self))
        return self._requires_grad

    @requires_grad.setter
    def requires_grad(self, value: bool) -> None:
        """Set requires_grad property so that autograd engine is aware of the change"""
        # Update the cached value and call parent class method to ensure autograd engine is aware
        self.requires_grad_(value)

    def requires_grad_(self, requires_grad: bool = True) -> QuantizedTensor:
        """Cache requires_grad property and call parent class method"""
        # pylint: disable=missing-function-docstring
        # Update the cached value
        self._requires_grad = requires_grad
        # Call parent class method to ensure autograd engine is aware
        super().requires_grad_(requires_grad)
        return self

    def _get_data(self) -> torch.Tensor:
        """Get tensor data property"""
        return super().data

    def _set_data(self, tensor: torch.Tensor) -> None:
        """Set tensor data property
        Updates the underlying tensor data and syncs the dtype cache.
        """
        # Update the parent class's data descriptor
        # pylint: disable=unnecessary-dunder-call
        super(QuantizedTensor, type(self)).data.__set__(self, tensor)
        # Update the dtype cache
        self._dtype = tensor.dtype

    # Create the data property with getter and setter
    data = property(_get_data, _set_data)

    def dequantize(self, *, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        """Convert quantized data to standard PyTorch tensor"""
        raise NotImplementedError(
            f"{self.__class__.__name__} class does not implement dequantize function"
        )

    def quantize_(self, tensor: torch.Tensor) -> QuantizedTensor:
        """Update quantized data in-place"""
        raise NotImplementedError(
            f"{self.__class__.__name__} class does not implement quantize_ function"
        )

    def detach(self) -> QuantizedTensor:
        """Create new quantized tensor with same data

        Output tensor must be detached from the current autograd
        graph.

        """
        raise NotImplementedError(
            f"{self.__class__.__name__} class does not implement detach function"
        )

    def clear(self):
        """Deallocate this tensor's memory. Typically not needed and must be used carefully"""
        raise NotImplementedError(
            f"{self.__class__.__name__} class does not implement clear function"
        )

    def __repr__(self, *, tensor_contents=None) -> str:
        return f"{self.__class__.__name__}(data={self.dequantize()})"

    def float(self) -> torch.Tensor:
        # pylint: disable=missing-function-docstring
        return self.dequantize(dtype=torch.float32)

    def bfloat16(self) -> torch.Tensor:
        # pylint: disable=missing-function-docstring
        return self.dequantize(dtype=torch.bfloat16)

    def half(self) -> torch.Tensor:
        # pylint: disable=missing-function-docstring
        return self.dequantize(dtype=torch.float16)

    def cpu(self, memory_format=torch.preserve_format) -> torch.Tensor:
        # pylint: disable=missing-function-docstring
        return self.dequantize().cpu(memory_format=memory_format)

    def expand_as(self, other: torch.Tensor) -> torch.Tensor:
        # pylint: disable=missing-function-docstring
        if other is self:
            # Note: expand_as is hackily used to create dummy autograd nodes
            # and access the backward graph (see
            # https://github.com/pytorch/pytorch/blob/238fb660851268f44ff88127887041fea352fe48/torch/nn/parallel/distributed.py#L1026).
            # We hackily add a dummy function to handle this case.
            return _IdentityFunc.apply(self)
        return super().expand_as(other)

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs=None):

        # Detach op
        if func == torch.ops.aten.detach.default:
            return args[0].detach()

        # In-place copy op
        if func == torch.ops.aten.copy_.default:
            dst = args[0]
            src = args[1]
            if (
                isinstance(dst, QuantizedTensor)
                and isinstance(src, QuantizedTensor)
                and type(dst._quantizer) is type(src._quantizer)
                and set(src.get_usages().keys()) == set(dst.get_usages().keys())
                and all(
                    src.get_usages()[usage] == dst.get_usages()[usage]
                    for usage in src.get_usages().keys()
                )
            ):

                dst_tensors, dst_tensor_obj = dst.prepare_for_saving()
                src_tensors, src_tensor_obj = src.prepare_for_saving()
                for dst_tensor, src_tensor in zip(dst_tensors, src_tensors):
                    if dst_tensor is not None:
                        dst_tensor.copy_(src_tensor, *args[2:], **kwargs)
                dst_tensor_obj.restore_from_saved(dst_tensors)
                src_tensor_obj.restore_from_saved(src_tensors)
                return None

            if isinstance(dst, QuantizedTensor):
                dst.quantize_(src)
            else:
                if isinstance(src, QuantizedTensor):
                    dtype = dst.dtype
                    if dtype not in (torch.float32, torch.float16, torch.bfloat16):
                        dtype = torch.float32
                    src = src.dequantize(dtype=dtype)
                dst.copy_(src)
            return None

        # View op
        if func == torch.ops.aten.view.default:
            raise NotImplementedError("{cls.__name__} class does not support tensor views")

        # New empty op (used by DCP async staging to create CPU copies)
        if func == torch.ops.aten.new_empty.default:
            tensor = args[0]
            size = args[1]
            dtype = kwargs.get("dtype", tensor.dtype)
            device = kwargs.get("device", tensor.device)
            pin_memory = kwargs.get("pin_memory", False)
            if tensor._quantizer is None:
                raise RuntimeError(
                    f"{type(tensor).__name__} does not have a quantizer; "
                    "cannot create new_empty QuantizedTensor"
                )
            out = tensor._quantizer.make_empty(
                shape=torch.Size(size),
                dtype=dtype,
                device=device,
                requires_grad=tensor.requires_grad,
                pin_memory=pin_memory,
            )
            return out

        # Empty like op
        if func == torch.ops.aten.empty_like.default:
            tensor = args[0]
            device = kwargs.get("device", tensor.device)
            requires_grad = kwargs.get("requires_grad", tensor.requires_grad)
            pin_memory = kwargs.get("pin_memory", False)
            usage = tensor.get_usages()
            quantizer_usage = tensor._quantizer.get_usages()
            tensor._quantizer.set_usage(**usage)
            out = tensor._quantizer.make_empty(
                shape=tensor.shape,
                dtype=tensor.dtype,
                device=device,
                requires_grad=requires_grad,
                pin_memory=pin_memory,
            )
            tensor._quantizer.set_usage(**quantizer_usage)
            return out

        if func == torch.ops.aten.numel.default:
            tensor = args[0]
            return math.prod(tensor.size())

        if func == torch.ops.aten.is_pinned.default:
            tensor = args[0]
            for t in tensor.get_data_tensors():
                if t is not None:
                    return func(t)
            return False  # Or error out?

        # Pass through registered custom ops without unwrapping
        if func in _quantized_tensor_passthrough_ops:
            if kwargs is None:
                kwargs = {}
            return super().__torch_dispatch__(func, types, args, kwargs)

        def maybe_unwrap(arg):
            if isinstance(arg, QuantizedTensor):
                return arg.dequantize()
            return arg

        def maybe_update_inplace(arg, new_arg, schema_arg):
            if (
                isinstance(arg, QuantizedTensor)
                and isinstance(new_arg, torch.Tensor)
                and hasattr(schema_arg, "alias_info")
                and hasattr(schema_arg.alias_info, "is_write")
                and schema_arg.alias_info.is_write
            ):
                arg.quantize_(new_arg)
            elif isinstance(arg, list) and isinstance(new_arg, list):
                # Recursively handle update for lists of tensors
                for a, na in zip(arg, new_arg):
                    maybe_update_inplace(a, na, schema_arg)

        # In-place op: dequantize, perform op, and quantize
        if func._schema.is_mutable:
            new_args = tree_map(maybe_unwrap, args)
            new_kwargs = tree_map(maybe_unwrap, kwargs)
            schema_args = func._schema.arguments
            args_len = len(args)
            super().__torch_dispatch__(func, types, new_args, new_kwargs)
            for arg, new_arg, schema_arg in zip(args, new_args, schema_args):
                maybe_update_inplace(arg, new_arg, schema_arg)
            for kwarg, new_kwarg, schema_arg in zip(kwargs, new_kwargs, schema_args[args_len:]):
                assert kwarg == new_kwarg == schema_arg.name, "name of the kw argument should match"
                maybe_update_inplace(kwargs[kwarg], new_kwargs[new_kwarg], schema_arg)
            return None

        # Default op: dequantize and perform op
        args = tree_map(maybe_unwrap, args)
        if kwargs is not None:
            kwargs = tree_map(maybe_unwrap, kwargs)
        out = super().__torch_dispatch__(func, types, args, kwargs)
        return out

    # Set as a class-level attribute rather than a ``@classmethod`` so that
    # Dynamo recognises the canonical "torch_function disabled" idiom
    # and can trace through custom-op calls that receive a
    # QuantizedTensor subclass as an argument. As a method override,
    # Dynamo bails with "cannot trace builtin
    # torch._C._disabled_torch_function_impl".
    __torch_function__ = torch._C._disabled_torch_function_impl

    def contiguous(
        self, memory_format: torch.memory_format = torch.contiguous_format
    ) -> QuantizedTensor:
        # pylint: disable=missing-function-docstring
        raise NotImplementedError(
            f"{self.__class__.__name__} class does not implement contiguous function"
        )

    def get_metadata(self) -> Dict[str, Any]:
        """Get keyword arguments for quantized tensor constructor

        Contains metadata so that the new quantized tensor has the
        same underlying quantized data.

        """
        raise NotImplementedError(
            f"{self.__class__.__name__} class does not implement get_metadata function"
        )

    # ------------------------------------------------------------------ #
    # PyTorch wrapper-subclass flatten / unflatten                       #
    # ------------------------------------------------------------------ #
    #
    # Driven by the same ``_FLATTEN_*_ATTRS`` / ``_FLATTEN_CTOR_KWARG``
    # declarations as :meth:`QuantizedTensorStorage._torch_compile_flatten`,
    # plus the :meth:`_flatten_meta_overrides` hook (Float8Tensor uses it
    # to bridge :class:`FP8DType` <-> ``tex.DType``).
    #
    # Per-subclass differences vs the storage path: PyTorch's protocol
    # carries only attributes living on ``self`` (no quantizer tensors,
    # no process group). Quantizers whose state contains tensors (e.g.
    # :class:`Float8Quantizer`'s ``scale`` / ``amax``,
    # :class:`NVFP4Quantizer`'s ``rht_matrix``) therefore round-trip via
    # :func:`_quantizer_subclass_snapshot`, which bails to ``None``; the
    # reconstructed tensor's ``_quantizer`` is ``None`` and downstream
    # code that needs the live quantizer sources it from the bucket-level
    # opaque metadata flowing alongside the inner op.

    def __tensor_flatten__(self) -> Tuple[list, dict]:
        if not type(self)._FLATTEN_TENSOR_ATTRS:
            raise NotImplementedError(
                f"{type(self).__name__} did not declare _FLATTEN_TENSOR_ATTRS"
            )
        inner: list = [
            attr for attr in self._FLATTEN_TENSOR_ATTRS if getattr(self, attr) is not None
        ]
        meta: Dict[str, Any] = {
            "quantizer_snapshot": _quantizer_subclass_snapshot(self._quantizer),
            "requires_grad": self.requires_grad,
        }
        for attr in self._FLATTEN_META_ATTRS:
            meta[self._flatten_ctor_kw(attr)] = getattr(self, attr)
        return inner, meta

    @classmethod
    def __tensor_unflatten__(
        cls,
        inner_tensors: dict,
        meta: dict,
        outer_size,
        outer_stride,
    ) -> "QuantizedTensor":
        meta = cls._flatten_meta_overrides(meta)
        quantizer = _quantizer_from_subclass_snapshot(meta.get("quantizer_snapshot"))
        kwargs: Dict[str, Any] = {
            "shape": outer_size,
            "dtype": meta["fake_dtype"],
            "requires_grad": meta.get("requires_grad", False),
            "quantizer": quantizer,
        }
        for attr in cls._FLATTEN_TENSOR_ATTRS:
            kw = cls._flatten_ctor_kw(attr)
            kwargs[kw] = inner_tensors.get(attr)
        for attr in cls._FLATTEN_META_ATTRS:
            kw = cls._flatten_ctor_kw(attr)
            kwargs[kw] = meta[kw]
        return cls(**kwargs)

    @classmethod
    def make_like(
        cls,
        tensor: QuantizedTensor,
        *,
        shape: Optional[Iterable[int]] = None,
        dtype: Optional[torch.dtype] = None,
        requires_grad: bool = False,
    ) -> QuantizedTensor:
        """Create new quantized tensor

        By default, new tensor has the same attributes and underlying
        data. This function is intended to create view of tensors.

        """
        shape = shape if shape is not None else tensor.shape
        dtype = dtype if dtype is not None else tensor.dtype
        kwargs = tensor.get_metadata()
        kwargs["fake_dtype"] = dtype
        return cls(shape=shape, dtype=dtype, requires_grad=requires_grad, **kwargs)

    def to_dtype(self, dtype: torch.dtype) -> QuantizedTensor:
        """Create `QuantizedTensor` with given nominal dtype

        The new tensor has the same underlying data.

        """
        return self.__class__.make_like(self, dtype=dtype)
