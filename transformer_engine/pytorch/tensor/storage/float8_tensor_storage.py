# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Mixin class holding data specific for Float8Tensor"""

from __future__ import annotations
import math
from typing import Any, Dict, List, Optional, Tuple
import torch

import transformer_engine_torch as tex
from transformer_engine_torch import DType as TE_DType

from ...quantized_tensor import QuantizedTensorStorage, Quantizer

from ...constants import TE_DType as torch_to_transformer_engine_dtype, TE_DType_To_Torch

from ...utils import is_non_tn_fp8_gemm_supported, _empty_tensor

try:
    from torch._library.opaque_object import is_opaque_value_type, register_opaque_type

    if not hasattr(TE_DType, "__fx_repr__"):
        TE_DType.__fx_repr__ = lambda self: (f"TE_DType({int(self)})", {"TE_DType": TE_DType})
    if not is_opaque_value_type(TE_DType):
        register_opaque_type(TE_DType, typ="value", members={})
except Exception:  # pragma: no cover - older torch / partial init
    pass


class _FromFloat8Func(torch.autograd.Function):
    """Cast from FP8 to other dtype"""

    @staticmethod
    def forward(
        _ctx: Optional[torch.autograd.function.FunctionCtx],  # unused
        tensor: Float8TensorStorage,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        # pylint: disable=missing-function-docstring
        te_dtype = torch_to_transformer_engine_dtype[dtype]

        # Make sure FP8 data is in expected format
        if tensor._data is not None:
            if tensor._data.numel() == 0:
                return torch.empty_like(tensor._data, dtype=dtype)
            if tensor._data.is_cpu:
                # CPU fallback: reinterpret uint8 as FP8, cast to target dtype, scale
                fp8_torch_dtype = TE_DType_To_Torch[tensor._fp8_dtype]
                return (
                    tensor._data.view(fp8_torch_dtype).float()
                    * tensor._scale_inv.to(tensor._data.device)
                ).to(dtype)
            # Cast from FP8
            return tex.dequantize(tensor, te_dtype)

        raise NotImplementedError("Casting back from the transpose not implemented yet!")

    @staticmethod
    def backward(
        _ctx: torch.autograd.function.FunctionCtx,  # unused
        grad: torch.Tensor,
    ) -> Tuple[Optional[torch.Tensor], ...]:
        # pylint: disable=missing-function-docstring
        # Assume that we want gradients in full precision
        return grad, None


class Float8TensorStorage(QuantizedTensorStorage):
    """Mixin class that holds data attributes of Float8Tensor.

    Float8Tensor inherits from the PyTorch tensor class and this mixin
    class. If this class is instantiated directly, it has the same
    data, lower CPU overhead, and less functionality. It should only
    be instantiated directly for performance-critical internal usage.

    """

    _data: Optional[torch.Tensor]
    _quantizer: Optional[Quantizer]
    _fp8_dtype: TE_DType
    _scale_inv: torch.Tensor

    # FP8 transpose cache
    _transpose: Optional[torch.Tensor]
    _transpose_invalid: bool

    # Upper bound on the number of inner tensors produced by
    # :meth:`_torch_compile_flatten`. Used by the wide-output layout in
    # :mod:`transformer_engine.pytorch.dynamo` to reserve enough slots in
    # the custom-op ``Tensor[]`` return for any storage-shaped output:
    # 3 data tensors (data / transpose / scale_inv) + up to 2 quantizer
    # tensors (Float8Quantizer carries scale / amax).
    _TORCH_COMPILE_MAX_INNER_TENSORS = 5

    def __new__(
        cls,
        *args,
        data: Optional[torch.Tensor],
        fp8_scale_inv: torch.Tensor,
        fp8_dtype: TE_DType,
        fake_dtype: Optional[torch.dtype] = None,
        data_transpose: Optional[torch.Tensor] = None,
        quantizer: Optional[Quantizer] = None,
        **kwargs,
    ):
        if cls is Float8TensorStorage:
            instance = object.__new__(cls)
            instance._dtype = fake_dtype if fake_dtype is not None else torch.float32
        else:
            instance = super().__new__(cls, *args, fake_dtype=fake_dtype, **kwargs)
        instance._data = data
        instance._quantizer = quantizer.copy() if quantizer is not None else None
        instance._fp8_dtype = fp8_dtype
        instance._scale_inv = fp8_scale_inv
        instance._transpose = data_transpose
        instance._transpose_invalid = instance._transpose is None

        return instance

    def clear(self):
        """Deallocate this tensor's memory. Typically not needed and must be used carefully.

        Scale-inv tensor is not deallocated because it's often shared
        between multiple FP8 tensors.

        """
        for t in (self._data, self._transpose):
            if t is not None:
                t.data = _empty_tensor()
        self._transpose_invalid = True

    def copy_from_storage(self, src: QuantizedTensorStorage) -> None:
        """Copy data buffers from another Float8TensorStorage."""
        if not isinstance(src, Float8TensorStorage):
            raise TypeError("copy_from_storage expects Float8TensorStorage")
        if self._fp8_dtype != src._fp8_dtype:
            raise RuntimeError("FP8 dtype mismatch in copy_from_storage")

        def _copy_optional(
            dst: Optional[torch.Tensor],
            src_tensor: Optional[torch.Tensor],
        ):
            if dst is not None and src_tensor is not None:
                dst.copy_(src_tensor)

        _copy_optional(self._data, src._data)
        _copy_optional(self._transpose, src._transpose)
        _copy_optional(self._scale_inv, src._scale_inv)

    def get_metadata(self) -> Dict[str, Any]:
        """Get this tensor's metadata."""
        return {
            "data": self._data,
            "fp8_scale_inv": self._scale_inv,
            "fp8_dtype": self._fp8_dtype,
            "data_transpose": self._transpose,
            "quantizer": self._quantizer,
            "device": (
                self._data.device
                if self._data is not None
                else (self._transpose.device if self._transpose is not None else None)
            ),
            "fake_dtype": self._dtype,
        }

    def prepare_for_saving(self) -> Tuple[list[Optional[torch.Tensor]], QuantizedTensorStorage]:
        """Prepare the tensor base for saving for backward"""
        tensors = [self._data, self._transpose, self._scale_inv]
        self._data = None
        self._transpose = None
        self._scale_inv = None
        return tensors, self

    def restore_from_saved(
        self, tensors: list[Optional[torch.Tensor]]
    ) -> list[Optional[torch.Tensor]]:
        """Restore the tensor base data from the saved tensors list"""
        self._data = tensors[0]
        self._transpose = tensors[1]
        self._scale_inv = tensors[2]
        # Re-derive ``_transpose_invalid`` from the restored buffer:
        # the saved transpose, if present, was valid at save time
        # (``prepare_for_saving`` never resets this flag, and forward
        # producers don't save stale transposes). Tying the flag to
        # ``self._transpose`` here makes restoration independent of
        # whichever shell carried the storage across the trace
        # boundary -- in particular ``torch.compile``'s save/restore
        # round-trip, which builds a fresh wrapper shell for backward
        # whose pre-restore ``_transpose_invalid`` would otherwise
        # come from :meth:`Float8TensorStorage.__new__` (``True``
        # whenever it sees ``data_transpose=None``) and trip
        # :meth:`update_usage` downstream.
        self._transpose_invalid = self._transpose is None
        return tensors[3:]

    def get_data_tensors(self, rowwise_data: bool = True, columnwise_data: bool = True):
        """Get this Tensor's data."""
        if rowwise_data and columnwise_data:
            return self._data, self._transpose
        if rowwise_data:
            return self._data
        if columnwise_data:
            return self._transpose
        raise ValueError("No data to get, both rowwise_data and columnwise_data are False")

    def dequantize(self, *, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        """Dequantize to a higher precision."""
        if dtype is None:
            dtype = self._dtype
        return _FromFloat8Func.forward(None, self, dtype)

    def size(self, *args, **kwargs):
        # pylint: disable=missing-function-docstring
        if self._data is not None:
            return self._data.size(*args, **kwargs)
        size = self._transpose.size(*args, **kwargs)
        return torch.Size([size[-1], math.prod(size[:-1])])

    @property
    def device(self):
        """Return the device of the tensor. Define this to avoid expensive PyObject lookups."""
        if self._data is not None:
            return self._data.device
        if self._transpose is not None:
            return self._transpose.device
        raise RuntimeError("Float8TensorStorage has no data!")

    def view(self, shape: torch.Size):
        # pylint: disable=missing-function-docstring
        out_data = self._data.view(shape)
        out_transpose = None if self._transpose_invalid else self._transpose
        if out_transpose is not None:
            out_transpose_shape = out_transpose.size()
            if out_transpose_shape[0] != shape[-1] or out_transpose_shape[1:] != shape[:-1]:
                out_transpose = None

        return Float8TensorStorage(
            data=out_data,
            fp8_scale_inv=self._scale_inv,
            fp8_dtype=self._fp8_dtype,
            fake_dtype=self._dtype,
            data_transpose=out_transpose,
            quantizer=self._quantizer,
        )

    def __repr__(self):
        # Must never raise: this runs from Inductor error formatters,
        # FX node dumps, Dynamo guards, etc. Crucially we must also
        # avoid any tensor->scalar materialization (``.item()``,
        # ``.tolist()``, ``dequantize()``): under fake-tensor mode they
        # allocate fresh unbacked symbols which then leak out of the
        # current op as "unreturned outputs" and crash the compile.
        # Stick to shape/dtype summaries.
        scale_shape = list(getattr(self._scale_inv, "shape", ()))
        if self._data is None:
            data_repr = "<no rowwise data (transpose-only)>"
        else:
            data_shape = list(getattr(self._data, "shape", ()))
            data_repr = f"<fp8 data shape={data_shape}>"
        return (
            "Float8TensorStorage("
            f"fp8_dtype={self._fp8_dtype}, "
            f"scale_inv=<shape={scale_shape}>, "
            f"data={data_repr}"
            ")"
        )

    def _torch_compile_flatten(self) -> Tuple[Any, Any, List[torch.Tensor]]:
        from transformer_engine.pytorch.dynamo import OpaqueSimpleMetadata

        tensors: List[torch.Tensor] = []

        def _append_if_present(tensor: Optional[torch.Tensor]) -> bool:
            if tensor is None:
                return False
            tensors.append(tensor)
            return True

        quantizer_meta = None
        process_group = None
        quantizer_tensors: List[torch.Tensor] = []
        if self._quantizer is not None:
            quantizer_meta, process_group, quantizer_tensors = self._quantizer._flatten()

        meta = OpaqueSimpleMetadata(
            {
                "_qstorage_cls": type(self).__qualname__,
                "is_tensor": isinstance(self, torch.Tensor),
                "shape": torch.Size(self.shape) if isinstance(self, torch.Tensor) else None,
                "requires_grad": self.requires_grad if isinstance(self, torch.Tensor) else False,
                "device": self.device if isinstance(self, torch.Tensor) else None,
                "fp8_dtype": self._fp8_dtype,
                "fake_dtype": self._dtype,
                "transpose_invalid": self._transpose_invalid,
                "has_data": _append_if_present(self._data),
                "has_transpose": _append_if_present(self._transpose),
                "has_scale_inv": _append_if_present(self._scale_inv),
                "quantizer_meta": quantizer_meta,
            }
        )
        tensors.extend(quantizer_tensors)
        return meta, process_group, tensors

    @classmethod
    def _torch_compile_do_unflatten(
        cls,
        meta: Any,
        process_group: Any,
        tensors: List[torch.Tensor],
    ) -> "Float8TensorStorage":
        tensor_iter = iter(tensors)
        data = next(tensor_iter) if meta["has_data"] else None
        transpose = next(tensor_iter) if meta["has_transpose"] else None
        scale_inv = next(tensor_iter) if meta["has_scale_inv"] else None
        quantizer = None
        if meta["quantizer_meta"] is not None:
            quantizer = Quantizer._unflatten(
                meta["quantizer_meta"], process_group, list(tensor_iter)
            )
        kwargs = {
            "data": data,
            "fp8_scale_inv": scale_inv,
            "fp8_dtype": meta["fp8_dtype"],
            "data_transpose": transpose,
            "quantizer": quantizer,
            "fake_dtype": meta["fake_dtype"],
        }
        if meta["is_tensor"]:
            kwargs.update(
                {
                    "shape": meta["shape"],
                    "dtype": meta["fake_dtype"],
                    "requires_grad": meta["requires_grad"],
                    "device": meta["device"],
                }
            )
        out = cls(**kwargs)
        # ``__new__`` already sets ``_transpose_invalid = (data_transpose
        # is None)``, which is exactly the post-restoration semantic we
        # want under :mod:`torch.compile`: a transpose buffer that the
        # producer chose to ship through the trace was valid at flatten
        # time (forward never emits stale transposes onto saved
        # tensors), so the unflattened storage must treat it as valid.
        # Trusting ``meta["transpose_invalid"]`` instead would re-pin the
        # stale ``True`` that Dynamo embeds into the metadata constant
        # because it cannot follow the in-place
        # :meth:`restore_from_saved` write through ``ctx.tensor_objects``
        # and would then fail the :meth:`update_usage`
        # ``not has_data_transpose`` guard in backward.
        return out

    def _create_transpose(self):
        """Update FP8 transpose cache"""
        data = self._data
        if not data.is_contiguous():
            data = data.contiguous()
        self._transpose = tex.fp8_transpose(data, self._fp8_dtype, out=self._transpose)
        self._transpose_invalid = False

    def update_usage(
        self,
        rowwise_usage: Optional[bool] = None,
        columnwise_usage: Optional[bool] = None,
    ):
        """
        Generate or remove FP8 data based on provided usage. For
        FP8, data cannot be generated even if transpose is available.
        """
        has_data = self._data is not None
        has_data_transpose = self._transpose is not None and not self._transpose_invalid
        needs_data = has_data
        needs_data_transpose = has_data_transpose
        if is_non_tn_fp8_gemm_supported():
            if rowwise_usage is not None and rowwise_usage:
                needs_data = True
            if columnwise_usage is not None and columnwise_usage:
                needs_data = True
            needs_data_transpose = False
        else:
            if rowwise_usage is not None:
                needs_data = rowwise_usage
            if columnwise_usage is not None:
                needs_data_transpose = columnwise_usage

        # Generate data that is required
        if needs_data and not has_data:
            raise RuntimeError("Cannot generate FP8 data, even from FP8 data transpose")
        if needs_data_transpose and not has_data_transpose:
            if not has_data:
                raise RuntimeError("FP8 data is required to generate FP8 data transpose")
            self._create_transpose()

        # Delete data that is not required
        if not needs_data:
            self._data = None
        if not needs_data_transpose:
            self._transpose = None
            self._transpose_invalid = True

    def get_usages(self) -> Dict[str, bool]:
        """Get the usage of the tensor"""
        usages = {"rowwise": self._data is not None}
        if is_non_tn_fp8_gemm_supported():
            usages["columnwise"] = self._data is not None
        else:
            usages["columnwise"] = self._transpose is not None and not self._transpose_invalid
        return usages
