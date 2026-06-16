# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Mixin class holding data specific for Float8Tensor"""

from __future__ import annotations
import math
from typing import Any, Dict, Optional, Tuple, Union
import torch

import transformer_engine_torch as tex

from ...quantized_tensor import QuantizedTensorStorage, Quantizer

from ...constants import TE_DType as torch_to_transformer_engine_dtype, TE_DType_To_Torch, DType

from ...utils import is_non_tn_fp8_gemm_supported, _empty_tensor


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
    _fp8_dtype: DType
    _scale_inv: torch.Tensor

    # FP8 transpose cache
    _transpose: Optional[torch.Tensor]
    _transpose_invalid: bool

    def __new__(
        cls,
        *args,
        data: Optional[torch.Tensor],
        fp8_scale_inv: torch.Tensor,
        fp8_dtype: Union[DType, tex.DType],
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
        instance._fp8_dtype = DType.cast(fp8_dtype)
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
        out_data = self._data.view(shape) if self._data is not None else None
        if out_data is not None:
            out_shape = out_data.size()
        else:
            out_shape = torch.empty(tuple(self.size()), device="meta").view(shape).shape
        out_transpose = None if self._transpose_invalid else self._transpose
        if out_transpose is not None:
            if len(out_shape) == 0:
                view_shape_for_transpose = out_shape
            else:
                view_shape_for_transpose = torch.Size((out_shape[-1], *out_shape[:-1]))
            if out_transpose.shape != view_shape_for_transpose:
                if self._data is None:
                    raise NotImplementedError(
                        "Float8TensorStorage view with columnwise-only data is only "
                        "supported when the requested shape preserves the columnwise layout"
                    )
                out_transpose = None
            else:
                out_transpose = out_transpose.view(*view_shape_for_transpose)
        if self._data is None and out_transpose is None:
            raise NotImplementedError(
                "Float8TensorStorage view with columnwise-only data requires a valid "
                "columnwise buffer"
            )

        return Float8TensorStorage(
            data=out_data,
            fp8_scale_inv=self._scale_inv,
            fp8_dtype=self._fp8_dtype,
            fake_dtype=self._dtype,
            data_transpose=out_transpose,
            quantizer=self._quantizer,
        )

    def __repr__(self):
        return (
            "Float8TensorStorage("
            f"fp8_dtype={self._fp8_dtype}, "
            f"scale_inv={self._scale_inv.item()}, "
            f"data={self.dequantize()}"
            ")"
        )

    def _create_transpose(self):
        """Update FP8 transpose cache"""
        data = self._data
        # Columnwise-only Float8Tensors (e.g. hybrid quantization sub-storages)
        # have _data=None — nothing to transpose.
        if data is None:
            return
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

    def fsdp_buffer_fields(self) -> Tuple[str, ...]:
        """Fields gathered by FSDP2 for per-tensor FP8.

        ``_scale_inv`` is a per-tensor scalar; it travels through the hook's
        metadata tuple (mirroring :meth:`Float8Tensor.fsdp_pre_all_gather`).

        Direction-aware: a vanilla Float8Tensor parameter has ``_data``
        populated, but a columnwise-only sub-storage (used inside
        ``HybridQuantizedTensor`` on Hopper / L40 where non-TN FP8 GEMM is
        not natively supported) holds its quantized data in ``_transpose``
        instead. Returning ``("_data",)`` unconditionally would have
        ``fsdp_extract_buffers`` produce ``(None,)`` and FSDP2 would
        all-gather a ``None`` tensor.

        The per-sub-storage direction is fixed at construction (pinned by
        ``HybridQuantizer.__init__`` via ``set_usage``), so this check is
        stable across iterations even though it inspects the current
        field state.
        """
        if self._data is not None:
            return ("_data",)
        if self._transpose is not None:
            return ("_transpose",)
        # Degenerate: fully empty storage. Fall back to ``_data`` so the
        # base ``fsdp_extract_buffers`` returns ``(None,)`` — same surface
        # the caller would have seen pre-direction-aware logic.
        return ("_data",)

    def fsdp_assign_gathered(
        self,
        gathered: Tuple[Optional[torch.Tensor], ...],
        meta: Dict[str, Any],
    ) -> None:
        """Write gathered Float8 buffers back, refreshing ``_transpose_invalid``.

        The base implementation just ``setattr``s the gathered tensors into
        the named fields. For Float8 we additionally need to clear
        ``_transpose_invalid`` when the gathered field is ``_transpose`` —
        otherwise a freshly-gathered transpose buffer is treated as stale
        on first use (see :attr:`_transpose_invalid` semantics in
        ``update_usage`` / ``get_usages``).
        """
        super().fsdp_assign_gathered(gathered, meta)
        if "_transpose" in meta["field_names"]:
            self._transpose_invalid = False
