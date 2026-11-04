# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Mixin class holding data specific for MXFP4Tensor"""

from __future__ import annotations
from typing import Annotated, Any, Dict, Optional, Tuple

import torch

import transformer_engine_torch as tex

from transformer_engine.pytorch.device import te_device_type
from ...quantized_tensor import InnerTensor, QuantizedTensorStorage, Quantizer
from .._quantization_helpers import safe_quantized_repr

from ...constants import TE_DType as torch_to_transformer_engine_dtype

from ...utils import _empty_tensor


class _FromMXFP4Func(torch.autograd.Function):
    """Cast from MXFP4 to other dtype"""

    @staticmethod
    def forward(
        _ctx: Optional[torch.autograd.function.FunctionCtx],  # unused
        tensor: MXFP4TensorStorage,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        # pylint: disable=missing-function-docstring
        if tensor._rowwise_data is None and tensor._columnwise_data is None:
            raise ValueError("Cannot dequantize MXFP4 tensor with no data")
        if tensor.numel() == 0:
            return torch.empty(tensor.size(), dtype=dtype, device=tensor.device)
        te_dtype = torch_to_transformer_engine_dtype[dtype]
        src_device = tensor.device
        if src_device.type != te_device_type():
            raise NotImplementedError("MXFP4 dequantization requires device-resident data")
        return tex.dequantize(tensor, te_dtype)

    @staticmethod
    def backward(
        _ctx: torch.autograd.function.FunctionCtx,  # unused
        grad: torch.Tensor,
    ) -> Tuple[Optional[torch.Tensor], ...]:
        # pylint: disable=missing-function-docstring
        return grad, None


class MXFP4TensorStorage(QuantizedTensorStorage):
    """Mixin class that holds data attributes of MXFP4Tensor.

    Data is FP4 E2M1 with an E8M0 scale per 32 consecutive elements,
    using the MXFP8 scaling mode and scale layouts. Both directions
    keep the logical (untransposed) orientation, with two adjacent
    elements of the last dimension packed per byte.

    """

    _rowwise_data: Annotated[Optional[torch.Tensor], InnerTensor("rowwise_data")]
    _rowwise_scale_inv: Annotated[Optional[torch.Tensor], InnerTensor("rowwise_scale_inv")]
    _columnwise_data: Annotated[Optional[torch.Tensor], InnerTensor("columnwise_data")]
    _columnwise_scale_inv: Annotated[Optional[torch.Tensor], InnerTensor("columnwise_scale_inv")]

    _quantizer: Optional[Quantizer]
    _with_gemm_swizzled_scales: bool

    def __new__(
        cls,
        rowwise_data: Optional[torch.Tensor],
        rowwise_scale_inv: Optional[torch.Tensor],
        columnwise_data: Optional[torch.Tensor],
        columnwise_scale_inv: Optional[torch.Tensor],
        quantizer: Optional[Quantizer],
        with_gemm_swizzled_scales: bool,
        *args,
        fake_dtype: Optional[torch.dtype] = None,
        **kwargs,
    ):
        if cls is MXFP4TensorStorage:
            instance = object.__new__(cls)
            instance._dtype = fake_dtype if fake_dtype is not None else torch.float32
        else:
            instance = super().__new__(cls, *args, fake_dtype=fake_dtype, **kwargs)
        instance._rowwise_data = rowwise_data
        instance._columnwise_data = columnwise_data
        instance._rowwise_scale_inv = rowwise_scale_inv
        instance._columnwise_scale_inv = columnwise_scale_inv
        instance._quantizer = quantizer.copy() if quantizer is not None else None
        instance._with_gemm_swizzled_scales = with_gemm_swizzled_scales
        return instance

    def clear(self):
        """Deallocate this tensor's memory. Typically not needed and must be used carefully."""
        for t in (
            self._rowwise_data,
            self._columnwise_data,
            self._rowwise_scale_inv,
            self._columnwise_scale_inv,
        ):
            if t is not None:
                t.data = _empty_tensor()

    def copy_from_storage(self, src: QuantizedTensorStorage) -> None:
        """Copy data buffers from another MXFP4TensorStorage."""
        if not isinstance(src, MXFP4TensorStorage):
            raise TypeError("copy_from_storage expects MXFP4TensorStorage")
        if self._with_gemm_swizzled_scales != src._with_gemm_swizzled_scales:
            raise RuntimeError("Scale layout mismatch in copy_from_storage")
        for dst_t, src_t in (
            (self._rowwise_data, src._rowwise_data),
            (self._columnwise_data, src._columnwise_data),
            (self._rowwise_scale_inv, src._rowwise_scale_inv),
            (self._columnwise_scale_inv, src._columnwise_scale_inv),
        ):
            if dst_t is not None and src_t is not None:
                dst_t.copy_(src_t)

    def get_metadata(self) -> Dict[str, Any]:
        """Get this tensor's metadata."""
        return {
            "rowwise_data": self._rowwise_data,
            "rowwise_scale_inv": self._rowwise_scale_inv,
            "columnwise_data": self._columnwise_data,
            "columnwise_scale_inv": self._columnwise_scale_inv,
            "quantizer": self._quantizer,
            "with_gemm_swizzled_scales": self._with_gemm_swizzled_scales,
            "fake_dtype": self._dtype,
        }

    def prepare_for_saving(self) -> Tuple[list[Optional[torch.Tensor]], MXFP4TensorStorage]:
        """Prepare the tensor base for saving for backward"""
        tensors = [
            self._rowwise_data,
            self._columnwise_data,
            self._rowwise_scale_inv,
            self._columnwise_scale_inv,
        ]
        self._rowwise_data = None
        self._columnwise_data = None
        self._rowwise_scale_inv = None
        self._columnwise_scale_inv = None
        return tensors, self

    def restore_from_saved(
        self, tensors: list[Optional[torch.Tensor]]
    ) -> list[Optional[torch.Tensor]]:
        """Restore the tensor base data from the saved tensors list."""
        self._rowwise_data = tensors[0]
        self._columnwise_data = tensors[1]
        self._rowwise_scale_inv = tensors[2]
        self._columnwise_scale_inv = tensors[3]
        return tensors[4:]

    def get_data_tensors(self, rowwise_data: bool = True, columnwise_data: bool = True):
        """Get this Tensor's packed data."""
        if rowwise_data and columnwise_data:
            return self._rowwise_data, self._columnwise_data
        if rowwise_data:
            return self._rowwise_data
        if columnwise_data:
            return self._columnwise_data
        raise ValueError("No data to get, both rowwise_data and columnwise_data are False")

    def dequantize(self, *, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        """Dequantize to a higher precision."""
        if dtype is None:
            dtype = self._dtype
        return _FromMXFP4Func.forward(None, self, dtype)

    def size(self, *args, **kwargs):
        # pylint: disable=missing-function-docstring
        data = self._rowwise_data if self._rowwise_data is not None else self._columnwise_data
        shape = torch.Size((*data.shape[:-1], 2 * data.shape[-1]))
        if args or kwargs:
            dim = args[0] if args else kwargs["dim"]
            return shape[dim]
        return shape

    def numel(self) -> int:
        # pylint: disable=missing-function-docstring
        return self.size().numel()

    @property
    def device(self):
        """Return the device of the tensor. Define this to avoid expensive PyObject lookups."""
        if self._rowwise_data is not None:
            return self._rowwise_data.device
        if self._columnwise_data is not None:
            return self._columnwise_data.device
        raise RuntimeError("MXFP4TensorStorage has no data!")

    def view(self, shape: torch.Size):
        # pylint: disable=missing-function-docstring
        cur_shape = self.size()
        if shape is None or tuple(shape) == tuple(cur_shape):
            return self
        shape = list(shape)
        if -1 in shape:
            known = 1
            for d in shape:
                if d != -1:
                    known *= d
            shape[shape.index(-1)] = cur_shape.numel() // known
        if shape[-1] != cur_shape[-1]:
            raise RuntimeError(
                "MXFP4Tensor does not support reshaping inner dimension "
                f"(attempted to reshape dims={tuple(cur_shape)} to {tuple(shape)})"
            )
        data_shape = (*shape[:-1], shape[-1] // 2)
        return MXFP4TensorStorage(
            rowwise_data=(
                self._rowwise_data.view(data_shape) if self._rowwise_data is not None else None
            ),
            rowwise_scale_inv=self._rowwise_scale_inv,
            columnwise_data=(
                self._columnwise_data.view(data_shape)
                if self._columnwise_data is not None
                else None
            ),
            columnwise_scale_inv=self._columnwise_scale_inv,
            quantizer=self._quantizer,
            with_gemm_swizzled_scales=self._with_gemm_swizzled_scales,
            fake_dtype=self._dtype,
        )

    def __repr__(self):
        try:
            return f"MXFP4TensorStorage(data={self.dequantize()})"
        except Exception as exc:  # pylint: disable=broad-except
            return safe_quantized_repr(self, "MXFP4TensorStorage", error=exc)

    def update_usage(
        self,
        rowwise_usage: Optional[bool] = None,
        columnwise_usage: Optional[bool] = None,
    ):
        """Drop unused representations.

        Each direction is quantized independently from the
        high-precision source, so missing usages cannot be generated.

        """
        if rowwise_usage is None:
            rowwise_usage = self._rowwise_data is not None
        if columnwise_usage is None:
            columnwise_usage = self._columnwise_data is not None

        if rowwise_usage:
            if self._rowwise_data is None or self._rowwise_scale_inv is None:
                raise RuntimeError(
                    "Requested row-wise usage, but MXFP4Tensor is missing row-wise data"
                )
        else:
            self._rowwise_data = None
            self._rowwise_scale_inv = None

        if columnwise_usage:
            if self._columnwise_data is None or self._columnwise_scale_inv is None:
                raise RuntimeError(
                    "Requested column-wise usage, but MXFP4Tensor is missing column-wise data"
                )
        else:
            self._columnwise_data = None
            self._columnwise_scale_inv = None

    def get_usages(self) -> Dict[str, bool]:
        """Get the usage of the tensor"""
        return {
            "rowwise": self._rowwise_data is not None,
            "columnwise": self._columnwise_data is not None,
        }

    def fsdp_buffer_fields(self) -> Tuple[str, ...]:
        """FSDP2 is not supported for MXFP4."""
        raise NotImplementedError("FSDP2 is not supported for MXFP4 tensors")
