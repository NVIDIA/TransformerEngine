# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""TODO: write comments"""

from __future__ import annotations
from typing import Optional, Dict, Any, Tuple, Union
from collections.abc import Iterable
import math
import warnings
import torch

import transformer_engine_torch as tex  # pylint: disable=unused-import
from transformer_engine_torch import DType as TE_DType

from ...quantized_tensor import QuantizedTensorStorage, Quantizer

from ...constants import (
    TE_DType as torch_to_transformer_engine_dtype,
)  # pylint: disable=unused-import

from ...utils import _empty_tensor, canonicalize_shape


class _FromFlexFunc(torch.autograd.Function):
    """Cast from MXFP8 to other dtype"""

    @staticmethod
    def forward(
        _ctx: Optional[torch.autograd.function.FunctionCtx],  # unused
        tensor: FlexTensorStorage,
        dtype: torch.dtype,
        quantizer: Quantizer,
    ) -> torch.Tensor:
        # pylint: disable=missing-function-docstring
        if tensor._rowwise_data is not None and tensor._rowwise_data.numel() == 0:
            return torch.empty(tensor.size(), dtype=dtype, device=tensor.device)
        if tensor._columnwise_data is not None and tensor._columnwise_data.numel() == 0:
            return torch.empty(tensor.size(), dtype=dtype, device=tensor.device)

        if tensor._rowwise_data is not None or tensor._columnwise_data is not None:
            return tex.dequantize_with_quantizer(tensor, dtype, quantizer)
        raise ValueError("Cannot dequantize Flex tensor with no data")

    @staticmethod
    def backward(
        _ctx: torch.autograd.function.FunctionCtx,  # unused
        grad: torch.Tensor,
    ) -> Tuple[Optional[torch.Tensor], ...]:
        # pylint: disable=missing-function-docstring
        # Assume that we want gradients in full precision
        return grad, None, None


class FlexTensorStorage(QuantizedTensorStorage):
    """Mixin class that holds data attributes of FlexTensor.

    FlexTensor inherits from the PyTorch tensor class and this mixin
    class. If this class is instantiated directly, it has the same
    data, lower CPU overhead, and less functionality. It should only
    be instantiated directly for performance-critical internal usage.

    The two directions may carry different quantization formats, tracked
    by ``_dtype_row`` and ``_dtype_column``.

    """

    # Row-scaled quantized data, None if not quantized in this direction
    _rowwise_data: Optional[torch.Tensor]
    # Column-scaled quantized data, None if not quantized in this direction
    _columnwise_data: Optional[torch.Tensor]
    # Block scaling factors for row-scaled data, None if not quantized in this direction
    _rowwise_scale_inv: Optional[torch.Tensor]
    # Block scaling factors for column-scaled data, None if not quantized in this direction
    _columnwise_scale_inv: Optional[torch.Tensor]
    # Input absolute maximum for row-scaled data if quantized in NVFP4 row-wisely
    # None if otherwise
    _amax_rowwise: Optional[torch.Tensor]
    # Input absolute maximum for column-scaled data if quantized in NVFP4 column-wisely
    # Nont if otherwise
    _amax_columnwise: Optional[torch.Tensor]

    # Builder class for casting to the flex format
    _quantizer: Optional[Quantizer]
    # Quantization format of the row-wise direction
    _dtype_row: Optional[TE_DType]
    # Quantization format of the column-wise direction
    _dtype_column: Optional[TE_DType]
    # Whether scaling factors are in the swizzled format expected by GEMM
    _with_gemm_swizzled_scales: bool

    def __new__(
        cls,
        rowwise_data: Optional[torch.Tensor],
        rowwise_scale_inv: Optional[torch.Tensor],
        columnwise_data: Optional[torch.Tensor],
        columnwise_scale_inv: Optional[torch.Tensor],
        amax_rowwise: Optional[torch.Tensor],
        amax_columnwise: Optional[torch.Tensor],
        dtype_row: Optional[TE_DType],
        dtype_column: Optional[TE_DType],
        quantizer: Optional[Quantizer],
        with_gemm_swizzled_scales: bool,
        *args,
        fake_dtype: Optional[torch.dtype] = None,
        **kwargs,
    ):
        if cls is FlexTensorStorage:
            instance = object.__new__(cls)
            instance._dtype = fake_dtype if fake_dtype is not None else torch.float32
        else:
            instance = super().__new__(cls, *args, fake_dtype=fake_dtype, **kwargs)
        instance._rowwise_data = rowwise_data
        instance._columnwise_data = columnwise_data
        instance._rowwise_scale_inv = rowwise_scale_inv
        instance._columnwise_scale_inv = columnwise_scale_inv
        instance._amax_rowwise = amax_rowwise
        instance._amax_columnwise = amax_columnwise
        instance._quantizer = quantizer.copy() if quantizer is not None else None
        instance._dtype_row = dtype_row
        instance._dtype_column = dtype_column
        instance._with_gemm_swizzled_scales = with_gemm_swizzled_scales

        return instance

    def clear(self):
        """Deallocate this tensor's memory. Typically not needed and must be used carefully."""
        for t in (
            self._rowwise_data,
            self._columnwise_data,
            self._rowwise_scale_inv,
            self._columnwise_scale_inv,
            self._amax_rowwise,
            self._amax_columnwise,
        ):
            if t is not None:
                t.data = _empty_tensor()

    def copy_from_storage(self, src: QuantizedTensorStorage) -> None:
        """Copy data buffers from another FlexTensorStorage."""
        if not isinstance(src, FlexTensorStorage):
            raise TypeError("copy_from_storage expects FlexTensorStorage")
        if self._dtype_row != src._dtype_row or self._dtype_column != src._dtype_column:
            raise RuntimeError("Flex dtype mismatch in copy_from_storage")
        if self._with_gemm_swizzled_scales != src._with_gemm_swizzled_scales:
            raise RuntimeError("Scale layout mismatch in copy_from_storage")

        def _copy_optional(dst: Optional[torch.Tensor], src_tensor: Optional[torch.Tensor]):
            if dst is not None and src_tensor is not None:
                dst.copy_(src_tensor)

        _copy_optional(self._rowwise_data, src._rowwise_data)
        _copy_optional(self._columnwise_data, src._columnwise_data)
        _copy_optional(self._rowwise_scale_inv, src._rowwise_scale_inv)
        _copy_optional(self._columnwise_scale_inv, src._columnwise_scale_inv)
        _copy_optional(self._amax_rowwise, src._amax_rowwise)
        _copy_optional(self._amax_columnwise, src._amax_columnwise)

    def get_metadata(self) -> Dict[str, Any]:
        """Get this tensor's metadata."""
        return {
            "rowwise_data": self._rowwise_data,
            "rowwise_scale_inv": self._rowwise_scale_inv,
            "columnwise_data": self._columnwise_data,
            "columnwise_scale_inv": self._columnwise_scale_inv,
            "amax_rowwise": self._amax_rowwise,
            "amax_columnwise": self._amax_columnwise,
            "dtype_row": self._dtype_row,
            "dtype_column": self._dtype_column,
            "quantizer": self._quantizer,
            "with_gemm_swizzled_scales": self._with_gemm_swizzled_scales,
            "fake_dtype": self._dtype,
        }

    def prepare_for_saving(self) -> Tuple[list[Optional[torch.Tensor]], FlexTensorStorage]:
        """Prepare the tensor base for saving for backward"""
        tensors = [
            self._rowwise_data,
            self._columnwise_data,
            self._rowwise_scale_inv,
            self._columnwise_scale_inv,
            self._amax_rowwise,
            self._amax_columnwise,
        ]
        self._rowwise_data = None
        self._columnwise_data = None
        self._rowwise_scale_inv = None
        self._columnwise_scale_inv = None
        self._amax_rowwise = None
        self._amax_columnwise = None
        return tensors, self

    def restore_from_saved(
        self, tensors: list[Optional[torch.Tensor]]
    ) -> list[Optional[torch.Tensor]]:
        """Restore the tensor base data from the saved tensors list."""
        self._rowwise_data = tensors[0]
        self._columnwise_data = tensors[1]
        self._rowwise_scale_inv = tensors[2]
        self._columnwise_scale_inv = tensors[3]
        self._amax_rowwise = tensors[4]
        self._amax_columnwise = tensors[5]
        return tensors[6:]

    def get_data_tensors(self, rowwise_data: bool = True, columnwise_data: bool = True):
        """Get this Tensor's data."""
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
        if self._rowwise_data is not None and self._rowwise_data.numel() == 0:
            return torch.empty(self.size(), dtype=dtype, device=self.device)
        return _FromFlexFunc.forward(None, self, dtype, self)

    def size(self, dim: Optional[int] = None) -> Union[torch.Size, int]:
        # pylint: disable=missing-function-docstring
        shape = None
        if self._rowwise_data is not None:
            if self.is_mxfp8_dtype(self._dtype_row):
                shape = self._rowwise_data.shape
            elif self.is_nvfp4_dtype(self._dtype_row):
                byte_shape = list(self._rowwise_data.size())
                shape = byte_shape[:-1] + [byte_shape[-1] * 2]
        elif self._columnwise_data is not None:
            if self.is_mxfp8_dtype(self._dtype_column):
                shape = self._columnwise_data
            elif self.is_nvfp4_dtype(self._dtype_column):
                warnings.warn("Attempting to get shape of NVFP4 tensor with only column-wise data.")
                byte_shape = list(self._columnwise_data.size())
                shape = byte_shape[1:-1] + [byte_shape[-1] * 2, byte_shape[0]]

        if shape is None:
            raise RuntimeError("Attempted to get shape of Flex tensor with no data")
        if dim is None:
            return torch.Size(shape)
        return shape[dim]

    @property
    def device(self):
        """Return the device of the tensor. Define this to avoid expensive PyObject lookups."""
        if self._rowwise_data is not None:
            return self._rowwise_data.device
        if self._columnwise_data is not None:
            return self._columnwise_data.device
        raise RuntimeError("FlexTensorStorage has no data!")

    def view(self, shape: torch.Size):
        # pylint: disable=missing-function-docstring

        # Return input tensor if view not needed
        cur_shape = self.size()
        if shape is None or shape == cur_shape:
            return self

        shape = canonicalize_shape(shape, cur_shape)
        if shape[-1] != cur_shape[-1]:
            raise RuntimeError(
                "FlexTensor does not support reshaping inner dimension "
                f"(attempted to reshape dims={tuple(cur_shape)} to {tuple(shape)})"
            )

        cur_rowwise_data = self._rowwise_data
        cur_columnwise_data = self._columnwise_data
        new_rowwise_data = None
        new_columnwise_data = None
        if self.is_mxfp8_dtype(self._dtype_row):
            new_rowwise_data = cur_rowwise_data.view(*shape)
        elif self.is_nvfp4_dtype(self._dtype_row):
            if shape[-1] % 2 != 0:
                raise ValueError(
                    "Cannot represent row-wise NVFP4 quantized data for Flex tensor "
                    f"with shape={shape} as byte array."
                )
            byte_shape = list(shape[:-1]) + [shape[-1] // 2]
            new_rowwise_data = self._rowwise_data.view(byte_shape)
        if self.is_mxfp8_dtype(self._dtype_column):
            new_columnwise_data = cur_columnwise_data.view(*shape)
        elif self.is_nvfp4_dtype(self._dtype_column):
            columnwise_shape = (shape[-1], math.prod(shape[:-1]))
            if columnwise_shape[-1] % 2 != 0:
                raise ValueError(
                    "Cannot represent column-wise NVFP4 quantized data for Flex tensor "
                    f"with shape={shape} as byte array."
                )
            byte_shape = (columnwise_shape[0], columnwise_shape[1] // 2)
            new_columnwise_data = self._columnwise_data.view(byte_shape)

        return FlexTensorStorage(
            rowwise_data=new_rowwise_data,
            rowwise_scale_inv=self._rowwise_scale_inv,
            columnwise_data=new_columnwise_data,
            columnwise_scale_inv=self._columnwise_scale_inv,
            amax_rowwise=self._amax_rowwise,
            amax_columnwise=self._amax_columnwise,
            dtype_row=self._dtype_row,
            dtype_column=self._dtype_column,
            quantizer=self._quantizer,
            with_gemm_swizzled_scales=self._with_gemm_swizzled_scales,
            fake_dtype=self._dtype,
        )

    def __repr__(self):
        return (
            "FlexTensorStorage("
            f"dtype_row={self._dtype_row}, "
            f"dtype_column={self._dtype_column}, "
            f"rowwise_scale_inv={self._rowwise_scale_inv}, "
            f"columnwise_scale_inv={self._columnwise_scale_inv}, "
            f"amax_rowwise={self._amax_rowwise},"
            f"amax_columnwise={self._amax_columnwise},"
            ")"
        )

    def update_usage(
        self,
        rowwise_usage: Optional[bool] = None,
        columnwise_usage: Optional[bool] = None,
    ):
        """
        TODO: figure out what to say here
        """

        # Default usage is based on available data
        if rowwise_usage is None:
            rowwise_usage = self._rowwise_data is not None
        if columnwise_usage is None:
            columnwise_usage = self._columnwise_data is not None

        # Update row-scaled data
        if rowwise_usage:
            if self._rowwise_data is None:
                raise RuntimeError(
                    "Requested row-wise usage, but FlexTensor is missing row-scaled data"
                )
            if self._rowwise_scale_inv is None:
                raise RuntimeError(
                    "Requested row-wise usage, but FlexTensor is missing row-scaled scale-inverses"
                )
            if self._amax_rowwise is None and self.is_nvfp4_dtype(self._dtype_row):
                raise RuntimeError(
                    "Requested row-wise NVFP4 usage, but FlexTensor is missing per tensor"
                    " row-scaled scale-inverse"
                )
        else:
            self._rowwise_data = None
            self._rowwise_scale_inv = None
            self._amax_rowwise = None

        # Update column-scaled data
        if columnwise_usage:
            if self._columnwise_data is None:
                raise RuntimeError(
                    "Requested column-wise usage, but FlexTensor is missing column-scaled data"
                )
            if self._columnwise_scale_inv is None:
                raise RuntimeError(
                    "Requested column-wise usage, "
                    "but FlexTensor is missing column-scaled scale-inverses"
                )
            if self._amax_columnwise is None and self.is_nvfp4_dtype(self._dtype_column):
                raise RuntimeError(
                    "Requested column-wise NVFP4 usage, "
                    "but FlexTensor is missing per tensor column-scaled scale-inverse"
                )
        else:
            self._columnwise_data = None
            self._columnwise_scale_inv = None
            self._amax_columnwise = None

    def get_usages(self) -> Dict[str, bool]:
        """Get the usage of the tensor"""
        return {
            "rowwise": self._rowwise_data is not None,
            "columnwise": self._columnwise_data is not None,
        }

    @staticmethod
    def is_mxfp8_dtype(dtype: TE_DType) -> bool:
        return dtype == tex.DType.kFloat8E4M3 or dtype == tex.DType.kFloat8E5M2

    @staticmethod
    def is_nvfp4_dtype(dtype: TE_DType) -> bool:
        return dtype == tex.DType.kFloat4E2M1
