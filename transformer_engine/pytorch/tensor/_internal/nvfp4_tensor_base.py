# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Mixin class holding data specific for NVFP4Tensor"""

from __future__ import annotations
from typing import Optional, Dict, Any, Tuple
import torch

import transformer_engine_torch as tex

from ..quantized_tensor import QuantizedTensorBase

from ...constants import TE_DType as torch_to_transformer_engine_dtype

from ..quantized_tensor import Quantizer

from ...utils import _empty_tensor


class _FromNVFP4Func(torch.autograd.Function):
    """Cast from NVFP4 to other dtype"""

    @staticmethod
    def forward(
        _ctx: Optional[torch.autograd.function.FunctionCtx],  # unused
        tensor: NVFP4TensorBase,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        # pylint: disable=missing-function-docstring
        dtype = torch_to_transformer_engine_dtype[dtype]

        # Make sure FP8 data is in expected format
        if tensor._rowwise_data is not None:
            return tex.dequantize(tensor, dtype)
        raise NotImplementedError("Casting back from the transpose not implemented yet!")

    @staticmethod
    def backward(
        _ctx: torch.autograd.function.FunctionCtx,  # unused
        grad: torch.Tensor,
    ) -> Tuple[Optional[torch.Tensor], ...]:
        # pylint: disable=missing-function-docstring
        # Assume that we want gradients in full precision
        return grad, None


class NVFP4TensorBase(QuantizedTensorBase):
    """Mixin class that holds data attributes of NVFP4Tensor.

    NVFP4Tensor inherits from the PyTorch tensor class and this mixin
    class. If this class is instantiated directly, it has the same
    data, lower CPU overhead, and less functionality. It should only
    be instantiated directly for performance-critical internal usage.

    """

    _rowwise_data: Optional[torch.Tensor]
    _columnwise_data: Optional[torch.Tensor]
    _quantizer: Optional[Quantizer]
    _rowwise_scale_inv: torch.Tensor
    _columnwise_scale_inv: torch.Tensor
    _per_tensor_rowwise_scale_inv: torch.Tensor

    def __new__(
        cls,
        *args,
        rowwise_data: Optional[torch.Tensor],
        rowwise_scale_inv: torch.Tensor,
        columnwise_data: Optional[torch.Tensor],
        columnwise_scale_inv: torch.Tensor,
        per_tensor_rowwise_scale_inv: torch.Tensor,
        quantizer: Optional[Quantizer] = None,
        **kwargs,
    ):
        instance = super().__new__(cls, *args, **kwargs)
        instance._rowwise_data = rowwise_data
        instance._columnwise_data = columnwise_data
        instance._quantizer = quantizer
        instance._rowwise_scale_inv = rowwise_scale_inv
        instance._columnwise_scale_inv = columnwise_scale_inv
        instance._per_tensor_rowwise_scale_inv = per_tensor_rowwise_scale_inv

        return instance

    def clear(self):
        """Deallocate this tensor's memory. Typically not needed and must be used carefully."""
        for t in (
            self._rowwise_data,
            self._columnwise_data,
            self._rowwise_scale_inv,
            self._columnwise_scale_inv,
            self._per_tensor_rowwise_scale_inv,
        ):
            if t is not None:
                t.data = _empty_tensor()

    def get_metadata(self) -> Dict[str, Any]:
        """Get this tensor's metadata."""
        return {
            "rowwise_data": self._rowwise_data,
            "rowwise_scale_inv": self._rowwise_scale_inv,
            "columnwise_data": self._columnwise_data,
            "columnwise_scale_inv": self._columnwise_scale_inv,
            "per_tensor_rowwise_scale_inv": self._per_tensor_rowwise_scale_inv,
            "quantizer": self._quantizer,
        }

    def prepare_for_saving(self) -> Tuple[list[Optional[torch.Tensor]], NVFP4TensorBase]:
        """Prepare the tensor base for saving for backward"""
        tensors = [
            self._rowwise_data,
            self._columnwise_data,
            self._rowwise_scale_inv,
            self._columnwise_scale_inv,
            self._per_tensor_rowwise_scale_inv,
        ]
        self._rowwise_data = None
        self._columnwise_data = None
        self._rowwise_scale_inv = None
        self._columnwise_scale_inv = None
        self._per_tensor_rowwise_scale_inv = None
        return tensors, self

    def restore_from_saved(
        self, tensors: list[Optional[torch.Tensor]]
    ) -> list[Optional[torch.Tensor]]:
        """Restore the tensor base data from the saved tensors list."""
        self._rowwise_data = tensors[0]
        self._columnwise_data = tensors[1]
        self._rowwise_scale_inv = tensors[2]
        self._columnwise_scale_inv = tensors[3]
        self._per_tensor_rowwise_scale_inv = tensors[4]
        return tensors[5:]

    def get_data_tensors(self):
        """Get this Tensor's data."""
        return self._rowwise_data, self._columnwise_data

    def dequantize(self, *, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """Dequantize to a higher precision."""
        return _FromNVFP4Func.forward(None, self, dtype)

    def size(self, *args, **kwargs):
        # pylint: disable=missing-function-docstring
        if self._rowwise_data is not None:
            return self._rowwise_data.size(*args, **kwargs)
        return self._columnwise_data.size(*args, **kwargs)

    def __repr__(self):
        data_rowwise = self.dequantize()

        return (
            "NVFP4TensorBase("
            f"rowwise_scaled_data={data_rowwise},"
            f"rowwise_scale_inv={self._rowwise_scale_inv},"
            f"per_tensor_rowwise_scale_inv={self._per_tensor_rowwise_scale_inv},"
            ")"
        )

    def update_usage(
        self,
        rowwise_usage: Optional[bool] = None,
        columnwise_usage: Optional[bool] = None,
    ):
        """
        For the NVFP4 format, columnwise scaled output is only produced by x2
        scaling kernels, so this function only disables usages.
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
                    "Requested row-wise usage, but NVFP4Tensor is missing row-scaled NVFP4 data"
                )
            if self._rowwise_scale_inv is None:
                raise RuntimeError(
                    "Requested row-wise usage, but NVFP4Tensor is missing row-scaled scale-inverses"
                )
            if self._per_tensor_rowwise_scale_inv is None:
                raise RuntimeError(
                    "Requested row-wise usage, but NVFP4Tensor is missing per tensor"
                    " row-scaled scale-inverse"
                )
        else:
            self._rowwise_data = None
            self._rowwise_scale_inv = None

        # Update column-scaled data
        if columnwise_usage:
            if self._columnwise_data is None:
                raise RuntimeError(
                    "Requested column-wise usage, but NVFP4Tensor is missing column-scaled FP8 data"
                )
            if self._columnwise_scale_inv is None:
                raise RuntimeError(
                    "Requested column-wise usage, "
                    "but NVFP4Tensor is missing column-scaled scale-inverses"
                )
        else:
            self._columnwise_data = None
            self._columnwise_scale_inv = None
