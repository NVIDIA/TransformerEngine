# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Mixin class holding data specific for NVFP4Tensor"""

from __future__ import annotations
from collections.abc import Iterable
import functools
import math
from typing import Any, Dict, Optional, Tuple, Union
import warnings

import torch

import transformer_engine_torch as tex
from transformer_engine_torch import DType as TE_DType

from ...quantized_tensor import QuantizedTensorStorage, Quantizer

from ...constants import TE_DType as torch_to_transformer_engine_dtype
from ...utils import _empty_tensor


@functools.lru_cache(maxsize=None)
def _fp4_e2m1_vals(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Values representable in FP4 E2M1 format"""
    return torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
        device=device,
        dtype=dtype,
    )


class _FromNVFP4Func(torch.autograd.Function):
    """Cast from NVFP4 to other dtype"""

    @staticmethod
    def forward(
        _ctx: Optional[torch.autograd.function.FunctionCtx],  # unused
        tensor: NVFP4TensorStorage,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        # pylint: disable=missing-function-docstring

        # Dequantize row-wise data
        if tensor._rowwise_data is not None:
            return tex.dequantize(tensor, torch_to_transformer_engine_dtype[dtype])

        if tensor._columnwise_data is not None:
            raise NotImplementedError("Dequantizing column-wise NVFP4 data is not implemented yet!")
        raise ValueError("Attempted to dequantize NVFP4 tensor with no data")

    @staticmethod
    def backward(
        _ctx: torch.autograd.function.FunctionCtx,  # unused
        grad: torch.Tensor,
    ) -> Tuple[Optional[torch.Tensor], ...]:
        # pylint: disable=missing-function-docstring
        # Assume that we want gradients in full precision
        return grad, None


class NVFP4TensorStorage(QuantizedTensorStorage):
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
    _fp4_dtype: TE_DType
    _amax_rowwise: torch.Tensor
    _amax_columnwise: torch.Tensor

    def __new__(
        cls,
        rowwise_data: Optional[torch.Tensor],
        rowwise_scale_inv: torch.Tensor,
        columnwise_data: Optional[torch.Tensor],
        columnwise_scale_inv: torch.Tensor,
        amax_rowwise: torch.Tensor,
        amax_columnwise: torch.Tensor,
        fp4_dtype: TE_DType,
        quantizer: Optional[Quantizer],
        *args,
        **kwargs,
    ):

        instance = super().__new__(cls, *args, **kwargs)

        instance._rowwise_data = rowwise_data
        instance._columnwise_data = columnwise_data
        instance._fp4_dtype = fp4_dtype
        instance._quantizer = quantizer.copy() if quantizer is not None else None
        instance._rowwise_scale_inv = rowwise_scale_inv
        instance._columnwise_scale_inv = columnwise_scale_inv
        instance._amax_rowwise = amax_rowwise
        instance._amax_columnwise = amax_columnwise

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

    def get_metadata(self) -> Dict[str, Any]:
        """Get this tensor's metadata."""
        return {
            "rowwise_data": self._rowwise_data,
            "rowwise_scale_inv": self._rowwise_scale_inv,
            "columnwise_data": self._columnwise_data,
            "columnwise_scale_inv": self._columnwise_scale_inv,
            "amax_rowwise": self._amax_rowwise,
            "amax_columnwise": self._amax_columnwise,
            "fp4_dtype": self._fp4_dtype,
            "quantizer": self._quantizer,
        }

    def prepare_for_saving(self) -> Tuple[list[Optional[torch.Tensor]], NVFP4TensorStorage]:
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

    def get_data_tensors(self):
        """Get this Tensor's data."""
        return self._rowwise_data, self._columnwise_data

    def dequantize(self, *, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """Dequantize to a higher precision."""
        return _FromNVFP4Func.forward(None, self, dtype)

    def size(self, dim: Optional[int] = None) -> Union[torch.Size, int]:
        # pylint: disable=missing-function-docstring

        # Infer tensor shape
        shape = None
        if self._rowwise_data is not None:
            byte_shape = list(self._rowwise_data.size())
            shape = byte_shape[:-1] + [byte_shape[-1] * 2]
        elif self._columnwise_data is not None:
            warnings.warn("Attempting to get shape of NVFP4 tensor with only column-wise data.")
            byte_shape = list(self._columnwise_data.size())
            shape = byte_shape[1:-1] + [byte_shape[-1] * 2, byte_shape[0]]
        if shape is None:
            raise RuntimeError("Attempted to get shape of NVFP4 tensor with no data")

        # Return shape or dim
        if dim is None:
            return torch.Size(shape)
        return shape[dim]

    def view(self, shape: torch.Size):
        # pylint: disable=missing-function-docstring

        # Return input tensor if view not needed
        cur_shape = self.size()
        if shape is None or shape == cur_shape:
            return self

        # Canonicalize shape
        if not isinstance(shape, Iterable):
            shape = [shape]
        elif len(shape) == 1 and isinstance(shape[0], Iterable):
            shape = shape[0]
        if -1 in shape:
            shape = list(shape)
            d_inferred = -math.prod(cur_shape) // math.prod(shape)
            for i, d in enumerate(shape):
                if d == -1:
                    shape[i] = d_inferred
                    break
        if shape[-1] != cur_shape[-1]:
            raise RuntimeError(
                "NVFP4Tensor does not support reshaping inner dimension "
                f"(attempted to reshape dims={tuple(cur_shape)} to {tuple(shape)})"
            )

        # Reshape data
        new_rowwise_data = None
        new_columnwise_data = None
        if self._rowwise_data is not None:
            if shape[-1] % 2 != 0:
                raise ValueError(
                    "Cannot represent row-wise data for NVFP4 tensor "
                    f"with shape={shape} as byte array."
                )
            byte_shape = list(shape[:-1]) + [shape[-1] // 2]
            new_rowwise_data = self._rowwise_data.view(byte_shape)
        if self._columnwise_data is not None:
            columnwise_shape = (shape[-1], math.prod(shape[:-1]))
            if columnwise_shape[-1] % 2 != 0:
                raise ValueError(
                    "Cannot represent column-wise data for NVFP4 tensor "
                    f"with shape={shape} as byte array."
                )
            byte_shape = (columnwise_shape[0], columnwise_shape[1] // 2)
            new_columnwise_data = self._columnwise_data.view(byte_shape)

        # Construct tensor
        return NVFP4TensorStorage(
            rowwise_data=new_rowwise_data,
            rowwise_scale_inv=self._rowwise_scale_inv,
            columnwise_data=new_columnwise_data,
            columnwise_scale_inv=self._columnwise_scale_inv,
            amax_rowwise=self._amax_rowwise,
            amax_columnwise=self._amax_columnwise,
            quantizer=self._quantizer,
            fp4_dtype=self._fp4_dtype,
        )

    def __repr__(self):
        data_rowwise = self.dequantize()

        return (
            "NVFP4TensorStorage("
            f"rowwise_scaled_data={data_rowwise},"
            f"rowwise_scale_inv={self._rowwise_scale_inv},"
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
            if self._amax_rowwise is None:
                raise RuntimeError(
                    "Requested row-wise usage, but NVFP4Tensor is missing per tensor"
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
                    "Requested column-wise usage, but NVFP4Tensor is missing column-scaled FP8 data"
                )
            if self._columnwise_scale_inv is None:
                raise RuntimeError(
                    "Requested column-wise usage, "
                    "but NVFP4Tensor is missing column-scaled scale-inverses"
                )
            if self._amax_columnwise is None:
                raise RuntimeError(
                    "Requested column-wise usage, "
                    "but NVFP4Tensor is missing per tensor column-scaled scale-inverse"
                )
        else:
            self._columnwise_data = None
            self._columnwise_scale_inv = None
            self._amax_columnwise = None
