# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Mixin class holding data specific for Float8Tensor"""

from __future__ import annotations
import math
from typing import Any, Dict, Optional, Tuple
import torch

import transformer_engine_torch as tex
from transformer_engine_torch import DType as TE_DType

from ..quantized_tensor import QuantizedTensorBase

from ...constants import TE_DType as torch_to_transformer_engine_dtype

from ..quantized_tensor import Quantizer

from ...utils import is_non_tn_fp8_gemm_supported, _empty_tensor


class _FromFloat8Func(torch.autograd.Function):
    """Cast from FP8 to other dtype"""

    @staticmethod
    def forward(
        _ctx: Optional[torch.autograd.function.FunctionCtx],  # unused
        tensor: Float8TensorBase,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        # pylint: disable=missing-function-docstring
        te_dtype = torch_to_transformer_engine_dtype[dtype]

        # Make sure FP8 data is in expected format
        if tensor._data is not None:
            if tensor._data.numel() == 0:
                return torch.empty_like(tensor._data, dtype=dtype)
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


class Float8TensorBase(QuantizedTensorBase):
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

    def __new__(
        cls,
        *args,
        data: Optional[torch.Tensor],
        fp8_scale_inv: torch.Tensor,
        fp8_dtype: TE_DType,
        data_transpose: Optional[torch.Tensor] = None,
        quantizer: Optional[Quantizer] = None,
        **kwargs,
    ):
        if cls is Float8TensorBase:
            instance = object.__new__(cls)
        else:
            instance = super().__new__(cls, *args, **kwargs)
        instance._data = data
        instance._quantizer = quantizer
        instance._fp8_dtype = fp8_dtype
        instance._scale_inv = fp8_scale_inv
        instance._transpose = data_transpose
        instance._transpose_invalid = instance._transpose is None

        return instance

    def clear(self):
        """Deallocate this tensor's memory. Typically not needed and must be used carefully."""
        for t in (self._data, self._transpose, self._scale_inv):
            if t is not None:
                t.data = _empty_tensor()
        self._transpose_invalid = True

    def get_metadata(self) -> Dict[str, Any]:
        """Get this tensor's metadata."""
        return {
            "data": self._data,
            "fp8_scale_inv": self._scale_inv,
            "fp8_dtype": self._fp8_dtype,
            "data_transpose": self._transpose,
            "quantizer": self._quantizer,
        }

    def prepare_for_saving(self) -> Tuple[list[Optional[torch.Tensor]], QuantizedTensorBase]:
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

    def get_data_tensors(self):
        """Get this Tensor's data."""
        return self._data, self._transpose

    def dequantize(self, *, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """Dequantize to a higher precision."""
        return _FromFloat8Func.forward(None, self, dtype)

    def size(self, *args, **kwargs):
        # pylint: disable=missing-function-docstring
        if self._data is not None:
            return self._data.size(*args, **kwargs)
        size = self._transpose.size(*args, **kwargs)
        return torch.Size([size[-1], math.prod(size[:-1])])

    def __repr__(self):
        return (
            "Float8TensorBase("
            f"fp8_dtype={self._fp8_dtype}, "
            f"scale_inv={self._scale_inv.item()}, "
            f"data={self.dequantize()}"
            ")"
        )

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
