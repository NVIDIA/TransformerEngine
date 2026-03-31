# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Tensor class with hybrid quantized data (different formats for rowwise vs columnwise)"""

from __future__ import annotations
from typing import Any, Dict, Iterable, Optional, Tuple

import torch

from .storage.hybrid_tensor_storage import HybridQuantizedTensorStorage
from ..quantized_tensor import QuantizedTensor, QuantizedTensorStorage, Quantizer

aten = torch.ops.aten


class HybridQuantizer(Quantizer):
    """Quantizer that composes two existing quantizers for different directions.

    Performs two-pass quantization: the rowwise_quantizer produces rowwise
    quantized data and the columnwise_quantizer produces columnwise quantized
    data. The results are wrapped in a HybridQuantizedTensor.

    Parameters
    ----------
    rowwise_quantizer : Quantizer
        Quantizer for the rowwise direction (e.g. MXFP8Quantizer).
    columnwise_quantizer : Quantizer
        Quantizer for the columnwise direction (e.g. NVFP4Quantizer).

    """

    rowwise_quantizer: Quantizer
    columnwise_quantizer: Quantizer

    def __init__(
        self,
        *,
        rowwise_quantizer: Quantizer,
        columnwise_quantizer: Quantizer,
    ) -> None:
        super().__init__(rowwise=True, columnwise=True)
        self.rowwise_quantizer = rowwise_quantizer
        self.columnwise_quantizer = columnwise_quantizer

        # Pin each sub-quantizer to its designated direction
        self.rowwise_quantizer.set_usage(rowwise=True, columnwise=False)
        self.columnwise_quantizer.set_usage(rowwise=False, columnwise=True)

    def quantize_impl(self, tensor: torch.Tensor) -> QuantizedTensor:
        rowwise_result = self.rowwise_quantizer.quantize(tensor)
        columnwise_result = self.columnwise_quantizer.quantize(tensor)

        if self.internal:
            return HybridQuantizedTensorStorage(
                rowwise_storage=rowwise_result,
                columnwise_storage=columnwise_result,
                rowwise_quantizer=self.rowwise_quantizer,
                columnwise_quantizer=self.columnwise_quantizer,
                quantizer=self,
                fake_dtype=tensor.dtype,
            )

        return HybridQuantizedTensor(
            shape=tensor.shape,
            dtype=tensor.dtype,
            rowwise_storage=rowwise_result,
            columnwise_storage=columnwise_result,
            rowwise_quantizer=self.rowwise_quantizer,
            columnwise_quantizer=self.columnwise_quantizer,
            quantizer=self,
        )

    def make_empty(
        self,
        shape: Iterable[int],
        *,
        dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None,
        requires_grad: bool = False,
        pin_memory: bool = False,
    ) -> HybridQuantizedTensor:
        self.rowwise_quantizer.internal = True
        rowwise_empty = self.rowwise_quantizer.make_empty(
            shape,
            dtype=dtype,
            device=device,
            pin_memory=pin_memory,
        )
        self.rowwise_quantizer.internal = False

        self.columnwise_quantizer.internal = True
        columnwise_empty = self.columnwise_quantizer.make_empty(
            shape,
            dtype=dtype,
            device=device,
            pin_memory=pin_memory,
        )
        self.columnwise_quantizer.internal = False

        return HybridQuantizedTensor(
            shape=shape,
            dtype=dtype,
            requires_grad=requires_grad,
            device=device,
            rowwise_storage=rowwise_empty,
            columnwise_storage=columnwise_empty,
            rowwise_quantizer=self.rowwise_quantizer,
            columnwise_quantizer=self.columnwise_quantizer,
            quantizer=self,
        )

    def set_usage(
        self, *, rowwise: Optional[bool] = None, columnwise: Optional[bool] = None
    ) -> None:
        super().set_usage(rowwise=rowwise, columnwise=columnwise)

    def _get_compatible_recipe(self):
        return None


class HybridQuantizedTensor(HybridQuantizedTensorStorage, QuantizedTensor):
    """Quantized tensor holding data in two different formats per direction.

    The tensor presents as having a standard, higher-precision dtype, but
    internally stores rowwise data in one quantized format and columnwise
    data in another.

    Parameters
    ----------
    shape : iterable of int
        Tensor dimensions.
    dtype : torch.dtype
        Nominal tensor datatype.
    rowwise_storage : QuantizedTensorStorage
        Sub-storage for rowwise quantized data.
    columnwise_storage : QuantizedTensorStorage
        Sub-storage for columnwise quantized data.
    rowwise_quantizer : Quantizer, optional
        Quantizer used for the rowwise sub-storage.
    columnwise_quantizer : Quantizer, optional
        Quantizer used for the columnwise sub-storage.
    quantizer : HybridQuantizer, optional
        Parent hybrid quantizer.
    requires_grad : bool, default = False
        Whether to compute gradients for this tensor.

    """

    def __new__(
        cls,
        *args,
        rowwise_storage: Optional[QuantizedTensorStorage],
        columnwise_storage: Optional[QuantizedTensorStorage],
        rowwise_quantizer: Optional[Quantizer] = None,
        columnwise_quantizer: Optional[Quantizer] = None,
        quantizer: Optional[Quantizer] = None,
        **kwargs,
    ):
        instance = super().__new__(
            cls,
            *args,
            rowwise_storage=rowwise_storage,
            columnwise_storage=columnwise_storage,
            rowwise_quantizer=rowwise_quantizer,
            columnwise_quantizer=columnwise_quantizer,
            quantizer=quantizer,
            **kwargs,
        )
        return instance

    def __repr__(self, *, tensor_contents=None):
        row_type = (
            type(self._rowwise_storage).__name__ if self._rowwise_storage is not None else "None"
        )
        col_type = (
            type(self._columnwise_storage).__name__
            if self._columnwise_storage is not None
            else "None"
        )
        return (
            f"HybridQuantizedTensor(rowwise={row_type}, columnwise={col_type}, dtype={self.dtype})"
        )

    def dequantize(self, *, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        if dtype is None:
            dtype = self.dtype
        return HybridQuantizedTensorStorage.dequantize(self, dtype=dtype)

    def detach(self) -> HybridQuantizedTensor:
        return HybridQuantizedTensor.make_like(self)

    def get_metadata(self) -> Dict[str, Any]:
        return HybridQuantizedTensorStorage.get_metadata(self)

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs=None):
        if func == aten.detach.default:
            return args[0].detach()

        return super().__torch_dispatch__(func, types, args, kwargs)
