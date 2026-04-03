# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Mixin class holding data specific for HybridQuantizedTensor"""

from __future__ import annotations
from typing import Any, Dict, Optional, Tuple

import torch

from ...quantized_tensor import QuantizedTensorStorage, Quantizer


class HybridQuantizedTensorStorage(QuantizedTensorStorage):
    """Storage that composes two QuantizedTensorStorage instances.

    One sub-storage provides rowwise quantized data and the other provides
    columnwise quantized data. This enables mixed-precision quantization
    where, for example, rowwise data is FP8 and columnwise data is FP4.

    """

    _rowwise_storage: Optional[QuantizedTensorStorage]
    _columnwise_storage: Optional[QuantizedTensorStorage]
    _rowwise_quantizer: Optional[Quantizer]
    _columnwise_quantizer: Optional[Quantizer]
    _quantizer: Optional[Quantizer]

    def __new__(
        cls,
        *args,
        rowwise_storage: Optional[QuantizedTensorStorage],
        columnwise_storage: Optional[QuantizedTensorStorage],
        rowwise_quantizer: Optional[Quantizer] = None,
        columnwise_quantizer: Optional[Quantizer] = None,
        quantizer: Optional[Quantizer] = None,
        fake_dtype: Optional[torch.dtype] = None,
        **kwargs,
    ):
        if cls is HybridQuantizedTensorStorage:
            instance = object.__new__(cls)
            instance._dtype = fake_dtype if fake_dtype is not None else torch.float32
        else:
            instance = super().__new__(cls, *args, fake_dtype=fake_dtype, **kwargs)

        instance._rowwise_storage = rowwise_storage
        instance._columnwise_storage = columnwise_storage
        instance._rowwise_quantizer = rowwise_quantizer
        instance._columnwise_quantizer = columnwise_quantizer
        instance._quantizer = quantizer
        return instance

    @property
    def rowwise_sub_storage(self) -> Optional[QuantizedTensorStorage]:
        """The sub-storage providing rowwise quantized data."""
        return self._rowwise_storage

    @property
    def columnwise_sub_storage(self) -> Optional[QuantizedTensorStorage]:
        """The sub-storage providing columnwise quantized data."""
        return self._columnwise_storage

    def update_usage(
        self,
        rowwise_usage: Optional[bool] = None,
        columnwise_usage: Optional[bool] = None,
    ):
        if rowwise_usage is not None and not rowwise_usage:
            self._rowwise_storage = None
        if columnwise_usage is not None and not columnwise_usage:
            self._columnwise_storage = None

    def get_usages(self) -> Dict[str, bool]:
        return {
            "rowwise": self._rowwise_storage is not None,
            "columnwise": self._columnwise_storage is not None,
        }

    def prepare_for_saving(
        self,
    ) -> Tuple[list[Optional[torch.Tensor]], HybridQuantizedTensorStorage]:
        tensors = []
        if self._rowwise_storage is not None:
            row_tensors, _ = self._rowwise_storage.prepare_for_saving()
            tensors.extend(row_tensors)
        if self._columnwise_storage is not None:
            col_tensors, _ = self._columnwise_storage.prepare_for_saving()
            tensors.extend(col_tensors)
        return tensors, self

    def restore_from_saved(
        self, tensors: list[Optional[torch.Tensor]]
    ) -> list[Optional[torch.Tensor]]:
        if self._rowwise_storage is not None:
            tensors = self._rowwise_storage.restore_from_saved(tensors)
        if self._columnwise_storage is not None:
            tensors = self._columnwise_storage.restore_from_saved(tensors)
        return tensors

    def dequantize(self, *, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        if dtype is None:
            dtype = self._dtype
        if self._rowwise_storage is not None:
            return self._rowwise_storage.dequantize(dtype=dtype)
        if self._columnwise_storage is not None:
            return self._columnwise_storage.dequantize(dtype=dtype)
        raise RuntimeError("HybridQuantizedTensorStorage has no data to dequantize")

    def get_data_tensors(self):
        row_tensors = ()
        col_tensors = ()
        if self._rowwise_storage is not None:
            result = self._rowwise_storage.get_data_tensors()
            row_tensors = result if isinstance(result, tuple) else (result,)
        if self._columnwise_storage is not None:
            result = self._columnwise_storage.get_data_tensors()
            col_tensors = result if isinstance(result, tuple) else (result,)
        return row_tensors + col_tensors

    def size(self, *args, **kwargs):
        if self._rowwise_storage is not None:
            return self._rowwise_storage.size(*args, **kwargs)
        if self._columnwise_storage is not None:
            return self._columnwise_storage.size(*args, **kwargs)
        raise RuntimeError("HybridQuantizedTensorStorage has no data")

    @property
    def device(self):
        if self._rowwise_storage is not None:
            return self._rowwise_storage.device
        if self._columnwise_storage is not None:
            return self._columnwise_storage.device
        raise RuntimeError("HybridQuantizedTensorStorage has no data")

    def view(self, shape: torch.Size):
        raise NotImplementedError("HybridQuantizedTensorStorage does not support view operations")

    def get_metadata(self) -> Dict[str, Any]:
        return {
            "rowwise_storage": self._rowwise_storage,
            "columnwise_storage": self._columnwise_storage,
            "rowwise_quantizer": self._rowwise_quantizer,
            "columnwise_quantizer": self._columnwise_quantizer,
            "quantizer": self._quantizer,
            "fake_dtype": self._dtype,
        }

    def __repr__(self):
        return (
            "HybridQuantizedTensorStorage("
            f"rowwise={type(self._rowwise_storage).__name__}, "
            f"columnwise={type(self._columnwise_storage).__name__}, "
            f"dtype={self._dtype})"
        )
