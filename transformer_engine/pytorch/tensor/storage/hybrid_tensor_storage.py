# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Mixin class holding data specific for HybridQuantizedTensor"""

from __future__ import annotations
from collections.abc import Iterable
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
    _quantizer: Quantizer

    def __new__(
        cls,
        *args,
        rowwise_storage: Optional[QuantizedTensorStorage],
        columnwise_storage: Optional[QuantizedTensorStorage],
        quantizer: Quantizer,
        fake_dtype: Optional[torch.dtype] = None,
        **kwargs,
    ):
        if quantizer is None or not isinstance(quantizer, Quantizer):
            raise TypeError(
                "HybridQuantizedTensorStorage requires a parent HybridQuantizer; "
                f"got {type(quantizer).__name__}."
            )
        if not hasattr(quantizer, "rowwise_quantizer") or not hasattr(
            quantizer, "columnwise_quantizer"
        ):
            raise TypeError(
                "HybridQuantizedTensorStorage requires a parent HybridQuantizer "
                "with rowwise_quantizer and columnwise_quantizer attributes; "
                f"got {type(quantizer).__name__}."
            )

        if cls is HybridQuantizedTensorStorage:
            instance = object.__new__(cls)
            instance._dtype = fake_dtype if fake_dtype is not None else torch.float32
        else:
            instance = super().__new__(cls, *args, fake_dtype=fake_dtype, **kwargs)

        instance._rowwise_storage = rowwise_storage
        instance._columnwise_storage = columnwise_storage
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

    def clear(self):
        """Deallocate both sub-storages' buffers.

        Delegates to each sub-storage's own ``clear()``; no-op when a
        sub-storage is ``None`` (columnwise-only or rowwise-only hybrid).

        Used by ``cpu_offload_v1`` after the offloader has taken its own
        reference to the extracted buffers, to release the GPU-resident
        originals.
        """
        if self._rowwise_storage is not None:
            self._rowwise_storage.clear()
        if self._columnwise_storage is not None:
            self._columnwise_storage.clear()

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
        """Dequantize using the first available sub-storage."""
        if dtype is None:
            dtype = self._dtype
        if self._rowwise_storage is not None:
            return self._rowwise_storage.dequantize(dtype=dtype)
        if self._columnwise_storage is not None:
            return self._columnwise_storage.dequantize(dtype=dtype)
        raise RuntimeError("HybridQuantizedTensorStorage has no data to dequantize")

    def get_data_tensors(self):
        """Return raw data tensors from both available sub-storages."""
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
        """Return the logical size from the first available sub-storage."""
        if self._rowwise_storage is not None:
            return self._rowwise_storage.size(*args, **kwargs)
        if self._columnwise_storage is not None:
            return self._columnwise_storage.size(*args, **kwargs)
        raise RuntimeError("HybridQuantizedTensorStorage has no data")

    @property
    def device(self):
        """Return the device from the first available sub-storage."""
        if self._rowwise_storage is not None:
            return self._rowwise_storage.device
        if self._columnwise_storage is not None:
            return self._columnwise_storage.device
        raise RuntimeError("HybridQuantizedTensorStorage has no data")

    def view(self, *shape):
        """View delegates to each sub-storage. Used by FSDP2 reset_sharded_param.

        Identity views are handled without forwarding a reshape to the
        sub-storages: the columnwise sub-storage's own shape is transposed
        relative to the hybrid for some formats (e.g. a 2D block-scaled
        Float8BlockwiseQTensor has shape ``(N, M)`` for an ``(M, N)`` weight),
        so forwarding the hybrid's row-major shape would be a spurious
        last-2-dims change that dequantizes it to a plain tensor.
        """
        flat_shape = shape[0] if len(shape) == 1 and isinstance(shape[0], Iterable) else shape
        if list(flat_shape) == list(self.size()):
            return HybridQuantizedTensorStorage(
                rowwise_storage=self._rowwise_storage,
                columnwise_storage=self._columnwise_storage,
                quantizer=self._quantizer,
                fake_dtype=self._dtype,
            )
        row_view = self._rowwise_storage.view(*shape) if self._rowwise_storage is not None else None
        col_view = (
            self._columnwise_storage.view(*shape) if self._columnwise_storage is not None else None
        )
        return HybridQuantizedTensorStorage(
            rowwise_storage=row_view,
            columnwise_storage=col_view,
            quantizer=self._quantizer,
            fake_dtype=self._dtype,
        )

    def get_metadata(self) -> Dict[str, Any]:
        """Return constructor metadata for make_like and serialization paths."""
        return {
            "rowwise_storage": self._rowwise_storage,
            "columnwise_storage": self._columnwise_storage,
            "quantizer": self._quantizer,
            "fake_dtype": self._dtype,
        }

    def __repr__(self):
        row_type = (
            type(self._rowwise_storage).__name__ if self._rowwise_storage is not None else "None"
        )
        col_type = (
            type(self._columnwise_storage).__name__
            if self._columnwise_storage is not None
            else "None"
        )
        return (
            "HybridQuantizedTensorStorage("
            f"rowwise={row_type}, "
            f"columnwise={col_type}, "
            f"dtype={self._dtype})"
        )
