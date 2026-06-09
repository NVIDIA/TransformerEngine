# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Mixin class holding data for IdentityTensor (high-precision passthrough)."""

from __future__ import annotations
from typing import Any, Dict, Optional, Tuple

import torch

from ...quantized_tensor import QuantizedTensorStorage, Quantizer
from ...utils import _empty_tensor


class IdentityTensorStorage(QuantizedTensorStorage):
    """Passthrough storage that holds a high-precision (unquantized) tensor.

    Produced by :class:`IdentityQuantizer`. It implements the
    ``QuantizedTensorStorage`` interface so it can flow through the same
    module / GEMM / save-for-backward / FSDP machinery as the real quantized
    storages, but it performs no quantization: it simply carries the original
    high-precision tensor. ``general_gemm`` materializes it back to that plain
    tensor (so the matmul runs in high precision).

    The data is direction-agnostic -- the same tensor serves both the rowwise
    and columnwise directions (the GEMM transposes via its layout flags), so a
    single buffer is stored. This is what lets a ``HybridQuantizer`` mix one
    quantized direction with one high-precision direction.
    """

    _hp_data: Optional[torch.Tensor]
    _quantizer: Optional[Quantizer]

    def __new__(
        cls,
        *args,
        hp_data: Optional[torch.Tensor],
        fake_dtype: Optional[torch.dtype] = None,
        quantizer: Optional[Quantizer] = None,
        **kwargs,
    ):
        if cls is IdentityTensorStorage:
            instance = object.__new__(cls)
            if fake_dtype is not None:
                instance._dtype = fake_dtype
            elif hp_data is not None:
                instance._dtype = hp_data.dtype
            else:
                instance._dtype = torch.float32
        else:
            instance = super().__new__(cls, *args, fake_dtype=fake_dtype, **kwargs)
        instance._hp_data = hp_data
        instance._quantizer = quantizer.copy() if quantizer is not None else None
        return instance

    def clear(self):
        """Deallocate the held tensor's memory."""
        if self._hp_data is not None:
            self._hp_data.data = _empty_tensor()

    def copy_from_storage(self, src: QuantizedTensorStorage) -> None:
        """Copy data from another IdentityTensorStorage."""
        if not isinstance(src, IdentityTensorStorage):
            raise TypeError("copy_from_storage expects IdentityTensorStorage")
        if self._hp_data is not None and src._hp_data is not None:
            self._hp_data.copy_(src._hp_data)

    def get_metadata(self) -> Dict[str, Any]:
        """Get this tensor's metadata."""
        return {
            "hp_data": self._hp_data,
            "quantizer": self._quantizer,
            "fake_dtype": self._dtype,
        }

    def prepare_for_saving(
        self,
    ) -> Tuple[list[Optional[torch.Tensor]], "IdentityTensorStorage"]:
        """Prepare the tensor base for saving for backward."""
        tensors = [self._hp_data]
        self._hp_data = None
        return tensors, self

    def restore_from_saved(
        self, tensors: list[Optional[torch.Tensor]]
    ) -> list[Optional[torch.Tensor]]:
        """Restore the held tensor from the saved tensors list."""
        self._hp_data = tensors[0]
        return tensors[1:]

    def get_data_tensors(self, rowwise_data: bool = True, columnwise_data: bool = True):
        """Get this tensor's data. The single HP buffer serves both directions."""
        if rowwise_data and columnwise_data:
            return self._hp_data, None
        if rowwise_data:
            return self._hp_data
        if columnwise_data:
            return self._hp_data
        raise ValueError("No data to get, both rowwise_data and columnwise_data are False")

    def dequantize(self, *, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        """Return the held high-precision tensor (no-op dequantization)."""
        if self._hp_data is None:
            raise RuntimeError("IdentityTensorStorage has no data to dequantize")
        if dtype is not None and self._hp_data.dtype != dtype:
            return self._hp_data.to(dtype)
        return self._hp_data

    def update_usage(
        self,
        rowwise_usage: Optional[bool] = None,
        columnwise_usage: Optional[bool] = None,
    ):
        """No-op: the single high-precision buffer serves both directions."""
        # High-precision data is not direction-specific, so there is nothing
        # to drop or synthesize. Honor the request only insofar as keeping the
        # buffer (a request to drop both would leave no data, which is invalid).

    def get_usages(self) -> Dict[str, bool]:
        """Get the usage of the tensor."""
        has_data = self._hp_data is not None
        return {"rowwise": has_data, "columnwise": has_data}

    def size(self, *args, **kwargs):
        # pylint: disable=missing-function-docstring
        if self._hp_data is None:
            raise RuntimeError("IdentityTensorStorage has no data")
        return self._hp_data.size(*args, **kwargs)

    @property
    def device(self):
        """Return the device of the held tensor."""
        if self._hp_data is None:
            raise RuntimeError("IdentityTensorStorage has no data!")
        return self._hp_data.device

    def view(self, *shape):
        # pylint: disable=missing-function-docstring
        flat_shape = shape[0] if len(shape) == 1 and not isinstance(shape[0], int) else shape
        return IdentityTensorStorage(
            hp_data=self._hp_data.view(*flat_shape) if self._hp_data is not None else None,
            fake_dtype=self._dtype,
            quantizer=self._quantizer,
        )

    def fsdp_buffer_fields(self) -> Tuple[str, ...]:
        """Field gathered by FSDP2 for the high-precision passthrough."""
        return ("_hp_data",)

    def __repr__(self):
        return f"IdentityTensorStorage(dtype={self._dtype}, data={self._hp_data})"
