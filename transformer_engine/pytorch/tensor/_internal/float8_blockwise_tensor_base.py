# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Mixin class holding data specific for Float8BlockwiseQTensor"""

from __future__ import annotations
import math
from typing import Optional, Dict, Any, Tuple
import torch

import transformer_engine_torch as tex
from transformer_engine_torch import DType as TE_DType

from ..quantized_tensor import QuantizedTensorBase

from ...constants import TE_DType_To_Torch

from ..quantized_tensor import Quantizer

from ...utils import _empty_tensor


class Float8BlockwiseQTensorBase(QuantizedTensorBase):
    """Mixin class that holds data attributes of Float8BlockwiseQTensor.

    Float8BlockwiseQTensor inherits from the PyTorch tensor class and this
    mixin class. If this class is instantiated directly, it has the same
    data, lower CPU overhead, and less functionality. It should only
    be instantiated directly for performance-critical internal usage.
    """

    _rowwise_data: Optional[torch.Tensor]
    _columnwise_data: Optional[torch.Tensor]
    _quantizer: Quantizer
    _fp8_dtype: TE_DType
    _rowwise_scale_inv: Optional[torch.Tensor]
    _columnwise_scale_inv: Optional[torch.Tensor]
    _is_2D_scaled: bool

    def __new__(
        cls,
        *args,
        rowwise_data: Optional[torch.Tensor],
        rowwise_scale_inv: Optional[torch.Tensor],
        columnwise_data: Optional[torch.Tensor],
        columnwise_scale_inv: Optional[torch.Tensor],
        fp8_dtype: TE_DType,
        quantizer: Quantizer,
        is_2D_scaled: bool,
        **kwargs,
    ):
        instance = super().__new__(cls, *args, **kwargs)
        instance._rowwise_data = rowwise_data
        instance._columnwise_data = columnwise_data
        instance._quantizer = quantizer
        instance._fp8_dtype = fp8_dtype
        instance._rowwise_scale_inv = rowwise_scale_inv
        instance._columnwise_scale_inv = columnwise_scale_inv
        instance._is_2D_scaled = is_2D_scaled

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

    def get_metadata(self) -> Dict[str, Any]:
        """Get this tensor's metadata."""
        return {
            "rowwise_data": self._rowwise_data,
            "rowwise_scale_inv": self._rowwise_scale_inv,
            "columnwise_data": self._columnwise_data,
            "columnwise_scale_inv": self._columnwise_scale_inv,
            "fp8_dtype": self._fp8_dtype,
            "quantizer": self._quantizer,
            "is_2D_scaled": self._is_2D_scaled,
        }

    def prepare_for_saving(
        self,
    ) -> Tuple[list[Optional[torch.Tensor]], Float8BlockwiseQTensorBase]:
        """
        Prepare the tensor base for saving for backward
        """
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

    def get_data_tensors(self):
        """Get this Tensor's data."""
        return self._rowwise_data, self._columnwise_data

    def _transpose_dq_columnwise_output(self, columnwise_dq: torch.Tensor) -> torch.Tensor:
        """Takes dequantized columnwise data and permutes to a rowwise shape"""
        if columnwise_dq.dim() < 2:
            return columnwise_dq
        permute_dims = list(range(1, columnwise_dq.dim()))
        permute_dims.append(0)
        return torch.permute(columnwise_dq, tuple(permute_dims)).contiguous()

    def _dequantize_vectorwise(self, *, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        block_len = 128

        q_M, q_K = 1, 1
        if self._rowwise_data is not None:
            q = self._rowwise_data
            scale_inv = self._rowwise_scale_inv
            transpose_output = False
            if len(q.shape) >= 1:
                q_K = q.shape[-1]
            for i in range(len(q.shape) - 1):
                q_M *= q.shape[i]
        else:
            assert self._columnwise_data is not None, "No data to dequantize"
            q = self._columnwise_data
            scale_inv = self._columnwise_scale_inv
            transpose_output = True
            if len(q.shape) >= 1:
                q_M = q.shape[0]
            for i in range(1, len(q.shape)):
                q_K *= q.shape[i]

        orig_shape = q.shape
        q = q.reshape(q_M, q_K)
        k_tiles, scale_m = scale_inv.shape
        if q_K % block_len != 0:
            k_pad_amount = (block_len - (q_K % block_len)) % block_len
            q = torch.nn.functional.pad(
                q, (0, k_pad_amount, 0, 0), mode="constant", value=0
            ).contiguous()
        _, padded_K = q.shape
        q_tiled = q.reshape(q_M, k_tiles, block_len)
        if scale_m > q_M:
            # scale_m is 4 element aligned.
            scale_inv = scale_inv[:, :q_M].contiguous()
        dq_scale = scale_inv.transpose(-2, -1).contiguous().reshape(q_M, k_tiles, 1)
        torch_q_dtype = TE_DType_To_Torch[self._fp8_dtype]
        result = q_tiled.view(torch_q_dtype).to(torch.float32) * dq_scale
        if padded_K != q_K:
            result = result.reshape(q_M, padded_K)[:, :q_K]
        result = result.to(dtype)
        if len(orig_shape) == 0:
            result = result.reshape([])
        else:
            result = result.reshape(*orig_shape).contiguous()

        if transpose_output:
            return self._transpose_dq_columnwise_output(result)
        return result

    def dequantize(self, *, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """
        Construct plain PyTorch tensor from Float8BlockwiseQTensor
        """
        block_len = 128
        if not self._is_2D_scaled:
            return self._dequantize_vectorwise(dtype=dtype)

        def format_scale_as_logical_shape(q_K, scales, block_len):
            # The GEMM for 2D blocks required padding in the scales.
            derived_scale_k_shape = math.ceil(q_K / block_len)
            _, scale_K = scales.shape
            if derived_scale_k_shape == scale_K:
                return scales
            return scales[:, :derived_scale_k_shape].contiguous()

        q_M, q_K = 1, 1
        if self._rowwise_data is not None:
            q = self._rowwise_data
            scale_inv = self._rowwise_scale_inv
            transpose_output = False
            if len(q.shape) >= 1:
                q_K = q.shape[-1]
            for i in range(len(q.shape) - 1):
                q_M *= q.shape[i]
        else:
            assert self._columnwise_data is not None, "No data to dequantize"
            q = self._columnwise_data
            scale_inv = self._columnwise_scale_inv
            transpose_output = True
            if len(q.shape) >= 1:
                q_M = q.shape[0]
            for i in range(1, len(q.shape)):
                q_K *= q.shape[i]

        orig_shape = q.shape
        q = q.reshape(q_M, q_K)
        formatted_scales = format_scale_as_logical_shape(q_K, scale_inv, block_len)
        assert len(formatted_scales.shape) == 2
        m_tiles, k_tiles = formatted_scales.shape
        unpadded_m, unpadded_k = q_M, q_K
        m_block_len = block_len
        k_block_len = block_len
        if q_M % m_block_len != 0 or q_K % k_block_len != 0:
            m_pad_amount = (m_block_len - (q_M % m_block_len)) % m_block_len
            k_pad_amount = (k_block_len - (q_K % k_block_len)) % k_block_len
            q = torch.nn.functional.pad(
                q, (0, k_pad_amount, 0, m_pad_amount), mode="constant", value=0
            ).contiguous()
        padded_M, padded_K = q.shape
        q_tiled = q.reshape(m_tiles, m_block_len, k_tiles, k_block_len)

        torch_q_dtype = TE_DType_To_Torch[self._fp8_dtype]

        result = q_tiled.view(torch_q_dtype).to(torch.float32) * formatted_scales.view(
            m_tiles, 1, k_tiles, 1
        )
        result = result.view(padded_M, padded_K).to(dtype)
        if padded_M != unpadded_m or padded_K != unpadded_k:
            result = result[:unpadded_m, :unpadded_k]
        if len(orig_shape) == 0:
            result = result.reshape([])
        else:
            result = result.reshape(*orig_shape).contiguous()
        if transpose_output:
            return self._transpose_dq_columnwise_output(result)
        return result

    def size(self, *args, **kwargs):
        # pylint: disable=missing-function-docstring
        if self._rowwise_data is not None:
            return self._rowwise_data.size(*args, **kwargs)
        dims = list(self._columnwise_data.size(*args, **kwargs))
        reordered = []
        for i in range(1, len(dims)):
            reordered.append(dims[i])
        reordered.append(dims[0])
        return torch.Size(reordered)

    def _create_columnwise(self):
        """
        Update columnwise data and columnwise scale inv. Can only be used when using 2D scaling.
        """
        assert self._is_2D_scaled, "Cannot create columnwise data when not using 2D scaling."

        rowwise_data = self._rowwise_data
        if not rowwise_data.is_contiguous():
            rowwise_data = rowwise_data.contiguous()
        self._columnwise_data = tex.fp8_transpose(
            rowwise_data, self._fp8_dtype, out=self._columnwise_data
        )

        if self._columnwise_scale_inv is None:
            assert self._quantizer is not None, (
                "._quantizer of Float8BlockwiseQTensor cannot be None because all the blockwise "
                "quantized tensors are supposed to be generated from the quantizer."
            )
            columnwise_scale_inv_shape = self._quantizer.get_scale_shape(rowwise_data.shape, True)
            self._columnwise_scale_inv = torch.empty(
                columnwise_scale_inv_shape,
                dtype=self._rowwise_scale_inv.dtype,
                device=self._rowwise_scale_inv.device,
            )
        assert len(self._rowwise_scale_inv.shape) == 2
        assert len(self._columnwise_scale_inv.shape) == 2
        rowwise_scale_inv = self._rowwise_scale_inv
        columnwise_scale_inv = rowwise_scale_inv.transpose(-2, -1)
        h = min(self._columnwise_scale_inv.shape[0], columnwise_scale_inv.shape[0])
        w = min(self._columnwise_scale_inv.shape[1], columnwise_scale_inv.shape[1])
        self._columnwise_scale_inv[0:h, 0:w].copy_(columnwise_scale_inv[0:h, 0:w])

    def __repr__(self):
        if self._rowwise_data is not None:
            data = self.dequantize()
            descriptor = "rowwise"
        else:
            data = self.dequantize()
            descriptor = "columnwise"
        return (
            "Float8BlockwiseQTensorBase("
            f"fp8_dtype={self._fp8_dtype}, "
            f"{descriptor}_scaled_data={data}"
        )

    def update_usage(
        self, rowwise_usage: Optional[bool] = None, columnwise_usage: Optional[bool] = None
    ):
        """
        update_usage can be used to clear out one of two possible copies of the data.
        """

        if rowwise_usage is None:
            rowwise_usage = self._rowwise_data is not None
        if columnwise_usage is None:
            columnwise_usage = self._columnwise_data is not None
        assert (
            columnwise_usage or rowwise_usage
        ), "Must retain some data either columnwise or rowwise"

        if columnwise_usage and rowwise_usage:
            if not self._is_2D_scaled:
                # For 1D scaling, we cannot create columnwise data/scale_inv from rowwise
                # data/scale_inv because their scale values are different.
                assert (
                    self._rowwise_data is not None
                    and self._rowwise_scale_inv is not None
                    and self._columnwise_data is not None
                    and self._columnwise_scale_inv is not None
                ), "Cannot update to rowwise and columnwise usage."
            else:
                # For 2D scaling, if columnwise data/scale_inv is None, we can create them from
                # rowwise data/scale_inv.
                assert (
                    self._rowwise_data is not None and self._rowwise_scale_inv is not None
                ), "Cannot update to rowwise and columnwise usage because rowwise data is None."
                if self._columnwise_data is None or self._columnwise_scale_inv is None:
                    self._create_columnwise()
            return

        if rowwise_usage:
            assert (
                self._rowwise_data is not None and self._rowwise_scale_inv is not None
            ), "Cannot update to rowwise usage."
            self._columnwise_data = None
            self._columnwise_scale_inv = None
            return
        if columnwise_usage:
            assert (
                self._columnwise_data is not None and self._columnwise_scale_inv is not None
            ), "Cannot update to columnwise usage."
            self._rowwise_data = None
            self._rowwise_scale_inv = None
            return

        return
