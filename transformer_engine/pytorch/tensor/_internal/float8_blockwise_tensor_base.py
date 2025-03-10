# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Mixin class holding data specific for Float8BlockwiseQTensor"""

from __future__ import annotations
import math
from typing import Optional, Dict, Any, Tuple
import torch

from transformer_engine_torch import DType as TE_DType

from ...constants import TE_DType_To_Torch

from ..quantized_tensor import Quantizer


class Float8BlockwiseQTensorBase:
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

    def __new__(
        cls,
        *args,
        rowwise_data: torch.Tensor,
        rowwise_scale_inv: torch.Tensor,
        columnwise_data: Optional[torch.Tensor],
        columnwise_scale_inv: Optional[torch.Tensor],
        fp8_dtype: TE_DType,
        quantizer: Quantizer,
        **kwargs,
    ):
        instance = super().__new__(cls, *args, **kwargs)
        instance._rowwise_data = rowwise_data
        instance._columnwise_data = columnwise_data
        instance._quantizer = quantizer
        instance._fp8_dtype = fp8_dtype
        instance._rowwise_scale_inv = rowwise_scale_inv
        instance._columnwise_scale_inv = columnwise_scale_inv

        return instance

    def get_metadata(self) -> Dict[str, Any]:
        """Get this tensor's metadata."""
        return {
            "rowwise_data": self._rowwise_data,
            "rowwise_scale_inv": self._rowwise_scale_inv,
            "columnwise_data": self._columnwise_data,
            "columnwise_scale_inv": self._columnwise_scale_inv,
            "fp8_dtype": self._fp8_dtype,
            "quantizer": self._quantizer,
        }

    def prepare_for_saving(
        self,
    ) -> Tuple[list[Optional[torch.Tensor]], Float8BlockwiseQTensorBase]:
        """Prepare the tensor base for saving for backward

        FIXME(kwyss): Set data tensors to None and consider saving/restoring scales.
        test_numerics.py fails when tensors are cleared at the moment in C++ shape logic.
        """
        tensors = [self._rowwise_data,
                   self._columnwise_data]
        return tensors, self

    def restore_from_saved(
        self, tensors: list[Optional[torch.Tensor]]
    ) -> list[Optional[torch.Tensor]]:
        """Restore the tensor base data from the saved tensors list."""
        self._rowwise_data = tensors[0]
        self._columnwise_data = tensors[1]
        return tensors[2:]

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
        assert self._quantizer is not None
        if self._quantizer.block_scaling_dim != 2:
            assert self._quantizer.block_scaling_dim == 1
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
