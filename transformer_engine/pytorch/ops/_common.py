# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Helper functions used in fusible operations."""

from __future__ import annotations
from typing import Optional

import torch

from transformer_engine_torch import FP8TensorMeta
from ..torch_version import torch_version
from ..quantization import FP8GlobalStateManager
from ..tensor.float8_tensor import Float8Tensor
from ..quantized_tensor import QuantizedTensorStorage
from ..utils import canonicalize_dtype
from ..module._common import noop_cat
from ..tensor import Quantizer
from ..tensor.grouped_tensor import GroupedTensor


def is_quantized_tensor(tensor: torch.Tensor | QuantizedTensorStorage) -> bool:
    """Check if tensor is a quantized tensor"""
    return isinstance(tensor, QuantizedTensorStorage)


def maybe_dequantize(
    tensor: torch.Tensor | QuantizedTensorStorage, dtype: torch.dtype | None = None
) -> torch.Tensor:
    """Dequantize tensor to given dtype or just convert if not a quantized tensor"""
    if is_quantized_tensor(tensor):
        return tensor.dequantize(dtype=dtype)
    if dtype is not None and tensor.dtype != dtype:
        tensor = tensor.to(dtype)
    if not tensor.is_contiguous():
        tensor = tensor.contiguous()
    return tensor


def maybe_autocast_dtype(
    *,
    device_type: str = "cuda",
    default_dtype: Optional[torch.dtype] = None,
) -> torch.dtype:
    """Get autocast dtype if enabled"""

    if torch_version() >= (2, 4, 3):
        if torch.is_autocast_enabled(device_type):
            return torch.get_autocast_dtype(device_type)
    else:
        if torch.is_autocast_enabled():
            return torch.get_autocast_gpu_dtype()
    return canonicalize_dtype(default_dtype)


def get_fp8_meta_from_fp8_tensor(tensor: Float8Tensor) -> tuple[FP8TensorMeta, int]:
    """Get FP8TensorMeta object and index corresponding to Float8Tensor

    Constructs FP8TensorMeta if needed.

    """

    # Check if tensor already has FP8 metadata
    if tensor._fp8_meta is not None:
        key = FP8GlobalStateManager.get_meta_tensor_key(
            forward=tensor._fp8_meta_forward,
        )
        return tensor._fp8_meta[key], tensor._fp8_meta_index

    # Create FP8TensorMeta class
    fp8_meta = FP8TensorMeta()
    fp8_meta.scale = tensor._scale_inv.reciprocal()
    fp8_meta.amax_history = torch.empty(1, 1, dtype=torch.float32, device=tensor.device)
    fp8_meta.scale_inv = tensor._scale_inv
    return fp8_meta, 0


def make_grouped_tensor_from_buffers(
    *,
    num_groups: int,
    data: torch.Tensor,
    split_sizes: torch.Tensor,
    columnwise_data: torch.Tensor = None,
    scale_inv: torch.Tensor = None,
    columnwise_scale_inv: torch.Tensor = None,
    tensor_offsets: torch.Tensor = None,
    logical_last_dim: int,
    dtype: torch.dtype,
    quantizer: Quantizer = None,
    with_gemm_swizzled_scales: bool = False,
) -> GroupedTensor:
    """Build GroupedTensor from FC1+SwiGLU / dSwiGLU kernel outputs.

    Scales are already in GEMM swizzled layout.
    """
    if tensor_offsets is None:
        tensor_offsets = GroupedTensor.make_tensor_offsets(split_sizes, logical_last_dim)
    logical_first_dim = data.shape[0] if data is not None else columnwise_data.shape[0]
    ndim = data.ndim if data is not None else columnwise_data.ndim
    if ndim == 1:
        logical_first_dim = logical_first_dim // logical_last_dim
    return GroupedTensor(
        shape=(logical_first_dim, logical_last_dim),
        dtype=dtype,
        quantizer=quantizer,
        num_tensors=num_groups,
        data=data,
        columnwise_data=columnwise_data,
        scale_inv=scale_inv,
        columnwise_scale_inv=columnwise_scale_inv,
        amax=None,
        columnwise_amax=None,
        scale=None,
        first_dims=split_sizes,
        last_dims=None,
        tensor_offsets=tensor_offsets,
        offsets=None,
        scale_inv_offsets=None,
        columnwise_scale_inv_offsets=None,
        with_gemm_swizzled_scales=with_gemm_swizzled_scales,
    )


def make_grouped_tensor_from_mxfp8_weights(
    weights: list,
    quantizer: Quantizer,
    device: torch.device,
    dtype: torch.dtype,
    with_gemm_swizzled_scales: bool = False,
) -> GroupedTensor:
    """Build a GroupedTensor from MXFP8 weight tensors by packing their buffers (no copy when contiguous)."""
    num_groups = len(weights)
    weight_shape = weights[0].shape
    O, I = weight_shape[0], weight_shape[1]
    logical_first_dim = num_groups * O
    logical_last_dim = I

    tensor_offsets = None
    data = None
    scale_inv = None
    scale_inv_offsets = None
    columnwise_data = None
    columnwise_scale_inv = None
    columnwise_scale_inv_offsets = None

    # Pack rowwise into data/scale_inv when available.
    # GEMM expects scales in swizzled layout (same as FC1 weight scales in grouped_gemm_swiglu).
    if weights[0]._rowwise_data is not None:
        data = noop_cat([w._rowwise_data.reshape(-1) for w in weights])
        rowwise_scales = noop_cat([w._rowwise_scale_inv for w in weights])
        if with_gemm_swizzled_scales:
            rowwise_scales = rowwise_scales.view(num_groups, O // 128, 4, 32, I // 128, 4)
            rowwise_scales = rowwise_scales.permute(0, 1, 4, 3, 2, 5).contiguous()
        scale_inv = rowwise_scales.reshape(-1)
    # Pack columnwise into columnwise_* when available.
    # GEMM expects columnwise scales in swizzled layout (same as FC2 weight scales in backward dSwiGLU kernel).
    if weights[0]._columnwise_data is not None:
        columnwise_data = noop_cat([w._columnwise_data.reshape(-1) for w in weights])
        columnwise_scales = noop_cat([w._columnwise_scale_inv for w in weights])
        if with_gemm_swizzled_scales:
            columnwise_scales = columnwise_scales.view(num_groups, O // 128, 4, I // 128, 4, 32)
            columnwise_scales = columnwise_scales.permute(0, 3, 1, 5, 4, 2).contiguous()
        columnwise_scale_inv = columnwise_scales.reshape(-1)

    return GroupedTensor(
        shape=(logical_first_dim, logical_last_dim),
        dtype=dtype,
        num_tensors=num_groups,
        quantizer=quantizer,
        data=data,
        columnwise_data=columnwise_data,
        scale_inv=scale_inv,
        columnwise_scale_inv=columnwise_scale_inv,
        amax=None,
        columnwise_amax=None,
        scale=None,
        first_dims=None,
        last_dims=None,
        tensor_offsets=tensor_offsets,
        offsets=None,
        scale_inv_offsets=scale_inv_offsets,
        columnwise_scale_inv_offsets=columnwise_scale_inv_offsets,
        with_gemm_swizzled_scales=with_gemm_swizzled_scales,
    )
