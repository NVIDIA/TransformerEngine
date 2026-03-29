# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Extract raw tensor data + quantization metadata from quantized tensor types.

Used by the stable ABI Python shim to convert quantized tensors into
raw buffers that can be passed to stable ABI ops.
"""

import torch

# TE DType values (must match transformer_engine/transformer_engine.h)
_DTYPE_MAP = {
    torch.float32: 4,  # kFloat32
    torch.float16: 5,  # kFloat16
    torch.bfloat16: 6,  # kBFloat16
    torch.uint8: 0,  # kByte (used for FP8 storage)
    torch.int32: 2,  # kInt32
    torch.int64: 3,  # kInt64
    torch.bool: 0,  # kByte
}

# FP8 dtype enum values — maps various string representations to TE dtype ints
_FP8_DTYPE_TO_TE = {
    "fp8e4m3": 7,  # kFloat8E4M3
    "fp8e5m2": 8,  # kFloat8E5M2
    "torch.float8_e4m3fn": 7,
    "torch.float8_e5m2": 8,
    "float8_e4m3fn": 7,
    "float8_e5m2": 8,
    # Integer DType enum values (set by _stable_torch_module.py)
    "7": 7,
    "8": 8,
}

# Scaling mode values (must match transformer_engine.h NVTEScalingMode enum)
NVTE_DELAYED_TENSOR_SCALING = 0
NVTE_MXFP8_1D_SCALING = 1
NVTE_BLOCK_SCALING_1D = 2
NVTE_BLOCK_SCALING_2D = 3
NVTE_NVFP4_1D_SCALING = 4


def _detect_scaling_mode(tensor):
    """Detect the NVTEScalingMode for a quantized tensor.

    Checks tensor attributes (_is_2D_scaled, _block_scaling_dim) first,
    then falls back to type-name detection for MXFP8/NVFP4 tensors which
    lack those attributes.
    """
    if hasattr(tensor, "_is_2D_scaled"):
        return NVTE_BLOCK_SCALING_2D if tensor._is_2D_scaled else NVTE_BLOCK_SCALING_1D
    if hasattr(tensor, "_block_scaling_dim"):
        return NVTE_BLOCK_SCALING_2D if tensor._block_scaling_dim == 2 else NVTE_BLOCK_SCALING_1D
    cls_name = type(tensor).__name__
    if "MXFP8" in cls_name:
        return NVTE_MXFP8_1D_SCALING
    if "NVFP4" in cls_name:
        return NVTE_NVFP4_1D_SCALING
    quantizer = getattr(tensor, "_quantizer", None)
    if quantizer is not None:
        q_name = type(quantizer).__name__
        if "MXFP8" in q_name:
            return NVTE_MXFP8_1D_SCALING
        if "NVFP4" in q_name:
            return NVTE_NVFP4_1D_SCALING
        if "Block" in q_name:
            block_dim = getattr(quantizer, "block_scaling_dim", None)
            return NVTE_BLOCK_SCALING_2D if block_dim == 2 else NVTE_BLOCK_SCALING_1D
    return NVTE_DELAYED_TENSOR_SCALING


def extract_tensor_data(tensor):
    """Extract raw data, dtype, scale_inv, and scaling_mode from a tensor.

    For regular PyTorch tensors, returns the tensor as-is with default metadata.
    For quantized TE tensor types, extracts the underlying raw buffers.

    Returns:
        (data, te_dtype, scale_inv, scaling_mode)
    """
    # Check for quantized TE tensor types FIRST (they subclass torch.Tensor)
    # TE quantized tensors have _rowwise_data or _data attributes
    if hasattr(tensor, "_rowwise_data") and tensor._rowwise_data is not None:
        data = tensor._rowwise_data
        scale_inv = getattr(tensor, "_rowwise_scale_inv", None)
        fp8_dtype = getattr(tensor, "_fp8_dtype", None)
        te_dtype = 0  # kByte
        if fp8_dtype is not None:
            te_dtype = _FP8_DTYPE_TO_TE.get(str(fp8_dtype), 7)
        if hasattr(tensor, "_is_2D_scaled"):
            sm = NVTE_BLOCK_SCALING_2D if tensor._is_2D_scaled else NVTE_BLOCK_SCALING_1D
        elif hasattr(tensor, "_block_scaling_dim"):
            sm = NVTE_BLOCK_SCALING_2D if tensor._block_scaling_dim == 2 else NVTE_BLOCK_SCALING_1D
        else:
            sm = NVTE_DELAYED_TENSOR_SCALING
        return data, te_dtype, scale_inv, sm

    # Columnwise-only block-scaling tensor (after update_usage(rowwise_usage=False)).
    # _rowwise_data is None but _columnwise_data exists — return columnwise data with
    # correct scaling_mode so callers don't fall through to the generic tensor path (sm=0).
    if (
        hasattr(tensor, "_rowwise_data")
        and tensor._rowwise_data is None
        and hasattr(tensor, "_columnwise_data")
        and tensor._columnwise_data is not None
        and not hasattr(tensor, "_data")
    ):
        col_data = tensor._columnwise_data
        col_si = getattr(tensor, "_columnwise_scale_inv", None)
        fp8_dtype = getattr(tensor, "_fp8_dtype", None)
        te_dtype = 0  # kByte
        if fp8_dtype is not None:
            te_dtype = _FP8_DTYPE_TO_TE.get(str(fp8_dtype), 7)
        if hasattr(tensor, "_is_2D_scaled"):
            sm = NVTE_BLOCK_SCALING_2D if tensor._is_2D_scaled else NVTE_BLOCK_SCALING_1D
        elif hasattr(tensor, "_block_scaling_dim"):
            sm = NVTE_BLOCK_SCALING_2D if tensor._block_scaling_dim == 2 else NVTE_BLOCK_SCALING_1D
        else:
            sm = NVTE_DELAYED_TENSOR_SCALING
        return col_data, te_dtype, col_si, sm

    if hasattr(tensor, "_data") and tensor._data is not None:
        data = tensor._data
        scale_inv = getattr(tensor, "_scale_inv", None)
        fp8_dtype = getattr(tensor, "_fp8_dtype", None)
        te_dtype = 0
        if fp8_dtype is not None:
            te_dtype = _FP8_DTYPE_TO_TE.get(str(fp8_dtype), 7)
        return data, te_dtype, scale_inv, NVTE_DELAYED_TENSOR_SCALING

    if isinstance(tensor, torch.Tensor):
        # Regular PyTorch tensor
        te_dtype = _DTYPE_MAP.get(tensor.dtype, 4)  # default kFloat32
        return tensor, te_dtype, None, NVTE_DELAYED_TENSOR_SCALING

    # Try Float8TensorStorage
    try:
        from transformer_engine.pytorch.tensor.storage.float8_tensor_storage import (
            Float8TensorStorage,
        )

        if isinstance(tensor, Float8TensorStorage):
            data = tensor._data  # uint8 tensor
            scale_inv = tensor._scale_inv  # float32 tensor
            fp8_dtype = str(tensor._fp8_dtype)
            te_dtype = _FP8_DTYPE_TO_TE.get(fp8_dtype, 7)  # default e4m3
            return data, te_dtype, scale_inv, NVTE_DELAYED_TENSOR_SCALING
    except ImportError:
        pass

    # Try MXFP8TensorStorage
    try:
        from transformer_engine.pytorch.tensor.storage.mxfp8_tensor_storage import (
            MXFP8TensorStorage,
        )

        if isinstance(tensor, MXFP8TensorStorage):
            data = tensor._rowwise_data
            scale_inv = tensor._rowwise_scale_inv
            fp8_dtype = str(tensor._fp8_dtype)
            te_dtype = _FP8_DTYPE_TO_TE.get(fp8_dtype, 7)
            return data, te_dtype, scale_inv, NVTE_MXFP8_1D_SCALING
    except ImportError:
        pass

    # Try Float8BlockwiseQTensorStorage
    try:
        from transformer_engine.pytorch.tensor.storage.float8_blockwise_tensor_storage import (
            Float8BlockwiseQTensorStorage,
        )

        if isinstance(tensor, Float8BlockwiseQTensorStorage):
            data = tensor._rowwise_data
            scale_inv = tensor._rowwise_scale_inv
            fp8_dtype = str(tensor._fp8_dtype)
            te_dtype = _FP8_DTYPE_TO_TE.get(fp8_dtype, 7)
            # Check 1D vs 2D block scaling
            sm = NVTE_BLOCK_SCALING_2D if tensor._is_2D_scaled else NVTE_BLOCK_SCALING_1D
            return data, te_dtype, scale_inv, sm
    except ImportError:
        pass

    # Try NVFP4TensorStorage
    try:
        from transformer_engine.pytorch.tensor.storage.nvfp4_tensor_storage import (
            NVFP4TensorStorage,
        )

        if isinstance(tensor, NVFP4TensorStorage):
            data = tensor._rowwise_data
            scale_inv = tensor._rowwise_scale_inv
            te_dtype = 10  # kFloat4E2M1
            return data, te_dtype, scale_inv, NVTE_NVFP4_1D_SCALING
    except ImportError:
        pass

    # Fallback: treat as regular tensor
    if hasattr(tensor, "data") and isinstance(tensor.data, torch.Tensor):
        return tensor.data, _DTYPE_MAP.get(tensor.data.dtype, 4), None, NVTE_DELAYED_TENSOR_SCALING

    raise TypeError(f"Cannot extract tensor data from type {type(tensor)}")
