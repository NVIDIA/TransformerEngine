# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Custom tensor classes"""

import torch

from ..quantized_tensor import (
    QuantizedTensorStorage,
    QuantizedTensor,
    Quantizer,
    prepare_for_saving,
    restore_from_saved,
)
from .storage.float8_tensor_storage import Float8TensorStorage
from .storage.mxfp8_tensor_storage import MXFP8TensorStorage
from .storage.float8_blockwise_tensor_storage import Float8BlockwiseQTensorStorage
from .storage.nvfp4_tensor_storage import NVFP4TensorStorage
from .float8_tensor import Float8Tensor, Float8Quantizer, Float8CurrentScalingQuantizer
from .mxfp8_tensor import MXFP8Tensor, MXFP8Quantizer
from .float8_blockwise_tensor import Float8BlockwiseQTensor, Float8BlockQuantizer
from .nvfp4_tensor import NVFP4Tensor, NVFP4Quantizer
from .utils import cast_master_weights_to_fp8, replace_raw_data

__all__ = [
    "Quantizer",
    "Float8Quantizer",
    "Float8CurrentScalingQuantizer",
    "MXFP8Quantizer",
    "Float8BlockQuantizer",
    "NVFP4Quantizer",
    "QuantizedTensorStorage",
    "Float8TensorStorage",
    "MXFP8TensorStorage",
    "Float8BlockwiseQTensorStorage",
    "NVFP4TensorStorage",
    "QuantizedTensor",
    "Float8Tensor",
    "MXFP8Tensor",
    "Float8BlockwiseQTensor",
    "NVFP4Tensor",
    "prepare_for_saving",
    "restore_from_saved",
]


def _make_module_cast_func(dtype):
    """Make module cast function that can handle QuantizedTensor"""
    cast_func_name = {
        torch.float32: "float",
        torch.float16: "half",
        torch.bfloat16: "bfloat16",
    }[dtype]

    def tensor_cast_func(tensor: torch.Tensor) -> torch.Tensor:
        """Cast tensor dtype"""
        if isinstance(tensor, QuantizedTensor):
            return tensor.__class__.make_like(tensor, dtype=dtype)
        if tensor.is_floating_point():
            return getattr(tensor, cast_func_name)()
        return tensor

    def module_cast_func(self: torch.nn.Module) -> torch.nn.Module:
        """Cast module dtype"""
        return self._apply(tensor_cast_func)

    return module_cast_func


# Monkey-patch module cast functions to handle QuantizedTensor
torch.nn.Module.float = _make_module_cast_func(torch.float32)
torch.nn.Module.half = _make_module_cast_func(torch.float16)
torch.nn.Module.bfloat16 = _make_module_cast_func(torch.bfloat16)


def get_all_tensor_types():
    """
    Get all tensor-like types that can be used in TE.
    """
    all_tensor_types = [
        torch.Tensor,
        torch.nn.Parameter,
        Float8Tensor,
        Float8TensorStorage,
        MXFP8Tensor,
        MXFP8TensorStorage,
        Float8BlockwiseQTensor,
        Float8BlockwiseQTensorStorage,
        NVFP4Tensor,
        NVFP4TensorStorage,
    ]
    return all_tensor_types
