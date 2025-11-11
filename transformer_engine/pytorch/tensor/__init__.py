# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Custom tensor classes"""

import torch

from .quantized_tensor import QuantizedTensor, Quantizer
from .utils import cast_master_weights_to_fp8, replace_raw_data

__all__ = [
    "QuantizedTensor",
    "Quantizer",
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
    from transformer_engine.pytorch.tensor.float8_tensor import Float8Tensor, Float8TensorBase
    from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Tensor, MXFP8TensorBase
    from transformer_engine.pytorch.tensor.float8_blockwise_tensor import (
        Float8BlockwiseQTensor,
        Float8BlockwiseQTensorBase,
    )
    from transformer_engine.pytorch.tensor.nvfp4_tensor import NVFP4Tensor, NVFP4TensorBase

    all_tensor_types = [
        torch.Tensor,
        torch.nn.Parameter,
        Float8Tensor,
        Float8TensorBase,
        MXFP8Tensor,
        MXFP8TensorBase,
        Float8BlockwiseQTensor,
        Float8BlockwiseQTensorBase,
        NVFP4Tensor,
        NVFP4TensorBase,
    ]
    return all_tensor_types
