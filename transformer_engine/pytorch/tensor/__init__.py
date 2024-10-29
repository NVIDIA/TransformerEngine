# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Custom tensor classes"""

import torch

from .float8_tensor import Float8Tensor
from .quantized_tensor import QuantizedTensor

__all__ = ["Float8Tensor", "QuantizedTensor"]


def _make_module_cast_func(dtype):
    """Make module cast function that can handle QuantizedTensor"""
    cast_func_name = {
        torch.float32: "float",
        torch.float16: "half",
        torch.bfloat16: "bfloat16",
    }[dtype]

    def tensor_cast_func(tensor: torch.Tensor) -> torch.Tensor:
        """Cast tensor dtype"""
        if isinstance(tensor, Float8Tensor):
            return Float8Tensor.make_like(
                tensor,
                data=tensor._data,
                fp8_attrs=tensor._fp8_attrs,
                dtype=dtype,
                requires_grad=tensor.requires_grad,
            )
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
