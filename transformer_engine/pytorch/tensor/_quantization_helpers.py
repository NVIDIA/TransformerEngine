# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Private helper functions and classes for quantized tensor implementations.

This module contains internal autograd functions and utilities that support
the quantization machinery.
"""

from __future__ import annotations
from typing import Callable, Optional, Tuple, Any, Dict, TYPE_CHECKING
import torch

if TYPE_CHECKING:
    from transformer_engine.pytorch.quantized_tensor import QuantizedTensor


class _QuantizeFunc(torch.autograd.Function):
    """Quantize tensor"""

    @staticmethod
    def forward(
        _ctx: Optional[torch.autograd.function.FunctionCtx],  # unused
        tensor: torch.Tensor,
        quantize_impl: Callable,
    ) -> QuantizedTensor:
        # pylint: disable=missing-function-docstring
        return quantize_impl(tensor)

    @staticmethod
    def backward(
        _ctx: torch.autograd.function.FunctionCtx,  # unused
        grad: torch.Tensor,
    ) -> Tuple[Optional[torch.Tensor], ...]:
        # pylint: disable=missing-function-docstring
        # Assume that we want gradients in full precision
        return grad, None


class _IdentityFunc(torch.autograd.Function):
    """Identity function

    If constructor keyword-arguments are provided, then construct a
    new Float8Tensor using the provided tensor's attributes.

    """

    @staticmethod
    def forward(
        ctx, tensor: QuantizedTensor, init_kwargs: Optional[Dict[str, Any]] = None
    ) -> QuantizedTensor:
        # pylint: disable=missing-function-docstring

        # Return input tensor if constructor kwargs are not provided
        if init_kwargs is None:
            return tensor.detach()

        # Construct new tensor if constructor kwargs are provided
        ctx.input_dtype = tensor.dtype
        kwargs = tensor.get_metadata()
        for key, val in init_kwargs.items():
            kwargs[key] = val
        return type(tensor)(tensor.shape, tensor.dtype, **kwargs)

    @staticmethod
    def backward(ctx, grad_output):
        # pylint: disable=missing-function-docstring
        grad_input = grad_output
        if grad_input.dtype == ctx.input_dtype:
            grad_input = grad_input.detach()
        else:
            grad_input = grad_input.to(ctx.input_dtype)
        return grad_input, None


def _stride_from_shape(shape: list[int]):
    """Calculate stride from shape for contiguous tensors"""
    if len(shape) == 0:
        return []
    rstride = [1]
    for d in reversed(shape[1:]):
        rstride.append(rstride[-1] * d)
    return list(reversed(rstride))
