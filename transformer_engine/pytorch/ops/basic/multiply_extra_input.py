# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Fusible operation for multiplying with extra input tensor."""

from __future__ import annotations
from collections.abc import Iterable
from typing import Any, Optional

import torch

from ...tensor import Quantizer
from ..op import BasicOperation, OperationContext
from .._common import maybe_dequantize


def _reduce_broadcast_dims(
    x: torch.Tensor,
    target_shape: Iterable[int],
) -> torch.Tensor:
    """Reduce a tensor down to a target shape.

    The input tensor shape and target shape are assumed to be
    broadcast-compatible. In other words, a tensor with the target
    shape can be broadcast to match the input tensor shape.

    """
    shape = tuple(x.size())
    target_shape = tuple(target_shape)

    # Return immediately if tensor already has correct shape
    if shape == target_shape:
        return x

    # Determine reduction dimensions
    reduce_dims = []
    if len(shape) < len(target_shape):
        raise ValueError(
            f"Invalid target shape (shape={shape} cannot be broadcast to shape={target_shape})."
        )
    if len(shape) > len(target_shape):
        reduce_dims.extend(range(len(shape) - len(target_shape)))
    for idx in range(-len(target_shape), 0):
        if shape[idx] == target_shape[idx]:
            pass
        elif target_shape[idx] != 1:
            raise ValueError(
                f"Invalid target shape (shape={shape} cannot be broadcast to shape={target_shape})."
            )
        else:
            reduce_dims.append(idx)

    # Perform reduction
    return x.sum(reduce_dims).reshape(target_shape)


class MultiplyExtraInput(BasicOperation):
    """Multiply with extra input tensor.

    If the tensor shapes do not match, they will follow NumPy
    broadcasting semantics.

    """

    # Operation expects extra input tensor
    num_extra_inputs: int = 1

    def op_forward(self, *args, **kwargs) -> None:
        raise RuntimeError(
            "{self.__class__.__name__} operation has "
            f"{self.num_extra_inputs} extra tensor inputs "
            f"and {self.num_extra_outputs} extra tensor outputs. "
            "It overrides `fuser_forward` instead of `op_forward`."
        )

    def op_backward(self, *args, **kwargs) -> None:
        raise RuntimeError(
            "{self.__class__.__name__} operation has "
            f"{self.num_extra_inputs} extra tensor inputs "
            f"and {self.num_extra_outputs} extra tensor outputs. "
            "It overrides `fuser_backward` instead of `op_backward`."
        )

    def fuser_forward(
        self,
        basic_op_ctxs: list[OperationContext],
        input_: torch.Tensor,
        *,
        basic_op_extra_inputs: list[tuple[torch.Tensor, ...]],
        prev_op_grad_output_quantizer: Optional[Quantizer],
        next_op_input_quantizer: Optional[Quantizer],
        basic_op_kwargs: list[dict[str, Any]],
    ) -> tuple[torch.Tensor, Iterable[Iterable[torch.Tensor]]]:
        extra_input = basic_op_extra_inputs[0][0]

        # Determine compute dtype
        if torch.is_autocast_enabled():
            dtype = torch.get_autocast_dtype("cuda")
        elif isinstance(input_, torch.Tensor):
            dtype = input_.dtype
        else:
            dtype = extra_input.dtype

        # Perform multiplication
        x1 = maybe_dequantize(input_, dtype)
        x2 = maybe_dequantize(extra_input, dtype)
        output = input_ * extra_input

        # Save state for backward pass
        ctx = basic_op_ctxs[0]
        if ctx.requires_grad:
            ctx.input_shape = x1.size()
            ctx.extra_input_shape = extra_input.size()
            ctx.input_requires_grad = True
            ctx.extra_input_requires_grad = extra_input.requires_grad
            ctx.save_for_backward(
                x1 if ctx.extra_input_requires_grad else None,
                x2 if ctx.input_requires_grad else None,
            )

        return output, [()]

    def fuser_backward(
        self,
        basic_op_ctxs: list[OperationContext],
        grad_output: torch.Tensor,
        *,
        basic_op_grad_extra_outputs: list[tuple[torch.Tensor, ...]],
    ) -> tuple[
        torch.Tensor,
        Iterable[Iterable[Optional[torch.Tensor]]],
        Iterable[Iterable[Optional[torch.Tensor]]],
    ]:
        ctx = basic_op_ctxs[0]
        input_, extra_input = ctx.saved_tensors
        grad_input = None
        if ctx.input_requires_grad:
            grad_input = _reduce_broadcast_dims(
                grad_output * extra_input,
                ctx.input_shape,
            )
        grad_extra_input = None
        if ctx.extra_input_requires_grad:
            grad_extra_input = _reduce_broadcast_dims(
                grad_output * input_,
                ctx.extra_input_shape,
            )
        return grad_input, [()], [(grad_extra_input,)]
