# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Fusible operation for reshape."""

from __future__ import annotations
from collections.abc import Iterable
from typing import Optional

import torch

from transformer_engine.pytorch.ops.op import (
    BasicOperation,
    OperationContext,
)
from ...tensor import Quantizer


class Reshape(BasicOperation):
    """Reshape tensor

    See ``torch.reshape``.

    Parameters
    ----------
    shape : iterable of int
        Output tensor dimensions. If one dimension is -1, it is
        inferred based on input tensor dimensions.

    """

    def __init__(self, shape: Iterable[int]) -> None:
        super().__init__()
        self._shape = tuple(shape)

    def op_forward_compute(
        self,
        input_: torch.Tensor,
        *,
        requires_grad: bool,
        prev_op_grad_output_quantizer: Optional[Quantizer] = None,
        next_op_input_quantizer: Optional[Quantizer] = None,
    ) -> tuple[torch.Tensor, tuple[()]]:
        return input_.reshape(*self._shape), ()

    def op_forward_save_ctx(
        self,
        ctx: OperationContext,
        input_: torch.Tensor,
        tensors_to_save: tuple[()],
        *,
        requires_grad: bool,
        prev_op_grad_output_quantizer: Optional[Quantizer] = None,
        next_op_input_quantizer: Optional[Quantizer] = None,
    ) -> None:
        if requires_grad:
            ctx.input_shape = input_.size()

    def op_backward(
        self,
        ctx: OperationContext,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[()]]:
        return grad_output.reshape(*ctx.input_shape), ()
