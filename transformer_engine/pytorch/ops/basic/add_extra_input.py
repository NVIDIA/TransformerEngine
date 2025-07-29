# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Fusible operation for adding extra input tensor."""

from __future__ import annotations
from collections.abc import Iterable
from typing import Any, Optional

import torch

from transformer_engine.pytorch.ops.op import (
    BasicOperation,
    OperationContext,
)

from transformer_engine.pytorch.tensor import Quantizer


class AddExtraInput(BasicOperation):
    """Add extra input tensor

    This operation requires an extra tensor input to the operation
    user. It returns the sum of the main input and the extra input.
    If in_place=True, the main input is added in-place to the extra
    input, and a view of the extra input is output.

    Using this operation with in_place=True is considered an advanced
    feature and most users are discouraged from it. In-place operations
    break some autograd assumptions and they can result in subtle, esoteric bugs.

    Compare to `MakeExtraOutput`, which does a similar operation in
    the backward pass.

    """

    # Operation expects buffer for output tensor
    num_extra_inputs: int = 1

    def __init__(self, *, in_place: bool = False):
        super().__init__()
        self._in_place = in_place

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
        if self._in_place:
            extra_input = extra_input.detach()
            extra_input += input_
            output = extra_input
        else:
            output = extra_input + input_
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
        return grad_output, [()], [(grad_output,)]
