# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Make extra tensor output in operation fuser."""

from __future__ import annotations
from collections.abc import Iterable
from typing import Any, Optional

import torch

from transformer_engine.pytorch.ops.op import (
    BasicOperation,
    OperationContext,
)
from ...tensor import Quantizer


class MakeExtraOutput(BasicOperation):
    """Make extra output in operation fuser

    If this operation is included in the operation fuser, then the
    operation fuser will return the intermediate tensor as an extra
    tensor output.

    In the backward pass, the gradient may be directly
    accumulated into the gradient w.r.t. the extra output. This is
    controlled by the in_place kwarg. Currently, the BackwardLinearAdd
    fusion is able to happen only with in_place=True.

    Using this operation with in_place=True is
    considered an advanced feature. Most users are discouraged
    from enabling it in-place gradient accumulation, as in-place
    operations break some autograd assumptions and they can result
    in subtle, esoteric bugs.

    Compare to `AddExtraInput`, which does a similar operation in the
    backward pass.

    """

    # Operation expects buffer for output tensor
    num_extra_outputs: int = 1

    def __init__(self, *, in_place: bool = False):
        super().__init__()
        self._in_place: bool = in_place

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
        return input_, [(input_,)]

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
        grad_extra_output = basic_op_grad_extra_outputs[0][0]
        if self._in_place:
            grad_extra_output += grad_output
            grad_input = grad_extra_output
        else:
            grad_input = grad_extra_output + grad_output
        return grad_input, [()], [()]
