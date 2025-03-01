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


class MakeExtraOutput(BasicOperation):
    """Make extra output in operation fuser

    If this operation is included in the operation fuser, then the
    operation fuser will return the intermediate tensor as an extra
    tensor output. In the backward pass, the gradient is directly
    accumulated into the gradient w.r.t. the extra output.

    This operation is considered an advanced feature and most users
    are discouraged from using it. In-place operations break some
    autograd assumptions and they can result in subtle, esoteric bugs.

    Compare to `AddInPlace`, which does a similar operation in the
    backward pass.

    """

    # Operation expects buffer for output tensor
    num_extra_outputs: int = 1

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
        basic_op_prev_ops: list[Optional[BasicOperation]],
        basic_op_next_ops: list[Optional[BasicOperation]],
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
        grad_input = basic_op_grad_extra_outputs[0][0]
        grad_input += grad_output
        return grad_input, [], [()]
