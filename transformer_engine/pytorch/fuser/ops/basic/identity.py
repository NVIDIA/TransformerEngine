# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

from __future__ import annotations

import torch

from transformer_engine.pytorch.fuser.ops.op import BasicOperation


class Identity(BasicOperation):
    """Return input tensor"""

    def __init__(self) -> None:
        super().__init__()

    def op_forward(
        self,
        ctx: OperationContext,
        input: torch.Tensor,
    ) -> torch.Tensor:
        return input

    def op_backward(
        self,
        ctx: OperationContext,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[()]]:
        return grad_output, ()
