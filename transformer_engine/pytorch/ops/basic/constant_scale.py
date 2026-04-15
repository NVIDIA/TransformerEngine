# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Fusible operation for constant scaling."""

from __future__ import annotations
from typing import Optional

import torch

from transformer_engine.pytorch.ops.op import (
    BasicOperation,
    OperationContext,
)
from ...tensor import Quantizer


class ConstantScale(BasicOperation):
    """Multiply by a constant"""

    def __init__(self, scale: float) -> None:
        super().__init__()
        self.scale = scale

    def op_forward_compute(
        self,
        input_: torch.Tensor,
        *,
        requires_grad: bool,
        prev_op_grad_output_quantizer: Optional[Quantizer] = None,
        next_op_input_quantizer: Optional[Quantizer] = None,
    ) -> tuple[torch.Tensor, tuple[()]]:
        return input_ * self.scale, ()

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
        pass

    def op_backward(
        self,
        ctx: OperationContext,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[()]]:
        return grad_output * self.scale, ()
