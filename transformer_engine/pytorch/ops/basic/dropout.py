# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Fusible operation for dropout."""

from __future__ import annotations
from typing import Optional

import torch

from transformer_engine.pytorch.ops.op import (
    BasicOperation,
    OperationContext,
)
from ...tensor import Quantizer


class Dropout(BasicOperation):
    """Randomly zero out tensor entries during training

    During training, tensor entries are randomly set to zero with
    probability :math:`p` and remaining entries are scaled by
    :math:`1/(1-p)`.

    """

    def __init__(self, p: float) -> None:
        super().__init__()
        self.dropout_probability = p

    def op_forward(
        self,
        ctx: OperationContext,
        input_: torch.Tensor,
        prev_op_grad_output_quantizer: Optional[Quantizer],
        next_op_input_quantizer: Optional[Quantizer],
    ) -> torch.Tensor:

        # Compute dropout if training
        out = input_
        is_training = self.training
        mask = None
        if is_training:
            keep_prob = 1 - self.dropout_probability
            mask = torch.empty_like(input_)
            mask.bernoulli_(keep_prob)
            mask *= 1 / keep_prob
            out = out * mask

        # Save context for backward
        if ctx.requires_grad:
            ctx.save_for_backward(mask)
            ctx.is_training = is_training

        return out

    def op_backward(
        self,
        ctx: OperationContext,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[()]]:
        (mask,) = ctx.saved_tensors
        grad_input = grad_output
        if ctx.is_training:
            grad_input = grad_input * mask
        return grad_input, ()
