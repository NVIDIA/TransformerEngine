# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Fusible operation for dropout."""

from __future__ import annotations
from typing import Optional

import torch
import transformer_engine_torch as tex
from ...tensor._internal.float8_tensor_base import Float8TensorBase
from ...tensor import Quantizer
from transformer_engine.pytorch.ops.op import (
    BasicOperation,
    OperationContext,
)


class Dropout(BasicOperation):
    """Randomly zero out tensor entries during training

    During training, tensor entries are randomly set to zero with
    probability :math:`p` and remaining entries are scaled by
    :math:`1/(1-p)`.

    """

    def __init__(self, p: float, otype: Optional[torch.dtype] = None) -> None:
        super().__init__()
        self.dropout_probability = p
        self.otype = otype

    def op_forward(
        self,
        ctx: OperationContext,
        input_: torch.Tensor,
        prev_op_grad_output_quantizer: Optional[Quantizer],
        next_op_input_quantizer: Optional[Quantizer],
    ) -> torch.Tensor:

        # Compute dropout if training
        is_training = self.training
        out = input_
        mask = None
        if is_training:
            if input_.numel() % 16 != 0:
                assert not isinstance(input_, Float8TensorBase), "fp8 dropout does not support non-16-element aligned inputs"
                keep_prob = 1 - self.dropout_probability
                mask = torch.empty_like(input_)
                mask.bernoulli_(keep_prob)
                mask *= 1 / keep_prob
                out = out * mask
            else:
                if isinstance(input_, Float8TensorBase):
                    assert self.otype is not None, "otype is not set for fp8 dropout"
                    assert self.otype in [torch.half, torch.bfloat16], "fp8 dropout only supports half and bfloat16 output types but got %s" % self.otype
                    out = torch.empty_like(input_, dtype=self.otype)
                    _, mask = tex.dropout_fwd_fp8(input_, out, self.dropout_probability)
                else:
                    out, mask = tex.dropout_fwd(input_, self.dropout_probability, is_training)
        else:
            assert not isinstance(input_, Float8TensorBase), "fp8 dropout does not support non-training mode"
        # Save context for backward
        if ctx.requires_grad:
            ctx.save_for_backward(mask)
            ctx.dropout_probability = self.dropout_probability
            #ctx.is_training = is_training
        return out

    def op_backward(
        self,
        ctx: OperationContext,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[()]]:
        (mask,) = ctx.saved_tensors
        # grad_input = grad_output
        # grad_input = grad_output * mask
        grad_input = tex.dropout_bwd(grad_output, mask, ctx.dropout_probability)
        return grad_input, ()
