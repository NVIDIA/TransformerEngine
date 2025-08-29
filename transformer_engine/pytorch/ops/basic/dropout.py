# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Fusible operation for dropout."""

from __future__ import annotations
from typing import Optional

import torch
import transformer_engine_torch as tex
from ...tensor import Quantizer
from ...tensor._internal.float8_tensor_base import Float8TensorBase
from .._common import maybe_autocast_dtype, maybe_dequantize
from ..op import BasicOperation, OperationContext


class Dropout(BasicOperation):
    """Randomly zero out tensor entries during training

    During training, tensor entries are randomly set to zero with
    probability :math:`p` and remaining entries are scaled by
    :math:`1/(1-p)`.

    """

    def __init__(self, p: float) -> None:
        super().__init__()
        self.dropout_probability: float = p

    def op_forward(
        self,
        ctx: OperationContext,
        input_: torch.Tensor,
        prev_op_grad_output_quantizer: Optional[Quantizer],
        next_op_input_quantizer: Optional[Quantizer],
    ) -> torch.Tensor:

        # Output dtype
        dtype = maybe_autocast_dtype(default_dtype=input_.dtype)

        # Choose implementation
        impl = None
        if not self.training:
            impl = "evaluation"
        elif input_.numel() % 16 == 0 and dtype in (torch.float16, torch.bfloat16):
            impl = "fused"
        else:
            impl = "unfused"

        # Perform dropout
        out: torch.Tensor
        mask: Optional[torch.Tensor] = None
        if impl == "evaluation":
            out = input_
        elif impl == "fused":
            x = input_
            if not isinstance(x, Float8TensorBase):
                x = maybe_dequantize(x, dtype=dtype)
            out, mask = tex.dropout_fwd(x, self.dropout_probability)
        elif impl == "unfused":
            x = maybe_dequantize(input_, dtype=dtype)
            keep_prob = 1 - self.dropout_probability
            mask = torch.empty_like(x)
            mask.bernoulli_(keep_prob)
            mask *= 1 / keep_prob
            out = x * mask
        else:
            raise ValueError(f"Unsupported forward implementation {impl}")

        # Save context for backward
        if ctx.requires_grad:
            ctx.save_for_backward(mask)
            ctx.impl = impl
            ctx.dropout_probability = self.dropout_probability
            ctx.dtype = dtype

        return out

    def op_backward(
        self,
        ctx: OperationContext,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[()]]:

        # Saved tensors from forward pass
        (mask,) = ctx.saved_tensors

        # Perform dropout backward pass
        grad_input: torch.Tensor
        if ctx.impl == "evaluation":
            grad_input = grad_output
        elif ctx.impl == "fused":
            dy = maybe_dequantize(grad_output, dtype=ctx.dtype)
            grad_input = tex.dropout_bwd(dy, mask, ctx.dropout_probability)
        elif ctx.impl == "unfused":
            dy = maybe_dequantize(grad_output, dtype=ctx.dtype)
            grad_input = dy * mask
        else:
            raise ValueError(f"Unsupported backward implementation {ctx.impl}")

        return grad_input, ()
