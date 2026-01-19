# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Fusible operation for identity."""

from __future__ import annotations
from typing import Optional, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from ..compile_compat.tensor_info import TensorInfo, PseudoForwardResult

from transformer_engine.pytorch.ops.op import (
    BasicOperation,
    OperationContext,
)
from ...tensor import Quantizer


class Identity(BasicOperation):
    """Return input tensor"""

    def op_forward(
        self,
        ctx: OperationContext,
        input_: torch.Tensor,
        prev_op_grad_output_quantizer: Optional[Quantizer],
        next_op_input_quantizer: Optional[Quantizer],
    ) -> torch.Tensor:
        return input_

    def op_backward(
        self,
        ctx: OperationContext,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[()]]:
        return grad_output, ()
