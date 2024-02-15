# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class OperationAutogradContext:

    to_save: Optional[tuple[Optional[torch.Tensor], ...]] = None
    saved_tensors: Optional[tuple[Optional[torch.Tensor], ...]] = None
    saved_tensors_range: Optional[tuple[int, int]] = None

    def save_for_backward(self, *tensors):
        self.to_save = tensors


class FusableOperation(torch.nn.Module):

    def __init__(self, *unfused_ops: FusableOperation) -> None:
        self._unfused_ops = unfused_ops

    @property
    def is_fused_op(self):
        return len(self._unfused_ops) > 0

    def unfused_op_forward(
        self,
        ctx: OperationAutogradContext,
        input: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError(
            "Forward pass is not implemented for unfused operation"
        )

    def fused_op_forward(
        self,
        ctx: tuple[OperationAutogradContext, ...],
        input: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError(
            "Forward pass is not implemented for fused operation"
        )

    def unfused_op_backward(
        self,
        ctx: OperationAutogradContext,
        grad_output: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError(
            "Backward pass is not implemented for unfused operation"
        )

    def fused_op_backward(
        self,
        ctx: tuple[OperationAutogradContext, ...],
        grad_output: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError(
            "Backward pass is not implemented for fused operation"
        )

    def _pipeline_forward(
        self,
        ctxs: tuple[OperationAutogradContext, ...],
        input_: torch.Tensor,
    ) -> torch.Tensor:
        if self.is_fused_op:
            return self.fused_op_forward(ctxs, input_)
        else:
            return self.unfused_op_forward(ctxs[0], input_)

    def _pipeline_backward(
        self,
        ctxs: tuple[OperationAutogradContext, ...],
        grad_output: torch.Tensor,
    ) -> torch.Tensor:
        if self.is_fused_op:
            return self.fused_op_backward(ctxs, grad_output)
        else:
            return self.unfused_op_backward(ctxs[0], grad_output)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        from ..pipeline import Pipeline
        return Pipeline([self], fuse_ops=False)(input)
