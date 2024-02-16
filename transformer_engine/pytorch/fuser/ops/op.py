# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class OperationContext:

    to_save: Optional[tuple[Optional[torch.Tensor], ...]] = None
    saved_tensors: Optional[tuple[Optional[torch.Tensor], ...]] = None
    saved_tensors_range: Optional[tuple[int, int]] = None

    def save_for_backward(self, *tensors):
        self.to_save = tensors


class FusableOperation(torch.nn.Module):

    def __init__(self, *unfused_ops: FusableOperation) -> None:
        super().__init__()
        self._unfused_ops = unfused_ops

    @property
    def is_fused_op(self):
        return len(self._unfused_ops) > 0

    def unfused_op_forward(
        self,
        ctx: OperationContext,
        input: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        raise NotImplementedError(
            "Forward pass is not implemented for unfused operation"
        )

    def fused_op_forward(
        self,
        unfused_op_ctxs: list[OperationContext],
        input: torch.Tensor,
        unfused_op_kwargs: list[dict[str, Any]],
    ) -> torch.Tensor:
        raise NotImplementedError(
            "Forward pass is not implemented for fused operation"
        )

    def unfused_op_backward(
        self,
        ctx: OperationContext,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor, Iterable[Optional[torch.Tensor]]]:
        raise NotImplementedError(
            "Backward pass is not implemented for unfused operation"
        )

    def fused_op_backward(
        self,
        unfused_op_ctxs: list[OperationContext],
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor, Iterable[Iterable[Optional[torch.Tensor]]]]:
        raise NotImplementedError(
            "Backward pass is not implemented for fused operation"
        )

    def _pipeline_forward(
        self,
        unfused_op_ctxs: list[OperationContext],
        input_: torch.Tensor,
        unfused_op_kwargs: list[dict[str, Any]],
    ) -> torch.Tensor:
        if self.is_fused_op:
            return self.fused_op_forward(
                unfused_op_ctxs,
                input_,
                unfused_op_kwargs,
            )
        else:
            return self.unfused_op_forward(
                unfused_op_ctxs[0],
                input_,
                **unfused_op_kwargs[0],
            )

    def _pipeline_backward(
        self,
        unfused_op_ctxs: list[OperationContext],
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor, Iterable[Iterable[Optional[torch.Tensor]]]]:
        if self.is_fused_op:
            return self.fused_op_backward(unfused_op_ctxs, grad_output)
        else:
            grad_input, grad_params = self.unfused_op_backward(
                unfused_op_ctxs[0],
                grad_output,
            )
            return grad_input, [grad_params]

    def forward(self, input: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        from ..pipeline import Pipeline
        return Pipeline([self], fuse_ops=False)(input, [kwargs])
