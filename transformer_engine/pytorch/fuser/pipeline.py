# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

from __future__ import annotations

from typing import Any, Optional

import torch

from .ops import FusableOperation, OperationAutogradContext

### TODO Handle no_grad
class _PipelineAutogradFunction(torch.autograd.Function):

    @staticmethod
    def forward(
        func_ctx: Any,
        op_ctxs: list[OperationAutogradContext],
        forward_ops: list[FusableOperation],
        backward_ops: list[FusableOperation],
        input_: torch.Tensor,
    ) -> torch.Tensor:

        # Apply forward ops
        x = input_
        for op, ctx_idxs in forward_ops:
            x = op._pipeline_forward(
                tuple(op_ctxs[idx] for idx in ctx_idxs),
                x,
            )

        # Flatten list of saved tensors
        to_save = []
        for ctx in op_ctxs:
            range_start = len(to_save)
            if ctx.to_save is not None:
                to_save.extend(ctx.to_save)
            range_end = len(to_save)
            ctx.to_save = None
            ctx.saved_tensors_range = (range_start, range_end)
        func_ctx.save_for_backward(*to_save)

        # Other context for backward pass
        func_ctx.backward_ops = backward_ops
        func_ctx.op_ctx = op_ctxs

        return x

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(
        func_ctx: Any,
        grad_output: torch.Tensor,
    ) -> tuple[Optional[torch.Tensor], ...]:

        # Unflatten list of saved tensors
        op_ctxs = func_ctx.op_ctxs
        saved_tensors = func_ctx.saved_tensors
        func_ctx.saved_tensors = None
        for ctx in op_ctxs:
            ctx.saved_tensors = saved_tensors[slice(ctx.saved_tensors_range)]
            ctx.saved_tensors_range = None
        del saved_tensors

        # Apply backward ops
        dx = grad_output
        for op, ctx_idxs in func_ctx.backward_ops:
            dx = op._pipeline_backward(
                tuple(op_ctxs[idx] for idx in ctx_idxs),
                dx,
            )

        return (
            None,  # op_ctxs
            None,  # forward_ops
            None,  # backward_ops
            dx,    # input_
        )


class Pipeline:

    def __init__(self, ops, fuse_ops=True):

        # Unfused ops for forward and backward pass
        self._num_unfused_ops: int = len(ops)
        self._forward_ops: list[tuple[FusableOperation, list[int]]]
        self._backward_ops: list[tuple[FusableOperation, list[int]]]
        self._forward_ops = [(op, (idx,)) for idx, op in enumerate(ops)]
        self._backward_ops = self._forward_ops.copy()
        self._backward_ops.reverse()

        # Fuse ops if needed
        if fuse_ops:
            self.fuse_ops()

    def _fuse_forward_ops(self, ops):
        return ops

    def _fuse_backward_ops(self, ops):
        return ops

    def fuse_ops(self) -> None:
        self._forward_ops = self._fuse_forward_ops(self._forward_ops)
        self._backward_ops = self._fuse_backward_ops(self._backward_ops)

    def __call__(self, input: torch.Tensor) -> torch.Tensor:
        ctxs = [OperationAutogradContext() for _ in range(self._num_unfused_ops)]
        return _PipelineAutogradFunction.apply(
            ctxs,
            self._forward_ops,
            self._backward_ops,
            input,
        )
