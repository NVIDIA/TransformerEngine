# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

from __future__ import annotations

from typing import Any, Optional

import torch

from transformer_engine.pytorch.fuser.ops.op import (
    FusableOperation,
    OperationContext,
    UnfusedOperation,
)
from transformer_engine.pytorch.fuser.ops.fused_forward import (
    fuse_forward_linear_bias_activation,
)
from transformer_engine.pytorch.utils import clear_tensor_data

### TODO Handle no_grad
class _PipelineAutogradFunction(torch.autograd.Function):

    @staticmethod
    def forward(
        func_ctx: Any,
        input_: torch.Tensor,
        forward_ops: list[FusableOperation],
        backward_ops: list[FusableOperation],
        unfused_ops: list[UnfusedOperation],
        unfused_op_ctxs: list[OperationContext],
        unfused_op_kwargs: list[dict[str, Any]],
        *params: torch.nn.Parameter,
    ) -> torch.Tensor:

        # Apply forward ops
        x = input_
        for op, unfused_op_idxs in forward_ops:
            x = op.pipeline_forward(
                [unfused_op_ctxs[idx] for idx in unfused_op_idxs],
                x,
                [unfused_op_kwargs[idx] for idx in unfused_op_idxs],
            )

        # Flatten list of saved tensors
        to_save = []
        for ctx in unfused_op_ctxs:
            range_start = len(to_save)
            if ctx.to_save is not None:
                to_save.extend(ctx.to_save)
            range_end = len(to_save)
            ctx.to_save = None
            ctx.saved_tensors_range = (range_start, range_end)
        func_ctx.save_for_backward(*to_save)

        # Other context for backward pass
        func_ctx.backward_ops = backward_ops
        func_ctx.unfused_ops = unfused_ops
        func_ctx.unfused_op_ctxs = unfused_op_ctxs

        return x

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(
        func_ctx: Any,
        grad_output: torch.Tensor,
    ) -> tuple[Optional[torch.Tensor], ...]:

        # Operations and autograd state
        backward_ops = func_ctx.backward_ops
        unfused_ops = func_ctx.unfused_ops
        unfused_op_ctxs = func_ctx.unfused_op_ctxs

        # Unflatten list of saved tensors
        saved_tensors = func_ctx.saved_tensors
        for ctx in unfused_op_ctxs:
            ctx.saved_tensors = saved_tensors[slice(*ctx.saved_tensors_range)]
            ctx.saved_tensors_range = None

        # Apply backward ops
        dx = grad_output
        grad_params = [None for _ in range(len(unfused_ops))]
        for op, unfused_op_idxs in backward_ops:
            dx, fused_op_dparams = op.pipeline_backward(
                [unfused_op_ctxs[idx] for idx in unfused_op_idxs],
                dx,
            )
            for idx, unfused_op_dparams in zip(unfused_op_idxs, fused_op_dparams):
                grad_params[idx] = unfused_op_dparams
                clear_tensor_data(*unfused_op_ctxs[idx].saved_tensors)

        # Flatten list of parameter gradients
        grad_params_flat = []
        for idx, dparams in enumerate(grad_params):
            params = list(unfused_ops[idx].parameters(recurse=False))
            if dparams is None:
                dparams = [None for _ in range(len(params))]
            else:
                dparams = list(dparams)
            if len(dparams) != len(params):
                raise RuntimeError(
                    f"Expected op {idx} to generate {len(params)} param grads, "
                    f"but got {len(dparams)}"
                )
            grad_params_flat.extend(dparams)

        return (
            dx,    # input_
            None,  # forward_ops
            None,  # backward_ops
            None,  # unfused_ops
            None,  # unfused_op_ctxs
            None,  # unfused_op_kwargs
            *grad_params_flat,  # params
        )

class Pipeline:

    def __init__(
        self,
        ops: list[FusableOperation],
        fuse_ops: bool = True,
    ):

        # Get list of unfused operations
        unfused_ops = []
        for op in ops:
            if op.is_fused_op:
                unfused_ops.extend(op.unfused_ops)
            else:
                unfused_ops.append(op)
        self._num_unfused_ops: int = len(unfused_ops)
        self._unfused_ops: list[UnfusedOperation] = unfused_ops

        # Ops for forward and backward pass
        self._forward_ops: list[tuple[FusableOperation, list[int]]]
        self._backward_ops: list[tuple[FusableOperation, list[int]]]
        self._forward_ops = [
            (op, (idx,))
            for idx, op in enumerate(self._unfused_ops)
        ]
        self._backward_ops = list(reversed(self._forward_ops))

        # Fuse ops if needed
        if fuse_ops:
            self.fuse_ops()

    def _fuse_forward_ops(self, ops):
        ops = fuse_forward_linear_bias_activation(ops)
        return ops

    def _fuse_backward_ops(self, ops):
        return ops

    def fuse_ops(self) -> None:
        self._forward_ops = self._fuse_forward_ops(self._forward_ops)
        self._backward_ops = self._fuse_backward_ops(self._backward_ops)

    def __call__(
        self,
        input: torch.Tensor,
        unfused_op_kwargs: Optional[list[dict[str, Any]]] = None,
    ) -> torch.Tensor:

        # Initialization before forward pass
        for op in self._unfused_ops:
            op.pre_forward()

        # Construct autograd contexts
        num_unfused_ops = len(self._unfused_ops)
        unfused_op_ctxs = [OperationContext() for _ in range(num_unfused_ops)]

        # Canonicalize op kwargs
        if unfused_op_kwargs is None:
            unfused_op_kwargs = [dict() for _ in range(num_unfused_ops)]

        # Flatten list of parameters
        params = []
        for op in self._unfused_ops:
            params.extend(op.parameters(recurse=False))

        # Pipeline forward pass
        return _PipelineAutogradFunction.apply(
            input,
            self._forward_ops,
            self._backward_ops,
            self._unfused_ops,
            unfused_op_ctxs,
            unfused_op_kwargs,
            *params,
        )
