# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

from __future__ import annotations

from typing import Any, Optional

import torch

from transformer_engine.pytorch.fp8 import FP8GlobalStateManager
from transformer_engine.pytorch.fuser.ops.op import (
    BasicOperation,
    FusableOperation,
    OperationContext,
)
from transformer_engine.pytorch.fuser.ops.fused_forward import (
    fuse_forward_linear_bias_activation,
)
from transformer_engine.pytorch.graph import is_graph_capturing
from transformer_engine.pytorch.utils import clear_tensor_data

class _OperationFuserAutogradFunction(torch.autograd.Function):
    """Autograd function for a pipeline of operations

    Autograd must be done at the pipeline level since we may apply
    different fusions in the forward and backward passes.

    """

    @staticmethod
    def forward(
        func_ctx: torch.autograd.function.FunctionCtx,
        input_: torch.Tensor,
        forward_ops: list[tuple[FusableOperation, list[int]]],
        backward_ops: list[tuple[FusableOperation, list[int]]],
        basic_ops: list[BasicOperation],
        basic_op_ctxs: list[OperationContext],
        basic_op_kwargs: list[dict[str, Any]],
        *params: torch.nn.Parameter,
    ) -> torch.Tensor:
        """Forward pass

        Parameters
        ----------
        func_ctx: torch.autograd.function.FunctionCtx
            Context for PyTorch autograd function
        input_: torch.Tensor
            Input to first operation in pipeline
        forward_ops: list of tuple
            Forward pass operations and the indices of the
            corresponding basic operations. The order should match
            basic_ops.
        backward_ops: list of tuple
            Backward pass operations and the indices of the
            corresponding basic operations. The order should be the
            reverse of basic_ops.
        basic_ops: list of BasicOperation
            Basic operations
        basic_op_ctxs: list of OperationContext
            Context for BasicOperation
        basic_op_kwargs: list of dict
            Keyword arguments to BasicOperation
        *params: torch.nn.Parameter
            Parameters in operation pipeline

        """

        # Apply forward ops
        x = input_
        requires_grad = x.requires_grad
        for op, basic_op_idxs in forward_ops:

            # Forward op
            x = op.fuser_forward(
                [basic_op_ctxs[idx] for idx in basic_op_idxs],
                x,
                [basic_op_kwargs[idx] for idx in basic_op_idxs],
            )

            # Check if backward op is required
            if not requires_grad:
                requires_grad = any(
                    param.requires_grad for param in op.parameters()
                )
            for idx in basic_op_idxs:
                basic_op_ctxs[idx]._requires_grad = requires_grad
            x.requires_grad_(requires_grad=requires_grad)

        # Flatten list of saved tensors
        to_save = []
        for ctx in basic_op_ctxs:
            range_start = len(to_save)
            if ctx.to_save is not None:
                to_save.extend(ctx.to_save)
            range_end = len(to_save)
            ctx._to_save = None
            ctx._saved_tensors_range = (range_start, range_end)
        func_ctx.save_for_backward(*to_save)

        # Other context for backward pass
        func_ctx.backward_ops = backward_ops
        func_ctx.basic_ops = basic_ops
        func_ctx.basic_op_ctxs = basic_op_ctxs
        func_ctx.is_first_module = FP8GlobalStateManager.is_first_fp8_module()

        return x

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(
        func_ctx: Any,
        grad_output: torch.Tensor,
    ) -> tuple[Optional[torch.Tensor], ...]:
        """Backward pass"""

        # Operations and autograd state
        backward_ops = func_ctx.backward_ops
        basic_ops = func_ctx.basic_ops
        basic_op_ctxs = func_ctx.basic_op_ctxs

        # Unflatten list of saved tensors
        saved_tensors = func_ctx.saved_tensors
        for ctx in basic_op_ctxs:
            ctx.saved_tensors = saved_tensors[slice(*ctx._saved_tensors_range)]
            ctx._saved_tensors_range = None

        # Apply backward ops
        dx = grad_output
        grad_params = [None for _ in range(len(basic_ops))]
        for op, basic_op_idxs in backward_ops:

            # Stop if no more gradients are required
            if all(
                not basic_op_ctxs[idx]._requires_grad
                for idx in basic_op_idxs
            ):
                dx = None
                break

            # Backward op
            dx, fused_op_dparams = op.fuser_backward(
                [basic_op_ctxs[idx] for idx in basic_op_idxs],
                dx,
            )
            for idx, basic_op_dparams in zip(basic_op_idxs, fused_op_dparams):
                grad_params[idx] = basic_op_dparams
                clear_tensor_data(*basic_op_ctxs[idx].saved_tensors)

        # Flatten list of parameter gradients
        grad_params_flat = []
        for idx, dparams in enumerate(grad_params):
            params = list(basic_ops[idx].parameters())
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

        # Update FP8 scaling factors
        if func_ctx.is_first_module and not is_graph_capturing():
            FP8GlobalStateManager.reduce_and_update_fp8_tensors(forward=False)

        return (
            dx,    # input_
            None,  # forward_ops
            None,  # backward_ops
            None,  # basic_ops
            None,  # basic_op_ctxs
            None,  # basic_op_kwargs
            *grad_params_flat,  # params
        )

class OperationFuser:
    """Manages forward and backward passes for a pipeline of operations

    Parameters
    ----------
    ops: list of FusableOperation
        Pipeline of operations
    fuse_ops: bool, default = `True`
        Whether to attempt fusing operations

    """

    def __init__(
        self,
        ops: list[FusableOperation],
        fuse_ops: bool = True,
    ):

        # Get list of basic operations
        basic_ops = []
        for op in ops:
            if op.is_fused_op:
                basic_ops.extend(op.basic_ops)
            else:
                basic_ops.append(op)
        self._num_basic_ops: int = len(basic_ops)
        self._basic_ops: list[BasicOperation] = basic_ops

        # Ops for forward and backward pass
        self._forward_ops: list[tuple[FusableOperation, list[int]]]
        self._backward_ops: list[tuple[FusableOperation, list[int]]]
        self._forward_ops = [
            (op, (idx,))
            for idx, op in enumerate(self._basic_ops)
        ]
        self._backward_ops = list(reversed(self._forward_ops))

        # Fuse ops if needed
        if fuse_ops:
            self.fuse_ops()

    def _fuse_forward_ops(self, ops):
        """Attempt to fuse operations in forward pass"""
        ops = fuse_forward_linear_bias_activation(ops)
        return ops

    def _fuse_backward_ops(self, ops):
        """Attempt to fuse operations in backward pass"""
        return ops

    def fuse_ops(self) -> None:
        """Attempt to fuse operations"""
        self._forward_ops = self._fuse_forward_ops(self._forward_ops)
        self._backward_ops = self._fuse_backward_ops(self._backward_ops)

    def __call__(
        self,
        input: torch.Tensor,
        basic_op_kwargs: Optional[list[dict[str, Any]]] = None,
    ) -> torch.Tensor:

        # Initialization before forward pass
        for op in self._basic_ops:
            op.pre_forward()

        # Construct autograd contexts
        num_basic_ops = len(self._basic_ops)
        basic_op_ctxs = []
        for idx, op in enumerate(self._basic_ops):
            next_op, prev_op = None, None
            if idx < num_basic_ops - 1:
                next_op = self._basic_ops[idx+1]
            if idx > 0:
                prev_op = self._basic_ops[idx-1]
            ctx = OperationContext(
                op=op,
                next_op=next_op,
                prev_op=prev_op,
            )
            basic_op_ctxs.append(ctx)

        # Canonicalize op kwargs
        if basic_op_kwargs is None:
            basic_op_kwargs = [dict() for _ in range(num_basic_ops)]

        # Flatten list of parameters
        params = []
        for op in self._basic_ops:
            params.extend(op.parameters())

        # Fuser forward pass
        return _OperationFuserAutogradFunction.apply(
            input,
            self._forward_ops,
            self._backward_ops,
            self._basic_ops,
            basic_op_ctxs,
            basic_op_kwargs,
            *params,
        )
