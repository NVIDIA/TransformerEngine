# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Manager class for a pipeline of fusible operations."""

from __future__ import annotations
from collections.abc import Callable
from typing import Any, Optional

import torch

from transformer_engine.pytorch.fp8 import FP8GlobalStateManager
from transformer_engine.pytorch.ops.op import (
    BasicOperation,
    FusibleOperation,
    OperationContext,
)
from transformer_engine.pytorch.ops.fused import (
    fuse_backward_linear_add,
    fuse_forward_linear_bias_activation,
    fuse_forward_linear_bias_add,
    fuse_userbuffers_backward_linear,
    fuse_userbuffers_forward_linear,
)


def _split_tuple(t: tuple, idx: int) -> tuple[tuple, tuple]:
    """Split tuple at index"""
    return t[:idx], t[idx:]


# Lazily imported function used in _is_graph_capturing
_is_graph_capturing_function: Optional[Callable[[], bool]] = None


def _is_graph_capturing() -> bool:
    """Whether function is called within `make_graphed_callables`

    Avoid circular import with lazy import.

    """
    global _is_graph_capturing_function
    if _is_graph_capturing_function is None:
        from ..graph import is_graph_capturing

        _is_graph_capturing_function = is_graph_capturing
    return _is_graph_capturing_function()


class _OperationFuserAutogradFunction(torch.autograd.Function):
    """Autograd function for a pipeline of operations

    Autograd must be done at the pipeline level since we may apply
    different fusions in the forward and backward passes.

    """

    # pylint: disable=unused-argument
    @staticmethod
    def forward(
        func_ctx: Optional[torch.autograd.function.FunctionCtx],
        input_: torch.Tensor,
        forward_ops: list[tuple[FusibleOperation, list[int]]],
        backward_ops: list[tuple[FusibleOperation, list[int]]],
        basic_ops: list[BasicOperation],
        basic_op_kwargs: list[dict[str, Any]],
        is_grad_enabled: bool,
        num_params: int,
        num_extra_inputs: int,
        *params_and_extra_inputs: torch.nn.Parameter,
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
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
        basic_op_kwargs: list of dict
            Keyword arguments to BasicOperation
        num_params: int
            Number of parameter tensors to include in autograd graph.
        *params_and_extra_inputs: torch.Tensor
            Other tensor inputs to include in autograd graph. Consists
            of parameter tensors, followed by extra operation inputs.

        Returns
        -------
        Output tensor(s). If none of the operations have any extra
        tensor outputs, then the pipeline's output tensor is returned.
        Otherwise, a tuple with the pipeline's output tensor and extra
        tensor outputs is returned.

        """

        # Operation autograd contexts
        basic_op_ctxs = [OperationContext() for _ in range(len(basic_ops))]

        # Unflatten list of parameters and extra tensor inputs
        if len(params_and_extra_inputs) != num_params + num_extra_inputs:
            raise ValueError(
                f"Expected {num_params + num_extra_inputs} extra tensor arguments "
                f"({num_params} parameters, {num_extra_inputs} extra inputs), "
                f"but got {len(params_and_extra_inputs)}"
            )
        _, extra_inputs = _split_tuple(params_and_extra_inputs, num_params)
        basic_op_extra_inputs = []
        for op in basic_ops:
            xs, extra_inputs = _split_tuple(extra_inputs, op.num_extra_inputs)
            basic_op_extra_inputs.append(xs)

        # Apply forward ops
        x = input_
        requires_grad = is_grad_enabled and x.requires_grad
        extra_outputs = [None for _ in range(len(basic_ops))]
        for op, basic_op_idxs in forward_ops:

            # Check if backward op is required
            if is_grad_enabled:
                if not requires_grad:
                    requires_grad = any(param.requires_grad for param in op.parameters())
                if not requires_grad:
                    requires_grad = any(any(x.requires_grad for x in xs) for xs in extra_inputs)
            for idx in basic_op_idxs:
                basic_op_ctxs[idx].requires_grad = requires_grad
            if requires_grad != x.requires_grad:
                if requires_grad:
                    x.requires_grad_()
                else:
                    x = x.detach()

            # Forward op
            extra_inputs = [basic_op_extra_inputs[idx] for idx in basic_op_idxs]
            prev_ops = [basic_ops[idx - 1] if idx > 0 else None for idx in basic_op_idxs]
            next_ops = [
                basic_ops[idx + 1] if (idx < len(basic_ops) - 1) else None for idx in basic_op_idxs
            ]
            x, fused_op_extra_outputs = op.fuser_forward(
                [basic_op_ctxs[idx] for idx in basic_op_idxs],
                x,
                basic_op_extra_inputs=extra_inputs,
                basic_op_prev_ops=prev_ops,
                basic_op_next_ops=next_ops,
                basic_op_kwargs=[basic_op_kwargs[idx] for idx in basic_op_idxs],
            )
            x.requires_grad_(requires_grad=requires_grad)
            for idx, ys in zip(basic_op_idxs, fused_op_extra_outputs):
                for y in ys:
                    y.requires_grad_(requires_grad=requires_grad)
                extra_outputs[idx] = ys

        # Flatten list of extra outputs
        extra_outputs_flat = []
        for idx, ys in enumerate(extra_outputs):
            ys = list(ys)
            num_extra_outputs = basic_ops[idx].num_extra_outputs
            if len(ys) != num_extra_outputs:
                raise RuntimeError(
                    f"Expected op {idx} to generate "
                    "{num_extra_outputs} extra inputs, "
                    f"but got {len(ys)}"
                )
            extra_outputs_flat.extend(ys)

        # Save context for backward pass
        if is_grad_enabled:

            # Flatten list of saved tensors
            to_save = []
            for ctx in basic_op_ctxs:
                range_start = len(to_save)
                if ctx.to_save is not None:
                    to_save.extend(ctx.to_save)
                range_end = len(to_save)
                ctx.to_save = None
                ctx._saved_tensors_range = (range_start, range_end)
            func_ctx.save_for_backward(*to_save)

            # Other context
            func_ctx.backward_ops = backward_ops
            func_ctx.basic_ops = basic_ops
            func_ctx.basic_op_ctxs = basic_op_ctxs
            func_ctx.basic_op_num_params = [sum(1 for _ in op.parameters()) for op in basic_ops]
            func_ctx.num_extra_inputs = num_extra_inputs
            func_ctx.num_extra_outputs = len(extra_outputs_flat)
            func_ctx.is_first_module = FP8GlobalStateManager.is_first_fp8_module()

        if extra_outputs_flat:
            return x, *extra_outputs_flat
        return x

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(
        func_ctx: Any,
        grad_output: torch.Tensor,
        *grad_extra_outputs: torch.Tensor,
    ) -> tuple[Optional[torch.Tensor], ...]:
        """Backward pass"""

        # Operations and autograd state
        backward_ops = func_ctx.backward_ops
        basic_ops = func_ctx.basic_ops
        basic_op_ctxs = func_ctx.basic_op_ctxs

        # Unflatten list of saved tensors
        for ctx in basic_op_ctxs:
            ctx.saved_tensors = func_ctx.saved_tensors[slice(*ctx._saved_tensors_range)]
            ctx._saved_tensors_range = None

        # Unflatten list of extra tensor output grads
        if len(grad_extra_outputs) != func_ctx.num_extra_outputs:
            raise ValueError(
                f"Expected grads for {func_ctx.num_extra_outputs} extra tensor outputs, "
                f"but got {len(grad_extra_outputs)}"
            )
        basic_op_grad_extra_outputs = []
        for op in basic_ops:
            dys, grad_extra_outputs = _split_tuple(grad_extra_outputs, op.num_extra_outputs)
            basic_op_grad_extra_outputs.append(dys)

        # Apply backward ops
        dx = grad_output
        grad_params = [None for _ in range(len(basic_ops))]
        grad_extra_inputs = [None for _ in range(len(basic_ops))]
        for op, basic_op_idxs in backward_ops:

            # Stop if no more gradients are required
            if all(not basic_op_ctxs[idx].requires_grad for idx in basic_op_idxs):
                dx = None
                break

            # Backward op
            grad_extra_outputs = [basic_op_grad_extra_outputs[idx] for idx in basic_op_idxs]
            dx, fused_op_grad_params, fused_op_grad_extra_inputs = op.fuser_backward(
                [basic_op_ctxs[idx] for idx in basic_op_idxs],
                dx,
                basic_op_grad_extra_outputs=grad_extra_outputs,
            )
            for idx, dparams in zip(basic_op_idxs, fused_op_grad_params):
                grad_params[idx] = dparams
                basic_op_ctxs[idx].saved_tensors = None
            for idx, dxs in zip(basic_op_idxs, fused_op_grad_extra_inputs):
                grad_extra_inputs[idx] = dxs

        # Flatten list of parameter gradients
        grad_params_flat = []
        for idx, dparams in enumerate(grad_params):
            num_params = func_ctx.basic_op_num_params[idx]
            if dparams is None:
                dparams = [None for _ in range(num_params)]
            else:
                dparams = list(dparams)
            if len(dparams) != num_params:
                raise RuntimeError(
                    f"Expected op {idx} to generate {num_params} param grads, "
                    f"but got {len(dparams)}"
                )
            grad_params_flat.extend(dparams)

        # Flatten list of parameter gradients
        grad_extra_inputs_flat = []
        for idx, dxs in enumerate(grad_extra_inputs):
            num_extra_inputs = basic_ops[idx].num_extra_inputs
            if dxs is None:
                dxs = [None for _ in range(num_extra_inputs)]
            else:
                dxs = list(dxs)
            if len(dxs) != num_extra_inputs:
                raise RuntimeError(
                    f"Expected op {idx} to generate grads "
                    f"for {num_extra_inputs} extra inputs, "
                    f"but got {len(dxs)}"
                )
            grad_extra_inputs_flat.extend(dxs)

        # Update FP8 scaling factors
        if func_ctx.is_first_module and not _is_graph_capturing():
            FP8GlobalStateManager.reduce_and_update_fp8_tensors(forward=False)

        return (
            dx,  # input_
            None,  # forward_ops
            None,  # backward_ops
            None,  # basic_ops
            None,  # basic_op_kwargs
            None,  # is_grad_enabled
            None,  # num_params
            None,  # num_extra_inputs
            *grad_params_flat,
            *grad_extra_inputs_flat,
        )


class OperationFuser:
    """Manages forward and backward passes for a pipeline of operations

    Parameters
    ----------
    ops: list of FusibleOperation
        Pipeline of operations
    fuse_ops: bool, default = `True`
        Whether to attempt fusing operations

    """

    def __init__(
        self,
        ops: list[FusibleOperation],
        fuse_ops: bool = True,
    ) -> None:

        # Get list of basic operations
        basic_ops = []
        for op in ops:
            if op.is_fused_op:
                basic_ops.extend(op.basic_ops)
            else:
                basic_ops.append(op)
        self._num_basic_ops: int = len(basic_ops)
        self._basic_ops: list[BasicOperation] = basic_ops

        # Number of extra tensor inputs
        self._num_extra_inputs: int = sum(op.num_extra_inputs for op in basic_ops)

        # Ops for forward and backward pass
        self._forward_ops: list[tuple[FusibleOperation, list[int]]]
        self._backward_ops: list[tuple[FusibleOperation, list[int]]]
        self._forward_ops = [(op, (idx,)) for idx, op in enumerate(self._basic_ops)]
        self._backward_ops = list(reversed(self._forward_ops))

        # Fuse ops if needed
        if fuse_ops:
            self.fuse_ops()

    @classmethod
    def _fuse_forward_ops(
        cls,
        ops: list[tuple[FusibleOperation, list[int]]],
    ) -> list[tuple[FusibleOperation, list[int]]]:
        """Attempt to fuse operations in forward pass"""
        ops = fuse_userbuffers_forward_linear(ops)
        ops = fuse_forward_linear_bias_add(ops)
        ops = fuse_forward_linear_bias_activation(ops)
        return ops

    @classmethod
    def _fuse_backward_ops(
        cls,
        ops: list[tuple[FusibleOperation, list[int]]],
    ) -> list[tuple[FusibleOperation, list[int]]]:
        """Attempt to fuse operations in backward pass"""
        ops = fuse_userbuffers_backward_linear(ops)
        ops = fuse_backward_linear_add(ops)
        return ops

    def fuse_ops(self) -> None:
        """Attempt to fuse operations"""
        self._forward_ops = self._fuse_forward_ops(self._forward_ops)
        self._backward_ops = self._fuse_backward_ops(self._backward_ops)

    def __call__(
        self,
        input: torch.Tensor,  # pylint: disable=redefined-builtin
        *extra_inputs: torch.Tensor,
        basic_op_kwargs: Optional[list[dict[str, Any]]] = None,
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:

        # Initialization before forward pass
        for op in self._basic_ops:
            op.pre_forward()

        # Canonicalize op kwargs
        if basic_op_kwargs is None:
            basic_op_kwargs = [{} for _ in range(len(self._basic_ops))]

        # Flatten list of parameters
        params = [param for op in self._basic_ops for param in op.parameters()]

        # Fuser forward pass
        is_grad_enabled = torch.is_grad_enabled()
        if is_grad_enabled:
            forward_func = _OperationFuserAutogradFunction.apply
            args = []
        else:
            forward_func = _OperationFuserAutogradFunction.forward
            args = [None]
        args += (
            input,
            self._forward_ops,
            self._backward_ops,
            self._basic_ops,
            basic_op_kwargs,
            is_grad_enabled,
            len(params),
            self._num_extra_inputs,
            *params,
            *extra_inputs,
        )
        return forward_func(*args)
