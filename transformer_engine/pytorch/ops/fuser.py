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
from transformer_engine.pytorch.tensor.quantized_tensor import (
    prepare_for_saving,
    restore_from_saved,
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
        fuser: OperationFuser,
        basic_op_kwargs: list[dict[str, Any]],
        is_grad_enabled: bool,
        *params_and_extra_inputs: torch.nn.Parameter,
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        """Forward pass

        Parameters
        ----------
        func_ctx: torch.autograd.function.FunctionCtx
            Context for PyTorch autograd function
        input_: torch.Tensor
            Input to first operation in pipeline
        fuser: OperationFuser
            Container for the pipeline of operations to run
        basic_op_kwargs: list of dict
            Keyword arguments to BasicOperation
        is_grad_enabled: bool
            Should context be saved for backward
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
        basic_op_ctxs = [OperationContext() for _ in range(fuser._num_basic_ops)]

        # Unflatten list of parameters and extra tensor inputs
        extra_inputs = params_and_extra_inputs[-fuser._num_extra_inputs :]
        basic_op_extra_inputs = []
        for op in fuser._basic_ops:
            xs, extra_inputs = _split_tuple(extra_inputs, op.num_extra_inputs)
            basic_op_extra_inputs.append(xs)

        # Apply forward ops
        x = input_
        requires_grad = is_grad_enabled and x.requires_grad
        extra_outputs = [None] * fuser._num_basic_ops
        for op, basic_op_idxs in fuser._forward_ops:

            # Check if backward op is required
            if is_grad_enabled:
                if not requires_grad:
                    requires_grad = any(param.requires_grad for param in op.parameters())
                if not requires_grad:
                    requires_grad = any(any(x.requires_grad for x in xs) for xs in extra_inputs)
            for idx in basic_op_idxs:
                basic_op_ctxs[idx].requires_grad = requires_grad

            # Forward op
            extra_inputs = [basic_op_extra_inputs[idx] for idx in basic_op_idxs]
            prev_op_idx = basic_op_idxs[0] - 1
            prev_op = fuser._basic_ops[prev_op_idx] if prev_op_idx > 0 else None
            prev_op_grad_input_quantizer = None
            if prev_op is not None:
                prev_op_grad_input_quantizer = prev_op.get_grad_input_quantizer()
            next_op_idx = basic_op_idxs[-1] + 1
            next_op = fuser._basic_ops[next_op_idx] if next_op_idx < fuser._num_basic_ops else None
            next_op_input_quantizer = None
            if next_op is not None:
                next_op_input_quantizer = next_op.get_input_quantizer()
            is_first_op = prev_op is None

            x, fused_op_extra_outputs = op.fuser_forward(
                [basic_op_ctxs[idx] for idx in basic_op_idxs],
                x,
                basic_op_extra_inputs=extra_inputs,
                prev_op_grad_input_quantizer=prev_op_grad_input_quantizer,
                next_op_input_quantizer=next_op_input_quantizer,
                is_first_op=is_first_op,
                basic_op_kwargs=[basic_op_kwargs[idx] for idx in basic_op_idxs],
            )
            for idx, ys in zip(basic_op_idxs, fused_op_extra_outputs):
                for y in ys:
                    y.requires_grad_(requires_grad)
                extra_outputs[idx] = ys

        # Flatten list of extra outputs
        extra_outputs_flat = []
        for idx, ys in enumerate(extra_outputs):
            ys = list(ys)
            num_extra_outputs = fuser._basic_ops[idx].num_extra_outputs
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

            # Save tensors for backward
            with_quantized_compute = FP8GlobalStateManager.is_fp8_enabled()
            if with_quantized_compute:
                tensors_to_save, tensor_objects = prepare_for_saving(*to_save)
                func_ctx.save_for_backward(*tensors_to_save)
                func_ctx.tensor_objects = tensor_objects
            else:
                func_ctx.save_for_backward(*to_save)

            # Other context
            func_ctx.backward_ops = fuser._backward_ops
            func_ctx.basic_ops = fuser._basic_ops
            func_ctx.basic_op_ctxs = basic_op_ctxs
            func_ctx.basic_op_num_params = fuser._num_list_basic_op_params
            func_ctx.num_extra_inputs = fuser._num_extra_inputs
            func_ctx.num_extra_outputs = len(extra_outputs_flat)
            func_ctx.is_first_module = FP8GlobalStateManager.is_first_fp8_module()
            func_ctx.with_quantized_compute = with_quantized_compute

        if extra_outputs_flat:
            return x, *extra_outputs_flat

        x.requires_grad_(requires_grad)

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

        # Restore saved tensors
        if func_ctx.with_quantized_compute:
            saved_tensors = restore_from_saved(func_ctx.tensor_objects, func_ctx.saved_tensors)
        else:
            saved_tensors = func_ctx.saved_tensors

        # Unflatten list of saved tensors
        for ctx in basic_op_ctxs:
            ctx.saved_tensors = saved_tensors[slice(*ctx._saved_tensors_range)]
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
            None,  # fuser
            None,  # basic_op_kwargs
            None,  # is_grad_enabled
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

        # Flatten list of parameters
        self._basic_op_params = [param for op in self._basic_ops for param in op.parameters()]
        self._num_list_basic_op_params = [sum(1 for _ in op.parameters()) for op in self._basic_ops]

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
        # Verify extra input count
        if len(extra_inputs) != self._num_extra_inputs:
            raise ValueError(
                f"Expected {self._num_extra_inputs} extra inputs but got {len(extra_inputs)}"
            )

        # Initialization before forward pass
        for op in self._basic_ops:
            op.pre_forward()

        # Canonicalize op kwargs
        if basic_op_kwargs is None:
            basic_op_kwargs = [{}] * self._num_basic_ops

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
            self,
            basic_op_kwargs,
            is_grad_enabled,
            *self._basic_op_params,
            *extra_inputs,
        )
        return forward_func(*args)
