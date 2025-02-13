# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Fused backward dgrad GEMM + add."""

from __future__ import annotations
from typing import Optional

import torch

from transformer_engine.pytorch.ops.basic import BasicLinear, MakeExtraOutput
from transformer_engine.pytorch.ops.op import (
    FusedOperation,
    FusibleOperation,
    OperationContext,
)
from ...utils import clear_tensor_data


class BackwardLinearAdd(FusedOperation):
    """Fused backward dgrad GEMM + add

    Column tensor parallelism is not supported since that requires
    communication immediately after the dgrad GEMM.

    """

    def __init__(
        self,
        *,
        linear: BasicLinear,
        backward_add: MakeExtraOutput,
    ) -> None:
        super().__init__((linear, backward_add))

    def fuser_backward(
        self,
        basic_op_ctxs: list[OperationContext],
        grad_output: torch.Tensor,
        *,
        basic_op_grad_extra_outputs: list[tuple[torch.Tensor, ...]],
    ) -> tuple[
        torch.Tensor,
        list[tuple[Optional[torch.Tensor], ...]],
        list[tuple[()]],
    ]:

        # Get basic operations
        linear_op = self.basic_ops[0]
        linear_op_ctx = basic_op_ctxs[0]

        # Saved tensors from forward pass
        (x_local,) = linear_op_ctx.saved_tensors

        # wgrad fusion
        accumulate_into_main_grad = linear_op._accumulate_into_main_grad
        grad_weight = None
        if linear_op_ctx.weight_requires_grad and accumulate_into_main_grad:
            if not hasattr(linear_op.weight, "main_grad"):
                raise RuntimeError(
                    "BasicLinear op is configured with "
                    "accumulate_into_main_grad=True, "
                    "but weight parameter does not have main_grad attribute"
                )
            grad_weight = linear_op.weight.main_grad.detach()
        else:
            accumulate_into_main_grad = False

        # Linear backward pass
        grad_input = basic_op_grad_extra_outputs[1][0]
        grad_input, grad_weight = BasicLinear._functional_backward(
            grad_output=grad_output,
            input=x_local,
            weight=linear_op.weight,
            input_requires_grad=linear_op_ctx.input_requires_grad,
            weight_requires_grad=linear_op_ctx.weight_requires_grad,
            dtype=grad_input.dtype,
            grad_weight=grad_weight,
            accumulate_into_grad_weight=accumulate_into_main_grad,
            grad_input=grad_input,
            accumulate_into_grad_input=True,
            tensor_parallel_mode=linear_op.tensor_parallel_mode,
            tensor_parallel_group=linear_op.tensor_parallel_group,
            sequence_parallel=linear_op.sequence_parallel,
            with_quantized_compute=linear_op_ctx.with_quantized_compute,
            input_quantizer=linear_op_ctx.input_quantizer,
            weight_quantizer=linear_op_ctx.weight_quantizer,
            grad_output_quantizer=linear_op_ctx.grad_output_quantizer,
            grad_input_quantizer=linear_op_ctx.grad_input_quantizer,
        )
        if accumulate_into_main_grad:
            grad_weight = None

        # Clear input tensor if possible
        if linear_op_ctx.has_prev_op:
            clear_tensor_data(x_local)

        return grad_input, [(grad_weight,), ()], [(), ()]


def fuse_backward_linear_add(
    ops: list[tuple[FusibleOperation, list[int]]],
) -> list[tuple[FusibleOperation, list[int]]]:
    """Fused backward dgrad GEMM + add

    Parameters
    ----------
    ops: list of tuples
        Forward pass operations and the indices of the corresponding
        basic operations.

    Returns
    -------
    ops: list of tuples
        Updated forward pass operations

    """

    # Scan through ops, fusing if possible
    out = []
    window = []
    while len(ops) >= 2:
        out.extend(window)

        # Check if first op is linear
        window, ops = ops[:1], ops[1:]
        op, _ = window[0]
        if not isinstance(op, BasicLinear):
            continue
        if op.tensor_parallel_mode == "column":
            # Row tensor-parallelism requires communication after the
            # GEMM
            continue

        # Check if second op is "make extra output"
        op, _ = ops[0]
        if not isinstance(op, MakeExtraOutput):
            continue
        window.extend(ops[:1])
        ops = ops[1:]

        # Replace window with fused op
        op = BackwardLinearAdd(
            linear=window[0][0],
            backward_add=window[1][0],
        )
        basic_op_idxs = [basic_op_idxs[0] for _, basic_op_idxs in window]
        window = [(op, basic_op_idxs)]

    # Return list of ops
    out.extend(window)
    out.extend(ops)
    return out
