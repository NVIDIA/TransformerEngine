# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Fused operation for forward GEMM + bias + add."""

from __future__ import annotations
from collections.abc import Iterable
from typing import Any, Optional

import torch

from transformer_engine.pytorch.fp8 import FP8GlobalStateManager
from transformer_engine.pytorch.ops.basic import AddExtraInput, BasicLinear, Bias
from transformer_engine.pytorch.ops.op import (
    FusedOperation,
    FusibleOperation,
    OperationContext,
)
from transformer_engine.pytorch.tensor import Quantizer


class ForwardLinearBiasAdd(FusedOperation):
    """Fused forward GEMM + bias + add

    Bias is optional. Row tensor parallelism is not supported since
    that requires communication immediately after the GEMM.

    """

    def __init__(
        self,
        *,
        linear: BasicLinear,
        bias: Optional[Bias],
        add: AddExtraInput,
    ) -> None:

        # Basic operations that comprise this fused operation
        op_idxs = {"linear": 0, "bias": None, "add": None}
        ops = [linear]
        if bias is not None:
            op_idxs["bias"] = len(ops)
            ops.append(bias)
        op_idxs["add"] = len(ops)
        ops.append(add)

        # Initialize base class
        super().__init__(ops)

        # Index of each basic operations
        self._op_idxs: dict[str, Optional[int]] = op_idxs

    def fuser_forward(
        self,
        basic_op_ctxs: list[OperationContext],
        input_: torch.Tensor,
        *,
        basic_op_extra_inputs: list[tuple[torch.Tensor, ...]],
        prev_op_grad_output_quantizer: Optional[Quantizer],
        next_op_input_quantizer: Optional[Quantizer],
        basic_op_kwargs: list[dict[str, Any]],
    ) -> tuple[torch.Tensor, Iterable[Iterable[torch.Tensor]]]:

        # Get basic operations
        idx = self._op_idxs["linear"]
        linear_op = self.basic_ops[idx]
        linear_op_ctx = basic_op_ctxs[idx]
        if self._op_idxs["bias"] is None:
            bias_op = None
            bias_op_ctx = None
            bias = None
        else:
            idx = self._op_idxs["bias"]
            bias_op = self.basic_ops[idx]
            bias_op_ctx = basic_op_ctxs[idx]
            bias = bias_op.bias
            if basic_op_kwargs[idx]:
                raise ValueError("Bias operation forward does not expect keyword arguments")

        # Check which grads are required
        input_requires_grad = linear_op_ctx.requires_grad
        weight_requires_grad = linear_op_ctx.requires_grad and linear_op.weight.requires_grad

        # FP8 metadata
        input_quantizer = linear_op.get_quantizer("forward", 0)
        weight_quantizer = linear_op.get_quantizer("forward", 1)
        output_quantizer = None
        grad_output_quantizer = linear_op.get_quantizer("backward", 0)
        grad_input_quantizer = prev_op_grad_output_quantizer
        with_quantized_compute = FP8GlobalStateManager.is_fp8_enabled()

        # Get autocast dtype if needed
        if torch.is_autocast_enabled():
            dtype = torch.get_autocast_dtype("cuda")
        else:
            dtype = linear_op.weight.dtype

        # Linear forward
        output = basic_op_extra_inputs[self._op_idxs["add"]][0]
        output, x_local, w = BasicLinear._functional_forward(
            input=input_,
            weight=linear_op.weight,
            bias=bias,
            dtype=output.dtype,
            out=output,
            accumulate_into_out=True,
            tensor_parallel_mode=linear_op.tensor_parallel_mode,
            tensor_parallel_group=linear_op.tensor_parallel_group,
            sequence_parallel=linear_op.sequence_parallel,
            with_quantized_compute=with_quantized_compute,
            input_quantizer=input_quantizer,
            weight_quantizer=weight_quantizer,
            output_quantizer=output_quantizer,
            input_requires_grad=input_requires_grad,
            weight_requires_grad=weight_requires_grad,
        )

        # Save state for backward pass
        if linear_op_ctx.requires_grad:
            linear_op_ctx.save_for_backward(x_local, w)
            linear_op_ctx.with_quantized_compute = with_quantized_compute
            linear_op_ctx.input_quantizer = input_quantizer
            linear_op_ctx.weight_quantizer = weight_quantizer
            linear_op_ctx.grad_output_quantizer = grad_output_quantizer
            linear_op_ctx.grad_input_quantizer = grad_input_quantizer
            linear_op_ctx.dtype = dtype
            linear_op_ctx.input_requires_grad = input_requires_grad
            linear_op_ctx.weight_requires_grad = weight_requires_grad
        if bias_op is not None and bias_op_ctx.requires_grad:
            bias_op_ctx.grad_input_quantizer = linear_op.get_grad_output_quantizer()

        return output, [() for _ in range(len(self.basic_ops))]


def fuse_forward_linear_bias_add(
    ops: list[tuple[FusibleOperation, list[int]]],
) -> list[tuple[FusibleOperation, list[int]]]:
    """Fuse forward GEMM + bias + add

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
        if op.tensor_parallel_mode == "row":
            # Row tensor-parallelism requires communication after the
            # GEMM
            continue
        linear = op
        op, _ = ops[0]

        # Check if next op is bias
        bias = None
        if isinstance(op, Bias):
            bias = op
            window.extend(ops[:1])
            ops = ops[1:]
            if len(ops) == 0:
                continue
            op, _ = ops[0]

        # Check if next op is in-place add extra input
        if not isinstance(op, AddExtraInput):
            continue
        if not op._in_place:
            continue
        add = op
        window.extend(ops[:1])
        ops = ops[1:]

        # Replace window with fused op
        op = ForwardLinearBiasAdd(
            linear=linear,
            bias=bias,
            add=add,
        )
        basic_op_idxs = [basic_op_idxs[0] for _, basic_op_idxs in window]
        window = [(op, basic_op_idxs)]

    # Return list of ops
    out.extend(window)
    out.extend(ops)
    return out
