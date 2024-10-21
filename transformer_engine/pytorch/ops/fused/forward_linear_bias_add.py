# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Fused operation for forward GEMM + bias + add."""

from __future__ import annotations
from collections.abc import Iterable
from typing import Any, Optional

import torch

from transformer_engine.pytorch.fp8 import FP8GlobalStateManager
from transformer_engine.pytorch.ops.basic import AddInPlace, BasicLinear, Bias
from transformer_engine.pytorch.ops.op import (
    BasicOperation,
    FusedOperation,
    FusibleOperation,
    OperationContext,
)


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
        add: AddInPlace,
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
        basic_op_prev_ops: list[Optional[BasicOperation]],
        basic_op_next_ops: list[Optional[BasicOperation]],
        basic_op_kwargs: list[dict[str, Any]],
    ) -> tuple[torch.Tensor, Iterable[Iterable[torch.Tensor]]]:

        # Get basic operations
        idx = self._op_idxs["linear"]
        linear_op = self.basic_ops[idx]
        linear_op_ctx = basic_op_ctxs[idx]
        if self._op_idxs["bias"] is None:
            bias_op = None
            bias = None
        else:
            idx = self._op_idxs["bias"]
            bias_op = self.basic_ops[idx]
            bias = bias_op.bias
            if basic_op_kwargs[idx]:
                raise ValueError("Bias operation forward does not expect keyword arguments")

        # FP8 metadata
        with_fp8_compute = FP8GlobalStateManager.is_fp8_enabled()
        input_fp8_meta = None
        weight_fp8_meta = None
        output_fp8_meta = None
        grad_output_fp8_meta = None
        grad_input_fp8_meta = None
        if with_fp8_compute:
            input_fp8_meta = linear_op.get_fp8_meta("input")
            weight_fp8_meta = linear_op.get_fp8_meta("param")
            grad_output_fp8_meta = linear_op.get_fp8_meta("grad_output")
            prev_op = basic_op_prev_ops[0]
            if prev_op is not None and prev_op.num_fp8_scales("grad_output") > 0:
                grad_input_fp8_meta = prev_op.get_fp8_meta("grad_output")

        # Get autocast dtype if needed
        dtype = None
        if torch.is_autocast_enabled():
            dtype = torch.get_autocast_dtype("cuda")

        # Linear forward
        output = basic_op_extra_inputs[self._op_idxs["add"]][0]
        output, x_local, _ = BasicLinear._functional_forward(
            input=input_,
            weight=linear_op.weight,
            bias=bias,
            device=linear_op.device,
            out=output,
            accumulate_into_out=True,
            tensor_parallel_mode=linear_op.tensor_parallel_mode,
            tensor_parallel_group=linear_op.tensor_parallel_group,
            sequence_parallel=linear_op.sequence_parallel,
            with_fp8_compute=with_fp8_compute,
            input_fp8_meta=input_fp8_meta,
            weight_fp8_meta=weight_fp8_meta,
            output_fp8_meta=output_fp8_meta,
        )

        # Save state for backward pass
        linear_op_ctx.save_for_backward(x_local)
        linear_op_ctx.with_fp8_compute = with_fp8_compute
        linear_op_ctx.weight_fp8_meta = weight_fp8_meta
        linear_op_ctx.grad_output_fp8_meta = grad_output_fp8_meta
        linear_op_ctx.grad_input_fp8_meta = grad_input_fp8_meta
        linear_op_ctx.dtype = dtype
        linear_op_ctx.input_dims = input_.size()
        linear_op_ctx.input_requires_grad = input_.requires_grad
        linear_op_ctx.weight_requires_grad = linear_op.weight.requires_grad
        linear_op_ctx.has_prev_op = basic_op_prev_ops[0] is not None

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

        # Check if next op is add in-place
        if not isinstance(op, AddInPlace):
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
