# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Fused backward RMNorm + add."""

from __future__ import annotations
from typing import Optional
import math

import torch

import transformer_engine_torch as tex
from transformer_engine.pytorch.ops.basic import MakeExtraOutput, RMSNorm

from transformer_engine.pytorch.ops.op import (
    FusedOperation,
    FusibleOperation,
    OperationContext,
)
from ...utils import clear_tensor_data
from .._common import maybe_dequantize


class BackwardAddRMSNorm(FusedOperation):
    """Fused backward RMNorm + add"""

    def __init__(self, *, add: MakeExtraOutput, rmsnorm: RMSNorm):
        super().__init__((add, rmsnorm))

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
        rmsnorm_op = self.basic_ops[1]
        rmsnorm_op_ctx = basic_op_ctxs[0]

        # Saved tensors from forward pass
        x, rstdevs = rmsnorm_op_ctx.saved_tensors

        # Tensor dims
        weight_dims = rmsnorm_op.weight.size()
        inner_dim = math.prod(weight_dims)

        # Check input tensors
        dtype = rmsnorm_op_ctx.dtype
        extra_grad = basic_op_grad_extra_outputs[1][0]
        dy = maybe_dequantize(grad_output.contiguous(), dtype).view(x.size())
        w = maybe_dequantize(rmsnorm_op.weight, dtype).view((inner_dim,))
        add = maybe_dequantize(extra_grad.contiguous(), dtype).view(x.size())

        # Compute RMSNorm backward pass
        dx, dw = tex.rmsnorm_bwd_add(
            dy,
            x,
            add,
            rstdevs,
            w,
            rmsnorm_op._sm_margins["backward"],
            rmsnorm_op.zero_centered_gamma,
        )

        # Clear saved tensors if possible
        clear_tensor_data(x)
        clear_tensor_data(rstdevs)

        # Reshape results
        grad_input = dx.view(grad_output.size())
        grad_weight = dw.view(weight_dims)

        return grad_input, [(grad_weight,), ()], [(), ()]


def fuse_backward_add_rmsnorm(
    ops: list[tuple[FusibleOperation, list[int]]],
) -> list[tuple[FusibleOperation, list[int]]]:
    """Fused backward RMNorm + add

    Parameters
    ----------
    ops: list of tuples
        Backward pass operations and the indices of the corresponding
        basic operations.

    Returns
    -------
    ops: list of tuples
        Updated backward pass operations

    """

    # Scan through ops, fusing if possible
    out = []
    window = []
    while len(ops) >= 2:
        out.extend(window)

        # Check if first op is linear
        window, ops = ops[:1], ops[1:]
        op, _ = window[0]
        if not isinstance(op, RMSNorm):
            continue

        # Check if second op is "make extra output"
        op, _ = ops[0]
        if not isinstance(op, MakeExtraOutput):
            continue
        if op._in_place:
            continue
        window.extend(ops[:1])
        ops = ops[1:]

        # Replace window with fused op
        op = BackwardAddRMSNorm(
            rmsnorm=window[0][0],
            add=window[1][0],
        )
        basic_op_idxs = [basic_op_idxs[0] for _, basic_op_idxs in window]
        window = [(op, basic_op_idxs)]

    # Return list of ops
    out.extend(window)
    out.extend(ops)
    return out
