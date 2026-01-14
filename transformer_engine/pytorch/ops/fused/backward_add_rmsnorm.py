# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
        rmsnorm_op_ctx = basic_op_ctxs[1]

        # Saved tensors from forward pass
        x, rstdevs = rmsnorm_op_ctx.saved_tensors

        # Tensor dims
        weight_dims = rmsnorm_op.weight.size()
        inner_dim = math.prod(weight_dims)

        # Check input tensors
        dtype = rmsnorm_op_ctx.dtype
        extra_grad = basic_op_grad_extra_outputs[0][0]
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

        return grad_input, [(), (grad_weight,)], [(), ()]

    @staticmethod
    def fuse_backward_ops(
        ops: list[FusibleOperation],
        **unused,  # pylint: disable=unused-argument
    ) -> list[FusibleOperation]:
        """Apply operation fusion for backward pass.

        Parameters
        ----------
        ops : list of FusibleOperation
            Backward pass operations.

        Returns
        -------
        ops : list of FusibleOperation
            Updated backward pass operations

        """

        # Scan through ops, fusing if possible
        out = []
        window = []
        while ops:

            # Shift window
            while len(window) >= 2:
                out.append(window[0])
                window = window[1:]
            while ops and len(window) < 2:
                window.append(ops[0])
                ops = ops[1:]

            # Construct fused op if window matches pattern
            if (
                len(window) == 2
                and isinstance(window[0], MakeExtraOutput)
                and isinstance(window[1], RMSNorm)
                and not window[0]._in_place
            ):
                op = BackwardAddRMSNorm(add=window[0], rmsnorm=window[1])
                window = [op]

        # Return list of ops
        out.extend(window)
        return out
