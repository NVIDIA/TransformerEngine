# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Fused backward dact + dbias + quantize."""

from __future__ import annotations
from typing import Optional

import torch

import transformer_engine_torch as tex
from transformer_engine.pytorch.quantization import Recipe
from transformer_engine.pytorch.ops.basic import Bias
from transformer_engine.pytorch.ops.basic.activation import (
    _ActivationOperation,
    GELU,
    ReLU,
)
from transformer_engine.pytorch.ops.op import (
    FusedOperation,
    FusibleOperation,
    OperationContext,
)
from ...utils import clear_tensor_data
from .._common import maybe_dequantize

_fused_activations = {GELU: tex.dbias_dgelu, ReLU: tex.dbias_drelu}
_fusible_activations = tuple(_fused_activations.keys())


class BackwardActivationBias(FusedOperation):
    """Fused backward dact + dbias + quantize

    Uses the next operation's input quantizer.

    """

    def __init__(self, *, bias: Bias, activation: _ActivationOperation):
        super().__init__((bias, activation))
        self._fused_function = _fused_activations[type(activation)]

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

        # Get basic operation contexts
        activation_op_ctx = basic_op_ctxs[0]
        bias_op_ctx = basic_op_ctxs[1]

        # Saved tensors from forward pass
        (act_input,) = activation_op_ctx.saved_tensors

        # Check activation input tensor
        act_input = maybe_dequantize(act_input.contiguous(), activation_op_ctx.dtype)

        # Check grad output tensor
        dy = maybe_dequantize(grad_output.contiguous(), act_input.dtype)

        # Get previous op quantizer
        quantizer = bias_op_ctx.grad_input_quantizer
        if quantizer is None:
            raise RuntimeError(
                "BackwardActivationBias requires previous op's grad output quantizer, "
                "but Bias context has no quantizer"
            )

        # Launch kernel
        db, dx = self._fused_function(dy, act_input, quantizer)

        # Clear activation input tensor
        clear_tensor_data(act_input)

        return dx, [(), (db,)], [(), ()]


def fuse_backward_activation_bias(
    ops: list[tuple[FusibleOperation, list[int]]],
    recipe: Optional[Recipe],
) -> list[tuple[FusibleOperation, list[int]]]:
    """Fused backward dact + dbias + quantize

    Parameters
    ----------
    ops: list of tuples
        Backward pass operations and the indices of the corresponding
        basic operations.
    recipe: Recipe, optional
        Used quantization recipe

    Returns
    -------
    ops: list of tuples
        Updated backward pass operations

    """

    # Check if recipe supports bias activation fusion
    if recipe is None:
        return ops

    # Scan through ops, fusing if possible
    out = []
    window = []
    while len(ops) >= 3:
        out.extend(window)

        # Check if first op is a supported activation
        window, ops = ops[:1], ops[1:]
        op, _ = window[0]
        if not isinstance(op, _fusible_activations):
            continue

        # Check if second op is bias
        op, _ = ops[0]
        if not isinstance(op, Bias):
            continue

        # Check if third op has a grad input quantizer
        op, _ = ops[1]
        if not op.num_quantizers("backward") > 0:
            continue

        window.extend(ops[:1])
        ops = ops[1:]

        # Replace window with fused op
        op = BackwardActivationBias(
            activation=window[0][0],
            bias=window[1][0],
        )
        basic_op_idxs = [basic_op_idxs[0] for _, basic_op_idxs in window]
        window = [(op, basic_op_idxs)]

    # Return list of ops
    out.extend(window)
    out.extend(ops)
    return out
