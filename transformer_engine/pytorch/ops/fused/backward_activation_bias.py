# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
        bias_op_ctx = basic_op_ctxs[0]
        activation_op_ctx = basic_op_ctxs[1]

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

        return dx, [(db,), ()], [(), ()]

    @staticmethod
    def fuse_backward_ops(
        ops: list[FusibleOperation],
        *,
        recipe: Optional[Recipe] = None,
        **unused,  # pylint: disable=unused-argument
    ) -> list[FusibleOperation]:
        """Apply operation fusion for backward pass.

        Parameters
        ----------
        ops : list of FusibleOperation
            Backward pass operations.
        recipe : Recipe, optional
            Quantization recipe.

        Returns
        -------
        ops : list of FusibleOperation
            Updated backward pass operations

        """

        # Check if recipe supports bias activation fusion
        if recipe is None:
            return ops

        # Scan through ops, fusing if possible
        out = []
        window, ops = ops[:3], ops[3:]
        while len(window) == 3:
            if (
                isinstance(window[2], _fusible_activations)
                and isinstance(window[1], Bias)
                and window[0].get_grad_output_quantizer() is not None
            ):
                # Construct fused op if window matches pattern
                op = BackwardActivationBias(bias=window[1], activation=window[2])
                window = [window[0], op]
            else:
                # Shift window if window doesn't match pattern
                out.extend(window[:-2])
                window = window[-2:]

            # Adjust window to expected size
            out.extend(window[:-3])
            window = window[-3:]
            while ops and len(window) < 3:
                window.append(ops[0])
                ops = ops[1:]

        # Return list of ops
        out.extend(window)
        return out
