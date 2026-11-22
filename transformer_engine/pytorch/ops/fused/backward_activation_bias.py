# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Fused backward dact + dbias + quantize."""

from __future__ import annotations
from typing import Optional
from dataclasses import dataclass

import transformer_engine_torch as tex
from transformer_engine.pytorch.quantization import Recipe
from transformer_engine.pytorch.ops.basic import Bias
from transformer_engine.pytorch.ops.basic.activation import (
    _ActivationOperation,
    ActivationBwdArgs,
    GELU,
    ReLU,
)
from transformer_engine.pytorch.ops.op import (
    FusedOperation,
    FusibleOperation,
)
from ...utils import clear_tensor_data
from ...dynamo.fake import fake_bgrad
from .._common import maybe_dequantize

_fused_activations = {GELU: tex.dbias_dgelu, ReLU: tex.dbias_drelu}
_fusible_activations = tuple(_fused_activations.keys())


@dataclass(slots=True)
class BackwardActivationBiasArgs(ActivationBwdArgs):
    """Activation gradient with a fused bias reduction."""

    activation: str


class BackwardActivationBias(FusedOperation):
    """Fused backward dact + dbias + quantize

    Uses the next operation's input quantizer.

    """

    def __init__(self, *, bias: Bias, activation: _ActivationOperation):
        super().__init__((bias, activation))

    bwd_args_type = BackwardActivationBiasArgs

    def compile_unsupported_reason(self, mode: str) -> Optional[str]:
        reason = super().compile_unsupported_reason(mode)
        return reason or self.basic_ops[1].compile_unsupported_reason(mode)

    def pack_backward_args(
        self, basic_op_ctxs, grad_output, **unused  # pylint: disable=unused-argument
    ) -> BackwardActivationBiasArgs:
        bias_ctx, act_ctx = basic_op_ctxs
        quantizer = bias_ctx.grad_input_quantizer
        if quantizer is None:
            raise RuntimeError(
                "BackwardActivationBias requires previous op's grad output quantizer, "
                "but Bias context has no quantizer"
            )
        return BackwardActivationBiasArgs(
            grad_output,
            act_ctx.saved_tensors[0],
            act_ctx.dtype,
            quantizer,
            type(self.basic_ops[1]).__name__.lower(),
        )

    @classmethod
    def backward_compute(cls, args: BackwardActivationBiasArgs, *, in_custom_op: bool = False):
        x = maybe_dequantize(args.input_, args.dtype).contiguous()
        dy = maybe_dequantize(args.grad_output, args.dtype).contiguous()
        compute = {"gelu": tex.dbias_dgelu, "relu": tex.dbias_drelu}[args.activation]
        db, dx = compute(dy, x, args.grad_input_quantizer)
        if not in_custom_op:
            clear_tensor_data(x)
        return dx, [(db,), ()], [(), ()]

    @classmethod
    def backward_compute_fake(cls, args: BackwardActivationBiasArgs):
        db, dx = fake_bgrad(args.grad_output, args.grad_input_quantizer, dtype=args.dtype)
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

        # Check if recipe supports bias activation fusion.
        # high_precision/dequantized backward overrides should use unfused backward ops.
        if recipe is None or recipe.backward_override is not None:
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
