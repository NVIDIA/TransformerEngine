# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Fusable operation for SiLU gating."""

from __future__ import annotations
from collections.abc import Iterable, Sequence
from typing import Any, Optional

import torch

from ...cpu_offload import is_cpu_offload_enabled, mark_activation_offload
from ...tensor import Quantizer
from ..op import BasicOperation, OperationContext
from .._common import maybe_autocast_dtype, maybe_dequantize


class SiLUGate(BasicOperation):
    r"""Multiply input by the SiLU activation of a gate tensor

    This operation takes the main input :math:`x` and an extra gate
    tensor :math:`g` and computes

    .. math::
        y = x * \mathrm{SiLU}(g) = \frac{x g}{1 + \exp(-g)}

    The gate tensor is an extra operation input and must have the same
    shape as the main input. Computation is performed in FP32 and the
    output is returned in the input dtype.

    This operation is typically used after a normalization operation to
    implement gated RMSNorm, as in Mamba2 and Qwen3-Next:

    .. code-block:: python

        op = Sequential(RMSNorm(hidden_size), SiLUGate())

    Output quantization is not supported.

    """

    num_extra_inputs: int = 1

    def fuser_forward(
        self,
        basic_op_ctxs: list[OperationContext],
        input_: torch.Tensor,
        *,
        basic_op_extra_inputs: list[tuple[torch.Tensor, ...]],
        prev_op_grad_output_quantizer: Optional[Quantizer],
        next_op_input_quantizer: Optional[Quantizer],
        basic_op_kwargs: list[dict[str, Any]],
    ) -> tuple[torch.Tensor, Sequence[Sequence[torch.Tensor]]]:
        # Output quantization is not supported. Return an unquantized
        # tensor and leave quantization to the next operation.
        del prev_op_grad_output_quantizer, next_op_input_quantizer, basic_op_kwargs

        # Check input tensors
        x = maybe_dequantize(input_.contiguous())
        gate = maybe_dequantize(basic_op_extra_inputs[0][0].contiguous())
        if gate.shape != x.shape:
            raise ValueError(
                f"Input tensor (shape={tuple(x.shape)}) "
                f"and gate tensor (shape={tuple(gate.shape)}) are not compatible"
            )
        dtype = maybe_autocast_dtype(default_dtype=x.dtype)
        x = maybe_dequantize(x, dtype)

        # Compute SiLU gate in FP32
        y = (x.float() * torch.nn.functional.silu(gate.float())).to(dtype)

        # Save state for backward pass
        ctx = basic_op_ctxs[0]
        if ctx.requires_grad:
            if is_cpu_offload_enabled():
                mark_activation_offload(x, gate)
            ctx.save_for_backward(x, gate)

        return y, [()]

    def fuser_backward(
        self,
        basic_op_ctxs: list[OperationContext],
        grad_output: torch.Tensor,
        *,
        basic_op_grad_extra_outputs: list[tuple[torch.Tensor, ...]],
    ) -> tuple[
        torch.Tensor,
        Iterable[Iterable[Optional[torch.Tensor]]],
        Iterable[Iterable[Optional[torch.Tensor]]],
    ]:
        del basic_op_grad_extra_outputs

        # Saved tensors from forward pass
        ctx = basic_op_ctxs[0]
        x, gate = ctx.saved_tensors

        dy = maybe_dequantize(grad_output.contiguous(), x.dtype)

        # Gradients of the SiLU gate, computed in FP32
        x_fp32 = x.float()
        gate_fp32 = gate.float()
        dy_fp32 = dy.float()
        sigmoid = torch.sigmoid(gate_fp32)
        silu_grad = sigmoid * (1 + gate_fp32 * (1 - sigmoid))
        dx = (dy_fp32 * torch.nn.functional.silu(gate_fp32)).to(x.dtype)
        dgate = (dy_fp32 * x_fp32 * silu_grad).to(gate.dtype)

        return dx, [()], [(dgate,)]
