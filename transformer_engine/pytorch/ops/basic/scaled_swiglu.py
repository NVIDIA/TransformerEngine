# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Fusible operation for multiplying with extra input tensor."""

from __future__ import annotations
from collections.abc import Iterable
from typing import Any, Optional

import torch

import transformer_engine_torch as tex
from ...tensor import Quantizer
from ..op import BasicOperation, OperationContext
from .._common import maybe_dequantize


class ScaledSwiGLU(BasicOperation):
    """SwiGLU with post-scaling
    """

    # Operation expects scales
    num_extra_inputs: int = 1

    def __init__(self, gate_interleave_size: Optional[int] = None):
        super().__init__()
        self.gate_interleave_size: Optional[int] = gate_interleave_size

    def op_forward(self, *args, **kwargs) -> None:
        raise RuntimeError(
            "{self.__class__.__name__} operation has "
            f"{self.num_extra_inputs} extra tensor inputs "
            f"and {self.num_extra_outputs} extra tensor outputs. "
            "It overrides `fuser_forward` instead of `op_forward`."
        )

    def op_backward(self, *args, **kwargs) -> None:
        raise RuntimeError(
            "{self.__class__.__name__} operation has "
            f"{self.num_extra_inputs} extra tensor inputs "
            f"and {self.num_extra_outputs} extra tensor outputs. "
            "It overrides `fuser_backward` instead of `op_backward`."
        )

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
        extra_input = basic_op_extra_inputs[0][0]

        # Determine compute dtype
        if torch.is_autocast_enabled():
            dtype = torch.get_autocast_dtype("cuda")
        elif isinstance(input_, torch.Tensor):
            dtype = input_.dtype
        else:
            dtype = extra_input.dtype

        # Make sure inputs are in correct dtype
        input_ = maybe_dequantize(input_, dtype)
        scales = maybe_dequantize(extra_input, dtype)

        # Remove gate interleaving if needed
        swiglu_in = input_
        if self.gate_interleave_size is not None:
            shape = swiglu_in.size()
            swiglu_in = swiglu_in.reshape(
                -1,
                shape[-1] // (2 * self.gate_interleave_size),
                2,
                self.gate_interleave_size,
            )
            swiglu_in = swiglu_in.transpose(1, 2).contiguous()
            swiglu_in = swiglu_in.view(shape)

        # Compute scaled SwiGLU
        swiglu_out = tex.swiglu(swiglu_in, None)
        out = swiglu_out * scales.unsqueeze(-1)

        # Save state for backward pass
        ctx = basic_op_ctxs[0]
        if ctx.requires_grad:
            ctx.input_requires_grad = True
            ctx.extra_input_requires_grad = extra_input.requires_grad
            ctx.dtype = dtype
            ctx.save_for_backward(
                input_,
                scales if ctx.input_requires_grad else None,
            )

        return out, [()]

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
        ctx = basic_op_ctxs[0]
        input_, scales = ctx.saved_tensors
        input_ = maybe_dequantize(input_, ctx.dtype)
        if scales is not None:
            scales = maybe_dequantize(scales, ctx.dtype)
        grad_output = maybe_dequantize(grad_output, ctx.dtype)

        # Remove gate interleaving if needed
        swiglu_in = input_
        if self.gate_interleave_size is not None:
            shape = swiglu_in.size()
            swiglu_in = swiglu_in.reshape(
                -1,
                shape[-1] // (2 * self.gate_interleave_size),
                2,
                self.gate_interleave_size,
            )
            swiglu_in = swiglu_in.transpose(1, 2).contiguous()
            swiglu_in = swiglu_in.view(shape)

        # Compute input grad
        grad_input = None
        if ctx.input_requires_grad:
            grad_swiglu_out = grad_output * scales.unsqueeze(-1)
            grad_swiglu_in = tex.dswiglu(grad_swiglu_out, swiglu_in, None)
            grad_input = grad_swiglu_in
            if self.gate_interleave_size is not None:
                shape = grad_input.size()
                grad_input = grad_input.reshape(
                    -1,
                    2,
                    shape[-1] // (2 * self.gate_interleave_size),
                    self.gate_interleave_size,
                )
                grad_input = grad_input.transpose(1, 2).contiguous()
                grad_input = grad_input.view(shape)

        # Compute scales grad by recomputing SwiGLU
        grad_extra_input = None
        if ctx.extra_input_requires_grad:
            swiglu_out = tex.swiglu(swiglu_in, None)
            grad_extra_input = torch.linalg.vecdot(swiglu_out, grad_output)

        return grad_input, [()], [(grad_extra_input,)]
