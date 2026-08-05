# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Fusible NCCL expert-parallel combine operation."""

from __future__ import annotations

from typing import Any, Optional

import torch

from ...ep import EpBuffer, _alloc_io, is_symm_backed
from ...tensor import Quantizer
from ..op import BasicOperation, OperationContext


def _validate_grad_buffer(
    tensor: Optional[torch.Tensor],
    *,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
) -> Optional[torch.Tensor]:
    if tensor is None:
        return None
    if tuple(tensor.shape) != shape:
        raise ValueError(f"grad_out shape {tuple(tensor.shape)} does not match {shape}.")
    if tensor.dtype is not dtype:
        raise TypeError(f"grad_out must have dtype {dtype}, got {tensor.dtype}.")
    if tensor.device != device:
        raise ValueError(f"grad_out must be on {device}, got {tensor.device}.")
    if not tensor.is_contiguous():
        raise ValueError("grad_out must be contiguous.")
    if tensor.requires_grad:
        raise ValueError("grad_out must not require gradients.")
    return tensor


class Combine(BasicOperation):
    """Combine pre-weighted local expert outputs with NCCL EP.

    The operation uses routing state produced by a :class:`Dispatch` with the
    same :class:`EpBuffer`.
    """

    def __init__(self, buffer: EpBuffer, *, num_local_tokens: Optional[int] = None) -> None:
        super().__init__()
        self.buffer = buffer
        self.num_local_tokens = (
            buffer.max_tokens_per_rank if num_local_tokens is None else int(num_local_tokens)
        )
        if self.num_local_tokens < 0:
            raise ValueError("num_local_tokens must be non-negative.")

    def op_forward(
        self,
        ctx: OperationContext,
        input_: torch.Tensor,
        *,
        prev_op_grad_output_quantizer: Optional[Quantizer],
        next_op_input_quantizer: Optional[Quantizer],
        **kwargs: Any,
    ) -> torch.Tensor:
        del prev_op_grad_output_quantizer, next_op_input_quantizer
        if input_.dtype is not torch.bfloat16:
            raise NotImplementedError(f"NCCL EP requires BF16 combine input, got {input_.dtype}.")
        if input_.ndim != 2 or input_.shape[-1] != self.buffer.hidden_dim:
            raise ValueError(
                f"Combine input must have shape (R, {self.buffer.hidden_dim}), "
                f"got {tuple(input_.shape)}."
            )

        expert_out = input_
        if self.buffer.zero_copy:
            expert_out = _alloc_io(
                tuple(input_.shape),
                input_.dtype,
                input_.device,
                True,
            )
            expert_out.copy_(input_)

        result = torch.empty(
            self.num_local_tokens,
            self.buffer.hidden_dim,
            dtype=input_.dtype,
            device=input_.device,
        )
        torch.ops.transformer_engine_ep.combine(
            self.buffer.handle_mem,
            expert_out,
            result,
        )

        if ctx.requires_grad:
            grad_out = kwargs.get("grad_out")
            if self.buffer.eager and grad_out is not None:
                raise ValueError(
                    "eager mode sizes combine gradients per step and cannot use "
                    "a caller-supplied grad_out"
                )
            grad_out = _validate_grad_buffer(
                grad_out,
                shape=tuple(input_.shape),
                dtype=input_.dtype,
                device=input_.device,
            )
            if self.buffer.zero_copy and grad_out is not None and not is_symm_backed(grad_out):
                raise ValueError("zero-copy Combine grad_out must be symmetric-memory-backed.")
            ctx.grad_out = grad_out
            ctx.input_shape = tuple(input_.shape)
            ctx.input_dtype = input_.dtype
            ctx.save_for_backward(self.buffer.handle_mem)

        return result

    def op_backward(
        self,
        ctx: OperationContext,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[()]]:
        (handle_mem,) = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        grad_input = ctx.grad_out
        if grad_input is None:
            grad_input = _alloc_io(
                ctx.input_shape,
                ctx.input_dtype,
                grad_output.device,
                self.buffer.zero_copy,
            )
        torch.ops.transformer_engine_ep.combine_bwd(
            handle_mem,
            grad_output,
            grad_input,
        )
        return grad_input, ()
