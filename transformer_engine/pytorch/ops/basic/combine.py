# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Fusible NCCL expert-parallel combine operation."""

from __future__ import annotations

from typing import Any, Iterable, Optional

import torch
import transformer_engine_torch as tex

from ...constants import DType, MXFP8_BLOCK_SCALING_SIZE
from ...ep import (
    _alloc_io,
    _make_grouped_mxfp8,
    _scale_alloc_io,
    is_symm_backed,
)
from ...quantization import QuantizerRole, Recipe
from ...tensor import MXFP8Quantizer, Quantizer
from .._common import is_quantized_tensor, maybe_dequantize, quantize_mxfp8_for_ep
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

    The operation consumes the routing handle and tokens-per-expert produced by
    a preceding :class:`Dispatch` through extra-tensor channels.
    """

    num_extra_inputs: int = 2

    def __init__(self, *, num_local_tokens: int) -> None:
        super().__init__()
        self.num_local_tokens = int(num_local_tokens)
        if self.num_local_tokens < 0:
            raise ValueError("num_local_tokens must be non-negative.")

    def num_quantizers(self, mode: str) -> int:
        return 1 if mode == "backward" else 0

    def get_quantizer_roles(self, mode: str) -> Optional[list[QuantizerRole]]:
        if mode == "backward":
            name = getattr(self, "name", "") or ""
            return [
                QuantizerRole(
                    module_type="combine",
                    tensor_type="grad_output",
                    name=name,
                )
            ]
        return None

    def pre_fuser_forward(self, *, requires_grad: bool) -> None:
        super().pre_fuser_forward(requires_grad=requires_grad)
        quantizer = self.get_quantizer("backward", 0)
        if quantizer is not None:
            quantizer.set_usage(rowwise=True, columnwise=False)
            quantizer.optimize_for_gemm = False

    def reset_recipe_state(self, *, recipe: Optional[Recipe]) -> None:
        super().reset_recipe_state(recipe=recipe)
        quantizer = self.get_quantizer("backward", 0)
        if quantizer is not None:
            quantizer.internal = True

    def _resolve_grad_output_quantizer(
        self,
        prev_op_grad_output_quantizer: Optional[Quantizer],
    ) -> Optional[MXFP8Quantizer]:
        quantizer = self.get_quantizer("backward", 0)
        if (
            quantizer is not None
            and prev_op_grad_output_quantizer is not None
            and quantizer is not prev_op_grad_output_quantizer
        ):
            raise ValueError(
                "Combine grad_output_quantizer and previous operation grad-output "
                "quantizer must be the same object when both are set."
            )
        if quantizer is None:
            quantizer = prev_op_grad_output_quantizer
        if quantizer is None:
            return None
        if not isinstance(quantizer, MXFP8Quantizer):
            raise TypeError(
                "NCCL EP Combine backward supports MXFP8Quantizer only, got "
                f"{type(quantizer).__name__}."
            )
        if quantizer.dtype != DType.kFloat8E4M3:
            raise NotImplementedError("NCCL EP Combine backward supports E4M3 MXFP8 only.")
        return quantizer

    def op_forward(self, *args: Any, **kwargs: Any) -> None:
        raise RuntimeError("Combine uses fuser_forward")

    def op_backward(self, *args: Any, **kwargs: Any) -> None:
        raise RuntimeError("Combine uses fuser_backward")

    def fuser_forward(
        self,
        basic_op_ctxs: list[OperationContext],
        input_: torch.Tensor,
        *,
        basic_op_extra_inputs: list[tuple[torch.Tensor, ...]],
        prev_op_grad_output_quantizer: Optional[Quantizer],
        next_op_input_quantizer: Optional[Quantizer],
        basic_op_kwargs: list[dict[str, Any]],
    ) -> tuple[torch.Tensor, list[tuple[()]]]:
        grad_output_quantizer = self._resolve_grad_output_quantizer(
            prev_op_grad_output_quantizer
        )
        handle_mem, tokens_per_expert = basic_op_extra_inputs[0]
        kwargs = basic_op_kwargs[0]
        if input_.dtype is not torch.bfloat16:
            raise NotImplementedError(f"NCCL EP requires BF16 combine input, got {input_.dtype}.")
        if input_.ndim != 2:
            raise ValueError(f"Combine input must be 2D, got shape {tuple(input_.shape)}.")
        if handle_mem.dtype is not torch.uint8 or handle_mem.device != input_.device:
            raise ValueError("Combine routing handle must be a uint8 tensor on the input device.")
        if tokens_per_expert.dtype is not torch.int64 or tokens_per_expert.device != input_.device:
            raise ValueError(
                "Combine tokens_per_expert must be an int64 tensor on the input device."
            )

        expert_out = input_
        zero_copy = bool(tex.ep_get_zero_copy())
        if zero_copy:
            expert_out = _alloc_io(
                tuple(input_.shape),
                input_.dtype,
                input_.device,
                True,
            )
            expert_out.copy_(input_)

        result = torch.empty(
            self.num_local_tokens,
            input_.shape[-1],
            dtype=input_.dtype,
            device=input_.device,
        )
        torch.ops.transformer_engine_ep.combine(
            handle_mem,
            expert_out,
            result,
        )

        ctx = basic_op_ctxs[0]
        if ctx.requires_grad:
            grad_out = kwargs.get("grad_out")
            if grad_output_quantizer is None:
                grad_out = _validate_grad_buffer(
                    grad_out,
                    shape=tuple(input_.shape),
                    dtype=input_.dtype,
                    device=input_.device,
                )
            elif grad_out is not None:
                if grad_out.device != input_.device:
                    raise ValueError(
                        f"grad_out must be on {input_.device}, got {grad_out.device}."
                    )
                if not grad_out.is_contiguous():
                    raise ValueError("MXFP8 grad_out storage must be contiguous.")
                if grad_out.requires_grad:
                    raise ValueError("grad_out must not require gradients.")
            if zero_copy and grad_out is not None and not is_symm_backed(grad_out):
                raise ValueError("zero-copy Combine grad_out must be symmetric-memory-backed.")
            ctx.grad_out = grad_out
            ctx.input_shape = tuple(input_.shape)
            ctx.input_dtype = input_.dtype
            ctx.zero_copy = zero_copy
            ctx.grad_output_quantizer = grad_output_quantizer
            ctx.save_for_backward(handle_mem, tokens_per_expert)

        if next_op_input_quantizer is not None and not is_quantized_tensor(result):
            result = next_op_input_quantizer(result)
        return result, [()]

    def fuser_backward(
        self,
        basic_op_ctxs: list[OperationContext],
        grad_output: torch.Tensor,
        *,
        basic_op_grad_extra_outputs: list[tuple[Optional[torch.Tensor], ...]],
    ) -> tuple[
        torch.Tensor,
        Iterable[Iterable[Optional[torch.Tensor]]],
        Iterable[Iterable[Optional[torch.Tensor]]],
    ]:
        del basic_op_grad_extra_outputs
        ctx = basic_op_ctxs[0]
        handle_mem, tokens_per_expert = ctx.saved_tensors
        quantizer = ctx.grad_output_quantizer
        if quantizer is None:
            grad_output = maybe_dequantize(grad_output, ctx.input_dtype).contiguous()
            grad_input = ctx.grad_out
            if grad_input is None:
                grad_input = _alloc_io(
                    ctx.input_shape,
                    ctx.input_dtype,
                    grad_output.device,
                    ctx.zero_copy,
                )
            torch.ops.transformer_engine_ep.combine_bwd(
                handle_mem,
                grad_output,
                grad_input,
            )
        else:
            quantized_grad, grad_scale_inv = quantize_mxfp8_for_ep(
                grad_output, quantizer
            )
            rows, hidden = ctx.input_shape
            scale_cols = hidden // MXFP8_BLOCK_SCALING_SIZE
            grad_data, grad_input_scale_inv = _scale_alloc_io(
                ctx.grad_out,
                rows,
                hidden,
                scale_cols,
                quantized_grad.rowwise_data.dtype,
                grad_scale_inv.dtype,
                grad_output.device,
                ctx.zero_copy,
            )
            torch.ops.transformer_engine_ep.combine_bwd(
                handle_mem,
                quantized_grad.rowwise_data.view(torch.float8_e4m3fn),
                grad_data.view(torch.float8_e4m3fn),
                grad_scale_inv,
                grad_input_scale_inv,
            )
            grad_input = _make_grouped_mxfp8(
                grad_data,
                grad_input_scale_inv,
                tokens_per_expert,
                quantized_grad._fp8_dtype,
                ctx.input_dtype,
            )
        return grad_input, [()], [(None, None)]
