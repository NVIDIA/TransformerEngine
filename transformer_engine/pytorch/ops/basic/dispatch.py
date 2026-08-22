# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Fusible NCCL expert-parallel dispatch operation."""

from __future__ import annotations

from typing import Any, Iterable, Optional

import torch

from ...constants import DType, MXFP8_BLOCK_SCALING_SIZE
from ...ep import (
    EpBuffer,
    _alloc_io,
    _make_grouped_mxfp8,
    _scale_alloc_io,
    ep_prepare,
)
from ...quantization import QuantizerRole, Recipe
from ...tensor import MXFP8Quantizer, Quantizer
from .._common import (
    is_quantized_tensor,
    maybe_dequantize,
    quantize_mxfp8_for_ep,
)
from ..op import BasicOperation, OperationContext


def _validate_output_buffer(
    name: str,
    tensor: Optional[torch.Tensor],
    *,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
) -> Optional[torch.Tensor]:
    if tensor is None:
        return None
    if tuple(tensor.shape) != shape:
        raise ValueError(f"{name} shape {tuple(tensor.shape)} does not match {shape}.")
    if tensor.dtype is not dtype:
        raise TypeError(f"{name} must have dtype {dtype}, got {tensor.dtype}.")
    if tensor.device != device:
        raise ValueError(f"{name} must be on {device}, got {tensor.device}.")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous.")
    if tensor.requires_grad:
        raise ValueError(f"{name} must not require gradients.")
    return tensor


class Dispatch(BasicOperation):
    """Dispatch BF16 tokens to local experts with NCCL EP.

    The extra inputs are routing indices and FP32 routing weights. The extra
    outputs are local tokens-per-expert and received routing weights.
    """

    num_extra_inputs: int = 2
    # tokens-per-expert, received routing weights, and the opaque NCCL EP
    # routing handle consumed by Combine.
    num_extra_outputs: int = 3

    def __init__(self, buffer: EpBuffer) -> None:
        super().__init__()
        self.buffer = buffer

    def num_quantizers(self, mode: str) -> int:
        return 1 if mode == "forward" else 0

    def get_quantizer_roles(self, mode: str) -> Optional[list[QuantizerRole]]:
        if mode == "forward":
            name = getattr(self, "name", "") or ""
            return [
                QuantizerRole(
                    module_type="dispatch",
                    tensor_type="input",
                    name=name,
                )
            ]
        return None

    def pre_fuser_forward(self, *, requires_grad: bool) -> None:
        super().pre_fuser_forward(requires_grad=requires_grad)
        quantizer = self.get_quantizer("forward", 0)
        if quantizer is not None:
            quantizer.set_usage(rowwise=True, columnwise=False)
            quantizer.optimize_for_gemm = False

    def reset_recipe_state(self, *, recipe: Optional[Recipe]) -> None:
        super().reset_recipe_state(recipe=recipe)
        quantizer = self.get_quantizer("forward", 0)
        if quantizer is not None:
            quantizer.internal = True

    def _resolve_input_quantizer(
        self,
        next_op_input_quantizer: Optional[Quantizer],
    ) -> Optional[MXFP8Quantizer]:
        quantizer = self.get_quantizer("forward", 0)
        if (
            quantizer is not None
            and next_op_input_quantizer is not None
            and quantizer is not next_op_input_quantizer
        ):
            raise ValueError(
                "Dispatch input_quantizer and next operation input quantizer "
                "must be the same object when both are set."
            )
        if quantizer is None:
            quantizer = next_op_input_quantizer
        if quantizer is None:
            return None
        if not isinstance(quantizer, MXFP8Quantizer):
            raise TypeError(
                f"NCCL EP Dispatch supports MXFP8Quantizer only, got {type(quantizer).__name__}."
            )
        if quantizer.dtype != DType.kFloat8E4M3:
            raise NotImplementedError("NCCL EP Dispatch supports E4M3 MXFP8 only.")
        return quantizer

    def op_forward(self, *args: Any, **kwargs: Any) -> None:
        raise RuntimeError("Dispatch uses fuser_forward")

    def op_backward(self, *args: Any, **kwargs: Any) -> None:
        raise RuntimeError("Dispatch uses fuser_backward")

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
        input_quantizer = self._resolve_input_quantizer(next_op_input_quantizer)
        topk_idx, topk_weights = basic_op_extra_inputs[0]
        kwargs = basic_op_kwargs[0]

        if input_.dtype is not torch.bfloat16:
            raise NotImplementedError(f"NCCL EP requires BF16 dispatch input, got {input_.dtype}.")
        if input_.ndim != 2 or input_.shape[-1] != self.buffer.hidden_dim:
            raise ValueError(
                f"Dispatch input must have shape (T, {self.buffer.hidden_dim}), "
                f"got {tuple(input_.shape)}."
            )
        if topk_idx.dtype not in (torch.int32, torch.int64):
            raise TypeError(f"topk_idx must be int32 or int64, got {topk_idx.dtype}.")
        expected_route_shape = (input_.shape[0], self.buffer.top_k)
        if tuple(topk_idx.shape) != expected_route_shape:
            raise ValueError(
                f"topk_idx shape must be {expected_route_shape}, got {tuple(topk_idx.shape)}."
            )
        if tuple(topk_weights.shape) != expected_route_shape:
            raise ValueError(
                f"topk_weights shape must be {expected_route_shape}, "
                f"got {tuple(topk_weights.shape)}."
            )
        if topk_weights.dtype is not torch.float32:
            raise TypeError(f"topk_weights must be float32, got {topk_weights.dtype}.")
        for name, tensor in (("topk_idx", topk_idx), ("topk_weights", topk_weights)):
            if tensor.device != input_.device:
                raise ValueError(f"{name} must be on {input_.device}, got {tensor.device}.")

        recv_tokens = kwargs.get("recv_tokens")
        recv_topk_weights = kwargs.get("recv_topk_weights")
        if self.buffer.eager and (recv_tokens is not None or recv_topk_weights is not None):
            raise ValueError(
                "eager mode sizes dispatch outputs per step and cannot use "
                "caller-supplied receive buffers"
            )

        tokens_per_expert = ep_prepare(self.buffer, topk_idx)
        rows = (
            self.buffer._host_total_recv_tokens
            if self.buffer.eager
            else self.buffer.recv_capacity_per_rank
        )
        if rows is None:
            raise RuntimeError("NCCL EP dispatch receive size is unavailable.")
        rows = int(rows)
        recv_shape = (rows, self.buffer.hidden_dim)
        if input_quantizer is None:
            recv_tokens = _validate_output_buffer(
                "recv_tokens",
                recv_tokens,
                shape=recv_shape,
                dtype=self.buffer.payload_dtype,
                device=self.buffer.device,
            )
        recv_topk_weights = _validate_output_buffer(
            "recv_topk_weights",
            recv_topk_weights,
            shape=(rows,),
            dtype=torch.float32,
            device=self.buffer.device,
        )
        if recv_topk_weights is None:
            recv_topk_weights = _alloc_io(
                (rows,),
                torch.float32,
                self.buffer.device,
                self.buffer.zero_copy,
            )

        if input_quantizer is None:
            if recv_tokens is None:
                recv_tokens = _alloc_io(
                    recv_shape,
                    self.buffer.payload_dtype,
                    self.buffer.device,
                    self.buffer.zero_copy,
                )
            torch.ops.transformer_engine_ep.dispatch(
                self.buffer.handle_mem,
                topk_idx,
                input_,
                topk_weights,
                recv_tokens,
                recv_topk_weights,
            )
        else:
            quantized_input, input_scale_inv = quantize_mxfp8_for_ep(input_, input_quantizer)
            scale_cols = self.buffer.hidden_dim // MXFP8_BLOCK_SCALING_SIZE
            recv_data, recv_scale_inv = _scale_alloc_io(
                recv_tokens,
                rows,
                self.buffer.hidden_dim,
                scale_cols,
                quantized_input.rowwise_data.dtype,
                input_scale_inv.dtype,
                self.buffer.device,
                self.buffer.zero_copy,
            )
            torch.ops.transformer_engine_ep.dispatch(
                self.buffer.handle_mem,
                topk_idx,
                quantized_input.rowwise_data.view(torch.float8_e4m3fn),
                topk_weights,
                recv_data.view(torch.float8_e4m3fn),
                recv_topk_weights,
                input_scale_inv,
                recv_scale_inv,
            )
            recv_tokens = _make_grouped_mxfp8(
                recv_data,
                recv_scale_inv,
                tokens_per_expert,
                quantized_input._fp8_dtype,
                input_.dtype,
            )

        ctx = basic_op_ctxs[0]
        if ctx.requires_grad:
            ctx.input_shape = tuple(input_.shape)
            ctx.input_dtype = input_.dtype
            ctx.topk_weights_shape = tuple(topk_weights.shape)
            ctx.prev_op_grad_output_quantizer = prev_op_grad_output_quantizer
            ctx.save_for_backward(self.buffer.handle_mem)

        return recv_tokens, [(tokens_per_expert, recv_topk_weights, self.buffer.handle_mem)]

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
        ctx = basic_op_ctxs[0]
        (handle_mem,) = ctx.saved_tensors
        grad_output = maybe_dequantize(grad_output, ctx.input_dtype).contiguous()

        grad_recv_weights = basic_op_grad_extra_outputs[0][1]
        if grad_recv_weights is None:
            grad_recv_weights = torch.zeros(
                grad_output.shape[0],
                dtype=torch.float32,
                device=grad_output.device,
            )
        else:
            grad_recv_weights = grad_recv_weights.to(dtype=torch.float32).contiguous()

        grad_input = torch.empty(
            ctx.input_shape,
            dtype=ctx.input_dtype,
            device=grad_output.device,
        )
        grad_topk_weights = torch.empty(
            ctx.topk_weights_shape,
            dtype=torch.float32,
            device=grad_output.device,
        )
        torch.ops.transformer_engine_ep.dispatch_bwd(
            handle_mem,
            grad_output,
            grad_recv_weights,
            grad_input,
            grad_topk_weights,
        )
        quantizer = ctx.prev_op_grad_output_quantizer
        if quantizer is not None and not is_quantized_tensor(grad_input):
            grad_input = quantizer(grad_input)
        return grad_input, [()], [(None, grad_topk_weights)]
