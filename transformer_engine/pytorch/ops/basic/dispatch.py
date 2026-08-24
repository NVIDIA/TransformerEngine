# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Fusible NCCL expert-parallel dispatch operation."""

from __future__ import annotations

from typing import Any, Iterable, Optional

import torch

from ...constants import MXFP8_BLOCK_SCALING_SIZE
from ...ep import (
    EpBuffer,
    _alloc_io,
    _make_grouped_mxfp8,
    _scale_alloc_io,
    ep_prepare,
)
from ...quantization import QuantizerRole, Recipe
from ...tensor import MXFP8Quantizer, Quantizer
from ...tensor.storage.mxfp8_tensor_storage import MXFP8TensorStorage
from .._common import (
    maybe_dequantize,
    quantize_for_ep,
    validate_buffer,
)
from ..op import BasicOperation, OperationContext


def _validate_dispatch_input(
    input_: torch.Tensor | MXFP8TensorStorage,
    buffer: EpBuffer,
) -> tuple[int, int]:
    """Validate the local token matrix."""
    input_shape = tuple(input_.shape)
    if len(input_shape) != 2 or input_shape[-1] != buffer.hidden_dim:
        raise ValueError(
            f"Dispatch input must have shape (T, {buffer.hidden_dim}), got {input_shape}."
        )
    if input_.device != buffer.device:
        raise ValueError(f"Dispatch input must be on {buffer.device}, got {input_.device}.")
    return input_shape


def _validate_routing_inputs(
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    *,
    device: torch.device,
) -> None:
    """Validate routing properties not checked by the native binding."""
    if topk_weights.dtype is not torch.float32:
        raise TypeError(f"topk_weights must be float32, got {topk_weights.dtype}.")
    for name, tensor in (("topk_idx", topk_idx), ("topk_weights", topk_weights)):
        if tensor.device != device:
            raise ValueError(f"{name} must be on {device}, got {tensor.device}.")


class Dispatch(BasicOperation):
    """Dispatch floating-point or MXFP8 tokens to local experts with NCCL EP.

    The extra inputs are routing indices and FP32 routing weights. The extra
    outputs are local tokens-per-expert, received routing weights, the routing
    handle, and routing indices for recovering the local token shape.
    """

    num_extra_inputs: int = 2
    # tokens-per-expert, received routing weights, and the opaque NCCL EP
    # routing handle and routing indices consumed by Combine.
    num_extra_outputs: int = 4

    def __init__(self, buffer: EpBuffer) -> None:
        super().__init__()
        self.buffer = buffer

    def num_quantizers(self, mode: str) -> int:
        # quantized dispatch_bwd/combine is not supported.
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
            # We just need data, scales for dispatch, and grouped tensor
            # will be recreated after dispatch op.
            quantizer.set_usage(rowwise=True, columnwise=False)
            quantizer.optimize_for_gemm = False
            quantizer.internal = True

    def op_forward(self, *args: Any, **kwargs: Any) -> None:
        raise RuntimeError("Dispatch uses fuser_forward")

    def op_backward(self, *args: Any, **kwargs: Any) -> None:
        raise RuntimeError("Dispatch uses fuser_backward")

    def _prepare_routing(self, topk_idx: torch.Tensor) -> tuple[torch.Tensor, int]:
        """Prepare NCCL routing state and return the receive row count."""
        tokens_per_expert = ep_prepare(self.buffer, topk_idx)
        num_recv_tokens = (
            self.buffer._host_total_recv_tokens
            if self.buffer.eager
            else self.buffer.recv_capacity_per_rank
        )
        return tokens_per_expert, int(num_recv_tokens)

    def _prepare_output_buffers(
        self,
        recv_tokens: Optional[torch.Tensor],
        recv_topk_weights: Optional[torch.Tensor],
        quantized_input: Optional[MXFP8TensorStorage],
        input_scale_inv: Optional[torch.Tensor],
        *,
        num_recv_tokens: int,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        """Validate caller buffers and allocate any missing dispatch outputs."""
        recv_shape = (num_recv_tokens, self.buffer.hidden_dim)
        recv_scale_inv = None
        if quantized_input is not None:
            recv_tokens = validate_buffer(
                "recv_tokens",
                recv_tokens,
                device=self.buffer.device,
            )
            scale_cols = self.buffer.hidden_dim // MXFP8_BLOCK_SCALING_SIZE
            if scale_cols * input_scale_inv.element_size() % 16:
                raise ValueError(
                    "MXFP8 NCCL EP transport requires hidden size divisible by "
                    f"{16 * MXFP8_BLOCK_SCALING_SIZE}, got {self.buffer.hidden_dim}."
                )
            recv_tokens, recv_scale_inv = _scale_alloc_io(
                recv_tokens,
                num_recv_tokens,
                self.buffer.hidden_dim,
                scale_cols,
                quantized_input._rowwise_data.dtype,
                input_scale_inv.dtype,
                self.buffer.device,
                self.buffer.zero_copy,
            )
        else:
            recv_tokens = validate_buffer(
                "recv_tokens",
                recv_tokens,
                shape=recv_shape,
                dtype=self.buffer.payload_dtype,
                device=self.buffer.device,
            )
            if recv_tokens is None:
                recv_tokens = _alloc_io(
                    recv_shape,
                    self.buffer.payload_dtype,
                    self.buffer.device,
                    self.buffer.zero_copy,
                )

        recv_topk_weights = validate_buffer(
            "recv_topk_weights",
            recv_topk_weights,
            shape=(num_recv_tokens,),
            dtype=torch.float32,
            device=self.buffer.device,
        )
        if recv_topk_weights is None:
            recv_topk_weights = _alloc_io(
                (num_recv_tokens,),
                torch.float32,
                self.buffer.device,
                self.buffer.zero_copy,
            )
        return recv_tokens, recv_scale_inv, recv_topk_weights

    def fuser_forward(
        self,
        basic_op_ctxs: list[OperationContext],
        input_: torch.Tensor | MXFP8TensorStorage,
        *,
        basic_op_extra_inputs: list[tuple[torch.Tensor, ...]],
        prev_op_grad_output_quantizer: Optional[Quantizer],
        next_op_input_quantizer: Optional[Quantizer],
        basic_op_kwargs: list[dict[str, Any]],
    ) -> tuple[torch.Tensor, Iterable[Iterable[torch.Tensor]]]:

        # Dispatch uses unquantized transport without an input quantizer and
        # MXFP8 transport with an MXFP8 input quantizer.
        input_quantizer = self.get_quantizer("forward", 0)
        topk_idx, topk_weights = basic_op_extra_inputs[0]
        kwargs = basic_op_kwargs[0]
        input_shape = _validate_dispatch_input(input_, self.buffer)
        _validate_routing_inputs(
            topk_idx,
            topk_weights,
            device=self.buffer.device,
        )
        # Prepare the routing handle, then size and allocate the receive outputs.
        tokens_per_expert, num_recv_tokens = self._prepare_routing(topk_idx)
        # Prepare the input
        input_scale_inv = None
        if input_quantizer is None:
            # Only BF16 dispatch is supported for now.
            input_ = maybe_dequantize(input_, torch.bfloat16)
            quantized_input = None
        elif isinstance(input_quantizer, MXFP8Quantizer):
            quantized_input, input_scale_inv = quantize_for_ep(input_, input_quantizer)
            input_ = quantized_input
        else:
            raise TypeError(
                "NCCL EP Dispatch supports MXFP8Quantizer only, got "
                f"{type(input_quantizer).__name__}."
            )
        # Prepare the output buffers
        # Eager mode discovers the receive size at runtime, so persistent
        # caller-owned output buffers cannot be used.
        recv_tokens = kwargs.get("recv_tokens")
        recv_topk_weights = kwargs.get("recv_topk_weights")
        if self.buffer.eager and (recv_tokens is not None or recv_topk_weights is not None):
            raise ValueError(
                "eager mode sizes dispatch outputs per step and cannot use "
                "caller-supplied receive buffers"
            )
        recv_tokens, recv_scale_inv, recv_topk_weights = self._prepare_output_buffers(
            recv_tokens,
            recv_topk_weights,
            quantized_input,
            input_scale_inv,
            num_recv_tokens=num_recv_tokens,
        )
        # Launch NCCL EP using the selected transport representation.
        if quantized_input is None:
            torch.ops.transformer_engine_ep.dispatch(
                self.buffer.handle_mem,
                topk_idx,
                input_,
                topk_weights,
                recv_tokens,
                recv_topk_weights,
            )
            output = recv_tokens
        else:
            torch.ops.transformer_engine_ep.dispatch(
                self.buffer.handle_mem,
                topk_idx,
                quantized_input._rowwise_data.view(torch.float8_e4m3fn),
                topk_weights,
                recv_tokens.view(torch.float8_e4m3fn),
                recv_topk_weights,
                input_scale_inv,
                recv_scale_inv,
            )
            output = _make_grouped_mxfp8(
                recv_tokens,
                recv_scale_inv,
                tokens_per_expert,
                quantized_input._fp8_dtype,
                torch.bfloat16,
            )
        # If next_op_input_quantizer is different from input_quantizer,
        # we need to requantize the data, which is handled in grouped_linear anyway.
        # We won't get any fusion benefit, so don't do it here.
        # Save only shape/dtype metadata and the opaque routing handle for backward.
        ctx = basic_op_ctxs[0]
        if ctx.requires_grad:
            ctx.input_shape = input_shape
            ctx.input_dtype = torch.bfloat16
            ctx.topk_weights_shape = tuple(topk_weights.shape)
            ctx.prev_op_grad_output_quantizer = prev_op_grad_output_quantizer
            ctx.save_for_backward(self.buffer.handle_mem)

        return output, [(tokens_per_expert, recv_topk_weights, self.buffer.handle_mem, topk_idx)]

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
        # Only BF16 Dispatch_bwd is supported for now.
        grad_output = maybe_dequantize(grad_output, ctx.input_dtype)

        grad_recv_weights = basic_op_grad_extra_outputs[0][1]
        if grad_recv_weights is None:
            grad_recv_weights = torch.zeros(
                grad_output.shape[0],
                dtype=torch.float32,
                device=grad_output.device,
            )
        else:
            grad_recv_weights = grad_recv_weights.to(dtype=torch.float32)

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
        if quantizer is not None:
            grad_input = quantizer(grad_input)
        return grad_input, [()], [(None, grad_topk_weights)]
