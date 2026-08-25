# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Fusible NCCL expert-parallel dispatch operation."""

from __future__ import annotations

from typing import Any, Iterable, Optional

import torch

from ...ep import (
    EpBuffer,
    _ep_dispatch_bwd,
    _ep_prepare_and_dispatch_fwd,
)
from ...quantization import QuantizerRole
from ...tensor import MXFP8Quantizer, Quantizer
from ...tensor.storage.mxfp8_tensor_storage import MXFP8TensorStorage
from .._common import (
    maybe_dequantize,
    quantize_for_ep,
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
        _validate_dispatch_input(input_, self.buffer)
        _validate_routing_inputs(
            topk_idx,
            topk_weights,
            device=self.buffer.device,
        )
        # Prepare the input
        input_scale_inv = None
        if input_quantizer is None:
            # Only BF16 dispatch is supported for now.
            input_ = maybe_dequantize(input_, torch.bfloat16)
        elif isinstance(input_quantizer, MXFP8Quantizer):
            input_, input_scale_inv = quantize_for_ep(input_, input_quantizer)
        else:
            raise TypeError(
                "NCCL EP Dispatch supports MXFP8Quantizer only, got "
                f"{type(input_quantizer).__name__}."
            )
        # Eager mode discovers the receive size at runtime, so persistent
        # caller-owned output buffers cannot be used.
        recv_tokens = kwargs.get("recv_tokens")
        recv_topk_weights = kwargs.get("recv_topk_weights")
        if self.buffer.eager and (recv_tokens is not None or recv_topk_weights is not None):
            raise ValueError(
                "eager mode sizes dispatch outputs per step and cannot use "
                "caller-supplied receive buffers"
            )
        output, recv_topk_weights, dispatch_state = _ep_prepare_and_dispatch_fwd(
            input_,
            topk_weights,
            topk_idx,
            self.buffer,
            recv_tokens,
            recv_topk_weights,
            input_scale_inv,
        )
        tokens_per_expert = self.buffer.tokens_per_expert
        # If next_op_input_quantizer is different from input_quantizer,
        # we need to requantize the data, which is handled in grouped_linear anyway.
        # We won't get any fusion benefit, so don't do it here.
        ctx = basic_op_ctxs[0]
        if ctx.requires_grad:
            ctx.input_dtype = torch.bfloat16
            ctx.dispatch_state = dispatch_state
            ctx.prev_op_grad_output_quantizer = prev_op_grad_output_quantizer

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

        grad_input, grad_topk_weights = _ep_dispatch_bwd(
            ctx.dispatch_state,
            grad_output,
            grad_recv_weights,
        )
        quantizer = ctx.prev_op_grad_output_quantizer
        if quantizer is not None:
            grad_input = quantizer(grad_input)
        return grad_input, [()], [(None, grad_topk_weights)]
