# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Fusible NCCL expert-parallel dispatch operation."""

from __future__ import annotations

from typing import Any, Iterable, Optional

import torch

from ...ep import (
    EpBuffer,
    EpConfig,
    _ep_dispatch_bwd,
    _ep_prepare_and_dispatch_fwd,
    quantize_for_ep,
)
from ...quantization import QuantizerRole
from ...tensor import MXFP8Quantizer, Quantizer
from ...tensor.storage.mxfp8_tensor_storage import MXFP8TensorStorage
from .._common import (
    maybe_dequantize,
    validate_ep_buffer,
    validate_ep_comms_recipe,
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
            f"MoeDispatch input must have shape (T, {buffer.hidden_dim}), got {input_shape}."
        )
    if input_.device != buffer.device:
        raise ValueError(f"MoeDispatch input must be on {buffer.device}, got {input_.device}.")
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


class MoeDispatch(BasicOperation):
    """Dispatch floating-point or MXFP8 tokens to local experts with NCCL EP.

    The extra inputs are routing indices and FP32 routing weights. The extra
    outputs are local tokens-per-expert and received routing weights.
    """

    num_extra_inputs: int = 2
    # tokens-per-expert and received routing weights consumed by the expert MLP.
    num_extra_outputs: int = 2

    def __init__(self, config: EpConfig) -> None:
        super().__init__()
        if not isinstance(config, EpConfig):
            raise TypeError(f"config must be an EpConfig, got {type(config).__name__}.")
        if config.zero_copy:
            raise NotImplementedError("MoeDispatch does not support zero-copy EP.")
        self.config = config

    def num_quantizers(self, mode: str) -> int:
        # quantized dispatch_bwd/combine is not supported.
        return 1 if mode == "forward" else 0

    def get_quantizer_roles(self, mode: str) -> Optional[list[QuantizerRole]]:
        if mode == "forward":
            name = getattr(self, "name", "") or ""
            return [
                QuantizerRole(
                    module_type="dispatch",
                    tensor_type="dispatch_input",
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
        raise RuntimeError("MoeDispatch uses fuser_forward")

    def op_backward(self, *args: Any, **kwargs: Any) -> None:
        raise RuntimeError("MoeDispatch uses fuser_backward")

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
        del next_op_input_quantizer
        # Dispatch uses unquantized transport without an input quantizer and
        # MXFP8 transport with an MXFP8 input quantizer.
        input_quantizer = self.get_quantizer("forward", 0)
        topk_idx, topk_weights = basic_op_extra_inputs[0]
        kwargs = basic_op_kwargs[0]
        buffer = validate_ep_buffer("MoeDispatch", self.config, kwargs.get("buffer"))
        validate_ep_comms_recipe(
            "MoeDispatch",
            input_quantizer,
            buffer.dispatch_fwd_quant_recipe,
        )
        input_shape = _validate_dispatch_input(input_, buffer)
        buffer.num_local_tokens = input_shape[0]
        _validate_routing_inputs(
            topk_idx,
            topk_weights,
            device=buffer.device,
        )
        # Prepare the input
        input_scale_inv = None
        if isinstance(input_quantizer, MXFP8Quantizer):
            input_, input_scale_inv = quantize_for_ep(input_, input_quantizer)
        else:
            # Only BF16 dispatch is supported for now.
            input_ = maybe_dequantize(input_, torch.bfloat16)
        output, recv_topk_weights, dispatch_state = _ep_prepare_and_dispatch_fwd(
            input_,
            topk_weights,
            topk_idx,
            buffer,
            None,
            None,
            input_scale_inv,
        )
        tokens_per_expert = buffer.tokens_per_expert
        # If next_op_input_quantizer is different from input_quantizer,
        # we need to requantize the data, which is handled in grouped_linear anyway.
        # We won't get any fusion benefit, so don't do it here.
        ctx = basic_op_ctxs[0]
        if ctx.requires_grad:
            ctx.dispatch_state = dispatch_state
            ctx.prev_op_grad_output_quantizer = prev_op_grad_output_quantizer

        return output, [(tokens_per_expert, recv_topk_weights)]

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
        grad_output = maybe_dequantize(grad_output, torch.bfloat16)

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
        return grad_input, [()], [(None, grad_topk_weights)]
