# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Fusible expert-parallel dispatch operation."""

from __future__ import annotations

from typing import Any, Iterable, Optional

import torch

from ...ep import (
    EpBuffer,
    EpConfig,
    _ep_dispatch_bwd,
    _ep_prepare_and_dispatch_fwd,
)
from ...tensor import Quantizer
from .._common import (
    is_quantized_tensor,
    maybe_dequantize,
    validate_ep_buffer,
)
from ..op import BasicOperation, OperationContext
from .moe_local import dispatch_backward, dispatch_forward, make_routing_plan, validate_local_config


def _validate_dispatch_input(
    input_: torch.Tensor,
    hidden_dim: int,
    device: torch.device,
) -> tuple[int, int]:
    """Validate the local token matrix."""
    if (
        not isinstance(input_, torch.Tensor)
        or is_quantized_tensor(input_)
        or input_.dtype is not torch.bfloat16
    ):
        raise TypeError(
            f"MoeDispatch input must be a plain BF16 tensor, got {type(input_).__name__}."
        )
    input_shape = tuple(input_.shape)
    if len(input_shape) != 2 or input_shape[-1] != hidden_dim:
        raise ValueError(f"MoeDispatch input must have shape (T, {hidden_dim}), got {input_shape}.")
    if input_.device != device:
        raise ValueError(
            "MoeDispatch input and routing metadata must share a device: input is on "
            f"{input_.device}, routing is on {device}."
        )
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
    """Route tokens to experts and distribute them across expert-parallel ranks

    The extra inputs are routing indices and FP32 routing weights. The extra
    outputs are local tokens-per-expert and received routing weights.

    The communication backend is selected by the ``buffer`` passed to the
    constructor: an ``EpBuffer`` dispatches with NCCL EP, and no buffer selects
    the PyTorch backend, which requires every expert to be local (EP=1). The
    quantization format of the communication is an EP backend detail, so the
    backend configures it rather than this operation's quantizers.

    """

    num_extra_inputs: int = 2
    # tokens-per-expert and received routing weights consumed by the expert MLP.
    num_extra_outputs: int = 2

    def __init__(self, config: EpConfig, buffer: Optional[EpBuffer] = None) -> None:
        # EpBuffer is specific to NCCL EP. Fused implementations using another communication
        # backend, such as NVSHMEM, do not need it.
        super().__init__()
        if not isinstance(config, EpConfig):
            raise TypeError(f"config must be an EpConfig, got {type(config).__name__}.")
        if config.zero_copy:
            raise NotImplementedError("MoeDispatch does not support zero-copy EP.")
        if buffer is None:
            validate_local_config(config)
        self.config = config
        self.buffer = buffer

    def op_forward(self, *args: Any, **kwargs: Any) -> None:
        raise RuntimeError("MoeDispatch uses fuser_forward")

    def op_backward(self, *args: Any, **kwargs: Any) -> None:
        raise RuntimeError("MoeDispatch uses fuser_backward")

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
        del next_op_input_quantizer, basic_op_kwargs
        topk_idx, topk_weights = basic_op_extra_inputs[0]
        ctx = basic_op_ctxs[0]

        if self.buffer is None:
            # PyTorch backend: tokens stay on this rank, so dispatch only sorts
            # them into the expert-major order that the expert MLP consumes.
            _validate_dispatch_input(input_, self.config.hidden_dim, topk_idx.device)
            plan = make_routing_plan(topk_idx, topk_weights, self.config)
            recv_tokens, recv_topk_weights = dispatch_forward(input_, plan)
            if ctx.requires_grad:
                ctx.routing_plan = plan
            return recv_tokens, [(plan.tokens_per_expert, recv_topk_weights)]

        buffer = validate_ep_buffer("MoeDispatch", self.config, self.buffer)
        input_shape = _validate_dispatch_input(input_, buffer.hidden_dim, buffer.device)
        buffer.num_local_tokens = input_shape[0]
        _validate_routing_inputs(
            topk_idx,
            topk_weights,
            device=buffer.device,
        )
        output, recv_topk_weights, dispatch_state = _ep_prepare_and_dispatch_fwd(
            input_,
            topk_weights,
            topk_idx,
            buffer,
            None,
            None,
        )
        tokens_per_expert = buffer.tokens_per_expert
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

        if self.buffer is None:
            grad_input, grad_topk_weights = dispatch_backward(
                ctx.routing_plan,
                grad_output,
                grad_recv_weights,
            )
        else:
            grad_input, grad_topk_weights = _ep_dispatch_bwd(
                ctx.dispatch_state,
                grad_output,
                grad_recv_weights,
            )
        return grad_input, [()], [(None, grad_topk_weights)]
