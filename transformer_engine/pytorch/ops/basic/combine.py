# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Fusible expert-parallel combine operation."""

from __future__ import annotations

from typing import Any, Iterable, Optional

import torch

from ...ep import (
    EpBuffer,
    EpConfig,
    _ep_combine_bwd,
    _ep_combine_fwd,
)
from ...tensor import Quantizer
from .._common import (
    is_quantized_tensor,
    maybe_dequantize,
    validate_ep_buffer,
)
from ..op import BasicOperation, OperationContext


def _validate_combine_inputs(
    input_: torch.Tensor,
    buffer: EpBuffer,
) -> tuple[int, int]:
    """Validate the expert output and routing metadata consumed by MoeCombine."""
    if input_.dtype is not torch.bfloat16:
        raise NotImplementedError(f"NCCL EP requires BF16 combine input, got {input_.dtype}.")
    if input_.ndim != 2:
        raise ValueError(f"MoeCombine input must be 2D, got shape {tuple(input_.shape)}.")
    if buffer.handle_mem.dtype is not torch.uint8 or buffer.handle_mem.device != input_.device:
        raise ValueError("MoeCombine routing handle must be a uint8 tensor on the input device.")
    if (
        buffer.tokens_per_expert.dtype is not torch.int64
        or buffer.tokens_per_expert.device != input_.device
    ):
        raise ValueError(
            "MoeCombine tokens_per_expert must be an int64 tensor on the input device."
        )
    return tuple(input_.shape)


def _validate_combine_grad_output(grad_output: torch.Tensor) -> None:
    """Validate the high-precision gradient combined back to the source tokens."""
    if (
        not isinstance(grad_output, torch.Tensor)
        or is_quantized_tensor(grad_output)
        or grad_output.dtype is not torch.bfloat16
    ):
        raise TypeError(
            "MoeCombine grad_output must be a plain BF16 tensor, "
            f"got {type(grad_output).__name__} with "
            f"dtype={getattr(grad_output, 'dtype', None)}."
        )


class MoeCombine(BasicOperation):
    """Combine pre-weighted expert outputs and return them to their source tokens

    The extra input is the routing index tensor that ``MoeDispatch`` consumes.
    NCCL EP carries the routing state in its ``EpBuffer`` and ignores it, while a
    backend that holds no communication state reads the expert-major order back
    out of it.

    The quantization format of the communication is an EP backend detail, so it
    is configured by ``EpConfig`` rather than this operation's quantizers.
    """

    num_extra_inputs: int = 1

    def __init__(self, config: EpConfig, buffer: Optional[EpBuffer] = None) -> None:
        # EpBuffer is specific to NCCL EP. Fused implementations using another communication
        # backend, such as NVSHMEM, do not need it.
        super().__init__()
        if not isinstance(config, EpConfig):
            raise TypeError(f"config must be an EpConfig, got {type(config).__name__}.")
        if config.zero_copy:
            raise NotImplementedError("MoeCombine does not support zero-copy EP.")
        self.config = config
        self.buffer = buffer

    def op_forward(self, *args: Any, **kwargs: Any) -> None:
        raise RuntimeError("MoeCombine uses fuser_forward")

    def op_backward(self, *args: Any, **kwargs: Any) -> None:
        raise RuntimeError("MoeCombine uses fuser_backward")

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
        # NCCL EP reads the routing state from the EpBuffer, so topk_idx is unused.
        del basic_op_extra_inputs, prev_op_grad_output_quantizer, next_op_input_quantizer
        # Only BF16 combine forward is supported for now.
        input_ = maybe_dequantize(input_, torch.bfloat16)
        ctx = basic_op_ctxs[0]
        kwargs = basic_op_kwargs[0]
        buffer = validate_ep_buffer("MoeCombine", self.config, kwargs.get("buffer", self.buffer))
        grad_out = kwargs.get("grad_out")
        if buffer.eager and grad_out is not None:
            raise ValueError("MoeCombine eager mode cannot use caller-supplied grad_out.")
        _validate_combine_inputs(input_, buffer)
        result, combine_state = _ep_combine_fwd(
            input_,
            grad_out,
            buffer,
            buffer.num_local_tokens,
            buffer.combine_bwd_quant_recipe,
        )
        if ctx.requires_grad:
            ctx.combine_state = combine_state

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
        _validate_combine_grad_output(grad_output)
        grad_output = grad_output.contiguous()
        grad_input = _ep_combine_bwd(ctx.combine_state, grad_output)
        return grad_input, [()], [(None,)]
