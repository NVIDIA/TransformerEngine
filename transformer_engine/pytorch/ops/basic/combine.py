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
from .moe_local import combine_backward, combine_forward, make_token_order, validate_local_config


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

    The communication backend is selected by the ``buffer`` passed to the
    constructor: an ``EpBuffer`` combines with NCCL EP, and no buffer selects the
    PyTorch backend, which requires every expert to be local (EP=1). The
    quantization format of the communication is an EP backend detail, so the
    backend configures it rather than this operation's quantizers.

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
        if buffer is None:
            validate_local_config(config)
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
        del (
            prev_op_grad_output_quantizer,
            next_op_input_quantizer,
            basic_op_kwargs,
        )
        (topk_idx,) = basic_op_extra_inputs[0]
        # Only BF16 combine forward is supported for now.
        input_ = maybe_dequantize(input_, torch.bfloat16)
        ctx = basic_op_ctxs[0]

        if self.buffer is None:
            # PyTorch backend: the expert outputs are already local, so combine
            # only needs the expert-major order to sum them into their tokens.
            token_index = make_token_order(topk_idx, self.config)
            output = combine_forward(input_, token_index, topk_idx.shape[0])
            if ctx.requires_grad:
                ctx.token_index = token_index
            return output, [()]

        buffer = validate_ep_buffer("MoeCombine", self.config, self.buffer)
        _validate_combine_inputs(input_, buffer)
        result, combine_state = _ep_combine_fwd(
            input_,
            None,
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
        if self.buffer is None:
            grad_input = combine_backward(ctx.token_index, grad_output)
        else:
            grad_input = _ep_combine_bwd(ctx.combine_state, grad_output)
        return grad_input, [()], [(None,)]
