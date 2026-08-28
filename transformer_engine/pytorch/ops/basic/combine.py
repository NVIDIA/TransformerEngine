# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Fusible NCCL expert-parallel combine operation."""

from __future__ import annotations

from typing import Any, Iterable, Optional

import torch

from ...ep import (
    EpBuffer,
    EpConfig,
    _alloc_io,
    _ep_combine_bwd,
    _ep_combine_fwd,
    is_symm_backed,
    quantize_for_ep,
)
from ...quantization import QuantizerRole
from ...tensor import MXFP8Quantizer, Quantizer
from .._common import (
    maybe_dequantize,
    validate_buffer,
    validate_ep_buffer,
    validate_ep_comms_recipe,
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


class MoeCombine(BasicOperation):
    """Combine pre-weighted local expert outputs with NCCL EP.

    The operation consumes routing state from a runtime :class:`EpBuffer`.
    """

    num_extra_inputs: int = 0

    def __init__(self, config: EpConfig) -> None:
        super().__init__()
        if not isinstance(config, EpConfig):
            raise TypeError(f"config must be an EpConfig, got {type(config).__name__}.")
        self.config = config

    def num_quantizers(self, mode: str) -> int:
        return 1 if mode == "backward" else 0

    def get_quantizer_roles(self, mode: str) -> Optional[list[QuantizerRole]]:
        if mode == "backward":
            # combine backward dispatches grad_output
            name = getattr(self, "name", "") or ""
            return [
                QuantizerRole(
                    module_type="combine",
                    tensor_type="dispatch_grad_output",
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
            quantizer.internal = True

    def op_forward(self, *args: Any, **kwargs: Any) -> None:
        raise RuntimeError("MoeCombine uses fuser_forward")

    def op_backward(self, *args: Any, **kwargs: Any) -> None:
        raise RuntimeError("MoeCombine uses fuser_backward")

    @staticmethod
    def _prepare_grad_buffer(
        grad_out: Optional[torch.Tensor],
        grad_output_quantizer: Optional[Quantizer],
        *,
        input_shape: tuple[int, int],
        input_dtype: torch.dtype,
        device: torch.device,
        zero_copy: bool,
    ) -> Optional[torch.Tensor]:
        """Validate caller storage for the expert-output gradient."""
        if not isinstance(grad_output_quantizer, MXFP8Quantizer):
            grad_out = validate_buffer(
                "grad_out",
                grad_out,
                shape=input_shape,
                dtype=input_dtype,
                device=device,
            )
        else:
            grad_out = validate_buffer(
                "MXFP8 grad_out storage",
                grad_out,
                device=device,
                contiguous=True,
            )
        if zero_copy and grad_out is not None and not is_symm_backed(grad_out):
            raise ValueError("zero-copy MoeCombine grad_out must be symmetric-memory-backed.")
        return grad_out

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
        # Combine's transport format is selected by Combine's own grad-output quantizer.
        # If the preceding op expects a different gradient quantized format,
        # it is requantized in that op's backward implementation (e.g., GroupedLinear backward).
        del basic_op_extra_inputs, prev_op_grad_output_quantizer, next_op_input_quantizer
        grad_output_quantizer = self.get_quantizer("backward", 0)
        transport_quantizer = (
            grad_output_quantizer if isinstance(grad_output_quantizer, MXFP8Quantizer) else None
        )
        kwargs = basic_op_kwargs[0]
        buffer = validate_ep_buffer("MoeCombine", self.config, kwargs.get("buffer"))
        validate_ep_comms_recipe(
            "MoeCombine",
            grad_output_quantizer,
            buffer.combine_bwd_quant_recipe,
        )
        input_shape = _validate_combine_inputs(input_, buffer)
        # Stage zero-copy input if needed, then restore local-token order.
        zero_copy = buffer.zero_copy
        expert_out = input_
        if zero_copy:
            expert_out = _alloc_io(tuple(input_.shape), input_.dtype, input_.device, True)
            expert_out.copy_(input_)
        # Preserve routing state and optional caller storage for backward.
        ctx = basic_op_ctxs[0]
        grad_out = None
        if ctx.requires_grad:
            grad_out = self._prepare_grad_buffer(
                kwargs.get("grad_out"),
                transport_quantizer,
                input_shape=input_shape,
                input_dtype=input_.dtype,
                device=input_.device,
                zero_copy=zero_copy,
            )
            ctx.input_dtype = input_.dtype
        result, combine_state = _ep_combine_fwd(
            expert_out,
            grad_out,
            handle_mem=buffer.handle_mem,
            token_counts=buffer.tokens_per_expert,
            num_local_tokens=buffer.num_local_tokens,
            hidden_dim=expert_out.shape[-1],
            bwd_quant_recipe=transport_quantizer,
            eager=buffer.eager,
            zero_copy=zero_copy,
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
        grad_output_quantizer = self.get_quantizer("backward", 0)
        grad_scale_inv = None
        # Prepare grad_output (Quantize if necessary)
        if isinstance(grad_output_quantizer, MXFP8Quantizer):
            quantized_grad, grad_scale_inv = quantize_for_ep(
                grad_output,
                grad_output_quantizer,
            )
            grad_output = quantized_grad
        else:
            grad_output = maybe_dequantize(grad_output, ctx.input_dtype).contiguous()
            quantized_grad = None
        grad_input = _ep_combine_bwd(
            ctx.combine_state,
            grad_output,
            quantized_grad,
            grad_scale_inv,
        )
        return grad_input, [()], [()]
