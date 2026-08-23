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
from ...tensor.storage.mxfp8_tensor_storage import MXFP8TensorStorage
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
    if tensor.requires_grad:
        raise ValueError(f"{name} must not require gradients.")
    return tensor


def _validate_dispatch_input(
    input_: torch.Tensor | MXFP8TensorStorage,
    buffer: EpBuffer,
) -> tuple[tuple[int, int], torch.dtype, bool]:
    """Validate the local token matrix and identify its transport format."""
    is_mxfp8 = isinstance(input_, MXFP8TensorStorage)
    if is_quantized_tensor(input_) and not is_mxfp8:
        raise TypeError(
            "NCCL EP Dispatch supports BF16 and MXFP8 inputs, "
            f"got {type(input_).__name__}."
        )

    input_shape = tuple(input_.shape)
    input_dtype = input_.dtype if isinstance(input_, torch.Tensor) else input_._dtype
    expected_hidden = buffer.hidden_dim
    if len(input_shape) != 2 or input_shape[-1] != expected_hidden:
        raise ValueError(
            f"Dispatch input must have shape (T, {expected_hidden}), got {input_shape}."
        )
    if input_.device != buffer.device:
        raise ValueError(f"Dispatch input must be on {buffer.device}, got {input_.device}.")
    if input_dtype is not torch.bfloat16:
        raise TypeError(
            "Dispatch input must be BF16 or an MXFP8 tensor representing BF16 values, "
            f"got logical dtype {input_dtype}."
        )

    if is_mxfp8:
        if input_._fp8_dtype != DType.kFloat8E4M3:
            raise NotImplementedError(
                f"NCCL EP Dispatch supports E4M3 MXFP8 only, got {input_._fp8_dtype}."
            )

    return input_shape, input_dtype, is_mxfp8


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


def _validate_mxfp8_output_buffer(
    tensor: Optional[torch.Tensor],
    *,
    device: torch.device,
) -> None:
    """Validate properties common to packed MXFP8 caller buffers.

    ``_scale_alloc_io`` validates contiguity and byte capacity after the data
    and scale sizes are known.
    """
    if tensor is None:
        return
    if tensor.device != device:
        raise ValueError(f"recv_tokens must be on {device}, got {tensor.device}.")
    if tensor.requires_grad:
        raise ValueError("recv_tokens must not require gradients.")


class Dispatch(BasicOperation):
    """Dispatch BF16 or MXFP8 tokens to local experts with NCCL EP.

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

    def _prepare_routing(self, topk_idx: torch.Tensor) -> tuple[torch.Tensor, int]:
        """Prepare NCCL routing state and return the receive row count."""
        tokens_per_expert = ep_prepare(self.buffer, topk_idx)
        num_recv_tokens = (
            self.buffer._host_total_recv_tokens
            if self.buffer.eager
            else self.buffer.recv_capacity_per_rank
        )
        if num_recv_tokens is None:
            raise RuntimeError("NCCL EP dispatch receive size is unavailable.")
        return tokens_per_expert, int(num_recv_tokens)

    def _prepare_output_buffers(
        self,
        recv_tokens: Optional[torch.Tensor],
        recv_topk_weights: Optional[torch.Tensor],
        *,
        num_recv_tokens: int,
        use_mxfp8: bool,
    ) -> tuple[Optional[torch.Tensor], torch.Tensor]:
        """Validate caller buffers and allocate any missing dispatch outputs."""
        recv_shape = (num_recv_tokens, self.buffer.hidden_dim)
        if use_mxfp8:
            _validate_mxfp8_output_buffer(recv_tokens, device=self.buffer.device)
        else:
            recv_tokens = _validate_output_buffer(
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

        recv_topk_weights = _validate_output_buffer(
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
        return recv_tokens, recv_topk_weights

    def _dispatch_impl(
        self,
        input_: torch.Tensor | MXFP8TensorStorage,
        input_dtype: torch.dtype,
        input_quantizer: Optional[MXFP8Quantizer],
        topk_idx: torch.Tensor,
        topk_weights: torch.Tensor,
        tokens_per_expert: torch.Tensor,
        recv_tokens: Optional[torch.Tensor],
        recv_topk_weights: torch.Tensor,
        *,
        num_recv_tokens: int,
        use_mxfp8: bool,
    ) -> torch.Tensor | MXFP8TensorStorage:
        """Route tokens in the selected transport format.

        Keep transport-specific setup in this function so future quantized
        formats can be added without branching in ``fuser_forward``.
        """
        if not use_mxfp8:
            if recv_tokens is None:
                raise RuntimeError("BF16 dispatch receive storage was not allocated.")
            torch.ops.transformer_engine_ep.dispatch(
                self.buffer.handle_mem,
                topk_idx,
                input_,
                topk_weights,
                recv_tokens,
                recv_topk_weights,
            )
            return recv_tokens

        quantized_input, input_scale_inv = quantize_mxfp8_for_ep(input_, input_quantizer)
        scale_cols = self.buffer.hidden_dim // MXFP8_BLOCK_SCALING_SIZE
        recv_data, recv_scale_inv = _scale_alloc_io(
            recv_tokens,
            num_recv_tokens,
            self.buffer.hidden_dim,
            scale_cols,
            quantized_input._rowwise_data.dtype,
            input_scale_inv.dtype,
            self.buffer.device,
            self.buffer.zero_copy,
        )
        torch.ops.transformer_engine_ep.dispatch(
            self.buffer.handle_mem,
            topk_idx,
            quantized_input._rowwise_data.view(torch.float8_e4m3fn),
            topk_weights,
            recv_data.view(torch.float8_e4m3fn),
            recv_topk_weights,
            input_scale_inv,
            recv_scale_inv,
        )
        return _make_grouped_mxfp8(
            recv_data,
            recv_scale_inv,
            tokens_per_expert,
            quantized_input._fp8_dtype,
            input_dtype,
        )

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
        # Resolve the transport format. A BF16 input uses MXFP8 transport when
        # an input quantizer is active; an existing MXFP8 input is passed through.
        input_quantizer = self._resolve_input_quantizer(next_op_input_quantizer)
        topk_idx, topk_weights = basic_op_extra_inputs[0]
        kwargs = basic_op_kwargs[0]
        input_shape, input_dtype, input_is_mxfp8 = _validate_dispatch_input(input_, self.buffer)
        use_mxfp8 = input_is_mxfp8 or input_quantizer is not None
        _validate_routing_inputs(
            topk_idx,
            topk_weights,
            device=self.buffer.device,
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

        # Prepare the routing handle, then size and allocate the receive outputs.
        tokens_per_expert, num_recv_tokens = self._prepare_routing(topk_idx)
        recv_tokens, recv_topk_weights = self._prepare_output_buffers(
            recv_tokens,
            recv_topk_weights,
            num_recv_tokens=num_recv_tokens,
            use_mxfp8=use_mxfp8,
        )

        # Launch NCCL EP using the selected transport representation.
        output = self._dispatch_impl(
            input_,
            input_dtype,
            input_quantizer,
            topk_idx,
            topk_weights,
            tokens_per_expert,
            recv_tokens,
            recv_topk_weights,
            num_recv_tokens=num_recv_tokens,
            use_mxfp8=use_mxfp8,
        )

        # Save only shape/dtype metadata and the opaque routing handle for backward.
        ctx = basic_op_ctxs[0]
        if ctx.requires_grad:
            ctx.input_shape = input_shape
            ctx.input_dtype = input_dtype
            ctx.topk_weights_shape = tuple(topk_weights.shape)
            ctx.prev_op_grad_output_quantizer = prev_op_grad_output_quantizer
            ctx.save_for_backward(self.buffer.handle_mem)

        return output, [
            (tokens_per_expert, recv_topk_weights, self.buffer.handle_mem, topk_idx)
        ]

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
