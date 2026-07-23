# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Fused scaled activation + grouped linear backward."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Optional

import torch

import transformer_engine_torch as tex
from ...quantization import Recipe
from ...tensor import Quantizer
from ...utils import clear_tensor_data
from .._common import maybe_dequantize
from ..basic import GroupedLinear, ScaledClampedQGeGLU, ScaledSReLU, ScaledSwiGLU
from ..op import FusedOperation, FusibleOperation, OperationContext
from .forward_activation_grouped_linear import (
    _SCALED_ACTIVATION_TYPES,
    _ScaledActivation,
    act_grouped_linear_fusion_supported,
)


def _grouped_scaled_dactivation(
    activation: _ScaledActivation,
    grad_output: torch.Tensor,
    input_: torch.Tensor,
    scales: torch.Tensor,
    *,
    quantizer: Quantizer,
    num_groups: int,
    split_sizes: torch.Tensor,
    tensor_offsets: torch.Tensor,
    compute_scale_grad: bool,
) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Dispatch a grouped scaled activation backward pass."""
    dy = grad_output.reshape(-1, grad_output.size(-1))
    x = input_.reshape(-1, input_.size(-1))
    s = scales.reshape(-1)
    if isinstance(activation, ScaledSwiGLU):
        return tex.grouped_scaled_dswiglu(
            dy,
            x,
            s,
            quantizer,
            num_groups,
            split_sizes,
            tensor_offsets,
            int(activation.glu_interleave_size or 0),
            compute_scale_grad,
        )
    if isinstance(activation, ScaledClampedQGeGLU):
        clamped = activation._clamped
        return tex.grouped_scaled_clamped_dswiglu(
            dy,
            x,
            s,
            quantizer,
            num_groups,
            split_sizes,
            tensor_offsets,
            clamped.limit,
            clamped.alpha,
            clamped.glu_linear_offset,
            int(activation.glu_interleave_size or 0),
            compute_scale_grad,
        )
    if isinstance(activation, ScaledSReLU):
        return tex.grouped_scaled_dsrelu(
            dy,
            x,
            s,
            quantizer,
            num_groups,
            split_sizes,
            tensor_offsets,
            compute_scale_grad,
        )
    raise TypeError(f"Unsupported scaled activation type ({type(activation).__name__})")


class BackwardScaledActivationGroupedLinear(FusedOperation):
    """Scaled activation backward + grouped quantize + grouped linear backward."""

    def __init__(self, *, linear: GroupedLinear, activation: _ScaledActivation) -> None:
        super().__init__((linear, activation))

    def fuser_backward(
        self,
        basic_op_ctxs: list[OperationContext],
        grad_output: torch.Tensor,
        *,
        basic_op_grad_extra_outputs: list[tuple[torch.Tensor, ...]],
    ) -> tuple[
        Optional[torch.Tensor],
        Iterable[Iterable[Optional[torch.Tensor]]],
        Iterable[Iterable[Optional[torch.Tensor]]],
    ]:
        linear = self.basic_ops[0]
        activation = self.basic_ops[1]
        linear_ctx, activation_ctx = basic_op_ctxs

        if not linear_ctx.requires_grad:
            _, _, act_grad_extra_inputs = activation.fuser_backward(
                [activation_ctx],
                grad_output,
                basic_op_grad_extra_outputs=[basic_op_grad_extra_outputs[1]],
            )
            return None, [(), ()], [(), act_grad_extra_inputs[0]]

        del basic_op_grad_extra_outputs
        input_, scales = activation_ctx.saved_tensors
        input_ = maybe_dequantize(input_, activation_ctx.dtype)
        scales = maybe_dequantize(scales, activation_ctx.dtype)
        grad_output = maybe_dequantize(grad_output, activation_ctx.dtype)

        split_sizes = linear_ctx.saved_tensors[0]
        split_sizes, (grad_output_tensor_offsets,) = tex.splits_to_offsets_multi(
            split_sizes,
            input_.device,
            strides=[linear.out_features],
            include_leading_zero=[True],
            dtypes=[torch.int64],
            bulk_allocate=False,
        )
        grad_output_quantizer = linear_ctx.grad_output_quantizers[0]
        grad_output_quantizer.set_usage(
            rowwise=linear_ctx.input_requires_grad,
            columnwise=linear_ctx.weight_requires_grad,
        )
        grad_output_quantizer.optimize_for_gemm = True
        grouped_dy, dense_dy, grad_scales = _grouped_scaled_dactivation(
            activation,
            grad_output,
            input_,
            scales,
            quantizer=grad_output_quantizer,
            num_groups=linear.num_groups,
            split_sizes=split_sizes,
            tensor_offsets=grad_output_tensor_offsets,
            compute_scale_grad=activation_ctx.extra_input_requires_grad,
        )

        grad_input, grad_params, grad_extra_inputs = linear._fuser_backward_grouped_tensor(
            ctx=linear_ctx,
            grad_output=dense_dy,
            grouped_grad_output=grouped_dy,
        )

        clear_tensor_data(activation_ctx.saved_tensors[0])
        return (
            grad_input,
            [grad_params[0], ()],
            [grad_extra_inputs[0], (grad_scales,)],
        )

    @staticmethod
    def fuse_backward_ops(
        ops: list[FusibleOperation],
        *,
        recipe: Optional[Recipe] = None,
        **unused,
    ) -> list[FusibleOperation]:
        """Fuse each supported GroupedLinear + ScaledActivation pair."""
        out: list[FusibleOperation] = []
        idx = 0
        while idx < len(ops):
            if (
                idx + 1 < len(ops)
                and isinstance(ops[idx], GroupedLinear)
                and isinstance(ops[idx + 1], _SCALED_ACTIVATION_TYPES)
                and act_grouped_linear_fusion_supported(ops[idx], ops[idx + 1], recipe)
            ):
                out.append(
                    BackwardScaledActivationGroupedLinear(
                        linear=ops[idx],
                        activation=ops[idx + 1],
                    )
                )
                idx += 2
            else:
                out.append(ops[idx])
                idx += 1
        return out
