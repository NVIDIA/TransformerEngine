# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Pair fusions between scaled activations and grouped linear operations."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Optional

import torch

import transformer_engine_torch as tex
from ...cpu_offload import is_cpu_offload_enabled, mark_activation_offload
from ...quantization import Recipe
from ...tensor import Quantizer
from ...utils import clear_tensor_data
from .._common import maybe_dequantize
from ..basic import GroupedLinear, ScaledClampedQGeGLU, ScaledSReLU, ScaledSwiGLU
from ..op import FusedOperation, FusibleOperation, OperationContext


_ScaledActivation = ScaledSwiGLU | ScaledClampedQGeGLU | ScaledSReLU
_SCALED_ACTIVATION_TYPES = (ScaledSwiGLU, ScaledClampedQGeGLU, ScaledSReLU)


def _grouped_scaled_activation(
    activation: _ScaledActivation,
    input_: torch.Tensor,
    scales: torch.Tensor,
    quantizer: Quantizer,
    num_groups: int,
    split_sizes: torch.Tensor,
    tensor_offsets: torch.Tensor,
) -> torch.Tensor:
    """Dispatch to the matching tex.grouped_scaled_* forward API."""
    x = input_.reshape(-1, input_.size(-1))
    s = scales.reshape(-1)
    if isinstance(activation, ScaledSwiGLU):
        return tex.grouped_scaled_swiglu(
            x,
            s,
            quantizer,
            num_groups,
            split_sizes,
            tensor_offsets,
            int(activation.glu_interleave_size or 0),
        )
    if isinstance(activation, ScaledClampedQGeGLU):
        clamped = activation._clamped
        return tex.grouped_scaled_clamped_swiglu(
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
        )
    return tex.grouped_scaled_srelu(x, s, quantizer, num_groups, split_sizes, tensor_offsets)


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
    """Dispatch to the matching tex.grouped_scaled_d* API.

    Returns ``(grouped_dx, dense_dx, dscales)`` where ``grouped_dx`` is the
    grouped-quantized grad input for the next grouped GEMM and ``dense_dx`` is
    the dense high-precision grad input (reused for the bias gradient so we do
    not have to dequantize ``grouped_dx``).
    """
    dy = grad_output.reshape(-1, grad_output.size(-1))
    x = input_.reshape(-1, input_.size(-1))
    s = scales.reshape(-1)
    if isinstance(activation, ScaledSwiGLU):
        dx, dense_dx, dscales = tex.grouped_scaled_dswiglu(
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
    elif isinstance(activation, ScaledClampedQGeGLU):
        clamped = activation._clamped
        dx, dense_dx, dscales = tex.grouped_scaled_clamped_dswiglu(
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
    else:
        dx, dense_dx, dscales = tex.grouped_scaled_dsrelu(
            dy,
            x,
            s,
            quantizer,
            num_groups,
            split_sizes,
            tensor_offsets,
            compute_scale_grad,
        )
    return dx, dense_dx, dscales


def act_grouped_linear_fusion_supported(
    linear: GroupedLinear,
    activation: _ScaledActivation,
    recipe: Optional[Recipe],
) -> bool:
    """Whether ScaledActivation + GroupedLinear can use grouped quantized compute."""
    if recipe is None or activation.activation_recompute_in_mlp:
        return False
    input_quantizers = [
        linear.get_quantizer("forward", 2 * group_idx) for group_idx in range(linear.num_groups)
    ]
    weight = linear.weight if linear.single_grouped_weight else linear.weight0
    dtype = torch.get_autocast_dtype("cuda") if torch.is_autocast_enabled() else weight.dtype
    return linear._is_graph_safe_path_supported(
        with_quantized_compute=True,
        input_quantizers=input_quantizers,
        dtype=dtype,
        single_grouped_weight=linear.single_grouped_weight,
    )


class ForwardScaledActivationGroupedLinear(FusedOperation):
    """Scaled activation + grouped quantize + grouped linear forward."""

    def __init__(self, *, activation: _ScaledActivation, linear: GroupedLinear) -> None:
        super().__init__((activation, linear))

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
        activation = self.basic_ops[0]
        linear = self.basic_ops[1]
        activation_ctx, linear_ctx = basic_op_ctxs
        if basic_op_kwargs[0] or basic_op_kwargs[1]:
            raise ValueError("Scaled activation and GroupedLinear do not expect keyword arguments")

        weight = linear.weight if linear.single_grouped_weight else linear.weight0
        device = weight.device
        dtype = torch.get_autocast_dtype("cuda") if torch.is_autocast_enabled() else weight.dtype
        input_ = maybe_dequantize(input_, dtype)
        scales = maybe_dequantize(basic_op_extra_inputs[0][0], dtype)

        split_sizes = basic_op_extra_inputs[1][0]
        if int(split_sizes.numel()) != linear.num_groups:
            raise ValueError(
                f"Expected {linear.num_groups} splits, but got {int(split_sizes.numel())}."
            )
        split_sizes = split_sizes.to(device=device, dtype=torch.int64)
        linear_scales = basic_op_extra_inputs[1][1] if linear._scale_bias else None
        split_sizes, grouped_tensor_offsets = tex.splits_to_offsets_multi(
            split_sizes,
            device,
            strides=[1, 1, linear.in_features, linear.out_features],
            include_leading_zero=[False, True, True, True],
            dtypes=[torch.int32, torch.int64, torch.int64, torch.int64],
            bulk_allocate=True,
        )

        input_quantizers = [
            linear.get_quantizer("forward", 2 * group_idx) for group_idx in range(linear.num_groups)
        ]
        weight_quantizers = [
            linear.get_quantizer("forward", 2 * group_idx + 1)
            for group_idx in range(linear.num_groups)
        ]
        input_quantizer = input_quantizers[0]
        weight_requires_grad = linear_ctx.requires_grad and weight.requires_grad
        input_quantizer.set_usage(rowwise=True, columnwise=weight_requires_grad)
        input_quantizer.optimize_for_gemm = True

        grouped_x = _grouped_scaled_activation(
            activation,
            input_,
            scales,
            input_quantizer,
            linear.num_groups,
            split_sizes,
            grouped_tensor_offsets[2],
        )

        if activation_ctx.requires_grad:
            if is_cpu_offload_enabled():
                mark_activation_offload(input_)
            activation_ctx.input_requires_grad = True
            activation_ctx.extra_input_requires_grad = basic_op_extra_inputs[0][0].requires_grad
            activation_ctx.dtype = dtype
            activation_ctx.save_for_backward(input_, scales)

        out, tensors_to_save = linear._fuser_forward_grouped_tensor(
            split_sizes=split_sizes,
            scales=linear_scales,
            with_quantized_compute=True,
            input_quantizers=input_quantizers,
            weight_quantizers=weight_quantizers,
            dtype=dtype,
            input_requires_grad=linear_ctx.requires_grad,
            weight_requires_grad=weight_requires_grad,
            device=device,
            grouped_input=grouped_x,
            grouped_tensor_offsets=grouped_tensor_offsets,
        )
        linear.fuser_forward_save_ctx(
            basic_op_ctxs=[linear_ctx],
            input_=input_,
            tensors_to_save=[tensors_to_save],
            requires_grad=[linear_ctx.requires_grad],
            basic_op_extra_inputs=[basic_op_extra_inputs[1]],
            prev_op_grad_output_quantizer=prev_op_grad_output_quantizer,
            next_op_input_quantizer=next_op_input_quantizer,
            basic_op_kwargs=[basic_op_kwargs[1]],
            use_grouped_tensor_path=True,
        )
        return out, [(), ()]

    @staticmethod
    def fuse_forward_ops(
        ops: list[FusibleOperation],
        *,
        recipe: Optional[Recipe] = None,
        **unused,
    ) -> list[FusibleOperation]:
        """Fuse each supported ScaledActivation + GroupedLinear pair."""
        out: list[FusibleOperation] = []
        idx = 0
        while idx < len(ops):
            if (
                idx + 1 < len(ops)
                and isinstance(ops[idx], _SCALED_ACTIVATION_TYPES)
                and isinstance(ops[idx + 1], GroupedLinear)
                and act_grouped_linear_fusion_supported(ops[idx + 1], ops[idx], recipe)
            ):
                out.append(
                    ForwardScaledActivationGroupedLinear(
                        activation=ops[idx],
                        linear=ops[idx + 1],
                    )
                )
                idx += 2
            else:
                out.append(ops[idx])
                idx += 1
        return out


class BackwardGroupedLinearScaledActivation(FusedOperation):
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
        torch.Tensor,
        Iterable[Iterable[Optional[torch.Tensor]]],
        Iterable[Iterable[Optional[torch.Tensor]]],
    ]:
        del basic_op_grad_extra_outputs
        linear = self.basic_ops[0]
        activation = self.basic_ops[1]
        linear_ctx, activation_ctx = basic_op_ctxs
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

        # Feed the grouped-quantized grad into the grouped GEMM while also
        # passing the dense high-precision grad so the bias gradient avoids a
        # lossy dequantize of ``grouped_dy``.
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
                    BackwardGroupedLinearScaledActivation(
                        linear=ops[idx],
                        activation=ops[idx + 1],
                    )
                )
                idx += 2
            else:
                out.append(ops[idx])
                idx += 1
        return out
