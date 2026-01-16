# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Fused operation for forward GEMM + scale + add."""

from __future__ import annotations
from collections.abc import Iterable
from typing import Any, Optional

import torch

import transformer_engine_torch as tex
from ...cpp_extensions import general_grouped_gemm
from ...cpu_offload import is_cpu_offload_enabled, mark_activation_offload
from ...quantization import FP8GlobalStateManager
from ...tensor import Quantizer
from ..basic import GroupedLinear, MultiplyExtraInput, SwiGLU
from ..fuser import register_forward_fusion
from ..op import FusedOperation, FusibleOperation, OperationContext
from .._common import is_quantized_tensor, maybe_dequantize


class ForwardGroupedMLP_CuTeGEMMSwiGLU(FusedOperation):

    def __init__(
        self,
        *,
        fc1: GroupedLinear,
        swiglu: SwiGLU,
        fc2: GroupedLinear,
        scale: MultiplyExtraInput,
    ) -> None:
        super().__init__((fc1, swiglu, fc2, scale))

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

        # Get basic operations
        fc1_op, swiglu_op, fc2_op, scale_op = self.basic_ops
        fc1_ctx, swiglu_ctx, fc2_ctx, scale_ctx = basic_op_ctxs

        group_size = fc1_op.group_size
        device = fc1_op.weight0.device
        in_shape = list(input_.size())

        # Check which grads are required
        requires_grad = any(ctx.requires_grad for ctx in basic_op_ctxs)
        input_requires_grad = requires_grad
        weight_requires_grad = (
            requires_grad
            and (fc1_op.weight0.requires_grad or fc2_op.weight0.requires_grad)
        )

        # Quantizers
        with_quantized_compute = FP8GlobalStateManager.is_fp8_enabled()
        fc1_input_quantizers = [None] * group_size
        fc1_weight_quantizers = [None] * group_size
        fc1_grad_output_quantizers = [None] * group_size
        fc2_input_quantizers = [None] * group_size
        fc2_weight_quantizers = [None] * group_size
        fc2_grad_output_quantizers = [None] * group_size
        if with_quantized_compute:
            for idx in range(group_size):
                fc1_input_quantizers[idx] = fc1_op.get_quantizer("forward", 2 * group_idx)
                fc1_weight_quantizers[idx] = fc1_op.get_quantizer("forward", 2 * group_idx + 1)
                fc1_grad_output_quantizers[idx] = fc1_op.get_quantizer("backward", group_idx)
                fc2_input_quantizers[idx] = fc2_op.get_quantizer("forward", 2 * group_idx)
                fc2_weight_quantizers[idx] = fc2_op.get_quantizer("forward", 2 * group_idx + 1)
                fc2_grad_output_quantizers[idx] = fc2_op.get_quantizer("backward", group_idx)

        # Get autocast dtype if needed
        if torch.is_autocast_enabled():
            dtype = torch.get_autocast_dtype("cuda")
        else:
            dtype = fc1_op.weight0.dtype

        # Extract split sizes from extra input
        fc1_split_sizes = basic_op_extra_inputs[0][0]
        fc2_split_sizes = basic_op_extra_inputs[2][0]
        if (
            fc1_split_sizes.size() != fc2_split_sizes.size()
            or fc1_split_sizes.data_ptr() != fc2_split_sizes.data_ptr()
        ):
            raise RuntimeError(
                f"{self.__class__.__name__} got different split points for FC1 and FC2."
            )
        split_sizes = fc1_split_sizes
        split_sizes_int = [int(s) for s in split_sizes.tolist()]
        if len(split_sizes_int) != group_size:
            raise ValueError(f"Expected {group_size} splits, but got {len(split_sizes_int)}.")

        # Extract params
        fc1_weights = [getattr(fc1_op, f"weight{idx}") for idx in range(group_size)]
        fc2_weights = [getattr(fc2_op, f"weight{idx}") for idx in range(group_size)]

        # Convert weight dtype if needed
        fc1_ws = []
        fc2_ws = []
        for w, quantizer in zip(fc1_weights, fc1_weight_quantizers):
            if not with_quantized_compute:
                w = maybe_dequantize(w, dtype)
            elif with_quantized_compute and not is_quantized_tensor(w):
                quantizer = weight_quantizers[group_idx]
                quantizer.set_usage(rowwise=True, columnwise=input_requires_grad)
                w = quantizer(w)
            fc1_ws.append(w)
        for w, quantizer in zip(fc2_weights, fc2_weight_quantizers):
            if not with_quantized_compute:
                w = maybe_dequantize(w, dtype)
            elif with_quantized_compute and not is_quantized_tensor(w):
                quantizer = weight_quantizers[group_idx]
                quantizer.set_usage(rowwise=True, columnwise=input_requires_grad)
                w = quantizer(w)
            fc2_ws.append(w)

        # Split input tensor and convert dtypes if needed
        fc1_x = maybe_dequantize(input_, dtype)
        fc1_xs = None
        if with_quantized_compute:
            for quantizer in fc1_input_quantizers:
                quantizer.set_usage(rowwise=True, columnwise=weight_requires_grad)
            fc1_xs = tex.split_quantize(fc1_x, split_sizes_int, fc1_input_quantizers)
        else:
            fc1_xs = torch.split(fc1_x, split_sizes_int)

        # FC1 GEMM
        fc1_out_shape = in_shape[:-1] + [fc1_op.out_features]
        fc1_out = torch.empty(fc1_out_shape, dtype=dtype, device=device)
        general_grouped_gemm(
            fc1_ws,
            fc1_xs,
            [fc1_out],
            [None] * group_size,  # quantization_params
            dtype,
            m_splits=split_sizes_int,
            bias=[None] * group_size,
            use_bias=False,
            single_output=True,
        )

        # SwiGLU
        swiglu_in = fc1_out
        swiglu_out = tex.swiglu(swiglu_in, None)

        # Split input tensor and convert dtypes if needed
        fc2_x = swiglu_out
        fc2_xs = None
        if with_quantized_compute:
            for quantizer in fc2_input_quantizers:
                quantizer.set_usage(rowwise=True, columnwise=weight_requires_grad)
            fc2_xs = tex.split_quantize(fc2_x, split_sizes_int, fc2_input_quantizers)
        else:
            fc2_xs = torch.split(fc2_x, split_sizes_int)

        # FC2 GEMM
        fc2_out_shape = in_shape[:-1] + [fc2_op.out_features]
        fc2_out = torch.empty(fc2_out_shape, dtype=dtype, device=device)
        general_grouped_gemm(
            fc2_ws,
            fc2_xs,
            [fc2_out],
            [None] * group_size,  # quantization_params
            dtype,
            m_splits=split_sizes_int,
            bias=[None] * group_size,
            use_bias=False,
            single_output=True,
        )

        # Post-scale
        scales = basic_op_extra_inputs[3][0]
        scales_shape = tuple(scales.size())
        if scales.numel() != scales_shape[0]:
            raise RuntimeError(
                f"{self.__class__.__name__} assumes scales are over leading dim, "
                f"but got shape={scales_shape}."
            )
        out = fc2_out * scales

        # Save state for backward pass
        if requires_grad:
            # FC1 state
            fc1_ctx.save_for_backward(split_sizes, *fc1_xs, *fc1_ws)
            fc1_ctx.with_quantized_compute = with_quantized_compute
            fc1_ctx.input_quantizers = fc1_input_quantizers
            fc1_ctx.weight_quantizers = fc1_weight_quantizers
            fc1_ctx.grad_output_quantizers = fc1_grad_output_quantizers
            fc1_ctx.grad_input_quantizers = None
            fc1_ctx.dtype = dtype
            fc1_ctx.input_requires_grad = input_requires_grad
            fc1_ctx.weight_requires_grad = weight_requires_grad

            # SwiGLU
            swiglu_ctx.save_for_backward(swiglu_in)
            swiglu_ctx.dtype = dtype
            swiglu_ctx.prev_op_grad_output_quantizer = None

            # FC2 state
            fc2_ctx.save_for_backward(split_sizes, *fc2_xs, *fc2_ws)
            fc2_ctx.with_quantized_compute = with_quantized_compute
            fc2_ctx.input_quantizers = fc2_input_quantizers
            fc2_ctx.weight_quantizers = fc2_weight_quantizers
            fc2_ctx.grad_output_quantizers = fc2_grad_output_quantizers
            fc2_ctx.grad_input_quantizers = None
            fc2_ctx.dtype = dtype
            fc2_ctx.input_requires_grad = input_requires_grad
            fc2_ctx.weight_requires_grad = weight_requires_grad

            # Scale
            scale_ctx.save_for_backward(fc2_out, scales)
            scale_ctx.input_shape = fc2_out.size()
            scale_ctx.extra_input_shape = scales_shape
            scale_ctx.input_requires_grad = True
            scale_ctx.extra_input_requires_grad = scales.requires_grad

        return out, [(), (), (), ()]

    @staticmethod
    def fuse_forward_ops(
        ops: list[FusibleOperation],
        *,
        recipe: Optional[Recipe] = None,
        **unused,  # pylint: disable=unused-argument
    ) -> list[FusibleOperation]:
        """Apply operation fusion for forward pass.

        Parameters
        ----------
        ops : list of FusibleOperation
            Forward pass operations.
        recipe : Recipe, optional
            Quantization recipe.

        Returns
        -------
        ops : list of FusibleOperation
            Updated forward pass operations

        """

        # Check if recipe is supported
        if recipe is not None:
            return ops

        # Scan through ops, fusing if possible
        out = []
        window, ops = ops[:4], ops[4:]
        while len(window) == 4:

            # Check if window matches pattern
            matches_pattern = True
            if not (
                isinstance(window[0], GroupedLinear)
                and isinstance(window[1], SwiGLU)
                and isinstance(window[2], GroupedLinear)
                and isinstance(window[3], MultiplyExtraInput)
            ):
                matches_pattern = False
            elif window[0].has_bias or window[2].has_bias:
                matches_pattern = False
            elif window[0].group_size != window[2].group_size:
                matches_pattern = False

            if matches_pattern:
                # Construct fused op if window matches pattern
                op = ForwardGroupedMLP_CuTeGEMMSwiGLU(
                    fc1=window[0],
                    swiglu=window[1],
                    fc2=window[2],
                    scale=window[3],
                )
                window = [op]
            else:
                # Shift window if window doesn't match pattern
                out.extend(window[:-3])
                window = window[-3:]

            # Adjust window to expected size
            out.extend(window[:-4])
            window = window[-4:]
            while ops and len(window) < 4:
                window.append(ops[0])
                ops = ops[1:]

        # Return list of ops
        out.extend(window)
        return out
