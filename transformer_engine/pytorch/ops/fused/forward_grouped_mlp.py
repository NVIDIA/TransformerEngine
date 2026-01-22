# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Fused operation for forward GEMM + scale + add."""

from __future__ import annotations
from collections.abc import Iterable
from typing import Any, Optional

import torch
from cudnn import grouped_gemm_swiglu_wrapper_sm100  ### TODO Check if available

import transformer_engine_torch as tex
from ...cpp_extensions import general_grouped_gemm
from ...cpu_offload import is_cpu_offload_enabled, mark_activation_offload
from ...quantization import FP8GlobalStateManager
from ...tensor import MXFP8Tensor, Quantizer
from ..basic import GroupedLinear, MultiplyExtraInput, SwiGLU
from ..fuser import register_forward_fusion
from ..op import FusedOperation, FusibleOperation, OperationContext
from .._common import is_quantized_tensor, maybe_dequantize


class ForwardGroupedMLP_CuTeGEMMSwiGLU_MXFP8(FusedOperation):

    def __init__(
        self,
        *,
        fc1: GroupedLinear,
        swiglu: SwiGLU,
        scale: MultiplyExtraInput,
        fc2: GroupedLinear,
    ) -> None:
        super().__init__((fc1, swiglu, scale, fc2))

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
        fc1_op, swiglu_op, scale_op, fc2_op = self.basic_ops
        fc1_ctx, swiglu_ctx, scale_ctx, fc2_ctx = basic_op_ctxs

        # Tensor properties
        in_shape = list(input_.size())
        assert len(in_shape) == 2, f"Expected 2D input tensor, got shape={in_shape}."
        fc1_weight_shape = (fc1_op.out_features, fc1_op.in_features)
        fc2_weight_shape = (fc2_op.out_features, fc2_op.in_features)
        group_size = fc1_op.group_size
        device = fc1_op.weight0.device
        if torch.is_autocast_enabled():
            dtype = torch.get_autocast_dtype("cuda")
        else:
            dtype = fc1_op.weight0.dtype

        # Check which grads are required
        requires_grad = any(ctx.requires_grad for ctx in basic_op_ctxs)
        input_requires_grad = requires_grad
        weight_requires_grad = (
            requires_grad
            and (fc1_op.weight0.requires_grad or fc2_op.weight0.requires_grad)
        )

        # Quantizers
        fc1_input_quantizers = [None] * group_size
        fc1_weight_quantizers = [None] * group_size
        fc1_grad_output_quantizers = [None] * group_size
        fc2_input_quantizers = [None] * group_size
        fc2_weight_quantizers = [None] * group_size
        fc2_grad_output_quantizers = [None] * group_size
        for idx in range(group_size):
            fc1_input_quantizers[idx] = fc1_op.get_quantizer("forward", 2 * idx)
            fc1_weight_quantizers[idx] = fc1_op.get_quantizer("forward", 2 * idx + 1)
            fc1_grad_output_quantizers[idx] = fc1_op.get_quantizer("backward", idx)
            fc2_input_quantizers[idx] = fc2_op.get_quantizer("forward", 2 * idx)
            fc2_weight_quantizers[idx] = fc2_op.get_quantizer("forward", 2 * idx + 1)
            fc2_grad_output_quantizers[idx] = fc2_op.get_quantizer("backward", idx)

        # Extract split sizes from extra input
        fc1_split_sizes = basic_op_extra_inputs[0][0]
        fc2_split_sizes = basic_op_extra_inputs[3][0]
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

        # Extract post-scales from extra input
        scales = basic_op_extra_inputs[2][0]
        scales_shape = tuple(scales.size())
        if scales.numel() != scales_shape[0]:
            raise RuntimeError(
                f"{self.__class__.__name__} assumes scales are over leading dim, "
                f"but got shape={scales_shape}."
            )

        # Extract params
        fc1_weights = [getattr(fc1_op, f"weight{idx}") for idx in range(group_size)]
        fc2_weights = [getattr(fc2_op, f"weight{idx}") for idx in range(group_size)]

        # Convert weight dtype if needed
        fc1_ws = []
        fc2_ws = []
        for w, quantizer in zip(fc1_weights, fc1_weight_quantizers):
            if not is_quantized_tensor(w):
                quantizer = weight_quantizers[group_idx]
                quantizer.set_usage(rowwise=True, columnwise=input_requires_grad)
                w = quantizer(w)
            fc1_ws.append(w)
        for w, quantizer in zip(fc2_weights, fc2_weight_quantizers):
            if not is_quantized_tensor(w):
                quantizer = weight_quantizers[group_idx]
                quantizer.set_usage(rowwise=True, columnwise=input_requires_grad)
                w = quantizer(w)
            fc2_ws.append(w)

        # Split input tensor and convert dtypes if needed
        fc1_x = maybe_dequantize(input_, dtype)
        fc1_xs = None
        for quantizer in fc1_input_quantizers:
            quantizer.set_usage(rowwise=True, columnwise=weight_requires_grad)
            quantizer.optimize_for_gemm = True
        fc1_xs = tex.split_quantize(fc1_x, split_sizes_int, fc1_input_quantizers)

        # Pack data tensors
        fc1_x_data = torch.cat([x._rowwise_data for x in fc1_xs])
        fc1_x_data = fc1_x_data.view(dtype=torch.float8_e4m3fn)
        fc1_x_data = fc1_x_data.unsqueeze(0).permute(1, 2, 0)
        fc1_x_scales = torch.cat([x._rowwise_scale_inv for x in fc1_xs])
        fc1_x_scales = fc1_x_scales.view(dtype=torch.float8_e8m0fnu)
        fc1_x_scales = fc1_x_scales.reshape(
            1,
            in_shape[0] // 128,
            in_shape[1] // 128,
            32,
            4,
            4,
        )
        fc1_x_scales = fc1_x_scales.permute(3, 4, 1, 5, 2, 0)

        # Pack weight tensors
        fc1_w_data = torch.stack([w._rowwise_data for w in fc1_weights])
        fc1_w_data = fc1_w_data.view(dtype=torch.float8_e4m3fn)
        fc1_w_data = fc1_w_data.reshape(group_size, 2, fc1_weight_shape[0] // 64, 32, fc1_weight_shape[1])
        fc1_w_data = fc1_w_data.transpose(1, 2)  # Interleave SwiGLU gate/activation
        fc1_w_data = fc1_w_data.flip(2).contiguous()  # Swap SwiGLU gate/activation
        fc1_w_data = fc1_w_data.reshape(group_size, fc1_weight_shape[0], fc1_weight_shape[1])
        fc1_w_data = fc1_w_data.permute(1, 2, 0)
        fc1_w_scales = torch.stack([w._rowwise_scale_inv for w in fc1_weights])
        fc1_w_scales = fc1_w_scales.view(dtype=torch.float8_e8m0fnu)
        fc1_w_scales = fc1_w_scales.reshape(group_size, 2, fc1_weight_shape[0] // 64, 32, fc1_weight_shape[1] // 32)
        fc1_w_scales = fc1_w_scales.transpose(1, 2)  # Interleave SwiGLU gate/activation
        fc1_w_scales = fc1_w_scales.flip(2).contiguous()  # Swap SwiGLU gate/activation
        fc1_w_scales = fc1_w_scales.reshape(group_size, fc1_weight_shape[0] // 128, 4, 32, fc1_weight_shape[1] // 128, 4)
        fc1_w_scales = fc1_w_scales.permute(0, 1, 4, 3, 2, 5).contiguous()  # Convert to swizzled layout
        fc1_w_scales = fc1_w_scales.permute(3, 4, 1, 5, 2, 0)

        # Kernel tile logic
        tile_idx_to_expert_idx = []
        cta_tile_m = 256  ### TODO ?
        for group_idx in range(group_size):
            num_tiles = split_sizes_int[group_idx] // cta_tile_m
            tile_idx_to_expert_idx.extend([group_idx] * num_tiles)
        num_non_exiting_tiles = torch.tensor([len(tile_idx_to_expert_idx)], device=device, dtype=torch.int32)
        tile_idx_to_expert_idx = torch.tensor(tile_idx_to_expert_idx, device=device, dtype=torch.int32)

        # Fused kernel for FC1 + SwiGLU + post-scale
        fc1_kernel_out = grouped_gemm_swiglu_wrapper_sm100(
            fc1_x_data,
            fc1_w_data,
            fc1_x_scales,
            fc1_w_scales,
            tile_idx_to_expert_idx,
            num_non_exiting_tiles,
            torch.ones(group_size, dtype=dtype, device=device),  # alpha_tensor
            split_sizes_int,
            torch.ones(1, dtype=dtype, device=device),  # norm_const_tensor
            scales.detach().reshape(-1, 1, 1),
            acc_dtype=torch.float32,
            c_dtype=torch.bfloat16,
            d_dtype=torch.float8_e4m3fn,
            cd_major="n",
            cluster_shape_mn=(2, 1),
            sf_vec_size=32,
            sf_dtype=torch.float8_e8m0fnu,
        )

        # Extract kernel outputs and construct MXFP8 tensors
        swiglu_in = fc1_kernel_out["c_tensor"]
        swiglu_in = swiglu_in.permute(2, 0, 1).contiguous()
        swiglu_in = swiglu_in.reshape(in_shape[0], fc1_weight_shape[0] // 64, 2, 32)
        swiglu_in = swiglu_in.transpose(1, 2)  # Undo interleaved SwiGLU gate/activation
        swiglu_in = swiglu_in.flip(1).contiguous()  # Undo swapped SwiGLU gate/activation
        swiglu_in = swiglu_in.reshape(in_shape[0], fc1_weight_shape[0])
        fc2_in_row_data = fc1_kernel_out["d_tensor"]
        fc2_in_row_data = fc2_in_row_data.permute(2, 0, 1).contiguous()
        fc2_in_row_data = fc2_in_row_data.reshape(in_shape[0], fc2_weight_shape[1]).contiguous()
        fc2_in_row_data = torch.split(fc2_in_row_data, split_sizes_int)
        fc2_in_row_scale = fc1_kernel_out["sfd_row_tensor"]
        fc2_in_row_scale = fc2_in_row_scale.permute(5, 2, 1, 0, 4, 3).contiguous()  # Convert to compact layout
        fc2_in_row_scale = fc2_in_row_scale.reshape(in_shape[0], fc2_weight_shape[1] // 32)
        fc2_in_row_scale = torch.split(fc2_in_row_scale, split_sizes_int)
        fc2_in_col_data = fc1_kernel_out["d_col_tensor"]
        fc2_in_col_data = fc2_in_col_data.permute(2, 0, 1).contiguous()
        fc2_in_col_data = fc2_in_col_data.reshape(in_shape[0], fc2_weight_shape[1]).contiguous()
        fc2_in_col_data = torch.split(fc2_in_col_data, split_sizes_int)
        fc2_in_col_scale = fc1_kernel_out["sfd_col_tensor"]
        fc2_in_col_scale = fc2_in_col_scale.permute(5, 4, 3, 2, 1, 0).contiguous()  # Convert to compact layout
        fc2_in_col_scale = fc2_in_col_scale.reshape(in_shape[0] // 32, fc2_weight_shape[1])
        fc2_in_col_scale = torch.split(fc2_in_col_scale, [s // 32 for s in split_sizes_int])

        # Construct MXFP8 tensors for FC2
        fc2_xs = []
        for group_idx in range(group_size):
            x = MXFP8Tensor(
                shape=(split_sizes_int[group_idx], fc2_weight_shape[1]),
                dtype=dtype,
                fp8_dtype=tex.DType.kFloat8E4M3,
                rowwise_data=fc2_in_row_data[group_idx],
                rowwise_scale_inv=fc2_in_row_scale[group_idx],
                columnwise_data=fc2_in_col_data[group_idx],
                columnwise_scale_inv=fc2_in_col_scale[group_idx],
                quantizer=fc2_input_quantizers[group_idx],
                requires_grad=False,
                with_gemm_swizzled_scales=False,
            )
            fc2_xs.append(x)

        # FC2 GEMM
        fc2_out_shape = in_shape[:-1] + [fc2_weight_shape[0]]
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

        # Save state for backward pass
        if requires_grad:
            # FC1 state
            fc1_ctx.save_for_backward(split_sizes, *fc1_xs, *fc1_ws)
            fc1_ctx.with_quantized_compute = True
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

            # Scale
            scale_ctx.save_for_backward(fc2_out, scales)
            scale_ctx.input_shape = fc2_out.size()
            scale_ctx.extra_input_shape = scales_shape
            scale_ctx.input_requires_grad = True
            scale_ctx.extra_input_requires_grad = scales.requires_grad

            # FC2 state
            fc2_ctx.save_for_backward(split_sizes, *fc2_xs, *fc2_ws)
            fc2_ctx.with_quantized_compute = True
            fc2_ctx.input_quantizers = fc2_input_quantizers
            fc2_ctx.weight_quantizers = fc2_weight_quantizers
            fc2_ctx.grad_output_quantizers = fc2_grad_output_quantizers
            fc2_ctx.grad_input_quantizers = None
            fc2_ctx.dtype = dtype
            fc2_ctx.input_requires_grad = input_requires_grad
            fc2_ctx.weight_requires_grad = weight_requires_grad

        return fc2_out, [(), (), (), ()]

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
    if recipe is None:
        return ops
    if not recipe.mxfp8():
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
            and isinstance(window[2], MultiplyExtraInput)
            and isinstance(window[3], GroupedLinear)
        ):
            matches_pattern = False
        elif window[0].has_bias or window[3].has_bias:
            matches_pattern = False
        elif window[0].group_size != window[3].group_size:
            matches_pattern = False
        elif (
            window[0].in_features % 256 != 0
            or window[0].out_features % 256 != 0
            or window[3].in_features % 256 != 0
            or window[3].out_features % 256 != 0
        ):
            matches_pattern = False

        if matches_pattern:
            # Construct fused op if window matches pattern
            op = ForwardGroupedMLP_CuTeGEMMSwiGLU_MXFP8(
                fc1=window[0],
                swiglu=window[1],
                scale=window[2],
                fc2=window[3],
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
