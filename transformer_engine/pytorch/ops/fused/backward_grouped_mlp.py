# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Fused operation for MoE grouped MLP."""

from __future__ import annotations
from collections.abc import Callable, Iterable
import functools
import itertools
from typing import Any, Optional

import torch

import transformer_engine_torch as tex
from ...cpp_extensions import general_grouped_gemm
from ...module._common import noop_cat
from ...module.base import get_dummy_wgrad
from ...quantization import Recipe
from ...tensor import MXFP8Tensor, Quantizer
from ...utils import clear_tensor_data, get_device_compute_capability
from ..basic import GroupedLinear, ScaledSwiGLU
from ..fuser import register_backward_fusion
from ..op import FusedOperation, FusibleOperation, OperationContext
from .._common import is_quantized_tensor, maybe_dequantize


class BackwardGroupedMLP_CuTeGEMMDSwiGLU_MXFP8(FusedOperation):
    """Fused op for MXFP8 GroupedLinear + ScaledSwiGLU + GroupedLinear

    Uses experimental CuTe DSL kernel from cuDNN front-end.

    """

    @classmethod
    @functools.lru_cache(maxsize=None)
    def grouped_gemm_dswiglu_kernel(cls) -> Callable:
        """Fused kernel for grouped GEMM, SwiGLU backward, and scale grad."""
        from cudnn import grouped_gemm_dswiglu_wrapper_sm100  # pylint: disable=no-name-in-module

        return grouped_gemm_dswiglu_wrapper_sm100

    @classmethod
    @functools.lru_cache(maxsize=None)
    def is_supported(cls) -> bool:
        """Whether this fused operation is supported on the current system."""
        if get_device_compute_capability() < (10, 0):
            # Kernel requires SM100+
            return False
        try:
            # Make sure kernel is available
            cls.grouped_gemm_dswiglu_kernel()
        except ImportError:
            return False
        return True

    def __init__(
        self,
        *,
        fc1: GroupedLinear,
        swiglu: ScaledSwiGLU,
        fc2: GroupedLinear,
    ) -> None:
        super().__init__((fc1, swiglu, fc2))

        # Check for unsupported configurations
        if not self.is_supported():
            self.grouped_gemm_dswiglu_kernel()  # Try triggering import error
            raise RuntimeError(f"{self.__class__.__name__} is not supported on this system.")
        if fc1.in_features % 256 != 0 or fc1.out_features % 256 != 0:
            raise ValueError(
                f"Unsupported dims for FC1 (num_groups={fc1.num_groups}, "
                f"in_features={fc1.in_features}, out_features={fc1.out_features})."
            )
        if fc2.in_features % 256 != 0 or fc2.out_features % 256 != 0:
            raise ValueError(
                f"Unsupported dims for FC2 (num_groups={fc2.num_groups}, "
                f"in_features={fc2.in_features}, out_features={fc2.out_features})."
            )
        if fc1.out_features != 2 * fc2.in_features or fc1.num_groups != fc2.num_groups:
            raise ValueError(
                f"FC1 (num_groups={fc1.num_groups}, in_features={fc1.in_features}, "
                f"out_features={fc1.out_features}) "
                f"and FC2 (num_groups={fc2.num_groups}, in_features={fc2.in_features}, "
                f"out_features={fc2.out_features}) do not match."
            )
        if fc1.has_bias or fc2.has_bias:
            raise ValueError("Fused kernel does not support bias.")
        if swiglu.glu_interleave_size != 32:
            raise ValueError(
                "Fused kernel requires 32-wide GLU interleaving, "
                f"but got glu_interleave_size={swiglu.glu_interleave_size}."
            )

    def fuser_backward(
        self,
        basic_op_ctxs: list[OperationContext],
        grad_output: torch.Tensor,
        **unused,
    ) -> tuple[
        torch.Tensor,
        list[tuple[Optional[torch.Tensor], ...]],
        list[tuple[()]],
    ]:

        # Get basic operations
        fc1_op, _, fc2_op = self.basic_ops
        fc1_ctx, swiglu_ctx, fc2_ctx = basic_op_ctxs

        # Tensor properties
        out_shape = list(grad_output.size())
        assert len(out_shape) == 2, f"Expected 2D grad output tensor, got shape={out_shape}."
        fc1_weight_shape = (fc1_op.out_features, fc1_op.in_features)
        fc2_weight_shape = (fc2_op.out_features, fc2_op.in_features)
        num_groups = fc1_op.num_groups
        device = fc1_op.weight0.device
        dtype = fc1_ctx.dtype

        # Saved tensors from FC1 forward
        saved_tensors = fc1_ctx.saved_tensors
        split_sizes, saved_tensors = saved_tensors[0], saved_tensors[1:]
        fc1_xs, saved_tensors = saved_tensors[:num_groups], saved_tensors[num_groups:]
        fc1_ws, saved_tensors = saved_tensors[:num_groups], saved_tensors[num_groups:]

        # Saved tensors from scaled SwiGLU forward
        swiglu_in, scales = swiglu_ctx.saved_tensors

        # Saved tensors from FC2 forward
        saved_tensors = fc2_ctx.saved_tensors
        _, saved_tensors = saved_tensors[0], saved_tensors[1:]  # Assume same split sizes as FC1
        fc2_xs, saved_tensors = saved_tensors[:num_groups], saved_tensors[num_groups:]
        fc2_ws, saved_tensors = saved_tensors[:num_groups], saved_tensors[num_groups:]

        # Group splits
        split_sizes_cpu = [int(s) for s in split_sizes.tolist()]
        if len(split_sizes_cpu) != num_groups:
            raise ValueError(f"Expected {num_groups} splits, but got {len(split_sizes_cpu)}.")
        split_sizes = split_sizes.to(dtype=torch.int, device=device)
        split_points = torch.cumsum(split_sizes, 0, dtype=torch.int)

        # Split grad output tensor and convert dtypes if needed
        fc2_dy = maybe_dequantize(grad_output, dtype)
        for quantizer in fc2_ctx.grad_output_quantizers:
            quantizer.set_usage(rowwise=True, columnwise=fc2_ctx.weight_requires_grad)
            quantizer.optimize_for_gemm = True
        fc2_dys = tex.split_quantize(fc2_dy, split_sizes_cpu, fc2_ctx.grad_output_quantizers)

        # Quantize FC2 weights to MXFP8 if needed
        if not is_quantized_tensor(fc2_ws[0]):
            for quantizer in fc2_ctx.weight_quantizers:
                quantizer.set_usage(rowwise=False, columnwise=True)
            fc2_ws = fc2_op._quantize_weights_mxfp8(fc2_ws, fc2_ctx.weight_quantizers)

        # Pack data tensors
        # Note: Fused kernel expects tensor with non-contiguous
        # logical dims.
        # Data actual shape: (1, sum(m), k)
        # Scale actual shape: (1, sum(m)/128, k/128, 32 (block row),
        #  4 (block row), 4 (block col))
        # Data logical shape: (sum(m), k, 1)
        # Scale logical shape: (32 (block row), 4 (block row),
        #   sum(m)/128, 4 (block col), k/128, 1)
        fc2_dy_data = noop_cat([dy._rowwise_data for dy in fc2_dys])
        fc2_dy_data = fc2_dy_data.view(dtype=torch.float8_e4m3fn)
        fc2_dy_data = fc2_dy_data.unsqueeze(0).permute(1, 2, 0)
        fc2_dy_scales = noop_cat([dy._rowwise_scale_inv for dy in fc2_dys])
        fc2_dy_scales = fc2_dy_scales.view(dtype=torch.float8_e8m0fnu)
        fc2_dy_scales = fc2_dy_scales.view(
            1,
            out_shape[0] // 128,
            out_shape[1] // 128,
            32,
            4,
            4,
        )
        fc2_dy_scales = fc2_dy_scales.permute(3, 4, 1, 5, 2, 0)

        # Pack weight tensors
        # Note: Fused kernel expects tensor with non-contiguous
        # logical dims.
        # Data actual shape: (num_groups, k, n)
        # Scale actual shape: (num_groups, n/128, k/128, 32 (block col),
        #  4 (block col), 4 (block row))
        # Data logical shape: (n, k, num_groups)
        # Scale logical shape: (32 (block col), 4 (block col), n/128,
        #   4 (block row), k/128, num_groups)
        fc2_w_data = noop_cat([w._columnwise_data for w in fc2_ws])
        fc2_w_data = fc2_w_data.view(dtype=torch.float8_e4m3fn)
        fc2_w_data = fc2_w_data.view(num_groups, fc2_weight_shape[0], fc2_weight_shape[1])
        fc2_w_data = fc2_w_data.permute(2, 1, 0)
        fc2_w_scales = noop_cat([w._columnwise_scale_inv for w in fc2_ws])
        fc2_w_scales = fc2_w_scales.view(dtype=torch.float8_e8m0fnu)
        fc2_w_scales = fc2_w_scales.view(
            num_groups, fc2_weight_shape[0] // 128, 4, fc2_weight_shape[1] // 128, 4, 32
        )  # Unswizzled layout
        fc2_w_scales = fc2_w_scales.permute(
            0, 3, 1, 5, 4, 2
        ).contiguous()  # Convert to swizzled layout
        fc2_w_scales = fc2_w_scales.permute(3, 4, 1, 5, 2, 0)

        # Grad for SwiGLU post-scales
        grad_scales = torch.zeros((scales.size(0), 1, 1), dtype=torch.float32, device=device)

        # Kernel scaling factors
        ones = torch.ones(num_groups, dtype=dtype, device=device)

        # Fused kernel for FC2 dgrad + dSwiGLU + grad scale
        fc2_dgrad_kernel_out = self.grouped_gemm_dswiglu_kernel()(
            fc2_dy_data,
            fc2_w_data,
            swiglu_in.unsqueeze(0).permute(1, 2, 0),
            fc2_dy_scales,
            fc2_w_scales,
            split_points,
            ones,  # alpha_tensor
            ones,  # beta_tensor
            scales.detach().reshape(-1, 1, 1),
            grad_scales,
            norm_const_tensor=ones[:1],
            d_dtype=torch.float8_e4m3fn,
            cd_major="n",
            sf_vec_size=32,
        )

        # Unpack kernel outputs
        # Note: Fused kernel outputs tensors with non-contiguous
        # logical dims.
        # Row-wise data logical shape: (sum(m), k, 1)
        # Row-wise scale logical shape: (32 (block row), 4 (block row),
        #   sum(m)/128, 4 (block col), k/128, 1)
        # Column-wise data logical shape: (k, sum(m), 1)
        # Column-wise scale logical shape: (32 (block col), 4 (block col),
        #   k/128, 4 (block row), sum(m)/128, 1)
        fc1_dy_row_data = fc2_dgrad_kernel_out["d_row_tensor"]
        fc1_dy_row_data = fc1_dy_row_data.permute(2, 0, 1)
        fc1_dy_row_data = fc1_dy_row_data.view(out_shape[0], fc1_weight_shape[0])
        fc1_dy_row_data = torch.split(fc1_dy_row_data.contiguous(), split_sizes_cpu)
        fc1_dy_row_scale = fc2_dgrad_kernel_out["sfd_row_tensor"]
        fc1_dy_row_scale = fc1_dy_row_scale.permute(5, 2, 4, 0, 1, 3)
        fc1_dy_row_scale = fc1_dy_row_scale.view(out_shape[0], fc1_weight_shape[0] // 32)
        fc1_dy_row_scale = torch.split(fc1_dy_row_scale.contiguous(), split_sizes_cpu)
        fc1_dy_col_data = fc2_dgrad_kernel_out["d_col_tensor"]
        fc1_dy_col_data = fc1_dy_col_data.permute(2, 0, 1)
        fc1_dy_col_data = fc1_dy_col_data.view(out_shape[0], fc1_weight_shape[0])
        fc1_dy_col_data = torch.split(fc1_dy_col_data.contiguous(), split_sizes_cpu)
        fc1_dy_col_scale = fc2_dgrad_kernel_out["sfd_col_tensor"]
        fc1_dy_col_scale = fc1_dy_col_scale.permute(5, 2, 4, 0, 1, 3)
        fc1_dy_col_scale = torch.split(fc1_dy_col_scale, [s // 128 for s in split_sizes_cpu], dim=2)
        fc1_dy_col_scale = [s.contiguous().view(-1, fc1_weight_shape[0]) for s in fc1_dy_col_scale]
        grad_scales = grad_scales.view(-1).to(dtype=dtype)

        # Construct MXFP8 tensors for FC1
        fc1_dys = []
        for group_idx in range(num_groups):
            dy = MXFP8Tensor(
                shape=(split_sizes_cpu[group_idx], fc1_weight_shape[0]),
                dtype=dtype,
                fp8_dtype=tex.DType.kFloat8E4M3,
                rowwise_data=fc1_dy_row_data[group_idx],
                rowwise_scale_inv=fc1_dy_row_scale[group_idx],
                columnwise_data=fc1_dy_col_data[group_idx],
                columnwise_scale_inv=fc1_dy_col_scale[group_idx],
                quantizer=fc1_ctx.grad_output_quantizers[group_idx],
                requires_grad=False,
                with_gemm_swizzled_scales=True,
            )
            fc1_dys.append(dy)

        # FC2 wgrad GEMM
        fc2_dws = [None] * num_groups
        if fc2_ctx.weight_requires_grad:

            # Initialize grad buffers
            accumulate_into_main_grad = False
            if fc2_op._accumulate_into_main_grad:
                # Megatron-LM wgrad fusion
                # Note: Get grad tensors from params so we can
                # accumulate directly into it.
                for group_idx in range(num_groups):
                    weight_param = getattr(fc2_op, f"weight{group_idx}")
                    if hasattr(weight_param, "__fsdp_param__"):
                        weight_param.main_grad = weight_param.get_main_grad()
                    fc2_dws[group_idx] = weight_param.main_grad
                accumulate_into_main_grad = not getattr(
                    fc2_op.weight0, "overwrite_main_grad", False
                )
            else:
                for group_idx in range(num_groups):
                    fc2_dws[group_idx] = torch.empty(
                        fc2_weight_shape,
                        dtype=dtype,
                        device=device,
                    )

            # Launch GEMM
            general_grouped_gemm(
                fc2_xs,
                fc2_dys,
                fc2_dws,
                [None] * num_groups,  # quantization_params
                dtype,
                layout="NT",
                m_splits=split_sizes_cpu,
                accumulate=accumulate_into_main_grad,
            )

            # Megatron-LM wgrad fusion
            # Note: Return dummy tensor for grad weight if needed.
            if accumulate_into_main_grad:
                for group_idx in range(num_groups):
                    weight_param = getattr(fc2_op, f"weight{group_idx}")
                    if hasattr(weight_param, "grad_added_to_main_grad"):
                        weight_param.grad_added_to_main_grad = True
                        fc2_dws[group_idx] = get_dummy_wgrad(
                            list(fc2_weight_shape),
                            weight_param.dtype,
                            zero=getattr(weight_param, "zero_out_wgrad", False),
                        )

        # Clear FC2 input tensor if possible
        clear_tensor_data(*fc2_xs)

        # FC1 dgrad GEMM
        grad_input = None
        if fc1_ctx.input_requires_grad:

            # Quantize weights to MXFP8 if needed
            if not is_quantized_tensor(fc1_ws[0]):
                for quantizer in fc1_ctx.weight_quantizers:
                    quantizer.set_usage(rowwise=False, columnwise=True)
                fc1_ws = fc1_op._quantize_weights_mxfp8(fc1_ws, fc1_ctx.weight_quantizers)

            # Launch GEMM
            in_shape = out_shape[:-1] + [fc1_weight_shape[1]]
            grad_input = torch.empty(in_shape, dtype=dtype, device=device)
            general_grouped_gemm(
                fc1_ws,
                fc1_dys,
                [grad_input],
                [None] * num_groups,  # quantization_params
                dtype,
                layout="NN",
                m_splits=split_sizes_cpu,
                single_output=True,
            )

        # FC1 wgrad GEMM
        fc1_dws = [None] * num_groups
        if fc1_ctx.weight_requires_grad:

            # Initialize grad buffers
            accumulate_into_main_grad = False
            if fc1_op._accumulate_into_main_grad:
                # Megatron-LM wgrad fusion
                # Note: Get grad tensors from params so we can
                # accumulate directly into it.
                for group_idx in range(num_groups):
                    weight_param = getattr(fc1_op, f"weight{group_idx}")
                    if hasattr(weight_param, "__fsdp_param__"):
                        weight_param.main_grad = weight_param.get_main_grad()
                    fc1_dws[group_idx] = weight_param.main_grad
                accumulate_into_main_grad = not getattr(
                    fc1_op.weight0, "overwrite_main_grad", False
                )
            else:
                fc1_dws = [
                    torch.empty(fc1_weight_shape, dtype=dtype, device=device)
                    for _ in range(num_groups)
                ]

            # Launch GEMM
            general_grouped_gemm(
                fc1_xs,
                fc1_dys,
                fc1_dws,
                [None] * num_groups,  # quantization_params
                dtype,
                layout="NT",
                m_splits=split_sizes_cpu,
                accumulate=accumulate_into_main_grad,
            )

            # Megatron-LM wgrad fusion
            # Note: Return dummy tensor for grad weight if needed.
            if accumulate_into_main_grad:
                for group_idx in range(num_groups):
                    weight_param = getattr(fc1_op, f"weight{group_idx}")
                    if hasattr(weight_param, "grad_added_to_main_grad"):
                        weight_param.grad_added_to_main_grad = True
                        fc1_dws[group_idx] = get_dummy_wgrad(
                            list(fc1_weight_shape),
                            weight_param.dtype,
                            zero=getattr(weight_param, "zero_out_wgrad", False),
                        )

        # Clear FC1 input tensor if possible
        clear_tensor_data(*fc1_xs)

        return grad_input, [fc1_dws, (), fc2_dws], [(None,), (grad_scales,), (None,)]


def fuse_backward_ops(
    ops: list[FusibleOperation],
    *,
    recipe: Optional[Recipe] = None,
    **unused,  # pylint: disable=unused-argument
) -> list[FusibleOperation]:
    """Apply operation fusion for backward pass.

    Parameters
    ----------
    ops : list of FusibleOperation
        Forward pass operations.
    recipe : Recipe, optional
        Quantization recipe.

    Returns
    -------
    ops : list of FusibleOperation
        Updated backward pass operations

    """

    # Return immediately if fused kernel is not supported
    if not BackwardGroupedMLP_CuTeGEMMDSwiGLU_MXFP8.is_supported():
        return ops

    # Check if recipe is supported
    if recipe is None:
        return ops
    if not recipe.mxfp8():
        return ops

    # Scan through ops, fusing if possible
    out = []
    window, ops = ops[:3], ops[3:]
    while len(window) == 3:

        # Check if window matches pattern
        matches_pattern = True
        if not (
            isinstance(window[0], GroupedLinear)
            and isinstance(window[1], ScaledSwiGLU)
            and isinstance(window[2], GroupedLinear)
        ):
            matches_pattern = False
        elif window[0].has_bias or window[2].has_bias:
            matches_pattern = False
        elif window[0].num_groups != window[2].num_groups:
            matches_pattern = False
        elif (
            window[0].in_features % 256 != 0
            or window[0].out_features % 256 != 0
            or window[2].in_features % 256 != 0
            or window[2].out_features % 256 != 0
        ):
            matches_pattern = False
        elif window[1].glu_interleave_size != 32:
            matches_pattern = False

        if matches_pattern:
            # Construct fused op if window matches pattern
            op = BackwardGroupedMLP_CuTeGEMMDSwiGLU_MXFP8(
                fc1=window[0],
                swiglu=window[1],
                fc2=window[2],
            )
            window = [op]
        else:
            # Shift window if window doesn't match pattern
            out.extend(window[:-2])
            window = window[-2:]

        # Adjust window to expected size
        out.extend(window[:-3])
        window = window[-3:]
        while ops and len(window) < 3:
            window.append(ops[0])
            ops = ops[1:]

    # Return list of ops
    out.extend(window)
    return out


# Register fusion if available
if BackwardGroupedMLP_CuTeGEMMDSwiGLU_MXFP8.is_supported():
    register_backward_fusion(fuse_backward_ops, prepend=True)
