# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Fused operation for MoE grouped MLP."""

from __future__ import annotations
from collections.abc import Callable
import functools
import inspect
import math
from typing import Optional

import torch

import transformer_engine_torch as tex
from cuda.bindings import driver as cuda
from ...cpp_extensions import (
    general_grouped_gemm_for_grouped_tensor,
)
from ...module._common import noop_cat
from ...module.base import get_dummy_wgrad
from ...quantization import Recipe
from ...tensor import Quantizer
from ...tensor.grouped_tensor import GroupedTensor
from ...utils import clear_tensor_data, get_cached_ones_tensor, get_device_compute_capability
from ...constants import MXFP8_BLOCK_SCALING_SIZE
from ..basic import GroupedLinear, ScaledSwiGLU
from ..fuser import register_backward_fusion
from ..op import FusedOperation, FusibleOperation, OperationContext
from .._common import (
    clone_grouped_tensor_storage,
    fuse_grouped_mlp_ops,
    is_quantized_tensor,
    make_grouped_tensor_from_buffers,
    maybe_dequantize,
)


@functools.lru_cache(maxsize=1)
def _dglu_wrapper_has_generate_dbias_arg() -> bool:
    """True if cudnn-frontend SM100 dGLU wrapper accepts ``generate_dbias``."""
    try:
        from cudnn import grouped_gemm_dglu_wrapper_sm100  # pylint: disable=import-outside-toplevel
    except ImportError:
        return False
    try:
        params = inspect.signature(grouped_gemm_dglu_wrapper_sm100).parameters
    except (TypeError, ValueError):
        return False
    return "generate_dbias" in params


class BackwardGroupedMLP_CuTeGEMMDSwiGLU_MXFP8(FusedOperation):
    """Fused op for MXFP8 GroupedLinear + ScaledSwiGLU + GroupedLinear

    Uses experimental CuTe DSL kernel from cuDNN front-end.

    """

    @classmethod
    @functools.lru_cache(maxsize=None)
    def grouped_gemm_dglu_kernel(cls) -> Callable:
        """Fused kernel for grouped GEMM, GLU activation backward, and scale grad."""
        from cudnn import grouped_gemm_dglu_wrapper_sm100  # pylint: disable=no-name-in-module

        return grouped_gemm_dglu_wrapper_sm100

    @classmethod
    @functools.lru_cache(maxsize=None)
    def grouped_gemm_quant_kernel(cls) -> Callable:
        """Grouped GEMM quant kernel for block-scaled inputs."""
        from cudnn import grouped_gemm_quant_wrapper_sm100  # pylint: disable=no-name-in-module

        return grouped_gemm_quant_wrapper_sm100

    @classmethod
    @functools.lru_cache(maxsize=None)
    def is_supported(cls) -> bool:
        """Whether this fused operation is supported on the current system."""
        if get_device_compute_capability() < (10, 0):
            return False
        try:
            cls.grouped_gemm_dglu_kernel()
            cls.grouped_gemm_quant_kernel()
        except ImportError:
            return False
        return True

    @classmethod
    def is_fc1_bias_supported(cls) -> bool:
        """Whether cudnn-frontend exposes ``generate_dbias`` on the dGLU SM100 wrapper (FC1 bias grad only)."""
        if not cls.is_supported():
            return False
        return _dglu_wrapper_has_generate_dbias_arg()

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
            self.grouped_gemm_dglu_kernel()  # Try triggering import error
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
        if swiglu.glu_interleave_size != 32:
            raise ValueError(
                "Fused kernel requires 32-wide GLU interleaving, "
                f"but got glu_interleave_size={swiglu.glu_interleave_size}."
            )

    def fuser_backward(
        self,
        basic_op_ctxs: list[OperationContext],
        grad_output: torch.Tensor,
        **unused,  # pylint: disable=unused-argument
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
        fc1_weight_param = fc1_op.weight if fc1_op.single_grouped_parameter else fc1_op.weight0
        device = fc1_weight_param.device
        dtype = fc1_ctx.dtype

        # Saved tensors from FC1 forward
        saved_tensors = fc1_ctx.saved_tensors
        split_sizes, split_points, saved_tensors = (
            saved_tensors[0],
            saved_tensors[1],
            saved_tensors[2:],
        )

        if fc1_op.single_grouped_parameter:
            grouped_fc1_weight, saved_tensors = saved_tensors[0], saved_tensors[1:]
        else:
            grouped_fc1_weight, saved_tensors = (
                saved_tensors[:num_groups],
                saved_tensors[num_groups:],
            )

        (
            fc1_x_data,
            fc1_x_col_data,
            fc1_x_scale,
            fc1_x_col_scale,
            fc1_x_tensor_offsets,
        ), saved_tensors = (
            saved_tensors[:5],
            saved_tensors[5:],
        )

        # Saved tensors from scaled SwiGLU forward
        swiglu_in, scales = swiglu_ctx.saved_tensors

        # Saved tensors from FC2 forward
        saved_tensors = fc2_ctx.saved_tensors
        _, saved_tensors = saved_tensors[0], saved_tensors[1:]  # Assume same split sizes as FC1
        if fc2_op.single_grouped_parameter:
            grouped_fc2_weight, saved_tensors = saved_tensors[0], saved_tensors[1:]
        else:
            grouped_fc2_weight, saved_tensors = (
                saved_tensors[:num_groups],
                saved_tensors[num_groups:],
            )

        (
            fc2_x_data,
            fc2_x_col_data,
            fc2_x_scale,
            fc2_x_col_scale,
            fc2_x_tensor_offsets,
        ), saved_tensors = (
            saved_tensors[:5],
            saved_tensors[5:],
        )

        # Group splits
        if int(split_sizes.numel()) != num_groups:
            raise ValueError(f"Expected {num_groups} splits, but got {int(split_sizes.numel())}.")
        split_sizes = split_sizes.to(dtype=torch.int64, device=device)
        split_points = split_points.to(dtype=torch.int, device=device)

        fc2_weight_for_gemm = grouped_fc2_weight
        if fc2_op.single_grouped_parameter:
            fc2_weight_for_gemm = clone_grouped_tensor_storage(grouped_fc2_weight)
            tex.swizzle_grouped_scales(fc2_weight_for_gemm, rowwise=False, columnwise=True)
        fc1_weight_for_gemm = grouped_fc1_weight
        if fc1_op.single_grouped_parameter:
            fc1_weight_for_gemm = clone_grouped_tensor_storage(grouped_fc1_weight)
            tex.swizzle_grouped_scales(fc1_weight_for_gemm, rowwise=False, columnwise=True)

        grouped_fc1_x = None
        if fc1_ctx.weight_requires_grad:
            grouped_fc1_x = make_grouped_tensor_from_buffers(
                num_groups=num_groups,
                data=fc1_x_data,
                columnwise_data=fc1_x_col_data,
                scale_inv=fc1_x_scale,
                columnwise_scale_inv=fc1_x_col_scale,
                split_sizes=split_sizes,
                logical_last_dim=fc1_weight_shape[1],
                dtype=dtype,
                quantizer=fc1_ctx.input_quantizers[0],
                with_gemm_swizzled_scales=True,
                tensor_offsets=fc1_x_tensor_offsets,
            )

        grouped_fc2_x = None
        if fc2_ctx.weight_requires_grad:
            grouped_fc2_x = make_grouped_tensor_from_buffers(
                num_groups=num_groups,
                data=fc2_x_data,
                columnwise_data=fc2_x_col_data,
                scale_inv=fc2_x_scale,
                columnwise_scale_inv=fc2_x_col_scale,
                split_sizes=split_sizes,
                logical_last_dim=fc2_weight_shape[1],
                dtype=dtype,
                quantizer=fc2_ctx.input_quantizers[0],
                with_gemm_swizzled_scales=True,
                tensor_offsets=fc2_x_tensor_offsets,
            )

        # Split grad output tensor and convert dtypes if needed
        fc2_dy = maybe_dequantize(grad_output, dtype)
        for quantizer in fc2_ctx.grad_output_quantizers:
            quantizer.set_usage(rowwise=True, columnwise=fc2_ctx.weight_requires_grad)
            quantizer.optimize_for_gemm = True
        output_fc2_dbias = fc2_op.has_bias
        gq_ret = tex.group_quantize(
            fc2_dy,
            fc2_ctx.grad_output_quantizers[0],
            num_groups,
            split_sizes,
            output_fc2_dbias,
        )
        if output_fc2_dbias:
            grouped_fc2_dy, fc2_dbias_packed = gq_ret
        else:
            grouped_fc2_dy = gq_ret
            fc2_dbias_packed = None

        fc2_bias_grads: Optional[list[Optional[torch.Tensor]]] = None
        fc2_bias_grad_packed: Optional[torch.Tensor] = None
        if fc2_dbias_packed is not None:
            if fc2_op.single_grouped_bias:
                fc2_bias_grad_packed = fc2_dbias_packed.to(dtype=dtype)
            else:
                fc2_bias_grads = [
                    fc2_dbias_packed[idx].to(dtype=dtype) for idx in range(num_groups)
                ]

        # Pack data tensors
        # Note: Fused kernel expects tensor with non-contiguous
        # logical dims.
        # Data actual shape: (1, sum(m), k)
        # Scale actual shape: (1, sum(m)/128, k/128, 32 (block row),
        #  4 (block row), 4 (block col))
        # Data logical shape: (sum(m), k, 1)
        # Scale logical shape: (32 (block row), 4 (block row),
        #   sum(m)/128, 4 (block col), k/128, 1)
        fc2_dy_data = grouped_fc2_dy.rowwise_data.view(out_shape[0], out_shape[1])
        fc2_dy_data = fc2_dy_data.view(dtype=torch.float8_e4m3fn)
        fc2_dy_data = fc2_dy_data.unsqueeze(0).permute(1, 2, 0)
        fc2_dy_scales = grouped_fc2_dy.scale_inv
        fc2_dy_scales = fc2_dy_scales.view(dtype=torch.float8_e8m0fnu)
        fc2_dy_scales = fc2_dy_scales.view(
            1,
            out_shape[0] // 128,
            out_shape[1] // 128,
            MXFP8_BLOCK_SCALING_SIZE,
            4,
            4,
        )
        fc2_dy_scales = fc2_dy_scales.permute(3, 4, 1, 5, 2, 0)

        # Kernel scaling factors
        alpha_tensor = get_cached_ones_tensor(num_groups, dtype, device)
        norm_const_tensor = get_cached_ones_tensor(1, dtype, device)
        current_stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

        if fc2_op.single_grouped_parameter:
            # Pack weight tensors for stacked kernel
            # Data actual shape: (num_groups, k, n)
            # Data logical shape: (n, k, num_groups)
            fc2_w_data = fc2_weight_for_gemm.columnwise_data
            fc2_w_data = fc2_w_data.view(dtype=torch.float8_e4m3fn)
            fc2_w_data = fc2_w_data.view(num_groups, fc2_weight_shape[0], fc2_weight_shape[1])
            fc2_w_data = fc2_w_data.permute(2, 1, 0)
            fc2_w_scales = fc2_weight_for_gemm.columnwise_scale_inv.view(dtype=torch.float8_e8m0fnu)
            fc2_w_scales = fc2_w_scales.view(
                num_groups,
                fc2_weight_shape[1] // 128,
                fc2_weight_shape[0] // 128,
                MXFP8_BLOCK_SCALING_SIZE,
                4,
                4,
            )
            fc2_w_scales = fc2_w_scales.permute(3, 4, 1, 5, 2, 0)

            prob_tensor = scales.detach().to(dtype=torch.float32).reshape(-1, 1, 1)
            dprob_tensor = torch.zeros_like(prob_tensor)

            fc2_dgrad_kernel_out = self.grouped_gemm_dglu_kernel()(
                a_tensor=fc2_dy_data,
                c_tensor=swiglu_in.unsqueeze(0).permute(1, 2, 0),
                sfa_tensor=fc2_dy_scales,
                padded_offsets=split_points,
                alpha_tensor=alpha_tensor,
                beta_tensor=alpha_tensor,
                prob_tensor=prob_tensor,
                dprob_tensor=dprob_tensor,
                b_tensor=fc2_w_data,
                sfb_tensor=fc2_w_scales,
                generate_dbias=fc1_op.has_bias,
                norm_const_tensor=norm_const_tensor,
                d_dtype=torch.float8_e4m3fn,
                cd_major="n",
                sf_vec_size=MXFP8_BLOCK_SCALING_SIZE,
                current_stream=current_stream,
                discrete_col_sfd=True,
                act_func="dswiglu",
                use_dynamic_sched=True,
            )
        else:
            fc2_b_ptrs, fc2_sfb_ptrs, _fc2_sw = tex.get_device_pointer_for_data_and_scales(
                [w._columnwise_data for w in grouped_fc2_weight],
                [w._columnwise_scale_inv for w in grouped_fc2_weight],
                swizzle=True,
                rowwise=False,
                data_dtype=grouped_fc2_weight[0]._fp8_dtype,
            )
            prob_tensor = scales.detach().to(dtype=torch.float32).reshape(-1, 1, 1)
            dprob_tensor = torch.zeros_like(prob_tensor)

            fc2_dgrad_kernel_out = self.grouped_gemm_dglu_kernel()(
                a_tensor=fc2_dy_data,
                c_tensor=swiglu_in.unsqueeze(0).permute(1, 2, 0),
                sfa_tensor=fc2_dy_scales,
                padded_offsets=split_points,
                alpha_tensor=alpha_tensor,
                beta_tensor=alpha_tensor,
                prob_tensor=prob_tensor,
                dprob_tensor=dprob_tensor,
                b_ptrs=fc2_b_ptrs,
                sfb_ptrs=fc2_sfb_ptrs,
                n=fc2_weight_shape[1],
                b_dtype=torch.float8_e4m3fn,
                b_major="n",
                generate_dbias=fc1_op.has_bias,
                norm_const_tensor=norm_const_tensor,
                d_dtype=torch.float8_e4m3fn,
                cd_major="n",
                sf_vec_size=MXFP8_BLOCK_SCALING_SIZE,
                current_stream=current_stream,
                discrete_col_sfd=True,
                act_func="dswiglu",
                use_dynamic_sched=True,
            )

        fc1_dy_row_data = fc2_dgrad_kernel_out["d_row_tensor"]
        fc1_dy_row_data = fc1_dy_row_data.view(out_shape[0], fc1_weight_shape[0]).contiguous()
        fc1_dy_row_scale = fc2_dgrad_kernel_out["sfd_row_tensor"]
        fc1_dy_col_data = fc2_dgrad_kernel_out["d_col_tensor"]
        fc1_dy_col_data = fc1_dy_col_data.view(out_shape[0], fc1_weight_shape[0]).contiguous()
        fc1_dy_col_scale = fc2_dgrad_kernel_out["sfd_col_tensor"]
        grad_scales = fc2_dgrad_kernel_out["dprob_tensor"]
        grad_scales = grad_scales.view(-1).to(dtype=dtype)

        fc1_bias_grads: Optional[list[Optional[torch.Tensor]]] = None
        fc1_bias_grad_packed: Optional[torch.Tensor] = None
        if fc1_op.has_bias:
            dbias_t = fc2_dgrad_kernel_out["dbias_tensor"]
            if dbias_t is not None:
                dbias_2d = dbias_t.squeeze(-1)
                if fc1_op.single_grouped_bias:
                    fc1_bias_grad_packed = dbias_2d.to(dtype=dtype)
                else:
                    fc1_bias_grads = [
                        dbias_2d[group_idx].to(dtype=dtype) for group_idx in range(num_groups)
                    ]

        # Autograd returns for weight grads when wgrad is deferred (multi-weight path only).
        fc1_autograd_weight_grads: Optional[list[Optional[torch.Tensor]]] = None
        fc2_autograd_weight_grads: Optional[list[Optional[torch.Tensor]]] = None

        # FC1 grad output for dgrad and wgrad GEMMs
        grouped_fc1_dy = make_grouped_tensor_from_buffers(
            num_groups=num_groups,
            data=fc1_dy_row_data,
            columnwise_data=fc1_dy_col_data,
            scale_inv=fc1_dy_row_scale,
            columnwise_scale_inv=fc1_dy_col_scale,
            split_sizes=split_sizes,
            logical_last_dim=fc1_weight_shape[0],
            dtype=dtype,
            quantizer=fc1_ctx.grad_output_quantizers[0],
            with_gemm_swizzled_scales=True,
        )

        # FC2 wgrad GEMM
        fc2_packed_wgrad = None
        fc2_weight_grads: list[Optional[torch.Tensor]]
        if fc2_op.single_grouped_parameter:
            fc2_weight_grads = [None]
        else:
            fc2_weight_grads = [None] * num_groups
        if fc2_ctx.weight_requires_grad:

            # Initialize grad buffers
            accumulate_into_main_grad = False
            if fc2_op.single_grouped_parameter:
                grouped_fc2_wgrad = None
                weight_param = fc2_op.weight
                if fc2_op._accumulate_into_main_grad:
                    # Megatron-LM wgrad fusion
                    # Note: Get grad tensors from params so we can
                    # accumulate directly into it.
                    if hasattr(weight_param, "__fsdp_param__"):
                        weight_param.main_grad = weight_param.get_main_grad()
                    main_grad = weight_param.main_grad
                    grouped_shape = (num_groups, *fc2_weight_shape)
                    if main_grad.shape != grouped_shape:
                        if main_grad.numel() != math.prod(grouped_shape):
                            raise RuntimeError(
                                "Grouped MLP fused backward expected FC2 main_grad to have "
                                f"shape {grouped_shape} or matching numel, "
                                f"but got shape {tuple(main_grad.shape)}"
                            )
                        # Keep aliasing with weight.main_grad; do not allow implicit copies.
                        try:
                            main_grad = main_grad.view(grouped_shape)
                        except RuntimeError as e:
                            raise RuntimeError(
                                "Grouped MLP fused backward requires FC2 main_grad to be viewable"
                                f" as {grouped_shape} without copy, but got shape"
                                f" {tuple(main_grad.shape)} and stride {tuple(main_grad.stride())}"
                            ) from e
                    accumulate_into_main_grad = not getattr(
                        weight_param, "overwrite_main_grad", False
                    )
                    if accumulate_into_main_grad:
                        grouped_fc2_wgrad = GroupedTensor.make_grouped_tensor_from_rowwise_data(
                            num_tensors=num_groups,
                            tensor_shape=fc2_weight_shape,
                            rowwise_data=main_grad,
                            dtype=main_grad.dtype,
                        )

                if grouped_fc2_wgrad is None:
                    # TODO:ksivaman: This is not CUDA Graph safe.
                    grouped_fc2_wgrad = GroupedTensor.make_grouped_tensor_with_shapes(
                        num_tensors=num_groups,
                        shapes=[fc2_weight_shape] * num_groups,
                        quantizer=None,
                        device=device,
                        dtype=dtype,
                    )
                delay_fc2_wgrad = (
                    fc2_op.wgrad_store is not None and fc2_op.wgrad_store.delay_wgrad_compute()
                )
                # Launch GEMM
                # A=grouped_input, B=grouped_fc2_dy; B's scales are GEMM-swizzled (see group_quantize above).
                if delay_fc2_wgrad:
                    fc2_op.wgrad_store.put(
                        [grouped_fc2_x, grouped_fc2_dy, grouped_fc2_wgrad],
                        functools.partial(
                            general_grouped_gemm_for_grouped_tensor,
                            layout="NT",
                            accumulate=accumulate_into_main_grad,
                        ),
                    )
                    if accumulate_into_main_grad and hasattr(
                        weight_param, "grad_added_to_main_grad"
                    ):
                        weight_param.grad_added_to_main_grad = True
                        fc2_packed_wgrad = get_dummy_wgrad(
                            list(weight_param.size()),
                            weight_param.dtype,
                            zero=getattr(weight_param, "zero_out_wgrad", False),
                        )
                else:
                    # A=grouped_input, B=grouped_fc2_dy; B's scales are GEMM-swizzled (see group_quantize above).
                    general_grouped_gemm_for_grouped_tensor(
                        grouped_fc2_x,
                        grouped_fc2_dy,
                        grouped_fc2_wgrad,
                        layout="NT",
                        accumulate=accumulate_into_main_grad,
                    )
                    fc2_packed_wgrad = grouped_fc2_wgrad.rowwise_data.view(
                        num_groups, *fc2_weight_shape
                    )
                    if accumulate_into_main_grad and hasattr(
                        weight_param, "grad_added_to_main_grad"
                    ):
                        weight_param.grad_added_to_main_grad = True
                        fc2_packed_wgrad = get_dummy_wgrad(
                            list(weight_param.size()),
                            weight_param.dtype,
                            zero=getattr(weight_param, "zero_out_wgrad", False),
                        )
            else:
                if fc2_op._accumulate_into_main_grad:
                    for idx in range(num_groups):
                        weight_param = getattr(fc2_op, f"weight{idx}")
                        if hasattr(weight_param, "__fsdp_param__"):
                            weight_param.main_grad = weight_param.get_main_grad()
                        fc2_weight_grads[idx] = weight_param.main_grad
                    accumulate_into_main_grad = not getattr(
                        fc2_op.weight0, "overwrite_main_grad", False
                    )
                else:
                    for idx in range(num_groups):
                        fc2_weight_grads[idx] = torch.empty(
                            fc2_weight_shape, dtype=dtype, device=device
                        )

                delay_fc2_wgrad = (
                    fc2_op.wgrad_store is not None and fc2_op.wgrad_store.delay_wgrad_compute()
                )
                if delay_fc2_wgrad:
                    fc2_wgrad_buffers = fc2_weight_grads
                    fc2_op.wgrad_store.put(
                        [grouped_fc2_x, grouped_fc2_dy, fc2_wgrad_buffers],
                        functools.partial(
                            general_grouped_gemm_for_grouped_tensor,
                            layout="NT",
                            accumulate=accumulate_into_main_grad,
                        ),
                    )
                    if accumulate_into_main_grad:
                        fc2_autograd_weight_grads = [
                            fc2_wgrad_buffers[i] for i in range(num_groups)
                        ]
                        for idx in range(num_groups):
                            weight_param = getattr(fc2_op, f"weight{idx}")
                            if hasattr(weight_param, "grad_added_to_main_grad"):
                                weight_param.grad_added_to_main_grad = True
                                fc2_autograd_weight_grads[idx] = get_dummy_wgrad(
                                    list(weight_param.size()),
                                    weight_param.dtype,
                                    zero=getattr(weight_param, "zero_out_wgrad", False),
                                )
                    else:
                        fc2_autograd_weight_grads = [None] * num_groups
                else:
                    general_grouped_gemm_for_grouped_tensor(
                        grouped_fc2_x,
                        grouped_fc2_dy,
                        fc2_weight_grads,
                        layout="NT",
                        accumulate=accumulate_into_main_grad,
                    )
                    if accumulate_into_main_grad:
                        for idx in range(num_groups):
                            weight_param = getattr(fc2_op, f"weight{idx}")
                            if hasattr(weight_param, "grad_added_to_main_grad"):
                                weight_param.grad_added_to_main_grad = True
                                fc2_weight_grads[idx] = get_dummy_wgrad(
                                    list(weight_param.size()),
                                    weight_param.dtype,
                                    zero=getattr(weight_param, "zero_out_wgrad", False),
                                )

        # Clear FC2 input tensor if possible
        if grouped_fc2_x is not None and not (
            fc2_ctx.weight_requires_grad
            and fc2_op.wgrad_store is not None
            and fc2_op.wgrad_store.delay_wgrad_compute()
        ):
            clear_tensor_data(
                grouped_fc2_x.data,
                grouped_fc2_x.columnwise_data,
                grouped_fc2_x.scale_inv,
                grouped_fc2_x.columnwise_scale_inv,
            )

        # FC1 dgrad GEMM
        grad_input = None
        if fc1_ctx.input_requires_grad:
            in_shape = out_shape[:-1] + [fc1_weight_shape[1]]

            if fc1_op.single_grouped_parameter:
                fc1_dgrad_a_data = fc2_dgrad_kernel_out["d_row_tensor"]
                fc1_dgrad_a_scales = fc2_dgrad_kernel_out["sfd_row_tensor"]

                fc1_w_data = fc1_weight_for_gemm.columnwise_data
                fc1_w_data = fc1_w_data.view(dtype=torch.float8_e4m3fn)
                fc1_w_data = fc1_w_data.view(num_groups, fc1_weight_shape[0], fc1_weight_shape[1])
                fc1_w_data = fc1_w_data.permute(2, 1, 0)
                fc1_w_scales = fc1_weight_for_gemm.columnwise_scale_inv.view(
                    dtype=torch.float8_e8m0fnu
                )
                fc1_w_scales = fc1_w_scales.view(
                    num_groups,
                    fc1_weight_shape[1] // 128,
                    fc1_weight_shape[0] // 128,
                    MXFP8_BLOCK_SCALING_SIZE,
                    4,
                    4,
                )
                fc1_w_scales = fc1_w_scales.permute(3, 4, 1, 5, 2, 0)

                fc1_dgrad_kernel_out = self.grouped_gemm_quant_kernel()(
                    a_tensor=fc1_dgrad_a_data,
                    sfa_tensor=fc1_dgrad_a_scales,
                    padded_offsets=split_points,
                    alpha_tensor=alpha_tensor.float(),
                    b_tensor=fc1_w_data,
                    sfb_tensor=fc1_w_scales,
                    norm_const_tensor=None,
                    prob_tensor=torch.ones(
                        (out_shape[0], 1, 1), dtype=torch.float32, device=device
                    ),
                    acc_dtype=torch.float32,
                    c_dtype=dtype,
                    d_dtype=dtype,
                    cd_major="n",
                    sf_vec_size=MXFP8_BLOCK_SCALING_SIZE,
                    current_stream=current_stream,
                    discrete_col_sfd=True,
                    use_dynamic_sched=True,
                )
                grad_input = fc1_dgrad_kernel_out["d_tensor"].view(in_shape)
            else:
                fc1_dgrad_a_data = fc2_dgrad_kernel_out["d_row_tensor"]
                fc1_dgrad_a_scales = fc2_dgrad_kernel_out["sfd_row_tensor"]

                fc1_b_ptrs, fc1_sfb_ptrs, _ = tex.get_device_pointer_for_data_and_scales(
                    [w._columnwise_data for w in grouped_fc1_weight],
                    [w._columnwise_scale_inv for w in grouped_fc1_weight],
                    swizzle=True,
                    rowwise=False,
                    data_dtype=grouped_fc1_weight[0]._fp8_dtype,
                )

                fc1_dgrad_kernel_out = self.grouped_gemm_quant_kernel()(
                    a_tensor=fc1_dgrad_a_data,
                    sfa_tensor=fc1_dgrad_a_scales,
                    padded_offsets=split_points,
                    alpha_tensor=alpha_tensor.float(),
                    b_ptrs=fc1_b_ptrs,
                    sfb_ptrs=fc1_sfb_ptrs,
                    n=fc1_weight_shape[1],
                    b_dtype=torch.float8_e4m3fn,
                    b_major="n",
                    norm_const_tensor=None,
                    prob_tensor=torch.ones(
                        (out_shape[0], 1, 1), dtype=torch.float32, device=device
                    ),
                    acc_dtype=torch.float32,
                    c_dtype=dtype,
                    d_dtype=dtype,
                    cd_major="n",
                    sf_vec_size=MXFP8_BLOCK_SCALING_SIZE,
                    current_stream=current_stream,
                    discrete_col_sfd=True,
                    use_dynamic_sched=True,
                )
                grad_input = fc1_dgrad_kernel_out["d_tensor"].view(in_shape)

        # FC1 wgrad GEMM
        fc1_packed_wgrad = None
        fc1_weight_grads: list[Optional[torch.Tensor]]
        if fc1_op.single_grouped_parameter:
            fc1_weight_grads = [None]
        else:
            fc1_weight_grads = [None] * num_groups
        if fc1_ctx.weight_requires_grad:

            # Initialize grad buffers
            accumulate_into_main_grad = False
            if fc1_op.single_grouped_parameter:
                grouped_fc1_wgrad = None
                weight_param = fc1_op.weight
                if fc1_op._accumulate_into_main_grad:
                    # Megatron-LM wgrad fusion
                    # Note: Get grad tensors from params so we can
                    # accumulate directly into it.
                    if hasattr(weight_param, "__fsdp_param__"):
                        weight_param.main_grad = weight_param.get_main_grad()
                    main_grad = weight_param.main_grad
                    grouped_shape = (num_groups, *fc1_weight_shape)
                    if main_grad.shape != grouped_shape:
                        if main_grad.numel() != math.prod(grouped_shape):
                            raise RuntimeError(
                                "Grouped MLP fused backward expected FC1 main_grad to have "
                                f"shape {grouped_shape} or matching numel, "
                                f"but got shape {tuple(main_grad.shape)}"
                            )
                        # Keep aliasing with weight.main_grad; do not allow implicit copies.
                        try:
                            main_grad = main_grad.view(grouped_shape)
                        except RuntimeError as e:
                            raise RuntimeError(
                                "Grouped MLP fused backward requires FC1 main_grad to be viewable"
                                f" as {grouped_shape} without copy, but got shape"
                                f" {tuple(main_grad.shape)} and stride {tuple(main_grad.stride())}"
                            ) from e
                    accumulate_into_main_grad = not getattr(
                        weight_param, "overwrite_main_grad", False
                    )
                    if accumulate_into_main_grad:
                        grouped_fc1_wgrad = GroupedTensor.make_grouped_tensor_from_rowwise_data(
                            num_tensors=num_groups,
                            tensor_shape=fc1_weight_shape,
                            rowwise_data=main_grad,
                            dtype=main_grad.dtype,
                        )

                if grouped_fc1_wgrad is None:
                    # TODO:ksivaman: This is not CUDA Graph safe.
                    grouped_fc1_wgrad = GroupedTensor.make_grouped_tensor_with_shapes(
                        num_tensors=num_groups,
                        shapes=[fc1_weight_shape] * num_groups,
                        quantizer=None,
                        device=device,
                        dtype=dtype,
                    )

                delay_fc1_wgrad = (
                    fc1_op.wgrad_store is not None and fc1_op.wgrad_store.delay_wgrad_compute()
                )
                if delay_fc1_wgrad:
                    fc1_op.wgrad_store.put(
                        [grouped_fc1_x, grouped_fc1_dy, grouped_fc1_wgrad],
                        functools.partial(
                            general_grouped_gemm_for_grouped_tensor,
                            layout="NT",
                            accumulate=accumulate_into_main_grad,
                        ),
                    )
                    if accumulate_into_main_grad and hasattr(
                        weight_param, "grad_added_to_main_grad"
                    ):
                        weight_param.grad_added_to_main_grad = True
                        fc1_packed_wgrad = get_dummy_wgrad(
                            list(weight_param.size()),
                            weight_param.dtype,
                            zero=getattr(weight_param, "zero_out_wgrad", False),
                        )
                else:
                    general_grouped_gemm_for_grouped_tensor(
                        grouped_fc1_x,
                        grouped_fc1_dy,
                        grouped_fc1_wgrad,
                        layout="NT",
                        accumulate=accumulate_into_main_grad,
                    )
                    fc1_packed_wgrad = grouped_fc1_wgrad.rowwise_data.view(
                        num_groups, *fc1_weight_shape
                    )
                    if accumulate_into_main_grad and hasattr(
                        weight_param, "grad_added_to_main_grad"
                    ):
                        weight_param.grad_added_to_main_grad = True
                        fc1_packed_wgrad = get_dummy_wgrad(
                            list(weight_param.size()),
                            weight_param.dtype,
                            zero=getattr(weight_param, "zero_out_wgrad", False),
                        )
            else:
                if fc1_op._accumulate_into_main_grad:
                    for idx in range(num_groups):
                        weight_param = getattr(fc1_op, f"weight{idx}")
                        if hasattr(weight_param, "__fsdp_param__"):
                            weight_param.main_grad = weight_param.get_main_grad()
                        fc1_weight_grads[idx] = weight_param.main_grad
                    accumulate_into_main_grad = not getattr(
                        fc1_op.weight0, "overwrite_main_grad", False
                    )
                else:
                    for idx in range(num_groups):
                        fc1_weight_grads[idx] = torch.empty(
                            fc1_weight_shape, dtype=dtype, device=device
                        )
                delay_fc1_wgrad = (
                    fc1_op.wgrad_store is not None and fc1_op.wgrad_store.delay_wgrad_compute()
                )
                if delay_fc1_wgrad:
                    fc1_wgrad_buffers = fc1_weight_grads
                    fc1_op.wgrad_store.put(
                        [grouped_fc1_x, grouped_fc1_dy, fc1_wgrad_buffers],
                        functools.partial(
                            general_grouped_gemm_for_grouped_tensor,
                            layout="NT",
                            accumulate=accumulate_into_main_grad,
                        ),
                    )
                    if accumulate_into_main_grad:
                        fc1_autograd_weight_grads = [
                            fc1_wgrad_buffers[i] for i in range(num_groups)
                        ]
                        for idx in range(num_groups):
                            weight_param = getattr(fc1_op, f"weight{idx}")
                            if hasattr(weight_param, "grad_added_to_main_grad"):
                                weight_param.grad_added_to_main_grad = True
                                fc1_autograd_weight_grads[idx] = get_dummy_wgrad(
                                    list(weight_param.size()),
                                    weight_param.dtype,
                                    zero=getattr(weight_param, "zero_out_wgrad", False),
                                )
                    else:
                        fc1_autograd_weight_grads = [None] * num_groups
                else:
                    general_grouped_gemm_for_grouped_tensor(
                        grouped_fc1_x,
                        grouped_fc1_dy,
                        fc1_weight_grads,
                        layout="NT",
                        accumulate=accumulate_into_main_grad,
                    )
                    if accumulate_into_main_grad:
                        for idx in range(num_groups):
                            weight_param = getattr(fc1_op, f"weight{idx}")
                            if hasattr(weight_param, "grad_added_to_main_grad"):
                                weight_param.grad_added_to_main_grad = True
                                fc1_weight_grads[idx] = get_dummy_wgrad(
                                    list(weight_param.size()),
                                    weight_param.dtype,
                                    zero=getattr(weight_param, "zero_out_wgrad", False),
                                )

        # Clear FC1 input tensor if possible
        if grouped_fc1_x is not None and not (
            fc1_ctx.weight_requires_grad
            and fc1_op.wgrad_store is not None
            and fc1_op.wgrad_store.delay_wgrad_compute()
        ):
            clear_tensor_data(
                grouped_fc1_x.data,
                grouped_fc1_x.columnwise_data,
                grouped_fc1_x.scale_inv,
                grouped_fc1_x.columnwise_scale_inv,
            )

        # Construct param grads in parameter registration order (see GroupedLinear.fuser_backward).
        if fc1_op.single_grouped_parameter:
            fc1_w_list = [fc1_packed_wgrad] if fc1_packed_wgrad is not None else [None]
        elif fc1_autograd_weight_grads is not None:
            fc1_w_list = fc1_autograd_weight_grads
        else:
            fc1_w_list = fc1_weight_grads
        if fc1_op.has_bias:
            if fc1_op.single_grouped_bias:
                if fc1_bias_grad_packed is not None:
                    fc1_gb = fc1_bias_grad_packed
                else:
                    fc1_gb = None
                fc1_grad_params = fc1_w_list + [fc1_gb]
            else:
                fc1_bias_list = (
                    fc1_bias_grads if fc1_bias_grads is not None else [None] * num_groups
                )
                if fc1_op.single_grouped_parameter:
                    fc1_grad_params = fc1_bias_list + fc1_w_list
                else:
                    fc1_grad_params = fc1_w_list + fc1_bias_list
        else:
            fc1_grad_params = fc1_w_list

        if fc2_op.single_grouped_parameter:
            fc2_w_list = [fc2_packed_wgrad] if fc2_packed_wgrad is not None else [None]
        elif fc2_autograd_weight_grads is not None:
            fc2_w_list = fc2_autograd_weight_grads
        else:
            fc2_w_list = fc2_weight_grads
        if fc2_op.has_bias:
            if fc2_op.single_grouped_bias:
                if fc2_bias_grad_packed is not None:
                    fc2_gb = fc2_bias_grad_packed
                else:
                    fc2_gb = None
                fc2_grad_params = fc2_w_list + [fc2_gb]
            else:
                fc2_bias_list = (
                    fc2_bias_grads if fc2_bias_grads is not None else [None] * num_groups
                )
                if fc2_op.single_grouped_parameter:
                    fc2_grad_params = fc2_bias_list + fc2_w_list
                else:
                    fc2_grad_params = fc2_w_list + fc2_bias_list
        else:
            fc2_grad_params = fc2_w_list

        return (
            grad_input,
            [fc1_grad_params, (), fc2_grad_params],
            [(None,), (grad_scales,), (None,)],
        )


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

    return fuse_grouped_mlp_ops(
        ops,
        recipe=recipe,
        fused_op_cls=BackwardGroupedMLP_CuTeGEMMDSwiGLU_MXFP8,
    )


# Register fusion if available
if BackwardGroupedMLP_CuTeGEMMDSwiGLU_MXFP8.is_supported():
    register_backward_fusion(fuse_backward_ops, prepend=True)
