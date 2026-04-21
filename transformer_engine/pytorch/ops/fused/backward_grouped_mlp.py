# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Fused operation for MoE grouped MLP."""

from __future__ import annotations
from collections.abc import Callable
import functools
import inspect
import math
import os
from typing import Optional

import torch

import transformer_engine_torch as tex
from ...module.base import get_dummy_wgrad
from ...quantization import Recipe
from ...tensor.grouped_tensor import GroupedTensor
from ...tensor.mxfp8_tensor import MXFP8Quantizer
from ...utils import clear_tensor_data, get_cached_ones_tensor, get_device_compute_capability
from ...constants import MXFP8_BLOCK_SCALING_SIZE
from ..basic import GroupedLinear, ScaledClampedQGeGLU, ScaledSwiGLU
from ..fuser import register_backward_fusion
from ..op import FusedOperation, FusibleOperation, OperationContext
from .._common import (
    _nvidia_cudnn_frontend_supports_wgrad,
    fuse_grouped_mlp_ops,
    maybe_dequantize,
    validate_grouped_mlp_dims,
)
from ...cpp_extensions import general_grouped_gemm_for_grouped_tensor
from ...module.base import _2X_ACC_WGRAD
from ...triton.grouped_dbias_dscales import _compute_grouped_dbias_dscales


def _cudnn_compute_wgrad(
    grouped_x: GroupedTensor,
    grouped_dy: GroupedTensor,
    wgrad_output,
    weight_shape: tuple,
    offsets: torch.Tensor,
    accumulate: bool,
    wgrad_kernel_fn,
    single_grouped_weight: bool,
    current_stream=None,
):
    """Compute wgrad using the cuDNN CuTe DSL grouped GEMM wgrad kernel.

    The cuDNN wgrad kernel computes:
        wgrad[e] = a[:, tok_start:tok_end] @ b[tok_start:tok_end, :]
    where a = DY^T = (out_features, total_tokens) row-major and
          b = X  = (total_tokens, in_features) column-major.
    """
    out_features, in_features = weight_shape
    total_tokens = grouped_dy.logical_shape[0]

    fp8_dtype = torch.float8_e4m3fn

    # a_tensor = DY^T = (out_features, total_tokens) row-major
    a_tensor = grouped_dy.columnwise_data.view(dtype=fp8_dtype).view(total_tokens, out_features).T
    # b_tensor = X = (total_tokens, in_features) column-major
    b_tensor = grouped_x.columnwise_data.view(dtype=fp8_dtype).view(total_tokens, in_features)

    sfa_leading_dim = ((out_features + 127) // 128) * 128
    sfb_leading_dim = ((in_features + 127) // 128) * 128
    sfa_tensor = grouped_dy.columnwise_scale_inv.view(sfa_leading_dim, -1).view(
        dtype=torch.float8_e8m0fnu
    )
    sfb_tensor = grouped_x.columnwise_scale_inv.view(sfb_leading_dim, -1).view(
        dtype=torch.float8_e8m0fnu
    )

    # Prepare wgrad output
    if single_grouped_weight:
        # Dense mode: single (num_groups, out_features, in_features) tensor
        wgrad_tensor = wgrad_output.rowwise_data.view(offsets.shape[0], out_features, in_features)
        wgrad_kernel_fn(
            a_tensor=a_tensor,
            b_tensor=b_tensor,
            sfa_tensor=sfa_tensor,
            sfb_tensor=sfb_tensor,
            offsets_tensor=offsets,
            output_mode="dense",
            wgrad_tensor=wgrad_tensor,
            acc_dtype=torch.float32,
            wgrad_dtype=wgrad_tensor.dtype,
            sf_vec_size=MXFP8_BLOCK_SCALING_SIZE,
            accumulate_on_output=accumulate,
            current_stream=current_stream,
        )
    else:
        # Discrete mode: per-expert wgrad device pointers
        (wgrad_ptrs,) = tex.convert_host_pointers_to_tensor([wgrad_output])
        wgrad_kernel_fn(
            a_tensor=a_tensor,
            b_tensor=b_tensor,
            sfa_tensor=sfa_tensor,
            sfb_tensor=sfb_tensor,
            offsets_tensor=offsets,
            output_mode="discrete",
            wgrad_ptrs=wgrad_ptrs,
            acc_dtype=torch.float32,
            wgrad_dtype=wgrad_output[0].dtype,
            sf_vec_size=MXFP8_BLOCK_SCALING_SIZE,
            accumulate_on_output=accumulate,
            current_stream=current_stream,
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


def _compute_grad_params(
    fc_op,
    ctx,
    num_groups,
    weight_shape,
    grouped_x,
    grouped_dy,
    dtype,
    device,
    bias_grads,
    bias_grad_packed,
    label="",
    *,
    cudnn_wgrad_kernel_fn,
    offsets,
):
    """Compute weight gradients and build grad_params for a GroupedLinear layer.
    Returns the grad_params list in parameter registration order.
    """

    # Allocate grad buffers, determine accumulate flag
    accumulate_into_main_grad = False
    grouped_wgrad = None
    wgrad_output = None
    if fc_op.single_grouped_weight:
        w_list = [None]
        if ctx.weight_requires_grad:
            weight_param = fc_op.weight
            if fc_op._accumulate_into_main_grad:
                if hasattr(weight_param, "__fsdp_param__"):
                    weight_param.main_grad = weight_param.get_main_grad()
                main_grad = weight_param.main_grad
                grouped_shape = (num_groups, *weight_shape)
                if main_grad.shape != grouped_shape:
                    if main_grad.numel() != math.prod(grouped_shape):
                        raise RuntimeError(
                            f"Grouped MLP fused backward expected {label} main_grad to have "
                            f"shape {grouped_shape} or matching numel, "
                            f"but got shape {tuple(main_grad.shape)}"
                        )
                    try:
                        main_grad = main_grad.view(grouped_shape)
                    except RuntimeError as e:
                        raise RuntimeError(
                            f"Grouped MLP fused backward requires {label} main_grad to be "
                            f"viewable as {grouped_shape} without copy, but got shape"
                            f" {tuple(main_grad.shape)} and stride"
                            f" {tuple(main_grad.stride())}"
                        ) from e
                accumulate_into_main_grad = not getattr(weight_param, "overwrite_main_grad", False)
                if accumulate_into_main_grad:
                    grouped_wgrad = GroupedTensor.make_grouped_tensor_from_rowwise_data(
                        num_tensors=num_groups,
                        tensor_shape=weight_shape,
                        rowwise_data=main_grad,
                        dtype=main_grad.dtype,
                    )

            if grouped_wgrad is None:
                grouped_wgrad = GroupedTensor.make_grouped_tensor_with_shapes(
                    num_tensors=num_groups,
                    shapes=[weight_shape] * num_groups,
                    quantizer=None,
                    device=device,
                    dtype=dtype,
                )
            wgrad_output = grouped_wgrad
    else:
        w_list = [None] * num_groups
        if ctx.weight_requires_grad:
            if fc_op._accumulate_into_main_grad:
                for idx in range(num_groups):
                    wp = getattr(fc_op, f"weight{idx}")
                    if hasattr(wp, "__fsdp_param__"):
                        wp.main_grad = wp.get_main_grad()
                    w_list[idx] = wp.main_grad
                accumulate_into_main_grad = not getattr(fc_op.weight0, "overwrite_main_grad", False)
            else:
                for idx in range(num_groups):
                    w_list[idx] = torch.empty(weight_shape, dtype=dtype, device=device)
            wgrad_output = w_list

    if ctx.weight_requires_grad:
        # Launch or defer the GEMM
        delay_wgrad = fc_op.wgrad_store is not None and fc_op.wgrad_store.delay_wgrad_compute()
        if cudnn_wgrad_kernel_fn is not None:
            offsets = offsets if offsets.dtype == torch.int32 else offsets.to(dtype=torch.int32)
            gemm_fn = functools.partial(
                _cudnn_compute_wgrad,
                weight_shape=weight_shape,
                offsets=offsets,
                accumulate=accumulate_into_main_grad,
                wgrad_kernel_fn=cudnn_wgrad_kernel_fn,
                single_grouped_weight=fc_op.single_grouped_weight,
                current_stream=torch.cuda.current_stream().cuda_stream,
            )
        else:
            gemm_fn = functools.partial(
                general_grouped_gemm_for_grouped_tensor,
                layout="NT",
                accumulate=accumulate_into_main_grad,
                use_split_accumulator=_2X_ACC_WGRAD,
            )

        if delay_wgrad:
            fc_op.wgrad_store.put([grouped_x, grouped_dy, wgrad_output], gemm_fn)
        else:
            gemm_fn(grouped_x, grouped_dy, wgrad_output)

        # Extract results, mark accumulated if needed
        if fc_op.single_grouped_weight:
            packed_wgrad = None
            if not delay_wgrad:
                packed_wgrad = grouped_wgrad.rowwise_data.view(num_groups, *weight_shape)
            if accumulate_into_main_grad and hasattr(weight_param, "grad_added_to_main_grad"):
                weight_param.grad_added_to_main_grad = True
                packed_wgrad = get_dummy_wgrad(
                    list(weight_param.size()),
                    weight_param.dtype,
                    zero=getattr(weight_param, "zero_out_wgrad", False),
                )
            w_list = [packed_wgrad]
        else:
            if delay_wgrad or accumulate_into_main_grad:
                w_list = [None] * num_groups
            if accumulate_into_main_grad:
                for idx in range(num_groups):
                    wp = getattr(fc_op, f"weight{idx}")
                    if hasattr(wp, "grad_added_to_main_grad"):
                        wp.grad_added_to_main_grad = True
                        w_list[idx] = get_dummy_wgrad(
                            list(wp.size()),
                            wp.dtype,
                            zero=getattr(wp, "zero_out_wgrad", False),
                        )

    # Assemble grad_params in parameter registration order.
    if not fc_op.has_bias:
        return w_list

    if fc_op.single_grouped_bias:
        return w_list + [bias_grad_packed]

    bias_list = bias_grads if bias_grads is not None else [None] * num_groups
    if fc_op.single_grouped_weight:
        return bias_list + w_list
    return w_list + bias_list


class BackwardGroupedMLP_CuTeGEMMDSwiGLU_MXFP8(FusedOperation):
    """Fused op for MXFP8 GroupedLinear + ScaledSwiGLU or ScaledClampedQGeGLU + GroupedLinear

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
    def grouped_gemm_wgrad_kernel(cls) -> Optional[Callable]:
        """CuTe DSL kernel for grouped GEMM wgrad on SM100+.
        Returns ``None`` when the cuDNN front-end package is older than
        1.23.0.
        """
        if not _nvidia_cudnn_frontend_supports_wgrad():
            return None
        from cudnn import grouped_gemm_wgrad_wrapper_sm100  # pylint: disable=no-name-in-module

        return grouped_gemm_wgrad_wrapper_sm100

    @classmethod
    @functools.lru_cache(maxsize=None)
    def is_supported(cls) -> bool:
        """Whether this fused operation is supported on the current system."""
        if int(os.environ.get("NVTE_CUTEDSL_FUSED_GROUPED_MLP", "0")) <= 0:
            return False
        if get_device_compute_capability()[0] != 10:
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
        swiglu: ScaledSwiGLU | ScaledClampedQGeGLU,
        fc2: GroupedLinear,
    ) -> None:
        super().__init__((fc1, swiglu, fc2))
        if not self.is_supported():
            self.grouped_gemm_dglu_kernel()  # Try triggering import error
            raise RuntimeError(f"{self.__class__.__name__} is not supported on this system.")
        validate_grouped_mlp_dims(fc1, swiglu, fc2)
        # The cuDNN dgeglu implementation corresponds to ScaledClampedQGeGLU.
        # The act_func string should be fixed on the cuDNN FE side.
        self._cudnn_dact_func: str = (
            "dgeglu" if isinstance(swiglu, ScaledClampedQGeGLU) else "dswiglu"
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
        fc1_weight_shape = (fc1_op.out_features, fc1_op.in_features)
        fc2_weight_shape = (fc2_op.out_features, fc2_op.in_features)
        grad_output = grad_output.reshape(-1, fc2_weight_shape[0])
        out_shape = list(grad_output.size())
        num_groups = fc1_op.num_groups
        fc1_weight_param = fc1_op.weight if fc1_op.single_grouped_weight else fc1_op.weight0
        device = fc1_weight_param.device
        dtype = fc1_ctx.dtype

        # Saved tensors from FC1 forward
        saved_tensors = fc1_ctx.saved_tensors
        split_sizes, split_points, saved_tensors = (
            saved_tensors[0],
            saved_tensors[1],
            saved_tensors[2:],
        )

        if fc1_op.single_grouped_weight:
            grouped_fc1_weight, saved_tensors = saved_tensors[0], saved_tensors[1:]
        else:
            grouped_fc1_weight, saved_tensors = (
                saved_tensors[:num_groups],
                saved_tensors[num_groups:],
            )

        (
            fc1_x_col_data,
            fc1_x_col_scale,
            fc1_x_tensor_offsets,
        ), saved_tensors = (
            saved_tensors[:3],
            saved_tensors[3:],
        )

        # Saved tensors from scaled SwiGLU forward
        swiglu_in, scales = swiglu_ctx.saved_tensors

        # Saved tensors from FC2 forward
        saved_tensors = fc2_ctx.saved_tensors
        _, saved_tensors = saved_tensors[0], saved_tensors[1:]  # Assume same split sizes as FC1
        if fc2_op.single_grouped_weight:
            grouped_fc2_weight, saved_tensors = saved_tensors[0], saved_tensors[1:]
        else:
            grouped_fc2_weight, saved_tensors = (
                saved_tensors[:num_groups],
                saved_tensors[num_groups:],
            )

        (
            fc2_x_col_data,
            fc2_x_col_scale,
            fc2_x_tensor_offsets,
        ), saved_tensors = (
            saved_tensors[:3],
            saved_tensors[3:],
        )

        # Group splits
        if int(split_sizes.numel()) != num_groups:
            raise ValueError(f"Expected {num_groups} splits, but got {int(split_sizes.numel())}.")
        scale_bias = fc2_op._scale_bias and fc2_op.has_bias

        grouped_fc1_x = None
        if fc1_ctx.weight_requires_grad:
            grouped_fc1_x = GroupedTensor(
                shape=(out_shape[0], fc1_weight_shape[1]),
                dtype=dtype,
                num_tensors=num_groups,
                quantizer=fc1_ctx.input_quantizer,
                columnwise_data=fc1_x_col_data,
                columnwise_scale_inv=fc1_x_col_scale,
                first_dims=split_sizes,
                tensor_offsets=fc1_x_tensor_offsets,
                with_gemm_swizzled_scales=True,
            )

        grouped_fc2_x = None
        if fc2_ctx.weight_requires_grad:
            grouped_fc2_x = GroupedTensor(
                shape=(out_shape[0], fc2_weight_shape[1]),
                dtype=dtype,
                num_tensors=num_groups,
                quantizer=fc2_ctx.input_quantizer,
                columnwise_data=fc2_x_col_data,
                columnwise_scale_inv=fc2_x_col_scale,
                first_dims=split_sizes,
                tensor_offsets=fc2_x_tensor_offsets,
                with_gemm_swizzled_scales=True,
            )

        # Split grad output tensor and convert dtypes if needed
        fc2_ctx.grad_output_quantizer.set_usage(
            rowwise=True, columnwise=fc2_ctx.weight_requires_grad
        )
        fc2_ctx.grad_output_quantizer.optimize_for_gemm = True
        output_fc2_dbias = fc2_op.has_bias
        fc2_dbias_packed = None
        fc2_dy = None
        if (
            not output_fc2_dbias
            and isinstance(grad_output, GroupedTensor)
            and isinstance(getattr(grad_output, "quantizer", None), MXFP8Quantizer)
        ):
            grouped_fc2_dy = grad_output
        else:
            fc2_dy = maybe_dequantize(grad_output, dtype)
            if output_fc2_dbias and not scale_bias:
                grouped_fc2_dy, fc2_dbias_packed = tex.bgrad_group_quantize(
                    fc2_dy,
                    fc2_ctx.grad_output_quantizer,
                    num_groups,
                    split_sizes,
                )
            else:
                grouped_fc2_dy = tex.group_quantize(
                    fc2_dy,
                    fc2_ctx.grad_output_quantizer,
                    num_groups,
                    split_sizes,
                )

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
            (out_shape[0] + 127) // 128,
            (out_shape[1] + 127) // 128,
            MXFP8_BLOCK_SCALING_SIZE,
            4,
            4,
        )
        fc2_dy_scales = fc2_dy_scales.permute(3, 4, 1, 5, 2, 0)

        # Kernel scaling factors
        alpha_tensor = get_cached_ones_tensor(num_groups, dtype, device)
        norm_const_tensor = get_cached_ones_tensor(1, dtype, device)
        current_stream = torch.cuda.current_stream().cuda_stream

        scales_f32 = scales.detach().to(dtype=torch.float32)
        scales_tensor = scales_f32.reshape(-1, 1, 1)
        dscales_tensor = torch.zeros_like(scales_tensor)

        fc2_dglu_kwargs = {
            "a_tensor": fc2_dy_data,
            "c_tensor": swiglu_in.unsqueeze(0).permute(1, 2, 0),
            "sfa_tensor": fc2_dy_scales,
            "padded_offsets": split_points,
            "alpha_tensor": alpha_tensor,
            "beta_tensor": alpha_tensor,
            "prob_tensor": scales_tensor,
            "dprob_tensor": dscales_tensor,
            "generate_dbias": fc1_op.has_bias,
            "norm_const_tensor": norm_const_tensor,
            "d_dtype": torch.float8_e4m3fn,
            "cd_major": "n",
            "sf_vec_size": MXFP8_BLOCK_SCALING_SIZE,
            "current_stream": current_stream,
            "discrete_col_sfd": True,
            "act_func": self._cudnn_dact_func,
            "use_dynamic_sched": True,
        }

        if fc2_op.single_grouped_weight:
            # Clone and swizzle scales for GEMM
            fc2_weight_for_gemm = grouped_fc2_weight.copy()
            tex.grouped_swizzle_for_gemm(fc2_weight_for_gemm, rowwise=False, columnwise=True)
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
                (fc2_weight_shape[1] + 127) // 128,
                (fc2_weight_shape[0] + 127) // 128,
                MXFP8_BLOCK_SCALING_SIZE,
                4,
                4,
            )
            fc2_w_scales = fc2_w_scales.permute(3, 4, 1, 5, 2, 0)

            fc2_dglu_kwargs["b_tensor"] = fc2_w_data
            fc2_dglu_kwargs["sfb_tensor"] = fc2_w_scales
        else:
            fc2_b_ptrs, fc2_sfb_ptrs, _fc2_sw = tex.get_device_pointer_for_data_and_scales(
                [w._columnwise_data for w in grouped_fc2_weight],
                [w._columnwise_scale_inv for w in grouped_fc2_weight],
                swizzle=True,
                rowwise=False,
                data_dtype=grouped_fc2_weight[0]._fp8_dtype,
            )
            fc2_dglu_kwargs["b_ptrs"] = fc2_b_ptrs
            fc2_dglu_kwargs["sfb_ptrs"] = fc2_sfb_ptrs
            fc2_dglu_kwargs["n"] = fc2_weight_shape[1]
            fc2_dglu_kwargs["b_dtype"] = torch.float8_e4m3fn
            fc2_dglu_kwargs["b_major"] = "n"

        fc2_dgrad_kernel_out = self.grouped_gemm_dglu_kernel()(**fc2_dglu_kwargs)

        fc1_dy_row_data = fc2_dgrad_kernel_out["d_row_tensor"]
        fc1_dy_row_data = fc1_dy_row_data.view(out_shape[0], fc1_weight_shape[0])
        # View scale in their actual swizzled shape
        fc1_dy_row_scale = fc2_dgrad_kernel_out["sfd_row_tensor"].permute(5, 2, 4, 0, 1, 3).view(-1)
        fc1_dy_col_data = fc2_dgrad_kernel_out["d_col_tensor"]
        fc1_dy_col_data = fc1_dy_col_data.view(out_shape[0], fc1_weight_shape[0])
        # View scale in their actual swizzled shape
        fc1_dy_col_scale = fc2_dgrad_kernel_out["sfd_col_tensor"].permute(5, 2, 4, 0, 1, 3).view(-1)
        grad_scales = fc2_dgrad_kernel_out["dprob_tensor"].view(-1)

        fc2_bias_grads: Optional[list[Optional[torch.Tensor]]] = None
        fc2_bias_grad_packed: Optional[torch.Tensor] = None
        if scale_bias:
            fc2_biases = fc2_op._get_bias_tensors(dtype)
            bias_packed = torch.stack(fc2_biases)
            fc2_dbias_packed_result, grad_scales = _compute_grouped_dbias_dscales(
                fc2_dy,
                scales_f32,
                bias_packed,
                offsets=fc1_ctx.base_split_offsets,
                dscales=grad_scales,
            )
            fc2_dbias_packed_result = fc2_dbias_packed_result.to(dtype=dtype)
            if fc2_op.single_grouped_bias:
                fc2_bias_grad_packed = fc2_dbias_packed_result
            else:
                fc2_bias_grads = [fc2_dbias_packed_result[idx] for idx in range(num_groups)]
        elif fc2_dbias_packed is not None:
            fc2_dbias_packed = fc2_dbias_packed.to(dtype=dtype)
            if fc2_op.single_grouped_bias:
                fc2_bias_grad_packed = fc2_dbias_packed
            else:
                fc2_bias_grads = [fc2_dbias_packed[idx] for idx in range(num_groups)]

        grad_scales = grad_scales.to(dtype=dtype)

        fc1_bias_grads: Optional[list[Optional[torch.Tensor]]] = None
        fc1_bias_grad_packed: Optional[torch.Tensor] = None
        if fc1_op.has_bias:
            dbias_t = fc2_dgrad_kernel_out["dbias_tensor"]
            if dbias_t is not None:
                dbias_2d = dbias_t.squeeze(-1).to(dtype=dtype)
                if fc1_op.single_grouped_bias:
                    fc1_bias_grad_packed = dbias_2d
                else:
                    fc1_bias_grads = [dbias_2d[group_idx] for group_idx in range(num_groups)]

        # FC1 grad output for dgrad and wgrad GEMMs
        fc1_dy_tensor_offsets = fc1_ctx.base_split_offsets * fc1_weight_shape[0]
        grouped_fc1_dy = GroupedTensor(
            shape=(out_shape[0], fc1_weight_shape[0]),
            dtype=dtype,
            num_tensors=num_groups,
            quantizer=fc1_ctx.grad_output_quantizer,
            data=fc1_dy_row_data,
            columnwise_data=fc1_dy_col_data,
            scale_inv=fc1_dy_row_scale,
            columnwise_scale_inv=fc1_dy_col_scale,
            first_dims=split_sizes,
            tensor_offsets=fc1_dy_tensor_offsets,
            with_gemm_swizzled_scales=True,
        )

        # FC2 wgrad GEMM
        fc2_grad_params = _compute_grad_params(
            fc_op=fc2_op,
            ctx=fc2_ctx,
            num_groups=num_groups,
            weight_shape=fc2_weight_shape,
            grouped_x=grouped_fc2_x,
            grouped_dy=grouped_fc2_dy,
            dtype=dtype,
            device=device,
            bias_grads=fc2_bias_grads,
            bias_grad_packed=fc2_bias_grad_packed,
            label="FC2",
            cudnn_wgrad_kernel_fn=self.grouped_gemm_wgrad_kernel(),
            offsets=split_points,
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

            fc1_dgrad_a_data = fc2_dgrad_kernel_out["d_row_tensor"]
            fc1_dgrad_a_scales = fc2_dgrad_kernel_out["sfd_row_tensor"]

            fc1_dgrad_kwargs = {
                "a_tensor": fc1_dgrad_a_data,
                "sfa_tensor": fc1_dgrad_a_scales,
                "padded_offsets": split_points,
                "alpha_tensor": alpha_tensor.float(),
                "norm_const_tensor": None,
                "prob_tensor": torch.ones((out_shape[0], 1, 1), dtype=torch.float32, device=device),
                "acc_dtype": torch.float32,
                "c_dtype": dtype,
                "d_dtype": dtype,
                "cd_major": "n",
                "sf_vec_size": MXFP8_BLOCK_SCALING_SIZE,
                "current_stream": current_stream,
                "discrete_col_sfd": True,
                "use_dynamic_sched": True,
            }

            if fc1_op.single_grouped_weight:
                # Clone and swizzle scales for GEMM
                fc1_weight_for_gemm = grouped_fc1_weight.copy()
                tex.grouped_swizzle_for_gemm(fc1_weight_for_gemm, rowwise=False, columnwise=True)

                fc1_w_data = fc1_weight_for_gemm.columnwise_data
                fc1_w_data = fc1_w_data.view(dtype=torch.float8_e4m3fn)
                fc1_w_data = fc1_w_data.view(num_groups, fc1_weight_shape[0], fc1_weight_shape[1])
                fc1_w_data = fc1_w_data.permute(2, 1, 0)
                fc1_w_scales = fc1_weight_for_gemm.columnwise_scale_inv.view(
                    dtype=torch.float8_e8m0fnu
                )
                fc1_w_scales = fc1_w_scales.view(
                    num_groups,
                    (fc1_weight_shape[1] + 127) // 128,
                    (fc1_weight_shape[0] + 127) // 128,
                    MXFP8_BLOCK_SCALING_SIZE,
                    4,
                    4,
                )
                fc1_w_scales = fc1_w_scales.permute(3, 4, 1, 5, 2, 0)

                fc1_dgrad_kwargs["b_tensor"] = fc1_w_data
                fc1_dgrad_kwargs["sfb_tensor"] = fc1_w_scales
            else:
                fc1_b_ptrs, fc1_sfb_ptrs, _ = tex.get_device_pointer_for_data_and_scales(
                    [w._columnwise_data for w in grouped_fc1_weight],
                    [w._columnwise_scale_inv for w in grouped_fc1_weight],
                    swizzle=True,
                    rowwise=False,
                    data_dtype=grouped_fc1_weight[0]._fp8_dtype,
                )

                fc1_dgrad_kwargs["b_ptrs"] = fc1_b_ptrs
                fc1_dgrad_kwargs["sfb_ptrs"] = fc1_sfb_ptrs
                fc1_dgrad_kwargs["n"] = fc1_weight_shape[1]
                fc1_dgrad_kwargs["b_dtype"] = torch.float8_e4m3fn
                fc1_dgrad_kwargs["b_major"] = "n"

            fc1_dgrad_kernel_out = self.grouped_gemm_quant_kernel()(**fc1_dgrad_kwargs)
            grad_input = fc1_dgrad_kernel_out["d_tensor"].view(in_shape)

        # FC1 wgrad GEMM
        fc1_grad_params = _compute_grad_params(
            fc_op=fc1_op,
            ctx=fc1_ctx,
            num_groups=num_groups,
            weight_shape=fc1_weight_shape,
            grouped_x=grouped_fc1_x,
            grouped_dy=grouped_fc1_dy,
            dtype=dtype,
            device=device,
            bias_grads=fc1_bias_grads,
            bias_grad_packed=fc1_bias_grad_packed,
            label="FC1",
            cudnn_wgrad_kernel_fn=self.grouped_gemm_wgrad_kernel(),
            offsets=split_points,
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

        fc2_grad_extra = (None, None) if fc2_op._scale_bias else (None,)
        return (
            grad_input,
            [fc1_grad_params, (), fc2_grad_params],
            [(None,), (grad_scales,), fc2_grad_extra],
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
