# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Fused operation for MoE grouped MLP."""

from __future__ import annotations
from collections.abc import Callable, Iterable
import functools
import os
from typing import Any, Optional

import torch

import transformer_engine_torch as tex
from ...cpp_extensions import general_gemm, general_grouped_gemm_for_grouped_tensor
from ...quantization import Recipe
from ...tensor import NVFP4Quantizer, NVFP4Tensor, Quantizer
from ...utils import (
    ceil_div,
    get_cached_ones_tensor,
    get_device_compute_capability,
    mark_grouped_tensor,
)
from ...tensor.grouped_tensor import GroupedTensor
from ...tensor.mxfp8_tensor import MXFP8Quantizer
from ...constants import MXFP8_BLOCK_SCALING_SIZE, NVFP4_BLOCK_SCALING_SIZE
from ..basic import GroupedLinear, ScaledSReLU, ScaledClampedQGeGLU
from ..fuser import register_forward_fusion
from ..op import FusedOperation, FusibleOperation, OperationContext
from .._common import (
    _cudnn_frontend_geglu_runtime_params,
    _cudnn_frontend_version_supported,
    _cudnn_frontend_supports_grouped_gemm_srelu,
    _group_quantize_for_grouped_mlp,
    _nvidia_cudnn_frontend_supports_wgrad,
    _nvfp4_amax,
    _nvfp4_single_tensor_from_grouped,
    fuse_grouped_mlp_ops,
    is_glu_activation,
    is_quantized_tensor,
    maybe_dequantize,
    validate_grouped_mlp_dims,
)


def _pack_grouped_linear_bias_for_cudnn(linear_op: GroupedLinear) -> Optional[torch.Tensor]:
    """Bias layout expected by cuDNN grouped GEMM: shape (n, num_groups), stride (1, n)."""
    if not linear_op.has_bias:
        return None
    num_groups = linear_op.num_groups
    grouped_bias = getattr(linear_op, "bias", None)
    if grouped_bias is not None:
        packed = grouped_bias.rowwise_data.view(num_groups, -1)
        return packed.transpose(0, 1)
    rows = [getattr(linear_op, f"bias{group_idx}") for group_idx in range(num_groups)]
    # stack to [num_groups, n] but cuDNN expects [n, num_groups] with stride [1, n].
    return torch.stack(rows, dim=0).transpose(0, 1)


@functools.lru_cache(maxsize=1)
def _grouped_gemm_dsrelu_backward_supported() -> bool:
    """Whether the cuDNN FE grouped GEMM dSReLU backward wrapper is available."""
    if int(os.environ.get("NVTE_CUTEDSL_FUSED_GROUPED_MLP", "0")) <= 0:
        return False
    if get_device_compute_capability()[0] != 10:
        return False
    if not _cudnn_frontend_supports_grouped_gemm_srelu():
        return False
    try:
        from cudnn import (
            grouped_gemm_dsrelu_wrapper_sm100,
        )  # pylint: disable=import-outside-toplevel
    except ImportError:
        return False
    return grouped_gemm_dsrelu_wrapper_sm100 is not None


class _ForwardGroupedMLP_CuTeGEMMBase(FusedOperation):
    """Base fused op for block-scaled GroupedLinear + activation + GroupedLinear.

    Uses experimental CuTe DSL kernel from cuDNN front-end.

    """

    @classmethod
    def grouped_gemm_activation_kernel(cls) -> Callable:
        """Fused kernel for grouped GEMM, activation, and post-multiplication."""
        raise NotImplementedError

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
        if int(os.environ.get("NVTE_CUTEDSL_FUSED_GROUPED_MLP", "0")) <= 0:
            return False
        if get_device_compute_capability()[0] != 10:
            return False
        if not _cudnn_frontend_version_supported():
            return False
        try:
            cls.grouped_gemm_activation_kernel()
            cls.grouped_gemm_quant_kernel()
        except ImportError:
            return False
        return True

    def __init__(
        self,
        *,
        fc1: GroupedLinear,
        activation: Optional[FusibleOperation],
        fc2: GroupedLinear,
    ) -> None:
        if activation is None:
            raise TypeError("Expected a grouped MLP activation op.")
        super().__init__((fc1, activation, fc2))
        if not self.is_supported():
            self.grouped_gemm_activation_kernel()  # Try triggering import error
            raise RuntimeError(f"{self.__class__.__name__} is not supported on this system.")
        validate_grouped_mlp_dims(fc1, activation, fc2)
        if not is_glu_activation(activation):
            # grouped_gemm_srelu_wrapper_sm100 is SReLU-specific and does not
            # take the GLU ``act_func`` selector.
            self._cudnn_act_func: Optional[str] = None
        else:
            # The cuDNN geglu implementation corresponds to ScaledClampedQGeGLU.
            # The act_func string should be fixed on the cuDNN FE side.
            self._cudnn_act_func = (
                "geglu" if isinstance(activation, ScaledClampedQGeGLU) else "swiglu"
            )

        # cuDNN-frontend >= 1.24.0 exposes runtime-configurable GeGLU
        # parameters; pass them through when the activation carries
        # non-default values (or always, if available).
        self._pass_geglu_runtime_params: bool = (
            isinstance(activation, ScaledClampedQGeGLU) and _cudnn_frontend_geglu_runtime_params()
        )
        if self._pass_geglu_runtime_params:
            self._cudnn_linear_offset: float = activation._clamped.glu_linear_offset
            self._cudnn_geglu_alpha: float = activation._clamped.alpha
            self._cudnn_glu_clamp_max: float = activation._clamped.limit
            self._cudnn_glu_clamp_min: float = -activation._clamped.limit

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
        fc1_op, _, fc2_op = self.basic_ops
        fc1_ctx, activation_ctx, fc2_ctx = basic_op_ctxs

        # Tensor properties
        fc1_weight_shape = (fc1_op.out_features, fc1_op.in_features)
        fc2_weight_shape = (fc2_op.out_features, fc2_op.in_features)
        input_ = input_.reshape(-1, fc1_weight_shape[1])
        in_shape = list(input_.size())
        assert in_shape[0] % 128 == 0, "Unsupported input shape for fused grouped MLP."

        num_groups = fc1_op.num_groups
        fc1_weight_param = fc1_op.weight if fc1_op.single_grouped_weight else fc1_op.weight0
        fc2_weight_param = fc2_op.weight if fc2_op.single_grouped_weight else fc2_op.weight0
        device = fc1_weight_param.device
        if torch.is_autocast_enabled():
            dtype = torch.get_autocast_dtype("cuda")
        else:
            dtype = fc1_weight_param.dtype

        # Check which grads are required
        requires_grad = any(ctx.requires_grad for ctx in basic_op_ctxs)
        input_requires_grad = requires_grad
        weight_requires_grad = requires_grad and (
            fc1_weight_param.requires_grad or fc2_weight_param.requires_grad
        )

        # Quantizers
        fc1_input_quantizer = fc1_op.get_quantizer("forward", 0)
        fc1_weight_quantizer = fc1_op.get_quantizer("forward", 1)
        fc1_grad_output_quantizer = fc1_op.get_quantizer("backward", 0)
        fc2_input_quantizer = fc2_op.get_quantizer("forward", 0)
        fc2_weight_quantizer = fc2_op.get_quantizer("forward", 1)
        fc2_grad_output_quantizer = fc2_op.get_quantizer("backward", 0)

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
        if int(split_sizes.numel()) != num_groups:
            raise ValueError(f"Expected {num_groups} splits, but got {int(split_sizes.numel())}.")
        split_sizes = split_sizes.to(dtype=torch.int64, device=device)
        base_split_offsets = tex.splits_to_offsets(split_sizes, 1)
        split_points = base_split_offsets[1:].to(dtype=torch.int)
        fc1_x_tensor_offsets = base_split_offsets * fc1_weight_shape[1]
        fc2_x_tensor_offsets = base_split_offsets * fc2_weight_shape[1]

        # Extract per-row activation probabilities from the middle op.
        scales = basic_op_extra_inputs[1][0]

        # Prepare FC1 grouped weight tensor for fused kernels.
        #  - single_grouped_weight=True: op.weight is already a GroupedTensor
        #  - single_grouped_weight=False: cute DSL kernel works with discrete weight tensors
        #   as long as host pointers for addresses are packed as contiguous device tensor.
        if fc1_op.single_grouped_weight:
            if not isinstance(fc1_op.weight, GroupedTensor):
                raise RuntimeError(
                    "FC1 expected GroupedTensor weight with single_grouped_weight=True."
                )
            if fc1_op.weight.quantizer is not None:
                fc1_weight_quantizer.set_usage(rowwise=True, columnwise=input_requires_grad)
                fc1_op.weight.quantizer = fc1_weight_quantizer
                grouped_fc1_weight = fc1_op.weight
            else:
                if fc1_op.weight.rowwise_data is None:
                    raise RuntimeError("FC1 grouped weight has no rowwise_data to quantize.")
                fc1_weight_quantizer.set_usage(rowwise=True, columnwise=input_requires_grad)
                grouped_fc1_weight = _group_quantize_for_grouped_mlp(
                    fc1_op.weight.rowwise_data.view(fc1_op.weight.logical_shape),
                    fc1_weight_quantizer,
                    num_groups,
                    None,
                )
        else:
            fc1_weights = [getattr(fc1_op, f"weight{idx}") for idx in range(num_groups)]
            quantized_fc1_weights = []
            for idx, weight in enumerate(fc1_weights):
                quantizer = fc1_op.get_quantizer("forward", 2 * idx + 1)
                if not is_quantized_tensor(weight):
                    quantizer.set_usage(rowwise=True, columnwise=input_requires_grad)
                    quantized_fc1_weights.append(quantizer(weight))
                else:
                    quantized_fc1_weights.append(weight)
            grouped_fc1_weight = quantized_fc1_weights

        # Prepare FC2 grouped weight tensor for fused kernels.
        if fc2_op.single_grouped_weight:
            if not isinstance(fc2_op.weight, GroupedTensor):
                raise RuntimeError(
                    "FC2 expected GroupedTensor weight with single_grouped_weight=True."
                )
            if fc2_op.weight.quantizer is not None:
                fc2_weight_quantizer.set_usage(rowwise=True, columnwise=input_requires_grad)
                fc2_op.weight.quantizer = fc2_weight_quantizer
                grouped_fc2_weight = fc2_op.weight
            else:
                if fc2_op.weight.rowwise_data is None:
                    raise RuntimeError("FC2 grouped weight has no rowwise_data to quantize.")
                fc2_weight_quantizer.set_usage(rowwise=True, columnwise=input_requires_grad)
                grouped_fc2_weight = _group_quantize_for_grouped_mlp(
                    fc2_op.weight.rowwise_data.view(fc2_op.weight.logical_shape),
                    fc2_weight_quantizer,
                    num_groups,
                    None,
                )
        else:
            fc2_weights = [getattr(fc2_op, f"weight{idx}") for idx in range(num_groups)]
            quantized_fc2_weights = []
            for idx, weight in enumerate(fc2_weights):
                quantizer = fc2_op.get_quantizer("forward", 2 * idx + 1)
                quantizer.set_usage(rowwise=True, columnwise=input_requires_grad)
                if not is_quantized_tensor(weight):
                    quantizer.set_usage(rowwise=True, columnwise=input_requires_grad)
                    quantized_fc2_weights.append(quantizer(weight))
                else:
                    quantized_fc2_weights.append(weight)
            grouped_fc2_weight = quantized_fc2_weights

        # Some wrapper-copy paths may drop grouped storage metadata; enforce defaults.
        if isinstance(grouped_fc1_weight, GroupedTensor) and not hasattr(
            grouped_fc1_weight, "_with_gemm_swizzled_scales"
        ):
            grouped_fc1_weight._with_gemm_swizzled_scales = False
        if isinstance(grouped_fc2_weight, GroupedTensor) and not hasattr(
            grouped_fc2_weight, "_with_gemm_swizzled_scales"
        ):
            grouped_fc2_weight._with_gemm_swizzled_scales = False

        # Group-quantize input tensor and convert dtypes if needed
        fc1_input_quantizer.set_usage(rowwise=True, columnwise=weight_requires_grad)
        fc1_input_quantizer.optimize_for_gemm = True
        input_quantizer = getattr(input_, "quantizer", None)
        if isinstance(input_, GroupedTensor) and (
            isinstance(fc1_input_quantizer, MXFP8Quantizer)
            and isinstance(input_quantizer, MXFP8Quantizer)
            or isinstance(fc1_input_quantizer, NVFP4Quantizer)
            and isinstance(input_quantizer, NVFP4Quantizer)
        ):
            grouped_fc1_x = input_
        else:
            fc1_x = maybe_dequantize(input_, dtype)
            grouped_fc1_x = _group_quantize_for_grouped_mlp(
                fc1_x,
                fc1_input_quantizer,
                num_groups,
                split_sizes,
                tensor_offsets=fc1_x_tensor_offsets,
            )

        use_nvfp4 = isinstance(fc1_input_quantizer, NVFP4Quantizer) or isinstance(
            fc1_weight_param, NVFP4Tensor
        )
        data_dtype = torch.float4_e2m1fn_x2 if use_nvfp4 else torch.float8_e4m3fn
        scale_view_dtype = torch.float8_e4m3fn if use_nvfp4 else torch.float8_e8m0fnu
        sf_vec_size = NVFP4_BLOCK_SCALING_SIZE if use_nvfp4 else MXFP8_BLOCK_SCALING_SIZE
        data_in_k = in_shape[1] // 2 if use_nvfp4 else in_shape[1]
        fc1_weight_k = fc1_weight_shape[1] // 2 if use_nvfp4 else fc1_weight_shape[1]
        k_sf_divisor = 2 * sf_vec_size if use_nvfp4 else 4 * sf_vec_size

        # Pack data tensors
        # Note: Fused kernel expects tensor with non-contiguous
        # logical dims.
        # Data actual shape: (1, sum(m), k)
        # Scale actual shape: (1, sum(m)/128, k/128, 32 (block row),
        #  4 (block row), 4 (block col))
        # Data logical shape: (sum(m), k, 1)
        # Scale logical shape: (32 (block row), 4 (block row),
        #   sum(m)/128, 4 (block col), k/128, 1)
        fc1_x_data = grouped_fc1_x.rowwise_data.view(dtype=data_dtype)
        fc1_x_data = fc1_x_data.view(in_shape[0], data_in_k)
        fc1_x_data = fc1_x_data.unsqueeze(0).permute(1, 2, 0)
        fc1_x_scales = grouped_fc1_x.scale_inv
        fc1_x_scales = fc1_x_scales.view(dtype=scale_view_dtype)
        with_gemm_swizzled_scales = grouped_fc1_x._with_gemm_swizzled_scales
        if use_nvfp4 and with_gemm_swizzled_scales:
            fc1_x_scales = fc1_x_scales.view(
                1,
                ceil_div(in_shape[0], 128),
                ceil_div(data_in_k, k_sf_divisor),
                32,
                4,
                4,
            )
            fc1_x_scales = fc1_x_scales.permute(3, 4, 1, 5, 2, 0)
        elif use_nvfp4:
            fc1_x_scales = fc1_x_scales.view(
                1,
                ceil_div(in_shape[0], 128),
                4,
                32,
                ceil_div(data_in_k, k_sf_divisor),
                4,
            )
            fc1_x_scales = fc1_x_scales.permute(3, 2, 1, 5, 4, 0)
        else:
            fc1_x_scales = fc1_x_scales.view(
                1,
                ceil_div(in_shape[0], 128),
                ceil_div(in_shape[1], k_sf_divisor),
                32,
                4,
                4,
            )
            fc1_x_scales = fc1_x_scales.permute(3, 4, 1, 5, 2, 0)

        alpha_tensor = get_cached_ones_tensor(num_groups, dtype, device)
        norm_const_tensor = get_cached_ones_tensor(1, torch.float32, device)
        current_stream = torch.cuda.current_stream().cuda_stream

        fc1_bias_packed = _pack_grouped_linear_bias_for_cudnn(fc1_op)
        fc2_bias_packed = _pack_grouped_linear_bias_for_cudnn(fc2_op)

        fc1_d_dtype = torch.bfloat16 if use_nvfp4 else torch.float8_e4m3fn
        fc1_prob_tensor = (
            scales.detach().to(dtype=torch.float32 if use_nvfp4 else dtype).reshape(-1, 1, 1)
        )
        fc1_norm_const_tensor = None if use_nvfp4 else norm_const_tensor
        if use_nvfp4:
            nvfp4_fp4_max = 6.0
            nvfp4_fp8_max = 448.0
            nvfp4_global_scale_denom = nvfp4_fp4_max * nvfp4_fp8_max
            # cuDNN receives NVFP4 block-scaled inputs without TE's per-group
            # global scale factors, so alpha supplies the product of the two
            # operand global scales.
            fc1_alpha_tensor = (
                _nvfp4_amax(grouped_fc1_x, columnwise=False)
                * _nvfp4_amax(grouped_fc1_weight, columnwise=False)
                / (nvfp4_global_scale_denom**2)
            ).to(torch.float32)
        else:
            fc1_alpha_tensor = alpha_tensor

        fc1_activation_kwargs = {
            "a_tensor": fc1_x_data,
            "sfa_tensor": fc1_x_scales,
            "padded_offsets": split_points,
            "alpha_tensor": fc1_alpha_tensor,
            "bias_tensor": fc1_bias_packed,
            "norm_const_tensor": fc1_norm_const_tensor,
            "prob_tensor": fc1_prob_tensor,
            "acc_dtype": torch.float32,
            "c_dtype": torch.bfloat16,
            "d_dtype": fc1_d_dtype,
            "cd_major": "n",
            "sf_vec_size": sf_vec_size,
            "current_stream": current_stream,
            "discrete_col_sfd": not use_nvfp4,
            "use_dynamic_sched": True,
        }
        if self._cudnn_act_func is not None:
            fc1_activation_kwargs["act_func"] = self._cudnn_act_func
        if self._pass_geglu_runtime_params:
            fc1_activation_kwargs.update(
                linear_offset=self._cudnn_linear_offset,
                geglu_alpha=self._cudnn_geglu_alpha,
                glu_clamp_max=self._cudnn_glu_clamp_max,
                glu_clamp_min=self._cudnn_glu_clamp_min,
            )

        if fc1_op.single_grouped_weight:
            # Clone and swizzle scales for GEMM.
            fc1_weight_for_gemm = grouped_fc1_weight.copy()
            tex.grouped_swizzle_for_gemm(fc1_weight_for_gemm, rowwise=True, columnwise=False)

            # Pack weight tensors for stacked kernel
            # Data actual shape: (num_groups, n, k)
            # Data logical shape: (n, k, num_groups)
            fc1_w_data = fc1_weight_for_gemm.rowwise_data
            fc1_w_data = fc1_w_data.view(dtype=data_dtype)
            fc1_w_data = fc1_w_data.view(num_groups, fc1_weight_shape[0], fc1_weight_k)
            fc1_w_data = fc1_w_data.permute(1, 2, 0)
            fc1_w_scales = fc1_weight_for_gemm.scale_inv.view(dtype=scale_view_dtype)
            fc1_w_scales = fc1_w_scales.view(
                num_groups,
                ceil_div(fc1_weight_shape[0], 128),
                ceil_div(fc1_weight_shape[1], k_sf_divisor),
                32,
                4,
                4,
            )
            fc1_w_scales = fc1_w_scales.permute(3, 4, 1, 5, 2, 0)

            fc1_activation_kwargs["b_tensor"] = fc1_w_data
            fc1_activation_kwargs["sfb_tensor"] = fc1_w_scales
        else:
            # Discrete-weight kernel: per-expert data/scale pointers
            fc1_b_ptrs, fc1_sfb_ptrs, _fc1_sfb_buffer = (
                tex.grouped_mlp_experimental.swizzle_scales_and_pack_ptrs_for_discrete_weights(
                    [w._rowwise_data for w in grouped_fc1_weight],
                    [w._rowwise_scale_inv for w in grouped_fc1_weight],
                    "nvfp4" if use_nvfp4 else "mxfp8_rowwise",
                    device,
                )
            )
            fc1_activation_kwargs["b_ptrs"] = fc1_b_ptrs
            fc1_activation_kwargs["sfb_ptrs"] = fc1_sfb_ptrs
            fc1_activation_kwargs["n"] = fc1_weight_shape[0]
            fc1_activation_kwargs["b_dtype"] = data_dtype
            fc1_activation_kwargs["b_major"] = "k"

        fc1_kernel_out = self.grouped_gemm_activation_kernel()(**fc1_activation_kwargs)

        # Unpack kernel outputs
        # Note: Fused kernel outputs tensors with non-contiguous
        # logical dims.
        # Row-wise data logical shape: (sum(m_splits), k, 1)
        # Row-wise scale logical shape: (32 (block row), 4 (block row),
        #   sum(m_splits)/128, 4 (block col), k/128, 1)
        # Column-wise data logical shape: (sum(m_splits), k, 1)
        # Column-wise scale logical shape: (32 (block col), 4 (block col),
        #   k/128, 4 (block row), sum(m_splits)/128, 1)
        activation_in = fc1_kernel_out["c_tensor"]
        activation_in = activation_in.view(in_shape[0], fc1_weight_shape[0])

        # FC2 GEMM
        fc2_out_shape = in_shape[:-1] + [fc2_weight_shape[0]]
        fc2_scales = basic_op_extra_inputs[2][1] if fc2_op._scale_bias else None

        if use_nvfp4:
            fc2_bias_for_gemm = None
            fc2_bias_scale = None
            if fc2_bias_packed is not None:
                fc2_bias_for_gemm = fc2_op._get_grouped_bias_for_gemm(dtype)
                if fc2_scales is not None:
                    fc2_bias_scale = fc2_scales.reshape(-1)
                    if fc2_bias_scale.dtype != torch.float32:
                        fc2_bias_scale = fc2_bias_scale.to(dtype=torch.float32)

            fc2_in = fc1_kernel_out["d_tensor"]
            fc2_in = fc2_in.view(in_shape[0], fc2_weight_shape[1]).contiguous()
            fc2_input_quantizer.set_usage(rowwise=True, columnwise=weight_requires_grad)
            fc2_input_quantizer.optimize_for_gemm = True
            grouped_fc2_x = _group_quantize_for_grouped_mlp(
                fc2_in,
                fc2_input_quantizer,
                num_groups,
                split_sizes,
                tensor_offsets=fc2_x_tensor_offsets,
            )

            fc2_out_buf = torch.empty(fc2_out_shape, dtype=dtype, device=device)
            if (
                num_groups == 1
                and grouped_fc2_x.columnwise_data is not None
                and grouped_fc2_x.columnwise_scale_inv is not None
            ):
                if fc2_op.single_grouped_weight:
                    fc2_w_single = grouped_fc2_weight.split_into_quantized_tensors()[0]
                else:
                    fc2_w_single = grouped_fc2_weight[0]
                fc2_x_single = _nvfp4_single_tensor_from_grouped(
                    grouped_fc2_x,
                    fc2_input_quantizer,
                    fp4_dtype=fc2_w_single._fp4_dtype,
                )
                general_gemm(
                    fc2_w_single,
                    fc2_x_single,
                    out_dtype=dtype,
                    out=fc2_out_buf,
                    layout="TN",
                    use_split_accumulator=False,
                )
                if fc2_bias_packed is not None:
                    token_bias = (
                        fc2_bias_packed.transpose(0, 1).contiguous().expand(in_shape[0], -1)
                    )
                    if fc2_scales is not None:
                        fc2_out_buf = fc2_out_buf + token_bias * fc2_scales.view(-1, 1)
                    else:
                        fc2_out_buf = fc2_out_buf + token_bias
            else:
                fc2_out_offsets = base_split_offsets * fc2_weight_shape[0]
                fc2_out_grouped = GroupedTensor(
                    shape=(in_shape[0], fc2_weight_shape[0]),
                    dtype=dtype,
                    num_tensors=num_groups,
                    quantizer=None,
                    data=fc2_out_buf.view(-1),
                    first_dims=split_sizes,
                    tensor_offsets=fc2_out_offsets,
                )
                general_grouped_gemm_for_grouped_tensor(
                    grouped_fc2_weight,
                    grouped_fc2_x,
                    fc2_out_grouped,
                    layout="TN",
                    bias=fc2_bias_for_gemm,
                    bias_scale=fc2_bias_scale,
                )
            fc2_out = fc2_out_buf
        else:
            fc2_in_row_data = fc1_kernel_out["d_tensor"]
            fc2_in_row_data = fc2_in_row_data.view(in_shape[0], fc2_weight_shape[1])
            fc2_in_row_scale = fc1_kernel_out["sfd_row_tensor"]
            fc2_in_row_scale = fc2_in_row_scale.permute(5, 2, 4, 0, 1, 3)

            fc2_in_col_data = fc1_kernel_out["d_col_tensor"]
            fc2_in_col_data = fc2_in_col_data.view(in_shape[0], fc2_weight_shape[1])
            fc2_in_col_scale = fc1_kernel_out["sfd_col_tensor"]
            fc2_in_col_scale = fc2_in_col_scale.permute(5, 2, 4, 0, 1, 3)

            grouped_fc2_x = GroupedTensor(
                shape=(in_shape[0], fc2_weight_shape[1]),
                dtype=dtype,
                num_tensors=num_groups,
                quantizer=fc2_input_quantizer,
                data=fc2_in_row_data.reshape(-1),
                columnwise_data=fc2_in_col_data.reshape(-1),
                scale_inv=fc2_in_row_scale.reshape(-1),
                columnwise_scale_inv=fc2_in_col_scale.reshape(-1),
                first_dims=split_sizes,
                tensor_offsets=fc2_x_tensor_offsets,
                with_gemm_swizzled_scales=True,
            )

            fc2_scales_tensor = (
                fc2_scales.detach().to(dtype=torch.float32).reshape(-1, 1, 1)
                if fc2_scales is not None
                else torch.ones((in_shape[0], 1, 1), dtype=torch.float32, device=device)
            )
            fc2_quant_kwargs = {
                "a_tensor": fc1_kernel_out["d_tensor"],
                "sfa_tensor": fc1_kernel_out["sfd_row_tensor"],
                "padded_offsets": split_points,
                "alpha_tensor": alpha_tensor,
                "bias_tensor": fc2_bias_packed,
                "norm_const_tensor": None,
                "prob_tensor": fc2_scales_tensor,
                "acc_dtype": torch.float32,
                "d_dtype": dtype,
                "cd_major": "n",
                "sf_vec_size": MXFP8_BLOCK_SCALING_SIZE,
                "current_stream": current_stream,
                "use_dynamic_sched": True,
            }

            if fc2_op.single_grouped_weight:
                # Clone and swizzle scales for GEMM (original stays unmodified for save_for_backward)
                fc2_weight_for_gemm = grouped_fc2_weight.copy()
                tex.grouped_swizzle_for_gemm(fc2_weight_for_gemm, rowwise=True, columnwise=False)

                fc2_w_data = fc2_weight_for_gemm.rowwise_data
                fc2_w_data = fc2_w_data.view(dtype=torch.float8_e4m3fn)
                fc2_w_data = fc2_w_data.view(num_groups, fc2_weight_shape[0], fc2_weight_shape[1])
                fc2_w_data = fc2_w_data.permute(1, 2, 0)

                fc2_w_scales = fc2_weight_for_gemm.scale_inv.view(dtype=torch.float8_e8m0fnu)
                fc2_w_scales = fc2_w_scales.view(
                    num_groups,
                    ceil_div(fc2_weight_shape[0], 128),
                    ceil_div(fc2_weight_shape[1], 128),
                    MXFP8_BLOCK_SCALING_SIZE,
                    4,
                    4,
                )
                fc2_w_scales = fc2_w_scales.permute(3, 4, 1, 5, 2, 0)
                fc2_quant_kwargs["b_tensor"] = fc2_w_data
                fc2_quant_kwargs["sfb_tensor"] = fc2_w_scales
            else:
                fc2_b_ptrs, fc2_sfb_ptrs, _fc2_sfb_buffer = (
                    tex.grouped_mlp_experimental.swizzle_scales_and_pack_ptrs_for_discrete_weights(
                        [w._rowwise_data for w in grouped_fc2_weight],
                        [w._rowwise_scale_inv for w in grouped_fc2_weight],
                        "nvfp4" if use_nvfp4 else "mxfp8_rowwise",
                        device,
                    )
                )
                fc2_quant_kwargs["b_ptrs"] = fc2_b_ptrs
                fc2_quant_kwargs["sfb_ptrs"] = fc2_sfb_ptrs
                fc2_quant_kwargs["n"] = fc2_weight_shape[0]
                fc2_quant_kwargs["b_dtype"] = torch.float8_e4m3fn
                fc2_quant_kwargs["b_major"] = "k"

            fc2_kernel_out = self.grouped_gemm_quant_kernel()(**fc2_quant_kwargs)
            fc2_out = fc2_kernel_out["d_tensor"].permute(2, 0, 1).view(fc2_out_shape).contiguous()

        # Save state for backward pass
        if requires_grad:
            mark_grouped_tensor(grouped_fc1_x, activation_in, scales, grouped_fc2_x)
            activation_op = self.basic_ops[1]
            activation_is_srelu = isinstance(activation_op, ScaledSReLU)
            activation_recompute_in_mlp = bool(
                getattr(activation_op, "activation_recompute_in_mlp", False)
            )
            recompute_srelu_fc2_x = (
                activation_is_srelu
                and activation_recompute_in_mlp
                and weight_requires_grad
                and _grouped_gemm_dsrelu_backward_supported()
                and _nvidia_cudnn_frontend_supports_wgrad()
            )
            saved_grouped_fc2_x = None if recompute_srelu_fc2_x else grouped_fc2_x

            # MXFP8 wgrad only needs columnwise tiles. NVFP4 generic GEMM fallbacks
            # need the full grouped tensor state, including rowwise data and amax.
            if not use_nvfp4:
                for grouped_fc_x in (grouped_fc1_x, saved_grouped_fc2_x):
                    if grouped_fc_x is not None:
                        grouped_fc_x.rowwise_data = None
                        grouped_fc_x.scale_inv = None

            # FC1 saved-tensor layout.
            #   [split_sizes, base_split_offsets, split_points,
            #    grouped_fc1_x, *fc1_weight_tensors]
            fc1_weight_tensors = (
                [grouped_fc1_weight] if fc1_op.single_grouped_weight else grouped_fc1_weight
            )
            fc1_ctx.save_for_backward(
                split_sizes,
                base_split_offsets,
                split_points,
                grouped_fc1_x,
                *fc1_weight_tensors,
            )
            fc1_ctx.use_grouped_tensor_path = True
            fc1_ctx.with_quantized_compute = True
            fc1_ctx.input_quantizers = [fc1_input_quantizer]
            fc1_ctx.weight_quantizers = [fc1_weight_quantizer]
            fc1_ctx.grad_output_quantizers = [fc1_grad_output_quantizer]
            fc1_ctx.grad_input_quantizers = None
            fc1_ctx.dtype = dtype
            fc1_ctx.input_requires_grad = input_requires_grad
            fc1_ctx.weight_requires_grad = weight_requires_grad

            # Activation
            activation_ctx.save_for_backward(activation_in, scales)
            activation_ctx.extra_input_requires_grad = True
            activation_ctx.input_requires_grad = True
            activation_ctx.dtype = dtype

            # FC2 saved-tensor layout. Matches the unfused
            # ``GroupedLinear._fuser_forward_grouped_tensor`` layout so the
            # unfused backward (basic/grouped_linear.py) can consume the same
            # ctx when the fused backward is unavailable.
            #   [split_sizes, base_split_offsets, split_points,
            #    (fc2_scales if _scale_bias),
            #    grouped_fc2_x, *fc2_weight_tensors]
            fc2_weight_tensors = (
                [grouped_fc2_weight] if fc2_op.single_grouped_weight else grouped_fc2_weight
            )
            fc2_saved: list[Optional[torch.Tensor]] = [
                split_sizes,
                base_split_offsets,
                split_points,
            ]
            if fc2_op._scale_bias:
                fc2_saved.append(fc2_scales)
            fc2_saved.append(saved_grouped_fc2_x)
            fc2_saved.extend(fc2_weight_tensors)
            fc2_ctx.save_for_backward(*fc2_saved)
            fc2_ctx.use_grouped_tensor_path = True
            fc2_ctx.with_quantized_compute = True
            fc2_ctx.input_quantizers = [fc2_input_quantizer]
            fc2_ctx.weight_quantizers = [fc2_weight_quantizer]
            fc2_ctx.grad_output_quantizers = [fc2_grad_output_quantizer]
            fc2_ctx.grad_input_quantizers = None
            fc2_ctx.dtype = dtype
            fc2_ctx.input_requires_grad = input_requires_grad
            fc2_ctx.weight_requires_grad = weight_requires_grad
            fc2_ctx.recompute_input_from_dsrelu = recompute_srelu_fc2_x

        return fc2_out, [(), (), ()]


class ForwardGroupedMLP_CuTeGEMMGLU(_ForwardGroupedMLP_CuTeGEMMBase):
    """Fused op for block-scaled GroupedLinear + scaled GLU + GroupedLinear."""

    @classmethod
    @functools.lru_cache(maxsize=None)
    def grouped_gemm_activation_kernel(cls) -> Callable:
        """Fused kernel for grouped GEMM, GLU activation, and post-multiplication."""
        from cudnn import grouped_gemm_glu_wrapper_sm100  # pylint: disable=no-name-in-module

        return grouped_gemm_glu_wrapper_sm100


class ForwardGroupedMLP_CuTeGEMMUnary(_ForwardGroupedMLP_CuTeGEMMBase):
    """Fused op for block-scaled GroupedLinear + scaled unary activation + GroupedLinear."""

    @classmethod
    @functools.lru_cache(maxsize=None)
    def is_supported(cls) -> bool:
        """Whether the SReLU fused operation is supported on the current system."""
        return _cudnn_frontend_supports_grouped_gemm_srelu() and super().is_supported()

    @classmethod
    @functools.lru_cache(maxsize=None)
    def grouped_gemm_activation_kernel(cls) -> Callable:
        """Fused kernel for grouped GEMM, SReLU activation, and post-multiplication."""
        from cudnn import grouped_gemm_srelu_wrapper_sm100  # pylint: disable=no-name-in-module

        return grouped_gemm_srelu_wrapper_sm100


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

    return fuse_grouped_mlp_ops(
        ops,
        recipe=recipe,
        fused_op_cls=ForwardGroupedMLP_CuTeGEMMGLU,
    )


def fuse_forward_srelu_ops(
    ops: list[FusibleOperation],
    *,
    recipe: Optional[Recipe] = None,
    **unused,  # pylint: disable=unused-argument
) -> list[FusibleOperation]:
    """Apply GroupedLinear + ScaledSReLU + GroupedLinear fusion for forward pass."""

    return fuse_grouped_mlp_ops(
        ops,
        recipe=recipe,
        fused_op_cls=ForwardGroupedMLP_CuTeGEMMUnary,
        activation_op_types=(ScaledSReLU,),
    )


# Register fusion if available
if ForwardGroupedMLP_CuTeGEMMGLU.is_supported():
    register_forward_fusion(fuse_forward_ops, prepend=True)
if ForwardGroupedMLP_CuTeGEMMUnary.is_supported():
    register_forward_fusion(fuse_forward_srelu_ops, prepend=True)
