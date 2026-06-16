# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Mega C++ grouped MLP forward fuser."""

from __future__ import annotations
from collections.abc import Iterable
import functools
import os
from typing import Any, NamedTuple, Optional

import torch

import transformer_engine_torch as tex
from ...cpp_extensions.gemm import (
    get_cublas_workspace_size_bytes,
    get_grouped_gemm_setup_workspace_size,
)
from ...quantization import Recipe
from ...tensor import Quantizer
from ...utils import get_device_compute_capability
from ..basic import GroupedLinear, ScaledSReLU, ScaledClampedQGeGLU, ScaledSwiGLU
from ..fuser import register_forward_fusion
from ..op import FusedOperation, FusibleOperation, OperationContext


def _megacpp_enabled() -> bool:
    """Whether the experimental grouped MLP C++ path is explicitly enabled."""
    return int(os.getenv("NVTE_MEGACPP_GROUPED_LINEAR", "0")) > 0


def _megacpp_supports_recipe(recipe: Optional[Recipe]) -> bool:
    """Whether megacpp is a valid candidate for the active quantization recipe.

    Today the C++ implementation is BF16/FP16-only, so only the no-recipe path
    is supported. Returning False for FP8 recipes is intentional: it leaves the
    op list unchanged so the existing MXFP8/NVFP4 CuTe DSL fusers can match.
    Future MXFP8/NVFP4 support should be enabled by changing this predicate,
    not by reordering fusion registrations.
    """
    if recipe is None:
        return True
    if recipe.mxfp8() or recipe.nvfp4():
        return False
    return False


@functools.lru_cache(maxsize=None)
def _cached_grouped_gemm_scratch(
    num_groups: int,
    device_index: int,
    _stream_handle: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Cached cuBLAS grouped GEMM scratch for one CUDA stream.

    ``_stream_handle`` is intentionally part of the cache key. The workspace is
    reused without recording extra streams, so it must not be shared by
    concurrent streams.
    """
    device = torch.device("cuda", device_index)
    with torch.cuda.device(device):
        setup_size = get_grouped_gemm_setup_workspace_size(num_groups)
        cublas_size = get_cublas_workspace_size_bytes()
    return (
        torch.ones(num_groups, dtype=torch.float32, device=device),
        torch.zeros(num_groups, dtype=torch.float32, device=device),
        torch.empty(setup_size, dtype=torch.uint8, device=device),
        torch.empty(cublas_size, dtype=torch.uint8, device=device),
    )


def _grouped_gemm_scratch(
    num_groups: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return cached GEMM resources for the current stream on ``device``."""
    device_index = torch.cuda.current_device() if device.index is None else device.index
    stream_handle = int(torch.cuda.current_stream(device_index).cuda_stream)
    return _cached_grouped_gemm_scratch(num_groups, device_index, stream_handle)


class _MegaCppActivationConfig(NamedTuple):
    """Activation semantics consumed by the C++ grouped MLP path."""

    name: str
    is_scaled: bool
    is_gated: bool
    glu_interleave_size: int
    limit: float = 0.0
    alpha: float = 0.0
    glu_linear_offset: float = 0.0


def _megacpp_activation_config(activation) -> _MegaCppActivationConfig:
    """Return activation parameters consumed by the C++ grouped MLP path."""
    glu_interleave_size = int(getattr(activation, "glu_interleave_size", None) or 0)
    if isinstance(activation, ScaledSwiGLU):
        return _MegaCppActivationConfig("swiglu", True, True, glu_interleave_size)
    if isinstance(activation, ScaledClampedQGeGLU):
        return _MegaCppActivationConfig(
            "clamped_swiglu",
            True,
            True,
            glu_interleave_size,
            float(activation._clamped.limit),
            float(activation._clamped.alpha),
            float(activation._clamped.glu_linear_offset),
        )
    if isinstance(activation, ScaledSReLU):
        return _MegaCppActivationConfig("srelu", True, False, 0)
    if getattr(activation, "num_extra_inputs", 0) == 0:
        return _MegaCppActivationConfig("plain_unsupported", False, False, 0)
    raise TypeError(
        "megacpp grouped MLP currently supports only ScaledSwiGLU, "
        "ScaledClampedQGeGLU, and ScaledSReLU."
    )


def _resolve_megacpp_grouped_mlp_config(
    fc1: GroupedLinear,
    activation,
    fc2: GroupedLinear,
) -> _MegaCppActivationConfig:
    """Resolve megacpp activation config and validate grouped MLP support."""
    config = _megacpp_activation_config(activation)
    if not config.is_scaled:
        raise RuntimeError(
            "megacpp grouped MLP keeps an optional-scale activation API, but plain "
            f"{activation.__class__.__name__} is not supported yet."
        )
    if fc1.in_features % 64 != 0 or fc1.out_features % 64 != 0:
        raise ValueError(
            f"Unsupported dims for FC1 (num_groups={fc1.num_groups}, "
            f"in_features={fc1.in_features}, out_features={fc1.out_features})."
        )
    if fc2.in_features % 64 != 0 or fc2.out_features % 64 != 0:
        raise ValueError(
            f"Unsupported dims for FC2 (num_groups={fc2.num_groups}, "
            f"in_features={fc2.in_features}, out_features={fc2.out_features})."
        )
    expected_fc1_out_features = 2 * fc2.in_features if config.is_gated else fc2.in_features
    if fc1.out_features != expected_fc1_out_features or fc1.num_groups != fc2.num_groups:
        raise ValueError(
            f"FC1 (num_groups={fc1.num_groups}, in_features={fc1.in_features}, "
            f"out_features={fc1.out_features}) "
            f"and FC2 (num_groups={fc2.num_groups}, in_features={fc2.in_features}, "
            f"out_features={fc2.out_features}) do not match."
        )
    if config.glu_interleave_size and fc1.out_features % (2 * config.glu_interleave_size) != 0:
        raise ValueError(
            "GLU interleaving requires FC1 out_features to be divisible by "
            f"2*glu_interleave_size, got out_features={fc1.out_features}, "
            f"glu_interleave_size={config.glu_interleave_size}."
        )
    return config


def _megacpp_weight_arg(
    linear_op: GroupedLinear,
    dtype: torch.dtype,
    *,
    input_requires_grad: bool,
) -> torch.Tensor | list[torch.Tensor]:
    """Return GEMM-ready high-precision weights for the current C++ path.

    Keep the layout policy in GroupedLinear. This handles quantized weights the
    same way as the Python grouped GEMM path: BF16/FP16 compute dequantizes when
    needed, while a future quantized-compute path can preserve quantized weights
    by switching ``with_quantized_compute``.
    """
    with_quantized_compute = False
    if linear_op.single_grouped_weight:
        grouped_weight = linear_op._get_grouped_weight_for_gemm(
            linear_op.weight,
            [linear_op.get_quantizer("forward", 1)],
            columnwise_usage=input_requires_grad,
            with_quantized_compute=with_quantized_compute,
            dtype=dtype,
        )
        if grouped_weight.rowwise_data is None:
            raise RuntimeError("megacpp grouped MLP expected dense grouped weight rowwise_data.")
        # Keep single grouped weight packed. The C++ path wraps this as a
        # uniform GroupedTensor and dispatches nvte_grouped_gemm instead of
        # expanding it into per-expert discrete tensors.
        return grouped_weight.rowwise_data.view(
            linear_op.num_groups,
            linear_op.out_features,
            linear_op.in_features,
        )
    return linear_op._get_discrete_weights_for_gemm(
        [getattr(linear_op, f"weight{idx}") for idx in range(linear_op.num_groups)],
        [linear_op.get_quantizer("forward", 2 * idx + 1) for idx in range(linear_op.num_groups)],
        columnwise_usage=input_requires_grad,
        with_quantized_compute=with_quantized_compute,
        dtype=dtype,
    )


def _megacpp_bias_arg(linear_op: GroupedLinear, dtype: torch.dtype) -> Optional[torch.Tensor]:
    """Return a packed [G, N] high-precision bias tensor or None."""
    grouped_bias = linear_op._get_grouped_bias_for_gemm(dtype)
    if grouped_bias is None:
        return None
    return grouped_bias.rowwise_data.view(linear_op.num_groups, linear_op.out_features)


class ForwardGroupedMLP_MegaCpp(FusedOperation):
    """Experimental BF16/FP16 grouped MLP forward implemented in C++.

    The C++ function returns plain tensors only. Python still owns autograd
    context layout; delayed wgrad is rejected by the matching backward op.
    """

    @classmethod
    @functools.lru_cache(maxsize=None)
    def is_supported(cls) -> bool:
        """Whether the C++ grouped MLP path can be dispatched."""
        if not torch.cuda.is_available():
            return False
        if get_device_compute_capability()[0] < 10:
            return False
        return hasattr(tex, "megacpp_grouped_mlp_forward")

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
        _resolve_megacpp_grouped_mlp_config(fc1, activation, fc2)
        if fc1._scale_bias or fc2._scale_bias:
            raise RuntimeError("megacpp grouped MLP does not support scale_bias yet.")

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
        del prev_op_grad_output_quantizer, next_op_input_quantizer, basic_op_kwargs
        fc1_op, activation_op, fc2_op = self.basic_ops
        fc1_ctx, activation_ctx, fc2_ctx = basic_op_ctxs
        num_groups = fc1_op.num_groups

        split_sizes = basic_op_extra_inputs[0][0]
        fc2_split_sizes = basic_op_extra_inputs[2][0]
        if (
            split_sizes.size() != fc2_split_sizes.size()
            or split_sizes.data_ptr() != fc2_split_sizes.data_ptr()
        ):
            raise RuntimeError(f"{self.__class__.__name__} got different split sizes for FC1/FC2.")
        if int(split_sizes.numel()) != num_groups:
            raise ValueError(f"Expected {num_groups} splits, got {int(split_sizes.numel())}.")

        activation_config = _megacpp_activation_config(activation_op)
        act_scales = basic_op_extra_inputs[1][0]
        fc1_weight_param = fc1_op.weight if fc1_op.single_grouped_weight else fc1_op.weight0
        fc2_weight_param = fc2_op.weight if fc2_op.single_grouped_weight else fc2_op.weight0
        dtype = (
            torch.get_autocast_dtype("cuda")
            if torch.is_autocast_enabled()
            else fc1_weight_param.dtype
        )
        if dtype not in (torch.bfloat16, torch.float16):
            raise RuntimeError(f"megacpp grouped MLP supports BF16/FP16 only, got {dtype}.")

        requires_grad = any(ctx.requires_grad for ctx in basic_op_ctxs)
        input_requires_grad = requires_grad
        fc1_weight_requires_grad = requires_grad and fc1_weight_param.requires_grad
        fc2_weight_requires_grad = requires_grad and fc2_weight_param.requires_grad

        fc1_weights = _megacpp_weight_arg(
            fc1_op,
            dtype,
            input_requires_grad=input_requires_grad,
        )
        fc2_weights = _megacpp_weight_arg(
            fc2_op,
            dtype,
            input_requires_grad=input_requires_grad,
        )
        gemm_scratch = _grouped_gemm_scratch(num_groups, input_.device)
        (
            fc2_out,
            x,
            split_sizes_i64,
            base_split_offsets,
            x_offsets,
            fc1_offsets,
            fc2_offsets,
            fc2_dy_offsets,
            fc1_activation_input,
            fc2_x,
        ) = tex.megacpp_grouped_mlp_forward(
            input_,
            dtype,
            split_sizes,
            fc1_weights,
            _megacpp_bias_arg(fc1_op, dtype),
            fc2_weights,
            _megacpp_bias_arg(fc2_op, dtype),
            act_scales,
            activation_config.name,
            activation_config.glu_interleave_size,
            activation_config.limit,
            activation_config.alpha,
            activation_config.glu_linear_offset,
            gemm_scratch,
        )

        if x.data_ptr() == input_.data_ptr():
            x._do_not_clear = True

        if requires_grad:
            fc1_saved_weights = (
                [fc1_weights] if isinstance(fc1_weights, torch.Tensor) else fc1_weights
            )
            fc2_saved_weights = (
                [fc2_weights] if isinstance(fc2_weights, torch.Tensor) else fc2_weights
            )

            fc1_ctx.save_for_backward(
                split_sizes_i64,
                base_split_offsets,
                x_offsets,
                fc1_offsets,
                x,
                fc1_activation_input,
                *fc1_saved_weights,
            )
            fc1_ctx.use_megacpp_grouped_mlp = True
            fc1_ctx.dtype = dtype
            fc1_ctx.input_requires_grad = input_requires_grad
            fc1_ctx.weight_requires_grad = fc1_weight_requires_grad
            fc1_ctx.single_weight_arg = isinstance(fc1_weights, torch.Tensor)

            activation_ctx.save_for_backward(fc1_activation_input, act_scales)
            activation_ctx.extra_input_requires_grad = act_scales.requires_grad
            activation_ctx.input_requires_grad = True
            activation_ctx.dtype = dtype

            fc2_ctx.save_for_backward(
                split_sizes_i64,
                base_split_offsets,
                fc2_offsets,
                fc2_dy_offsets,
                fc2_x,
                *fc2_saved_weights,
            )
            fc2_ctx.use_megacpp_grouped_mlp = True
            fc2_ctx.dtype = dtype
            fc2_ctx.input_requires_grad = input_requires_grad
            fc2_ctx.weight_requires_grad = fc2_weight_requires_grad
            fc2_ctx.single_weight_arg = isinstance(fc2_weights, torch.Tensor)

        return fc2_out, [(), (), ()]


def fuse_forward_megacpp_ops(
    ops: list[FusibleOperation],
    *,
    recipe: Optional[Recipe] = None,
    **unused,  # pylint: disable=unused-argument
) -> list[FusibleOperation]:
    """Apply opt-in C++ grouped MLP fusion for BF16/FP16."""
    if not _megacpp_enabled():
        return ops
    if not _megacpp_supports_recipe(recipe):
        return ops
    if not ForwardGroupedMLP_MegaCpp.is_supported():
        return ops

    out = []
    window, ops = ops[:3], ops[3:]
    while len(window) == 3:
        matches_pattern = True
        if not (isinstance(window[0], GroupedLinear) and isinstance(window[2], GroupedLinear)):
            matches_pattern = False
        elif window[0]._scale_bias or window[2]._scale_bias:
            matches_pattern = False
        else:
            try:
                _resolve_megacpp_grouped_mlp_config(window[0], window[1], window[2])
            except (TypeError, ValueError, RuntimeError):
                matches_pattern = False

        if matches_pattern:
            window = [
                ForwardGroupedMLP_MegaCpp(
                    fc1=window[0],
                    activation=window[1],
                    fc2=window[2],
                )
            ]
        else:
            out.extend(window[:-2])
            window = window[-2:]

        out.extend(window[:-3])
        window = window[-3:]
        while ops and len(window) < 3:
            window.append(ops[0])
            ops = ops[1:]

    out.extend(window)
    return out


# Explicit env opt-in gives megacpp first chance. Unsupported recipes intentionally
# return the ops unchanged so lower-priority recipe-specific fusers remain the
# fallback path.
register_forward_fusion(fuse_forward_megacpp_ops, prepend=True)
