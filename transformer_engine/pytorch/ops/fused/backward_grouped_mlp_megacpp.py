# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Mega C++ grouped MLP backward fuser."""

from __future__ import annotations
import functools
from typing import Optional

import torch

import transformer_engine_torch as tex
from ...quantization import Recipe
from ...utils import clear_tensor_data, get_device_compute_capability
from ...triton.grouped_dbias_dscales import compute_grouped_dbias
from ..basic import GroupedLinear
from ..fuser import register_backward_fusion
from ..op import FusedOperation, FusibleOperation, OperationContext
from .._common import (
    get_accumulate_flag_in_param,
    get_dummy_wgrads_for_params,
    get_main_grad_from_param,
    view_main_grad_as_grouped_buffer,
)
from .forward_grouped_mlp_megacpp import (
    _grouped_gemm_scratch,
    _megacpp_activation_config,
    _megacpp_enabled,
    _megacpp_supports_recipe,
    _resolve_megacpp_grouped_mlp_config,
)


def _megacpp_saved_weight_arg(
    saved_tensors: tuple[torch.Tensor, ...],
    *,
    single_weight_arg: bool,
    num_groups: int,
) -> tuple[torch.Tensor | list[torch.Tensor], tuple[torch.Tensor, ...]]:
    """Unpack saved C++ weight argument in the same shape used by forward."""
    if single_weight_arg:
        return saved_tensors[0], saved_tensors[1:]
    return list(saved_tensors[:num_groups]), saved_tensors[num_groups:]


def _delay_wgrad(fc_op: GroupedLinear, ctx: OperationContext) -> bool:
    """Whether this FC op requested unsupported delayed wgrad."""
    return bool(
        ctx.weight_requires_grad
        and fc_op.wgrad_store is not None
        and fc_op.wgrad_store.delay_wgrad_compute()
    )


def _compute_bias_grad_params(
    fc_op: GroupedLinear,
    dy_2d: torch.Tensor,
    base_offsets: torch.Tensor,
    *,
    num_groups: int,
    dtype: torch.dtype,
) -> tuple[Optional[list[torch.Tensor]], Optional[torch.Tensor]]:
    """Compute bias grads in GroupedLinear parameter layout."""
    if not fc_op.has_bias:
        return None, None
    dbias_packed = compute_grouped_dbias(dy_2d, base_offsets, num_groups).to(dtype=dtype)
    if fc_op.single_grouped_bias:
        return None, dbias_packed
    return [dbias_packed[idx] for idx in range(num_groups)], None


def _prepare_cpp_wgrad_output(
    fc_op: GroupedLinear,
    ctx: OperationContext,
    *,
    num_groups: int,
    weight_shape: tuple[int, int],
    label: str,
) -> tuple[Optional[torch.Tensor | list[torch.Tensor]], bool, bool, list[Optional[torch.Tensor]]]:
    """Return an optional externally-owned wgrad buffer for C++.

    If Megatron has already installed ``main_grad`` buffers, C++ writes into
    them. Otherwise this returns ``None`` and C++ allocates/returns a packed
    ``[num_groups, out_features, in_features]`` wgrad tensor.
    """
    weights = fc_op._get_weight_tensors()
    weight_grads: list[Optional[torch.Tensor]] = (
        [None] if fc_op.single_grouped_weight else [None] * num_groups
    )
    if _delay_wgrad(fc_op, ctx):
        raise ValueError("megacpp grouped MLP does not support delay_wgrad_compute=True.")
    if not ctx.weight_requires_grad:
        return None, False, False, weight_grads

    accumulate_into_main_grad = False
    if fc_op.single_grouped_weight:
        if fc_op._accumulate_into_main_grad:
            main_grad = get_main_grad_from_param(weights[0], op_label=label)
            wgrad_output = view_main_grad_as_grouped_buffer(
                main_grad,
                num_groups,
                weight_shape,
                label=f"{label} weight",
            )
            accumulate_into_main_grad = get_accumulate_flag_in_param(weights[0])
            weight_grads = get_dummy_wgrads_for_params(weights)
        else:
            wgrad_output = None
    else:
        if fc_op._accumulate_into_main_grad:
            wgrad_output = [get_main_grad_from_param(w, op_label=label) for w in weights]
            accumulate_into_main_grad = get_accumulate_flag_in_param(weights[0])
            weight_grads = get_dummy_wgrads_for_params(weights)
        else:
            wgrad_output = None

    return wgrad_output, True, accumulate_into_main_grad, weight_grads


def _assemble_grad_params(
    fc_op: GroupedLinear,
    weight_grads: list[Optional[torch.Tensor]],
    bias_grads: Optional[list[torch.Tensor]],
    bias_grad_packed: Optional[torch.Tensor],
    *,
    num_groups: int,
) -> list[Optional[torch.Tensor]]:
    """Assemble parameter grads in GroupedLinear registration order."""
    if not fc_op.has_bias:
        return weight_grads
    if fc_op.single_grouped_bias:
        return weight_grads + [bias_grad_packed]
    bias_list = bias_grads if bias_grads is not None else [None] * num_groups
    if fc_op.single_grouped_weight:
        return bias_list + weight_grads
    return weight_grads + bias_list


class BackwardGroupedMLP_MegaCpp(FusedOperation):
    """Experimental C++ grouped MLP backward for BF16/FP16.

    Weight gradients are computed in C++. Delayed wgrad is intentionally not
    supported in this first implementation to keep ownership and lifetime rules
    simple.
    """

    @classmethod
    @functools.lru_cache(maxsize=None)
    def is_supported(cls) -> bool:
        if not torch.cuda.is_available():
            return False
        if get_device_compute_capability()[0] < 10:
            return False
        return hasattr(tex, "megacpp_grouped_mlp_backward")

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
        fc1_op, activation_op, fc2_op = self.basic_ops
        fc1_ctx, activation_ctx, fc2_ctx = basic_op_ctxs
        num_groups = fc1_op.num_groups
        dtype = fc1_ctx.dtype

        fc1_saved = fc1_ctx.saved_tensors
        split_sizes, base_offsets, x_offsets, fc1_offsets = fc1_saved[:4]
        x, fc1_activation_input = fc1_saved[4:6]
        fc1_weight_arg, _ = _megacpp_saved_weight_arg(
            fc1_saved[6:],
            single_weight_arg=bool(getattr(fc1_ctx, "single_weight_arg", False)),
            num_groups=num_groups,
        )

        activation_config = _megacpp_activation_config(activation_op)
        _, act_scales = activation_ctx.saved_tensors

        fc2_saved = fc2_ctx.saved_tensors
        fc2_offsets = fc2_saved[2]
        fc2_dy_offsets = fc2_saved[3]
        fc2_x = fc2_saved[4]
        fc2_weight_arg, _ = _megacpp_saved_weight_arg(
            fc2_saved[5:],
            single_weight_arg=bool(getattr(fc2_ctx, "single_weight_arg", False)),
            num_groups=num_groups,
        )

        (
            fc1_wgrad_output,
            fc1_compute_wgrad,
            fc1_accumulate_wgrad,
            fc1_weight_grads,
        ) = _prepare_cpp_wgrad_output(
            fc1_op,
            fc1_ctx,
            num_groups=num_groups,
            weight_shape=(fc1_op.out_features, fc1_op.in_features),
            label="Grouped MLP megacpp backward (FC1)",
        )
        (
            fc2_wgrad_output,
            fc2_compute_wgrad,
            fc2_accumulate_wgrad,
            fc2_weight_grads,
        ) = _prepare_cpp_wgrad_output(
            fc2_op,
            fc2_ctx,
            num_groups=num_groups,
            weight_shape=(fc2_op.out_features, fc2_op.in_features),
            label="Grouped MLP megacpp backward (FC2)",
        )
        (
            grad_input,
            fc1_dy,
            grad_act_scales,
            fc1_owned_weight_grads,
            fc2_owned_weight_grads,
        ) = tex.megacpp_grouped_mlp_backward(
            grad_output,
            dtype,
            split_sizes,
            x_offsets,
            fc1_offsets,
            fc2_offsets,
            fc2_dy_offsets,
            base_offsets,
            x,
            fc1_activation_input,
            fc2_x,
            act_scales,
            fc1_weight_arg,
            fc2_weight_arg,
            fc1_wgrad_output,
            fc1_compute_wgrad,
            fc1_accumulate_wgrad,
            fc2_wgrad_output,
            fc2_compute_wgrad,
            fc2_accumulate_wgrad,
            activation_config.name,
            activation_config.glu_interleave_size,
            activation_config.limit,
            activation_config.alpha,
            activation_config.glu_linear_offset,
            bool(activation_ctx.extra_input_requires_grad),
            bool(fc1_ctx.input_requires_grad),
            _grouped_gemm_scratch(num_groups, grad_output.device),
        )
        if not fc1_ctx.input_requires_grad:
            grad_input = None

        grad_output_2d = grad_output.reshape(-1, fc2_op.out_features).to(dtype=dtype)
        fc2_bias_grads, fc2_bias_grad_packed = _compute_bias_grad_params(
            fc2_op,
            grad_output_2d,
            base_offsets,
            num_groups=num_groups,
            dtype=dtype,
        )
        fc1_bias_grads, fc1_bias_grad_packed = _compute_bias_grad_params(
            fc1_op,
            fc1_dy,
            base_offsets,
            num_groups=num_groups,
            dtype=dtype,
        )

        # Wgrad ownership cases:
        # 1. No weight grad: keep [None] placeholders prepared above.
        # 2. Megatron-owned main_grad: C++ wrote into the provided buffer;
        #    keep dummy wgrads prepared above for autograd.
        # 3. C++-owned allocation: replace the placeholder list with returned
        #    wgrads. Single grouped weight returns [packed], discrete weights
        #    return one tensor per expert.
        if fc2_ctx.weight_requires_grad and not fc2_op._accumulate_into_main_grad:
            expected_wgrads = 1 if fc2_op.single_grouped_weight else num_groups
            if len(fc2_owned_weight_grads) != expected_wgrads:
                raise RuntimeError(f"FC2 expected {expected_wgrads} owned wgrad tensors.")
            fc2_weight_grads = fc2_owned_weight_grads
        fc2_grad_params = _assemble_grad_params(
            fc2_op,
            fc2_weight_grads,
            fc2_bias_grads,
            fc2_bias_grad_packed,
            num_groups=num_groups,
        )
        clear_tensor_data(fc2_x)

        # Same ownership policy as FC2. Megatron-owned main_grad keeps the
        # prepared dummy grads; C++-owned allocation uses the returned wgrads.
        if fc1_ctx.weight_requires_grad and not fc1_op._accumulate_into_main_grad:
            expected_wgrads = 1 if fc1_op.single_grouped_weight else num_groups
            if len(fc1_owned_weight_grads) != expected_wgrads:
                raise RuntimeError(f"FC1 expected {expected_wgrads} owned wgrad tensors.")
            fc1_weight_grads = fc1_owned_weight_grads
        fc1_grad_params = _assemble_grad_params(
            fc1_op,
            fc1_weight_grads,
            fc1_bias_grads,
            fc1_bias_grad_packed,
            num_groups=num_groups,
        )
        clear_tensor_data(x)

        # d(act_scales) belongs to the extra input, so match act_scales.dtype
        activation_grad_extra = (
            (grad_act_scales.to(dtype=act_scales.dtype),)
            if activation_ctx.extra_input_requires_grad
            else (None,)
        )

        return (
            grad_input,
            [fc1_grad_params, (), fc2_grad_params],
            [
                (None,),
                activation_grad_extra,
                (None,),
            ],
        )


def fuse_backward_megacpp_ops(
    ops: list[FusibleOperation],
    *,
    recipe: Optional[Recipe] = None,
    **unused,  # pylint: disable=unused-argument
) -> list[FusibleOperation]:
    """Apply opt-in C++ grouped MLP backward fusion for BF16/FP16."""
    if not _megacpp_enabled():
        return ops
    if not _megacpp_supports_recipe(recipe):
        return ops
    if not BackwardGroupedMLP_MegaCpp.is_supported():
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
                BackwardGroupedMLP_MegaCpp(
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


# Use the same opt-in and recipe gate as forward. Unsupported recipes fall
# through unchanged so the matching recipe-specific backward fuser can run.
register_backward_fusion(fuse_backward_megacpp_ops, prepend=True)
