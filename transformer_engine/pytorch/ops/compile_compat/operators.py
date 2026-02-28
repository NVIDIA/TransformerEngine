# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Custom operators for torch.compile compatibility."""

from __future__ import annotations
from typing import Any, Optional

import torch
from torch._library.opaque_object import register_opaque_type

from transformer_engine.common.recipe import (
    Recipe,
    DelayedScaling,
    Float8CurrentScaling,
    MXFP8BlockScaling,
    Float8BlockScaling,
    NVFP4BlockScaling,
    CustomRecipe,
)

from .tensor_info import TensorInfo, TensorInfoList, PseudoForwardResult
from .opaque_kwargs import OpaqueKwargs
from .ops_container import OpsContainer


# -----------------------------------------------------------------------------
# Recipe opaque type registration
# -----------------------------------------------------------------------------

# Register Recipe and all subclasses as reference types
# (delayed scaling and other recipes may mutate internal state)
# Guard on recipe identity - recompile if recipe instance changes
_recipe_classes = [
    Recipe,
    DelayedScaling,
    Float8CurrentScaling,
    MXFP8BlockScaling,
    Float8BlockScaling,
    NVFP4BlockScaling,
    CustomRecipe,
]

for _recipe_cls in _recipe_classes:
    register_opaque_type(
        _recipe_cls,
        typ="reference",
        guard_fn=lambda r: (id(r),),
    )


class NoneRecipe(Recipe):
    """Singleton recipe representing None (FP8 disabled).

    This class is used to pass through torch.compile custom ops when recipe is None.
    Inside the operator implementation, NoneRecipe is immediately converted to None.

    PyTorch's @torch.library.custom_op does not support Optional[OpaqueType]
    in function signatures, so we use this sentinel class instead.
    """

    _instance: Optional["NoneRecipe"] = None

    def __new__(cls) -> "NoneRecipe":
        if cls._instance is None:
            cls._instance = object.__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return "NoneRecipe()"


# Register NoneRecipe - use fixed guard (singleton always same)
register_opaque_type(
    NoneRecipe,
    typ="reference",
    guard_fn=lambda r: (),  # No values to guard on - always same instance
)


# Singleton instance - created at module import time (outside torch.compile regions)
NONE_RECIPE = NoneRecipe()


def get_recipe_or_none(recipe: Recipe) -> Optional[Recipe]:
    """Convert recipe to Optional[Recipe], handling NoneRecipe sentinel.

    Args:
        recipe: Recipe instance (may be NoneRecipe singleton)

    Returns:
        None if recipe is NoneRecipe, otherwise the recipe itself.
    """
    if isinstance(recipe, NoneRecipe):
        return None
    return recipe


# -----------------------------------------------------------------------------
# te_ops::fused_forward
# -----------------------------------------------------------------------------


def _filter_non_aliased_tensors(
    tensors_to_save: list[torch.Tensor],
    tensor_sources: list[int],
) -> list[torch.Tensor]:
    """Filter tensors_to_save to only include non-aliased tensors (source == -1)."""
    return [t for t, src in zip(tensors_to_save, tensor_sources) if src == -1]


@torch.library.custom_op("te_ops::fused_forward", mutates_args=())
def fused_forward_impl(
    x: torch.Tensor,
    ops_container: OpsContainer,
    recipe: Recipe,
    kwargs_opaque: OpaqueKwargs,
    params: list[torch.Tensor],
    extra_inputs: list[torch.Tensor],
) -> list[torch.Tensor]:
    """Perform fused forward pass.

    Fusion logic happens inside run_forward (invisible to torch.compile).

    Args:
        x: Input tensor
        ops_container: Container with operations
        recipe: FP8 recipe (NoneRecipe if FP8 is disabled)
        kwargs_opaque: Per-op keyword arguments
        params: Flattened list of parameters
        extra_inputs: Flattened list of extra inputs

    Returns:
        [output, *non_aliased_tensors_to_save, *extra_outputs] as flat list
        Tensors that alias inputs (x, params, extra_inputs) are NOT included in
        tensors_to_save - they will be reconstructed in backward using tensor_sources.
    """
    # Convert NoneRecipe to None
    actual_recipe = get_recipe_or_none(recipe)

    output, tensors_to_save, extra_outputs = ops_container.run_forward(
        x, actual_recipe, kwargs_opaque, params, extra_inputs
    )

    # Get tensor_sources from pseudo_forward to know which tensors are input aliases
    input_info = TensorInfo(tuple(x.shape), x.dtype, x.requires_grad)
    extra_inputs_info = TensorInfoList([TensorInfo.from_tensor(t) for t in extra_inputs])
    pseudo_result = ops_container.run_pseudo_forward(input_info, extra_inputs_info, kwargs_opaque)
    tensor_sources = pseudo_result.tensor_sources

    # Filter out tensors that alias inputs
    non_aliased_tensors = _filter_non_aliased_tensors(tensors_to_save, tensor_sources)

    return [output] + non_aliased_tensors + extra_outputs


@fused_forward_impl.register_fake
def fused_forward_fake(
    x: torch.Tensor,
    ops_container: OpsContainer,
    recipe: Recipe,
    kwargs_opaque: OpaqueKwargs,
    params: list[torch.Tensor],
    extra_inputs: list[torch.Tensor],
) -> list[torch.Tensor]:
    """Fake kernel - uses pseudo_forward for shape inference."""
    input_info = TensorInfo(tuple(x.shape), x.dtype, x.requires_grad)
    extra_inputs_info = TensorInfoList([TensorInfo.from_tensor(t) for t in extra_inputs])
    result = ops_container.run_pseudo_forward(input_info, extra_inputs_info, kwargs_opaque)

    output = torch.empty(result.output_info.shape, device=x.device, dtype=result.output_info.dtype)

    # Only create tensors for non-aliased sources
    tensor_sources = result.tensor_sources
    non_aliased_tensors = [
        torch.empty(info.shape, device=x.device, dtype=info.dtype)
        for info, src in zip(result.tensors_to_save_info, tensor_sources)
        if src == -1
    ]

    extra_outputs = [
        torch.empty(info.shape, device=x.device, dtype=info.dtype)
        for info in result.extra_outputs_info
    ]

    return [output] + non_aliased_tensors + extra_outputs


# -----------------------------------------------------------------------------
# te_ops::fused_backward
# -----------------------------------------------------------------------------


@torch.library.custom_op("te_ops::fused_backward", mutates_args=())
def fused_backward_impl(
    grad_output: torch.Tensor,
    grad_extra_outputs: list[torch.Tensor],
    ops_container: OpsContainer,
    recipe: Recipe,
    kwargs_opaque: OpaqueKwargs,
    input_info: TensorInfo,
    extra_inputs_info: TensorInfoList,
    tensors_saved: list[torch.Tensor],
    params: list[torch.Tensor],
) -> list[torch.Tensor]:
    """Perform fused backward pass.

    Fusion logic happens inside run_backward (invisible to torch.compile).

    Args:
        grad_output: Gradient w.r.t. output
        grad_extra_outputs: Gradients w.r.t. extra outputs
        ops_container: Container with operations
        recipe: FP8 recipe (NoneRecipe if FP8 is disabled)
        kwargs_opaque: Per-op keyword arguments
        input_info: TensorInfo of original input (for ctx reconstruction)
        extra_inputs_info: TensorInfo of extra inputs (for ctx reconstruction)
        tensors_saved: Saved tensors from forward
        params: Flattened list of parameters

    Returns:
        [grad_input, *grad_params, *grad_extra_inputs] as flat list
    """
    # Convert NoneRecipe to None
    actual_recipe = get_recipe_or_none(recipe)

    # Reconstruct ctx by running pseudo_forward with saved info
    pseudo_result = ops_container.run_pseudo_forward(input_info, extra_inputs_info, kwargs_opaque)
    ctx_data = pseudo_result.ctx_data

    # Run actual backward with ctx_data + tensors_saved
    grad_input, grad_params, grad_extra_inputs = ops_container.run_backward(
        grad_output, grad_extra_outputs, ctx_data, tensors_saved, params, actual_recipe
    )

    # Ensure grad_input doesn't alias grad_output (custom ops don't allow output-input aliasing)
    # This can happen with ops like Bias where grad_input = grad_output directly
    if grad_input is not None and grad_input.data_ptr() == grad_output.data_ptr():
        grad_input = grad_input.clone()

    # Return as flat list
    result = [grad_input] if grad_input is not None else [torch.zeros_like(grad_output)]
    result.extend(grad_params)
    result.extend(grad_extra_inputs)
    return result


@fused_backward_impl.register_fake
def fused_backward_fake(
    grad_output: torch.Tensor,
    grad_extra_outputs: list[torch.Tensor],
    ops_container: OpsContainer,
    recipe: Recipe,
    kwargs_opaque: OpaqueKwargs,
    input_info: TensorInfo,
    extra_inputs_info: TensorInfoList,
    tensors_saved: list[torch.Tensor],
    params: list[torch.Tensor],
) -> list[torch.Tensor]:
    """Fake kernel for backward - compute output shapes."""
    # grad_input has same shape as original input
    grad_input = torch.empty(
        input_info.shape,
        device=grad_output.device,
        dtype=input_info.dtype,
    )

    # grad_params have same shapes as params
    grad_params = [torch.empty_like(p) for p in params]

    # grad_extra_inputs have same shapes as extra_inputs
    grad_extra_inputs = [
        torch.empty(info.shape, device=grad_output.device, dtype=info.dtype)
        for info in extra_inputs_info.items
    ]

    return [grad_input] + grad_params + grad_extra_inputs


# -----------------------------------------------------------------------------
# Autograd registration
# -----------------------------------------------------------------------------


def _setup_context(ctx, inputs, output):
    """Save context for backward pass."""
    x, ops_container, recipe, kwargs_opaque, params, extra_inputs = inputs

    # output is [output_tensor, *non_aliased_tensors_to_save, *extra_outputs]
    flat_output = output

    # Get tensor_sources from pseudo_forward
    input_info = TensorInfo(tuple(x.shape), x.dtype, x.requires_grad)
    extra_inputs_info = TensorInfoList([TensorInfo.from_tensor(t) for t in extra_inputs])
    pseudo_result = ops_container.run_pseudo_forward(input_info, extra_inputs_info, kwargs_opaque)
    tensor_sources = pseudo_result.tensor_sources

    # Count non-aliased tensors
    num_non_aliased = sum(1 for src in tensor_sources if src == -1)
    num_extra_outputs = ops_container.num_extra_outputs

    output_tensor = flat_output[0]
    non_aliased_tensors = flat_output[1 : 1 + num_non_aliased]
    extra_outputs = flat_output[1 + num_non_aliased :]

    # Save tensors for backward: x, params, extra_inputs, and non-aliased tensors
    ctx.save_for_backward(x, *params, *extra_inputs, *non_aliased_tensors)

    # Save non-tensor state
    ctx.ops_container = ops_container
    ctx.recipe = recipe
    ctx.kwargs_opaque = kwargs_opaque
    ctx.num_params = len(params)
    ctx.num_extra_inputs = len(extra_inputs)
    ctx.num_non_aliased = num_non_aliased
    ctx.num_extra_outputs = num_extra_outputs

    # Save TensorInfo for pseudo_forward reconstruction in backward
    ctx.input_info = input_info
    ctx.extra_inputs_info = extra_inputs_info


def _backward(ctx, grads_list):
    """Backward pass."""
    # grads_list: [grad_output, *grad_non_aliased_tensors, *grad_extra_outputs]

    grad_output = grads_list[0]
    num_non_aliased = ctx.num_non_aliased
    num_extra_outputs = ctx.num_extra_outputs

    grad_non_aliased = grads_list[1 : 1 + num_non_aliased]  # Usually None
    grad_extra_outputs = list(grads_list[1 + num_non_aliased :])

    # Retrieve saved tensors: (x, *params, *extra_inputs, *non_aliased_tensors)
    saved = ctx.saved_tensors
    x = saved[0]
    params = list(saved[1 : 1 + ctx.num_params])
    extra_inputs = list(saved[1 + ctx.num_params : 1 + ctx.num_params + ctx.num_extra_inputs])
    non_aliased_tensors = list(saved[1 + ctx.num_params + ctx.num_extra_inputs :])

    # Run pseudo_forward to get tensor_sources
    pseudo_result = ctx.ops_container.run_pseudo_forward(
        ctx.input_info, ctx.extra_inputs_info, ctx.kwargs_opaque
    )
    tensor_sources = pseudo_result.tensor_sources

    # Build list of all inputs for lookup
    # source encoding: 0=x, 1..num_params=params, num_params+1..=extra_inputs
    all_inputs = [x] + params + extra_inputs

    # Reconstruct full tensors_saved using tensor_sources
    tensors_saved = []
    non_aliased_idx = 0
    for src in tensor_sources:
        if src == -1:
            # Non-aliased tensor - take from saved non_aliased_tensors
            tensors_saved.append(non_aliased_tensors[non_aliased_idx])
            non_aliased_idx += 1
        else:
            # Aliased tensor - reconstruct from inputs
            tensors_saved.append(all_inputs[src])

    backward_result = fused_backward_impl(
        grad_output,
        grad_extra_outputs,
        ctx.ops_container,
        ctx.recipe,
        ctx.kwargs_opaque,
        ctx.input_info,
        ctx.extra_inputs_info,
        tensors_saved,
        params,
    )

    # backward_result is [grad_input, *grad_params, *grad_extra_inputs]
    grad_input = backward_result[0]
    grad_params = backward_result[1 : 1 + ctx.num_params]
    grad_extra_inputs = backward_result[1 + ctx.num_params :]

    # Return grads for: x, ops_container, recipe, kwargs_opaque, params, extra_inputs
    # ops_container, recipe, kwargs_opaque are non-differentiable
    return (grad_input, None, None, None, grad_params, grad_extra_inputs)


# Register autograd
fused_forward_impl.register_autograd(_backward, setup_context=_setup_context)
