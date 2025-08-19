# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""Activation functions for Transformer Engine in JAX.

This module provides optimized activation functions with quantization support.
"""

from typing import Sequence, Union, Callable, Optional
from functools import partial

import jax
import jax.numpy as jnp

from . import cpp_extensions as tex

from .quantize.tensor import ScaledTensor
from .quantize.quantizer import Quantizer
from .sharding import global_mesh_resource, global_shard_guard


def activation(
    x: jnp.ndarray,
    activation_type: Sequence[Union[str, Callable]],
    quantizer: Optional[Quantizer] = None,
) -> Union[jnp.ndarray, ScaledTensor]:
    """Apply activation functions to input tensor with optional quantization.

    This function applies a sequence of activation functions to the input tensor.
    It supports string-based activation types (e.g., 'relu', 'gelu', ('gelu', 'linear')).

    Args:
        x: Input tensor to apply activations to
        activation_type: Sequence of activation functions
        quantizer: Optional quantizer for quantizing the output

    Returns:
        Activated output tensor
    """
    assert x.shape[-1] % len(activation_type) == 0
    mesh_resource = global_mesh_resource()
    output = _activation(x, activation_type, quantizer, mesh_resource)
    return output


@partial(jax.custom_vjp, nondiff_argnums=(1, 3))
def _activation(x, activation_type, quantizer, mesh_resource):
    """Internal implementation of activation with custom VJP.

    This function implements the core activation logic with support for
    custom vector-Jacobian product (VJP) for automatic differentiation.

    Args:
        x: Input tensor
        activation_type: Sequence of activation functions
        quantizer: Optional quantizer
        mesh_resource: Mesh resource for sharding

    Returns:
        Activated tensor
    """
    _output, _ = _activation_fwd_rule(x, activation_type, quantizer, mesh_resource)
    return _output


def _activation_fwd_rule(x, activation_type, quantizer, mesh_resource):
    """Forward pass rule for activation function.

    Args:
        x: Input tensor
        activation_type: Sequence of activation functions
        quantizer: Optional quantizer
        mesh_resource: Mesh resource for sharding

    Returns:
        Tuple of (output, context) for backward pass
    """
    with global_shard_guard(mesh_resource):
        fwd_output = tex.act_lu(x, activation_type, quantizer)
        if isinstance(fwd_output, ScaledTensor):
            fwd_output = fwd_output.dequantize()
        return fwd_output, (x, quantizer)


def _activation_bwd_rule(activation_type, mesh_resource, ctx, g):
    """Backward pass rule for activation function.

    Args:
        activation_type: Sequence of activation functions
        ctx: Context from forward pass
        g: Gradient from upstream
        mesh_resource: Mesh resource for sharding

    Returns:
        Gradient with respect to input
    """
    with global_shard_guard(mesh_resource):
        (x, _) = ctx
        assert x.dtype == g.dtype
        dx = tex.dact_lu(g, x, activation_type)
        return (dx, None)


_activation.defvjp(_activation_fwd_rule, _activation_bwd_rule)
