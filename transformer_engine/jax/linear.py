# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""Linear transformation operations for Transformer Engine in JAX.

This module provides optimized linear transformation operations for transformer
architectures, including support for quantization and automatic differentiation.
It implements matrix multiplication with optional bias addition and supports
customizable contracting dimensions for flexible tensor operations.
"""

from typing import Tuple, Sequence
from functools import partial
import jax
import jax.numpy as jnp

from . import cpp_extensions as tex
from .quantize import QuantizerSet, noop_quantizer_set


def linear(
    x: jnp.ndarray,
    kernel: jnp.ndarray,
    bias: jnp.ndarray = None,
    contracting_dims: Tuple[Sequence[int], Sequence[int]] = ((1,), (0,)),
    quantizer_set: QuantizerSet = noop_quantizer_set,
):
    """Perform linear transformation with optional quantization.

    This function implements matrix multiplication with optional bias addition,
    supporting quantization and custom contracting dimensions. It's optimized
    for transformer architectures and supports automatic differentiation.

    Args:
        x: Input tensor
        kernel: Weight matrix for the linear transformation
        bias: Optional bias tensor to add after the transformation
        contracting_dims: Tuple of sequences specifying which dimensions to contract
        quantizer_set: QuantizerSet which contains quantizers for different tensor types

    Returns:
        Transformed output tensor
    """
    # Remove when tex.quantize() can handle quantizer=None
    if quantizer_set == noop_quantizer_set:
        output = tex.gemm(x, kernel, contracting_dims)
        if bias is not None:
            bias_new_shape = (1,) * (output.ndim - bias.ndim) + bias.shape
            output += jnp.reshape(bias, bias_new_shape)
    else:
        output = _linear(x, kernel, bias, contracting_dims, quantizer_set)
    return output


@partial(jax.custom_vjp, nondiff_argnums=(3,))
def _linear(x, kernel, bias, contracting_dims, quantizer_set):
    """Internal implementation of linear transformation with custom VJP.

    This function implements the core linear transformation logic with support
    for custom vector-Jacobian product (VJP) for automatic differentiation.

    Args:
        x: Input tensor
        kernel: Weight matrix
        bias: Optional bias tensor
        contracting_dims: Contracting dimensions specification
        quantizer_set: QuantizerSet which contains quantizers for different tensor types

    Returns:
        Transformed output tensor
    """
    output, _ = _linear_fwd_rule(x, kernel, bias, contracting_dims, quantizer_set)
    return output


def _linear_fwd_rule(x, kernel, bias, contracting_dims, quantizer_set):
    """Forward pass rule for linear transformation.

    Args:
        x: Input tensor
        kernel: Weight matrix
        bias: Optional bias tensor
        contracting_dims: Contracting dimensions specification
        quantizer_set: QuantizerSet which contains quantizers for different tensor types

    Returns:
        Tuple of (output, context) for backward pass
    """
    x_contracting_dims, k_contracting_dims = contracting_dims

    casted_x = tex.quantize(x, quantizer_set.x)
    casted_kernel = tex.quantize(kernel, quantizer_set.kernel)

    # GEMM NN
    output = tex.gemm(
        casted_x.get_rowwise_tensor(),
        casted_kernel.get_colwise_tensor(),
        (x_contracting_dims, k_contracting_dims),
    )
    use_bias = bias is not None
    if use_bias:
        bias_new_shape = (1,) * (output.ndim - bias.ndim) + bias.shape
        output += jnp.reshape(bias, bias_new_shape)

    ctx = (
        casted_x.get_colwise_tensor() if quantizer_set.x.is_2x2x() else None,
        casted_kernel.get_rowwise_tensor() if quantizer_set.kernel.is_2x2x() else None,
        x.shape,
        kernel.shape,
        use_bias,
        quantizer_set,
    )
    return output, ctx


def _linear_bwd_rule(contracting_dims, ctx, grad):  # pylint: disable=unused-argument
    """Backward pass rule for linear transformation.

    Args:
        contracting_dims: Contracting dimensions specification
        ctx: Context from forward pass
        grad: Gradient from upstream

    Returns:
        Tuple of gradients with respect to inputs
    """
    fwd_x_contracting_dims, fwd_k_contracting_dims = contracting_dims

    (
        colwise_casted_x,
        rowwise_casted_kernel,
        x_shape,
        kernel_shape,
        use_bias,
        quantizer_set,
    ) = ctx

    casted_grad, dbias = tex.quantize_dbias(grad, is_dbias=use_bias, quantizer=quantizer_set.dgrad)

    # GEMM NT
    # k_non_contracting_dims calibrated with the shape difference of grad.ndim vs kernel.ndim
    g_constracting_dim = tuple(
        range(grad.ndim - len(kernel_shape) + len(fwd_k_contracting_dims), grad.ndim)
    )
    # k_non_contracting_dims
    k_constracting_dim = tuple(
        dim for dim in range(len(kernel_shape)) if dim not in fwd_k_contracting_dims
    )
    dgrad = tex.gemm(
        casted_grad.get_rowwise_tensor(),
        rowwise_casted_kernel,
        (g_constracting_dim, k_constracting_dim),
    )

    # GEMM TN
    # x_non_contracting_dims
    g_constracting_dim = x_constracting_dim = tuple(
        range(0, len(x_shape) - len(fwd_x_contracting_dims))
    )

    wgrad = tex.gemm(
        colwise_casted_x, casted_grad.get_colwise_tensor(), (x_constracting_dim, g_constracting_dim)
    )

    return dgrad, wgrad, dbias, quantizer_set


_linear.defvjp(_linear_fwd_rule, _linear_bwd_rule)
