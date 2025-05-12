# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""Dense layer transformation operations for Transformer Engine in JAX.

This module provides optimized dense layer transformation operations for transformer
architectures, including support for quantization and automatic differentiation.
It implements matrix multiplication with optional bias addition and supports
customizable contracting dimensions for flexible tensor operations.
"""

from typing import Tuple, Sequence
from functools import partial
import jax
import jax.numpy as jnp

from . import cpp_extensions as tex
from .quantize import (
    QuantizerSet,
    noop_quantizer_set,
    with_sharding_constraint_by_logical_axes,
)


def dense(
    x: jnp.ndarray,
    kernel: jnp.ndarray,
    bias: jnp.ndarray = None,
    contracting_dims: Tuple[Sequence[int], Sequence[int]] = ((1,), (0,)),
    input_axes: Tuple[str, ...] = None,
    kernel_axes: Tuple[str, ...] = None,
    quantizer_set: QuantizerSet = noop_quantizer_set,
):
    """Perform dense layer transformation with optional quantization.

    This function implements matrix multiplication with optional bias addition,
    supporting quantization and custom contracting dimensions. It's optimized
    for transformer architectures and supports automatic differentiation.

    Args:
        x: Input tensor
        kernel: Weight matrix for the dense layer transformation
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
        output = _dense(x, kernel, bias, contracting_dims, input_axes, kernel_axes, quantizer_set)
    return output


@partial(jax.custom_vjp, nondiff_argnums=(3, 4, 5))
def _dense(x, kernel, bias, contracting_dims, input_axes, kernel_axes, quantizer_set):
    """Internal implementation of dense layer transformation with custom VJP.

    This function implements the core dense layer transformation logic with support
    for custom vector-Jacobian product (VJP) for automatic differentiation.

    Args:
        x: Input tensor
        kernel: Weight matrix
        bias: Optional bias tensor
        contracting_dims: Contracting dimensions specification
        input_axes: Logical axes for sharding the activation input
        kernel_axes: Logical axes for sharding the weight matrix
        quantizer_set: QuantizerSet which contains quantizers for different tensor types

    Returns:
        Transformed output tensor
    """
    output, _ = _dense_fwd_rule(
        x, kernel, bias, contracting_dims, input_axes, kernel_axes, quantizer_set
    )
    return output


def _dense_fwd_rule(x, kernel, bias, contracting_dims, input_axes, kernel_axes, quantizer_set):
    """Forward pass rule for dense layer transformation.

    Returns:
        Tuple of (output, context) for backward pass
    """
    x_contracting_dims, k_contracting_dims = contracting_dims

    flatten_axis_x = -len(x_contracting_dims)
    flatten_axis_k = len(k_contracting_dims) - len(kernel.shape)

    casted_x = tex.quantize(x, flatten_axis=flatten_axis_x, quantizer=quantizer_set.x)
    casted_x = with_sharding_constraint_by_logical_axes(casted_x, input_axes)

    casted_kernel = tex.quantize(
        kernel, flatten_axis=flatten_axis_k, quantizer=quantizer_set.kernel
    )
    casted_kernel = with_sharding_constraint_by_logical_axes(casted_kernel, kernel_axes)

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
        flatten_axis_k,
    )
    return output, ctx


def _dense_bwd_rule(
    contracting_dims, input_axes, kernel_axes, ctx, grad
):  # pylint: disable=unused-argument
    """Backward pass rule for dense layer transformation.

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
        flatten_axis_k,
    ) = ctx

    casted_grad, dbias = tex.quantize_dbias(
        grad, is_dbias=use_bias, flatten_axis=flatten_axis_k, quantizer=quantizer_set.dgrad
    )

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
    dgrad = with_sharding_constraint_by_logical_axes(dgrad, input_axes)

    # GEMM TN
    # x_non_contracting_dims
    g_constracting_dim = x_constracting_dim = tuple(
        range(0, len(x_shape) - len(fwd_x_contracting_dims))
    )

    wgrad = tex.gemm(
        colwise_casted_x, casted_grad.get_colwise_tensor(), (x_constracting_dim, g_constracting_dim)
    )
    wgrad = with_sharding_constraint_by_logical_axes(wgrad, kernel_axes)

    return dgrad, wgrad, dbias, quantizer_set


_dense.defvjp(_dense_fwd_rule, _dense_bwd_rule)


def grouped_dense(
    x: jnp.ndarray,
    kernel: jnp.ndarray,
    group_sizes: jnp.ndarray,
    contracting_dims: Tuple[Sequence[int], Sequence[int]] = ((1,), (1,)),
    bias: jnp.ndarray = None,
    precision: jax.lax.Precision = jax.lax.Precision.DEFAULT,
    preferred_element_type: jnp.dtype = None,
    group_offset: jnp.array = None,
    quantizer_set: QuantizerSet = noop_quantizer_set,
):
    # Perform grouped_dense layer transformation with optional quantization.
    if quantizer_set == noop_quantizer_set:
        # Code duplication for now, we will unify the two when we have NoopQuantizer
        output = _grouped_dense_no_quant(
            x,
            kernel,
            group_sizes,
            contracting_dims,
            bias,
            precision,
            preferred_element_type,
            group_offset,
        )
    else:
        output = _grouped_dense(
            x,
            kernel,
            group_sizes,
            contracting_dims,
            bias,
            precision,
            preferred_element_type,
            group_offset,
            quantizer_set,
        )
    return output


@partial(jax.custom_vjp, nondiff_argnums=(3, 5, 6, 7))
def _grouped_dense(
    x,
    kernel,
    group_sizes,
    contracting_dims,
    bias,
    precision,
    preferred_element_type,
    group_offset,
    quantizer_set,
):
    output, _ = _grouped_dense_fwd_rule(
        x,
        kernel,
        group_sizes,
        contracting_dims,
        bias,
        precision,
        preferred_element_type,
        group_offset,
        quantizer_set,
    )
    return output


def _grouped_dense_fwd_rule(
    x,
    kernel,
    group_sizes,
    contracting_dims,
    bias,
    precision,
    preferred_element_type,
    group_offset,
    quantizer_set,
):
    use_bias = bias is not None
    assert not use_bias, "Fused Bias is not yet supported!"

    x_contracting_dims, k_contracting_dims = contracting_dims
    flatten_axis_x = -len(x_contracting_dims)
    flatten_axis_k = len(k_contracting_dims) - len(kernel.shape) + 1  # +1 for G axis

    casted_x = tex.grouped_quantize(x, quantizer_set.x, group_sizes, flatten_axis=flatten_axis_x)
    casted_kernel = tex.grouped_quantize(kernel, quantizer_set.kernel, flatten_axis=flatten_axis_k)
    output = tex.grouped_gemm(
        casted_x.get_rowwise_tensor(),
        casted_kernel.get_colwise_tensor(),
        group_sizes,
        contracting_dims,
        bias,
        precision,
        preferred_element_type,
        group_offset,
    )

    ctx = (
        group_sizes,
        casted_x.get_colwise_tensor() if quantizer_set.x.is_2x2x() else None,
        casted_kernel.get_rowwise_tensor() if quantizer_set.kernel.is_2x2x() else None,
        x.shape,
        kernel.shape,
        use_bias,
        quantizer_set,
        flatten_axis_k,
    )
    return output, ctx


def _grouped_dense_bwd_rule(
    contracting_dims, precision, preferred_element_type, group_offset, ctx, grad
):
    fwd_x_contracting_dims, fwd_k_contracting_dims = contracting_dims

    (
        group_sizes,
        colwise_casted_x,
        rowwise_casted_kernel,
        x_shape,
        kernel_shape,
        _,
        quantizer_set,
        flatten_axis_k,
    ) = ctx

    casted_grad = tex.grouped_quantize(
        grad, quantizer_set.dgrad, group_sizes, flatten_axis=flatten_axis_k
    )

    # GEMM NT
    # k_non_contracting_dims calibrated with the shape difference of grad.ndim vs kernel.ndim
    g_constracting_dim = tuple(
        range(grad.ndim - len(kernel_shape) + len(fwd_k_contracting_dims), grad.ndim)
    )
    # k_non_contracting_dims
    k_constracting_dim = tuple(
        dim for dim in range(len(kernel_shape)) if dim not in fwd_k_contracting_dims
    )
    dgrad = tex.grouped_gemm(
        casted_grad.get_rowwise_tensor(),
        rowwise_casted_kernel,
        group_sizes,
        (g_constracting_dim, k_constracting_dim),
        precision=precision,
        preferred_element_type=preferred_element_type,
        group_offset=group_offset,
    )

    # GEMM TN
    # x_non_contracting_dims
    g_constracting_dim = x_constracting_dim = tuple(
        range(0, len(x_shape) - len(fwd_x_contracting_dims))
    )

    wgrad = tex.grouped_gemm(
        colwise_casted_x,
        casted_grad.get_colwise_tensor(),
        group_sizes,
        (x_constracting_dim, g_constracting_dim),
        precision=precision,
        preferred_element_type=preferred_element_type,
        group_offset=group_offset,
    )
    group_sizes_grad = dbias = None

    return dgrad, wgrad, group_sizes_grad, dbias, quantizer_set


_grouped_dense.defvjp(_grouped_dense_fwd_rule, _grouped_dense_bwd_rule)


@partial(jax.custom_vjp, nondiff_argnums=(3, 5, 6, 7))
def _grouped_dense_no_quant(
    x, kernel, group_sizes, contracting_dims, bias, precision, preferred_element_type, group_offset
):
    output, _ = _grouped_dense_no_quant_fwd_rule(
        x,
        kernel,
        group_sizes,
        contracting_dims,
        bias,
        precision,
        preferred_element_type,
        group_offset,
    )
    return output


def _grouped_dense_no_quant_fwd_rule(
    x, kernel, group_sizes, contracting_dims, bias, precision, preferred_element_type, group_offset
):
    use_bias = bias is not None
    assert not use_bias, "Fused Bias is not yet supported!"

    output = tex.grouped_gemm(
        x,
        kernel,
        group_sizes,
        contracting_dims,
        bias,
        precision,
        preferred_element_type,
        group_offset,
    )

    ctx = (
        group_sizes,
        x,
        kernel,
        x.shape,
        kernel.shape,
        use_bias,
    )
    return output, ctx


def _grouped_dense_no_quant_bwd_rule(
    contracting_dims, precision, preferred_element_type, group_offset, ctx, grad
):
    fwd_x_contracting_dims, fwd_k_contracting_dims = contracting_dims

    (
        group_sizes,
        x,
        kernel,
        x_shape,
        kernel_shape,
        _,
    ) = ctx

    # GEMM NT
    # k_non_contracting_dims calibrated with the shape difference of grad.ndim vs kernel.ndim
    g_constracting_dim = tuple(
        range(grad.ndim - len(kernel_shape) + len(fwd_k_contracting_dims), grad.ndim)
    )
    # k_non_contracting_dims
    k_constracting_dim = tuple(
        dim for dim in range(len(kernel_shape)) if dim not in fwd_k_contracting_dims
    )
    dgrad = tex.grouped_gemm(
        grad,
        kernel,
        group_sizes,
        (g_constracting_dim, k_constracting_dim),
        precision=precision,
        preferred_element_type=preferred_element_type,
        group_offset=group_offset,
    )

    # GEMM TN
    # x_non_contracting_dims
    g_constracting_dim = x_constracting_dim = tuple(
        range(0, len(x_shape) - len(fwd_x_contracting_dims))
    )

    wgrad = tex.grouped_gemm(
        x,
        grad,
        group_sizes,
        (x_constracting_dim, g_constracting_dim),
        precision=precision,
        preferred_element_type=preferred_element_type,
        group_offset=group_offset,
    )
    group_sizes_grad = dbias = None

    return dgrad, wgrad, group_sizes_grad, dbias


_grouped_dense_no_quant.defvjp(_grouped_dense_no_quant_fwd_rule, _grouped_dense_no_quant_bwd_rule)
