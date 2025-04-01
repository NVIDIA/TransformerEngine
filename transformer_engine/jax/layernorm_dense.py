# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""Fused Layer normalization and dense layer transformation operations for Transformer Engine in JAX.

This module provides optimized implementations of layer normalization followed by
dense layer transformation (GEMM) operations, which are commonly used in transformer
architectures. It supports various normalization types, quantization, and
distributed training through sharding constraints.
"""

from functools import partial
from typing import Tuple

import jax
import jax.numpy as jnp

from . import cpp_extensions as tex

from .quantize import (
    QuantizerSet,
    noop_quantizer_set,
    with_sharding_constraint_by_logical_axes,
)


def layernorm_dense(
    x: jnp.ndarray,
    kernel: jnp.ndarray,
    gamma: jnp.ndarray,
    beta: jnp.ndarray,
    bias: jnp.ndarray = None,
    norm_type: str = "layernorm",
    zero_centered_gamma: bool = False,
    epsilon: float = 1e-6,
    # The logic axes of sharding constraint to the layernorm input.
    layernorm_input_axes: Tuple[str, ...] = None,
    # The logic axes of sharding constraint to the dot input.
    dot_input_axes: Tuple[str, ...] = None,
    quantizer_set: QuantizerSet = noop_quantizer_set,
) -> jnp.ndarray:
    """Apply layer normalization followed by dense layer transformation.

    This function implements the following sequence of operations:
        1. Layer normalization: (x - mean) / sqrt(var + epsilon) * gamma + beta
        2. Linear transformation: y = x * kernel + bias

    Args:
        x: Input tensor with shape [batch..., hidden_in]
        kernel: Weight matrix with shape [hidden_in, hidden_out]
        gamma: Scale parameter for normalization with shape [hidden_in]
        beta: Bias parameter for normalization with shape [hidden_in]
        bias: Optional bias term for dense layer transformation with shape [hidden_out]
        norm_type: Type of normalization ("layernorm" or "rmsnorm")
        zero_centered_gamma: Whether to use zero-centered gamma for normalization
        epsilon: Small constant for numerical stability in normalization
        layernorm_input_axes: Logical axes for sharding the layernorm input
        dot_input_axes: Logical axes for sharding the matrix multiplication input
        quantizer_set: Set of quantizers for different tensor types

    Returns:
        Output tensor with shape [batch..., hidden_out]

    Note:
        - For RMSNorm (norm_type="rmsnorm"), beta must be None and zero_centered_gamma
          must be False
        - The function supports automatic differentiation through JAX's custom VJP
        - Quantization is applied to both the normalized input and kernel
    """
    output = _layernorm_dense(
        x,
        kernel,
        gamma,
        beta,
        bias,
        norm_type,
        zero_centered_gamma,
        epsilon,
        layernorm_input_axes,
        dot_input_axes,
        quantizer_set,
    )
    return output


@partial(
    jax.custom_vjp,
    nondiff_argnums=(
        5,
        6,
        7,
        8,
        9,
    ),
)
def _layernorm_dense(
    x: jnp.ndarray,
    kernel: jnp.ndarray,
    gamma: jnp.ndarray,
    beta: jnp.ndarray,
    bias: jnp.ndarray,
    norm_type: str,
    zero_centered_gamma: bool,
    epsilon: float,
    layernorm_input_axes: Tuple[str, ...],
    dot_input_axes: Tuple[str, ...],
    quantizer_set,
):
    """Internal implementation of layernorm_dense with custom VJP.

    This function implements the forward pass of layernorm_dense with support for
    automatic differentiation. It handles the normalization and dense layer transformation
    operations, including quantization and sharding constraints.

    Args:
        x: Input tensor
        kernel: Weight matrix
        gamma: Scale parameter for normalization
        beta: Bias parameter for normalization
        bias: Optional bias term
        norm_type: Type of normalization
        zero_centered_gamma: Whether to use zero-centered gamma
        epsilon: Small constant for numerical stability
        layernorm_input_axes: Logical axes for layernorm sharding
        dot_input_axes: Logical axes for matrix multiplication sharding
        quantizer_set: Set of quantizers

    Returns:
        Output tensor from the combined operations
    """
    output, _ = _layernorm_dense_fwd_rule(
        x,
        kernel,
        gamma,
        beta,
        bias,
        norm_type,
        zero_centered_gamma,
        epsilon,
        layernorm_input_axes,
        dot_input_axes,
        quantizer_set,
    )
    return output


def _layernorm_dense_fwd_rule(
    x,
    kernel,
    gamma,
    beta,
    bias,
    norm_type,
    zero_centered_gamma,
    epsilon,
    layernorm_input_axes,
    dot_input_axes,
    quantizer_set,
):
    """Forward pass rule for layernorm_dense.

    Implements the forward pass computation including:
    1. Layer normalization with quantization
    2. Matrix multiplication with quantized kernel
    3. Optional bias addition
    4. Sharding constraints

    Returns:
        Tuple of (output, context) for automatic differentiation
    """
    x_contracting_dims = (len(x.shape) - 1,)
    k_contracting_dims = (0,)
    assert x.shape[-1] == kernel.shape[0]
    assert len(kernel.shape) == 2  # Otherwise need to merge dims in quantize

    x = with_sharding_constraint_by_logical_axes(x, layernorm_input_axes)

    casted_ln_out, mu, rsigma = tex.normalization_fwd(
        x,
        gamma,
        beta,
        zero_centered_gamma,
        epsilon,
        norm_type,
        quantizer_set.x,
    )

    # Kernel in (hidden_in, hidden_out...)
    casted_kernel = tex.quantize(kernel, quantizer_set.kernel)

    casted_ln_out = with_sharding_constraint_by_logical_axes(casted_ln_out, dot_input_axes)

    # NN GEMM
    # (batch..., hidden_in) x (hidden_in, hidden_out...)
    output = tex.gemm(
        casted_ln_out.get_rowwise_tensor(),
        casted_kernel.get_colwise_tensor(),
        (x_contracting_dims, k_contracting_dims),
    )

    use_bias = bias is not None
    if use_bias:
        bias_new_shape = (1,) * (output.ndim - bias.ndim) + bias.shape
        output += jnp.reshape(bias, bias_new_shape)

    ctx = (
        casted_ln_out.get_colwise_tensor() if quantizer_set.x.is_2x2x() else None,
        casted_kernel.get_rowwise_tensor() if quantizer_set.kernel.is_2x2x() else None,
        x.shape,
        kernel.shape,
        mu,
        rsigma,
        x,
        gamma,
        beta,
        x_contracting_dims,
        k_contracting_dims,
        use_bias,
        quantizer_set,
    )

    return output, ctx


def _layernorm_dense_bwd_rule(
    norm_type,
    zero_centered_gamma,
    epsilon,
    layernorm_input_axes,
    dot_input_axes,  # pylint: disable=unused-argument
    ctx,
    grad,
):
    """Backward pass rule for layernorm_dense.

    Implements the backward pass computation including:
    1. Gradient computation for matrix multiplication
    2. Gradient computation for layer normalization
    3. Gradient computation for bias terms
    4. Proper handling of quantization

    Returns:
        Tuple of gradients for all input parameters
    """
    (
        colwise_casted_ln_out,
        rowwise_casted_kernel,
        x_shape,
        kernel_shape,
        mu,
        rsigma,
        x,
        gamma,
        beta,
        x_contracting_dims_in_fwd,
        k_contracting_dims_in_fwd,
        use_bias,
        quantizer_set,
    ) = ctx

    grad = with_sharding_constraint_by_logical_axes(grad, dot_input_axes)

    casted_grad, dbias = tex.quantize_dbias(grad, is_dbias=use_bias, quantizer=quantizer_set.dgrad)

    # k_non_contracting_dims calibrated with the shape difference of grad.ndim vs kernel.ndim
    g_constracting_dim = tuple(
        range(grad.ndim - len(kernel_shape) + len(k_contracting_dims_in_fwd), grad.ndim)
    )
    # k_non_contracting_dims
    k_constracting_dim = tuple(
        dim for dim in range(len(kernel_shape)) if dim not in k_contracting_dims_in_fwd
    )

    # NT GEMM
    dgrad = tex.gemm(
        casted_grad.get_rowwise_tensor(),
        rowwise_casted_kernel,
        (g_constracting_dim, k_constracting_dim),
    )

    dgrad = with_sharding_constraint_by_logical_axes(dgrad, layernorm_input_axes)

    g_constracting_dim = x_constracting_dim = tuple(
        range(0, len(x_shape) - len(x_contracting_dims_in_fwd))
    )

    # TN GEMM
    wgrad = tex.gemm(
        colwise_casted_ln_out,
        casted_grad.get_colwise_tensor(),
        (x_constracting_dim, g_constracting_dim),
    )

    dx, dgamma, dbeta = tex.normalization_bwd(
        dgrad,
        x,
        mu,
        rsigma,
        gamma,
        beta,
        zero_centered_gamma=zero_centered_gamma,
        epsilon=epsilon,
        norm_type=norm_type,
    )

    return dx, wgrad, dgamma, dbeta, dbias, quantizer_set


_layernorm_dense.defvjp(_layernorm_dense_fwd_rule, _layernorm_dense_bwd_rule)
