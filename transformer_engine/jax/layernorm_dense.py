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
    layernorm_input_axes: Tuple[str, ...] = None,
    dot_input_axes: Tuple[str, ...] = None,
    kernel_axes: Tuple[str, ...] = None,
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
        kernel_axes: Logical axes for sharding the weight matrix
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
        kernel_axes,
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
        10,
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
    kernel_axes: Tuple[str, ...],
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
        kernel_axes,
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
    kernel_axes,
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
    x_cdims = (x.ndim - 1,)
    k_cdims = (0,)
    assert x.shape[-1] == kernel.shape[0]

    # Apply layernorm with quantized output if quantizer_set is given
    x = with_sharding_constraint_by_logical_axes(x, layernorm_input_axes)
    casted_ln_out, mu, rsigma = tex.normalization_fwd(
        x,
        gamma,
        beta,
        zero_centered_gamma,
        epsilon,
        norm_type,
        quantizer_set.x,
        noop_scaled_tensor=True,
    )
    casted_ln_out = with_sharding_constraint_by_logical_axes(casted_ln_out, dot_input_axes)

    # Layernorm output (batch..., hidden_in)
    rowwise_ln_out = casted_ln_out.get_rowwise_tensor()
    colwise_ln_out = casted_ln_out.get_colwise_tensor()

    # Kernel (hidden_in, hidden_out)
    flatten_axis = 1 - kernel.ndim
    casted_kernel = tex.quantize(kernel, flatten_axis=flatten_axis, quantizer=quantizer_set.kernel,
                                 noop_scaled_tensor=True)
    casted_kernel = with_sharding_constraint_by_logical_axes(casted_kernel, kernel_axes)

    rowwise_kernel = casted_kernel.get_rowwise_tensor()
    colwise_kernel = casted_kernel.get_colwise_tensor()

    # FPROP GEMM: (batch..., hidden_in) x (hidden_in, hidden_out) = (batch..., hidden_out)
    # FPROP FP8 GEMM: (batch..., hidden_in) x (hidden_out, hidden_in)^T = (batch..., hidden_out)
    use_bias = bias is not None
    output = tex.gemm(
        rowwise_ln_out,
        colwise_kernel,
        bias=bias if not tex.gemm_uses_jax_dot() else None,
        fuse_bias=use_bias if not tex.gemm_uses_jax_dot() else False,
        contracting_dims=(x_cdims, k_cdims),
        grad=False,
    )
    if use_bias and tex.gemm_uses_jax_dot():
        bias_new_shape = (1,) * (output.ndim - bias.ndim) + bias.shape
        output += jnp.reshape(bias, bias_new_shape)

    ctx = (
        colwise_ln_out,
        rowwise_kernel,
        mu,
        rsigma,
        x,
        gamma,
        beta,
        x_cdims,
        k_cdims,
        use_bias,
        quantizer_set,
    )

    return output, ctx


def _layernorm_dense_bwd_rule(
    norm_type,
    zero_centered_gamma,
    epsilon,
    layernorm_input_axes,
    dot_input_axes,
    kernel_axes,
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
        colwise_ln_out,
        rowwise_kernel,
        mu,
        rsigma,
        x,
        gamma,
        beta,
        fwd_x_cdims,
        fwd_k_cdims,
        use_bias,
        quantizer_set,
    ) = ctx
    # Original non-contracting dimensions in the forward pass are contracting dimensions for the
    # backward pass.
    fwd_x_non_cdims = tex.get_non_contracting_dims(colwise_ln_out.ndim, fwd_x_cdims)
    fwd_k_non_cdims = tex.get_non_contracting_dims(rowwise_kernel.ndim, fwd_k_cdims)

    # Axis boundary for the gradient is the number of non-contracting dimensions of the FWD input
    flatten_axis_grad = len(fwd_x_non_cdims)
    casted_grad, dbias = tex.quantize_dbias(
        grad, is_dbias=use_bias, flatten_axis=flatten_axis_grad, quantizer=quantizer_set.dgrad,
        noop_scaled_tensor=True,
    )

    # Prepare DGRAD and WGRAD operands and contracting dims
    rowwise_g = casted_grad.get_rowwise_tensor()
    rowwise_g_cdims = tuple(range(flatten_axis_grad, grad.ndim))
    colwise_g = casted_grad.get_colwise_tensor()
    colwise_ln_out_cdims = fwd_x_non_cdims
    colwise_g_cdims = tex.get_non_contracting_dims(grad.ndim, rowwise_g_cdims)

    # DGRAD GEMM: (batch..., hidden_out) x (hidden_in, hidden_out)^T = (batch..., hidden_in)
    dgrad = tex.gemm(
        rowwise_g,
        rowwise_kernel,
        contracting_dims=(rowwise_g_cdims, fwd_k_non_cdims),
        grad=True
    )
    dgrad = with_sharding_constraint_by_logical_axes(dgrad, dot_input_axes)

    # WGRAD GEMM: (batch..., hidden_in)^T x (batch..., hidden_out) = (hidden_in, hidden_out)
    # WGRAD FP8 GEMM: (hidden_in, batch...) x (hidden_out, batch...)^T = (hidden_in, hidden_out)
    wgrad = tex.gemm(
        colwise_ln_out,
        colwise_g,
        contracting_dims=(colwise_ln_out_cdims, colwise_g_cdims),
        grad=True,
    )
    wgrad = with_sharding_constraint_by_logical_axes(wgrad, kernel_axes)

    # Layernorm gradient
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
    dx = with_sharding_constraint_by_logical_axes(dx, layernorm_input_axes)

    return dx, wgrad, dgamma, dbeta, dbias, quantizer_set


_layernorm_dense.defvjp(_layernorm_dense_fwd_rule, _layernorm_dense_bwd_rule)
