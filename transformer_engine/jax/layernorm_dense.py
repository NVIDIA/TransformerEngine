# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""Fused Layer normalization and dense layer transformation operations for Transformer Engine in JAX.

This module provides optimized implementations of layer normalization followed by
dense layer transformation (GEMM) operations, which are commonly used in transformer
architectures. It supports various normalization types, quantization, and
distributed training through sharding constraints.
"""

import warnings
from functools import partial
from typing import Tuple

import jax
import jax.numpy as jnp

from . import cpp_extensions as tex

from .quantize import (
    QuantizerSet,
    noop_quantizer_set,
    with_sharding_constraint_by_logical_axes,
    TensorUsage,
)
from .sharding import get_sequence_parallel_dim


LAYERNORM_DENSE_BATCH_FIRST_WARNING_ISSUED = False


def _issue_batch_first_warning(msg):
    global LAYERNORM_DENSE_BATCH_FIRST_WARNING_ISSUED
    if not LAYERNORM_DENSE_BATCH_FIRST_WARNING_ISSUED:
        warnings.warn(msg, UserWarning)
        LAYERNORM_DENSE_BATCH_FIRST_WARNING_ISSUED = True


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
    batch_first: bool = True,
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
        batch_first: Assume that X is batched in the first dimension if it has more than 2 dims.
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
        batch_first,
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
        11,
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
    batch_first: bool,
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
        batch_first: Assume that X is batched in the first dimension.
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
        batch_first,
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
    batch_first,
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

    x_bdim = None
    if x.ndim > 2:
        if not batch_first:
            _issue_batch_first_warning(
                "TE/JAX `layernorm_dense()` fused-layer implementation does not officially "
                "support sequence-first inputs and may produce incorrect results when "
                "`batch_first=False` or `transpose_batch_sequence=True`. Use sequence-first "
                "inputs at your own discretion."
            )
        x_bdim = 0 if batch_first else x.ndim - 2

    x = with_sharding_constraint_by_logical_axes(x, layernorm_input_axes)

    casted_ln_out, mu, rsigma = tex.normalization_fwd(
        x,
        gamma,
        beta,
        zero_centered_gamma,
        epsilon,
        norm_type,
        quantizer=quantizer_set.x,
        noop_scaled_tensor=True,
    )
    casted_ln_out = with_sharding_constraint_by_logical_axes(casted_ln_out, dot_input_axes)

    # Kernel in (hidden_in, hidden_out...)
    flatten_axis = 1 - len(kernel.shape)
    casted_kernel = tex.quantize(
        kernel, flatten_axis=flatten_axis, quantizer=quantizer_set.kernel, noop_scaled_tensor=True
    )
    casted_kernel = with_sharding_constraint_by_logical_axes(casted_kernel, kernel_axes)

    # NN GEMM
    # (batch..., hidden_in) x (hidden_in, hidden_out...)
    use_bias = bias is not None
    output = tex.gemm(
        casted_ln_out.get_tensor(TensorUsage.LHS),
        casted_kernel.get_tensor(TensorUsage.RHS),
        contracting_dims=(x_contracting_dims, k_contracting_dims),
        batched_dims=((x_bdim,), ()),
        bias=bias if not tex.gemm_uses_jax_dot() else None,
        fuse_bias=use_bias if not tex.gemm_uses_jax_dot() else False,
    )

    if use_bias and tex.gemm_uses_jax_dot():
        bias_new_shape = (1,) * (output.ndim - bias.ndim) + bias.shape
        output += jnp.reshape(bias, bias_new_shape)

    ctx = (
        casted_ln_out.get_tensor(TensorUsage.LHS_TRANS),
        casted_kernel.get_tensor(TensorUsage.RHS_TRANS),
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
        flatten_axis,
        x_bdim,
    )

    return output, ctx


def _layernorm_dense_bwd_rule(
    norm_type,
    zero_centered_gamma,
    epsilon,
    layernorm_input_axes,
    dot_input_axes,  # pylint: disable=unused-argument
    kernel_axes,
    batch_first,  # pylint: disable=unused-argument
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
        casted_ln_out,
        casted_kernel,
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
        flatten_axis,
        x_bdim,
    ) = ctx

    casted_grad, dbias = tex.quantize_dbias(
        grad,
        is_dbias=use_bias,
        flatten_axis=flatten_axis,
        quantizer=quantizer_set.dgrad,
        noop_scaled_tensor=True,
    )

    # k_non_contracting_dims calibrated with the shape difference of grad.ndim vs kernel.ndim
    g_constracting_dim = tuple(
        range(grad.ndim - len(kernel_shape) + len(k_contracting_dims_in_fwd), grad.ndim)
    )
    # k_non_contracting_dims
    k_constracting_dim = tuple(
        dim for dim in range(len(kernel_shape)) if dim not in k_contracting_dims_in_fwd
    )

    # NT GEMM
    sequence_dim = get_sequence_parallel_dim(
        layernorm_input_axes, x_contracting_dims_in_fwd, (x_bdim,)
    )
    dgrad = tex.gemm(
        casted_grad.get_tensor(TensorUsage.LHS),
        casted_kernel,
        contracting_dims=(g_constracting_dim, k_constracting_dim),
        batched_dims=((x_bdim,), ()),
        sequence_parallel_output=sequence_dim is not None and not tex.gemm_uses_jax_dot(),
        sequence_dim=sequence_dim if not tex.gemm_uses_jax_dot() else None,
    )

    dgrad = with_sharding_constraint_by_logical_axes(dgrad, layernorm_input_axes)

    g_constracting_dim = x_constracting_dim = tuple(
        range(0, len(x_shape) - len(x_contracting_dims_in_fwd))
    )

    # TN GEMM
    wgrad = tex.gemm(
        casted_ln_out,
        casted_grad.get_tensor(TensorUsage.RHS),
        contracting_dims=(x_constracting_dim, g_constracting_dim),
        batched_dims=((x_bdim,), (x_bdim,)),
    )

    wgrad = with_sharding_constraint_by_logical_axes(wgrad, kernel_axes)

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
