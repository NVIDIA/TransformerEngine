# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""Multi-layer perceptron (MLP) operations with layer normalization for Transformer Engine in JAX.

This module provides optimized implementations of MLP blocks commonly used in transformer
architectures. Each MLP block consists of:
1. Layer normalization
2. First dense layer transformation (GEMM1) with bias and activation
3. Second dense layer transformation (GEMM2) with bias

The implementation supports various normalization types, activation functions,
quantization, and distributed training through sharding constraints.
"""

from typing import List, Tuple, Sequence, Union, Callable
from functools import partial

import jax
import jax.numpy as jnp
from jax.ad_checkpoint import checkpoint_name

from . import cpp_extensions as tex
from .layernorm import canonicalize_norm_type
from .quantize import with_sharding_constraint_by_logical_axes, QuantizerSet, noop_quantizer_set


def layernorm_mlp(
    x: jnp.ndarray,
    gamma: jnp.ndarray,
    beta: jnp.ndarray,
    kernels: List[jnp.ndarray],
    biases: List[jnp.ndarray],
    norm_type: str,
    zero_centered_gamma: bool = False,
    epsilon: float = 1e-6,
    norm_input_axes: Tuple[str, ...] = None,
    dot_1_input_axes: Tuple[str, ...] = None,
    dot_2_input_axes: Tuple[str, ...] = None,
    ffn1_ckpt_name: str = "ffn1",
    ffn2_ckpt_name: str = "ffn2",
    activation_type: Sequence[Union[str, Callable]] = ("gelu",),
    quantizer_sets: Tuple[QuantizerSet] = (noop_quantizer_set, noop_quantizer_set),
) -> jnp.ndarray:
    """Apply layer normalization followed by MLP block.

    This function implements the following sequence of operations:
        1. Layer normalization: (x - mean) / sqrt(var + epsilon) * gamma + beta
        2. First dense layer transformation: y1 = x * kernel1 + bias1
        3. Activation function: y2 = activation(y1)
        4. Second dense layer transformation: y3 = y2 * kernel2 + bias2

    Args:
        x: Input tensor with shape [batch..., hidden_in]
        gamma: Scale parameter for normalization with shape [hidden_in]
        beta: Bias parameter for normalization with shape [hidden_in]
        kernels: List of two weight matrices:
            - kernel1: [hidden_in, intermediate]
            - kernel2: [intermediate, hidden_in]
        biases: List of two bias terms:
            - bias1: [intermediate]
            - bias2: [hidden_in]
        norm_type: Type of normalization ("layernorm" or "rmsnorm")
        zero_centered_gamma: Whether to use zero-centered gamma for normalization
        epsilon: Small constant for numerical stability in normalization
        norm_input_axes: Logical axes for sharding the layernorm input
        dot_1_input_axes: Logical axes for sharding the first matrix multiplication
        dot_2_input_axes: Logical axes for sharding the second matrix multiplication
        ffn1_ckpt_name: Name for checkpointing the first feed-forward network
        ffn2_ckpt_name: Name for checkpointing the second feed-forward network
        activation_type: Activation function(s) to apply after the first dense layer transformation
        quantizer_sets: Tuple of two quantizer sets for the two dense layer transformations

    Returns:
        Output tensor with shape [batch..., hidden_in]

    Note:
        - For RMSNorm (norm_type="rmsnorm"), beta must be None and zero_centered_gamma
          must be False
        - The function supports automatic differentiation through JAX's custom VJP
        - Quantization is applied to both dense layer transformations
        - Checkpointing is applied to both feed-forward networks for memory efficiency
    """
    assert len(kernels) == 2

    kernel_1 = kernels[0]
    kernel_2 = kernels[1]
    bias_1 = biases[0]
    bias_2 = biases[1]

    norm_type = canonicalize_norm_type(norm_type)
    if norm_type == "rmsnorm":
        assert beta is None, "beta should be None if norm_type is 'rmsnorm'"
        assert (
            not zero_centered_gamma
        ), "zero_centered_gamma is not supported if norm_type is 'rmsnorm'"

    output = _layernorm_mlp(
        x,
        gamma,
        beta,
        kernel_1,
        kernel_2,
        bias_1,
        bias_2,
        norm_type,
        zero_centered_gamma,
        epsilon,
        norm_input_axes,
        dot_1_input_axes,
        dot_2_input_axes,
        ffn1_ckpt_name,
        ffn2_ckpt_name,
        activation_type,
        quantizer_sets,
    )
    return output


@partial(jax.custom_vjp, nondiff_argnums=(7, 8, 9, 10, 11, 12, 13, 14, 15))
def _layernorm_mlp(
    x: jnp.ndarray,
    gamma: jnp.ndarray,
    beta: jnp.ndarray,
    kernel_1: jnp.ndarray,
    kernel_2: jnp.ndarray,
    bias_1: jnp.ndarray,
    bias_2: jnp.ndarray,
    norm_type: str,
    zero_centered_gamma: bool,
    epsilon: float,
    norm_input_axes: Tuple[str, ...],
    dot_1_input_axes: Tuple[str, ...],
    dot_2_input_axes: Tuple[str, ...],
    ffn1_ckpt_name: str,
    ffn2_ckpt_name: str,
    activation_type: Sequence[Union[str, Callable]],
    quantizer_sets,
):
    """Internal implementation of layernorm_mlp with custom VJP.

    This function implements the forward pass of layernorm_mlp with support for
    automatic differentiation. It handles the normalization, dense layer transformations,
    activation, and quantization operations.

    Args:
        x: Input tensor
        gamma: Scale parameter for normalization
        beta: Bias parameter for normalization
        kernel_1: First weight matrix
        kernel_2: Second weight matrix
        bias_1: First bias term
        bias_2: Second bias term
        norm_type: Type of normalization
        zero_centered_gamma: Whether to use zero-centered gamma
        epsilon: Small constant for numerical stability
        norm_input_axes: Logical axes for layernorm sharding
        dot_1_input_axes: Logical axes for first matrix multiplication sharding
        dot_2_input_axes: Logical axes for second matrix multiplication sharding
        ffn1_ckpt_name: Name for first feed-forward network checkpointing
        ffn2_ckpt_name: Name for second feed-forward network checkpointing
        activation_type: Activation function(s)
        quantizer_sets: Tuple of quantizer sets

    Returns:
        Output tensor from the combined operations
    """
    output, _ = _layernorm_mlp_fwd_rule(
        x,
        gamma,
        beta,
        kernel_1,
        kernel_2,
        bias_1,
        bias_2,
        norm_type,
        zero_centered_gamma,
        epsilon,
        norm_input_axes,
        dot_1_input_axes,
        dot_2_input_axes,
        ffn1_ckpt_name,
        ffn2_ckpt_name,
        activation_type,
        quantizer_sets,
    )
    return output


def _layernorm_mlp_fwd_rule(
    x,
    gamma,
    beta,
    kernel_1,
    kernel_2,
    bias_1,
    bias_2,
    norm_type,
    zero_centered_gamma,
    epsilon,
    norm_input_axes,
    dot_1_input_axes,
    dot_2_input_axes,
    ffn1_ckpt_name,
    ffn2_ckpt_name,
    activation_type,
    quantizer_sets,
):
    """Forward pass rule for layernorm_mlp.

    Implements the forward pass computation including:
    1. Layer normalization with quantization
    2. First matrix multiplication with quantized kernel
    3. Activation function application
    4. Second matrix multiplication with quantized kernel
    5. Optional bias additions
    6. Sharding constraints
    7. Checkpointing for memory efficiency

    Returns:
        Tuple of (output, context) for automatic differentiation
    """
    ffn1_quantizer_set, ffn2_quantizer_set = quantizer_sets

    # x should be in shape of (batch..., hidden)
    # Kernel_1 should be in shape of (hidden_in, activation_len * intermediate)
    # Kernel_2 should be in shape of (intermediate, hidden_in)
    assert len(kernel_1.shape) == 2
    assert len(kernel_2.shape) == 2
    assert kernel_1.shape[1] == kernel_2.shape[0] * len(activation_type)

    x_contracting_dims = (len(x.shape) - 1,)
    k_contracting_dims = (0,)

    assert x.shape[x_contracting_dims[0]] == kernel_1.shape[k_contracting_dims[0]]
    assert kernel_1.shape[1] == len(activation_type) * kernel_2.shape[0]

    use_bias_1 = bias_1 is not None
    use_bias_2 = bias_1 is not None

    x = with_sharding_constraint_by_logical_axes(x, norm_input_axes)

    casted_ln_out, mu, rsigma = tex.normalization_fwd(
        x,
        gamma,
        beta,
        zero_centered_gamma,
        epsilon,
        norm_type,
        quantizer=ffn1_quantizer_set.x,
    )

    casted_kernel_1 = tex.quantize(kernel_1, quantizer=ffn1_quantizer_set.kernel)

    casted_ln_out = with_sharding_constraint_by_logical_axes(casted_ln_out, dot_1_input_axes)

    # NN GEMM
    # (batch..., hidden_in) x (hidden_in, hidden_out)
    dot_1_output = tex.gemm(
        casted_ln_out.get_rowwise_tensor(),
        casted_kernel_1.get_colwise_tensor(),
        (x_contracting_dims, k_contracting_dims),
    )
    if use_bias_1:
        bias_1_shape = bias_1.shape
        bias_1_new_shape = (1,) * (dot_1_output.ndim - bias_1.ndim) + bias_1_shape
        dot_1_output += jnp.reshape(bias_1, bias_1_new_shape)

    dot_1_output = checkpoint_name(dot_1_output, ffn1_ckpt_name)

    # (batch..., hidden_in) -> (batch..., hidden)
    casted_act_out = tex.act_lu(dot_1_output, activation_type, quantizer=ffn2_quantizer_set.x)

    casted_act_out = with_sharding_constraint_by_logical_axes(casted_act_out, dot_2_input_axes)

    casted_kernel_2 = tex.quantize(kernel_2, quantizer=ffn2_quantizer_set.kernel)

    # NN GEMM
    # (batch..., hidden_in) x (hidden_out, hidden_in)
    dot_2_output = tex.gemm(
        casted_act_out.get_rowwise_tensor(),
        casted_kernel_2.get_colwise_tensor(),
        (x_contracting_dims, k_contracting_dims),
    )

    if use_bias_2:
        bias_2_shape = bias_2.shape
        bias_2_new_shape = (1,) * (dot_2_output.ndim - bias_2.ndim) + bias_2_shape
        dot_2_output += jnp.reshape(bias_2, bias_2_new_shape)

    dot_2_output = checkpoint_name(dot_2_output, ffn2_ckpt_name)

    ctx = (
        x,
        mu,
        rsigma,
        gamma,
        beta,
        casted_ln_out.get_colwise_tensor(),
        casted_kernel_1.get_rowwise_tensor(),
        dot_1_output,
        casted_act_out.get_colwise_tensor(),
        casted_kernel_2.get_rowwise_tensor(),
        x_contracting_dims,
        k_contracting_dims,
        kernel_1.shape,
        kernel_2.shape,
        use_bias_1,
        use_bias_2,
        quantizer_sets,
    )

    return dot_2_output, ctx


def _layernorm_mlp_bwd_rule(
    norm_type,
    zero_centered_gamma,
    epsilon,
    norm_input_axes,
    dot_1_input_axes,
    dot_2_input_axes,
    ffn1_ckpt_name,  # pylint: disable=unused-argument
    ffn2_ckpt_name,  # pylint: disable=unused-argument
    activation_type,
    ctx,
    grad,
):
    """Backward pass rule for layernorm_mlp.

    Implements the backward pass computation including:
    1. Gradient computation for second matrix multiplication
    2. Gradient computation for activation function
    3. Gradient computation for first matrix multiplication
    4. Gradient computation for layer normalization
    5. Gradient computation for bias terms
    6. Proper handling of quantization

    Returns:
        Tuple of gradients for all input parameters
    """
    (
        x,
        mu,
        rsigma,
        gamma,
        beta,
        colwise_casted_ln_out,
        rowwise_casted_kernel_1,
        dot_1_output,
        colwise_casted_act_out,
        rowwise_casted_kernel_2,
        x_contracting_dims_in_fwd,
        k_contracting_dims_in_fwd,
        kernel_1_shape,
        kernel_2_shape,
        use_bias_1,
        use_bias_2,
        quantizer_sets,
    ) = ctx

    ffn1_quantizer_set, ffn2_quantizer_set = quantizer_sets

    # Since the sharding of outputs should be the same as dot_1's input
    grad = with_sharding_constraint_by_logical_axes(grad, dot_1_input_axes)

    casted_grad, dbias_2 = tex.quantize_dbias(
        grad, is_dbias=use_bias_2, quantizer=ffn1_quantizer_set.dgrad
    )

    # k_non_contracting_dims calibrated with the shape difference of grad.ndim vs kernel_1.ndim
    g_constracting_dim_2 = tuple(
        range(grad.ndim - len(kernel_2_shape) + len(k_contracting_dims_in_fwd), grad.ndim)
    )
    # k_non_contracting_dims
    k_constracting_dim_2 = tuple(
        dim for dim in range(len(kernel_2_shape)) if dim not in k_contracting_dims_in_fwd
    )

    # NT GEMM
    # (batch..., hidden_out) x (hidden_in, hidden_out)
    dgrad_2 = tex.gemm(
        casted_grad.get_rowwise_tensor(),
        rowwise_casted_kernel_2,
        (g_constracting_dim_2, k_constracting_dim_2),
    )

    dgrad_2 = with_sharding_constraint_by_logical_axes(dgrad_2, dot_2_input_axes)

    x_constracting_dim = g_constracting_dim = tuple(
        range(0, len(x.shape) - len(x_contracting_dims_in_fwd))
    )

    # TN GEMM
    # (hidden, batch...,) x (hidden, batch...)
    wgrad_2 = tex.gemm(
        colwise_casted_act_out,
        casted_grad.get_colwise_tensor(),
        (x_constracting_dim, g_constracting_dim),
    )

    casted_dact_out, dbias_1 = tex.quantize_dact_dbias(
        dgrad_2,
        dot_1_output,
        activation_type=activation_type,
        is_dbias=use_bias_1,
        quantizer=ffn2_quantizer_set.dgrad,
    )

    # k_non_contracting_dims calibrated with the shape difference of grad.ndim vs kernel_1.ndim
    g_constracting_dim_1 = tuple(
        range(dgrad_2.ndim - len(kernel_1_shape) + len(k_contracting_dims_in_fwd), dgrad_2.ndim)
    )
    # k_non_contracting_dims
    k_constracting_dim_1 = tuple(
        dim for dim in range(len(kernel_1_shape)) if dim not in k_contracting_dims_in_fwd
    )

    # NT GEMM
    dgrad_1 = tex.gemm(
        casted_dact_out.get_rowwise_tensor(),
        rowwise_casted_kernel_1,
        (g_constracting_dim_1, k_constracting_dim_1),
    )

    dgrad_1 = with_sharding_constraint_by_logical_axes(dgrad_1, norm_input_axes)

    # TN GEMM
    # (hidden, batch...) x (hidden, batch...)
    wgrad_1 = tex.gemm(
        colwise_casted_ln_out,
        casted_dact_out.get_colwise_tensor(),
        (x_constracting_dim, g_constracting_dim),
    )

    dx, dgamma, dbeta = tex.normalization_bwd(
        dgrad_1,
        x,
        mu,
        rsigma,
        gamma,
        beta,
        zero_centered_gamma=zero_centered_gamma,
        epsilon=epsilon,
        norm_type=norm_type,
    )

    return (dx, dgamma, dbeta, wgrad_1, wgrad_2, dbias_1, dbias_2, quantizer_sets)


_layernorm_mlp.defvjp(_layernorm_mlp_fwd_rule, _layernorm_mlp_bwd_rule)
