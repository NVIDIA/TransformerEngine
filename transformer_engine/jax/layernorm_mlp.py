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
from .quantize import (
    with_sharding_constraint_by_logical_axes,
    QuantizerSet,
    noop_quantizer_set
)
from .sharding import get_non_contracting_logical_axes


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
    kernel_1_axes: Tuple[str, ...] = None,
    kernel_2_axes: Tuple[str, ...] = None,
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
        kernel_1_axes: Logical axes for sharding the first weight matrix
        kernel_2_axes: Logical axes for sharding the second weight matrix
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
        kernel_1_axes,
        kernel_2_axes,
        ffn1_ckpt_name,
        ffn2_ckpt_name,
        activation_type,
        quantizer_sets,
    )
    return output


@partial(jax.custom_vjp, nondiff_argnums=(7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17))
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
    kernel_1_axes: Tuple[str, ...],
    kernel_2_axes: Tuple[str, ...],
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
        kernel_1_axes,
        kernel_2_axes,
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
    kernel_1_axes,
    kernel_2_axes,
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
    del kernel_2_axes

    ffn1_quantizer_set, ffn2_quantizer_set = quantizer_sets

    # x should be in shape of (batch..., hidden)
    # Kernel_1 should be in shape of (hidden_in, activation_len, intermediate)
    # Kernel_2 should be in shape of (intermediate, hidden_in)
    assert len(kernel_1.shape) == 3
    assert len(kernel_2.shape) == 2
    assert kernel_1.shape[-2] == len(activation_type)

    x_cdims = (x.ndim - 1,)
    k_cdims = (0,)

    assert x.shape[x_cdims[0]] == kernel_1.shape[k_cdims[0]]

    use_bias_1 = bias_1 is not None
    use_bias_2 = bias_1 is not None

    # Apply layernorm with quantized output if quantizer_set is given
    x = with_sharding_constraint_by_logical_axes(x, norm_input_axes)
    casted_ln_out, mu, rsigma = tex.normalization_fwd(
        x,
        gamma,
        beta,
        zero_centered_gamma,
        epsilon,
        norm_type,
        quantizer=ffn1_quantizer_set.x,
        noop_scaled_tensor=True,
    )
    casted_ln_out = with_sharding_constraint_by_logical_axes(casted_ln_out, dot_1_input_axes)

    # FC1 kernel (hidden_in, act_len, hidden_out)
    casted_kernel_1 = tex.quantize(kernel_1, flatten_axis=-2, quantizer=ffn1_quantizer_set.kernel,
                                   noop_scaled_tensor=True)

    # Prepare FC1 FPROP operands and layouts
    rowwise_ln_out = casted_ln_out.get_rowwise_tensor()
    rowwise_kernel_1 = casted_kernel_1.get_rowwise_tensor()
    colwise_ln_out = casted_ln_out.get_colwise_tensor()
    colwise_kernel_1 = casted_kernel_1.get_colwise_tensor()

    # FC1 GEMM:
    #   (batch..., hidden_in) x (hidden_in, act_len, hidden_out) = (batch..., act_len, hidden_out)
    # FC1 FP8 GEMM:
    #   (batch..., hidden_in) x (hidden_out, act_len, hidden_in)^T = (batch..., act_len, hidden_out)
    use_bias_1 = bias_1 is not None
    dot_1_output = tex.gemm(
        rowwise_ln_out,
        colwise_kernel_1,
        bias=bias_1 if not tex.gemm_uses_jax_dot() else None,
        fuse_bias=use_bias_1 if not tex.gemm_uses_jax_dot() else False,
        contracting_dims=(x_cdims, k_cdims),
        grad=False,
    )

    if dot_1_input_axes is not None and kernel_1_axes is not None:
        dot_1_output_axes = (
            *get_non_contracting_logical_axes(x.ndim, dot_1_input_axes, x_cdims),
            *get_non_contracting_logical_axes(kernel_1.ndim, kernel_1_axes, k_cdims),
        )
        dot_1_output = with_sharding_constraint_by_logical_axes(dot_1_output, dot_1_output_axes)

    if use_bias_1 and tex.gemm_uses_jax_dot():
        bias_1_shape = bias_1.shape
        bias_1_new_shape = (1,) * (dot_1_output.ndim - bias_1.ndim) + bias_1_shape
        dot_1_output += jnp.reshape(bias_1, bias_1_new_shape)

    dot_1_output = checkpoint_name(dot_1_output, ffn1_ckpt_name)

    # Activation (batch..., act_len, hidden_out) -> (batch..., hidden_out)
    casted_act_out = tex.act_lu(dot_1_output, activation_type, quantizer=ffn2_quantizer_set.x,
                                noop_scaled_tensor=True)
    casted_act_out = with_sharding_constraint_by_logical_axes(casted_act_out, dot_2_input_axes)

    # FC2 kernel (hidden_out, hidden_in)
    casted_kernel_2 = tex.quantize(kernel_2, quantizer=ffn2_quantizer_set.kernel,
                                   noop_scaled_tensor=True)

    # Prepare FC2 FPROP operands and layouts
    rowwise_act_out = casted_act_out.get_rowwise_tensor()
    rowwise_kernel_2 = casted_kernel_2.get_rowwise_tensor()
    colwise_act_out = casted_act_out.get_colwise_tensor()
    colwise_kernel_2 = casted_kernel_2.get_colwise_tensor()

    # FC2 GEMM:
    #   (batch..., hidden_out) x (hidden_out, hidden_in) = (batch..., hidden_in)
    # FC2 FP8 GEMM:
    #   (batch..., hidden_out) x (hidden_in, hidden_out)^T = (batch..., hidden_in)
    dot_2_output = tex.gemm(
        rowwise_act_out,
        colwise_kernel_2,
        bias=bias_2 if not tex.gemm_uses_jax_dot() else None,
        fuse_bias=use_bias_2 if not tex.gemm_uses_jax_dot() else False,
        contracting_dims=(x_cdims, k_cdims),
        grad=False
    )

    if use_bias_2 and tex.gemm_uses_jax_dot():
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
        colwise_ln_out,
        rowwise_kernel_1,
        dot_1_output,
        colwise_act_out,
        rowwise_kernel_2,
        x_cdims,
        k_cdims,
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
    kernel_1_axes,
    kernel_2_axes,
    ffn1_ckpt_name,
    ffn2_ckpt_name,
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
    del ffn1_ckpt_name, ffn2_ckpt_name
    (
        x,
        mu,
        rsigma,
        gamma,
        beta,
        colwise_ln_out,
        rowwise_kernel_1,
        dot_1_output,
        colwise_act_out,
        rowwise_kernel_2,
        fwd_x_cdims,
        fwd_k_cdims,
        use_bias_1,
        use_bias_2,
        quantizer_sets,
    ) = ctx

    ffn1_quantizer_set, ffn2_quantizer_set = quantizer_sets

    # Axis boundary for the gradient is the number of non-contracting dimensions of the FWD input
    fwd_x_non_cdims = tex.get_non_contracting_dims(colwise_ln_out.ndim, fwd_x_cdims)
    flatten_axis_grad = len(fwd_x_non_cdims)
    grad = with_sharding_constraint_by_logical_axes(grad, dot_1_input_axes)
    casted_grad, dbias_2 = tex.quantize_dbias(
        grad, is_dbias=use_bias_2, flatten_axis=flatten_axis_grad,
        quantizer=ffn2_quantizer_set.dgrad, noop_scaled_tensor=True,
    )

    # Prepare FC2 DGRAD and WGRAD operands and contracting dims
    rowwise_g = casted_grad.get_rowwise_tensor()
    rowwise_g_cdims = tuple(range(flatten_axis_grad, grad.ndim))
    fwd_k2_non_cdims = tex.get_non_contracting_dims(rowwise_kernel_2.ndim, fwd_k_cdims)

    colwise_g = casted_grad.get_colwise_tensor()
    colwise_g_cdims = tex.get_non_contracting_dims(grad.ndim, rowwise_g_cdims)
    colwise_act_out_cdims = tex.get_non_contracting_dims(colwise_act_out.ndim, fwd_x_cdims)

    # FC2 DGRAD GEMM: (batch..., hidden_in) x (hidden_out, hidden_in)^T = (batch..., hidden_out)
    dgrad_2 = tex.gemm(
        rowwise_g,
        rowwise_kernel_2,
        contracting_dims=(rowwise_g_cdims, fwd_k2_non_cdims),
        grad=True
    )
    dgrad_2 = with_sharding_constraint_by_logical_axes(dgrad_2, dot_2_input_axes)

    # FC2 WGRAD GEMM:
    #   (batch..., hidden_out)^T x (batch..., hidden_in) = (hidden_out, hidden_in)
    # FC2 WGRAD FP8 GEMM:
    #   (hidden_out, batch...) x (hidden_in, batch...)^T = (hidden_out, hidden_in)
    wgrad_2 = tex.gemm(
        colwise_act_out,
        colwise_g,
        contracting_dims=(colwise_act_out_cdims, colwise_g_cdims),
        grad=True,
    )
    wgrad_2 = with_sharding_constraint_by_logical_axes(wgrad_2, kernel_2_axes)

    # Activation gradient w/ bias fusion (batch..., hidden_out) -> (batch.., act_len, hidden_out)
    casted_dact_out, dbias_1 = tex.quantize_dact_dbias(
        dgrad_2,
        dot_1_output,
        activation_type=activation_type,
        is_dbias=use_bias_1,
        quantizer=ffn1_quantizer_set.dgrad,
        noop_scaled_tensor=True,
    )

    # Prepare FC1 DGRAD and WGRAD operands and contracting dims
    rowwise_dact_out = casted_dact_out.get_rowwise_tensor()
    rowwise_dact_out_cdims = tuple(range(flatten_axis_grad, rowwise_dact_out.ndim))
    colwise_dact_out = casted_dact_out.get_colwise_tensor()
    colwise_dact_out_cdims = tex.get_non_contracting_dims(casted_dact_out.ndim, rowwise_dact_out_cdims)
    fwd_k1_non_cdims = tex.get_non_contracting_dims(rowwise_kernel_1.ndim, fwd_k_cdims)

    # FC1 DGRAD GEMM:
    #   (batch..., act_len, hidden_out) x (hidden_in, act_len, hidden_out)^T = (batch..., hidden_in)
    dgrad_1 = tex.gemm(
        rowwise_dact_out,
        rowwise_kernel_1,
        contracting_dims=(rowwise_dact_out_cdims, fwd_k1_non_cdims),
        grad=True
    )
    dgrad_1 = with_sharding_constraint_by_logical_axes(dgrad_1, dot_1_input_axes)

    # FC1 WGRAD GEMM:
    #   (batch..., hidden_in)^T x (batch..., act_len, hidden_out) = (hidden_in, act_len, hidden_out)
    # FC1 WGRAD FP8 GEMM:
    #   (hidden_in, batch...) x (hidden_out, act_len, batch...)^T = (hidden_in, act_len, hidden_out)
    wgrad_1 = tex.gemm(
        colwise_ln_out,
        colwise_dact_out,
        contracting_dims=(fwd_x_non_cdims, colwise_dact_out_cdims),
        grad=True,
    )
    wgrad_1 = with_sharding_constraint_by_logical_axes(wgrad_1, kernel_1_axes)

    # Layernorm gradient
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
    dx = with_sharding_constraint_by_logical_axes(dx, norm_input_axes)

    return (dx, dgamma, dbeta, wgrad_1, wgrad_2, dbias_1, dbias_2, quantizer_sets)


_layernorm_mlp.defvjp(_layernorm_mlp_fwd_rule, _layernorm_mlp_bwd_rule)
