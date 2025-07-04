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
    noop_quantizer_set,
    TensorUsage,
)
from .sharding import get_non_contracting_logical_axes


def layernorm_mlp(
    x: jnp.ndarray,
    gamma: jnp.ndarray,
    beta: jnp.ndarray,
    kernels: List[jnp.ndarray],
    biases: List[jnp.ndarray],
    norm_type: str = "layernorm",
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
    batch_first: bool = True,
    ffn1_comm_overlaps: tex.CommOverlapHelperSet = tex.CommOverlapHelperSet(),
    ffn2_comm_overlaps: tex.CommOverlapHelperSet = tex.CommOverlapHelperSet(),
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
        ffn1_comm_overlaps: A set of CommOverlapHelper objects for FFN1 FPROP, DGRAD and WGRAD.
        ffn2_comm_overlaps: A set of CommOverlapHelper objects for FFN2 FPROP, DGRAD and WGRAD.
        batch_first: Assume that X is batched in the first dimension.
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
        batch_first,
        ffn1_comm_overlaps,
        ffn2_comm_overlaps,
        quantizer_sets,
    )
    return output


@partial(jax.custom_vjp, nondiff_argnums=(7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20))
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
    batch_first: bool,
    ffn1_comm_overlaps: tex.CommOverlapHelperSet,
    ffn2_comm_overlaps: tex.CommOverlapHelperSet,
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
        batch_first: Assume that X is batched in the first dimension.
        ffn1_comm_overlaps: A set of CommOverlapHelper objects for FFN1 FPROP, DGRAD and WGRAD.
        ffn2_comm_overlaps: A set of CommOverlapHelper objects for FFN2 FPROP, DGRAD and WGRAD.
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
        batch_first,
        ffn1_comm_overlaps,
        ffn2_comm_overlaps,
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
    batch_first,
    ffn1_comm_overlaps,
    ffn2_comm_overlaps,
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
    # Kernel_1 should be in shape of (hidden_in, activation_len, intermediate)
    # Kernel_2 should be in shape of (intermediate, hidden_in)
    assert len(kernel_1.shape) == 3
    assert len(kernel_2.shape) == 2
    assert kernel_1.shape[-2] == len(activation_type)

    x_contracting_dims = (len(x.shape) - 1,)
    k_contracting_dims = (0,)

    assert x.shape[x_contracting_dims[0]] == kernel_1.shape[k_contracting_dims[0]]

    x_bdim = None
    if x.ndim > 2:
        x_bdim = 0 if batch_first else x.ndim - 2

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
        noop_scaled_tensor=True,
    )
    casted_ln_out = with_sharding_constraint_by_logical_axes(casted_ln_out, dot_1_input_axes)

    casted_kernel_1 = tex.quantize(
        kernel_1, flatten_axis=-2, quantizer=ffn1_quantizer_set.kernel, noop_scaled_tensor=True
    )
    casted_kernel_1 = with_sharding_constraint_by_logical_axes(casted_kernel_1, kernel_1_axes)

    # NN GEMM
    # (batch..., sequence, hidden_in) x (hidden_in, hidden_out)
    # NOTE: Comm+GEMM overlap can only do AG->GEMM here to all-gather a sequence-parallel layernorm
    #       output.
    dot_1_output = tex.gemm(
        casted_ln_out.get_tensor(TensorUsage.LHS),
        casted_kernel_1.get_tensor(TensorUsage.RHS),
        dimension_numbers=((x_contracting_dims, k_contracting_dims), ((x_bdim,), ())),
        bias=bias_1 if not tex.gemm_uses_jax_dot() else None,
        fuse_bias=use_bias_1 if not tex.gemm_uses_jax_dot() else False,
        comm_overlap=ffn1_comm_overlaps.fprop,
    )
    dot_1_output = with_sharding_constraint_by_logical_axes(
        dot_1_output,
        ffn1_comm_overlaps.fprop.get_logical_output_axes(
            dot_1_input_axes,
            kernel_1_axes,
            ((x_contracting_dims, k_contracting_dims), ((x_bdim,), ())),
        ),
    )

    if use_bias_1 and tex.gemm_uses_jax_dot():
        bias_1_shape = bias_1.shape
        bias_1_new_shape = (1,) * (dot_1_output.ndim - bias_1.ndim) + bias_1_shape
        dot_1_output += jnp.reshape(bias_1, bias_1_new_shape)

    dot_1_output = checkpoint_name(dot_1_output, ffn1_ckpt_name)

    # (batch..., hidden_in) -> (batch..., hidden)
    casted_act_out = tex.act_lu(
        dot_1_output, activation_type, quantizer=ffn2_quantizer_set.x, noop_scaled_tensor=True
    )

    casted_act_out = with_sharding_constraint_by_logical_axes(casted_act_out, dot_2_input_axes)

    casted_kernel_2 = tex.quantize(
        kernel_2, quantizer=ffn2_quantizer_set.kernel, noop_scaled_tensor=True
    )
    casted_kernel_2 = with_sharding_constraint_by_logical_axes(casted_kernel_2, kernel_2_axes)

    # NN GEMM
    # (batch..., hidden_in) x (hidden_out, hidden_in)
    # NOTE: Comm+GEMM overlap can only do GEMM->RS to reduce-scatter the FFN2 output. We don't need
    #       an auxiliary input/output here for this because it's already handled in the custom op
    #       and the returned array is the final reduce-scattered result.
    dot_2_output = tex.gemm(
        casted_act_out.get_tensor(TensorUsage.LHS),
        casted_kernel_2.get_tensor(TensorUsage.RHS),
        dimension_numbers=((x_contracting_dims, k_contracting_dims), ((x_bdim,), ())),
        bias=bias_2 if not tex.gemm_uses_jax_dot() else None,
        fuse_bias=use_bias_2 if not tex.gemm_uses_jax_dot() else False,
        comm_overlap=ffn2_comm_overlaps.fprop,
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
        casted_ln_out.get_tensor(TensorUsage.LHS_TRANS),
        casted_kernel_1.get_tensor(TensorUsage.RHS_TRANS),
        dot_1_output,
        casted_act_out.get_tensor(TensorUsage.LHS_TRANS),
        casted_kernel_2.get_tensor(TensorUsage.RHS_TRANS),
        x_contracting_dims,
        k_contracting_dims,
        kernel_1.shape,
        kernel_2.shape,
        use_bias_1,
        use_bias_2,
        quantizer_sets,
        x_bdim,
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
    batch_first,
    ffn1_comm_overlaps,
    ffn2_comm_overlaps,
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
    del norm_input_axes, ffn1_ckpt_name, ffn2_ckpt_name, batch_first
    (
        x,
        mu,
        rsigma,
        gamma,
        beta,
        casted_ln_out,
        casted_kernel_1,
        dot_1_output,
        casted_act_out,
        casted_kernel_2,
        x_contracting_dims_in_fwd,
        k_contracting_dims_in_fwd,
        kernel_1_shape,
        kernel_2_shape,
        use_bias_1,
        use_bias_2,
        quantizer_sets,
        x_bdim,
    ) = ctx

    ffn1_quantizer_set, ffn2_quantizer_set = quantizer_sets

    casted_grad, dbias_2 = tex.quantize_dbias(
        grad, is_dbias=use_bias_2, quantizer=ffn1_quantizer_set.dgrad, noop_scaled_tensor=True
    )
    casted_grad = with_sharding_constraint_by_logical_axes(
        casted_grad,
        ffn2_comm_overlaps.fprop.get_logical_output_axes(
            dot_2_input_axes,
            kernel_2_axes,
            ((x_contracting_dims_in_fwd, k_contracting_dims_in_fwd), ((x_bdim,), ())),
        ),
    )

    # k_non_contracting_dims calibrated with the shape difference of grad.ndim vs kernel_1.ndim
    g_contracting_dims_2 = tuple(
        range(grad.ndim - len(kernel_2_shape) + len(k_contracting_dims_in_fwd), grad.ndim)
    )
    # k_non_contracting_dims
    k_contracting_dims_2 = tuple(
        dim for dim in range(len(kernel_2_shape)) if dim not in k_contracting_dims_in_fwd
    )

    # NT GEMM
    # (batch..., hidden_out) x (hidden_in, hidden_out)
    # NOTE: The only possible comm. overlap with FFN2 DGRAD is an AG+GEMM with all-gathered
    #       gradient returned in the auxiliary output to be re-used in the FFN2 WGRAD GEMM.
    dgrad_2 = tex.gemm(
        casted_grad.get_tensor(TensorUsage.LHS),
        casted_kernel_2,
        dimension_numbers=((g_contracting_dims_2, k_contracting_dims_2), ((x_bdim,), ())),
        comm_overlap=ffn2_comm_overlaps.dgrad,
    )

    x_contracting_dims = g_contracting_dims = tuple(
        range(0, len(x.shape) - len(x_contracting_dims_in_fwd))
    )

    # TN GEMM
    # (hidden, batch...,) x (hidden, batch...)
    # NOTE: There is no possible comm. overlap with FFN2 WGRAD, but we need to re-use the
    #       all-gathered gradient returned in the auxiliary output of FFN2 DGRAD.
    casted_grad_rhs = casted_grad.get_tensor(usage=TensorUsage.RHS)
    if ffn2_comm_overlaps.dgrad.is_enabled:
        casted_grad_rhs.data = (
            dgrad_2[-1].transpose(
                *range(casted_grad_rhs.flatten_axis, casted_grad_rhs.ndim),
                *range(casted_grad_rhs.flatten_axis)
            )
            if casted_grad_rhs.data_layout == "T"
            else dgrad_2[-1]
        )
        dgrad_2 = dgrad_2[0]

    wgrad_2 = tex.gemm(
        casted_act_out,
        casted_grad.get_tensor(TensorUsage.RHS),
        dimension_numbers=((x_contracting_dims, g_contracting_dims), ((x_bdim,), (x_bdim,))),
        comm_overlap=ffn2_comm_overlaps.wgrad,
    )

    dgrad_2 = with_sharding_constraint_by_logical_axes(dgrad_2, dot_2_input_axes)
    wgrad_2 = with_sharding_constraint_by_logical_axes(wgrad_2, kernel_2_axes)

    casted_dact_out, dbias_1 = tex.quantize_dact_dbias(
        dgrad_2,
        dot_1_output,
        activation_type=activation_type,
        is_dbias=use_bias_1,
        quantizer=ffn2_quantizer_set.dgrad,
        noop_scaled_tensor=True,
    )
    casted_dact_out = with_sharding_constraint_by_logical_axes(
        casted_dact_out,
        ffn1_comm_overlaps.fprop.get_logical_output_axes(
            dot_1_input_axes,
            kernel_1_axes,
            ((x_contracting_dims_in_fwd, k_contracting_dims_in_fwd), ((x_bdim,), ())),
        ),
    )

    # k_non_contracting_dims calibrated with the shape difference of grad.ndim vs kernel_1.ndim
    dact_out_ndim = casted_dact_out.get_tensor(TensorUsage.LHS).data.ndim
    g_contracting_dims_1 = tuple(
        range(dact_out_ndim - len(kernel_1_shape) + len(k_contracting_dims_in_fwd), dact_out_ndim)
    )
    # k_non_contracting_dims
    k_contracting_dims_1 = tuple(
        dim for dim in range(len(kernel_1_shape)) if dim not in k_contracting_dims_in_fwd
    )

    # If FFN1 DGRAD is bulk all-gathering the layernorm output, but the layernorm output
    # has transposed data layout, we need to un-transpose it here before the all-gather and
    # transpose it again before using it in FFN1 WGRAD. Also make sure we do not already have the
    # the gathered layernorm output from FPROP.
    # NOTE: This transpose should not be necessary if the tensor usages work correctly!
    dgrad_1_aux_in = None
    ln_out_transposed_dims = (
        *tuple(range(casted_ln_out.flatten_axis, casted_ln_out.ndim)),
        *tuple(range(casted_ln_out.flatten_axis)),
    )
    casted_ln_out = with_sharding_constraint_by_logical_axes(casted_ln_out, dot_1_input_axes)
    if ffn1_comm_overlaps.dgrad.is_bulk() and not ffn1_comm_overlaps.fprop.output_all_gathered_lhs:
        dgrad_1_aux_in = (
            casted_ln_out.data.transpose(ln_out_transposed_dims)
            if casted_ln_out.data_layout == "T"
            else casted_ln_out.data
        )

    # NT GEMM
    dgrad_1 = tex.gemm(
        casted_dact_out.get_tensor(TensorUsage.LHS),
        casted_kernel_1,
        dimension_numbers=((g_contracting_dims_1, k_contracting_dims_1), ((x_bdim,), ())),
        comm_overlap=ffn1_comm_overlaps.dgrad,
        aux_in=dgrad_1_aux_in,
    )

    if ffn1_comm_overlaps.dgrad.is_bulk() and not ffn1_comm_overlaps.fprop.output_all_gathered_lhs:
        casted_ln_out.data = (
            dgrad_1[-1].transpose(ln_out_transposed_dims)
            if casted_ln_out.data_layout == "T"
            else dgrad_1[-1]
        )
        dgrad_1 = dgrad_1[0]

    # TN GEMM
    # (hidden, batch...) x (hidden, batch...)
    wgrad_1 = tex.gemm(
        casted_ln_out,
        casted_dact_out.get_tensor(TensorUsage.RHS),
        dimension_numbers=((x_contracting_dims, g_contracting_dims), ((x_bdim,), (x_bdim,))),
        comm_overlap=ffn1_comm_overlaps.wgrad,
        aux_in=(dgrad_1 if ffn1_comm_overlaps.wgrad.is_bulk() else None),
    )
    if ffn1_comm_overlaps.wgrad.is_bulk():
        # FFN1 DGRAD was bulk reduce-scattered during FFN2 WGRAD and returned as auxiliary output
        dgrad_1 = wgrad_1[-1]
        wgrad_1 = wgrad_1[0]

    dgrad_1 = with_sharding_constraint_by_logical_axes(dgrad_1, dot_1_input_axes)
    wgrad_1 = with_sharding_constraint_by_logical_axes(wgrad_1, kernel_1_axes)

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
