# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""Dense layer transformation operations for Transformer Engine in JAX.

This module provides optimized dense layer transformation operations for transformer
architectures, including support for quantization and automatic differentiation.
It implements matrix multiplication with optional bias addition and supports
customizable contracting dimensions for flexible tensor operations.
"""
import warnings
from typing import Tuple, Sequence
from functools import partial
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

DENSE_BATCH_FIRST_WARNING_ISSUED = False


def _issue_batch_first_warning(msg):
    global DENSE_BATCH_FIRST_WARNING_ISSUED
    if not DENSE_BATCH_FIRST_WARNING_ISSUED:
        warnings.warn(msg, UserWarning)
        DENSE_BATCH_FIRST_WARNING_ISSUED = True


def dense(
    x: jnp.ndarray,
    kernel: jnp.ndarray,
    bias: jnp.ndarray = None,
    contracting_dims: Tuple[Sequence[int], Sequence[int]] = ((1,), (0,)),
    input_axes: Tuple[str, ...] = None,
    kernel_axes: Tuple[str, ...] = None,
    batch_first: bool = True,
    sequence_parallel_output: bool = False,
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
        batch_first: Assume that X is batched in the first dimension.
        sequence_parallel_output: Produce an output that sharded in the first non-batched dim. Only
                                  supported for TE custom GEMM with row-parallel kernel axes.
        quantizer_set: QuantizerSet which contains quantizers for different tensor types

    Returns:
        Transformed output tensor
    """
    # Remove when tex.quantize() can handle quantizer=None
    if quantizer_set == noop_quantizer_set and tex.gemm_uses_jax_dot():
        x = with_sharding_constraint_by_logical_axes(x, input_axes)
        output = tex.gemm(x, kernel, contracting_dims=contracting_dims)
        if bias is not None:
            bias_new_shape = (1,) * (output.ndim - bias.ndim) + bias.shape
            output += jnp.reshape(bias, bias_new_shape)
    else:
        output = _dense(
            x,
            kernel,
            bias,
            contracting_dims,
            input_axes,
            kernel_axes,
            batch_first,
            sequence_parallel_output,
            quantizer_set,
        )
    return output


@partial(jax.custom_vjp, nondiff_argnums=(3, 4, 5, 6, 7))
def _dense(
    x,
    kernel,
    bias,
    contracting_dims,
    input_axes,
    kernel_axes,
    batch_first,
    sequence_parallel_output,
    quantizer_set,
):
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
        batch_first: Assume that X is batched in the first dimension if it has more than 2 dims.
        sequence_parallel_output: Produce an output that sharded in the first non-batched dim. Only
                                  supported for TE custom GEMM with row-parallel kernel axes.
        quantizer_set: QuantizerSet which contains quantizers for different tensor types

    Returns:
        Transformed output tensor
    """
    output, _ = _dense_fwd_rule(
        x,
        kernel,
        bias,
        contracting_dims,
        input_axes,
        kernel_axes,
        batch_first,
        sequence_parallel_output,
        quantizer_set,
    )
    return output


def _dense_fwd_rule(
    x,
    kernel,
    bias,
    contracting_dims,
    input_axes,
    kernel_axes,
    batch_first,
    sequence_parallel_output,
    quantizer_set,
):
    """Forward pass rule for dense layer transformation.

    Returns:
        Tuple of (output, context) for backward pass
    """
    x_contracting_dims, k_contracting_dims = map(
        tex.sanitize_dims, (x.ndim, kernel.ndim), contracting_dims
    )

    # Check supported input layout
    x_is_transposed = x.ndim - 1 not in x_contracting_dims
    k_is_transposed = kernel.ndim - 1 in k_contracting_dims
    assert (
        not x_is_transposed and not k_is_transposed
    ), "Dense layer only supports `NN` layout inputs, i.e. non-transposed X and Kernel."

    # Determine X batch dimension
    # - If `batch_first=True` -> (batch, leading..., contracting...)
    # - Otherwise             -> (leading..., batch, contracting...)
    # NOTE: Always assume a single batch dimension
    x_bdim = None
    num_cdims = len(x_contracting_dims)
    if x.ndim >= num_cdims + 2:
        # Assume X is batched if it has at least +2 dimensions more than the number of contracting
        # dimensions.
        if not batch_first:
            _issue_batch_first_warning(
                "TE/JAX `dense()` layer implementation does not officially support sequence-first "
                "inputs and may produce incorrect results when `batch_first=False`. Use "
                "sequence-first inputs at your own discretion.",
            )
        x_bdim = 0 if batch_first else x.ndim - num_cdims - 1

    flatten_axis_x = -len(x_contracting_dims)
    flatten_axis_k = len(k_contracting_dims) - len(kernel.shape)

    casted_x = tex.quantize(
        x, flatten_axis=flatten_axis_x, quantizer=quantizer_set.x, noop_scaled_tensor=True
    )
    casted_x = with_sharding_constraint_by_logical_axes(casted_x, input_axes)

    casted_kernel = tex.quantize(
        kernel,
        flatten_axis=flatten_axis_k,
        quantizer=quantizer_set.kernel,
        noop_scaled_tensor=True,
    )
    casted_kernel = with_sharding_constraint_by_logical_axes(casted_kernel, kernel_axes)

    # GEMM NN
    use_bias = bias is not None
    output = tex.gemm(
        casted_x.get_tensor(usage=TensorUsage.LHS),
        casted_kernel.get_tensor(usage=TensorUsage.RHS),
        contracting_dims=(x_contracting_dims, k_contracting_dims),
        batched_dims=((x_bdim,), ()),
        bias=bias if not tex.gemm_uses_jax_dot() else None,
        fuse_bias=use_bias if not tex.gemm_uses_jax_dot() else False,
        sequence_parallel_output=sequence_parallel_output and not tex.gemm_uses_jax_dot(),
    )

    if use_bias and tex.gemm_uses_jax_dot():
        bias_new_shape = (1,) * (output.ndim - bias.ndim) + bias.shape
        output += jnp.reshape(bias, bias_new_shape)

    ctx = (
        casted_x.get_tensor(usage=TensorUsage.LHS_TRANS),
        casted_kernel.get_tensor(usage=TensorUsage.RHS_TRANS),
        x.shape,
        kernel.shape,
        use_bias,
        quantizer_set,
        flatten_axis_k,
        x_bdim,
    )
    return output, ctx


def _dense_bwd_rule(
    contracting_dims, input_axes, kernel_axes, batch_first, sequence_parallel_output, ctx, grad
):  # pylint: disable=unused-argument
    """Backward pass rule for dense layer transformation.

    Returns:
        Tuple of gradients with respect to inputs
    """
    (
        casted_x_lhs,
        casted_kernel_rhs,
        x_shape,
        kernel_shape,
        use_bias,
        quantizer_set,
        flatten_axis_k,
        x_bdim,
    ) = ctx

    fwd_x_contracting_dims, fwd_k_contracting_dims = map(
        tex.sanitize_dims, (casted_x_lhs.ndim, casted_kernel_rhs.ndim), contracting_dims
    )

    casted_grad, dbias = tex.quantize_dbias(
        grad,
        is_dbias=use_bias,
        flatten_axis=flatten_axis_k,
        quantizer=quantizer_set.dgrad,
        noop_scaled_tensor=True,
    )

    # GEMM NT
    # k_non_contracting_dims calibrated with the shape difference of grad.ndim vs kernel.ndim
    g_contracting_dim = tuple(
        range(grad.ndim - len(kernel_shape) + len(fwd_k_contracting_dims), grad.ndim)
    )
    # k_non_contracting_dims
    k_contracting_dim = tuple(
        dim for dim in range(len(kernel_shape)) if dim not in fwd_k_contracting_dims
    )

    # Get sequence-parallel dimension of the FWD input (if it exists)
    sequence_dim = get_sequence_parallel_dim(input_axes, fwd_x_contracting_dims, (x_bdim,))
    dgrad = tex.gemm(
        casted_grad.get_tensor(usage=TensorUsage.LHS),
        casted_kernel_rhs,
        contracting_dims=(g_contracting_dim, k_contracting_dim),
        batched_dims=((x_bdim,), ()),
        sequence_parallel_output=(
            sequence_dim is not None
            and not sequence_parallel_output
            and not tex.gemm_uses_jax_dot()
        ),
        sequence_dim=(
            None if sequence_parallel_output or tex.gemm_uses_jax_dot() else sequence_dim
        ),
    )
    dgrad = with_sharding_constraint_by_logical_axes(dgrad, input_axes)

    # GEMM TN
    # x_non_contracting_dims
    g_contracting_dim = x_contracting_dim = tuple(
        range(0, len(x_shape) - len(fwd_x_contracting_dims))
    )

    wgrad = tex.gemm(
        casted_x_lhs,
        casted_grad.get_tensor(usage=TensorUsage.RHS),
        contracting_dims=(x_contracting_dim, g_contracting_dim),
        batched_dims=((x_bdim,), (x_bdim,)),
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
    """
    Perform grouped dense (linear) layer transformation with optional quantization.

    Args:
        x: Input tensor of shape (M, K)
        kernel: Weight matrix of shape (G, K, N)
        group_sizes: 1D array of shape (G,) specifying the size of each group
        contracting_dims: Tuple of sequences specifying which dimensions to contract
                          (currently only supports ((1,), (1,)))
        bias: Bias tensor of shape (G, N)
        precision: JAX precision for the GEMM operation
        preferred_element_type: Preferred data type for the output tensor
        group_offset: 1D array containing offsets for each group (not yet implemented)
        quantizer_set: Set of quantizers for FP8 quantization of the input and output

    Returns:
        A jnp.ndarray containing the result of the grouped linear operation
    """
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
    is_noop_quantizer_set = quantizer_set == noop_quantizer_set

    if is_noop_quantizer_set:
        grouped_gemm_x = x
        grouped_gemm_kernel = kernel
        ctx_x = x
        ctx_kernel = kernel
        flatten_axis_k = None
    else:
        x_contracting_dims, k_contracting_dims = contracting_dims
        flatten_axis_x = -len(x_contracting_dims)
        flatten_axis_k = len(k_contracting_dims) - len(kernel.shape) + 1  # +1 for G axis

        assert x.ndim == 2, "Grouped dense expects a 2D input tensor of shape (M, K)"
        assert kernel.ndim == 3, "Grouped dense expects a 3D kernel tensor of shape (G, K, N)"
        # Expected k_contracting_dims == (1,), need to tweak it for grouped_gemm FP8 extra transpose
        # TODO(Hua): Do we have a better way for this? What if is_gemm_with_all_layouts_supported()?
        assert x_contracting_dims == (1,) and k_contracting_dims == (1,), (
            "grouped_dense for FP8 can only handle x_contracting_dims=(1,) "
            "and k_contracting_dims=(1,) for now, "
            f"got {x_contracting_dims=} and {k_contracting_dims=}"
        )

        casted_x = tex.grouped_quantize(
            x, quantizer_set.x, group_sizes, flatten_axis=flatten_axis_x
        )
        casted_kernel = tex.grouped_quantize(
            kernel, quantizer_set.kernel, flatten_axis=flatten_axis_k
        )
        contracting_dims = (x_contracting_dims, k_contracting_dims)

        # For x_contracting_dims == (1,) and k_contracting_dims == (1,), we should have
        # rowwise_casted_x.original_shape == (M, K)
        # colwise_casted_kernel.original_shape == (G, N, K)
        grouped_gemm_x = casted_x.get_tensor(usage=TensorUsage.LHS)
        grouped_gemm_kernel = casted_kernel.get_tensor(usage=TensorUsage.RHS)
        ctx_x = casted_x.get_tensor(usage=TensorUsage.LHS_TRANS)
        ctx_kernel = casted_kernel.get_tensor(usage=TensorUsage.RHS_TRANS)

    output = tex.grouped_gemm(
        grouped_gemm_x,
        grouped_gemm_kernel,
        group_sizes,
        contracting_dims,
        bias,
        precision,
        preferred_element_type,
        group_offset,
    )

    ctx = (
        group_sizes,
        ctx_x,
        ctx_kernel,
        x.shape,
        kernel.shape,
        use_bias,
        is_noop_quantizer_set,
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
        ctx_x,
        ctx_kernel,
        x_shape,
        kernel_shape,
        use_bias,
        is_noop_quantizer_set,
        quantizer_set,
        flatten_axis_k,
    ) = ctx

    if is_noop_quantizer_set:
        # The 1 in range is for excluding the group dimension (shall we use the hardcoded results below?)
        # g_contracting_dim = (1, )
        # k_contracting_dim = (2, )
        g_contracting_dim = tuple(
            range(1 + grad.ndim - len(kernel_shape) + len(fwd_k_contracting_dims), grad.ndim)
        )
        k_contracting_dim = tuple(
            dim for dim in range(1, len(kernel_shape)) if dim not in fwd_k_contracting_dims
        )
        dgrad_contracting_dims = (g_contracting_dim, k_contracting_dim)
        dgrad_grad = grad
        dgrad_kernel_T = ctx_kernel

        # g_contracting_dim = (0, )
        # x_contracting_dim = (0, )
        g_contracting_dim = x_contracting_dim = tuple(
            range(0, len(x_shape) - len(fwd_x_contracting_dims))
        )
        wgrad_contracting_dims = (x_contracting_dim, g_contracting_dim)
        wgrad_x_T = ctx_x
        wgrad_grad = grad
    else:
        casted_grad = tex.grouped_quantize(
            grad, quantizer_set.dgrad, group_sizes, flatten_axis=flatten_axis_k
        )

        # For x_contracting_dims == (1,) and k_contracting_dims == (1,), we need to use
        # g_contracting_dim = (1,) and k_contracting_dim = (2,) to make it work after the
        # extra transpose for FP8 in grouped_gemm
        # TODO(Hua): Do we have a better way for this? What if is_gemm_with_all_layouts_supported()?
        g_contracting_dim = (1,)
        k_contracting_dim = (2,)
        dgrad_contracting_dims = (g_contracting_dim, k_contracting_dim)
        dgrad_grad = casted_grad.get_tensor(usage=TensorUsage.LHS)
        dgrad_kernel_T = ctx_kernel

        # We need to use g_contracting_dim = (0,) and x_contracting_dim = (0,) to make it work
        # after the extra transpose for FP8 in grouped_gemm
        # TODO(Hua): Do we have a better way for this? What if is_gemm_with_all_layouts_supported()?
        g_contracting_dim = (0,)
        x_contracting_dim = (0,)
        wgrad_contracting_dims = (x_contracting_dim, g_contracting_dim)
        wgrad_x_T = ctx_x
        wgrad_grad = casted_grad.get_tensor(usage=TensorUsage.RHS)

    dgrad = tex.grouped_gemm(
        dgrad_grad,
        dgrad_kernel_T,
        group_sizes,
        dgrad_contracting_dims,
        precision=precision,
        preferred_element_type=preferred_element_type,
        group_offset=group_offset,
    )

    wgrad = tex.grouped_gemm(
        wgrad_x_T,
        wgrad_grad,
        group_sizes,
        wgrad_contracting_dims,
        precision=precision,
        preferred_element_type=preferred_element_type,
        group_offset=group_offset,
    )

    group_sizes_grad = None
    dbias = tex.grouped_dbias(grad, group_sizes) if use_bias else None

    return dgrad, wgrad, group_sizes_grad, dbias, quantizer_set


_grouped_dense.defvjp(_grouped_dense_fwd_rule, _grouped_dense_bwd_rule)
