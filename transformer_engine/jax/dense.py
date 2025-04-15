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
from .quantize import QuantizerSet, noop_quantizer_set


def dense(
    x: jnp.ndarray,
    kernel: jnp.ndarray,
    bias: jnp.ndarray = None,
    contracting_dims: Tuple[Sequence[int], Sequence[int]] = ((1,), (0,)),
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
        output = _dense(x, kernel, bias, contracting_dims, quantizer_set)
    return output


@partial(jax.custom_vjp, nondiff_argnums=(3,))
def _dense(x, kernel, bias, contracting_dims, quantizer_set):
    """Internal implementation of dense layer transformation with custom VJP.

    This function implements the core dense layer transformation logic with support
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
    output, _ = _dense_fwd_rule(x, kernel, bias, contracting_dims, quantizer_set)
    return output


def _dense_fwd_rule(x, kernel, bias, contracting_dims, quantizer_set):
    """Forward pass rule for dense layer transformation.

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


def _dense_bwd_rule(contracting_dims, ctx, grad):  # pylint: disable=unused-argument
    """Backward pass rule for dense layer transformation.

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


_dense.defvjp(_dense_fwd_rule, _dense_bwd_rule)


def collective_dense(
    x,
    kernel,
    bias = None,
    contracting_dims: Tuple[Sequence[int], Sequence[int]] = ((1,), (0,)),
    quantizer_set: QuantizerSet = None,
    fuse_gelu: bool = False,
    accumulate: bool = False,
    use_split_accumulator: bool = False,
):
    """Non-FP8 collective/distributed `nvte_cublas_gemm()` with GELU and bias-add fusions."""
    return _collective_dense(x,
                             kernel,
                             bias,
                             contracting_dims,
                             quantizer_set,
                             fuse_gelu,
                             accumulate,
                             use_split_accumulator
                             )


@partial(jax.custom_vjp, nondiff_argnums=(3, 4, 5, 6, 7))
def _collective_dense(
    x,
    kernel,
    bias,
    contracting_dims,
    quantizer_set,
    fuse_gelu,
    accumulate,
    use_split_accumulator,
):
    out, _ = _collective_dense_fwd_rule(
        x,
        kernel,
        bias,
        contracting_dims,
        quantizer_set,
        fuse_gelu,
        accumulate,
        use_split_accumulator,
    )
    return out


def _collective_dense_fwd_rule(
    x,
    kernel,
    bias,
    contracting_dims,
    quantizer_set,
    fuse_gelu,
    accumulate,
    use_split_accumulator,
):
    assert (
        kernel.ndim == 2
    ), "TE/JAX Collective GEMM custom op does not support batched RHS operand in forward mode."

    fuse_bias = bias is not None

    if quantizer_set is None:
        x_rowwise = x
        x_colwise = x
        kernel_colwise = kernel
        kernel_rowwise = kernel
        x_shape = x.shape
        kernel_shape = kernel.shape
    else:
        q_x = tex.quantize(x, quantizer_set.x)
        q_kernel = tex.quantize(kernel, quantizer_set.kernel)
        x_rowwise = q_x.get_rowwise_tensor()
        x_colwise = q_x.get_colwise_tensor()
        kernel_colwise = q_kernel.get_colwise_tensor()
        kernel_rowwise = q_kernel.get_rowwise_tensor()
        x_shape = x_rowwise.data.shape
        kernel_shape = kernel_rowwise.data.shape

    # AG+GEMM:    ([B], M/P, K) --(AG)--> ([B], M, K) x (K, N/P) ------> ([B], M, N/P)
    # (DP, TP, None) --(AG)--> (DP, None, None) x (None, TP) --> (DP, None, TP)
    #
    # GEMM+AR: ([B], M, K/P) x (K/P, N) --(AR)--> ([B], M, N)
    #     (DP, None, TP) x (TP, None) --(AR)--> (DP, None, None)
    out, pre_gelu_out, _ = tex.collective_gemm(
        x_rowwise,
        kernel_colwise,
        bias=bias,
        batched_output=(x.ndim > 2),
        contracting_dims=contracting_dims,
        fuse_gelu=fuse_gelu,
        fuse_bias=fuse_bias,
        accumulate=accumulate,
        use_split_accumulator=use_split_accumulator,
    )

    ctx = (
        x_colwise,
        kernel_rowwise,
        x_shape,
        kernel_shape,
        pre_gelu_out if fuse_gelu else None,
        fuse_bias,
        quantizer_set,
    )

    return out, ctx


def _collective_dense_bwd_rule(
    contracting_dims,
    fuse_gelu,
    accumulate,
    use_split_accumulator,
    ctx,
    grad,
):
    (
        x_colwise,
        kernel_rowwise,
        x_shape,
        kernel_shape,
        pre_gelu_out,
        fuse_bias,
        quantizer_set,
    ) = ctx

    if quantizer_set is None:
        casted_grad = grad
        bgrad = tex.quantization._jax_dbias(grad)
        grad_rowwise = grad
        grad_colwise = grad
    else:
        casted_grad, bgrad = tex.quantize_dbias(
            grad, is_dbias=fuse_bias, quantizer=quantizer_set.dgrad
        )
        grad_rowwise = casted_grad.get_rowwise_tensor()
        grad_colwise = casted_grad.get_colwise_tensor()

    fwd_x_contracting_dims, fwd_k_contracting_dims = contracting_dims
    g_contracting_dim = tuple(
        range(grad.ndim - len(kernel_shape) + len(fwd_k_contracting_dims), grad.ndim)
    )
    k_contracting_dim = tuple(
        dim for dim in range(len(kernel_shape)) if dim not in fwd_k_contracting_dims
    )
    dgrad_contracting_dims = (g_contracting_dim, k_contracting_dim)

    # GEMM TN
    g_contracting_dim = x_contracting_dim = tuple(
        range(0, len(x_shape) - len(fwd_x_contracting_dims))
    )
    wgrad_contracting_dims = (x_contracting_dim, g_contracting_dim)

    # FWD MODE:
    #     AG+GEMM: ([B], M/P, K) --(AG)--> ([B], M, K) x (K, N/P) ------> ([B], M, N/P)
    #  (DP, TP, None) --(AG)--> (DP, None, None) x (None, TP) --> (DP, None, TP)
    #
    #     GEMM+AR: ([B], M, K/P) x (K/P, N) --(AR)--> ([B], M, N)
    #         (DP, None, TP) x (TP, None) --(AR)--> (DP, None, None)

    # DGRAD:
    #    AG+GEMM: ([B], M, N/P) x (K, N/P)^T ----(AR)----> ([B], M, K)
    #        (DP, None, TP) x (None, TP)^T --(AR)--> (DP, None, None)
    #
    #    GEMM+AR:   ([B], M, N) x (K/P, N)^T ------> ([B], M, K/P)
    #        (DP, None, None) x (TP, None)^T --> (DP, None, TP)
    dgrad, dgelu, _ = tex.collective_gemm(
        grad_rowwise,
        kernel_rowwise,
        gelu_input=pre_gelu_out,
        batched_output=(x_colwise.ndim > 2),
        contracting_dims=dgrad_contracting_dims,
        fuse_gelu=fuse_gelu,
        fuse_bias=False,
        grad=True,
        accumulate=accumulate,
        use_split_accumulator=use_split_accumulator,
    )

    # WGRAD:
    #    AG+GEMM: ([B], M/P, K)^T --(AG)--> ([B], M, K)^T x ([B], M, N/P) --> (K, N/P)
    #  (DP, 'tp', None)^T --(AG)-->(DP, None, None)^T x (DP, None, 'tp') --> (None, 'tp')
    #
    #    GEMM+AR: ([B], M, K/P)^T --(AG)--> ([B], M, K)^T x ([B], M, N) ---------> (K/P, N)
    #     (DP, None, 'tp')^T --(AG)--> (DP, None, None)^T x (DP, None, None) ----> (None, None)
    #     Make XLA scatter output in first dim.
    wgrad_rhs = dgelu if fuse_gelu else grad_colwise
    wgrad, _, bgrad = tex.collective_gemm(
        x_colwise,
        wgrad_rhs,
        gelu_input=pre_gelu_out,
        batched_output=False,
        contracting_dims=wgrad_contracting_dims,
        fuse_gelu=False,
        fuse_bias=fuse_bias,
        grad=True,
        accumulate=accumulate,
        use_split_accumulator=use_split_accumulator,
    )

    if not fuse_bias:
        bgrad = None

    return dgrad, wgrad, bgrad


_collective_dense.defvjp(_collective_dense_fwd_rule, _collective_dense_bwd_rule)


def grouped_dense(
    x_list,
    kernel_list,
    bias_list,
    contracting_dims_list,
    quantizer_set_list=None,
):
    """
    Perform grouped_dense layer transformation with optional quantization.

    """
    output_list = _grouped_dense(
        x_list, kernel_list, bias_list, contracting_dims_list, quantizer_set_list
    )
    return output_list


@partial(jax.custom_vjp, nondiff_argnums=(3,))
def _grouped_dense(x_list, kernel_list, bias_list, contracting_dims_list, quantizer_set_list):
    output_list, _ = _grouped_dense_fwd_rule(
        x_list, kernel_list, bias_list, contracting_dims_list, quantizer_set_list
    )
    return output_list


def _grouped_dense_fwd_rule(
    x_list, kernel_list, bias_list, contracting_dims_list, quantizer_set_list
):
    use_bias = bias_list is not None
    output_list = []
    x_rowwise_list = []
    x_colwise_list = []
    kernel_colwise_list = []
    kernel_rowwise_list = []
    x_shape_list = []
    kernel_shape_list = []
    if quantizer_set_list is None:
        x_rowwise_list = x_list
        x_colwise_list = x_list
        kernel_colwise_list = kernel_list
        kernel_rowwise_list = kernel_list
        x_shape_list = [x.shape for x in x_list]
        kernel_shape_list = [kernel.shape for kernel in kernel_list]
    else:
        for i in range(len(x_list)):  # pylint: disable=consider-using-enumerate
            q_x = tex.quantize(x_list[i], quantizer_set_list[i].x)
            q_kernel = tex.quantize(kernel_list[i], quantizer_set_list[i].kernel)
            x_rowwise_list.append(q_x.get_rowwise_tensor())
            x_colwise_list.append(q_x.get_colwise_tensor())
            kernel_colwise_list.append(q_kernel.get_colwise_tensor())
            kernel_rowwise_list.append(q_kernel.get_rowwise_tensor())
            x_shape_list.append(x_rowwise_list[-1].data.shape)
            kernel_shape_list.append(kernel_rowwise_list[-1].data.shape)

    output_list = tex.grouped_gemm(
        x_rowwise_list, kernel_colwise_list, contracting_dims_list, bias_list
    )

    ctx = (
        x_colwise_list,
        kernel_rowwise_list,
        x_shape_list,
        kernel_shape_list,
        use_bias,
        quantizer_set_list,
    )
    return output_list, ctx


def _grouped_dense_bwd_rule(contracting_dims_list, ctx, grad_list):
    (
        colwise_x_list,
        rowwise_kernel_list,
        x_shape_list,
        kernel_shape_list,
        use_bias,
        quantizer_set_list,
    ) = ctx

    group_size = len(grad_list)
    dbias_list = []
    grad_rowwise_list = []
    grad_colwise_list = []
    dgrad_contracting_dims_list = []
    wgrad_contracting_dims_list = []
    for i in range(group_size):
        grad = grad_list[i]
        x_shape = x_shape_list[i]
        kernel_shape = kernel_shape_list[i]
        fwd_contracting_dims = contracting_dims_list[i]

        if quantizer_set_list is None:
            casted_grad = grad
            dbias = tex.quantization._jax_dbias(grad)
            grad_rowwise_list.append(grad)
            grad_colwise_list.append(grad)
        else:
            quantizer_set = quantizer_set_list[i]
            casted_grad, dbias = tex.quantize_dbias(
                grad, is_dbias=use_bias, quantizer=quantizer_set.dgrad
            )
            grad_rowwise_list.append(casted_grad.get_rowwise_tensor())
            grad_colwise_list.append(casted_grad.get_colwise_tensor())
        dbias_list.append(dbias)

        # GEMM NT
        fwd_x_contracting_dims, fwd_k_contracting_dims = fwd_contracting_dims
        g_contracting_dim = tuple(
            range(grad.ndim - len(kernel_shape) + len(fwd_k_contracting_dims), grad.ndim)
        )
        k_contracting_dim = tuple(
            dim for dim in range(len(kernel_shape)) if dim not in fwd_k_contracting_dims
        )
        dgrad_contracting_dims = (g_contracting_dim, k_contracting_dim)
        dgrad_contracting_dims_list.append(dgrad_contracting_dims)

        # GEMM TN
        g_contracting_dim = x_contracting_dim = tuple(
            range(0, len(x_shape) - len(fwd_x_contracting_dims))
        )
        wgrad_contracting_dims = (x_contracting_dim, g_contracting_dim)
        wgrad_contracting_dims_list.append(wgrad_contracting_dims)

    dgrad_list = tex.grouped_gemm(
        grad_rowwise_list, rowwise_kernel_list, dgrad_contracting_dims_list
    )
    wgrad_list = tex.grouped_gemm(colwise_x_list, grad_colwise_list, wgrad_contracting_dims_list)

    return dgrad_list, wgrad_list, dbias_list, quantizer_set_list


_grouped_dense.defvjp(_grouped_dense_fwd_rule, _grouped_dense_bwd_rule)
