# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""JAX MLP modules"""

from typing import List, Tuple, Sequence, Union, Callable
from functools import partial

import jax
import jax.numpy as jnp
from jax.ad_checkpoint import checkpoint_name

from . import cpp_extensions as tex
from .dot import fp8_dot_impl, get_precision_of_fp8_dot, quantize
from .layernorm import canonicalize_layernorm_type
from .fp8 import FP8Helper, FP8MetaPackage
from .sharding import with_sharding_constraint_by_logical_axes


def activation_lu(x: jnp.ndarray, activation_type: Sequence[Union[str, Callable]]):
    """
    Activation Unit
    """
    if len(activation_type) > 1:
        assert x.shape[-2] == 2  # Linear + GeLU
    output = _activation_lu(x, activation_type)
    return output


@partial(jax.custom_vjp, nondiff_argnums=(1,))
def _activation_lu(x: jnp.ndarray, activation_type: Sequence[Union[str, Callable]]):

    _output, _ = _activation_lu_fwd_rule(x, activation_type)

    return _output


def _activation_lu_fwd_rule(x, activation_type):
    fwd_output = tex.act_lu(x, activation_type)
    return fwd_output, (x,)


def _activation_lu_bwd_rule(activation_type, ctx, g):
    (x,) = ctx
    assert x.dtype == g.dtype

    dx = tex.dact_lu(g, x, activation_type)
    dx = jnp.reshape(dx, x.shape)
    return (dx,)


_activation_lu.defvjp(_activation_lu_fwd_rule, _activation_lu_bwd_rule)


def fused_layernorm_fp8_mlp(
    x: jnp.ndarray,
    gamma: jnp.ndarray,
    beta: jnp.ndarray,
    kernels: List[jnp.ndarray],
    biases: List[jnp.ndarray],
    fp8_meta_pkgs: List[FP8MetaPackage],
    layernorm_type: str,
    zero_centered_gamma: bool = False,
    epsilon: float = 1e-6,
    layernorm_input_axes: Tuple[str, ...] = None,
    dot_1_input_axes: Tuple[str, ...] = None,
    dot_2_input_axes: Tuple[str, ...] = None,
    ffn1_ckpt_name: str = "ffn1",
    ffn2_ckpt_name: str = "ffn2",
    activation_type: Sequence[Union[str, Callable]] = ("gelu",),
    use_bias: bool = True,
) -> jnp.ndarray:
    """
    Layernorm + GEMM1 + bias + activation + GEMM2 + bias
    """

    assert len(kernels) == 2
    assert len(fp8_meta_pkgs) == len(kernels)

    kernel_1 = kernels[0]
    kernel_2 = kernels[1]
    bias_1 = biases[0]
    bias_2 = biases[1]
    amax_list_1 = fp8_meta_pkgs[0].amax_list
    amax_list_2 = fp8_meta_pkgs[1].amax_list
    scale_list_1 = fp8_meta_pkgs[0].scale_list
    scale_list_2 = fp8_meta_pkgs[1].scale_list

    fwd_dtype = FP8Helper.FWD_DTYPE
    bwd_dtype = FP8Helper.BWD_DTYPE

    layernorm_type = canonicalize_layernorm_type(layernorm_type)
    if layernorm_type == "rmsnorm":
        assert beta is None, "beta should be None if layernorm_type is 'rmsnorm'"
        assert (
            not zero_centered_gamma
        ), "zero_centered_gamma is not supported if layernorm_type is 'rmsnorm'"

    output = _fused_layernorm_fp8_mlp(
        x,
        gamma,
        beta,
        kernel_1,
        kernel_2,
        bias_1,
        bias_2,
        amax_list_1,
        amax_list_2,
        scale_list_1,
        scale_list_2,
        fwd_dtype,
        bwd_dtype,
        layernorm_type,
        zero_centered_gamma,
        epsilon,
        layernorm_input_axes,
        dot_1_input_axes,
        dot_2_input_axes,
        ffn1_ckpt_name,
        ffn2_ckpt_name,
        activation_type,
        use_bias,
    )
    return output


@partial(jax.custom_vjp, nondiff_argnums=(11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22))
def _fused_layernorm_fp8_mlp(
    x: jnp.ndarray,
    gamma: jnp.ndarray,
    beta: jnp.ndarray,
    kernel_1: jnp.ndarray,
    kernel_2: jnp.ndarray,
    bias_1: jnp.ndarray,
    bias_2: jnp.ndarray,
    amax_list_1: List[jnp.ndarray],
    amax_list_2: List[jnp.ndarray],
    scale_list_1: List[jnp.ndarray],
    scale_list_2: List[jnp.ndarray],
    fwd_dtype: jnp.dtype,
    bwd_dtype: jnp.dtype,
    layernorm_type: str,
    zero_centered_gamma: bool,
    epsilon: float,
    layernorm_input_axes: Tuple[str, ...],
    dot_1_input_axes: Tuple[str, ...],
    dot_2_input_axes: Tuple[str, ...],
    ffn1_ckpt_name: str,
    ffn2_ckpt_name: str,
    activation_type: Sequence[Union[str, Callable]],
    use_bias: bool,
):
    output, _ = _fused_layernorm_fp8_mlp_fwd_rule(
        x,
        gamma,
        beta,
        kernel_1,
        kernel_2,
        bias_1,
        bias_2,
        amax_list_1,
        amax_list_2,
        scale_list_1,
        scale_list_2,
        fwd_dtype,
        bwd_dtype,
        layernorm_type,
        zero_centered_gamma,
        epsilon,
        layernorm_input_axes,
        dot_1_input_axes,
        dot_2_input_axes,
        ffn1_ckpt_name,
        ffn2_ckpt_name,
        activation_type,
        use_bias,
    )
    return output


def _fused_layernorm_fp8_mlp_fwd_rule(
    x,
    gamma,
    beta,
    kernel_1,
    kernel_2,
    bias_1,
    bias_2,
    amax_list_1,
    amax_list_2,
    scale_list_1,
    scale_list_2,
    fwd_dtype,
    bwd_dtype,  # pylint: disable=unused-argument
    layernorm_type,
    zero_centered_gamma,
    epsilon,
    layernorm_input_axes,
    dot_1_input_axes,
    dot_2_input_axes,
    ffn1_ckpt_name,
    ffn2_ckpt_name,
    activation_type,
    use_bias,
):

    # x should be in shape of (batch..., hidden)
    # Kernel_1 should be in shape of (Hidden_in, 1, Hidden_out)
    # Kernel_2 should be in shape of (Hidden_in, Hidden_out)
    assert len(kernel_1.shape) == 3
    assert kernel_1.shape[-2] == len(activation_type)
    assert len(kernel_2.shape) == 2

    x_contracting_dims = (len(x.shape) - 1,)
    xt_batch_dims = tuple(range(1, x.ndim))

    assert x.shape[x_contracting_dims[0]] == kernel_1.shape[0]
    assert kernel_1.shape[-1] == kernel_2.shape[0]

    maybe_fm32_to_fp32, maybe_fp32_to_fm32 = FP8Helper.generate_fp8_meta_dtype_converter_pair(
        *amax_list_1, *scale_list_1, *amax_list_2, *scale_list_2
    )
    amax_list_1 = maybe_fm32_to_fp32(*amax_list_1)
    scale_list_1 = maybe_fm32_to_fp32(*scale_list_1)
    amax_list_2 = maybe_fm32_to_fp32(*amax_list_2)
    scale_list_2 = maybe_fm32_to_fp32(*scale_list_2)

    fp8_dtype_list = [fwd_dtype, fwd_dtype, bwd_dtype]
    scale_list_1, scale_inv_list_1 = FP8MetaPackage.update_fp8_scale(
        amax_list_1, scale_list_1, fp8_dtype_list
    )
    amax_list_1 = FP8MetaPackage.update_amax_list(amax_list_1)
    scale_list_2, scale_inv_list_2 = FP8MetaPackage.update_fp8_scale(
        amax_list_2, scale_list_2, fp8_dtype_list
    )
    amax_list_2 = FP8MetaPackage.update_amax_list(amax_list_2)

    x_amax = amax_list_1[FP8MetaPackage.INPUT_IDX][0:1]
    x_scale = scale_list_1[FP8MetaPackage.INPUT_IDX]
    x_scale_inv = scale_inv_list_1[FP8MetaPackage.INPUT_IDX]

    x = with_sharding_constraint_by_logical_axes(x, layernorm_input_axes)

    if layernorm_type == "layernorm":
        ln_out, mu, rsigma, updated_x_amax = tex.layernorm_fwd_fp8(
            x,
            gamma,
            beta,
            x_amax,
            x_scale,
            x_scale_inv,
            out_dtype=fwd_dtype,
            zero_centered_gamma=zero_centered_gamma,
            epsilon=epsilon,
        )
    else:
        assert (
            not zero_centered_gamma
        ), "zero_centered_gamma is not supported if layernorm_type is 'rmsnorm'"
        ln_out, rsigma, updated_x_amax = tex.rmsnorm_fwd_fp8(
            x, gamma, x_amax, x_scale, x_scale_inv, out_dtype=fwd_dtype, epsilon=epsilon
        )
        mu = None

    assert x.shape == ln_out.shape

    kernel_1_amax = amax_list_1[FP8MetaPackage.WEIGHT_IDX][0:1]
    kernel_1_scale = scale_list_1[FP8MetaPackage.WEIGHT_IDX]
    kernel_1_scale_inv = scale_inv_list_1[FP8MetaPackage.WEIGHT_IDX]

    # Note (Ming Huang): Use cast only to allow XLA handle tranpose for avoiding
    # unnecessary copy to break FP8 GEMM pattern matching.
    casted_kernel_1, updated_kernel_1_amax = tex.cast_fp8(
        kernel_1, kernel_1_amax, kernel_1_scale, kernel_1_scale_inv, fwd_dtype
    )

    ln_out = with_sharding_constraint_by_logical_axes(ln_out, dot_1_input_axes)

    # (batch..., hidden_in) x (hidden_in, hidden_out)
    dot_1_output = fp8_dot_impl(
        ln_out,
        casted_kernel_1,
        x_scale_inv,
        kernel_1_scale_inv,
        x.dtype,
        (x_contracting_dims, (0,)),
        get_precision_of_fp8_dot(FP8Helper.FP8_2X_ACC_FPROP),
    )
    if use_bias:
        bias_1_shape = bias_1.shape
        bias_1_new_shape = (1,) * (dot_1_output.ndim - bias_1.ndim) + bias_1_shape
        dot_1_output += jnp.reshape(bias_1, bias_1_new_shape)
    else:
        bias_1_shape = None
    dot_1_output = checkpoint_name(dot_1_output, ffn1_ckpt_name)

    activation_lu_out_amax = amax_list_2[FP8MetaPackage.INPUT_IDX][0:1]
    activation_lu_out_scale = scale_list_2[FP8MetaPackage.INPUT_IDX]
    activation_lu_out_scale_inv = scale_inv_list_2[FP8MetaPackage.INPUT_IDX]

    # (batch..., hidden_in) -> (batch..., hidden)
    casted_activation_lu_out, updated_activation_lu_amax = tex.act_lu_fp8(
        dot_1_output,
        activation_lu_out_amax,
        activation_lu_out_scale,
        activation_lu_out_scale_inv,
        fwd_dtype,
        activation_type,
    )

    casted_activation_lu_out = with_sharding_constraint_by_logical_axes(
        casted_activation_lu_out, dot_2_input_axes
    )

    kernel_2_scale = scale_list_2[FP8MetaPackage.WEIGHT_IDX]
    kernel_2_scale_inv = scale_inv_list_2[FP8MetaPackage.WEIGHT_IDX]
    # Note (Ming Huang): Use native cast to allow XLA handle tranpose for avoiding
    # unnecessary copy to break FP8 GEMM pattern matching.
    casted_kernel_2, updated_kernel_2_amax = quantize(kernel_2, fwd_dtype, kernel_2_scale)

    # (batch..., hidden_in) x (hidden_out, hidden_in)
    dot_2_output = fp8_dot_impl(
        casted_activation_lu_out,
        casted_kernel_2,
        activation_lu_out_scale_inv,
        kernel_2_scale_inv,
        x.dtype,
        (x_contracting_dims, (0,)),
        get_precision_of_fp8_dot(FP8Helper.FP8_2X_ACC_FPROP),
    )

    if use_bias:
        bias_2_shape = bias_2.shape
        bias_2_new_shape = (1,) * (dot_2_output.ndim - bias_2.ndim) + bias_2_shape
        dot_2_output += jnp.reshape(bias_2, bias_2_new_shape)
    else:
        bias_2_shape = None

    dot_2_output = checkpoint_name(dot_2_output, ffn2_ckpt_name)

    ctx = (
        x,
        ln_out,
        mu,
        rsigma,
        gamma,
        dot_1_output,
        casted_activation_lu_out,
        casted_kernel_1,
        casted_kernel_2,
        amax_list_1,
        amax_list_2,
        scale_list_1,
        scale_list_2,
        scale_inv_list_1,
        scale_inv_list_2,
        updated_x_amax,
        updated_activation_lu_amax,
        updated_kernel_1_amax,
        updated_kernel_2_amax,
        x_contracting_dims,
        xt_batch_dims,
        bias_1_shape,
        bias_2_shape,
        maybe_fp32_to_fm32,
    )

    return dot_2_output, ctx


def _fused_layernorm_fp8_mlp_bwd_rule(
    fwd_dtype,  # pylint: disable=unused-argument
    bwd_dtype,
    layernorm_type,
    zero_centered_gamma,
    epsilon,
    layernorm_input_axes,
    dot_1_input_axes,
    dot_2_input_axes,
    ffn1_ckpt_name,  # pylint: disable=unused-argument
    ffn2_ckpt_name,  # pylint: disable=unused-argument
    activation_type,
    use_bias,
    ctx,
    grad,
):
    (
        x,
        ln_out,
        mu,
        rsigma,
        gamma,
        dot_1_output,
        casted_activation_lu_out,
        casted_kernel_1,
        casted_kernel_2,
        amax_list_1,
        amax_list_2,
        scale_list_1,
        scale_list_2,
        scale_inv_list_1,
        scale_inv_list_2,
        updated_x_amax,
        updated_activation_lu_amax,
        updated_kernel_1_amax,
        updated_kernel_2_amax,
        x_contracting_dims,
        xt_batch_dims,
        bias_1_shape,
        bias_2_shape,
        maybe_fp32_to_fm32,
    ) = ctx

    grad_amax = amax_list_2[FP8MetaPackage.GRAD_IDX][0:1]
    grad_scale = scale_list_2[FP8MetaPackage.GRAD_IDX]
    grad_scale_inv = scale_inv_list_2[FP8MetaPackage.GRAD_IDX]

    # Since the sharding of outputs should be the same as dot_1's input
    grad = with_sharding_constraint_by_logical_axes(grad, dot_1_input_axes)
    if use_bias:
        casted_grad, casted_grad_t, dbias_2, updated_grad_amax = tex.dbias_cast_transpose(
            grad,
            grad_amax,
            grad_scale,
            grad_scale_inv,
            bwd_dtype,
            static_axis_boundary=-1,
            transpose_axis_boundary=-1,
        )
        dbias_2 = jnp.reshape(dbias_2, bias_2_shape)
    else:
        casted_grad, casted_grad_t, updated_grad_amax = tex.cast_transpose(
            grad,
            grad_amax,
            grad_scale,
            grad_scale_inv,
            bwd_dtype,
            static_axis_boundary=-1,
            transpose_axis_boundary=-1,
        )
        dbias_2 = None

    casted_activation_lu_out_t = tex.transpose(
        casted_activation_lu_out, static_axis_boundary=-1, transpose_axis_boundary=-1
    )

    # (hidden, batch...,) x (hidden, batch...)
    gemm2_x_scale_inv = scale_inv_list_2[FP8MetaPackage.INPUT_IDX]
    wgrad_2 = fp8_dot_impl(
        casted_activation_lu_out_t,
        casted_grad_t,
        gemm2_x_scale_inv,
        grad_scale_inv,
        grad.dtype,
        (xt_batch_dims, xt_batch_dims),
        get_precision_of_fp8_dot(FP8Helper.FP8_2X_ACC_WGRAD),
    )

    # (batch..., hidden_out) x (hidden_in, hidden_out)
    kernel_2_scale_inv = scale_inv_list_2[FP8MetaPackage.WEIGHT_IDX]
    dgrad_2 = fp8_dot_impl(
        casted_grad,
        casted_kernel_2,
        grad_scale_inv,
        kernel_2_scale_inv,
        grad.dtype,
        (x_contracting_dims, (1,)),
        get_precision_of_fp8_dot(FP8Helper.FP8_2X_ACC_DGRAD),
    )

    dgrad_2 = with_sharding_constraint_by_logical_axes(dgrad_2, dot_2_input_axes)

    dactivation_lu_amax = amax_list_1[FP8MetaPackage.GRAD_IDX][0:1]
    dactivation_lu_scale = scale_list_1[FP8MetaPackage.GRAD_IDX]
    dactivation_lu_scale_inv = scale_inv_list_1[FP8MetaPackage.GRAD_IDX]

    if len(activation_type) > 1:  # if gated
        if use_bias:
            dactivation_lu = tex.dact_lu(dgrad_2, dot_1_output, activation_type)
            casted_dactivation_lu, casted_dactivation_lu_t, dbias_1, updated_dactivation_lu_amax = (
                tex.dbias_cast_transpose(
                    dactivation_lu,
                    dactivation_lu_amax,
                    dactivation_lu_scale,
                    dactivation_lu_scale_inv,
                    bwd_dtype,
                    static_axis_boundary=-1,
                    transpose_axis_boundary=-2,
                )
            )
            dbias_1 = jnp.reshape(dbias_1, bias_1_shape)
        else:
            casted_dactivation_lu, casted_dactivation_lu_t, updated_dactivation_lu_amax = (
                tex.dgated_act_lu_cast_transpose(
                    dgrad_2,
                    dot_1_output,
                    dactivation_lu_amax,
                    dactivation_lu_scale,
                    dactivation_lu_scale_inv,
                    bwd_dtype,
                    static_axis_boundary=-1,
                    activation_type=activation_type,
                )
            )
            dbias_1 = None
    else:
        if use_bias:
            casted_dactivation_lu, casted_dactivation_lu_t, dbias_1, updated_dactivation_lu_amax = (
                tex.dact_lu_dbias_cast_transpose(
                    dgrad_2,
                    dot_1_output,
                    dactivation_lu_amax,
                    dactivation_lu_scale,
                    dactivation_lu_scale_inv,
                    bwd_dtype,
                    static_axis_boundary=-1,
                    transpose_axis_boundary=-2,
                    activation_type=activation_type,
                )
            )
            dbias_1 = jnp.reshape(dbias_1, bias_1_shape)
        else:
            dactivation_lu = tex.dact_lu(dgrad_2, dot_1_output, activation_type)
            casted_dactivation_lu, casted_dactivation_lu_t, updated_dactivation_lu_amax = (
                tex.cast_transpose(
                    dactivation_lu,
                    dactivation_lu_amax,
                    dactivation_lu_scale,
                    dactivation_lu_scale_inv,
                    bwd_dtype,
                    static_axis_boundary=-1,
                    transpose_axis_boundary=-2,
                )
            )
            dbias_1 = None

    ln_out_t = tex.transpose(ln_out, static_axis_boundary=-1, transpose_axis_boundary=-1)

    # (hidden, batch...) x (hidden, batch...)
    gemm1_x_scale_inv = scale_inv_list_1[FP8MetaPackage.INPUT_IDX]
    xt_batch_dims_2 = tuple(i + 1 for i in xt_batch_dims)
    wgrad_1 = fp8_dot_impl(
        ln_out_t,
        casted_dactivation_lu_t,
        gemm1_x_scale_inv,
        dactivation_lu_scale_inv,
        grad.dtype,
        (xt_batch_dims, xt_batch_dims_2),
        get_precision_of_fp8_dot(FP8Helper.FP8_2X_ACC_WGRAD),
    )

    x_contracting_dims = (
        (min(x_contracting_dims),) + tuple(i + 1 for i in x_contracting_dims),
        (1, 2),
    )
    kernel_1_scale_inv = scale_inv_list_1[FP8MetaPackage.WEIGHT_IDX]
    dgrad_1 = fp8_dot_impl(
        casted_dactivation_lu,
        casted_kernel_1,
        dactivation_lu_scale_inv,
        kernel_1_scale_inv,
        grad.dtype,
        x_contracting_dims,
        get_precision_of_fp8_dot(FP8Helper.FP8_2X_ACC_DGRAD),
    )

    dgrad_1 = with_sharding_constraint_by_logical_axes(dgrad_1, layernorm_input_axes)

    if layernorm_type == "layernorm":
        dx, dgamma, dbeta = tex.layernorm_bwd(
            dgrad_1, x, mu, rsigma, gamma, zero_centered_gamma=zero_centered_gamma, epsilon=epsilon
        )
    else:
        assert (
            not zero_centered_gamma
        ), "zero_centered_gamma is not supported if layernorm_type is 'rmsnorm'"
        dx, dgamma = tex.rmsnorm_bwd(dgrad_1, x, rsigma, gamma, epsilon=epsilon)
        dbeta = None

    amax_list_1[FP8MetaPackage.INPUT_IDX] = (
        amax_list_1[FP8MetaPackage.INPUT_IDX].at[0].set(updated_x_amax[0])
    )
    amax_list_1[FP8MetaPackage.WEIGHT_IDX] = (
        amax_list_1[FP8MetaPackage.WEIGHT_IDX].at[0].set(updated_kernel_1_amax[0])
    )
    amax_list_1[FP8MetaPackage.GRAD_IDX] = (
        amax_list_1[FP8MetaPackage.GRAD_IDX].at[0].set(updated_dactivation_lu_amax[0])
    )
    amax_list_2[FP8MetaPackage.INPUT_IDX] = (
        amax_list_2[FP8MetaPackage.INPUT_IDX].at[0].set(updated_activation_lu_amax[0])
    )
    amax_list_2[FP8MetaPackage.WEIGHT_IDX] = (
        amax_list_2[FP8MetaPackage.WEIGHT_IDX].at[0].set(updated_kernel_2_amax)
    )
    amax_list_2[FP8MetaPackage.GRAD_IDX] = (
        amax_list_2[FP8MetaPackage.GRAD_IDX].at[0].set(updated_grad_amax[0])
    )

    amax_list_1 = maybe_fp32_to_fm32(*amax_list_1)
    scale_list_1 = maybe_fp32_to_fm32(*scale_list_1)
    amax_list_2 = maybe_fp32_to_fm32(*amax_list_2)
    scale_list_2 = maybe_fp32_to_fm32(*scale_list_2)

    return (
        dx,
        dgamma,
        dbeta,
        wgrad_1,
        wgrad_2,
        dbias_1,
        dbias_2,
        amax_list_1,
        amax_list_2,
        scale_list_1,
        scale_list_2,
    )


_fused_layernorm_fp8_mlp.defvjp(
    _fused_layernorm_fp8_mlp_fwd_rule, _fused_layernorm_fp8_mlp_bwd_rule
)
