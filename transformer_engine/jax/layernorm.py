# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""JAX layernorm modules"""

from functools import partial
from typing import List, Tuple

import jax
import jax.numpy as jnp

from . import cpp_extensions as tex
from .dot import fp8_dot_impl, get_precision_of_fp8_dot
from .fp8 import FP8Helper, FP8MetaPackage
from .sharding import with_sharding_constraint_by_logical_axes


def canonicalize_layernorm_type(x):
    """
    Canonicalize the layernorm type
    """
    canonicalized = x.lower().strip().replace("-", "").replace("_", "")
    assert canonicalized in ["layernorm", "rmsnorm"]
    return canonicalized


def layernorm(
    inputs: jnp.ndarray,
    gamma: jnp.ndarray,
    beta: jnp.ndarray,
    layernorm_type: str,
    zero_centered_gamma: bool = False,
    epsilon: float = 1e-6,
):
    """
    LN/RMSNorm  wrapper
    Only support layernorm_type in ['layernorm', 'rmsnorm']
    """
    output = _layernorm(
        inputs,
        gamma,
        beta,
        layernorm_type=layernorm_type,
        zero_centered_gamma=zero_centered_gamma,
        epsilon=epsilon,
    )
    return output


@partial(jax.custom_vjp, nondiff_argnums=(3, 4, 5))
def _layernorm(
    x, gamma, beta, layernorm_type: str, zero_centered_gamma: bool = False, epsilon: float = 1e-6
):
    output, _ = _layernorm_fwd_rule(x, gamma, beta, layernorm_type, zero_centered_gamma, epsilon)
    return output


def _layernorm_fwd_rule(
    x, gamma, beta, layernorm_type: str, zero_centered_gamma: bool = False, epsilon: float = 1e-6
):
    layernorm_type = canonicalize_layernorm_type(layernorm_type)
    if layernorm_type == "layernorm":
        output, mu, rsigma = tex.layernorm_fwd(x, gamma, beta, zero_centered_gamma, epsilon)
    elif layernorm_type == "rmsnorm":
        assert (
            not zero_centered_gamma
        ), "zero_centered_gamma is not supported if layernorm_type is 'rmsnorm'"
        output, rsigma = tex.rmsnorm_fwd(x, gamma, epsilon)
        mu = None
    else:
        raise ValueError(f"{layernorm_type=} is not supported.")
    return output, (x, mu, rsigma, gamma)


def _layernorm_bwd_rule(layernorm_type, zero_centered_gamma, epsilon, ctx, dz):
    x, mu, rsigma, gamma = ctx
    if layernorm_type == "layernorm":
        dx, dgamma, dbeta = tex.layernorm_bwd(
            dz, x, mu, rsigma, gamma, zero_centered_gamma=zero_centered_gamma, epsilon=epsilon
        )
    elif layernorm_type == "rmsnorm":
        assert (
            not zero_centered_gamma
        ), "zero_centered_gamma is not supported if layernorm_type is 'rmsnorm'"
        dx, dgamma = tex.rmsnorm_bwd(dz, x, rsigma, gamma, epsilon=epsilon)
        dbeta = None
    else:
        raise ValueError(f"{layernorm_type=} is not supported.")

    return dx, dgamma, dbeta


_layernorm.defvjp(_layernorm_fwd_rule, _layernorm_bwd_rule)


def layernorm_fp8_dot(
    x: jnp.ndarray,
    kernel: jnp.ndarray,
    gamma: jnp.ndarray,
    beta: jnp.ndarray,
    fp8_meta_pkg: FP8MetaPackage,
    layernorm_type: str,
    zero_centered_gamma: bool = False,
    epsilon: float = 1e-6,
    layernorm_input_axes: Tuple[
        str, ...
    ] = None,  # The logic axes of sharding constraint to the layernorm input.
    dot_input_axes: Tuple[
        str, ...
    ] = None,  # The logic axes of sharding constraint to the dot input.
) -> jnp.ndarray:
    """
    Layernorm + FP8 GEMM
    """
    amax_list = fp8_meta_pkg.amax_list
    scale_list = fp8_meta_pkg.scale_list
    fwd_dtype = FP8Helper.FWD_DTYPE
    bwd_dtype = FP8Helper.BWD_DTYPE
    output = _layernorm_fp8_dot(
        x,
        kernel,
        gamma,
        beta,
        amax_list,
        scale_list,
        layernorm_type,
        fwd_dtype,
        bwd_dtype,
        zero_centered_gamma,
        epsilon,
        layernorm_input_axes,
        dot_input_axes,
    )
    return output


@partial(jax.custom_vjp, nondiff_argnums=(6, 7, 8, 9, 10, 11, 12))
def _layernorm_fp8_dot(
    x: jnp.ndarray,
    kernel: jnp.ndarray,
    gamma: jnp.ndarray,
    beta: jnp.ndarray,
    amax_list: List[jnp.ndarray],
    scale_list: List[jnp.ndarray],
    layernorm_type: str,
    fwd_dtype: jnp.dtype,
    bwd_dtype: jnp.dtype,
    zero_centered_gamma: bool,
    epsilon: float,
    layernorm_input_axes: Tuple[str, ...],
    dot_input_axes: Tuple[str, ...],
):
    output, _ = _layernorm_fp8_dot_fwd_rule(
        x,
        kernel,
        gamma,
        beta,
        amax_list,
        scale_list,
        layernorm_type,
        fwd_dtype,
        bwd_dtype,
        zero_centered_gamma,
        epsilon,
        layernorm_input_axes,
        dot_input_axes,
    )
    return output


def _layernorm_fp8_dot_fwd_rule(
    x,
    kernel,
    gamma,
    beta,
    amax_list,
    scale_list,
    layernorm_type,
    fwd_dtype,
    bwd_dtype,  # pylint: disable=unused-argument
    zero_centered_gamma,
    epsilon,
    layernorm_input_axes,
    dot_input_axes,
):

    x_contracting_dims = (len(x.shape) - 1,)
    k_contracting_dims = (0,)
    assert x.shape[-1] == kernel.shape[0]

    maybe_fm32_to_fp32, maybe_fp32_to_fm32 = FP8Helper.generate_fp8_meta_dtype_converter_pair(
        *amax_list, *scale_list
    )
    amax_list = maybe_fm32_to_fp32(*amax_list)
    scale_list = maybe_fm32_to_fp32(*scale_list)

    fp8_dtype_list = [fwd_dtype, fwd_dtype, bwd_dtype]
    scale_list, scale_inv_list = FP8MetaPackage.update_fp8_scale(
        amax_list, scale_list, fp8_dtype_list
    )
    amax_list = FP8MetaPackage.update_amax_list(amax_list)

    x_amax = amax_list[FP8MetaPackage.INPUT_IDX][0:1]
    x_scale = scale_list[FP8MetaPackage.INPUT_IDX]
    x_scale_inv = scale_inv_list[FP8MetaPackage.INPUT_IDX]

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

    kernel_amax = amax_list[FP8MetaPackage.WEIGHT_IDX][0:1]
    kernel_scale = scale_list[FP8MetaPackage.WEIGHT_IDX]
    kernel_scale_inv = scale_inv_list[FP8MetaPackage.WEIGHT_IDX]

    # Kernel in (hidden_in, hidden_out...)
    # Note (Ming Huang): Use cast only to allow XLA handle tranpose for avoiding
    # unnecessary copy to break FP8 GEMM pattern matching.
    casted_kernel, updated_kernel_amax = tex.cast_fp8(
        kernel, kernel_amax, kernel_scale, kernel_scale_inv, fwd_dtype
    )

    ln_out = with_sharding_constraint_by_logical_axes(ln_out, dot_input_axes)

    # (batch..., hidden_in) x (hidden_in, hidden_out...)
    output = fp8_dot_impl(
        ln_out,
        casted_kernel,
        x_scale_inv,
        kernel_scale_inv,
        x.dtype,
        (x_contracting_dims, k_contracting_dims),
        get_precision_of_fp8_dot(FP8Helper.FP8_2X_ACC_FPROP),
    )

    ctx = (
        ln_out,
        casted_kernel,
        amax_list,
        scale_list,
        scale_inv_list,
        updated_x_amax,
        updated_kernel_amax,
        x.shape,
        kernel.shape,
        mu,
        rsigma,
        x,
        gamma,
        x_contracting_dims,
        k_contracting_dims,
        maybe_fp32_to_fm32,
    )

    return output, ctx


def _layernorm_fp8_dot_bwd_rule(
    layernorm_type,
    fwd_dtype,  # pylint: disable=unused-argument
    bwd_dtype,
    zero_centered_gamma,
    epsilon,
    layernorm_input_axes,
    dot_input_axes,  # pylint: disable=unused-argument
    ctx,
    grad,
):
    (
        ln_out_,
        casted_kernel,
        amax_list,
        scale_list,
        scale_inv_list,
        updated_x_amax,
        updated_kernel_amax,
        x_shape,
        kernel_shape,
        mu,
        rsigma,
        x,
        gamma,
        x_contracting_dims,
        k_contracting_dims,
        maybe_fp32_to_fm32,
    ) = ctx

    ln_out_t = tex.transpose(ln_out_, static_axis_boundary=-1, transpose_axis_boundary=-1)

    grad_amax = amax_list[FP8MetaPackage.GRAD_IDX][0:1]
    grad_scale = scale_list[FP8MetaPackage.GRAD_IDX]
    grad_scale_inv = scale_inv_list[FP8MetaPackage.GRAD_IDX]

    casted_grad, casted_grad_t, updated_grad_amax = tex.cast_transpose(
        grad,
        grad_amax,
        grad_scale,
        grad_scale_inv,
        bwd_dtype,
        static_axis_boundary=-1,
        transpose_axis_boundary=min(x_contracting_dims),
    )

    xt_constracting_dim = tuple(range(len(x_contracting_dims), len(x_shape)))
    gt_constracting_dim = tuple(range(grad.ndim - len(xt_constracting_dim), grad.ndim))
    x_scale_inv = scale_inv_list[FP8MetaPackage.INPUT_IDX]
    wgrad = fp8_dot_impl(
        ln_out_t,
        casted_grad_t,
        x_scale_inv,
        grad_scale_inv,
        grad.dtype,
        (xt_constracting_dim, gt_constracting_dim),
        get_precision_of_fp8_dot(FP8Helper.FP8_2X_ACC_WGRAD),
    )

    g_for_dgrad_constracting_dim = tuple(
        range(grad.ndim - len(kernel_shape) + len(k_contracting_dims), grad.ndim)
    )
    k_constracting_dim = tuple(range(len(k_contracting_dims), len(kernel_shape)))
    kernel_scale_inv = scale_inv_list[FP8MetaPackage.WEIGHT_IDX]
    dgrad = fp8_dot_impl(
        casted_grad,
        casted_kernel,
        grad_scale_inv,
        kernel_scale_inv,
        grad.dtype,
        (g_for_dgrad_constracting_dim, k_constracting_dim),
        get_precision_of_fp8_dot(FP8Helper.FP8_2X_ACC_DGRAD),
    )

    dgrad = with_sharding_constraint_by_logical_axes(dgrad, layernorm_input_axes)
    if layernorm_type == "layernorm":
        dx, dgamma, dbeta = tex.layernorm_bwd(
            dgrad, x, mu, rsigma, gamma, zero_centered_gamma=zero_centered_gamma, epsilon=epsilon
        )
    else:
        assert (
            not zero_centered_gamma
        ), "zero_centered_gamma is not supported if layernorm_type is 'rmsnorm'"
        dx, dgamma = tex.rmsnorm_bwd(dgrad, x, rsigma, gamma, epsilon=epsilon)
        dbeta = None

    amax_list[FP8MetaPackage.INPUT_IDX] = (
        amax_list[FP8MetaPackage.INPUT_IDX].at[0].set(updated_x_amax[0])
    )
    amax_list[FP8MetaPackage.WEIGHT_IDX] = (
        amax_list[FP8MetaPackage.WEIGHT_IDX].at[0].set(updated_kernel_amax[0])
    )
    amax_list[FP8MetaPackage.GRAD_IDX] = (
        amax_list[FP8MetaPackage.GRAD_IDX].at[0].set(updated_grad_amax[0])
    )

    amax_list = maybe_fp32_to_fm32(*amax_list)
    scale_list = maybe_fp32_to_fm32(*scale_list)

    return dx, wgrad, dgamma, dbeta, amax_list, scale_list


_layernorm_fp8_dot.defvjp(_layernorm_fp8_dot_fwd_rule, _layernorm_fp8_dot_bwd_rule)
