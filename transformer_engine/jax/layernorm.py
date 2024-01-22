# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""JAX layernorm modules"""

from functools import partial
from typing import Tuple

import jax
import jax.numpy as jnp

from .cpp_extensions import cast_fp8, cast_transpose, transpose
from .cpp_extensions import rmsnorm_fwd, rmsnorm_fwd_fp8, rmsnorm_bwd
from .cpp_extensions import layernorm_fwd, layernorm_fwd_fp8, layernorm_bwd
from .dot import fp8_dot_impl, get_precision_of_fp8_dot
from .fp8 import FP8Helper, FP8MetaPackage
from .sharding import with_sharding_constraint_by_logical_axes


def canonicalize_layernorm_type(x):
    '''
    Canonicalize the layernorm type
    '''
    canonicalized = x.lower().strip().replace('-', '').replace('_', '')
    assert canonicalized in ['layernorm', 'rmsnorm']
    return canonicalized


def layernorm(inputs: jnp.ndarray,
              gamma: jnp.ndarray,
              beta: jnp.ndarray,
              layernorm_type: str,
              zero_centered_gamma: bool = False,
              epsilon: float = 1e-6):
    """
    LN/RMSNorm  wrapper
    Only support layernorm_type in ['layernorm', 'rmsnorm']
    """
    output = _layernorm(inputs,
                        gamma,
                        beta,
                        layernorm_type=layernorm_type,
                        zero_centered_gamma=zero_centered_gamma,
                        epsilon=epsilon)
    return output


@partial(jax.custom_vjp, nondiff_argnums=(3, 4, 5))
def _layernorm(x,
               gamma,
               beta,
               layernorm_type: str,
               zero_centered_gamma: bool = False,
               epsilon: float = 1e-6):
    output, _ = _layernorm_fwd_rule(x, gamma, beta, layernorm_type, zero_centered_gamma, epsilon)
    return output


def _layernorm_fwd_rule(x,
                        gamma,
                        beta,
                        layernorm_type: str,
                        zero_centered_gamma: bool = False,
                        epsilon: float = 1e-6):
    layernorm_type = canonicalize_layernorm_type(layernorm_type)
    if layernorm_type == 'layernorm':
        output, mu, rsigma = layernorm_fwd(x, gamma, beta, zero_centered_gamma, epsilon)
    elif layernorm_type == 'rmsnorm':
        assert not zero_centered_gamma, "zero_centered_gamma is not supported " \
            "if layernorm_type is 'rmsnorm'"
        output, rsigma = rmsnorm_fwd(x, gamma, epsilon)
        mu = None
    else:
        raise ValueError(f"{layernorm_type=} is not supported.")
    return output, (x, mu, rsigma, gamma)


def _layernorm_bwd_rule(layernorm_type, zero_centered_gamma, epsilon, ctx, dz):
    x, mu, rsigma, gamma = ctx
    if layernorm_type == 'layernorm':
        dx, dgamma, dbeta = layernorm_bwd(dz,
                                          x,
                                          mu,
                                          rsigma,
                                          gamma,
                                          zero_centered_gamma=zero_centered_gamma,
                                          epsilon=epsilon)
    elif layernorm_type == 'rmsnorm':
        assert not zero_centered_gamma, "zero_centered_gamma is not supported " \
            "if layernorm_type is 'rmsnorm'"
        dx, dgamma = rmsnorm_bwd(dz, x, rsigma, gamma, epsilon=epsilon)
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
        str, ...] = None,    # The logic axes of sharding constraint to the layernorm input.
    dot_input_axes: Tuple[str,
                          ...] = None    # The logic axes of sharding constraint to the dot input.
) -> jnp.ndarray:
    """
    Layernorm + FP8 GEMM
    """
    fp8_max = fp8_meta_pkg.fp8_max
    amax = fp8_meta_pkg.amax
    scale = fp8_meta_pkg.scale
    scale_inv = fp8_meta_pkg.scale_inv
    fwd_dtype = FP8Helper.FWD_DTYPE
    bwd_dtype = FP8Helper.BWD_DTYPE
    output = _layernorm_fp8_dot(x, kernel, gamma, beta, fp8_max, amax, scale, scale_inv,
                                layernorm_type, fwd_dtype, bwd_dtype, zero_centered_gamma, epsilon,
                                layernorm_input_axes, dot_input_axes)
    return output


@partial(jax.custom_vjp, nondiff_argnums=(8, 9, 10, 11, 12, 13, 14))
def _layernorm_fp8_dot(x: jnp.ndarray, kernel: jnp.ndarray, gamma: jnp.ndarray, beta: jnp.ndarray,
                       fp8_max: jnp.ndarray, amax: jnp.ndarray, scale: jnp.ndarray,
                       scale_inv: jnp.ndarray, layernorm_type: str, fwd_dtype: jnp.dtype,
                       bwd_dtype: jnp.dtype, zero_centered_gamma: bool, epsilon: float,
                       layernorm_input_axes: Tuple[str, ...], dot_input_axes: Tuple[str, ...]):
    output, _ = _layernorm_fp8_dot_fwd_rule(x, kernel, gamma, beta, fp8_max, amax, scale, scale_inv,
                                            layernorm_type, fwd_dtype, bwd_dtype,
                                            zero_centered_gamma, epsilon, layernorm_input_axes,
                                            dot_input_axes)
    return output


def _layernorm_fp8_dot_fwd_rule(
        x,
        kernel,
        gamma,
        beta,
        fp8_max,
        amax,
        scale,
        scale_inv,
        layernorm_type,
        fwd_dtype,
        bwd_dtype,    # pylint: disable=unused-argument
        zero_centered_gamma,
        epsilon,
        layernorm_input_axes,
        dot_input_axes):

    x_contracting_dims = (len(x.shape) - 1,)
    k_contracting_dims = (0,)
    assert x.shape[-1] == kernel.shape[0]

    amax = FP8Helper.update_amax_history(amax)

    gemm_x_idx, gemm_kernel_idx, _ = FP8Helper.get_fp8_meta_indices(0)

    x_amax = amax[gemm_x_idx, 0:1]
    x_scale = scale[gemm_x_idx]
    x_scale_inv = scale_inv[gemm_x_idx]

    x = with_sharding_constraint_by_logical_axes(x, layernorm_input_axes)

    if layernorm_type == 'layernorm':
        ln_out, mu, rsigma, updated_x_amax = layernorm_fwd_fp8(
            x,
            gamma,
            beta,
            x_amax,
            x_scale,
            x_scale_inv,
            out_dtype=fwd_dtype,
            zero_centered_gamma=zero_centered_gamma,
            epsilon=epsilon)
    else:
        assert not zero_centered_gamma, "zero_centered_gamma is not supported " \
            "if layernorm_type is 'rmsnorm'"
        ln_out, rsigma, updated_x_amax = rmsnorm_fwd_fp8(x,
                                                         gamma,
                                                         x_amax,
                                                         x_scale,
                                                         x_scale_inv,
                                                         out_dtype=fwd_dtype,
                                                         epsilon=epsilon)
        mu = None

    assert x.shape == ln_out.shape

    kernel_amax = amax[gemm_kernel_idx, 0:1]
    kernel_scale = scale[gemm_kernel_idx]
    kernel_scale_inv = scale_inv[gemm_kernel_idx]

    # Kernel in (hidden_in, hidden_out...)
    # Note (Ming Huang): Use cast only to allow XLA handle tranpose for avoiding
    # unnecessary copy to break FP8 GEMM pattern matching.
    casted_kernel, updated_kernel_amax = \
        cast_fp8(kernel, kernel_amax, kernel_scale, kernel_scale_inv, fwd_dtype)

    ln_out = with_sharding_constraint_by_logical_axes(ln_out, dot_input_axes)

    # (batch..., hidden_in) x (hidden_in, hidden_out...)
    output = fp8_dot_impl(ln_out, casted_kernel, x_scale_inv, kernel_scale_inv, x.dtype,
                          (x_contracting_dims, k_contracting_dims),
                          get_precision_of_fp8_dot(FP8Helper.FP8_2X_ACC_FPROP))

    ctx = (ln_out, casted_kernel, fp8_max, amax, scale, scale_inv, updated_x_amax,
           updated_kernel_amax, x.shape, kernel.shape, mu, rsigma, x, gamma, x_contracting_dims,
           k_contracting_dims)

    return output, ctx


def _layernorm_fp8_dot_bwd_rule(
        layernorm_type,
        fwd_dtype,    # pylint: disable=unused-argument
        bwd_dtype,
        zero_centered_gamma,
        epsilon,
        layernorm_input_axes,
        dot_input_axes,    # pylint: disable=unused-argument
        ctx,
        grad):
    ln_out_, casted_kernel, fp8_max, amax, scale, scale_inv, \
    updated_x_amax, updated_kernel_amax, \
    x_shape, kernel_shape, mu, rsigma, x, gamma, \
    x_contracting_dims, k_contracting_dims = ctx

    ln_out_t = transpose(ln_out_, static_axis_boundary=-1, transpose_axis_boundary=-1)

    gemm_x_idx, gemm_kernel_idx, gemm_grad_idx = FP8Helper.get_fp8_meta_indices(0)

    grad_amax = amax[gemm_grad_idx, 0:1]
    grad_scale = scale[gemm_grad_idx]
    grad_scale_inv = scale_inv[gemm_grad_idx]

    casted_grad, casted_grad_t, updated_grad_amax = \
        cast_transpose(grad, grad_amax, grad_scale, grad_scale_inv, bwd_dtype,
                       static_axis_boundary=-1, transpose_axis_boundary=min(x_contracting_dims))

    xt_constracting_dim = tuple(range(len(x_contracting_dims), len(x_shape)))
    gt_constracting_dim = tuple(range(grad.ndim - len(xt_constracting_dim), grad.ndim))
    x_scale_inv = scale_inv[gemm_x_idx]
    wgrad = fp8_dot_impl(ln_out_t, casted_grad_t, x_scale_inv, grad_scale_inv, grad.dtype,
                         (xt_constracting_dim, gt_constracting_dim),
                         get_precision_of_fp8_dot(FP8Helper.FP8_2X_ACC_WGRAD))

    g_for_dgrad_constracting_dim = tuple(
        range(grad.ndim - len(kernel_shape) + len(k_contracting_dims), grad.ndim))
    k_constracting_dim = tuple(range(len(k_contracting_dims), len(kernel_shape)))
    kernel_scale_inv = scale_inv[gemm_kernel_idx]
    dgrad = fp8_dot_impl(casted_grad, casted_kernel, grad_scale_inv, kernel_scale_inv, grad.dtype,
                         (g_for_dgrad_constracting_dim, k_constracting_dim),
                         get_precision_of_fp8_dot(FP8Helper.FP8_2X_ACC_DGRAD))

    dgrad = with_sharding_constraint_by_logical_axes(dgrad, layernorm_input_axes)
    if layernorm_type == 'layernorm':
        dx, dgamma, dbeta = layernorm_bwd(dgrad,
                                          x,
                                          mu,
                                          rsigma,
                                          gamma,
                                          zero_centered_gamma=zero_centered_gamma,
                                          epsilon=epsilon)
    else:
        assert not zero_centered_gamma, "zero_centered_gamma is not supported " \
            "if layernorm_type is 'rmsnorm'"
        dx, dgamma = rmsnorm_bwd(dgrad, x, rsigma, gamma, epsilon=epsilon)
        dbeta = None

    amax = amax.at[gemm_x_idx, 0].set(updated_x_amax[0])
    amax = amax.at[gemm_kernel_idx, 0].set(updated_kernel_amax[0])
    amax = amax.at[gemm_grad_idx, 0].set(updated_grad_amax[0])

    scale, scale_inv = FP8Helper.update_fp8_scale(fp8_max, amax, scale)

    return dx, wgrad, \
           dgamma, dbeta, \
           fp8_max, amax, scale, scale_inv


_layernorm_fp8_dot.defvjp(_layernorm_fp8_dot_fwd_rule, _layernorm_fp8_dot_bwd_rule)
