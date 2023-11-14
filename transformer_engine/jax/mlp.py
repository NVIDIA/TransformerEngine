# Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""JAX MLP modules"""

from typing import List
from functools import partial

import jax
import jax.numpy as jnp

from .cpp_extensions import transpose, cast_transpose
from .cpp_extensions import gated_gelu, gated_gelu_fp8
from .cpp_extensions import dgated_gelu, dgated_gelu_cast_transpose
from .cpp_extensions import rmsnorm_fwd_fp8, rmsnorm_bwd
from .cpp_extensions import layernorm_fwd_fp8, layernorm_bwd
from .dot import fp8_dot_impl
from .layernorm import canonicalize_layernorm_type
from .fp8 import FP8Helper, FP8MetaPackage


def geglu(x: jnp.ndarray):
    """
    Gated gelu
    """
    assert x.shape[-2] == 2    # Linear + GeLU

    output = _geglu(x)

    return output


@partial(jax.custom_vjp)
def _geglu(x: jnp.ndarray):

    geglu_output, _ = _geglu_fwd_rule(x)

    return geglu_output


def _geglu_fwd_rule(x):
    geglu_output = gated_gelu(x)
    return geglu_output, (x,)


def _geglu_bwd_rule(ctx, g):
    x, = ctx
    assert x.dtype == g.dtype

    dgelu = dgated_gelu(g, x)
    dgelu = jnp.reshape(dgelu, x.shape)
    return (dgelu,)


_geglu.defvjp(_geglu_fwd_rule, _geglu_bwd_rule)


def layernrom_geglu_fp8_mlp(x: jnp.ndarray,
                            gamma: jnp.ndarray,
                            beta: jnp.ndarray,
                            kernels: List[jnp.ndarray],
                            fp8_gemm_pkg: FP8MetaPackage,
                            layernorm_type: str,
                            zero_centered_gamma: bool = False,
                            epsilon: float = 1e-6) -> jnp.ndarray:
    """
    Layernorm + GEMM1 + GeGLU + GEMM2
    """

    assert len(kernels) == 2
    assert fp8_gemm_pkg.num_of_gemm == len(kernels)

    kernel_1 = kernels[0]
    kernel_2 = kernels[1]
    fp8_max = fp8_gemm_pkg.fp8_max
    amax = fp8_gemm_pkg.amax
    scale = fp8_gemm_pkg.scale
    scale_inv = fp8_gemm_pkg.scale_inv

    fwd_dtype = FP8Helper.FWD_DTYPE
    bwd_dtype = FP8Helper.BWD_DTYPE

    layernorm_type = canonicalize_layernorm_type(layernorm_type)
    if layernorm_type == 'rmsnorm':
        assert beta is None, "beta should be None if layernorm_type is 'rmsnorm'"
        assert not zero_centered_gamma, "zero_centered_gamma is not supported " \
            "if layernorm_type is 'rmsnorm'"

    output = _layernrom_geglu_fp8_mlp(x, gamma, beta, kernel_1, kernel_2, fp8_max, amax, scale,
                                      scale_inv, fwd_dtype, bwd_dtype, layernorm_type,
                                      zero_centered_gamma, epsilon)
    return output


@partial(jax.custom_vjp, nondiff_argnums=(9, 10, 11, 12, 13))
def _layernrom_geglu_fp8_mlp(x: jnp.ndarray, gamma: jnp.ndarray, beta: jnp.ndarray,
                             kernel_1: jnp.ndarray, kernel_2: jnp.ndarray, fp8_max: jnp.ndarray,
                             amax: jnp.ndarray, scale: jnp.ndarray, scale_inv: jnp.ndarray,
                             fwd_dtype: jnp.dtype, bwd_dtype: jnp.dtype, layernorm_type: str,
                             zero_centered_gamma: bool, epsilon: float):
    output, _ = _layernrom_geglu_fp8_mlp_fwd_rule(x, gamma, beta, kernel_1, kernel_2, fp8_max, amax,
                                                  scale, scale_inv, fwd_dtype, bwd_dtype,
                                                  layernorm_type, zero_centered_gamma, epsilon)
    return output


def _layernrom_geglu_fp8_mlp_fwd_rule(
        x,
        gamma,
        beta,
        kernel_1,
        kernel_2,
        fp8_max,
        amax,
        scale,
        scale_inv,
        fwd_dtype,
        bwd_dtype,    # pylint: disable=unused-argument
        layernorm_type,
        zero_centered_gamma,
        epsilon):

    # x should be in shape of (batch..., hidden)
    # Kernel_1 should be in shape of (Hidden_in, 2, Hidden_out)
    # Kernel_2 should be in shape of (Hidden_in, Hidden_out)
    assert len(kernel_1.shape) == 3
    assert kernel_1.shape[-2] == 2
    assert len(kernel_2.shape) == 2

    x_contracting_dims = (len(x.shape) - 1,)
    xt_batch_dims = tuple(range(1, x.ndim))

    assert x.shape[x_contracting_dims[0]] == kernel_1.shape[0]
    assert kernel_1.shape[-1] == kernel_2.shape[0]

    amax = FP8Helper.update_amax_history(amax)

    gemm1_x_idx, gemm1_kernel_idx, _ = FP8Helper.get_fp8_meta_indices(0)

    x_amax = amax[gemm1_x_idx, 0:1]
    x_scale = scale[gemm1_x_idx]
    x_scale_inv = scale_inv[gemm1_x_idx]

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

    kernel_1_amax = amax[gemm1_kernel_idx, 0:1]
    kernel_1_scale = scale[gemm1_kernel_idx]
    kernel_1_scale_inv = scale_inv[gemm1_kernel_idx]

    casted_kerenl_1, casted_kerenl_1_t, updated_kernel_1_amax = \
        cast_transpose(kernel_1, kernel_1_amax, kernel_1_scale, kernel_1_scale_inv, fwd_dtype,
                       static_axis_boundary=-1, transpose_axis_boundary=-2)

    # (batch..., hidden_in) x (2, hidden_out, hidden_in)
    dot_1_output = fp8_dot_impl(ln_out, casted_kerenl_1_t, x_scale_inv, kernel_1_scale_inv, x.dtype,
                                (x_contracting_dims, (2,)))

    gemm2_x_idx, gemm2_kernel_idx, _ = FP8Helper.get_fp8_meta_indices(1)

    geglu_out_amax = amax[gemm2_x_idx, 0:1]
    geglu_out_scale = scale[gemm2_x_idx]
    geglu_out_scale_inv = scale_inv[gemm2_x_idx]

    # (batch..., hidden_in) -> (batch..., hidden)
    casted_geglu_out, updated_geglu_amax = gated_gelu_fp8(dot_1_output, geglu_out_amax,
                                                          geglu_out_scale, geglu_out_scale_inv,
                                                          fwd_dtype)

    kernel_2_amax = amax[gemm2_kernel_idx, 0:1]
    kernel_2_scale = scale[gemm2_kernel_idx]
    kernel_2_scale_inv = scale_inv[gemm2_kernel_idx]

    casted_kerenl_2, casted_kerenl_2_t, updated_kernel_2_amax = \
        cast_transpose(kernel_2, kernel_2_amax, kernel_2_scale, kernel_2_scale_inv, fwd_dtype,
                       static_axis_boundary=-1, transpose_axis_boundary=-1)

    # (batch..., hidden_in) x (hidden_out, hidden_in)
    dot_2_output = fp8_dot_impl(casted_geglu_out, casted_kerenl_2_t, geglu_out_scale_inv,
                                kernel_2_scale_inv, x.dtype, (x_contracting_dims, (1,)))

    ctx = (x, ln_out, mu, rsigma, gamma, dot_1_output, casted_geglu_out, casted_kerenl_1,
           casted_kerenl_2, fp8_max, amax, scale, scale_inv, updated_x_amax, updated_geglu_amax,
           updated_kernel_1_amax, updated_kernel_2_amax, x_contracting_dims, xt_batch_dims)

    return dot_2_output, ctx


def _layernrom_geglu_fp8_mlp_bwd_rule(
        fwd_dtype,    # pylint: disable=unused-argument
        bwd_dtype,
        layernorm_type,
        zero_centered_gamma,
        epsilon,
        ctx,
        grad):
    x, ln_out, mu, rsigma, gamma, dot_1_output, casted_geglu_out, \
    casted_kerenl_1, casted_kerenl_2, fp8_max, amax, scale, scale_inv, updated_x_amax, \
    updated_geglu_amax, updated_kernel_1_amax, updated_kernel_2_amax, \
    x_contracting_dims, xt_batch_dims = ctx

    gemm2_x_idx, gemm2_kernel_idx, gemm2_grad_idx = FP8Helper.get_fp8_meta_indices(1)

    grad_amax = amax[gemm2_grad_idx, 0:1]
    grad_scale = scale[gemm2_grad_idx]
    grad_scale_inv = scale_inv[gemm2_grad_idx]

    casted_grad, casted_grad_t, updated_grad_amax = \
        cast_transpose(grad, grad_amax, grad_scale, grad_scale_inv, bwd_dtype,
                       static_axis_boundary=-1, transpose_axis_boundary=-1)

    casted_geglu_out_t = transpose(casted_geglu_out,
                                   static_axis_boundary=-1,
                                   transpose_axis_boundary=-1)

    # (hidden, batch...,) x (hidden, batch...)
    gemm2_x_scale_inv = scale_inv[gemm2_x_idx]
    wgrad_2 = fp8_dot_impl(casted_geglu_out_t, casted_grad_t, gemm2_x_scale_inv, grad_scale_inv,
                           grad.dtype, (xt_batch_dims, xt_batch_dims))

    # (batch..., hidden_out) x (hidden_in, hidden_out)
    kernel_2_scale_inv = scale_inv[gemm2_kernel_idx]
    dgrad_2 = fp8_dot_impl(casted_grad, casted_kerenl_2, grad_scale_inv, kernel_2_scale_inv,
                           grad.dtype, (x_contracting_dims, (1,)))

    gemm1_x_idx, gemm1_kernel_idx, gemm1_grad_idx = FP8Helper.get_fp8_meta_indices(0)

    dgeglu_amax = amax[gemm1_grad_idx, 0:1]
    dgeglu_scale = scale[gemm1_grad_idx]
    dgeglu_scale_inv = scale_inv[gemm1_grad_idx]

    casted_dgeglu, casted_dgeglu_t, updated_dgeglu_amax = dgated_gelu_cast_transpose(
        dgrad_2,
        dot_1_output,
        dgeglu_amax,
        dgeglu_scale,
        dgeglu_scale_inv,
        bwd_dtype,
        static_axis_boundary=-1)

    ln_out_t = transpose(ln_out, static_axis_boundary=-1, transpose_axis_boundary=-1)

    # (hidden, batch...) x (2, hidden, batch...)
    xt_batch_dims_plus_act_dim = tuple(i + 1 for i in xt_batch_dims)
    gemm1_x_scale_inv = scale_inv[gemm1_x_idx]
    wgrad_1 = fp8_dot_impl(ln_out_t, casted_dgeglu_t, gemm1_x_scale_inv, dgeglu_scale_inv,
                           grad.dtype, (xt_batch_dims, xt_batch_dims_plus_act_dim))

    # (batch..., 2, hidden_out) x (hidden_in, 2, hidden_out)
    x_contracting_dims_plus_act_dim = (min(x_contracting_dims),) + tuple(
        i + 1 for i in x_contracting_dims)
    kernel_1_scale_inv = scale_inv[gemm1_kernel_idx]
    dgrad_1 = fp8_dot_impl(casted_dgeglu, casted_kerenl_1, dgeglu_scale_inv, kernel_1_scale_inv,
                           grad.dtype, (x_contracting_dims_plus_act_dim, (
                               1,
                               2,
                           )))

    if layernorm_type == 'layernorm':
        dx, dgamma, dbeta = layernorm_bwd(dgrad_1,
                                          x,
                                          mu,
                                          rsigma,
                                          gamma,
                                          zero_centered_gamma=zero_centered_gamma,
                                          epsilon=epsilon)
    else:
        assert not zero_centered_gamma, "zero_centered_gamma is not supported " \
            "if layernorm_type is 'rmsnorm'"
        dx, dgamma = rmsnorm_bwd(dgrad_1, x, rsigma, gamma, epsilon=epsilon)
        dbeta = None

    amax = amax.at[gemm1_x_idx, 0].set(updated_x_amax[0])
    amax = amax.at[gemm1_kernel_idx, 0].set(updated_kernel_1_amax[0])
    amax = amax.at[gemm1_grad_idx, 0].set(updated_dgeglu_amax[0])
    amax = amax.at[gemm2_x_idx, 0].set(updated_geglu_amax[0])
    amax = amax.at[gemm2_kernel_idx, 0].set(updated_kernel_2_amax[0])
    amax = amax.at[gemm2_grad_idx, 0].set(updated_grad_amax[0])

    scale, scale_inv = FP8Helper.update_fp8_scale(fp8_max, amax, scale)

    return dx, dgamma, dbeta, wgrad_1, wgrad_2, \
           fp8_max, amax, scale, scale_inv


_layernrom_geglu_fp8_mlp.defvjp(_layernrom_geglu_fp8_mlp_fwd_rule,
                                _layernrom_geglu_fp8_mlp_bwd_rule)
