# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""JAX MLP modules"""

from typing import List, Tuple
from functools import partial

import jax
import jax.numpy as jnp
from jax.ad_checkpoint import checkpoint_name

from .cpp_extensions import cast_fp8, transpose, cast_transpose
from .cpp_extensions import gelu as te_gelu
from .cpp_extensions import gelu_fp8, dgelu, dgelu_dbias_cast_transpose
from .cpp_extensions import gated_gelu, gated_gelu_fp8
from .cpp_extensions import dgated_gelu, dgated_gelu_cast_transpose
from .cpp_extensions import rmsnorm_fwd_fp8, rmsnorm_bwd
from .cpp_extensions import layernorm_fwd_fp8, layernorm_bwd
from .dot import fp8_dot_impl, get_precision_of_fp8_dot, quantize
from .layernorm import canonicalize_layernorm_type
from .fp8 import FP8Helper, FP8MetaPackage
from .sharding import with_sharding_constraint_by_logical_axes


def gelu(x: jnp.ndarray):
    """
    Gelu
    """
    output = _gelu(x)
    return output


@partial(jax.custom_vjp)
def _gelu(x: jnp.ndarray):

    geglu_output, _ = _gelu_fwd_rule(x)

    return geglu_output


def _gelu_fwd_rule(x):
    geglu_output = te_gelu(x)
    return geglu_output, (x,)


def _gelu_bwd_rule(ctx, g):
    x, = ctx
    assert x.dtype == g.dtype

    dx = dgelu(g, x)
    dx = jnp.reshape(dx, x.shape)
    return (dx,)


_gelu.defvjp(_gelu_fwd_rule, _gelu_bwd_rule)


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

    dx = dgated_gelu(g, x)
    dx = jnp.reshape(dx, x.shape)
    return (dx,)


_geglu.defvjp(_geglu_fwd_rule, _geglu_bwd_rule)


def layernorm_geglu_fp8_mlp(x: jnp.ndarray,
                            gamma: jnp.ndarray,
                            beta: jnp.ndarray,
                            kernels: List[jnp.ndarray],
                            fp8_gemm_pkg: FP8MetaPackage,
                            layernorm_type: str,
                            zero_centered_gamma: bool = False,
                            epsilon: float = 1e-6,
                            layernorm_input_axes: Tuple[str, ...] = None,
                            dot_1_input_axes: Tuple[str, ...] = None,
                            dot_2_input_axes: Tuple[str, ...] = None,
                            ffn1_ckpt_name: str = 'ffn1',
                            ffn2_ckpt_name: str = 'ffn2') -> jnp.ndarray:
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

    output = _layernorm_geglu_fp8_mlp(x, gamma, beta, kernel_1, kernel_2, fp8_max, amax, scale,
                                      scale_inv, fwd_dtype, bwd_dtype, layernorm_type,
                                      zero_centered_gamma, epsilon, layernorm_input_axes,
                                      dot_1_input_axes, dot_2_input_axes, ffn1_ckpt_name,
                                      ffn2_ckpt_name)
    return output


@partial(jax.custom_vjp, nondiff_argnums=(9, 10, 11, 12, 13, 14, 15, 16, 17, 18))
def _layernorm_geglu_fp8_mlp(x: jnp.ndarray, gamma: jnp.ndarray, beta: jnp.ndarray,
                             kernel_1: jnp.ndarray, kernel_2: jnp.ndarray, fp8_max: jnp.ndarray,
                             amax: jnp.ndarray, scale: jnp.ndarray, scale_inv: jnp.ndarray,
                             fwd_dtype: jnp.dtype, bwd_dtype: jnp.dtype, layernorm_type: str,
                             zero_centered_gamma: bool, epsilon: float,
                             layernorm_input_axes: Tuple[str, ...],
                             dot_1_input_axes: Tuple[str, ...], dot_2_input_axes: Tuple[str, ...],
                             ffn1_ckpt_name: str, ffn2_ckpt_name: str):
    output, _ = _layernorm_geglu_fp8_mlp_fwd_rule(x, gamma, beta, kernel_1, kernel_2, fp8_max, amax,
                                                  scale, scale_inv, fwd_dtype, bwd_dtype,
                                                  layernorm_type, zero_centered_gamma, epsilon,
                                                  layernorm_input_axes, dot_1_input_axes,
                                                  dot_2_input_axes, ffn1_ckpt_name, ffn2_ckpt_name)
    return output


def _layernorm_geglu_fp8_mlp_fwd_rule(
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
        epsilon,
        layernorm_input_axes,
        dot_1_input_axes,
        dot_2_input_axes,
        ffn1_ckpt_name,
        ffn2_ckpt_name):

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

    kernel_1_amax = amax[gemm1_kernel_idx, 0:1]
    kernel_1_scale = scale[gemm1_kernel_idx]
    kernel_1_scale_inv = scale_inv[gemm1_kernel_idx]

    # Note (Ming Huang): Use cast only to allow XLA handle tranpose for avoiding
    # unnecessary copy to break FP8 GEMM pattern matching.
    casted_kernel_1, updated_kernel_1_amax = \
        cast_fp8(kernel_1, kernel_1_amax, kernel_1_scale, kernel_1_scale_inv, fwd_dtype)

    ln_out = with_sharding_constraint_by_logical_axes(ln_out, dot_1_input_axes)

    # (batch..., hidden_in) x (hidden_in, 2, hidden_out)
    dot_1_output = fp8_dot_impl(ln_out, casted_kernel_1, x_scale_inv, kernel_1_scale_inv, x.dtype,
                                (x_contracting_dims, (0,)),
                                get_precision_of_fp8_dot(FP8Helper.FP8_2X_ACC_FPROP))
    dot_1_output = checkpoint_name(dot_1_output, ffn1_ckpt_name)

    gemm2_x_idx, gemm2_kernel_idx, _ = FP8Helper.get_fp8_meta_indices(1)

    geglu_out_amax = amax[gemm2_x_idx, 0:1]
    geglu_out_scale = scale[gemm2_x_idx]
    geglu_out_scale_inv = scale_inv[gemm2_x_idx]

    # (batch..., hidden_in) -> (batch..., hidden)
    casted_geglu_out, updated_geglu_amax = gated_gelu_fp8(dot_1_output, geglu_out_amax,
                                                          geglu_out_scale, geglu_out_scale_inv,
                                                          fwd_dtype)

    casted_geglu_out = with_sharding_constraint_by_logical_axes(casted_geglu_out, dot_2_input_axes)

    kernel_2_scale = scale[gemm2_kernel_idx]
    kernel_2_scale_inv = scale_inv[gemm2_kernel_idx]
    # Note (Ming Huang): Use native cast to allow XLA handle tranpose for avoiding
    # unnecessary copy to break FP8 GEMM pattern matching.
    casted_kernel_2, updated_kernel_2_amax = quantize(kernel_2, fwd_dtype, kernel_2_scale)

    # (batch..., hidden_in) x (hidden_out, hidden_in)
    dot_2_output = fp8_dot_impl(casted_geglu_out, casted_kernel_2, geglu_out_scale_inv,
                                kernel_2_scale_inv, x.dtype, (x_contracting_dims, (0,)),
                                get_precision_of_fp8_dot(FP8Helper.FP8_2X_ACC_FPROP))
    dot_2_output = checkpoint_name(dot_2_output, ffn2_ckpt_name)

    ctx = (x, ln_out, mu, rsigma, gamma, dot_1_output, casted_geglu_out, casted_kernel_1,
           casted_kernel_2, fp8_max, amax, scale, scale_inv, updated_x_amax, updated_geglu_amax,
           updated_kernel_1_amax, updated_kernel_2_amax, x_contracting_dims, xt_batch_dims)

    return dot_2_output, ctx


def _layernorm_geglu_fp8_mlp_bwd_rule(
        fwd_dtype,    # pylint: disable=unused-argument
        bwd_dtype,
        layernorm_type,
        zero_centered_gamma,
        epsilon,
        layernorm_input_axes,
        dot_1_input_axes,
        dot_2_input_axes,
        ffn1_ckpt_name,    # pylint: disable=unused-argument
        ffn2_ckpt_name,    # pylint: disable=unused-argument
        ctx,
        grad):
    x, ln_out, mu, rsigma, gamma, dot_1_output, casted_geglu_out, \
    casted_kernel_1, casted_kernel_2, fp8_max, amax, scale, scale_inv, updated_x_amax, \
    updated_geglu_amax, updated_kernel_1_amax, updated_kernel_2_amax, \
    x_contracting_dims, xt_batch_dims = ctx

    gemm2_x_idx, gemm2_kernel_idx, gemm2_grad_idx = FP8Helper.get_fp8_meta_indices(1)

    grad_amax = amax[gemm2_grad_idx, 0:1]
    grad_scale = scale[gemm2_grad_idx]
    grad_scale_inv = scale_inv[gemm2_grad_idx]

    # Since the sharding of outputs should be the same as dot_1's input
    grad = with_sharding_constraint_by_logical_axes(grad, dot_1_input_axes)

    casted_grad, casted_grad_t, updated_grad_amax = \
        cast_transpose(grad, grad_amax, grad_scale, grad_scale_inv, bwd_dtype,
                       static_axis_boundary=-1, transpose_axis_boundary=-1)

    casted_geglu_out_t = transpose(casted_geglu_out,
                                   static_axis_boundary=-1,
                                   transpose_axis_boundary=-1)

    # (hidden, batch...,) x (hidden, batch...)
    gemm2_x_scale_inv = scale_inv[gemm2_x_idx]
    wgrad_2 = fp8_dot_impl(casted_geglu_out_t, casted_grad_t, gemm2_x_scale_inv, grad_scale_inv,
                           grad.dtype, (xt_batch_dims, xt_batch_dims),
                           get_precision_of_fp8_dot(FP8Helper.FP8_2X_ACC_WGRAD))

    # (batch..., hidden_out) x (hidden_in, hidden_out)
    kernel_2_scale_inv = scale_inv[gemm2_kernel_idx]
    dgrad_2 = fp8_dot_impl(casted_grad, casted_kernel_2, grad_scale_inv, kernel_2_scale_inv,
                           grad.dtype, (x_contracting_dims, (1,)),
                           get_precision_of_fp8_dot(FP8Helper.FP8_2X_ACC_DGRAD))

    dgrad_2 = with_sharding_constraint_by_logical_axes(dgrad_2, dot_2_input_axes)

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
                           grad.dtype, (xt_batch_dims, xt_batch_dims_plus_act_dim),
                           get_precision_of_fp8_dot(FP8Helper.FP8_2X_ACC_WGRAD))

    # (batch..., 2, hidden_out) x (hidden_in, 2, hidden_out)
    x_contracting_dims_plus_act_dim = (min(x_contracting_dims),) + tuple(
        i + 1 for i in x_contracting_dims)
    kernel_1_scale_inv = scale_inv[gemm1_kernel_idx]
    dgrad_1 = fp8_dot_impl(casted_dgeglu, casted_kernel_1, dgeglu_scale_inv, kernel_1_scale_inv,
                           grad.dtype, (x_contracting_dims_plus_act_dim, (1, 2)),
                           get_precision_of_fp8_dot(FP8Helper.FP8_2X_ACC_DGRAD))

    dgrad_1 = with_sharding_constraint_by_logical_axes(dgrad_1, layernorm_input_axes)

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
    amax = amax.at[gemm2_kernel_idx, 0].set(updated_kernel_2_amax)
    amax = amax.at[gemm2_grad_idx, 0].set(updated_grad_amax[0])

    scale, scale_inv = FP8Helper.update_fp8_scale(fp8_max, amax, scale)

    return dx, dgamma, dbeta, wgrad_1, wgrad_2, \
           fp8_max, amax, scale, scale_inv


_layernorm_geglu_fp8_mlp.defvjp(_layernorm_geglu_fp8_mlp_fwd_rule,
                                _layernorm_geglu_fp8_mlp_bwd_rule)


def layernorm_gelu_fp8_mlp(x: jnp.ndarray,
                           gamma: jnp.ndarray,
                           beta: jnp.ndarray,
                           kernels: List[jnp.ndarray],
                           biases: List[jnp.ndarray],
                           fp8_gemm_pkg: FP8MetaPackage,
                           layernorm_type: str,
                           zero_centered_gamma: bool = False,
                           epsilon: float = 1e-6,
                           layernorm_input_axes: Tuple[str, ...] = None,
                           dot_1_input_axes: Tuple[str, ...] = None,
                           dot_2_input_axes: Tuple[str, ...] = None,
                           ffn1_ckpt_name: str = 'ffn1',
                           ffn2_ckpt_name: str = 'ffn2') -> jnp.ndarray:
    """
    Layernorm + GEMM1 + bias + GeLU + GEMM2 + bias
    """

    assert len(kernels) == 2
    assert fp8_gemm_pkg.num_of_gemm == len(kernels)

    kernel_1 = kernels[0]
    kernel_2 = kernels[1]
    bias_1 = biases[0]
    bias_2 = biases[1]
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

    output = _layernorm_gelu_fp8_mlp(x, gamma, beta, kernel_1, kernel_2, bias_1, bias_2, fp8_max,
                                     amax, scale, scale_inv, fwd_dtype, bwd_dtype, layernorm_type,
                                     zero_centered_gamma, epsilon, layernorm_input_axes,
                                     dot_1_input_axes, dot_2_input_axes, ffn1_ckpt_name,
                                     ffn2_ckpt_name)
    return output


@partial(jax.custom_vjp, nondiff_argnums=(11, 12, 13, 14, 15, 16, 17, 18, 19, 20))
def _layernorm_gelu_fp8_mlp(x: jnp.ndarray, gamma: jnp.ndarray, beta: jnp.ndarray,
                            kernel_1: jnp.ndarray, kernel_2: jnp.ndarray, bias_1: jnp.ndarray,
                            bias_2: jnp.ndarray, fp8_max: jnp.ndarray, amax: jnp.ndarray,
                            scale: jnp.ndarray, scale_inv: jnp.ndarray, fwd_dtype: jnp.dtype,
                            bwd_dtype: jnp.dtype, layernorm_type: str, zero_centered_gamma: bool,
                            epsilon: float, layernorm_input_axes: Tuple[str, ...],
                            dot_1_input_axes: Tuple[str, ...], dot_2_input_axes: Tuple[str, ...],
                            ffn1_ckpt_name: str, ffn2_ckpt_name: str):
    output, _ = _layernorm_gelu_fp8_mlp_fwd_rule(x, gamma, beta, kernel_1, kernel_2, bias_1, bias_2,
                                                 fp8_max, amax, scale, scale_inv, fwd_dtype,
                                                 bwd_dtype, layernorm_type, zero_centered_gamma,
                                                 epsilon, layernorm_input_axes, dot_1_input_axes,
                                                 dot_2_input_axes, ffn1_ckpt_name, ffn2_ckpt_name)
    return output


def _layernorm_gelu_fp8_mlp_fwd_rule(
        x,
        gamma,
        beta,
        kernel_1,
        kernel_2,
        bias_1,
        bias_2,
        fp8_max,
        amax,
        scale,
        scale_inv,
        fwd_dtype,
        bwd_dtype,    # pylint: disable=unused-argument
        layernorm_type,
        zero_centered_gamma,
        epsilon,
        layernorm_input_axes,
        dot_1_input_axes,
        dot_2_input_axes,
        ffn1_ckpt_name,
        ffn2_ckpt_name):

    # x should be in shape of (batch..., hidden)
    # Kernel_1 should be in shape of (Hidden_in, 1, Hidden_out)
    # Kernel_2 should be in shape of (Hidden_in, Hidden_out)
    assert len(kernel_1.shape) == 3
    assert kernel_1.shape[-2] == 1
    assert len(kernel_2.shape) == 2

    x_contracting_dims = (len(x.shape) - 1,)
    xt_batch_dims = tuple(range(1, x.ndim))

    assert x.shape[x_contracting_dims[0]] == kernel_1.shape[0]
    assert kernel_1.shape[-1] == kernel_2.shape[0]

    # Squeeze act axis
    # (hidden_in, 1, hidden_out) -> (hidden_in, hidden_out)
    kernel_1 = jnp.squeeze(kernel_1, axis=-2)

    amax = FP8Helper.update_amax_history(amax)

    gemm1_x_idx, gemm1_kernel_idx, _ = FP8Helper.get_fp8_meta_indices(0)

    x_amax = amax[gemm1_x_idx, 0:1]
    x_scale = scale[gemm1_x_idx]
    x_scale_inv = scale_inv[gemm1_x_idx]

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

    kernel_1_amax = amax[gemm1_kernel_idx, 0:1]
    kernel_1_scale = scale[gemm1_kernel_idx]
    kernel_1_scale_inv = scale_inv[gemm1_kernel_idx]

    # Note (Ming Huang): Use cast only to allow XLA handle tranpose for avoiding
    # unnecessary copy to break FP8 GEMM pattern matching.
    casted_kernel_1, updated_kernel_1_amax = \
        cast_fp8(kernel_1, kernel_1_amax, kernel_1_scale, kernel_1_scale_inv, fwd_dtype)

    ln_out = with_sharding_constraint_by_logical_axes(ln_out, dot_1_input_axes)

    # (batch..., hidden_in) x (hidden_in, hidden_out)
    dot_1_output = fp8_dot_impl(ln_out, casted_kernel_1, x_scale_inv, kernel_1_scale_inv, x.dtype,
                                (x_contracting_dims, (0,)),
                                get_precision_of_fp8_dot(FP8Helper.FP8_2X_ACC_FPROP))

    bias_1_shape = (1,) * (dot_1_output.ndim - bias_1.ndim) + bias_1.shape
    dot_1_output += jnp.reshape(bias_1, bias_1_shape)
    dot_1_output = checkpoint_name(dot_1_output, ffn1_ckpt_name)

    gemm2_x_idx, gemm2_kernel_idx, _ = FP8Helper.get_fp8_meta_indices(1)

    gelu_out_amax = amax[gemm2_x_idx, 0:1]
    gelu_out_scale = scale[gemm2_x_idx]
    gelu_out_scale_inv = scale_inv[gemm2_x_idx]

    # (batch..., hidden_in) -> (batch..., hidden)
    casted_gelu_out, updated_gelu_amax = gelu_fp8(dot_1_output, gelu_out_amax, gelu_out_scale,
                                                  gelu_out_scale_inv, fwd_dtype)

    casted_gelu_out = with_sharding_constraint_by_logical_axes(casted_gelu_out, dot_2_input_axes)

    kernel_2_scale = scale[gemm2_kernel_idx]
    kernel_2_scale_inv = scale_inv[gemm2_kernel_idx]
    # Note (Ming Huang): Use native cast to allow XLA handle tranpose for avoiding
    # unnecessary copy to break FP8 GEMM pattern matching.
    casted_kernel_2, updated_kernel_2_amax = quantize(kernel_2, fwd_dtype, kernel_2_scale)

    # (batch..., hidden_in) x (hidden_out, hidden_in)
    dot_2_output = fp8_dot_impl(casted_gelu_out, casted_kernel_2, gelu_out_scale_inv,
                                kernel_2_scale_inv, x.dtype, (x_contracting_dims, (0,)),
                                get_precision_of_fp8_dot(FP8Helper.FP8_2X_ACC_FPROP))

    bias_2_shape = (1,) * (dot_2_output.ndim - bias_2.ndim) + bias_2.shape
    dot_2_output += jnp.reshape(bias_2, bias_2_shape)
    dot_2_output = checkpoint_name(dot_2_output, ffn2_ckpt_name)

    ctx = (x, ln_out, mu, rsigma, gamma, dot_1_output, casted_gelu_out, casted_kernel_1,
           casted_kernel_2, fp8_max, amax, scale, scale_inv, updated_x_amax, updated_gelu_amax,
           updated_kernel_1_amax, updated_kernel_2_amax, x_contracting_dims, xt_batch_dims,
           bias_1.shape, bias_2.shape)

    return dot_2_output, ctx


def _layernorm_gelu_fp8_mlp_bwd_rule(
        fwd_dtype,    # pylint: disable=unused-argument
        bwd_dtype,
        layernorm_type,
        zero_centered_gamma,
        epsilon,
        layernorm_input_axes,
        dot_1_input_axes,
        dot_2_input_axes,
        ffn1_ckpt_name,    # pylint: disable=unused-argument
        ffn2_ckpt_name,    # pylint: disable=unused-argument
        ctx,
        grad):
    x, ln_out, mu, rsigma, gamma, dot_1_output, casted_gelu_out, \
    casted_kernel_1, casted_kernel_2, fp8_max, amax, scale, scale_inv, updated_x_amax, \
    updated_gelu_amax, updated_kernel_1_amax, updated_kernel_2_amax, \
    x_contracting_dims, xt_batch_dims, bias_1_shape, bias_2_shape= ctx

    gemm2_x_idx, gemm2_kernel_idx, gemm2_grad_idx = FP8Helper.get_fp8_meta_indices(1)

    grad_amax = amax[gemm2_grad_idx, 0:1]
    grad_scale = scale[gemm2_grad_idx]
    grad_scale_inv = scale_inv[gemm2_grad_idx]

    # Since the sharding of outputs should be the same as dot_1's input
    grad = with_sharding_constraint_by_logical_axes(grad, dot_1_input_axes)

    casted_grad, casted_grad_t, updated_grad_amax = \
        cast_transpose(grad, grad_amax, grad_scale, grad_scale_inv, bwd_dtype,
                       static_axis_boundary=-1, transpose_axis_boundary=-1)

    casted_gelu_out_t = transpose(casted_gelu_out,
                                  static_axis_boundary=-1,
                                  transpose_axis_boundary=-1)

    dbias_2 = jnp.sum(grad, axis=(i for i in range(grad.ndim - 1)))
    dbias_2 = jnp.reshape(dbias_2, bias_2_shape)

    # (hidden, batch...,) x (hidden, batch...)
    gemm2_x_scale_inv = scale_inv[gemm2_x_idx]
    wgrad_2 = fp8_dot_impl(casted_gelu_out_t, casted_grad_t, gemm2_x_scale_inv, grad_scale_inv,
                           grad.dtype, (xt_batch_dims, xt_batch_dims),
                           get_precision_of_fp8_dot(FP8Helper.FP8_2X_ACC_WGRAD))

    # (batch..., hidden_out) x (hidden_in, hidden_out)
    kernel_2_scale_inv = scale_inv[gemm2_kernel_idx]
    dgrad_2 = fp8_dot_impl(casted_grad, casted_kernel_2, grad_scale_inv, kernel_2_scale_inv,
                           grad.dtype, (x_contracting_dims, (1,)),
                           get_precision_of_fp8_dot(FP8Helper.FP8_2X_ACC_DGRAD))

    dgrad_2 = with_sharding_constraint_by_logical_axes(dgrad_2, dot_2_input_axes)

    gemm1_x_idx, gemm1_kernel_idx, gemm1_grad_idx = FP8Helper.get_fp8_meta_indices(0)

    dgelu_amax = amax[gemm1_grad_idx, 0:1]
    dgelu_scale = scale[gemm1_grad_idx]
    dgelu_scale_inv = scale_inv[gemm1_grad_idx]

    casted_dgelu, casted_dgelu_t, dbias_1, updated_dgelu_amax = dgelu_dbias_cast_transpose(
        dgrad_2,
        dot_1_output,
        dgelu_amax,
        dgelu_scale,
        dgelu_scale_inv,
        bwd_dtype,
        static_axis_boundary=-1,
        transpose_axis_boundary=-1)

    dbias_1 = jnp.reshape(dbias_1, bias_1_shape)

    ln_out_t = transpose(ln_out, static_axis_boundary=-1, transpose_axis_boundary=-1)

    # (hidden, batch...) x (hidden, batch...)
    gemm1_x_scale_inv = scale_inv[gemm1_x_idx]
    wgrad_1 = fp8_dot_impl(ln_out_t, casted_dgelu_t, gemm1_x_scale_inv, dgelu_scale_inv, grad.dtype,
                           (xt_batch_dims, xt_batch_dims),
                           get_precision_of_fp8_dot(FP8Helper.FP8_2X_ACC_WGRAD))
    # Expand act axis to match the shape with the given kernel_1
    wgrad_1 = jnp.expand_dims(wgrad_1, axis=-2)

    # (batch..., hidden_out) x (hidden_in, hidden_out)
    kernel_1_scale_inv = scale_inv[gemm1_kernel_idx]
    dgrad_1 = fp8_dot_impl(casted_dgelu, casted_kernel_1, dgelu_scale_inv, kernel_1_scale_inv,
                           grad.dtype, (x_contracting_dims, (1,)),
                           get_precision_of_fp8_dot(FP8Helper.FP8_2X_ACC_DGRAD))

    dgrad_1 = with_sharding_constraint_by_logical_axes(dgrad_1, layernorm_input_axes)

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
    amax = amax.at[gemm1_grad_idx, 0].set(updated_dgelu_amax[0])
    amax = amax.at[gemm2_x_idx, 0].set(updated_gelu_amax[0])
    amax = amax.at[gemm2_kernel_idx, 0].set(updated_kernel_2_amax)
    amax = amax.at[gemm2_grad_idx, 0].set(updated_grad_amax[0])

    scale, scale_inv = FP8Helper.update_fp8_scale(fp8_max, amax, scale)

    return dx, dgamma, dbeta, wgrad_1, wgrad_2, dbias_1, dbias_2, \
           fp8_max, amax, scale, scale_inv


_layernorm_gelu_fp8_mlp.defvjp(_layernorm_gelu_fp8_mlp_fwd_rule, _layernorm_gelu_fp8_mlp_bwd_rule)
