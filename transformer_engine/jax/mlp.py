# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""JAX MLP modules"""

from typing import List, Tuple, Sequence, Union, Callable
from functools import partial

import jax
import jax.numpy as jnp
from jax.ad_checkpoint import checkpoint_name

from .cpp_extensions import cast_fp8, transpose, cast_transpose, dbias_cast_transpose
from .cpp_extensions import gelu
from .cpp_extensions import gelu_fp8, dgelu, dgelu_dbias_cast_transpose
from .cpp_extensions import gated_gelu, gated_gelu_fp8
from .cpp_extensions import dgated_gelu, dgated_gelu_cast_transpose
from .cpp_extensions import rmsnorm_fwd_fp8, rmsnorm_bwd
from .cpp_extensions import layernorm_fwd_fp8, layernorm_bwd
from .dot import fp8_dot_impl, get_precision_of_fp8_dot, quantize
from .layernorm import canonicalize_layernorm_type
from .fp8 import FP8Helper, FP8MetaPackage
from .sharding import with_sharding_constraint_by_logical_axes


activation_dict = {
    ('gelu',): {'fwd': gelu,
                "bwd": dgelu},
    ('gelu', 'linear'): {'fwd': gated_gelu,
                         'bwd': dgated_gelu}
}

activation_fp8_dict = {
    ('gelu',): {'fwd': gelu_fp8,
                'bwd': dgelu_dbias_cast_transpose},
    ('gelu', 'linear'): {'fwd': gated_gelu_fp8,
                         'bwd': dgated_gelu_cast_transpose}
}


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
    fwd_output = activation_dict[activation_type]["fwd"](x)
    return fwd_output, (x,)


def _activation_lu_bwd_rule(activation_type, ctx, g):
    x, = ctx
    assert x.dtype == g.dtype

    dx = activation_dict[activation_type]["bwd"](g, x)
    dx = jnp.reshape(dx, x.shape)
    return (dx,)

_activation_lu.defvjp(_activation_lu_fwd_rule, _activation_lu_bwd_rule)


def fused_layernorm_fp8_mlp(x: jnp.ndarray,
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
                           ffn2_ckpt_name: str = 'ffn2',
                           activation_type: Sequence[Union[str, Callable]] = ('gelu',),
                           use_bias: bool = True) -> jnp.ndarray:
    """
    Layernorm + GEMM1 + bias + activation + GEMM2 + bias
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

    output = _fused_layernorm_fp8_mlp(x, gamma, beta, kernel_1, kernel_2, bias_1, bias_2, fp8_max,
                                     amax, scale, scale_inv, fwd_dtype, bwd_dtype, layernorm_type,
                                     zero_centered_gamma, epsilon, layernorm_input_axes,
                                     dot_1_input_axes, dot_2_input_axes, ffn1_ckpt_name,
                                     ffn2_ckpt_name, activation_type, use_bias)
    return output


@partial(jax.custom_vjp, nondiff_argnums=(11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22))
def _fused_layernorm_fp8_mlp(x: jnp.ndarray, gamma: jnp.ndarray, beta: jnp.ndarray,
                            kernel_1: jnp.ndarray, kernel_2: jnp.ndarray, bias_1: jnp.ndarray,
                            bias_2: jnp.ndarray, fp8_max: jnp.ndarray, amax: jnp.ndarray,
                            scale: jnp.ndarray, scale_inv: jnp.ndarray, fwd_dtype: jnp.dtype,
                            bwd_dtype: jnp.dtype, layernorm_type: str, zero_centered_gamma: bool,
                            epsilon: float, layernorm_input_axes: Tuple[str, ...],
                            dot_1_input_axes: Tuple[str, ...], dot_2_input_axes: Tuple[str, ...],
                            ffn1_ckpt_name: str, ffn2_ckpt_name: str,
                            activation_type: Sequence[Union[str, Callable]],
                            use_bias: bool):
    output, _ = _fused_layernorm_fp8_mlp_fwd_rule(x, gamma, beta, kernel_1, kernel_2, bias_1,
                                                  bias_2, fp8_max, amax, scale, scale_inv,
                                                  fwd_dtype, bwd_dtype, layernorm_type,
                                                  zero_centered_gamma, epsilon,
                                                  layernorm_input_axes, dot_1_input_axes,
                                                  dot_2_input_axes, ffn1_ckpt_name, ffn2_ckpt_name,
                                                  activation_type, use_bias)
    return output


def _fused_layernorm_fp8_mlp_fwd_rule(
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
        ffn2_ckpt_name,
        activation_type,
        use_bias):

    is_gated = len(activation_type) > 1
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

    # Squeeze act axis
    # (hidden_in, 1, hidden_out) -> (hidden_in, hidden_out)
    if not is_gated:
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
    if use_bias:
        bias_1_shape = (1,) * (dot_1_output.ndim - bias_1.ndim) + bias_1.shape
        dot_1_output += jnp.reshape(bias_1, bias_1_shape)
    dot_1_output = checkpoint_name(dot_1_output, ffn1_ckpt_name)

    gemm2_x_idx, gemm2_kernel_idx, _ = FP8Helper.get_fp8_meta_indices(1)

    activation_lu_out_amax = amax[gemm2_x_idx, 0:1]
    activation_lu_out_scale = scale[gemm2_x_idx]
    activation_lu_out_scale_inv = scale_inv[gemm2_x_idx]

    activation_lu_fp8 = activation_fp8_dict[activation_type]["fwd"]

    # (batch..., hidden_in) -> (batch..., hidden)
    casted_activation_lu_out, updated_activation_lu_amax = activation_lu_fp8(dot_1_output,
                                                    activation_lu_out_amax, activation_lu_out_scale,
                                                    activation_lu_out_scale_inv, fwd_dtype)

    casted_activation_lu_out = with_sharding_constraint_by_logical_axes(casted_activation_lu_out,
                                                                        dot_2_input_axes)

    kernel_2_scale = scale[gemm2_kernel_idx]
    kernel_2_scale_inv = scale_inv[gemm2_kernel_idx]
    # Note (Ming Huang): Use native cast to allow XLA handle tranpose for avoiding
    # unnecessary copy to break FP8 GEMM pattern matching.
    casted_kernel_2, updated_kernel_2_amax = quantize(kernel_2, fwd_dtype, kernel_2_scale)

    # (batch..., hidden_in) x (hidden_out, hidden_in)
    dot_2_output = fp8_dot_impl(casted_activation_lu_out, casted_kernel_2,
                                activation_lu_out_scale_inv,
                                kernel_2_scale_inv, x.dtype, (x_contracting_dims, (0,)),
                                get_precision_of_fp8_dot(FP8Helper.FP8_2X_ACC_FPROP))

    if use_bias:
        bias_2_shape = (1,) * (dot_2_output.ndim - bias_2.ndim) + bias_2.shape
        dot_2_output += jnp.reshape(bias_2, bias_2_shape)

    dot_2_output = checkpoint_name(dot_2_output, ffn2_ckpt_name)

    ctx = (x, ln_out, mu, rsigma, gamma, dot_1_output, casted_activation_lu_out, casted_kernel_1,
           casted_kernel_2, fp8_max, amax, scale, scale_inv, updated_x_amax,
           updated_activation_lu_amax, updated_kernel_1_amax, updated_kernel_2_amax,
           x_contracting_dims, xt_batch_dims, bias_1.shape, bias_2.shape)

    return dot_2_output, ctx


def _fused_layernorm_fp8_mlp_bwd_rule(
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
        activation_type,
        use_bias,
        ctx,
        grad):
    x, ln_out, mu, rsigma, gamma, dot_1_output, casted_activation_lu_out, \
    casted_kernel_1, casted_kernel_2, fp8_max, amax, scale, scale_inv, updated_x_amax, \
    updated_activation_lu_amax, updated_kernel_1_amax, updated_kernel_2_amax, \
    x_contracting_dims, xt_batch_dims, bias_1_shape, bias_2_shape= ctx

    is_gated = len(activation_type) > 1

    gemm2_x_idx, gemm2_kernel_idx, gemm2_grad_idx = FP8Helper.get_fp8_meta_indices(1)

    grad_amax = amax[gemm2_grad_idx, 0:1]
    grad_scale = scale[gemm2_grad_idx]
    grad_scale_inv = scale_inv[gemm2_grad_idx]

    # Since the sharding of outputs should be the same as dot_1's input
    grad = with_sharding_constraint_by_logical_axes(grad, dot_1_input_axes)

    casted_grad, casted_grad_t, updated_grad_amax = \
        cast_transpose(grad, grad_amax, grad_scale, grad_scale_inv, bwd_dtype,
                       static_axis_boundary=-1, transpose_axis_boundary=-1)

    casted_activation_lu_out_t = transpose(casted_activation_lu_out,
                                           static_axis_boundary=-1,
                                           transpose_axis_boundary=-1)
    if use_bias:
        dbias_2 = jnp.sum(grad, axis=(i for i in range(grad.ndim - 1)))
        dbias_2 = jnp.reshape(dbias_2, bias_2_shape)
    else:
        dbias_2 = jnp.zeros(bias_2_shape, grad.dtype)

    # (hidden, batch...,) x (hidden, batch...)
    gemm2_x_scale_inv = scale_inv[gemm2_x_idx]
    wgrad_2 = fp8_dot_impl(casted_activation_lu_out_t, casted_grad_t, gemm2_x_scale_inv,
                           grad_scale_inv, grad.dtype, (xt_batch_dims, xt_batch_dims),
                           get_precision_of_fp8_dot(FP8Helper.FP8_2X_ACC_WGRAD))

    # (batch..., hidden_out) x (hidden_in, hidden_out)
    kernel_2_scale_inv = scale_inv[gemm2_kernel_idx]
    dgrad_2 = fp8_dot_impl(casted_grad, casted_kernel_2, grad_scale_inv, kernel_2_scale_inv,
                           grad.dtype, (x_contracting_dims, (1,)),
                           get_precision_of_fp8_dot(FP8Helper.FP8_2X_ACC_DGRAD))

    dgrad_2 = with_sharding_constraint_by_logical_axes(dgrad_2, dot_2_input_axes)

    gemm1_x_idx, gemm1_kernel_idx, gemm1_grad_idx = FP8Helper.get_fp8_meta_indices(0)

    dactivation_lu_amax = amax[gemm1_grad_idx, 0:1]
    dactivation_lu_scale = scale[gemm1_grad_idx]
    dactivation_lu_scale_inv = scale_inv[gemm1_grad_idx]

    dactivation_lu_cast_transpose = activation_fp8_dict[activation_type]["bwd"]
    if not is_gated and use_bias:
        casted_dactivation_lu, casted_dactivation_lu_t, dbias_1, updated_dactivation_lu_amax = \
        dactivation_lu_cast_transpose(
            dgrad_2,
            dot_1_output,
            dactivation_lu_amax,
            dactivation_lu_scale,
            dactivation_lu_scale_inv,
            bwd_dtype,
            static_axis_boundary=-1,
            transpose_axis_boundary=-1)

    elif is_gated and not use_bias:
        casted_dactivation_lu, casted_dactivation_lu_t, updated_dactivation_lu_amax = \
        dactivation_lu_cast_transpose(
            dgrad_2,
            dot_1_output,
            dactivation_lu_amax,
            dactivation_lu_scale,
            dactivation_lu_scale_inv,
            bwd_dtype,
            static_axis_boundary=-1)
        dbias_1 = jnp.zeros(bias_1_shape, grad.dtype)
        print(casted_dactivation_lu_t.shape)
    else:  # d<activation> + fused cast transpose
        dactivation_lu = activation_dict[activation_type]["bwd"](dgrad_2, dot_1_output)
        dactivation_lu_shape = dactivation_lu.shape
        if is_gated:
            dactivation_lu = jnp.reshape(dactivation_lu, (dactivation_lu.shape[0], -1))
        if use_bias:
            casted_dactivation_lu, casted_dactivation_lu_t, dbias_1, updated_dactivation_lu_amax = \
            dbias_cast_transpose(
                dactivation_lu,
                dactivation_lu_amax,
                dactivation_lu_scale,
                dactivation_lu_scale_inv,
                bwd_dtype,
                static_axis_boundary=-1,
                transpose_axis_boundary=-1)
        else:
            casted_dactivation_lu, casted_dactivation_lu_t, updated_dactivation_lu_amax = \
            cast_transpose(
                dactivation_lu,
                dactivation_lu_amax,
                dactivation_lu_scale,
                dactivation_lu_scale_inv,
                bwd_dtype,
                static_axis_boundary=-1,
                transpose_axis_boundary=-1)
            dbias_1 = jnp.zeros(bias_1_shape, bwd_dtype)
        if is_gated:
            casted_dactivation_lu = jnp.reshape(casted_dactivation_lu, dactivation_lu_shape)
            # TODO tmr
            casted_dactivation_lu_t = jnp.split(casted_dactivation_lu_t, casted_dactivation_lu_t.shape[0]//2, axis=0)

    dbias_1 = jnp.reshape(dbias_1, bias_1_shape)

    ln_out_t = transpose(ln_out, static_axis_boundary=-1, transpose_axis_boundary=-1)

    # (hidden, batch...) x (hidden, batch...)
    gemm1_x_scale_inv = scale_inv[gemm1_x_idx]
    # Check if not gated
    xt_batch_dims_2 = xt_batch_dims if not is_gated \
        else tuple(i + 1 for i in xt_batch_dims)
    wgrad_1 = fp8_dot_impl(ln_out_t, casted_dactivation_lu_t, gemm1_x_scale_inv,
                           dactivation_lu_scale_inv, grad.dtype,
                           (xt_batch_dims, xt_batch_dims_2),
                           get_precision_of_fp8_dot(FP8Helper.FP8_2X_ACC_WGRAD))
    # Expand act axis to match the shape with the given kernel_1
    if not is_gated:
        wgrad_1 = jnp.expand_dims(wgrad_1, axis=-2)

    # (batch..., hidden_out) x (hidden_in, hidden_out)
    if is_gated:
        x_contracting_dims = ((min(x_contracting_dims),) + tuple(
            i + 1 for i in x_contracting_dims), (1,2))
    else:
        x_contracting_dims = (x_contracting_dims, (1,))
    kernel_1_scale_inv = scale_inv[gemm1_kernel_idx]
    dgrad_1 = fp8_dot_impl(casted_dactivation_lu, casted_kernel_1,
                           dactivation_lu_scale_inv, kernel_1_scale_inv,
                           grad.dtype, x_contracting_dims,
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
    amax = amax.at[gemm1_grad_idx, 0].set(updated_dactivation_lu_amax[0])
    amax = amax.at[gemm2_x_idx, 0].set(updated_activation_lu_amax[0])
    amax = amax.at[gemm2_kernel_idx, 0].set(updated_kernel_2_amax)
    amax = amax.at[gemm2_grad_idx, 0].set(updated_grad_amax[0])

    scale, scale_inv = FP8Helper.update_fp8_scale(fp8_max, amax, scale)

    return dx, dgamma, dbeta, wgrad_1, wgrad_2, dbias_1, dbias_2, \
           fp8_max, amax, scale, scale_inv


_fused_layernorm_fp8_mlp.defvjp(_fused_layernorm_fp8_mlp_fwd_rule,
                                        _fused_layernorm_fp8_mlp_bwd_rule)
