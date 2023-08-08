# Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""JAX layernorm modules"""

from typing import Tuple, Sequence
from functools import partial, reduce
import operator
import jax
import jax.numpy as jnp

from transformer_engine_jax import DType as TEDType
from .cpp_extensions import cast_transpose, gemm, jax_dtype_to_te_dtype
from .cpp_extensions import transpose
from .cpp_extensions import rmsnorm_fwd, rmsnorm_fwd_fp8, rmsnorm_bwd
from .cpp_extensions import layernorm_fwd, layernorm_fwd_fp8, layernorm_bwd
from .fp8 import FP8Helper, FP8GemmPackage
from .sharding import ShardingType, get_elementwise_sharding_meta
from .sharding import get_dot_sharding_meta, get_fp8_meta_sharding_meta
from .sharding import is_dp_enabled, is_tp_enabled, merge_axis_resources
from .sharding import xmap_runner, extend_fsdp_sharding_meta

jax.config.update('experimental_xmap_spmd_lowering', True)
jax.config.update('experimental_xmap_spmd_lowering_manual', True)


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
              epsilon: float = 1e-6,
              sharding_type: ShardingType = ShardingType.SINGLE,
              dp_dim_index: int = 0):
    """
    Layernorm wrapper
    """
    assert sharding_type not in (ShardingType.TP_ROW, ShardingType.DP_TP_ROW), \
        "layernorm does not support row-split tensor parallelism currently."

    layernorm_type = canonicalize_layernorm_type(layernorm_type)
    if layernorm_type == 'rmsnorm':
        assert beta is None, "beta should be None if layernorm_type is 'rmsnorm'"
        assert not zero_centered_gamma, "zero_centered_gamma is not supported " \
            "if layernorm_type is 'rmsnorm'"

    if sharding_type is ShardingType.SINGLE:
        output = _layernorm(inputs,
                            gamma,
                            beta,
                            layernorm_type=layernorm_type,
                            zero_centered_gamma=zero_centered_gamma,
                            epsilon=epsilon,
                            sharding_type=sharding_type,
                            dp_axis_name="",
                            fsdp_axis_name="")
    else:
        dp_axis_name = "batch"
        tp_axis_name = "model"
        sharding_meta = get_elementwise_sharding_meta(sharding_type, inputs.shape, gamma.shape,
                                                      dp_dim_index, dp_axis_name, tp_axis_name)

        sharding_meta, fsdp_axis_name = extend_fsdp_sharding_meta(sharding_meta, {0: dp_dim_index})
        inputs_ = jnp.reshape(inputs, sharding_meta.input_shapes[0])    # 0 for input
        gamma_ = jnp.reshape(gamma, sharding_meta.input_shapes[1])    # 1 for gamma
        beta_ = beta
        beta_in_axis = {}
        if beta_ is not None:
            beta_ = jnp.reshape(beta_, sharding_meta.input_shapes[1])    # 1 for beta
            beta_in_axis = sharding_meta.in_axes[1]

        in_axes = (*sharding_meta.in_axes, beta_in_axis)

        partial_ln = partial(_layernorm,
                             layernorm_type=layernorm_type,
                             zero_centered_gamma=zero_centered_gamma,
                             epsilon=epsilon,
                             sharding_type=sharding_type,
                             dp_axis_name=dp_axis_name,
                             fsdp_axis_name=fsdp_axis_name)

        output = xmap_runner(partial_ln, in_axes, sharding_meta.out_axes,
                             sharding_meta.axis_resources, (inputs_, gamma_, beta_))

        output = jnp.reshape(output, sharding_meta.output_shapes[0])

    return output


@partial(jax.custom_vjp, nondiff_argnums=(3, 4, 5, 6, 7, 8))
def _layernorm(x, gamma, beta, layernorm_type, zero_centered_gamma, epsilon, sharding_type,
               dp_axis_name, fsdp_axis_name):
    output, _ = _layernorm_fwd(x, gamma, beta, layernorm_type, zero_centered_gamma, epsilon,
                               sharding_type, dp_axis_name, fsdp_axis_name)
    return output


def _layernorm_fwd(
        x,
        gamma,
        beta,
        layernorm_type,
        zero_centered_gamma,
        epsilon,
        sharding_type,    # pylint: disable=unused-argument
        dp_axis_name,    # pylint: disable=unused-argument
        fsdp_axis_name    # pylint: disable=unused-argument
):
    if layernorm_type == 'layernorm':
        output, mu, rsigma = layernorm_fwd(x, gamma, beta, zero_centered_gamma, epsilon)
    else:
        assert not zero_centered_gamma, "zero_centered_gamma is not supported " \
            "if layernorm_type is 'rmsnorm'"
        output, rsigma = rmsnorm_fwd(x, gamma, epsilon)
        mu = None
    return output, (mu, rsigma, x, gamma)


def _layernorm_bwd(layernorm_type, zero_centered_gamma, epsilon, sharding_type, dp_axis_name,
                   fsdp_axis_name, ctx, g):
    mu, rsigma, x, gamma = ctx

    if layernorm_type == 'layernorm':
        grad_input, grad_gamma, grad_beta = layernorm_bwd(g,
                                                          mu,
                                                          rsigma,
                                                          x,
                                                          gamma,
                                                          zero_centered_gamma=zero_centered_gamma,
                                                          epsilon=epsilon)
    else:
        assert not zero_centered_gamma, "zero_centered_gamma is not supported " \
            "if layernorm_type is 'rmsnorm'"
        grad_input, grad_gamma = rmsnorm_bwd(g, rsigma, x, gamma, epsilon=epsilon)
        grad_beta = None

    if is_dp_enabled(sharding_type.value[0]):
        grad_gamma = jax.lax.psum(grad_gamma, dp_axis_name)
        if grad_beta is not None:
            grad_beta = jax.lax.psum(grad_beta, dp_axis_name)
    if len(fsdp_axis_name) > 0:
        grad_gamma = jax.lax.psum(grad_gamma, fsdp_axis_name)
        if grad_beta is not None:
            grad_beta = jax.lax.psum(grad_beta, fsdp_axis_name)

    return grad_input, grad_gamma, grad_beta


_layernorm.defvjp(_layernorm_fwd, _layernorm_bwd)


def layernorm_fp8_dot(fp8_gemm_pkg: FP8GemmPackage,
                      gamma: jnp.ndarray,
                      beta: jnp.ndarray,
                      layernorm_type: str,
                      fwd_dtype: TEDType,
                      bwd_dtype: TEDType,
                      contracting_dims: Tuple[Sequence[int], Sequence[int]] = ((-1,), (0,)),
                      zero_centered_gamma: bool = False,
                      epsilon: float = 1e-6,
                      sharding_type: ShardingType = ShardingType.SINGLE,
                      dp_dim_index: int = 0) -> jnp.ndarray:
    """
    LN + fp8 dot fusion wrapper
    """
    assert sharding_type not in (ShardingType.TP_ROW, ShardingType.DP_TP_ROW), \
        "layernorm_fp8_dot does not support row-split tensor parallelism currently."

    layernorm_type = canonicalize_layernorm_type(layernorm_type)
    if layernorm_type == 'rmsnorm':
        assert beta is None, "beta should be None if layernorm_type is 'rmsnorm'"
        assert not zero_centered_gamma, "zero_centered_gamma is not supported " \
            "if layernorm_type is 'rmsnorm'"

    assert fp8_gemm_pkg.num_of_gemm == 1
    inputs = fp8_gemm_pkg.inputs
    kernel = fp8_gemm_pkg.kernels[0]
    fp8_max = fp8_gemm_pkg.fp8_max
    amax = fp8_gemm_pkg.amax
    scale = fp8_gemm_pkg.scale
    scale_inv = fp8_gemm_pkg.scale_inv

    if sharding_type is ShardingType.SINGLE:
        output = _layernorm_fp8_dot(inputs,
                                    kernel,
                                    gamma,
                                    beta,
                                    fp8_max,
                                    amax,
                                    scale,
                                    scale_inv,
                                    layernorm_type,
                                    fwd_dtype,
                                    bwd_dtype,
                                    contracting_dims,
                                    zero_centered_gamma=zero_centered_gamma,
                                    epsilon=epsilon,
                                    sharding_type=sharding_type,
                                    dp_axis_name="",
                                    tp_axis_name="",
                                    fsdp_axis_name="")
    else:
        dp_axis_name = "batch"
        tp_axis_name = "model"

        ln_sharding_meta = get_elementwise_sharding_meta(sharding_type, inputs.shape, gamma.shape,
                                                         dp_dim_index, dp_axis_name, tp_axis_name)
        ln_sharding_meta, _ = extend_fsdp_sharding_meta(ln_sharding_meta, {0: dp_dim_index})
        inputs_ = jnp.reshape(inputs, ln_sharding_meta.input_shapes[0])    # 0 for input
        gamma_ = jnp.reshape(gamma, ln_sharding_meta.input_shapes[1])    # 1 for gamma
        beta_ = beta
        beta_in_axis = {}
        if beta_ is not None:
            beta_ = jnp.reshape(beta_, ln_sharding_meta.input_shapes[1])    # 1 for beta
            beta_in_axis = ln_sharding_meta.in_axes[1]

        kernel_tp_index = None
        # TODO (Ming Huang): Should we add a new argument to support general sharding to kernel? # pylint: disable=fixme
        if sharding_type in (ShardingType.TP_COL, ShardingType.DP_TP_COL):
            kernel_tp_index = len(kernel.shape) - 1
        elif sharding_type in (ShardingType.TP_ROW, ShardingType.DP_TP_ROW):
            kernel_tp_index = 0

        input_tp_index = len(inputs.shape) - 1
        dot_sharding_meta = get_dot_sharding_meta(sharding_type, inputs.shape, kernel.shape,
                                                  dp_dim_index, input_tp_index, kernel_tp_index,
                                                  contracting_dims, dp_axis_name, tp_axis_name)
        dot_sharding_meta, fsdp_axis_name = extend_fsdp_sharding_meta(dot_sharding_meta,
                                                                      {0: dp_dim_index})
        kernel_ = jnp.reshape(kernel, dot_sharding_meta.input_shapes[1])    # 1 for kernel

        num_of_fp8_meta_kind = 4    # fp8_max, amax, scale, scale_inv
        fp8_sharding_meta = get_fp8_meta_sharding_meta(sharding_type, num_of_fp8_meta_kind,
                                                       dp_axis_name, tp_axis_name)

        axis_resource = merge_axis_resources([
            ln_sharding_meta.axis_resources, dot_sharding_meta.axis_resources,
            fp8_sharding_meta.axis_resources
        ])

        partial_ln_fp8_dot = partial(_layernorm_fp8_dot,
                                     layernorm_type=layernorm_type,
                                     fwd_dtype=fwd_dtype,
                                     bwd_dtype=bwd_dtype,
                                     contracting_dims=contracting_dims,
                                     zero_centered_gamma=zero_centered_gamma,
                                     epsilon=epsilon,
                                     sharding_type=sharding_type,
                                     dp_axis_name=dp_axis_name,
                                     tp_axis_name=tp_axis_name,
                                     fsdp_axis_name=fsdp_axis_name)

        # input, kernel, gamma, beta, fp8_metas
        in_axes = (ln_sharding_meta.in_axes[0], dot_sharding_meta.in_axes[1],
                   ln_sharding_meta.in_axes[1], beta_in_axis, *fp8_sharding_meta.in_axes)

        output = xmap_runner(partial_ln_fp8_dot, in_axes, dot_sharding_meta.out_axes, axis_resource,
                             (inputs_, kernel_, gamma_, beta_, fp8_max, amax, scale, scale_inv))

        output = jnp.reshape(output, dot_sharding_meta.output_shapes[0])
    return output


@partial(jax.custom_vjp, nondiff_argnums=(8, 9, 10, 11, 12, 13, 14, 15, 16, 17))
def _layernorm_fp8_dot(inputs: jnp.ndarray, kernel: jnp.ndarray, gamma: jnp.ndarray,
                       beta: jnp.ndarray, fp8_maxs: jnp.ndarray, amax: jnp.ndarray,
                       scale: jnp.ndarray, scale_inv: jnp.ndarray, layernorm_type: str,
                       fwd_dtype: TEDType, bwd_dtype: TEDType,
                       contracting_dims: Tuple[Sequence[int], Sequence[int]],
                       zero_centered_gamma: bool, epsilon: float, sharding_type: ShardingType,
                       dp_axis_name: str, tp_axis_name: str, fsdp_axis_name: str) -> jnp.ndarray:
    output, _ = _layernorm_fp8_dot_fwd(inputs, kernel, gamma, beta, fp8_maxs, amax, scale,
                                       scale_inv, layernorm_type, fwd_dtype, bwd_dtype,
                                       contracting_dims, zero_centered_gamma, epsilon,
                                       sharding_type, dp_axis_name, tp_axis_name, fsdp_axis_name)
    return output


def _layernorm_fp8_dot_fwd(
        inputs,
        kernel,
        gamma,
        beta,
        fp8_maxs,
        amax,
        scale,
        scale_inv,
        layernorm_type,
        fwd_dtype,
        bwd_dtype,    # pylint: disable=unused-argument
        contracting_dims,
        zero_centered_gamma,
        epsilon,
        sharding_type,
        dp_axis_name,    # pylint: disable=unused-argument
        tp_axis_name,
        fsdp_axis_name):    # pylint: disable=unused-argument

    lhs_contracting_dims, rhs_contracting_dims = contracting_dims
    input_shape_pre = inputs.shape[:min(lhs_contracting_dims)]
    input_shape_suf = inputs.shape[min(lhs_contracting_dims):]
    kernel_shape_pre = kernel.shape[:max(rhs_contracting_dims) + 1]
    kernel_shape_suf = kernel.shape[max(rhs_contracting_dims) + 1:]
    input_contracting_size = reduce(operator.mul, input_shape_suf)
    kernel_contracting_size = reduce(operator.mul, kernel_shape_pre)
    assert input_contracting_size == kernel_contracting_size

    amax = FP8Helper.update_amax_history(amax)

    gemm_input_idx, gemm_kernel_idx, _ = FP8Helper.get_fp8_meta_indices(0)

    input_amax = amax[gemm_input_idx, 0:1]
    input_scale = scale[gemm_input_idx]
    input_scale_inv = scale_inv[gemm_input_idx]
    if layernorm_type == 'layernorm':
        ln_out, mu, rsigma, input_amax = layernorm_fwd_fp8(inputs,
                                                           gamma,
                                                           beta,
                                                           input_amax,
                                                           input_scale,
                                                           input_scale_inv,
                                                           zero_centered_gamma=zero_centered_gamma,
                                                           epsilon=epsilon)
    else:
        assert not zero_centered_gamma, "zero_centered_gamma is not supported " \
            "if layernorm_type is 'rmsnorm'"
        ln_out, rsigma, input_amax = rmsnorm_fwd_fp8(inputs,
                                                     gamma,
                                                     input_amax,
                                                     input_scale,
                                                     input_scale_inv,
                                                     epsilon=epsilon)
        mu = None

    assert inputs.shape == ln_out.shape
    ln_out_ = jnp.reshape(ln_out, (-1, input_contracting_size))
    kernel_ = jnp.reshape(kernel, (kernel_contracting_size, -1))

    kernel_amax = amax[gemm_kernel_idx, 0:1]
    kernel_scale = scale[gemm_kernel_idx]
    kernel_scale_inv = scale_inv[gemm_kernel_idx]
    kernel_cast, kernel_cast_trans, kernel_amax = cast_transpose(kernel_, kernel_amax, kernel_scale,
                                                                 kernel_scale_inv, fwd_dtype)

    output = gemm(kernel_cast_trans, kernel_scale_inv, fwd_dtype, True, ln_out_, input_scale_inv,
                  fwd_dtype, False, jax_dtype_to_te_dtype(inputs.dtype), FP8Helper.FP8_2X_ACC_FPROP)

    if sharding_type in (ShardingType.TP_ROW, ShardingType.DP_TP_ROW):
        output = jax.lax.psum(output, tp_axis_name)

    # (input_shape_pre, input_shape_suf)
    # x (kernel_shape_pre, kernel_shape_suf)
    # = (input_shape_pre, kernel_shape_suf)
    output_shape = input_shape_pre + kernel_shape_suf
    output = jnp.reshape(output, output_shape)

    ctx = (ln_out_, kernel_cast, fp8_maxs, amax, scale, scale_inv, input_amax, kernel_amax,
           inputs.shape, kernel.shape, mu, rsigma, inputs, gamma)
    return output, ctx


def _layernorm_fp8_dot_bwd(
        layernorm_type,
        fwd_dtype,
        bwd_dtype,
        contracting_dims,    # pylint: disable=unused-argument
        zero_centered_gamma,
        epsilon,
        sharding_type,
        dp_axis_name,
        tp_axis_name,
        fsdp_axis_name,
        ctx,
        g):
    ln_out_, kernel_cast, \
    fp8_maxs, amax, scale, scale_inv, \
    input_amax, kernel_amax, \
    inputs_shape, kernel_shape, \
    mu, rsigma, inputs, gamma = ctx

    gemm_input_idx, gemm_kernel_idx, gemm_grad_idx = \
        FP8Helper.get_fp8_meta_indices(0)

    grad_amax = amax[gemm_grad_idx, 0:1]
    grad_scale = scale[gemm_grad_idx]
    grad_scale_inv = scale_inv[gemm_grad_idx]

    ln_out_trans = transpose(ln_out_, fwd_dtype)
    g = jnp.reshape(g, (ln_out_trans.shape[1], -1))

    # cast and transpose the grad_output
    grad_cast, grad_cast_trans, grad_amax = cast_transpose(g, grad_amax, grad_scale, grad_scale_inv,
                                                           bwd_dtype)

    input_scale_inv = scale_inv[gemm_input_idx]
    wgrad = gemm(grad_cast_trans, grad_scale_inv, bwd_dtype, True, ln_out_trans, input_scale_inv,
                 fwd_dtype, False, jax_dtype_to_te_dtype(g.dtype), FP8Helper.FP8_2X_ACC_WGRAD)

    kernel_scale_inv = scale_inv[gemm_kernel_idx]
    dgrad = gemm(kernel_cast, kernel_scale_inv, fwd_dtype, True, grad_cast, grad_scale_inv,
                 bwd_dtype, False, jax_dtype_to_te_dtype(g.dtype), FP8Helper.FP8_2X_ACC_DGRAD)

    dgrad = jnp.reshape(dgrad, inputs_shape)

    if sharding_type in (ShardingType.TP_COL, ShardingType.DP_TP_COL):
        dgrad = jax.lax.psum(dgrad, tp_axis_name)

    if layernorm_type == 'layernorm':
        grad_input, grad_gamma, grad_beta = layernorm_bwd(dgrad,
                                                          mu,
                                                          rsigma,
                                                          inputs,
                                                          gamma,
                                                          zero_centered_gamma=zero_centered_gamma,
                                                          epsilon=epsilon)
    else:
        assert not zero_centered_gamma, "zero_centered_gamma is not supported " \
            "if layernorm_type is 'rmsnorm'"
        grad_input, grad_gamma = rmsnorm_bwd(dgrad, rsigma, inputs, gamma, epsilon=epsilon)
        grad_beta = None

    amax = amax.at[gemm_input_idx, 0].set(input_amax[0])
    amax = amax.at[gemm_kernel_idx, 0].set(kernel_amax[0])
    amax = amax.at[gemm_grad_idx, 0].set(grad_amax[0])

    if is_dp_enabled(sharding_type.value[0]):
        wgrad = jax.lax.psum(wgrad, dp_axis_name)
        grad_gamma = jax.lax.psum(grad_gamma, dp_axis_name)
        if grad_beta is not None:
            grad_beta = jax.lax.psum(grad_beta, dp_axis_name)
        amax = jax.lax.pmax(amax, dp_axis_name)

    if len(fsdp_axis_name) > 0:
        wgrad = jax.lax.psum(wgrad, fsdp_axis_name)
        grad_gamma = jax.lax.psum(grad_gamma, fsdp_axis_name)
        if grad_beta is not None:
            grad_beta = jax.lax.psum(grad_beta, fsdp_axis_name)
        amax = jax.lax.pmax(amax, fsdp_axis_name)

    if is_tp_enabled(sharding_type.value[0]):
        amax = jax.lax.pmax(amax, tp_axis_name)

    wgrad = jnp.reshape(wgrad, kernel_shape)
    return grad_input, wgrad, \
           grad_gamma, grad_beta, \
           fp8_maxs, amax, scale, scale_inv


_layernorm_fp8_dot.defvjp(_layernorm_fp8_dot_fwd, _layernorm_fp8_dot_bwd)
