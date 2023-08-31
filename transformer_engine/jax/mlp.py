# Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""JAX MLP modules"""

from typing import Tuple, Sequence, Union, Callable
from functools import partial, reduce
import operator

import jax
import jax.numpy as jnp
from jax.interpreters import pxla

from transformer_engine_jax import DType as TEDType
from .cpp_extensions import jax_dtype_to_te_dtype
from .cpp_extensions import transpose, cast_transpose
from .cpp_extensions import gated_gelu, gated_gelu_fp8
from .cpp_extensions import dgated_gelu, dgated_gelu_cast_transpose
from .cpp_extensions import rmsnorm_fwd_fp8, rmsnorm_bwd
from .cpp_extensions import layernorm_fwd_fp8, layernorm_bwd
from .cpp_extensions import gemm
from .sharding import MajorShardingType, ShardingType
from .sharding import get_elementwise_sharding_meta
from .sharding import get_dot_sharding_meta, get_fp8_meta_sharding_meta
from .sharding import merge_axis_resources, infer_sharding_type
from .sharding import xmap_runner, extend_fsdp_sharding_meta
from .layernorm import canonicalize_layernorm_type
from .fp8 import FP8Helper, FP8GemmPackage

jax.config.update('experimental_xmap_spmd_lowering', True)
jax.config.update('experimental_xmap_spmd_lowering_manual', True)

thread_resources = pxla.thread_resources


def geglu(
        inputs: jnp.ndarray,
        contracting_dims: Sequence[int] = (-1,),
        sharding_type: ShardingType = ShardingType.SINGLE,
        dp_dim_index: int = 0,    # pylint: disable=unused-argument
):
    """
    Gated gelu
    """
    input_shape_suf_size = reduce(operator.mul, inputs.shape[min(contracting_dims):])
    assert input_shape_suf_size % 2 == 0
    output_shape = (*inputs.shape[:min(contracting_dims)], input_shape_suf_size // 2)

    if sharding_type is ShardingType.SINGLE:
        output = _geglu(inputs, contracting_dims)
    else:
        dp_axis_name = "batch"
        tp_axis_name = "model"

        sharding_meta = get_elementwise_sharding_meta(sharding_type, inputs.shape, None,
                                                      dp_dim_index, dp_axis_name, tp_axis_name)
        sharding_meta, _ = extend_fsdp_sharding_meta(sharding_meta, {0: dp_dim_index})

        inputs_ = jnp.reshape(inputs, sharding_meta.input_shapes[0])    # 0 for input

        partial_geglu = partial(_geglu, contracting_dims=contracting_dims)

        output = xmap_runner(partial_geglu, sharding_meta.in_axes, sharding_meta.out_axes,
                             sharding_meta.axis_resources, (inputs_,))

    output = jnp.reshape(output, output_shape)
    return output


@partial(jax.custom_vjp, nondiff_argnums=(1,))
def _geglu(inputs: jnp.ndarray, contracting_dims: Sequence[int] = (-1,)):

    geglu_output, _ = _geglu_fwd(inputs, contracting_dims)

    return geglu_output


def _geglu_fwd(inputs, contracting_dims):
    inputs_real_shape = (*inputs.shape[:min(contracting_dims)],
                         reduce(operator.mul, inputs.shape[min(contracting_dims):]))
    inputs_ = jnp.reshape(inputs, inputs_real_shape)
    geglu_output = gated_gelu(inputs_)
    geglu_output = jnp.expand_dims(geglu_output, min(contracting_dims))
    return geglu_output, (inputs_, inputs.shape)


def _geglu_bwd(contracting_dims, ctx, g):
    inputs_, inputs_shape = ctx
    g = jnp.squeeze(g, min(contracting_dims))
    assert inputs_.dtype == g.dtype

    dgelu = dgated_gelu(g, inputs_)
    dgelu = jnp.reshape(dgelu, inputs_shape)
    return (dgelu,)


_geglu.defvjp(_geglu_fwd, _geglu_bwd)


def fp8_ln_mlp(
    fp8_gemm_pkg: FP8GemmPackage,
    ln_scale: jnp.ndarray,
    ln_bias: jnp.ndarray,
    layernorm_type: str,
    fwd_dtype: TEDType,
    bwd_dtype: TEDType,
    zero_centered_gamma: bool = False,
    epsilon: float = 1e-6,
    contracting_dims: Tuple[Sequence[int], Sequence[int]] = ((-1,), (0,)),
    major_sharding_type: MajorShardingType = MajorShardingType.SINGLE,
    dp_dim_index: int = 0,    # pylint: disable=unused-argument
    activations: Sequence[Union[str, Callable]] = ('gelu', 'linear')
) -> jnp.ndarray:
    """
    FP8 layernorm MLP wrapper
    (LN + Dense + act + Dense)
    """
    assert fp8_gemm_pkg.num_of_gemm == 2
    inputs = fp8_gemm_pkg.inputs
    kernel_1 = fp8_gemm_pkg.kernels[0]
    kernel_2 = fp8_gemm_pkg.kernels[1]
    fp8_max = fp8_gemm_pkg.fp8_max
    amax = fp8_gemm_pkg.amax
    scale = fp8_gemm_pkg.scale
    scale_inv = fp8_gemm_pkg.scale_inv

    layernorm_type = canonicalize_layernorm_type(layernorm_type)
    if layernorm_type == 'rmsnorm':
        assert ln_bias is None, "ln_bias should be None if layernorm_type is 'rmsnorm'"
        assert not zero_centered_gamma, "zero_centered_gamma is not supported " \
            "if layernorm_type is 'rmsnorm'"

    assert activations == ('gelu', 'linear')
    if major_sharding_type is MajorShardingType.SINGLE:
        res = _fp8_mlp(inputs, ln_scale, ln_bias, kernel_1, kernel_2, fp8_max, amax, scale,
                       scale_inv, layernorm_type, activations, zero_centered_gamma, epsilon,
                       fwd_dtype, bwd_dtype, contracting_dims, major_sharding_type, "", "", "")
    else:
        dp_axis_name = "batch"
        tp_axis_name = "model"

        first_part_st, second_part_st = infer_sharding_type(major_sharding_type)

        ln_sharding_meta = get_elementwise_sharding_meta(first_part_st, inputs.shape,
                                                         ln_scale.shape, dp_dim_index, dp_axis_name,
                                                         tp_axis_name)
        ln_sharding_meta, _ = extend_fsdp_sharding_meta(ln_sharding_meta, {0: dp_dim_index})

        input_tp_index = len(inputs.shape) - 1
        first_dot_sharding_meta = get_dot_sharding_meta(first_part_st, inputs.shape, kernel_1.shape,
                                                        dp_dim_index, input_tp_index, 2,
                                                        contracting_dims, dp_axis_name,
                                                        tp_axis_name)
        first_dot_sharding_meta, fsdp_axis_name = extend_fsdp_sharding_meta(
            first_dot_sharding_meta, {0: dp_dim_index})
        second_input_shape = (*first_dot_sharding_meta.output_shapes[0][:-2],
                              first_dot_sharding_meta.output_shapes[0][-1])
        second_dot_sharding_meta = get_dot_sharding_meta(second_part_st, second_input_shape,
                                                         kernel_2.shape, dp_dim_index,
                                                         len(second_input_shape) - 1, 0,
                                                         contracting_dims, dp_axis_name,
                                                         tp_axis_name)
        second_dot_sharding_meta, _ = extend_fsdp_sharding_meta(second_dot_sharding_meta,
                                                                {0: dp_dim_index})

        num_of_fp8_meta_kind = 4    # fp8_max, amax, scale, scale_inv
        fp8_sharding_meta = get_fp8_meta_sharding_meta(first_part_st, num_of_fp8_meta_kind,
                                                       dp_axis_name, tp_axis_name)

        inputs_ = jnp.reshape(inputs, ln_sharding_meta.input_shapes[0])    # 0 for input
        ln_scale_ = jnp.reshape(ln_scale, ln_sharding_meta.input_shapes[1])    # 1 for gamma
        ln_bias_ = ln_bias
        ln_bias_in_axis = {}
        if ln_bias_ is not None:
            ln_bias_ = jnp.reshape(ln_bias_, ln_sharding_meta.input_shapes[1])    # 1 for beta
            ln_bias_in_axis = ln_sharding_meta.in_axes[1]
        kernel_1_ = jnp.reshape(kernel_1, first_dot_sharding_meta.input_shapes[1])    # 1 for kernel
        kernel_2_ = jnp.reshape(kernel_2,
                                second_dot_sharding_meta.input_shapes[1])    # 1 for kernel

        axis_resource = merge_axis_resources([
            ln_sharding_meta.axis_resources, first_dot_sharding_meta.axis_resources,
            second_dot_sharding_meta.axis_resources, fp8_sharding_meta.axis_resources
        ])

        partial_fp8_mlp = partial(_fp8_mlp,
                                  layernorm_type=layernorm_type,
                                  activations=activations,
                                  zero_centered_gamma=zero_centered_gamma,
                                  epsilon=epsilon,
                                  fwd_dtype=fwd_dtype,
                                  bwd_dtype=bwd_dtype,
                                  contracting_dims=contracting_dims,
                                  major_sharding_type=major_sharding_type,
                                  dp_axis_name=dp_axis_name,
                                  tp_axis_name=tp_axis_name,
                                  fsdp_axis_name=fsdp_axis_name)
        in_axes = (ln_sharding_meta.in_axes[0], ln_sharding_meta.in_axes[1], ln_bias_in_axis,
                   first_dot_sharding_meta.in_axes[1], second_dot_sharding_meta.in_axes[1],
                   *fp8_sharding_meta.in_axes)

        res = xmap_runner(
            partial_fp8_mlp, in_axes, second_dot_sharding_meta.out_axes, axis_resource,
            (inputs_, ln_scale_, ln_bias_, kernel_1_, kernel_2_, fp8_max, amax, scale, scale_inv))
        res = jnp.reshape(res, second_dot_sharding_meta.output_shapes[0])

    return res


@partial(jax.custom_vjp, nondiff_argnums=(9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19))
def _fp8_mlp(inputs: jnp.ndarray, ln_scale: jnp.ndarray, ln_bias: jnp.ndarray,
             kernel_1: jnp.ndarray, kernel_2: jnp.ndarray, fp8_maxs: jnp.ndarray, amax: jnp.ndarray,
             scale: jnp.ndarray, scale_inv: jnp.ndarray, layernorm_type: str,
             activations: Sequence[Union[str, Callable]], zero_centered_gamma: bool, epsilon: float,
             fwd_dtype: TEDType, bwd_dtype: TEDType, contracting_dims: Tuple[Sequence[int],
                                                                             Sequence[int]],
             major_sharding_type: MajorShardingType, dp_axis_name: str, tp_axis_name: str,
             fsdp_axis_name: str):
    res, _ = _fp8_mlp_fwd(inputs,
                          ln_scale,
                          ln_bias,
                          kernel_1,
                          kernel_2,
                          fp8_maxs,
                          amax,
                          scale,
                          scale_inv,
                          layernorm_type,
                          activations,
                          zero_centered_gamma,
                          epsilon,
                          fwd_dtype,
                          bwd_dtype,
                          contracting_dims=contracting_dims,
                          major_sharding_type=major_sharding_type,
                          dp_axis_name=dp_axis_name,
                          tp_axis_name=tp_axis_name,
                          fsdp_axis_name=fsdp_axis_name)
    return res


def _fp8_mlp_fwd(
        inputs,
        gamma,
        beta,
        kernel_1,
        kernel_2,
        fp8_maxs,
        amax,
        scale,
        scale_inv,
        layernorm_type,
        activations,
        zero_centered_gamma,
        epsilon,
        fwd_dtype,
        bwd_dtype,    # pylint: disable=unused-argument
        contracting_dims,
        major_sharding_type,
        dp_axis_name,    # pylint: disable=unused-argument
        tp_axis_name,
        fsdp_axis_name):    # pylint: disable=unused-argument
    if activations != ('gelu', 'linear'):
        raise NotImplementedError("activations only support ('gelu', 'linear') for now.")
    lhs_contracting_dims, rhs_contracting_dims = contracting_dims
    input_shape_pre = inputs.shape[:min(lhs_contracting_dims)]
    input_shape_suf = inputs.shape[min(lhs_contracting_dims):]
    kernel_1_shape_pre = kernel_1.shape[:max(rhs_contracting_dims) + 1]
    kernel_1_shape_suf = kernel_1.shape[max(rhs_contracting_dims) + 1:]
    kernel_2_shape_pre = kernel_2.shape[:max(rhs_contracting_dims) + 1]
    kernel_2_shape_suf = kernel_2.shape[max(rhs_contracting_dims) + 1:]
    input_contracting_size = reduce(operator.mul, input_shape_suf)
    kernel_1_pre_size = reduce(operator.mul, kernel_1_shape_pre)
    kernel_1_suf_size = reduce(operator.mul, kernel_1_shape_suf)
    kernel_2_pre_size = reduce(operator.mul, kernel_2_shape_pre)
    assert input_contracting_size == kernel_1_pre_size
    assert kernel_1_suf_size == kernel_2_pre_size * len(activations)
    inputs_ = jnp.reshape(inputs, (-1, input_contracting_size))
    kernel_1_ = jnp.reshape(kernel_1, (kernel_1_pre_size, -1))
    kernel_2_ = jnp.reshape(kernel_2, (kernel_2_pre_size, -1))

    amax = FP8Helper.update_amax_history(amax)

    gemm1_input_idx, gemm1_kernel_idx, _ = FP8Helper.get_fp8_meta_indices(0)

    input_amax = amax[gemm1_input_idx, 0:1]
    input_scale = scale[gemm1_input_idx]
    input_scale_inv = scale_inv[gemm1_input_idx]
    if layernorm_type == 'layernorm':
        ln_out, mu, rsigma, ln_out_amax = layernorm_fwd_fp8(inputs_,
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
        ln_out, rsigma, ln_out_amax = rmsnorm_fwd_fp8(inputs_,
                                                      gamma,
                                                      input_amax,
                                                      input_scale,
                                                      input_scale_inv,
                                                      epsilon=epsilon)
        mu = None

    kernel_1_amax = amax[gemm1_kernel_idx, 0:1]
    kernel_1_scale = scale[gemm1_kernel_idx]
    kernel_1_scale_inv = scale_inv[gemm1_kernel_idx]
    kernel_1_cast, kernel_1_cast_trans, kernel_1_amax = cast_transpose(
        kernel_1_, kernel_1_amax, kernel_1_scale, kernel_1_scale_inv, fwd_dtype)
    dense_1_output = gemm(kernel_1_cast_trans, kernel_1_scale_inv, fwd_dtype, True, ln_out,
                          scale_inv[gemm1_input_idx], fwd_dtype, False,
                          jax_dtype_to_te_dtype(inputs.dtype), FP8Helper.FP8_2X_ACC_FPROP)

    gemm2_input_idx, gemm2_kernel_idx, _ = FP8Helper.get_fp8_meta_indices(1)

    kernel_2_amax = amax[gemm2_kernel_idx, 0:1]
    kernel_2_scale = scale[gemm2_kernel_idx]
    kernel_2_scale_inv = scale_inv[gemm2_kernel_idx]
    kernel_2_cast, kernel_2_cast_trans, kernel_2_amax = cast_transpose(
        kernel_2_, kernel_2_amax, kernel_2_scale, kernel_2_scale_inv, fwd_dtype)

    dense_1_out_amax = amax[gemm2_input_idx, 0:1]
    dense_1_out_scale = scale[gemm2_input_idx]
    dense_1_out_scale_inv = scale_inv[gemm2_input_idx]
    gated_gelu_output_cast, gated_gelu_amax = gated_gelu_fp8(dense_1_output, dense_1_out_amax,
                                                             dense_1_out_scale,
                                                             dense_1_out_scale_inv, fwd_dtype)
    res = gemm(kernel_2_cast_trans, kernel_2_scale_inv, fwd_dtype, True,
               gated_gelu_output_cast, dense_1_out_scale_inv, fwd_dtype, False,
               jax_dtype_to_te_dtype(inputs.dtype), FP8Helper.FP8_2X_ACC_FPROP)

    if major_sharding_type in (MajorShardingType.TP, MajorShardingType.DPTP):
        res = jax.lax.psum(res, tp_axis_name)

    # (input_shape_pre, input_shape_suf)
    # x (kernel_1_shape_pre, kernel_1_shape_suf)
    # x (kernel_2_shape_pre, kernel_2_shape_suf)
    # = (input_shape_pre, kernel_2_shape_suf)
    output_shape = input_shape_pre + kernel_2_shape_suf
    res = jnp.reshape(res, output_shape)

    ctx = (inputs_, ln_out, mu, rsigma, gamma, dense_1_output, gated_gelu_output_cast,
           kernel_1_cast, kernel_2_cast, fp8_maxs, amax, scale, scale_inv, ln_out_amax,
           gated_gelu_amax, kernel_1_amax, kernel_2_amax, inputs.shape, kernel_1.shape,
           kernel_2.shape)

    return res, ctx


def _fp8_mlp_bwd(
        layernorm_type,
        activations,    # pylint: disable=unused-argument
        zero_centered_gamma,
        epsilon,
        fwd_dtype,
        bwd_dtype,
        contracting_dims,    # pylint: disable=unused-argument
        major_sharding_type,
        dp_axis_name,
        tp_axis_name,
        fsdp_axis_name,
        ctx,
        g):
    inputs_, ln_out, mu, rsigma, gamma, \
    dense_1_output, gated_gelu_output_cast, \
    kernel_1_cast, kernel_2_cast, \
    fp8_maxs, amax, scale, scale_inv, \
    ln_out_amax, gated_gelu_amax, kernel_1_amax, kernel_2_amax, \
    input_shape, kernel_1_shape, kernel_2_shape = ctx

    g = jnp.reshape(g, (ln_out.shape[0], -1))

    gemm2_input_idx, gemm2_kernel_idx, gemm2_grad_idx = FP8Helper.get_fp8_meta_indices(1)

    grad_amax = amax[gemm2_grad_idx, 0:1]
    grad_scale = scale[gemm2_grad_idx]
    grad_scale_inv = scale_inv[gemm2_grad_idx]

    grad_cast, grad_cast_trans, grad_amax = cast_transpose(g, grad_amax, grad_scale, grad_scale_inv,
                                                           bwd_dtype)
    gated_gelu_output_cast_trans = transpose(gated_gelu_output_cast, fwd_dtype)

    gemm2_input_scale_inv = scale_inv[gemm2_input_idx]
    wgrad_2 = gemm(grad_cast_trans, grad_scale_inv, bwd_dtype, True,
                   gated_gelu_output_cast_trans, gemm2_input_scale_inv, fwd_dtype, False,
                   jax_dtype_to_te_dtype(g.dtype), FP8Helper.FP8_2X_ACC_WGRAD)
    kernel_2_scale_inv = scale_inv[gemm2_kernel_idx]
    dgrad_2 = gemm(kernel_2_cast, kernel_2_scale_inv, fwd_dtype, True, grad_cast, grad_scale_inv,
                   bwd_dtype, False, jax_dtype_to_te_dtype(g.dtype), FP8Helper.FP8_2X_ACC_DGRAD)

    gemm1_input_idx, gemm1_kernel_idx, gemm1_grad_idx = FP8Helper.get_fp8_meta_indices(0)

    dgrad_2_amax = amax[gemm1_grad_idx, 0:1]
    dgrad_2_scale = scale[gemm1_grad_idx]
    dgrad_2_scale_inv = scale_inv[gemm1_grad_idx]
    dgelu, dgelu_trans, dgelu_amax = dgated_gelu_cast_transpose(dgrad_2, dense_1_output,
                                                                dgrad_2_amax, dgrad_2_scale,
                                                                dgrad_2_scale_inv, bwd_dtype)
    ln_out_trans = transpose(ln_out, fwd_dtype)

    gemm1_input_scale_inv = scale_inv[gemm1_input_idx]
    wgrad_1 = gemm(dgelu_trans, dgrad_2_scale_inv, bwd_dtype, True,
                   ln_out_trans, gemm1_input_scale_inv, fwd_dtype, False,
                   jax_dtype_to_te_dtype(g.dtype), FP8Helper.FP8_2X_ACC_WGRAD)

    kernel_1_scale_inv = scale_inv[gemm1_kernel_idx]
    dgrad_1 = gemm(kernel_1_cast, kernel_1_scale_inv, fwd_dtype, True, dgelu, dgrad_2_scale_inv,
                   bwd_dtype, False, jax_dtype_to_te_dtype(g.dtype), FP8Helper.FP8_2X_ACC_DGRAD)
    if major_sharding_type in (MajorShardingType.TP, MajorShardingType.DPTP):
        dgrad_1 = jax.lax.psum(dgrad_1, tp_axis_name)

    if layernorm_type == 'layernorm':
        grad_input, grad_gamma, grad_beta = layernorm_bwd(dgrad_1,
                                                          mu,
                                                          rsigma,
                                                          inputs_,
                                                          gamma,
                                                          zero_centered_gamma=zero_centered_gamma,
                                                          epsilon=epsilon)
    else:
        assert not zero_centered_gamma, "zero_centered_gamma is not supported " \
            "if layernorm_type is 'rmsnorm'"
        grad_input, grad_gamma = rmsnorm_bwd(dgrad_1, rsigma, inputs_, gamma, epsilon=epsilon)
        grad_beta = None

    amax = amax.at[gemm1_input_idx, 0].set(ln_out_amax[0])
    amax = amax.at[gemm1_kernel_idx, 0].set(kernel_1_amax[0])
    amax = amax.at[gemm1_grad_idx, 0].set(dgelu_amax[0])
    amax = amax.at[gemm2_input_idx, 0].set(gated_gelu_amax[0])
    amax = amax.at[gemm2_kernel_idx, 0].set(kernel_2_amax[0])
    amax = amax.at[gemm2_grad_idx, 0].set(grad_amax[0])

    if major_sharding_type in (MajorShardingType.DP, MajorShardingType.DPTP):
        wgrad_1 = jax.lax.psum(wgrad_1, dp_axis_name)
        wgrad_2 = jax.lax.psum(wgrad_2, dp_axis_name)
        grad_gamma = jax.lax.psum(grad_gamma, dp_axis_name)
        if grad_beta is not None:
            grad_beta = jax.lax.psum(grad_beta, dp_axis_name)
        amax = jax.lax.pmax(amax, dp_axis_name)

    if len(fsdp_axis_name) > 0:
        wgrad_1 = jax.lax.psum(wgrad_1, fsdp_axis_name)
        wgrad_2 = jax.lax.psum(wgrad_2, fsdp_axis_name)
        grad_gamma = jax.lax.psum(grad_gamma, fsdp_axis_name)
        if grad_beta is not None:
            grad_beta = jax.lax.psum(grad_beta, fsdp_axis_name)
        amax = jax.lax.pmax(amax, fsdp_axis_name)

    if major_sharding_type in (MajorShardingType.TP, MajorShardingType.DPTP):
        amax = jax.lax.pmax(amax, tp_axis_name)

    grad_input = jnp.reshape(grad_input, input_shape)
    wgrad_1 = jnp.reshape(wgrad_1, kernel_1_shape)
    wgrad_2 = jnp.reshape(wgrad_2, kernel_2_shape)
    return grad_input, grad_gamma, grad_beta, \
           wgrad_1, wgrad_2, \
           fp8_maxs, amax, scale, scale_inv


_fp8_mlp.defvjp(_fp8_mlp_fwd, _fp8_mlp_bwd)
