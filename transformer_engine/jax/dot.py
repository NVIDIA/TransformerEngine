# Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""JAX te modules"""

from typing import Tuple, Sequence
from functools import partial, reduce
import operator
import jax
import jax.numpy as jnp

from transformer_engine_jax import DType as TEDType
from .cpp_extensions import cast_transpose, gemm, jax_dtype_to_te_dtype
from .fp8 import FP8Helper, FP8GemmPackage
from .sharding import ShardingType, get_dot_sharding_meta, get_fp8_meta_sharding_meta
from .sharding import is_dp_enabled, is_tp_enabled, merge_axis_resources
from .sharding import xmap_runner, extend_fsdp_sharding_meta

jax.config.update('experimental_xmap_spmd_lowering', True)
jax.config.update('experimental_xmap_spmd_lowering_manual', True)


def fp8_dot(fp8_gemm_pkg: FP8GemmPackage,
            fwd_dtype: TEDType,
            bwd_dtype: TEDType,
            contracting_dims: Tuple[Sequence[int], Sequence[int]] = ((-1,), (0,)),
            sharding_type: ShardingType = ShardingType.SINGLE,
            dp_dim_index: int = 0) -> jnp.ndarray:
    """
    FP8 dot wrapper
    """
    assert fp8_gemm_pkg.num_of_gemm == 1
    inputs = fp8_gemm_pkg.inputs
    kernel = fp8_gemm_pkg.kernels[0]
    fp8_max = fp8_gemm_pkg.fp8_max
    amax = fp8_gemm_pkg.amax
    scale = fp8_gemm_pkg.scale
    scale_inv = fp8_gemm_pkg.scale_inv

    if sharding_type is ShardingType.SINGLE:
        res = _fp8_dot(inputs,
                       kernel,
                       fp8_max,
                       amax,
                       scale,
                       scale_inv,
                       fwd_dtype=fwd_dtype,
                       bwd_dtype=bwd_dtype,
                       contracting_dims=contracting_dims,
                       sharding_type=sharding_type,
                       dp_axis_name="",
                       tp_axis_name="",
                       fsdp_axis_name="")
    else:
        dp_axis_name = "batch"
        tp_axis_name = "model"
        kernel_tp_index = None
        # TODO (Ming Huang): Should we add a new argument to support general sharding to kernel? # pylint: disable=fixme
        if sharding_type in (ShardingType.TP_COL, ShardingType.DP_TP_COL):
            kernel_tp_index = len(kernel.shape) - 1
        elif sharding_type in (ShardingType.TP_ROW, ShardingType.DP_TP_ROW):
            kernel_tp_index = 0

        input_tp_index = len(inputs.shape) - 1
        sharding_meta = get_dot_sharding_meta(sharding_type, inputs.shape, kernel.shape,
                                              dp_dim_index, input_tp_index, kernel_tp_index,
                                              contracting_dims, dp_axis_name, tp_axis_name)
        sharding_meta, fsdp_axis_name = extend_fsdp_sharding_meta(sharding_meta, {0: dp_dim_index})
        inputs_ = jnp.reshape(inputs, sharding_meta.input_shapes[0])    # 0 for input
        kernel_ = jnp.reshape(kernel, sharding_meta.input_shapes[1])    # 1 for kernel

        num_of_fp8_meta_kind = 4    # fp8_max, amax, scale, scale_inv
        fp8_sharding_meta = get_fp8_meta_sharding_meta(sharding_type, num_of_fp8_meta_kind,
                                                       dp_axis_name, tp_axis_name)

        axis_resources = merge_axis_resources(
            [sharding_meta.axis_resources, fp8_sharding_meta.axis_resources])

        partial_fp8_dot = partial(_fp8_dot,
                                  fwd_dtype=fwd_dtype,
                                  bwd_dtype=bwd_dtype,
                                  contracting_dims=contracting_dims,
                                  sharding_type=sharding_type,
                                  dp_axis_name=dp_axis_name,
                                  tp_axis_name=tp_axis_name,
                                  fsdp_axis_name=fsdp_axis_name)
        res = xmap_runner(partial_fp8_dot, (*sharding_meta.in_axes, *fp8_sharding_meta.in_axes),
                          sharding_meta.out_axes, axis_resources,
                          (inputs_, kernel_, fp8_max, amax, scale, scale_inv))

        res = jnp.reshape(res, sharding_meta.output_shapes[0])

    return res


@partial(jax.custom_vjp, nondiff_argnums=(6, 7, 8, 9, 10, 11, 12))
def _fp8_dot(inputs: jnp.ndarray, kernel: jnp.ndarray, fp8_maxs: jnp.ndarray, amax: jnp.ndarray,
             scale: jnp.ndarray, scale_inv: jnp.ndarray, fwd_dtype: TEDType, bwd_dtype: TEDType,
             contracting_dims: Tuple[Sequence[int], Sequence[int]], sharding_type: ShardingType,
             dp_axis_name: str, tp_axis_name: str, fsdp_axis_name: str):
    res, _ = _fp8_dot_fwd(inputs,
                          kernel,
                          fp8_maxs,
                          amax,
                          scale,
                          scale_inv,
                          fwd_dtype,
                          bwd_dtype,
                          contracting_dims=contracting_dims,
                          sharding_type=sharding_type,
                          dp_axis_name=dp_axis_name,
                          tp_axis_name=tp_axis_name,
                          fsdp_axis_name=fsdp_axis_name)
    return res


def _fp8_dot_fwd(
        inputs,
        kernel,
        fp8_maxs,
        amax,
        scale,
        scale_inv,
        fwd_dtype,
        bwd_dtype,    # pylint: disable=unused-argument
        contracting_dims,
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
    inputs_ = jnp.reshape(inputs, (-1, input_contracting_size))
    kernel_ = jnp.reshape(kernel, (kernel_contracting_size, -1))

    amax = FP8Helper.update_amax_history(amax)

    gemm_input_idx, gemm_kernel_idx, _ = FP8Helper.get_fp8_meta_indices(0)

    input_amax = amax[gemm_input_idx, 0:1]
    input_scale = scale[gemm_input_idx]
    input_scale_inv = scale_inv[gemm_input_idx]
    input_cast, input_cast_trans, input_amax = cast_transpose(inputs_, input_amax, input_scale,
                                                              input_scale_inv, fwd_dtype)

    kernel_amax = amax[gemm_kernel_idx, 0:1]
    kernel_scale = scale[gemm_kernel_idx]
    kernel_scale_inv = scale_inv[gemm_kernel_idx]
    kernel_cast, kernel_cast_trans, kernel_amax = cast_transpose(kernel_, kernel_amax, kernel_scale,
                                                                 kernel_scale_inv, fwd_dtype)
    res = gemm(kernel_cast_trans, kernel_scale_inv, fwd_dtype, True, input_cast, input_scale_inv,
               fwd_dtype, False, jax_dtype_to_te_dtype(inputs.dtype), FP8Helper.FP8_2X_ACC_FPROP)

    if sharding_type in (ShardingType.TP_ROW, ShardingType.DP_TP_ROW):
        res = jax.lax.psum(res, tp_axis_name)

    # (input_shape_pre, input_shape_suf)
    # x (kernel_shape_pre, kernel_shape_suf)
    # = (input_shape_pre, kernel_shape_suf)
    output_shape = input_shape_pre + kernel_shape_suf
    res = jnp.reshape(res, output_shape)

    ctx = (input_cast_trans, kernel_cast, fp8_maxs, amax, scale, scale_inv, input_amax, kernel_amax,
           inputs.shape, kernel.shape)
    return res, ctx


def _fp8_dot_bwd(
        fwd_dtype,
        bwd_dtype,
        contracting_dims,    # pylint: disable=unused-argument
        sharding_type,
        dp_axis_name,
        tp_axis_name,
        fsdp_axis_name,
        ctx,
        g):
    input_cast_trans, kernel_cast, \
    fp8_maxs, amax, scale, scale_inv, \
    input_amax, kernel_amax, \
    inputs_shape, kernel_shape = ctx

    gemm_input_idx, gemm_kernel_idx, gemm_grad_idx = FP8Helper.get_fp8_meta_indices(0)

    grad_amax = amax[gemm_grad_idx, 0:1]
    grad_scale = scale[gemm_grad_idx]
    grad_scale_inv = scale_inv[gemm_grad_idx]
    g = jnp.reshape(g, (input_cast_trans.shape[1], -1))
    grad_cast, grad_cast_trans, grad_amax = cast_transpose(g, grad_amax, grad_scale, grad_scale_inv,
                                                           bwd_dtype)

    input_scale_inv = scale_inv[gemm_input_idx]
    wgrad = gemm(grad_cast_trans, grad_scale_inv, bwd_dtype,
                 True, input_cast_trans, input_scale_inv, fwd_dtype, False,
                 jax_dtype_to_te_dtype(g.dtype), FP8Helper.FP8_2X_ACC_WGRAD)

    kernel_scale_inv = scale_inv[gemm_kernel_idx]
    dgrad = gemm(kernel_cast, kernel_scale_inv, fwd_dtype, True, grad_cast, grad_scale_inv,
                 bwd_dtype, False, jax_dtype_to_te_dtype(g.dtype), FP8Helper.FP8_2X_ACC_DGRAD)

    amax = amax.at[gemm_input_idx, 0].set(input_amax[0])
    amax = amax.at[gemm_kernel_idx, 0].set(kernel_amax[0])
    amax = amax.at[gemm_grad_idx, 0].set(grad_amax[0])

    if is_dp_enabled(sharding_type.value[0]):
        wgrad = jax.lax.psum(wgrad, dp_axis_name)
        amax = jax.lax.pmax(amax, dp_axis_name)

    if len(fsdp_axis_name) > 0:
        wgrad = jax.lax.psum(wgrad, fsdp_axis_name)
        amax = jax.lax.pmax(amax, fsdp_axis_name)

    if is_tp_enabled(sharding_type.value[0]):
        amax = jax.lax.pmax(amax, tp_axis_name)

    if sharding_type in (ShardingType.TP_COL, ShardingType.DP_TP_COL):
        dgrad = jax.lax.psum(dgrad, tp_axis_name)

    dgrad = jnp.reshape(dgrad, inputs_shape)
    wgrad = jnp.reshape(wgrad, kernel_shape)
    return dgrad, wgrad, fp8_maxs, amax, scale, scale_inv


_fp8_dot.defvjp(_fp8_dot_fwd, _fp8_dot_bwd)
