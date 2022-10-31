# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""JAX te modules"""

from typing import Tuple, Sequence
from functools import partial, reduce
import operator
import jax
import jax.numpy as jnp
from jax.interpreters import pxla
from jax.experimental.maps import xmap

from transformer_engine_jax import DType as TEDType
from .cpp_extensions import te_cast_transpose, te_gemm, _jax_dtype_to_te_dtype
from .fp8 import FP8Helper

jax.config.update('experimental_xmap_spmd_lowering', True)
jax.config.update('experimental_xmap_spmd_lowering_manual', True)

thread_resources = pxla.thread_resources


def fp8_dot(inputs: jnp.ndarray,
            kernel: jnp.ndarray,
            fp8_maxs: jnp.ndarray,
            amax: jnp.ndarray,
            scale: jnp.ndarray,
            scale_inv: jnp.ndarray,
            amax_history_idx: int,
            fwd_ctype: TEDType,
            bwd_ctype: TEDType,
            contracting_dims: Tuple[Sequence[int],
                                    Sequence[int]] = ((-1, ), (0, )),
            batch_axis_resource: str = 'data',
            batch_dim_index: int = 0) -> jnp.ndarray:
    """
    FP8 dot wrapper
    """
    mesh = thread_resources.env.physical_mesh
    if batch_axis_resource and batch_axis_resource in mesh.axis_names:
        batch_axis_name = 'batch'
        device_num = mesh.shape[batch_axis_resource]
        axis_resources = {batch_axis_name: batch_axis_resource}
        fake_in_axes = {}
        axis_names = [n for n in mesh.axis_names if n != batch_axis_resource]
        axis_num = len(axis_names)
        for i in range(axis_num):
            axis_name = 'axis_' + str(i)
            axis_resources[axis_name] = axis_names[i]
            fake_in_axes[i] = axis_name
        inputs_ = jnp.reshape(inputs,
                              (*inputs.shape[:batch_dim_index], device_num, -1,
                               *inputs.shape[batch_dim_index + 1:]))
        res = xmap(lambda x, y, a, b, c, d, _: _fp8_dot(
            x,
            y,
            a,
            b,
            c,
            d,
            amax_history_idx=amax_history_idx,
            fwd_ctype=fwd_ctype,
            bwd_ctype=bwd_ctype,
            contracting_dims=contracting_dims,
            batch_axis_name=batch_axis_name),
                   in_axes=[{
                       batch_dim_index: batch_axis_name
                   }, {}, {}, {}, {}, {}, fake_in_axes],
                   out_axes=({
                       batch_dim_index: batch_axis_name
                   }),
                   axis_resources=axis_resources)(
                       inputs_, kernel, fp8_maxs, amax, scale, scale_inv,
                       jnp.zeros(tuple(64 for _ in range(axis_num))))
        res = jnp.reshape(res, (*res.shape[:batch_dim_index], -1,
                                *res.shape[batch_dim_index + 2:]))
    else:
        res = _fp8_dot(inputs,
                       kernel,
                       fp8_maxs,
                       amax,
                       scale,
                       scale_inv,
                       amax_history_idx=amax_history_idx,
                       fwd_ctype=fwd_ctype,
                       bwd_ctype=bwd_ctype,
                       contracting_dims=contracting_dims,
                       batch_axis_name="")
    return res


@partial(jax.custom_vjp, nondiff_argnums=(6, 7, 8, 9, 10))
def _fp8_dot(inputs: jnp.ndarray, kernel: jnp.ndarray, fp8_maxs: jnp.ndarray,
             amax: jnp.ndarray, scale: jnp.ndarray, scale_inv: jnp.ndarray,
             amax_history_idx: int, fwd_ctype: TEDType, bwd_ctype: TEDType,
             contracting_dims: Tuple[Sequence[int],
                                     Sequence[int]], batch_axis_name: str):
    res, _ = _fp8_dot_fwd(inputs,
                          kernel,
                          fp8_maxs,
                          amax,
                          scale,
                          scale_inv,
                          amax_history_idx,
                          fwd_ctype,
                          bwd_ctype,
                          contracting_dims=contracting_dims,
                          batch_axis_name=batch_axis_name)
    return res


def _fp8_dot_fwd(
        inputs,
        kernel,
        fp8_maxs,
        amax,
        scale,
        scale_inv,
        amax_history_idx,  # pylint: disable=unused-argument
        fwd_ctype,
        bwd_ctype,  # pylint: disable=unused-argument
        contracting_dims,
        batch_axis_name):  # pylint: disable=unused-argument
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
    input_cast, input_cast_trans, input_amax = te_cast_transpose(
        inputs_, amax[FP8Helper.INPUT_META_IDX_PER_GEMM],
        scale[FP8Helper.INPUT_META_IDX_PER_GEMM],
        scale_inv[FP8Helper.INPUT_META_IDX_PER_GEMM], fwd_ctype)
    kernel_cast, kernel_cast_trans, kernel_amax = te_cast_transpose(
        kernel_, amax[FP8Helper.KERNEL_META_IDX_PER_GEMM],
        scale[FP8Helper.KERNEL_META_IDX_PER_GEMM],
        scale_inv[FP8Helper.KERNEL_META_IDX_PER_GEMM], fwd_ctype)
    res = te_gemm(kernel_cast_trans,
                  scale_inv[FP8Helper.KERNEL_META_IDX_PER_GEMM], fwd_ctype,
                  True, input_cast,
                  scale_inv[FP8Helper.INPUT_META_IDX_PER_GEMM], fwd_ctype,
                  False, _jax_dtype_to_te_dtype(inputs.dtype))
    return jnp.reshape(res, input_shape_pre + kernel_shape_suf), \
           (input_cast_trans, kernel_cast,
            fp8_maxs, amax, scale, scale_inv, input_amax, kernel_amax,
            inputs.shape, kernel.shape)


def _fp8_dot_bwd(
        amax_history_idx,
        fwd_ctype,
        bwd_ctype,
        contracting_dims,  # pylint: disable=unused-argument
        batch_axis_name,
        ctx,
        g):
    input_cast_trans, kernel_cast, \
        fp8_maxs, amax, scale, scale_inv, input_amax, kernel_amax, \
            inputs_shape, kernel_shape = ctx
    g = jnp.reshape(g, (input_cast_trans.shape[1], -1))
    grad_cast, grad_cast_trans, grad_amax = te_cast_transpose(
        g, amax[FP8Helper.GRAD_META_IDX_PER_GEMM],
        scale[FP8Helper.GRAD_META_IDX_PER_GEMM],
        scale_inv[FP8Helper.GRAD_META_IDX_PER_GEMM], bwd_ctype)
    wgrad = te_gemm(grad_cast_trans,
                    scale_inv[FP8Helper.GRAD_META_IDX_PER_GEMM], bwd_ctype,
                    True, input_cast_trans,
                    scale_inv[FP8Helper.INPUT_META_IDX_PER_GEMM], fwd_ctype,
                    False, _jax_dtype_to_te_dtype(g.dtype))
    dgrad = te_gemm(kernel_cast, scale_inv[FP8Helper.KERNEL_META_IDX_PER_GEMM],
                    fwd_ctype, True, grad_cast,
                    scale_inv[FP8Helper.GRAD_META_IDX_PER_GEMM], bwd_ctype,
                    False, _jax_dtype_to_te_dtype(g.dtype))
    if batch_axis_name:
        wgrad = jax.lax.psum(wgrad, batch_axis_name)

    amax = amax.at[FP8Helper.INPUT_META_IDX_PER_GEMM,
                   amax_history_idx].set(input_amax[0])
    amax = amax.at[FP8Helper.KERNEL_META_IDX_PER_GEMM,
                   amax_history_idx].set(kernel_amax[0])
    amax = amax.at[FP8Helper.GRAD_META_IDX_PER_GEMM,
                   amax_history_idx].set(grad_amax[0])
    return jnp.reshape(dgrad, inputs_shape), jnp.reshape(
        wgrad, kernel_shape), fp8_maxs, amax, scale, scale_inv


_fp8_dot.defvjp(_fp8_dot_fwd, _fp8_dot_bwd)
