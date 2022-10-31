# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""JAX MLP modules"""

from typing import Tuple, Sequence, Union, Callable
from functools import partial, reduce
import operator

import jax
import jax.numpy as jnp
from jax.interpreters import pxla
from jax.experimental.maps import xmap

from transformer_engine_jax import DType as TEDType
from .cpp_extensions import _jax_dtype_to_te_dtype
from .cpp_extensions import te_transpose, te_cast_transpose
from .cpp_extensions import te_gated_gelu, te_cast_transpose_dgated_gelu
from .cpp_extensions import te_rmsnorm_fwd_fp8, te_rmsnorm_bwd
from .cpp_extensions import te_gemm

from .fp8 import FP8Helper

jax.config.update('experimental_xmap_spmd_lowering', True)
jax.config.update('experimental_xmap_spmd_lowering_manual', True)

thread_resources = pxla.thread_resources


def fp8_ln_mlp(
    inputs: jnp.ndarray,
    ln_scale: jnp.ndarray,
    kernel_1: jnp.ndarray,
    kernel_2: jnp.ndarray,
    fp8_maxs: jnp.ndarray,
    amax: jnp.ndarray,
    scale: jnp.ndarray,
    scale_inv: jnp.ndarray,
    amax_history_idx: int,
    fwd_ctype: TEDType,
    bwd_ctype: TEDType,
    epsilon: float = 1e-6,
    contracting_dims: Tuple[Sequence[int], Sequence[int]] = ((-1, ), (0, )),
    batch_axis_resource: str = 'data',  # pylint: disable=unused-argument
    batch_dim_index: int = 0,  # pylint: disable=unused-argument
    activations: Sequence[Union[str, Callable]] = ('gelu', 'linear')
) -> jnp.ndarray:
    """
    FP8 layernorm MLP wrapper
    (LN + Dense + act + Dense)
    """
    assert activations == ('gelu', 'linear')
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
        res = xmap(lambda x, lns, k1, k2, a, b, c, d, _: _fp8_mlp(
            x,
            lns,
            k1,
            k2,
            a,
            b,
            c,
            d,
            amax_history_idx=amax_history_idx,
            activations=activations,
            epsilon=epsilon,
            fwd_ctype=fwd_ctype,
            bwd_ctype=bwd_ctype,
            contracting_dims=contracting_dims,
            batch_axis_name=batch_axis_name),
                   in_axes=[{
                       batch_dim_index: batch_axis_name
                   }, {}, {}, {}, {}, {}, {}, {}, fake_in_axes],
                   out_axes=({
                       batch_dim_index: batch_axis_name
                   }),
                   axis_resources=axis_resources)(
                       inputs_, ln_scale, kernel_1, kernel_2, fp8_maxs, amax,
                       scale, scale_inv,
                       jnp.zeros(tuple(64 for _ in range(axis_num))))
        res = jnp.reshape(res, (*res.shape[:batch_dim_index], -1,
                                *res.shape[batch_dim_index + 2:]))
    else:
        res = _fp8_mlp(inputs, ln_scale, kernel_1, kernel_2, fp8_maxs, amax,
                       scale, scale_inv, amax_history_idx, activations,
                       epsilon, fwd_ctype, bwd_ctype, contracting_dims, "")
    return res


@partial(jax.custom_vjp, nondiff_argnums=(8, 9, 10, 11, 12, 13, 14))
def _fp8_mlp(inputs: jnp.ndarray, ln_scale: jnp.ndarray, kernel_1: jnp.ndarray,
             kernel_2: jnp.ndarray, fp8_maxs: jnp.ndarray, amax: jnp.ndarray,
             scale: jnp.ndarray, scale_inv: jnp.ndarray, amax_history_idx: int,
             activations: Sequence[Union[str, Callable]], epsilon: float,
             fwd_ctype: TEDType, bwd_ctype: TEDType,
             contracting_dims: Tuple[Sequence[int],
                                     Sequence[int]], batch_axis_name: str):
    res, _ = _fp8_mlp_fwd(inputs,
                          ln_scale,
                          kernel_1,
                          kernel_2,
                          fp8_maxs,
                          amax,
                          scale,
                          scale_inv,
                          amax_history_idx,
                          activations,
                          epsilon,
                          fwd_ctype,
                          bwd_ctype,
                          contracting_dims=contracting_dims,
                          batch_axis_name=batch_axis_name)
    return res


def _fp8_mlp_fwd(
        inputs,
        gamma,
        kernel_1,
        kernel_2,
        fp8_maxs,
        amax,
        scale,
        scale_inv,
        amax_history_idx,  # pylint: disable=unused-argument
        activations,
        epsilon,
        fwd_ctype,
        bwd_ctype,  # pylint: disable=unused-argument
        contracting_dims,
        batch_axis_name):  # pylint: disable=unused-argument
    if activations != ('gelu', 'linear'):
        raise NotImplementedError(
            "activations only support ('gelu', 'linear') for now.")
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

    ln_out, invvar, ln_out_amax = te_rmsnorm_fwd_fp8(
        inputs_,
        gamma,
        amax[FP8Helper.NUM_META_PER_GEMM * 0 +
             FP8Helper.INPUT_META_IDX_PER_GEMM],
        scale[FP8Helper.NUM_META_PER_GEMM * 0 +
              FP8Helper.INPUT_META_IDX_PER_GEMM],
        scale_inv[FP8Helper.NUM_META_PER_GEMM * 0 +
                  FP8Helper.INPUT_META_IDX_PER_GEMM],
        epsilon=epsilon)
    kernel_1_cast, kernel_1_cast_trans, kernel_1_amax = te_cast_transpose(
        kernel_1_, amax[FP8Helper.NUM_META_PER_GEMM * 0 +
                        FP8Helper.KERNEL_META_IDX_PER_GEMM],
        scale[FP8Helper.NUM_META_PER_GEMM * 0 +
              FP8Helper.KERNEL_META_IDX_PER_GEMM],
        scale_inv[FP8Helper.NUM_META_PER_GEMM * 0 +
                  FP8Helper.KERNEL_META_IDX_PER_GEMM], fwd_ctype)
    dense_1_output = te_gemm(
        kernel_1_cast_trans, scale_inv[FP8Helper.NUM_META_PER_GEMM * 0 +
                                       FP8Helper.KERNEL_META_IDX_PER_GEMM],
        fwd_ctype, True, ln_out, scale_inv[FP8Helper.NUM_META_PER_GEMM * 0 +
                                           FP8Helper.INPUT_META_IDX_PER_GEMM],
        fwd_ctype, False, _jax_dtype_to_te_dtype(inputs.dtype))
    kernel_2_cast, kernel_2_cast_trans, kernel_2_amax = te_cast_transpose(
        kernel_2_, amax[FP8Helper.NUM_META_PER_GEMM * 1 +
                        FP8Helper.KERNEL_META_IDX_PER_GEMM],
        scale[FP8Helper.NUM_META_PER_GEMM * 1 +
              FP8Helper.KERNEL_META_IDX_PER_GEMM],
        scale_inv[FP8Helper.NUM_META_PER_GEMM * 1 +
                  FP8Helper.KERNEL_META_IDX_PER_GEMM], fwd_ctype)
    gated_gelu_output_cast, gated_gelu_amax = te_gated_gelu(
        dense_1_output, amax[FP8Helper.NUM_META_PER_GEMM * 1 +
                             FP8Helper.INPUT_META_IDX_PER_GEMM],
        scale[FP8Helper.NUM_META_PER_GEMM * 1 +
              FP8Helper.INPUT_META_IDX_PER_GEMM],
        scale_inv[FP8Helper.NUM_META_PER_GEMM * 1 +
                  FP8Helper.INPUT_META_IDX_PER_GEMM], fwd_ctype)
    res = te_gemm(
        kernel_2_cast_trans, scale_inv[FP8Helper.NUM_META_PER_GEMM * 1 +
                                       FP8Helper.KERNEL_META_IDX_PER_GEMM],
        fwd_ctype, True, gated_gelu_output_cast,
        scale_inv[FP8Helper.NUM_META_PER_GEMM * 1 +
                  FP8Helper.INPUT_META_IDX_PER_GEMM], fwd_ctype, False,
        _jax_dtype_to_te_dtype(inputs.dtype))

    return jnp.reshape(res, input_shape_pre + kernel_2_shape_suf), \
           (inputs_, ln_out, invvar, gamma, dense_1_output, gated_gelu_output_cast, \
                kernel_1_cast, kernel_2_cast,
                    fp8_maxs, amax, scale, scale_inv, \
                        ln_out_amax, gated_gelu_amax, kernel_1_amax, kernel_2_amax,
                            inputs.shape, kernel_1.shape, kernel_2.shape)


def _fp8_mlp_bwd(
        amax_history_idx,
        activations,  # pylint: disable=unused-argument
        epsilon,
        fwd_ctype,
        bwd_ctype,
        contracting_dims,  # pylint: disable=unused-argument
        batch_axis_name,
        ctx,
        g):
    inputs_, ln_out, invvar, gamma, dense_1_output, gated_gelu_output_cast, \
        kernel_1_cast, kernel_2_cast, \
            fp8_maxs, amax, scale, scale_inv, \
                ln_out_amax, gated_gelu_amax, kernel_1_amax, kernel_2_amax, \
                    input_shape, kernel_1_shape, kernel_2_shape = ctx

    g = jnp.reshape(g, (ln_out.shape[0], -1))

    grad_cast, grad_cast_trans, grad_amax = te_cast_transpose(
        g, amax[FP8Helper.NUM_META_PER_GEMM * 1 +
                FP8Helper.GRAD_META_IDX_PER_GEMM],
        scale[FP8Helper.NUM_META_PER_GEMM * 1 +
              FP8Helper.GRAD_META_IDX_PER_GEMM],
        scale_inv[FP8Helper.NUM_META_PER_GEMM * 1 +
                  FP8Helper.GRAD_META_IDX_PER_GEMM], bwd_ctype)
    gated_gelu_output_cast_trans = te_transpose(gated_gelu_output_cast,
                                                fwd_ctype)
    wgrad_2 = te_gemm(
        grad_cast_trans, scale_inv[FP8Helper.NUM_META_PER_GEMM * 1 +
                                   FP8Helper.GRAD_META_IDX_PER_GEMM],
        bwd_ctype, True, gated_gelu_output_cast_trans,
        scale_inv[FP8Helper.NUM_META_PER_GEMM * 1 +
                  FP8Helper.INPUT_META_IDX_PER_GEMM], fwd_ctype, False,
        _jax_dtype_to_te_dtype(g.dtype))
    dgrad_2 = te_gemm(
        kernel_2_cast, scale_inv[FP8Helper.NUM_META_PER_GEMM * 1 +
                                 FP8Helper.KERNEL_META_IDX_PER_GEMM],
        fwd_ctype, True, grad_cast,
        scale_inv[FP8Helper.NUM_META_PER_GEMM * 1 +
                  FP8Helper.GRAD_META_IDX_PER_GEMM], bwd_ctype, False,
        _jax_dtype_to_te_dtype(g.dtype))
    dgelu, dgelu_trans, dgelu_amax = te_cast_transpose_dgated_gelu(
        dgrad_2, dense_1_output, amax[FP8Helper.NUM_META_PER_GEMM * 0 +
                                      FP8Helper.GRAD_META_IDX_PER_GEMM],
        scale[FP8Helper.NUM_META_PER_GEMM * 0 +
              FP8Helper.GRAD_META_IDX_PER_GEMM],
        scale_inv[FP8Helper.NUM_META_PER_GEMM * 0 +
                  FP8Helper.GRAD_META_IDX_PER_GEMM], bwd_ctype)
    ln_out_trans = te_transpose(ln_out, fwd_ctype)
    wgrad_1 = te_gemm(
        dgelu_trans, scale_inv[FP8Helper.NUM_META_PER_GEMM * 0 +
                               FP8Helper.GRAD_META_IDX_PER_GEMM], bwd_ctype,
        True, ln_out_trans, scale_inv[FP8Helper.NUM_META_PER_GEMM * 0 +
                                      FP8Helper.INPUT_META_IDX_PER_GEMM],
        fwd_ctype, False, _jax_dtype_to_te_dtype(g.dtype))
    dgrad_1 = te_gemm(
        kernel_1_cast, scale_inv[FP8Helper.NUM_META_PER_GEMM * 0 +
                                 FP8Helper.KERNEL_META_IDX_PER_GEMM],
        fwd_ctype, True, dgelu, scale_inv[FP8Helper.NUM_META_PER_GEMM * 0 +
                                          FP8Helper.GRAD_META_IDX_PER_GEMM],
        bwd_ctype, False, _jax_dtype_to_te_dtype(g.dtype))
    grad_input, grad_gamma = te_rmsnorm_bwd(dgrad_1,
                                            invvar,
                                            inputs_,
                                            gamma,
                                            epsilon=epsilon)
    if batch_axis_name:
        wgrad_1 = jax.lax.psum(wgrad_1, batch_axis_name)
        wgrad_2 = jax.lax.psum(wgrad_2, batch_axis_name)
        grad_gamma = jax.lax.psum(grad_gamma, batch_axis_name)

    amax = amax.at[FP8Helper.NUM_META_PER_GEMM * 0 +
                   FP8Helper.INPUT_META_IDX_PER_GEMM,
                   amax_history_idx].set(ln_out_amax[0])
    amax = amax.at[FP8Helper.NUM_META_PER_GEMM * 0 +
                   FP8Helper.KERNEL_META_IDX_PER_GEMM,
                   amax_history_idx].set(kernel_1_amax[0])
    amax = amax.at[FP8Helper.NUM_META_PER_GEMM * 0 +
                   FP8Helper.GRAD_META_IDX_PER_GEMM,
                   amax_history_idx].set(dgelu_amax[0])
    amax = amax.at[FP8Helper.NUM_META_PER_GEMM * 1 +
                   FP8Helper.INPUT_META_IDX_PER_GEMM,
                   amax_history_idx].set(gated_gelu_amax[0])
    amax = amax.at[FP8Helper.NUM_META_PER_GEMM * 1 +
                   FP8Helper.KERNEL_META_IDX_PER_GEMM,
                   amax_history_idx].set(kernel_2_amax[0])
    amax = amax.at[FP8Helper.NUM_META_PER_GEMM * 1 +
                   FP8Helper.GRAD_META_IDX_PER_GEMM,
                   amax_history_idx].set(grad_amax[0])
    return jnp.reshape(grad_input, input_shape), grad_gamma, jnp.reshape(
        wgrad_1, kernel_1_shape), jnp.reshape(
            wgrad_2, kernel_2_shape), fp8_maxs, amax, scale, scale_inv


_fp8_mlp.defvjp(_fp8_mlp_fwd, _fp8_mlp_bwd)
