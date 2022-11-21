# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""JAX layernorm modules"""

from typing import Tuple, Sequence
from functools import partial, reduce
import operator
import jax
import jax.numpy as jnp
from jax.interpreters import pxla
from jax.experimental.maps import xmap

from transformer_engine_jax import DType as TEDType
from .cpp_extensions import te_cast_transpose, te_gemm, _jax_dtype_to_te_dtype
from .cpp_extensions import te_transpose
from .cpp_extensions import te_rmsnorm_fwd, te_rmsnorm_fwd_fp8, te_rmsnorm_bwd
from .fp8 import FP8Helper

jax.config.update('experimental_xmap_spmd_lowering', True)
jax.config.update('experimental_xmap_spmd_lowering_manual', True)

thread_resources = pxla.thread_resources


def layernorm(inputs: jnp.ndarray,
              gamma: jnp.ndarray,
              batch_axis_resource: str = 'data',
              batch_dim_index: int = 0,
              epsilon: float = 1e-6):
    """
    Layernorm wrapper
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

        output = xmap(lambda x, y, _: _layernorm(
            x, y, batch_axis_name=batch_axis_name, epsilon=epsilon),
                      in_axes=[{
                          batch_dim_index: batch_axis_name
                      }, {}, fake_in_axes],
                      out_axes=({
                          batch_dim_index: batch_axis_name
                      }),
                      axis_resources=axis_resources)(
                          inputs_, gamma,
                          jnp.zeros(tuple(64 for _ in range(axis_num))))
        output = jnp.reshape(output, (*output.shape[:batch_dim_index], -1,
                                      *output.shape[batch_dim_index + 2:]))
    else:
        output = _layernorm(inputs, gamma, batch_axis_name="", epsilon=epsilon)
    return output


@partial(jax.custom_vjp, nondiff_argnums=(
    2,
    3,
))
def _layernorm(x, gamma, batch_axis_name, epsilon=1e-6):
    output, _ = _layernorm_fwd(x, gamma, batch_axis_name, epsilon)
    return output


def _layernorm_fwd(
        x,
        gamma,
        batch_axis_name,  # pylint: disable=unused-argument
        epsilon):
    output, invvar = te_rmsnorm_fwd(x, gamma, epsilon)
    return output, (invvar, x, gamma)


def _layernorm_bwd(batch_axis_name, epsilon, ctx, g):
    invvar, x, gamma = ctx
    grad_input, grad_gamma = te_rmsnorm_bwd(g,
                                            invvar,
                                            x,
                                            gamma,
                                            epsilon=epsilon)
    if batch_axis_name:
        grad_gamma = jax.lax.psum(grad_gamma, batch_axis_name)
    return grad_input, grad_gamma


_layernorm.defvjp(_layernorm_fwd, _layernorm_bwd)


def layernorm_fp8_dot(inputs: jnp.ndarray,
                      kernel: jnp.ndarray,
                      gamma: jnp.ndarray,
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
                      batch_dim_index: int = 0,
                      epsilon: float = 1e-6) -> jnp.ndarray:
    """
    LN + fp8 dot fusion wrapper
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
        output = xmap(lambda x, y, z, a, b, c, d, _: _layernorm_fp8_dot(
            x,
            y,
            z,
            a,
            b,
            c,
            d,
            amax_history_idx=amax_history_idx,
            fwd_ctype=fwd_ctype,
            bwd_ctype=bwd_ctype,
            contracting_dims=contracting_dims,
            batch_axis_name=batch_axis_name,
            epsilon=epsilon),
                      in_axes=[{
                          batch_dim_index: batch_axis_name
                      }, {}, {}, {}, {}, {}, {}, fake_in_axes],
                      out_axes=({
                          batch_dim_index: batch_axis_name
                      }),
                      axis_resources=axis_resources)(
                          inputs_, kernel, gamma, fp8_maxs, amax, scale,
                          scale_inv,
                          jnp.zeros(tuple(64 for _ in range(axis_num))))
        output = jnp.reshape(output, (*output.shape[:batch_dim_index], -1,
                                      *output.shape[batch_dim_index + 2:]))
    else:
        output = _layernorm_fp8_dot(inputs,
                                    kernel,
                                    gamma,
                                    fp8_maxs,
                                    amax,
                                    scale,
                                    scale_inv,
                                    amax_history_idx,
                                    fwd_ctype,
                                    bwd_ctype,
                                    contracting_dims,
                                    batch_axis_name="",
                                    epsilon=epsilon)
    return output


@partial(jax.custom_vjp, nondiff_argnums=(
    7,
    8,
    9,
    10,
    11,
    12,
))
def _layernorm_fp8_dot(inputs: jnp.ndarray,
                       kernel: jnp.ndarray,
                       gamma: jnp.ndarray,
                       fp8_maxs: jnp.ndarray,
                       amax: jnp.ndarray,
                       scale: jnp.ndarray,
                       scale_inv: jnp.ndarray,
                       amax_history_idx: int,
                       fwd_ctype: TEDType,
                       bwd_ctype: TEDType,
                       contracting_dims: Tuple[Sequence[int], Sequence[int]],
                       batch_axis_name: str,
                       epsilon: float = 1e-6) -> jnp.ndarray:
    output, _ = _layernorm_fp8_dot_fwd(inputs, kernel, gamma, fp8_maxs, amax,
                                       scale, scale_inv, amax_history_idx,
                                       fwd_ctype, bwd_ctype, contracting_dims,
                                       batch_axis_name, epsilon)
    return output


def _layernorm_fp8_dot_fwd(
        inputs,
        kernel,
        gamma,
        fp8_maxs,
        amax,
        scale,
        scale_inv,
        amax_history_idx,  # pylint: disable=unused-argument
        fwd_ctype,
        bwd_ctype,  # pylint: disable=unused-argument
        contracting_dims,
        batch_axis_name,  # pylint: disable=unused-argument
        epsilon):

    lhs_contracting_dims, rhs_contracting_dims = contracting_dims
    input_shape_pre = inputs.shape[:min(lhs_contracting_dims)]
    input_shape_suf = inputs.shape[min(lhs_contracting_dims):]
    kernel_shape_pre = kernel.shape[:max(rhs_contracting_dims) + 1]
    kernel_shape_suf = kernel.shape[max(rhs_contracting_dims) + 1:]
    input_contracting_size = reduce(operator.mul, input_shape_suf)
    kernel_contracting_size = reduce(operator.mul, kernel_shape_pre)
    assert input_contracting_size == kernel_contracting_size

    ln_out, invvar, input_amax = te_rmsnorm_fwd_fp8(
        inputs,
        gamma,
        amax[FP8Helper.INPUT_META_IDX_PER_GEMM],
        scale[FP8Helper.INPUT_META_IDX_PER_GEMM],
        scale_inv[FP8Helper.INPUT_META_IDX_PER_GEMM],
        epsilon=epsilon)

    assert inputs.shape == ln_out.shape
    ln_out_ = jnp.reshape(ln_out, (-1, input_contracting_size))
    kernel_ = jnp.reshape(kernel, (kernel_contracting_size, -1))

    kernel_cast, kernel_cast_trans, kernel_amax = te_cast_transpose(
        kernel_, amax[FP8Helper.KERNEL_META_IDX_PER_GEMM],
        scale[FP8Helper.KERNEL_META_IDX_PER_GEMM],
        scale_inv[FP8Helper.KERNEL_META_IDX_PER_GEMM], fwd_ctype)

    output = te_gemm(kernel_cast_trans,
                     scale_inv[FP8Helper.KERNEL_META_IDX_PER_GEMM], fwd_ctype,
                     True, ln_out_,
                     scale_inv[FP8Helper.INPUT_META_IDX_PER_GEMM], fwd_ctype,
                     False, _jax_dtype_to_te_dtype(inputs.dtype))
    return jnp.reshape(output, input_shape_pre +
                       kernel_shape_suf), (ln_out_, kernel_cast, fp8_maxs,
                                           amax, scale, scale_inv, input_amax,
                                           kernel_amax, inputs.shape,
                                           kernel.shape, invvar, inputs, gamma)


def _layernorm_fp8_dot_bwd(amax_history_idx, fwd_ctype, bwd_ctype,
                           contracting_dims, batch_axis_name, epsilon, ctx, g):  # pylint: disable=unused-argument
    (
        ln_out_,
        kernel_cast,
        fp8_maxs,
        amax,
        scale,
        scale_inv,
        input_amax,
        kernel_amax,
        inputs_shape,  # pylint: disable=unused-variable
        kernel_shape,  # pylint: disable=unused-variable
        invvar,
        inputs,
        gamma) = ctx

    ln_out_trans = te_transpose(ln_out_, fwd_ctype)
    g = jnp.reshape(g, (ln_out_trans.shape[1], -1))

    # cast and transpose the grad_output
    grad_cast, grad_cast_trans, grad_amax = te_cast_transpose(
        g, amax[FP8Helper.GRAD_META_IDX_PER_GEMM],
        scale[FP8Helper.GRAD_META_IDX_PER_GEMM],
        scale_inv[FP8Helper.GRAD_META_IDX_PER_GEMM], bwd_ctype)

    wgrad = te_gemm(grad_cast_trans,
                    scale_inv[FP8Helper.GRAD_META_IDX_PER_GEMM], bwd_ctype,
                    True, ln_out_trans,
                    scale_inv[FP8Helper.INPUT_META_IDX_PER_GEMM], fwd_ctype,
                    False, _jax_dtype_to_te_dtype(g.dtype))

    dgrad = te_gemm(kernel_cast, scale_inv[FP8Helper.KERNEL_META_IDX_PER_GEMM],
                    fwd_ctype, True, grad_cast,
                    scale_inv[FP8Helper.GRAD_META_IDX_PER_GEMM], bwd_ctype,
                    False, _jax_dtype_to_te_dtype(g.dtype))

    dgrad = jnp.reshape(dgrad, inputs_shape)

    grad_input, grad_gamma = te_rmsnorm_bwd(dgrad,
                                            invvar,
                                            inputs,
                                            gamma,
                                            epsilon=epsilon)

    if batch_axis_name:
        wgrad = jax.lax.psum(wgrad, batch_axis_name)
        grad_gamma = jax.lax.psum(grad_gamma, batch_axis_name)

    amax = amax.at[FP8Helper.INPUT_META_IDX_PER_GEMM,
                   amax_history_idx].set(input_amax[0])
    amax = amax.at[FP8Helper.KERNEL_META_IDX_PER_GEMM,
                   amax_history_idx].set(kernel_amax[0])
    amax = amax.at[FP8Helper.GRAD_META_IDX_PER_GEMM,
                   amax_history_idx].set(grad_amax[0])

    return grad_input, jnp.reshape(
        wgrad, kernel_shape), grad_gamma, fp8_maxs, amax, scale, scale_inv


_layernorm_fp8_dot.defvjp(_layernorm_fp8_dot_fwd, _layernorm_fp8_dot_bwd)
