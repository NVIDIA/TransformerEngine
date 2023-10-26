# Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""JAX te modules"""

from typing import Tuple, Sequence
from functools import partial, reduce
import operator
import jax
import jax.numpy as jnp

from .cpp_extensions import cast_transpose
from .fp8 import FP8Helper, FP8MetaPackage


def type_safe_dot_general(
    x,
    kernel,
    fp8_meta_pkg: FP8MetaPackage = None,
    contracting_dims: Tuple[Sequence[int], Sequence[int]] = ((1,), (0,))
) -> jnp.ndarray:
    """
    Type safe dot_general, including FP8.
    """

    if fp8_meta_pkg is None:
        kernel = jnp.asarray(kernel, x.dtype)
        return jax.lax.dot_general(x, kernel, (contracting_dims, ((), ())))

    fp8_max = fp8_meta_pkg.fp8_max
    amax = fp8_meta_pkg.amax
    scale = fp8_meta_pkg.scale
    scale_inv = fp8_meta_pkg.scale_inv
    fwd_dtype = FP8Helper.FWD_DTYPE
    bwd_dtype = FP8Helper.BWD_DTYPE
    return _fp8_dot(x, kernel, fp8_max, amax, scale, scale_inv, fwd_dtype, bwd_dtype,
                    contracting_dims)


def dequantize(x, dq_dtype, scale_inv):
    """
    Dequantize with scale_inv.
    """
    return x.astype(dq_dtype) * scale_inv.astype(dq_dtype)


# Apply jit to guarantee correctness of FP8 GEMM.
@partial(jax.jit, static_argnums=(4, 5, 6))
def fp8_dot_impl(
        q_lhs: jnp.ndarray,
        q_rhs: jnp.ndarray,
        lhs_scale_inv: jnp.ndarray,
        rhs_scale_inv: jnp.ndarray,
        ctype: jnp.dtype,    # computing type
        already_trans_lhs: bool = False,
        already_trans_rhs: bool = False):
    """
    FP8 GEMM for XLA pattern match
    """
    lhs_dim = 0 if already_trans_lhs else 1
    rhs_dim = 1 if already_trans_rhs else 0

    dim_nums = (((lhs_dim,), (rhs_dim,)), ((), ()))

    lhs = dequantize(q_lhs, ctype, lhs_scale_inv)
    rhs = dequantize(q_rhs, ctype, rhs_scale_inv)

    return jax.lax.dot_general(lhs, rhs, dim_nums)


@partial(jax.custom_vjp, nondiff_argnums=(6, 7, 8))
def _fp8_dot(x: jnp.ndarray, kernel: jnp.ndarray, fp8_max: jnp.ndarray, amax: jnp.ndarray,
             scale: jnp.ndarray, scale_inv: jnp.ndarray, fwd_dtype: jnp.dtype, bwd_dtype: jnp.dtype,
             contracting_dims: Tuple[Sequence[int], Sequence[int]]):
    output, _ = _fp8_dot_fwd_rule(x, kernel, fp8_max, amax, scale, scale_inv, fwd_dtype, bwd_dtype,
                                  contracting_dims)
    return output


def _fp8_dot_fwd_rule(
        x,
        kernel,
        fp8_max,
        amax,
        scale,
        scale_inv,
        fwd_dtype,
        bwd_dtype,    # pylint: disable=unused-argument
        contracting_dims):
    lhs_contracting_dims, rhs_contracting_dims = contracting_dims
    x_shape_pre = x.shape[:min(lhs_contracting_dims)]
    x_shape_suf = x.shape[min(lhs_contracting_dims):]
    kernel_shape_pre = kernel.shape[:max(rhs_contracting_dims) + 1]
    kernel_shape_suf = kernel.shape[max(rhs_contracting_dims) + 1:]
    x_contracting_size = reduce(operator.mul, x_shape_suf)
    kernel_contracting_size = reduce(operator.mul, kernel_shape_pre)
    assert x_contracting_size == kernel_contracting_size
    x_ = jnp.reshape(x, (-1, x_contracting_size))
    kernel_ = jnp.reshape(kernel, (kernel_contracting_size, -1))

    output_shape = x_shape_pre + kernel_shape_suf

    amax = FP8Helper.update_amax_history(amax)

    gemm_x_idx, gemm_kernel_idx, _ = FP8Helper.get_fp8_meta_indices(0)

    x_amax = amax[gemm_x_idx, 0:1]
    x_scale = scale[gemm_x_idx]
    x_scale_inv = scale_inv[gemm_x_idx]

    casted_x, casted_xt, updated_x_amax = \
        cast_transpose(x_, x_amax, x_scale, x_scale_inv, fwd_dtype, static_axis_boundary=-1)

    kernel_amax = amax[gemm_kernel_idx, 0:1]
    kernel_scale = scale[gemm_kernel_idx]
    kernel_scale_inv = scale_inv[gemm_kernel_idx]

    casted_kerenl, casted_kerenl_t, updated_kernel_amax = \
        cast_transpose(kernel_, kernel_amax, kernel_scale, kernel_scale_inv,
                       fwd_dtype, static_axis_boundary=-1)

    output = fp8_dot_impl(casted_x, casted_kerenl_t, x_scale_inv, kernel_scale_inv, x.dtype, False,
                          True)
    output = jnp.reshape(output, output_shape)

    ctx = (casted_xt, casted_kerenl, fp8_max, amax, scale, scale_inv, updated_x_amax,
           updated_kernel_amax, x.shape, kernel.shape)
    return output, ctx


def _fp8_dot_bwd_rule(fwd_dtype, bwd_dtype, contracting_dims, ctx, grad):    # pylint: disable=unused-argument
    casted_xt, casted_kerenl, fp8_max, amax, scale, scale_inv, \
        updated_x_amax, updated_kernel_amax, x_shape, kernel_shape = ctx

    gemm_x_idx, gemm_kernel_idx, gemm_grad_idx = FP8Helper.get_fp8_meta_indices(0)

    grad = jnp.reshape(grad, (casted_xt.shape[1], -1))

    grad_amax = amax[gemm_grad_idx, 0:1]
    grad_scale = scale[gemm_grad_idx]
    grad_scale_inv = scale_inv[gemm_grad_idx]

    casted_grad, casted_grad_t, updated_grad_amax = \
        cast_transpose(grad, grad_amax, grad_scale, grad_scale_inv,
                                                           bwd_dtype, static_axis_boundary=-1)

    x_scale_inv = scale_inv[gemm_x_idx]
    wgrad = fp8_dot_impl(casted_xt, casted_grad_t, x_scale_inv, grad_scale_inv, grad.dtype, False,
                         True)

    kernel_scale_inv = scale_inv[gemm_kernel_idx]
    dgrad = fp8_dot_impl(casted_grad, casted_kerenl, grad_scale_inv, kernel_scale_inv, grad.dtype,
                         False, True)

    dgrad = jnp.reshape(dgrad, x_shape)
    wgrad = jnp.reshape(wgrad, kernel_shape)

    amax = amax.at[gemm_x_idx, 0].set(updated_x_amax[0])
    amax = amax.at[gemm_kernel_idx, 0].set(updated_kernel_amax[0])
    amax = amax.at[gemm_grad_idx, 0].set(updated_grad_amax[0])

    scale, scale_inv = FP8Helper.update_fp8_scale(fp8_max, amax, scale)

    return dgrad, wgrad, fp8_max, amax, scale, scale_inv


_fp8_dot.defvjp(_fp8_dot_fwd_rule, _fp8_dot_bwd_rule)
