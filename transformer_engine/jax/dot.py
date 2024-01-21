# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""JAX te modules"""

from typing import Tuple, Sequence
from functools import partial
import jax
import jax.numpy as jnp

from .cpp_extensions import cast_transpose
from .fp8 import FP8Helper, FP8MetaPackage

Precision = jax.lax.Precision


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


def quantize(x, q_dtype, scale):
    """
    Quantize with scale.
    """
    updated_amax = jnp.max(jnp.abs(x)).astype(scale.dtype)
    dtype_max = (jnp.finfo(q_dtype).max).astype(x.dtype)
    scale = scale.astype(x.dtype)
    clipped_scaled_x = jnp.clip((x * scale), -dtype_max, dtype_max)
    return clipped_scaled_x.astype(q_dtype), updated_amax


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
        contracting_dims: Tuple[Sequence[int], Sequence[int]],
        precision: Precision = None):
    """
    FP8 GEMM for XLA pattern match
    """
    dim_nums = (contracting_dims, ((), ()))

    lhs = dequantize(q_lhs, ctype, lhs_scale_inv)
    rhs = dequantize(q_rhs, ctype, rhs_scale_inv)

    return jax.lax.dot_general(lhs, rhs, dim_nums, precision=precision)


def get_precision_of_fp8_dot(enable_2xACC: bool):
    """
    Get Precision of FP8 DOT.
    """
    return jax.lax.Precision.HIGHEST if enable_2xACC else jax.lax.Precision.DEFAULT


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

    x_shape_suf = x.shape[min(lhs_contracting_dims):]
    kernel_shape_pre = kernel.shape[:max(rhs_contracting_dims) + 1]
    assert x_shape_suf == kernel_shape_pre

    amax = FP8Helper.update_amax_history(amax)

    gemm_x_idx, gemm_kernel_idx, _ = FP8Helper.get_fp8_meta_indices(0)

    x_scale = scale[gemm_x_idx]
    x_scale_inv = scale_inv[gemm_x_idx]
    # Note (Ming Huang): Use native cast to allow XLA handle tranpose for avoiding
    # unnecessary copy to break FP8 GEMM pattern matching.
    casted_x, updated_x_amax = quantize(x, fwd_dtype, x_scale)

    kernel_scale = scale[gemm_kernel_idx]
    kernel_scale_inv = scale_inv[gemm_kernel_idx]
    # Note (Ming Huang): Use native cast to allow XLA handle tranpose for avoiding
    # unnecessary copy to break FP8 GEMM pattern matching.
    casted_kernel, updated_kernel_amax = quantize(kernel, fwd_dtype, kernel_scale)

    output = fp8_dot_impl(casted_x, casted_kernel, x_scale_inv, kernel_scale_inv, x.dtype,
                          (lhs_contracting_dims, rhs_contracting_dims),
                          get_precision_of_fp8_dot(FP8Helper.FP8_2X_ACC_FPROP))

    ctx = (casted_x, casted_kernel, fp8_max, amax, scale, scale_inv, updated_x_amax,
           updated_kernel_amax, x.shape, kernel.shape)
    return output, ctx


def _fp8_dot_bwd_rule(fwd_dtype, bwd_dtype, contracting_dims, ctx, grad):    # pylint: disable=unused-argument
    lhs_contracting_dims, rhs_contracting_dims = contracting_dims

    casted_x, casted_kernel, fp8_max, amax, scale, scale_inv, \
        updated_x_amax, updated_kernel_amax, x_shape, kernel_shape = ctx

    gemm_x_idx, gemm_kernel_idx, gemm_grad_idx = FP8Helper.get_fp8_meta_indices(0)

    grad_amax = amax[gemm_grad_idx, 0:1]
    grad_scale = scale[gemm_grad_idx]
    grad_scale_inv = scale_inv[gemm_grad_idx]

    casted_grad, casted_grad_t, updated_grad_amax = \
        cast_transpose(grad, grad_amax, grad_scale, grad_scale_inv,
                       bwd_dtype, static_axis_boundary=-1,
                       transpose_axis_boundary=min(lhs_contracting_dims))

    x_constracting_dim = tuple(range(0, len(x_shape) - len(lhs_contracting_dims)))
    gt_constracting_dim = tuple(range(grad.ndim - len(x_constracting_dim), grad.ndim))
    x_scale_inv = scale_inv[gemm_x_idx]
    wgrad = fp8_dot_impl(casted_x, casted_grad_t, x_scale_inv, grad_scale_inv, grad.dtype,
                         (x_constracting_dim, gt_constracting_dim),
                         get_precision_of_fp8_dot(FP8Helper.FP8_2X_ACC_WGRAD))

    g_constracting_dim = tuple(
        range(grad.ndim - len(kernel_shape) + len(rhs_contracting_dims), grad.ndim))
    k_constracting_dim = tuple(range(len(rhs_contracting_dims), len(kernel_shape)))
    kernel_scale_inv = scale_inv[gemm_kernel_idx]
    dgrad = fp8_dot_impl(casted_grad, casted_kernel, grad_scale_inv, kernel_scale_inv, grad.dtype,
                         (g_constracting_dim, k_constracting_dim),
                         get_precision_of_fp8_dot(FP8Helper.FP8_2X_ACC_DGRAD))

    amax = amax.at[gemm_x_idx, 0].set(updated_x_amax)
    amax = amax.at[gemm_kernel_idx, 0].set(updated_kernel_amax)
    amax = amax.at[gemm_grad_idx, 0].set(updated_grad_amax[0])

    scale, scale_inv = FP8Helper.update_fp8_scale(fp8_max, amax, scale)

    return dgrad, wgrad, fp8_max, amax, scale, scale_inv


_fp8_dot.defvjp(_fp8_dot_fwd_rule, _fp8_dot_bwd_rule)
