# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""JAX te modules"""

from typing import List, Tuple, Sequence
from functools import partial
import jax
import jax.numpy as jnp

from . import cpp_extensions as tex
from .fp8 import FP8Helper, FP8MetaPackage

Precision = jax.lax.Precision


def type_safe_dot_general(
    x,
    kernel,
    fp8_meta_pkg: FP8MetaPackage = None,
    contracting_dims: Tuple[Sequence[int], Sequence[int]] = ((1,), (0,)),
) -> jnp.ndarray:
    """
    Type safe dot_general, including FP8.
    """

    if fp8_meta_pkg is None:
        kernel = jnp.asarray(kernel, x.dtype)
        return jax.lax.dot_general(x, kernel, (contracting_dims, ((), ())))

    amax_list = fp8_meta_pkg.amax_list
    scale_list = fp8_meta_pkg.scale_list
    fwd_dtype = FP8Helper.FWD_DTYPE
    bwd_dtype = FP8Helper.BWD_DTYPE
    return _fp8_dot(x, kernel, amax_list, scale_list, fwd_dtype, bwd_dtype, contracting_dims)


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
    ctype: jnp.dtype,  # computing type
    contracting_dims: Tuple[Sequence[int], Sequence[int]],
    precision: Precision = None,
):
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


@partial(jax.custom_vjp, nondiff_argnums=(4, 5, 6))
def _fp8_dot(
    x: jnp.ndarray,
    kernel: jnp.ndarray,
    amax_list: List[jnp.ndarray],
    scale_list: List[jnp.ndarray],
    fwd_dtype: jnp.dtype,
    bwd_dtype: jnp.dtype,
    contracting_dims: Tuple[Sequence[int], Sequence[int]],
):
    output, _ = _fp8_dot_fwd_rule(
        x, kernel, amax_list, scale_list, fwd_dtype, bwd_dtype, contracting_dims
    )
    return output


def _fp8_dot_fwd_rule(
    x,
    kernel,
    amax_list,
    scale_list,
    fwd_dtype,
    bwd_dtype,  # pylint: disable=unused-argument
    contracting_dims,
):

    maybe_fm32_to_fp32, maybe_fp32_to_fm32 = FP8Helper.generate_fp8_meta_dtype_converter_pair(
        *amax_list, *scale_list
    )
    amax_list = maybe_fm32_to_fp32(*amax_list)
    scale_list = maybe_fm32_to_fp32(*scale_list)

    lhs_contracting_dims, rhs_contracting_dims = contracting_dims

    x_shape_suf = x.shape[min(lhs_contracting_dims) :]
    kernel_shape_pre = kernel.shape[: max(rhs_contracting_dims) + 1]
    assert x_shape_suf == kernel_shape_pre

    fp8_dtype_list = [fwd_dtype, fwd_dtype, bwd_dtype]
    scale_list, scale_inv_list = FP8MetaPackage.update_fp8_scale(
        amax_list, scale_list, fp8_dtype_list
    )
    amax_list = FP8MetaPackage.update_amax_list(amax_list)

    x_scale = scale_list[FP8MetaPackage.INPUT_IDX]
    x_scale_inv = scale_inv_list[FP8MetaPackage.INPUT_IDX]
    # Note (Ming Huang): Use native cast to allow XLA handle tranpose for avoiding
    # unnecessary copy to break FP8 GEMM pattern matching.
    casted_x, updated_x_amax = quantize(x, fwd_dtype, x_scale)

    kernel_scale = scale_list[FP8MetaPackage.WEIGHT_IDX]
    kernel_scale_inv = scale_inv_list[FP8MetaPackage.WEIGHT_IDX]
    # Note (Ming Huang): Use native cast to allow XLA handle tranpose for avoiding
    # unnecessary copy to break FP8 GEMM pattern matching.
    casted_kernel, updated_kernel_amax = quantize(kernel, fwd_dtype, kernel_scale)

    output = fp8_dot_impl(
        casted_x,
        casted_kernel,
        x_scale_inv,
        kernel_scale_inv,
        x.dtype,
        (lhs_contracting_dims, rhs_contracting_dims),
        get_precision_of_fp8_dot(FP8Helper.FP8_2X_ACC_FPROP),
    )

    ctx = (
        casted_x,
        casted_kernel,
        amax_list,
        scale_list,
        scale_inv_list,
        updated_x_amax,
        updated_kernel_amax,
        x.shape,
        kernel.shape,
        maybe_fp32_to_fm32,
    )
    return output, ctx


def _fp8_dot_bwd_rule(
    fwd_dtype, bwd_dtype, contracting_dims, ctx, grad
):  # pylint: disable=unused-argument
    lhs_contracting_dims, rhs_contracting_dims = contracting_dims

    (
        casted_x,
        casted_kernel,
        amax_list,
        scale_list,
        scale_inv_list,
        updated_x_amax,
        updated_kernel_amax,
        x_shape,
        kernel_shape,
        maybe_fp32_to_fm32,
    ) = ctx

    grad_amax = amax_list[FP8MetaPackage.GRAD_IDX][0:1]
    grad_scale = scale_list[FP8MetaPackage.GRAD_IDX]
    grad_scale_inv = scale_inv_list[FP8MetaPackage.GRAD_IDX]

    casted_grad, casted_grad_t, updated_grad_amax = tex.cast_transpose(
        grad,
        grad_amax,
        grad_scale,
        grad_scale_inv,
        bwd_dtype,
        static_axis_boundary=-1,
        transpose_axis_boundary=min(lhs_contracting_dims),
    )

    x_constracting_dim = tuple(range(0, len(x_shape) - len(lhs_contracting_dims)))
    gt_constracting_dim = tuple(range(grad.ndim - len(x_constracting_dim), grad.ndim))
    x_scale_inv = scale_inv_list[FP8MetaPackage.INPUT_IDX]
    wgrad = fp8_dot_impl(
        casted_x,
        casted_grad_t,
        x_scale_inv,
        grad_scale_inv,
        grad.dtype,
        (x_constracting_dim, gt_constracting_dim),
        get_precision_of_fp8_dot(FP8Helper.FP8_2X_ACC_WGRAD),
    )

    g_constracting_dim = tuple(
        range(grad.ndim - len(kernel_shape) + len(rhs_contracting_dims), grad.ndim)
    )
    k_constracting_dim = tuple(range(len(rhs_contracting_dims), len(kernel_shape)))
    kernel_scale_inv = scale_inv_list[FP8MetaPackage.WEIGHT_IDX]
    dgrad = fp8_dot_impl(
        casted_grad,
        casted_kernel,
        grad_scale_inv,
        kernel_scale_inv,
        grad.dtype,
        (g_constracting_dim, k_constracting_dim),
        get_precision_of_fp8_dot(FP8Helper.FP8_2X_ACC_DGRAD),
    )

    amax_list[FP8MetaPackage.INPUT_IDX] = (
        amax_list[FP8MetaPackage.INPUT_IDX].at[0].set(updated_x_amax)
    )
    amax_list[FP8MetaPackage.WEIGHT_IDX] = (
        amax_list[FP8MetaPackage.WEIGHT_IDX].at[0].set(updated_kernel_amax)
    )
    amax_list[FP8MetaPackage.GRAD_IDX] = (
        amax_list[FP8MetaPackage.GRAD_IDX].at[0].set(updated_grad_amax[0])
    )

    amax_list = maybe_fp32_to_fm32(*amax_list)
    scale_list = maybe_fp32_to_fm32(*scale_list)

    return dgrad, wgrad, amax_list, scale_list


_fp8_dot.defvjp(_fp8_dot_fwd_rule, _fp8_dot_bwd_rule)
