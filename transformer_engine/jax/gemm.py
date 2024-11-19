# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
import operator
from functools import partial, reduce
from typing import Optional, Tuple, Union

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from .fp8 import FP8Helper, FP8MetaPackage
from .cpp_extensions import (
    gemm_impl,
    fp8_gemm_impl,
    cast_transpose,
    dact_lu,
    dbias_cast_transpose,
    dact_lu_dbias_cast_transpose,
)
from .cpp_extensions.gemm import sanitize_dims


__all__ = [
    "gemm",
    "fp8_gemm",
    "type_safe_gemm",
]


def gemm(
    x: ArrayLike,
    kernel: ArrayLike,
    bias: Optional[ArrayLike] = None,
    contracting_dims: Tuple[int, int] = (1, 0),
    fuse_gelu: bool = False,
    accumulate: bool = False,
    use_split_accumulator: bool = False,
) -> ArrayLike:
    """Non-FP8 collective/distributed `nvte_cublas_gemm()` with GELU and bias-add fusions."""
    return _gemm(x, kernel, bias, contracting_dims, fuse_gelu, accumulate, use_split_accumulator)


@partial(jax.custom_vjp, nondiff_argnums=(3, 4, 5, 6))
def _gemm(
    x: ArrayLike,
    kernel: ArrayLike,
    bias: Union[ArrayLike, None],
    contracting_dims: Tuple[int, int],
    fuse_gelu: bool,
    accumulate: bool,
    use_split_accumulator: bool,
) -> ArrayLike:
    out, _ = _gemm_fwd_rule(
        x, kernel, bias, contracting_dims, fuse_gelu, accumulate, use_split_accumulator
    )
    return out


def _gemm_fwd_rule(
    x: ArrayLike,
    kernel: ArrayLike,
    bias: ArrayLike,
    contracting_dims: Tuple[int, int],
    fuse_gelu: bool,
    accumulate: bool,
    use_split_accumulator: bool,
) -> Tuple[ArrayLike, ...]:
    assert (
        kernel.ndim == 2
    ), "TE/JAX Collective GEMM custom op does not support batched RHS operand in forward mode."

    fuse_bias = bias is not None

    out, pre_gelu_out = gemm_impl(
        x,
        kernel,
        bias=bias,
        contracting_dims=contracting_dims,
        fuse_gelu=fuse_gelu,
        fuse_bias=fuse_bias,
        accumulate=accumulate,
        use_split_accumulator=use_split_accumulator,
    )

    ctx = (
        x,
        kernel,
        pre_gelu_out if fuse_gelu else None,
        fuse_bias,
    )

    return out, ctx


def _gemm_bwd_rule(
    contracting_dims,
    fuse_gelu,
    accumulate,
    use_split_accumulator,
    ctx,
    grad,
):
    x, kernel, pre_gelu_out, fuse_bias = ctx
    x_inner_dim, kernel_inner_dim = map(sanitize_dims, contracting_dims, (x.ndim, kernel.ndim))
    x_outer_dim = x.ndim - 1 if x_inner_dim != x.ndim - 1 else x.ndim - 2
    kernel_outer_dim = kernel.ndim - 2 if kernel_inner_dim == kernel.ndim - 1 else kernel.ndim - 1

    # DGRAD: ([B], M, N) x (K, N)^T = ([B], M, K)
    dgrad, dgelu, _ = gemm_impl(
        grad,
        kernel,
        gelu_input=pre_gelu_out,
        contracting_dims=(-1, kernel_outer_dim),
        fuse_gelu=fuse_gelu,
        fuse_bias=False,
        grad=True,
        accumulate=accumulate,
        use_split_accumulator=use_split_accumulator,
    )

    # WGRAD: ([B], M, K)^T x ([B], M, N) = (K, N)
    wgrad_rhs = dgelu if fuse_gelu else grad
    wgrad, _, bgrad = gemm_impl(
        x,
        wgrad_rhs,
        gelu_input=pre_gelu_out,
        contracting_dims=(x_outer_dim, wgrad_rhs.ndim - 2),
        fuse_gelu=False,
        fuse_bias=fuse_bias,
        grad=True,
        accumulate=accumulate,
        use_split_accumulator=use_split_accumulator,
    )

    if not fuse_bias:
        bgrad = None

    return dgrad, wgrad, bgrad


_gemm.defvjp(_gemm_fwd_rule, _gemm_bwd_rule)


def fp8_gemm(
    x: ArrayLike,
    kernel_t: ArrayLike,
    fp8_meta: FP8MetaPackage,
    bias: Optional[ArrayLike] = None,
    out_dtype: jnp.dtype = jnp.bfloat16,
    fuse_gelu: bool = False,
    accumulate: bool = False,
    use_split_accumulator: bool = False,
) -> ArrayLike:
    """Non-FP8 `nvte_cublas_gemm()` with optional GELU and bias-add fusions."""
    return _fp8_gemm(
        x,
        kernel_t,
        bias,
        fp8_meta.amax_list,
        fp8_meta.scale_list,
        out_dtype,
        fuse_gelu,
        accumulate,
        use_split_accumulator,
    )


@partial(jax.custom_vjp, nondiff_argnums=(5, 6, 7, 8, 9))
def _fp8_gemm(
    x: ArrayLike,
    kernel_t: ArrayLike,
    bias: ArrayLike,
    amax_list: ArrayLike,
    scale_list: ArrayLike,
    out_dtype: jnp.dtype,
    fuse_gelu: bool,
    accumulate: bool,
    use_split_accumulator: bool,
) -> ArrayLike:
    out, _ = _fp8_gemm_fwd_rule(
        x,
        kernel_t,
        bias,
        amax_list,
        scale_list,
        out_dtype,
        fuse_gelu,
        accumulate,
        use_split_accumulator,
    )
    return out


def _fp8_gemm_fwd_rule(
    x: ArrayLike,
    kernel_t: ArrayLike,
    bias: ArrayLike,
    amax_list: ArrayLike,
    scale_list: ArrayLike,
    out_dtype: jnp.dtype,
    fuse_gelu: bool,
    accumulate: bool,
    use_split_accumulator: bool,
) -> Tuple[ArrayLike, ...]:
    assert (
        kernel_t.ndim == 2
    ), "TE/JAX Collective GEMM custom op does not support batched RHS operand in forward mode."

    fuse_bias = bias is not None

    maybe_fm32_to_fp32, maybe_fp32_to_fm32 = FP8Helper.generate_fp8_meta_dtype_converter_pair(
        *amax_list,
        *scale_list,
    )
    amax_list = maybe_fm32_to_fp32(*amax_list)
    scale_list = maybe_fm32_to_fp32(*scale_list)

    fwd_dtype = FP8Helper.FWD_DTYPE
    bwd_dtype = FP8Helper.BWD_DTYPE
    fp8_dtype_list = [fwd_dtype, fwd_dtype, bwd_dtype, fwd_dtype]
    scale_list, scale_inv_list = FP8MetaPackage.update_fp8_scale(
        amax_list, scale_list, fp8_dtype_list
    )
    amax_list = FP8MetaPackage.update_amax_list(amax_list)

    x_amax = amax_list[FP8MetaPackage.INPUT_IDX][0:1]
    x_scale = scale_list[FP8MetaPackage.INPUT_IDX]
    x_scale_inv = scale_inv_list[FP8MetaPackage.INPUT_IDX]
    if x.dtype not in [jnp.float8_e4m3fn, jnp.float8_e5m2]:
        casted_x, casted_x_t, updated_x_amax = cast_transpose(
            x,
            x_amax,
            x_scale,
            x_scale_inv,
            fwd_dtype,
            static_axis_boundary=-1,
            transpose_axis_boundary=-1,
        )
    else:
        casted_x = x
        casted_x_t = jnp.matrix_transpose(x)
        updated_x_amax = x_amax

    kernel_amax = amax_list[FP8MetaPackage.WEIGHT_IDX][0:1]
    kernel_scale = scale_list[FP8MetaPackage.WEIGHT_IDX]
    kernel_scale_inv = scale_inv_list[FP8MetaPackage.WEIGHT_IDX]
    if kernel_t.dtype not in [jnp.float8_e4m3fn, jnp.float8_e5m2]:
        casted_kernel_t, casted_kernel, updated_kernel_amax = cast_transpose(
            kernel_t,
            kernel_amax,
            kernel_scale,
            kernel_scale_inv,
            fwd_dtype,
            static_axis_boundary=-1,
            transpose_axis_boundary=-1,
        )
    else:
        casted_kernel = jnp.matrix_transpose(kernel_t)
        casted_kernel_t = kernel_t
        updated_kernel_amax = kernel_amax

    out_amax = (
        amax_list[FP8MetaPackage.OUTPUT_IDX][0:1]
        if out_dtype in [jnp.float8_e4m3fn, jnp.float8_e5m2]
        else None
    )
    out_scale = (
        scale_list[FP8MetaPackage.OUTPUT_IDX][0:1]
        if out_dtype in [jnp.float8_e4m3fn, jnp.float8_e5m2]
        else None
    )
    out, updated_out_amax, updated_out_scale, pre_gelu_out = fp8_gemm_impl(
        casted_x,
        x_scale_inv,
        casted_kernel_t,
        kernel_scale_inv,
        bias=bias,
        out_amax=out_amax,
        out_scale=out_scale,
        out_dtype=out_dtype,
        fuse_gelu=fuse_gelu,
        fuse_bias=fuse_bias,
        accumulate=accumulate,
        use_split_accumulator=use_split_accumulator,
    )
    if out_dtype not in [jnp.float8_e4m3fn, jnp.float8_e5m2]:
        updated_out_amax = None
        updated_out_scale = None

    ctx = (
        casted_x_t,
        casted_kernel,
        amax_list,
        scale_list,
        scale_inv_list,
        updated_x_amax,
        updated_kernel_amax,
        updated_out_amax,
        pre_gelu_out if fuse_gelu else None,
        fuse_bias,
        maybe_fp32_to_fm32,
    )

    return (out, updated_out_scale), ctx


def _fp8_gemm_bwd_rule(
    out_dtype,
    fuse_gelu,
    accumulate,
    use_split_accumulator,
    ctx,
    grad,
):
    (
        casted_x_t,
        casted_kernel,
        amax_list,
        scale_list,
        scale_inv_list,
        updated_x_amax,
        updated_kernel_amax,
        updated_out_amax,
        pre_gelu_out,
        fuse_bias,
        maybe_fp32_to_fm32,
    ) = ctx

    bwd_dtype = FP8Helper.BWD_DTYPE

    grad_amax = amax_list[FP8MetaPackage.GRAD_IDX][0:1]
    grad_scale = scale_list[FP8MetaPackage.GRAD_IDX]
    grad_scale_inv = scale_inv_list[FP8MetaPackage.GRAD_ID]
    if fuse_gelu:
        if fuse_bias:
            # Fuse dbias into this dGELU.
            casted_grad, casted_grad_t, bgrad, updated_grad_amax = dact_lu_dbias_cast_transpose(
                grad,
                pre_gelu_out,
                grad_amax,
                grad_scale,
                grad_scale_inv,
                bwd_dtype,
                static_axis_boundary=-1,
                transpose_axis_boundary=-1,
                activation_type=("gelu",),
            )
        else:
            # No bias to fuse so we just do dGELU.
            casted_grad, casted_grad_t, updated_grad_amax = dact_lu(grad, pre_gelu_out, ("gelu",))
            bgrad = None
    else:
        if fuse_bias:
            # Since there is no GELU fusion, we need to fuse dbias into this cast_transpose.
            casted_grad, casted_grad_t, bgrad, updated_grad_amax = dbias_cast_transpose(
                grad,
                grad_amax,
                grad_scale,
                grad_scale_inv,
                bwd_dtype,
                static_axis_boundary=-1,
                transpose_axis_boundary=-1,
            )
        else:
            # If both bias and GELU is fused into the forward pass, we will fuse dbias later with
            # dGELU. No need to do it here.
            casted_grad, casted_grad_t, updated_grad_amax = cast_transpose(
                grad,
                grad_amax,
                grad_scale,
                grad_scale_inv,
                bwd_dtype,
                static_axis_boundary=-1,
                transpose_axis_boundary=-1,
            )
            bgrad = None

    kernel_scale_inv = scale_inv_list[FP8MetaPackage.WEIGHT_IDX]
    dgrad, *_ = fp8_gemm_impl(
        casted_grad,
        grad_scale_inv,
        casted_kernel,
        kernel_scale_inv,
        accumulate=accumulate,
        use_split_accumulator=use_split_accumulator,
    )

    x_scale_inv = scale_inv_list[FP8MetaPackage.INPUT_IDX]
    wgrad, *_ = fp8_gemm_impl(
        casted_x_t,
        x_scale_inv,
        casted_grad_t,
        grad_scale_inv,
        accumulate=accumulate,
        use_split_accumulator=use_split_accumulator,
    )

    amax_list[FP8MetaPackage.INPUT_IDX] = (
        amax_list[FP8MetaPackage.INPUT_IDX].at[0].set(updated_x_amax[0])
    )
    amax_list[FP8MetaPackage.WEIGHT_IDX] = (
        amax_list[FP8MetaPackage.WEIGHT_IDX].at[0].set(updated_kernel_amax[0])
    )
    amax_list[FP8MetaPackage.GRAD_IDX] = (
        amax_list[FP8MetaPackage.GRAD_IDX].at[0].set(updated_grad_amax[0])
    )
    if out_dtype in [jnp.float8_e4m3fn, jnp.float8_e5m2]:
        amax_list[FP8MetaPackage.OUTPUT_IDX] = (
            amax_list[FP8MetaPackage.OUTPUT_IDX].at[0].set(updated_out_amax[0])
        )

    amax_list = maybe_fp32_to_fm32(*amax_list)
    scale_list = maybe_fp32_to_fm32(*scale_list)

    return dgrad, wgrad, bgrad, amax_list, scale_list


_fp8_gemm.defvjp(_fp8_gemm_fwd_rule, _fp8_gemm_bwd_rule)


def type_safe_gemm(
    x: ArrayLike,
    kernel: ArrayLike,
    bias: Optional[ArrayLike] = None,
    fp8_meta: Optional[FP8MetaPackage] = None,
    out_dtype: Optional[jnp.dtype] = None,
    contracting_dims: Tuple[int, int] = (1, 0),
    fuse_gelu: bool = False,
    accumulate: bool = False,
    use_split_accumulator: bool = False,
) -> ArrayLike:
    if x.dtype in [jnp.float8_e4m3fn, jnp.float8_e5m2] or kernel.dtype in [
        jnp.float8_e4m3fn,
        jnp.float8_e5m2,
    ]:
        assert fp8_meta is not None, "GEMM operands have FP8 dtypes but FP8MetaPackage is None."

    if fp8_meta is not None:
        x_inner_dim, kernel_inner_dim = map(sanitize_dims, contracting_dims, (x.ndim, kernel.ndim))
        assert x_inner_dim == x.ndim - 1 and kernel_inner_dim == kernel.ndim - 1, (
            "FP8 GEMM requires non-transposed X (LHS) and transposed kernel (RHS), "
            + "i.e. contracting_dims=(-1, -1)."
        )
        return fp8_gemm(
            x,
            kernel,
            bias,
            fp8_meta,
            out_dtype,
            fuse_gelu,
            accumulate,
            use_split_accumulator,
        )
    else:
        return gemm(x, kernel, bias, contracting_dims, fuse_gelu, accumulate, use_split_accumulator)
