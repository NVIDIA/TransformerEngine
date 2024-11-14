# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""JAX/TE custom ops for cuBlasLt GEMM"""
import warnings
import operator
from functools import reduce
from typing import Optional, Union, Tuple

import jax
import jax.numpy as jnp
from jax import dtypes
from jax.interpreters import mlir
from jax.interpreters.mlir import ir
from jax.sharding import PartitionSpec, NamedSharding
from jax.extend import ffi
from jax.typing import ArrayLike

from transformer_engine import transformer_engine_jax as tex
from .base import BasePrimitive, register_primitive
from .custom_call import custom_caller, CustomCallArgsWrapper
from .misc import (
    jax_dtype_to_te_dtype,
    jax_dtype_is_fp8,
    get_padded_spec,
    is_ffi_enabled,
    check_valid_batch_dims,
)
from ..sharding import (
    global_mesh_resource,
    lax_paral_op,
    all_reduce_max_along_all_axes_except_PP,
)


__all__ = [
    "fp8_gemm_impl",
    "gemm_impl",
]


def get_cublas_workspace_size_bytes() -> None:
    """Return 32 MiB if using hopper, 4 MiB for all other architectures."""
    if tex.get_device_compute_capability() >= 90:
        return 33_554_432
    return 4_194_304


class CollectiveGemmPrimitive(BasePrimitive):
    """
    cuBlasLt GEMM Primitive w/ support for distributed inputs
    """

    name = "te_gemm"
    impl_static_args = (8, 9, 10, 11, 12, 13, 14)
    multiple_results = True
    inner_primitive = None
    outer_primitive = None

    @staticmethod
    def abstract(lhs_aval, lhs_scale_inv_aval, rhs_aval, rhs_scale_inv_aval, bias_aval,
                 gelu_input_aval, out_amax_aval, out_scale_aval, out_dtype, contracting_dims,
                 fuse_gelu, fuse_bias, grad, accumulate, use_split_accumulator):
        """
        cuBlasLt GEMM abstract
        """
        del grad, accumulate, use_split_accumulator

        # Validate operand dtypes
        lhs_dtype = dtypes.canonicalize_dtype(lhs_aval.dtype)
        rhs_dtype = dtypes.canonicalize_dtype(rhs_aval.dtype)
        assert lhs_dtype == rhs_dtype, "Mismatched matrix dtypes for GEMM."
        is_fp8 = False
        if jax_dtype_is_fp8(lhs_dtype):
            assert (
                lhs_scale_inv_aval.size == 1
                and dtypes.canonicalize_dtype(lhs_scale_inv_aval.dtype) == jnp.float32
            ), "Missing LHS operand scale inverse in FP8 GEMM."
            is_fp8 = True
        if jax_dtype_is_fp8(rhs_dtype):
            assert (
                rhs_scale_inv_aval.size == 1
                and dtypes.canonicalize_dtype(rhs_scale_inv_aval.dtype) == jnp.float32
            ), "Missing RHS operand scale inverse in FP8 GEMM."

        # Validate operand layouts
        lhs_inner_dim, rhs_inner_dim = map(
            lambda inner_dim, ndims: (ndims - inner_dim) if inner_dim < 0 else inner_dim,
            contracting_dims,
            (lhs_aval.ndim, rhs_aval.ndim)
        )
        assert (
            lhs_aval.shape[lhs_inner_dim] == rhs_aval.shape[rhs_inner_dim]
        ), f"Incompatible operand sizes: {lhs_aval.shape} x {rhs_aval.shape}."

        lhs_trans = lhs_inner_dim != lhs_aval.ndim - 1
        rhs_trans = rhs_inner_dim == rhs_aval.ndim - 1
        assert (
            not (lhs_trans and rhs_trans)
        ), "GEMM does not support transposed LHS and transposed RHS at the same time."
        if is_fp8:
            assert not lhs_trans, "FP8 GEMM does not support transposed LHS."
            assert rhs_trans, "FP8 GEMM requires transposed RHS."

        # Validate output dtype
        if jax_dtype_is_fp8(out_dtype):
            assert (
                jax_dtype_is_fp8(lhs_dtype) and jax_dtype_is_fp8(rhs_dtype)
            ), "FP8 GEMM output requires FP8 inputs."
            assert (
                out_amax_aval.size == out_scale_aval.size == 1
            ), "Invalid/missing output amax and scale."
            out_amax_updated_dtype = dtypes.canonicalize_dtype(out_amax_aval.dtype)
            out_scale_updated_dtype = dtypes.canonicalize_dtype(out_scale_aval.dtype)
            assert (
                out_amax_updated_dtype == out_scale_updated_dtype == jnp.float32
            ), "Invalid output amax or scale dtype."
        else:
            out_dtype = lhs_dtype
            out_amax_updated_dtype = jnp.float32
            out_scale_updated_dtype = jnp.float32

        # Infer output shape
        lhs_outer_dim = lhs_aval.ndim - 1 if lhs_trans else lhs_aval.ndim - 2
        lhs_bdims = [dim for dim in range(lhs_aval.ndim)
                     if dim not in [lhs_outer_dim, lhs_inner_dim]]
        lhs_batch_shape = [lhs_aval.shape[dim] for dim in lhs_bdims]
        lhs_batch_size = reduce(operator.mul, lhs_batch_shape, 1)
        rhs_outer_dim = rhs_aval.ndim - 2 if rhs_trans else rhs_aval.ndim - 1
        rhs_bdims = [dim for dim in range(rhs_aval.ndim)
                     if dim not in [rhs_outer_dim, rhs_inner_dim]]
        rhs_batch_size = reduce(operator.mul, rhs_bdims, 1)
        assert (
            lhs_batch_size == rhs_batch_size
        ), "LHS and RHS operands must have the same batched sizes."
        out_shape = (*lhs_batch_shape, lhs_aval.shape[lhs_outer_dim], rhs_aval.shape[rhs_outer_dim])

        # Validate bias/bias_grad shape against inferred output
        bias_dtype = jnp.bfloat16 if jax_dtype_is_fp8(out_dtype) else out_dtype
        if fuse_bias:
            assert (
                bias_aval.size > 0
                and bias_aval.ndim == 1
                and bias_aval.shape[0] == out_shape[-1]
            ), "Incorrect bias shape."
            bias_dtype = dtypes.canonicalize_dtype(bias_aval.dtype)
        else:
            assert bias_aval.size == 0, "Internal TE error."

        # Validate GELU input/output
        if fuse_gelu:
            assert (
                all([gelu_input_aval.shape[i] == out_shape[i] for i in len(out_shape)])
            ), "Invalid GELU input shape."
            assert gelu_input_aval.dtype == bias_dtype, "Invalid GELU dtype."
        else:
            assert gelu_input_aval.size == 0, "Internal TE error."

        # Create abstract arrays for all outputs
        out_aval = lhs_aval.update(shape=out_shape, dtype=out_dtype)
        out_amax_updated_aval = out_amax_aval.update(shape=out_amax_aval.shape,
                                                     dtype=out_amax_updated_dtype)
        out_scale_updated_aval = out_scale_aval.update(shape=out_scale_aval.shape,
                                                       dtype=out_scale_updated_dtype)
        pre_gelu_out_aval = gelu_input_aval.update(shape=gelu_input_aval.shape, dtype=bias_dtype)
        bias_grad_aval = bias_aval.update(shape=bias_aval.shape, dtype=bias_dtype)
        workspace_aval = jax.core.ShapedArray(shape=(get_cublas_workspace_size_bytes(), ),
                                              dtype=jnp.uint8)

        return (
            out_aval,
            out_amax_updated_aval,
            out_scale_updated_aval,
            pre_gelu_out_aval,
            bias_grad_aval,
            workspace_aval
        )

    @staticmethod
    def outer_abstract(*args, **kwargs):
        """
        cuBlasLt GEMM outer abstract
        """
        (
            out_aval,
            out_amax_aval,
            out_scale_aval,
            pre_gelu_out_aval,
            bias_grad_aval,
            _
        ) = CollectiveGemmPrimitive.abstract(*args, **kwargs)
        return out_aval, out_amax_aval, out_scale_aval, pre_gelu_out_aval, bias_grad_aval

    @staticmethod
    def lowering(ctx, lhs, lhs_scale_inv, rhs, rhs_scale_inv, bias, gelu_input, out_amax, out_scale,
                 *, out_dtype, contracting_dims, fuse_gelu, fuse_bias, grad, accumulate,
                 use_split_accumulator):
        """
        Fused attention fwd lowering rules
        """
        lhs_aval, _, rhs_aval, _, bias_aval, *_ = ctx.avals_in
        lhs_inner_dim, rhs_inner_dim = map(
            lambda inner_dim, ndims: (ndims - inner_dim) if inner_dim < 0 else inner_dim,
            contracting_dims,
            (lhs_aval.ndim, rhs_aval.ndim)
        )
        lhs_trans = lhs_inner_dim != lhs_aval.ndim - 1
        rhs_trans = rhs_inner_dim == rhs_aval.ndim - 1

        operand_output_aliases = {
            4: 4,  # bias        <-->  bias_grad
            5: 3,  # gelu_input  <-->  pre_gelu_out
            6: 1,  # out_amax    <-->  out_amax_updated
            7: 2,  # out_scale   <-->  out_scale_updated
        }

        if is_ffi_enabled():
            name = "te_gemm_ffi"
            return ffi.ffi_lowering(name, operand_output_aliases=operand_output_aliases)(
                ctx,
                lhs,
                lhs_scale_inv,
                rhs,
                rhs_scale_inv,
                bias,
                gelu_input,
                out_amax,
                out_scale,
                lhs_trans=lhs_trans,
                rhs_trans=rhs_trans,
                fuse_gelu=fuse_gelu,
                fuse_bias=fuse_bias,
                grad=grad,
                accumulate=accumulate,
                use_split_accumulator=use_split_accumulator
            )
        else:
            operands = [
                lhs,
                lhs_scale_inv,
                rhs,
                rhs_scale_inv,
                bias,
                gelu_input,
                out_amax,
                out_scale,
            ]
            operand_shapes = map(lambda x: ir.RankedTensorType(x.type).shape, operands)
            out_types = [
                ir.RankedTensorType.get(output.shape, mlir.dtype_to_ir_dtype(output.dtype))
                for output in ctx.avals_out
            ]
            args = CustomCallArgsWrapper(out_types, operands, operand_shapes)

            lhs_outer_dim = lhs_aval.ndim - 1 if lhs_trans else lhs_aval.ndim - 2
            rhs_outer_dim = rhs_aval.ndim - 2 if rhs_trans else rhs_aval.ndim - 1
            m = lhs_aval.shape[lhs_outer_dim]
            k = rhs_aval.shape[rhs_inner_dim]
            n = rhs_aval.shape[rhs_outer_dim]
            workspace_size = get_cublas_workspace_size_bytes()
            operand_dtype = jax_dtype_to_te_dtype(lhs_aval.dtype)
            bias_dtype = jax_dtype_to_te_dtype(bias_aval.dtype)
            opaque = tex.pack_gemm_descriptor(m, n, k, workspace_size, operand_dtype,
                                              jax_dtype_to_te_dtype(out_dtype), bias_dtype,
                                              lhs_trans, rhs_trans, fuse_gelu, fuse_bias, grad,
                                              accumulate, use_split_accumulator)

            return custom_caller(
                CollectiveGemmPrimitive.name,
                args,
                opaque,
                has_side_effect=False,
                operand_output_aliases=operand_output_aliases,
            )

    @staticmethod
    def impl(lhs, lhs_scale_inv, rhs, rhs_scale_inv, bias, gelu_input, out_amax, out_scale,
             out_dtype, contracting_dims, fuse_gelu, fuse_bias, grad, accumulate,
             use_split_accumulator):
        assert CollectiveGemmPrimitive.inner_primitive is not None

        (
            out,
            out_amax_updated,
            out_scale_updated,
            pre_gelu_out,
            bias_grad,
            _,
        ) = CollectiveGemmPrimitive.inner_primitive.bind(
            lhs,
            lhs_scale_inv,
            rhs,
            rhs_scale_inv,
            bias,
            gelu_input,
            out_amax,
            out_scale,
            out_dtype=out_dtype,
            contracting_dims=contracting_dims,
            fuse_gelu=fuse_gelu,
            fuse_bias=fuse_bias,
            grad=grad,
            accumulate=accumulate,
            use_split_accumulator=use_split_accumulator,
        )
        return out, out_amax_updated, out_scale_updated, pre_gelu_out, bias_grad

    @staticmethod
    def batcher(batched_args, batch_dims, *, out_dtype, contracting_dims, fuse_gelu, fuse_bias, grad,
                accumulate, use_split_accumulator):
        assert CollectiveGemmPrimitive.outer_primitive is not None
        check_valid_batch_dims(batch_dims)
        lhs_bdims, *_, bias_bdims, gelu_input_bdims, out_amax_bdims, out_scale_bdims = batch_dims

         # FP8 GEMM only supports non-transposed LHS and transposed RHS
        lhs, _, rhs, *_ = batched_args
        lhs_trans = contracting_dims[0] != lhs.ndim - 1
        rhs_trans = contracting_dims[1] == rhs.ndim - 1
        lhs = jnp.matrix_transpose(lhs) if lhs_trans and jax_dtype_is_fp8(lhs.dtype) else lhs
        rhs = jnp.matrix_transpose(rhs) if not rhs_trans and jax_dtype_is_fp8(rhs.dtype) else rhs
        contracting_dims = (1, 1)

        return (
            CollectiveGemmPrimitive.outer_primitive.bind(
                lhs,
                batched_args[1],
                rhs,
                *batched_args[3:],
                out_dtype=out_dtype,
                contracting_dims=contracting_dims,
                fuse_gelu=fuse_gelu,
                fuse_bias=fuse_bias,
                grad=grad,
                accumulate=accumulate,
                use_split_accumulator=use_split_accumulator,
            )
            (lhs_bdims, out_amax_bdims, out_scale_bdims, gelu_input_bdims, bias_bdims)
        )

    @staticmethod
    def infer_sharding_from_operands(out_dtype, contracting_dims, fuse_gelu, fuse_bias, grad,
                                     accumulate, use_split_accumulator, mesh, arg_infos,
                                     result_infos):
        del out_dtype, accumulate, use_split_accumulator, result_infos
        lhs, _, rhs, *_ = arg_infos
        lhs_spec, rhs_spec = map(get_padded_spec, [lhs, rhs])

        lhs_inner_dim, rhs_inner_dim = map(
            lambda inner_dim, ndims: (ndims - inner_dim) if inner_dim < 0 else inner_dim,
            contracting_dims,
            (lhs.ndim, rhs.ndim)
        )
        if lhs_spec[lhs_inner_dim] != rhs_spec[rhs_inner_dim] and not grad:
            warnings.warn("Forcing the inner dimension of LHS to match the sharding of inner "
                          + "dimension of RHS. This can trigger additional communication if LHS is "
                          + "not already partitioned correctly.")

        lhs_trans = lhs_inner_dim != lhs.ndim - 1
        rhs_trans = rhs_inner_dim == rhs.ndim - 1
        lhs_outer_dim = lhs.ndim - 1 if lhs_trans else lhs.ndim - 2
        rhs_outer_dim = rhs.ndim - 2 if rhs_trans else rhs.ndim - 1
        lhs_bdims = [dim for dim in range(lhs.ndim) if dim not in [lhs_outer_dim, lhs_inner_dim]]
        batch_specs = [lhs_spec[bdim] for bdim in lhs_bdims]
        rhs_outer_spec = rhs_spec[rhs_outer_dim]

        if rhs_spec[rhs_inner_dim] is not None and rhs_outer_spec is not None:
            raise RuntimeError("Both inner and outer dimensions of RHS cannot be sharded.")

        # Outer (sequence) dimension of the GEMM output is always unsharded
        out_spec = [*batch_specs, None, rhs_outer_spec]
        out_sharding = NamedSharding(mesh, PartitionSpec(*out_spec))

        # FP8 metas are always unsharded
        fp8_meta_sharding = NamedSharding(mesh, PartitionSpec(None))

        # Pre-GELU output matches output spec if GELU fusion is turned on, otherwise unsharded
        gelu_spec = out_spec if fuse_gelu else [None]
        gelu_sharding = NamedSharding(mesh, PartitionSpec(*gelu_spec))

        # Bias gradient spec matches outer dimension of output if bias fusion is turned on
        bias_sharding = NamedSharding(mesh, PartitionSpec(rhs_outer_spec if fuse_bias else None))

        return (out_sharding, fp8_meta_sharding, fp8_meta_sharding, gelu_sharding, bias_sharding)

    @staticmethod
    def partition(out_dtype, contracting_dims, fuse_gelu, fuse_bias, grad, accumulate,
                  use_split_accumulator, mesh, arg_infos, result_infos):
        del result_infos
        lhs, _, rhs, *_ = arg_infos
        lhs_spec, rhs_spec = map(get_padded_spec, [lhs, rhs])

        lhs_inner_dim, rhs_inner_dim = map(
            lambda inner_dim, ndims: (ndims - inner_dim) if inner_dim < 0 else inner_dim,
            contracting_dims,
            (lhs.ndim, rhs.ndim)
        )

        lhs_trans = lhs_inner_dim != lhs.ndim - 1
        rhs_trans = rhs_inner_dim == rhs.ndim - 1
        lhs_outer_dim = lhs.ndim - 1 if lhs_trans else lhs.ndim - 2
        rhs_outer_dim = rhs.ndim - 2 if rhs_trans else rhs.ndim - 1
        lhs_bdims = [dim for dim in range(lhs.ndim) if dim not in [lhs_outer_dim, lhs_inner_dim]]
        batch_specs = [lhs_spec[bdim] for bdim in lhs_bdims]
        rhs_outer_spec = rhs_spec[rhs_outer_dim]

        # Force all-gather the outer (sequence) dimension of the LHS operand
        lhs_spec_new = [spec for spec in lhs_spec]
        lhs_spec_new[lhs_outer_dim] = None
        lhs_spec_new[lhs_inner_dim] = rhs_spec[rhs_inner_dim]
        lhs_sharding = NamedSharding(mesh, PartitionSpec(*lhs_spec_new))

        # RHS operand is unchanged, we already enforce that only one dimension can be sharded
        rhs_sharding = NamedSharding(mesh, PartitionSpec(*rhs_spec))

        # Bias is sharded to match outer dimension spec of the RHS operand (also the output)
        bias_sharding = NamedSharding(mesh, PartitionSpec(rhs_outer_spec if fuse_bias else None))

        # FP8 metas are always unsharded
        fp8_meta_sharding = NamedSharding(mesh, PartitionSpec(None))

        # Outer (sequence) dimension of the GEMM output is always unsharded
        out_spec = [*batch_specs, None, rhs_outer_spec]
        out_sharding = NamedSharding(mesh, PartitionSpec(*out_spec))

        # Pre-GELU output matches output spec if GELU fusion is turned on, otherwise unsharded
        gelu_spec = out_spec if fuse_gelu else [None]
        gelu_sharding = NamedSharding(mesh, PartitionSpec(*gelu_spec))

        arg_shardings = (lhs_sharding, fp8_meta_sharding, rhs_sharding, fp8_meta_sharding,
                         bias_sharding, gelu_sharding, fp8_meta_sharding, fp8_meta_sharding)
        out_shardings = (out_sharding, fp8_meta_sharding, fp8_meta_sharding, gelu_sharding,
                         bias_sharding)

        def sharded_impl(lhs, lhs_scale_inv, rhs, rhs_scale_inv, bias, gelu_input, out_amax,
                         out_scale):
            (
                out,
                out_amax_updated,
                out_scale_updated,
                pre_gelu_out,
                bias_grad,
            ) = CollectiveGemmPrimitive.impl(
                lhs,
                lhs_scale_inv,
                rhs,
                rhs_scale_inv,
                bias,
                gelu_input,
                out_amax,
                out_scale,
                out_dtype=out_dtype,
                contracting_dims=contracting_dims,
                fuse_gelu=fuse_gelu,
                fuse_bias=fuse_bias,
                grad=grad,
                accumulate=accumulate,
                use_split_accumulator=use_split_accumulator,
            )

            # FP8 amax reduction
            if jax_dtype_is_fp8(lhs.dtype):
                out_amax_updated = all_reduce_max_along_all_axes_except_PP(out_amax_updated, mesh)

            if rhs_spec[rhs_inner_dim] is not None:
                # GEMM output needs to be all-reduced when the contracting dimension is sharded.
                # If the layer is sequence-parallel, we also need to scatter the output, which we
                # can combine into a reduce-scatter here.
                out = lax_paral_op(out, jax.lax.psum, global_mesh_resource().cp_resource,
                                      mesh)
                if fuse_gelu:
                    pre_gelu_out = lax_paral_op(
                        pre_gelu_out, jax.lax.psum, global_mesh_resource().cp_resource, mesh
                    )

            return out, out_amax_updated, out_scale_updated, pre_gelu_out, bias_grad

        return mesh, sharded_impl, out_shardings, arg_shardings


register_primitive(CollectiveGemmPrimitive)


def fp8_gemm_impl(
    lhs: ArrayLike,
    lhs_scale_inv: ArrayLike,
    rhs: ArrayLike,
    rhs_scale_inv: ArrayLike,
    bias:  Optional[ArrayLike] = None,
    gelu_input: Optional[ArrayLike] = None,
    out_amax:  Optional[ArrayLike] = None,
    out_scale:  Optional[ArrayLike] = None,
    out_dtype: jnp.dtype = jnp.bfloat16,
    contracting_dims: Tuple[int, int] = (1, 1),
    fuse_gelu: bool = False,
    fuse_bias: bool = False,
    accumulate: bool = False,
    use_split_accumulator: bool = False,
) -> Tuple[ArrayLike, ...]:
    """FP8 mat-mul with `nvte_cublas_gemm()` custom op."""
    if out_dtype is not None and jax_dtype_is_fp8(out_dtype):
        assert out_amax is not None and out_scale is not None, "Missing output amax and scale."
    else:
        out_amax = jnp.zeros(0, dtype=jnp.float32)
        out_scale = jnp.zeros(0, dtype=jnp.float32)

    if not fuse_bias:
        bias = jnp.zeros(0, dtype=jnp.bfloat16)
    else:
        assert (
            bias is not None
        ), "Missing bias in forward GEMM when bias epilogue is enabled."

    if not fuse_gelu:
        gelu_input = jnp.zeros(0, dtype=bias.dtype)
    elif gelu_input is None:
        lhs_outer_dim = lhs.ndim - 1 if contracting_dims[0] == 1 else lhs.ndim - 2
        rhs_outer_dim = rhs.ndim - 2 if contracting_dims[1] == 0 else rhs.ndim - 1
        out_shape = (*lhs.shape[:-2], lhs.shape[lhs_outer_dim], rhs.shape[rhs_outer_dim])
        gelu_input = jnp.zeros(out_shape, dtype=bias.dtype)

    out, out_amax, out_scale, pre_gelu_out, _ = CollectiveGemmPrimitive.outer_primitive.bind(
        rhs,
        rhs_scale_inv,
        lhs,
        lhs_scale_inv,
        bias,
        gelu_input,
        out_amax,
        out_scale,
        out_dtype=out_dtype,
        contracting_dims=tuple(reversed(contracting_dims)),
        fuse_gelu=fuse_gelu,
        fuse_bias=fuse_bias,
        grad=False,
        accumulate=accumulate,
        use_split_accumulator=use_split_accumulator,
    )

    return out, out_amax, out_scale, pre_gelu_out


def gemm_impl(
    lhs: ArrayLike,
    rhs: ArrayLike,
    bias:  Optional[ArrayLike] = None,
    gelu_input:  Optional[ArrayLike] = None,
    contracting_dims: Tuple[int, int] = (1, 0),
    fuse_gelu: bool = False,
    fuse_bias: bool = False,
    grad: bool = False,
    accumulate: bool = False,
    use_split_accumulator: bool = False,
) -> Tuple[ArrayLike, ...]:
    """Non-FP8 mat-mul with `nvte_cublas_gemm()` custom op."""
    dummy_fp8_meta = jnp.zeros(0, dtype=jnp.float32)

    lhs_outer_dim = lhs.ndim - 1 if contracting_dims[0] == 1 else lhs.ndim - 2
    rhs_outer_dim = rhs.ndim - 2 if contracting_dims[1] == 0 else rhs.ndim - 1
    out_shape = (*lhs.shape[:-2], lhs.shape[lhs_outer_dim], rhs.shape[rhs_outer_dim])

    if not fuse_bias:
        bias = jnp.zeros(0, dtype=lhs.dtype)
    elif grad:
        bias = jnp.zeros(out_shape[-1], dtype=lhs.dtype)
    else:
        assert (
            bias is not None
        ), "Missing bias in forward GEMM when bias epilogue is enabled."

    if not fuse_gelu:
        gelu_input = jnp.zeros(0, dtype=lhs.dtype)
    elif grad:
        assert (
            gelu_input is not None
        ), "Backward GEMM with dGELU epilogue requires pre-GELU output from forward GEMM."
    elif gelu_input is None:
        gelu_input = jnp.zeros(out_shape, dtype=lhs.dtypes)

    out, _, _, pre_gelu_out, bias_grad = CollectiveGemmPrimitive.outer_primitive.bind(
        lhs,
        dummy_fp8_meta,
        rhs,
        dummy_fp8_meta,
        bias,
        gelu_input,
        dummy_fp8_meta,
        dummy_fp8_meta,
        out_dtype=lhs.dtype,
        contracting_dims=contracting_dims,
        fuse_gelu=fuse_gelu,
        fuse_bias=fuse_bias,
        grad=grad,
        accumulate=accumulate,
        use_split_accumulator=use_split_accumulator,
    )

    if grad:
        return out, pre_gelu_out, bias_grad
    else:
        return out, pre_gelu_out
