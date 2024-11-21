# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
import warnings
import operator
from functools import reduce
from typing import Optional, Tuple
from collections.abc import Iterable

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


def sanitize_dims(dim, ndims):
    return (ndims + dim) if dim < 0 else dim


def mirror_dim(dim, ndims):
    return ndims - 2 if dim == ndims - 1 else ndims - 1


def remove_fsdp_specs(pspecs):
    fsdp_resource = global_mesh_resource().fsdp_resource
    if fsdp_resource is None:
        return list(pspecs).copy()

    new_pspecs = []
    for spec in pspecs:
        if spec is None:
            new_pspecs.append(None)

        elif isinstance(spec, Iterable) and not isinstance(spec, str):
            new_spec = []
            for s in spec:
                if s == fsdp_resource:
                    new_spec.append(None)
                else:
                    new_spec.append(s)

            if len(new_spec) > 1:
                new_pspecs.append(new_spec)
            elif len(new_spec) == 1:
                new_pspecs.append(new_spec[0])
            else:
                new_pspecs.append(None)

        elif isinstance(spec, str):
            if spec == fsdp_resource:
                new_pspecs.append(None)
            else:
                new_pspecs.append(spec)

        else:
            new_pspecs.append(spec)

    assert len(new_pspecs) == len(pspecs), (
        "Length of partition specs changed when removing FSDP sharding!\n"
        + f"Original: {pspecs}\n"
        + f"Filtered: {new_pspecs}\n"
    )

    return new_pspecs


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
    impl_static_args = (8, 9, 10, 11, 12, 13, 14, 15)
    multiple_results = True
    inner_primitive = None
    outer_primitive = None

    @staticmethod
    def abstract(
        lhs_aval,
        lhs_scale_inv_aval,
        rhs_aval,
        rhs_scale_inv_aval,
        bias_aval,
        gelu_input_aval,
        out_amax_aval,
        out_scale_aval,
        out_dtype,
        batched_output,
        contracting_dims,
        fuse_gelu,
        fuse_bias,
        grad,
        accumulate,
        use_split_accumulator,
    ):
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
            sanitize_dims, contracting_dims, (lhs_aval.ndim, rhs_aval.ndim)
        )
        assert (
            lhs_aval.shape[lhs_inner_dim] == rhs_aval.shape[rhs_inner_dim]
        ), f"Incompatible operand sizes: {lhs_aval.shape} x {rhs_aval.shape}."

        lhs_trans = lhs_inner_dim != lhs_aval.ndim - 1
        rhs_trans = rhs_inner_dim == rhs_aval.ndim - 1
        assert not (
            lhs_trans and rhs_trans
        ), "GEMM does not support transposed LHS and transposed RHS at the same time."
        if is_fp8:
            assert not lhs_trans, "FP8 GEMM does not support transposed LHS."
            assert rhs_trans, "FP8 GEMM requires transposed RHS."

        # Validate output dtype
        if jax_dtype_is_fp8(out_dtype):
            assert jax_dtype_is_fp8(lhs_dtype) and jax_dtype_is_fp8(
                rhs_dtype
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

        # Make sure leading dimensions of RHS is broadcast-compatible with LHS
        lhs_outer_dim, rhs_outer_dim = map(
            mirror_dim,
            (lhs_inner_dim, rhs_inner_dim),
            (lhs_aval.ndim, rhs_aval.ndim),
        )
        lhs_bdims = [
            dim for dim in range(lhs_aval.ndim) if dim not in [lhs_outer_dim, lhs_inner_dim]
        ]
        lhs_batch_shape = [lhs_aval.shape[dim] for dim in lhs_bdims]
        lhs_batch_size = reduce(operator.mul, lhs_batch_shape, 1)

        # Infer output shape
        if batched_output:
            assert (
                lhs_aval.ndim > 2 and rhs_aval.ndim == 2
            ), "Batched output requires batched LHS and non-batched RHS operands."
            out_shape = (
                *lhs_batch_shape,
                lhs_aval.shape[lhs_outer_dim],
                rhs_aval.shape[rhs_outer_dim],
            )
        else:
            assert (
                lhs_aval.ndim == rhs_aval.ndim
            ), "Non-batched output requires LHS and RHS operands with same number of dimensions."
            if lhs_aval.ndim > 2:
                rhs_bdims = [
                    dim for dim in range(rhs_aval.ndim) if dim not in [rhs_outer_dim, rhs_inner_dim]
                ]
                rhs_batch_shape = [rhs_aval.shape[dim] for dim in rhs_bdims]
                rhs_batch_size = reduce(operator.mul, rhs_batch_shape, 1)
                assert lhs_batch_size == rhs_batch_size, (
                    f"Leading dimensins of RHS ({rhs_aval.shape=}) is not broadcast-compatible "
                    + f"with the leading dimensions of LHS ({lhs_aval.shape=})."
                )
            out_shape = (lhs_aval.shape[lhs_outer_dim], rhs_aval.shape[rhs_outer_dim])

        # Validate bias/bias_grad shape against inferred output
        bias_dtype = jnp.bfloat16 if jax_dtype_is_fp8(out_dtype) else out_dtype
        if fuse_bias:
            assert (
                bias_aval.size > 0 and bias_aval.ndim == 1 and bias_aval.shape[0] == out_shape[-1]
            ), "Incorrect bias shape."
            bias_dtype = dtypes.canonicalize_dtype(bias_aval.dtype)
        else:
            assert bias_aval.size == 0, "Internal TE error."

        # Validate GELU input/output
        gelu_shape = (0,)
        if fuse_gelu:
            gelu_shape = (
                (reduce(operator.mul, out_shape[:-1], 1), out_shape[-1])
                if len(out_shape) > 2
                else out_shape
            )
            assert gelu_input_aval.ndim == 2 and all(
                [gelu_input_aval.shape[i] == gelu_shape[i] for i in len(gelu_shape)]
            ), "Invalid GELU input shape."
            assert gelu_input_aval.dtype == bias_dtype, "Invalid GELU dtype."
        else:
            assert gelu_input_aval.size == 0, "Internal TE error."

        # Create abstract arrays for all outputs
        out_aval = lhs_aval.update(shape=out_shape, dtype=out_dtype)
        out_amax_updated_aval = out_amax_aval.update(
            shape=out_amax_aval.shape, dtype=out_amax_updated_dtype
        )
        out_scale_updated_aval = out_scale_aval.update(
            shape=out_scale_aval.shape, dtype=out_scale_updated_dtype
        )
        pre_gelu_out_aval = gelu_input_aval.update(shape=gelu_shape, dtype=bias_dtype)
        bias_grad_aval = bias_aval.update(shape=bias_aval.shape, dtype=bias_dtype)
        workspace_aval = jax.core.ShapedArray(
            shape=(get_cublas_workspace_size_bytes(),), dtype=jnp.uint8
        )

        return (
            out_aval,
            out_amax_updated_aval,
            out_scale_updated_aval,
            pre_gelu_out_aval,
            bias_grad_aval,
            workspace_aval,
        )

    @staticmethod
    def outer_abstract(*args, **kwargs):
        """
        cuBlasLt GEMM outer abstract
        """
        (out_aval, out_amax_aval, out_scale_aval, pre_gelu_out_aval, bias_grad_aval, _) = (
            CollectiveGemmPrimitive.abstract(*args, **kwargs)
        )
        return out_aval, out_amax_aval, out_scale_aval, pre_gelu_out_aval, bias_grad_aval

    @staticmethod
    def lowering(
        ctx,
        lhs,
        lhs_scale_inv,
        rhs,
        rhs_scale_inv,
        bias,
        gelu_input,
        out_amax,
        out_scale,
        *,
        out_dtype,
        batched_output,
        contracting_dims,
        fuse_gelu,
        fuse_bias,
        grad,
        accumulate,
        use_split_accumulator,
    ):
        """
        Fused attention fwd lowering rules
        """
        del batched_output
        lhs_aval, _, rhs_aval, _, bias_aval, *_ = ctx.avals_in
        lhs_inner_dim, rhs_inner_dim = map(
            sanitize_dims, contracting_dims, (lhs_aval.ndim, rhs_aval.ndim)
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
                use_split_accumulator=use_split_accumulator,
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

            lhs_outer_dim, rhs_outer_dim = map(
                mirror_dim,
                (lhs_inner_dim, rhs_inner_dim),
                (lhs.ndim, rhs.ndim),
            )
            m = lhs_aval.shape[lhs_outer_dim]
            k = rhs_aval.shape[rhs_inner_dim]
            n = rhs_aval.shape[rhs_outer_dim]
            workspace_size = get_cublas_workspace_size_bytes()
            operand_dtype = jax_dtype_to_te_dtype(lhs_aval.dtype)
            bias_dtype = jax_dtype_to_te_dtype(bias_aval.dtype)
            opaque = tex.pack_gemm_descriptor(
                m,
                n,
                k,
                workspace_size,
                operand_dtype,
                jax_dtype_to_te_dtype(out_dtype),
                bias_dtype,
                lhs_trans,
                rhs_trans,
                fuse_gelu,
                fuse_bias,
                grad,
                accumulate,
                use_split_accumulator,
            )

            return custom_caller(
                CollectiveGemmPrimitive.name,
                args,
                opaque,
                has_side_effect=False,
                operand_output_aliases=operand_output_aliases,
            )

    @staticmethod
    def impl(
        lhs,
        lhs_scale_inv,
        rhs,
        rhs_scale_inv,
        bias,
        gelu_input,
        out_amax,
        out_scale,
        out_dtype,
        batched_output,
        contracting_dims,
        fuse_gelu,
        fuse_bias,
        grad,
        accumulate,
        use_split_accumulator,
    ):
        assert CollectiveGemmPrimitive.inner_primitive is not None

        lhs_inner_dim, rhs_inner_dim = map(sanitize_dims, contracting_dims, (lhs.ndim, rhs.ndim))
        lhs_outer_dim, rhs_outer_dim = map(
            mirror_dim, (lhs_inner_dim, rhs_inner_dim), (lhs.ndim, rhs.ndim)
        )

        # Infer output shape and collapse batch dimensions
        lhs_2d_shape = rhs_2d_shape = None
        lhs_layout = rhs_layout = None
        lhs_batch_dims = [
            dim for dim in range(lhs.ndim) if dim not in [lhs_inner_dim, lhs_outer_dim]
        ]
        lhs_batch_shape = [lhs.shape[dim] for dim in lhs_batch_dims]
        lhs_batch_size = reduce(operator.mul, lhs_batch_shape, 1)
        contracting_dims_2d = list(contracting_dims).copy()
        if batched_output:
            # If output is batched, the LSH batch dimension collapses into the outer dimension
            # and RHS cannot be batched
            lhs_2d_shape = (lhs_batch_size * lhs.shape[lhs_outer_dim], lhs.shape[lhs_inner_dim])
            lhs_layout = (*lhs_batch_dims, lhs_outer_dim, lhs_inner_dim)
            contracting_dims_2d[0] = 1
        else:
            # If the output is not batched, both LHS and RHS batch  dimensions collapse into the
            # contracting dimensions
            lhs_2d_shape = (lhs_batch_size * lhs.shape[lhs_inner_dim], lhs.shape[lhs_outer_dim])
            lhs_layout = (*lhs_batch_dims, lhs_inner_dim, lhs_outer_dim)
            contracting_dims_2d[0] = 0

            rhs_batch_dims = [
                dim for dim in range(rhs.ndim) if dim not in [rhs_inner_dim, rhs_outer_dim]
            ]
            rhs_batch_shape = [rhs.shape[dim] for dim in rhs_batch_dims]
            rhs_batch_size = reduce(operator.mul, rhs_batch_shape, 1)
            rhs_2d_shape = (rhs_batch_size * rhs.shape[rhs_inner_dim], rhs.shape[rhs_outer_dim])
            rhs_layout = (*rhs_batch_dims, rhs_inner_dim, rhs_outer_dim)
            contracting_dims_2d[1] = 0

        # Reshape LHS and RHS into 2D and fix layouts for FP8 GEMM
        if lhs_2d_shape is not None and lhs.ndim > 2:
            lhs = jax.lax.reshape(lhs, lhs_2d_shape, dimensions=lhs_layout)
            if jax_dtype_is_fp8(lhs.dtype):
                lhs = jax.lax.transpose(lhs, (1, 0))
                contracting_dims_2d[0] = 1
        else:
            contracting_dims_2d[0] = contracting_dims[0]

        if rhs_2d_shape is not None and rhs.ndim > 2:
            rhs = jax.lax.reshape(rhs, rhs_2d_shape, dimensions=rhs_layout)
            if jax_dtype_is_fp8(rhs.dtype):
                rhs = jax.lax.transpose(rhs, (1, 0))
                contracting_dims_2d[1] = 1
        else:
            contracting_dims_2d[1] = contracting_dims[1]

        # Invoke GEMM with guaranteed 2D inputs, so batched_output=False
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
            batched_output=False,
            contracting_dims=contracting_dims_2d,
            fuse_gelu=fuse_gelu,
            fuse_bias=fuse_bias,
            grad=grad,
            accumulate=accumulate,
            use_split_accumulator=use_split_accumulator,
        )

        # Recover batched dimensions in the output
        if batched_output:
            out_shape = (*lhs_batch_shape, out.shape[-2] // lhs_batch_size, out.shape[-1])
            out = jax.lax.reshape(out, out_shape)

        return out, out_amax_updated, out_scale_updated, pre_gelu_out, bias_grad

    @staticmethod
    def batcher(
        batched_args,
        batch_dims,
        *,
        out_dtype,
        batched_output,
        contracting_dims,
        fuse_gelu,
        fuse_bias,
        grad,
        accumulate,
        use_split_accumulator,
    ):
        assert CollectiveGemmPrimitive.outer_primitive is not None
        check_valid_batch_dims(batch_dims)
        lhs_bdims, *_, bias_bdims, gelu_input_bdims, out_amax_bdims, out_scale_bdims = batch_dims

        return (
            CollectiveGemmPrimitive.outer_primitive.bind(
                *batched_args,
                out_dtype=out_dtype,
                batched_output=batched_output,
                contracting_dims=contracting_dims,
                fuse_gelu=fuse_gelu,
                fuse_bias=fuse_bias,
                grad=grad,
                accumulate=accumulate,
                use_split_accumulator=use_split_accumulator,
            ),
            (lhs_bdims, out_amax_bdims, out_scale_bdims, gelu_input_bdims, bias_bdims),
        )

    @staticmethod
    def infer_sharding_from_operands(
        out_dtype,
        batched_output,
        contracting_dims,
        fuse_gelu,
        fuse_bias,
        grad,
        accumulate,
        use_split_accumulator,
        mesh,
        arg_infos,
        result_infos,
    ):
        del out_dtype, accumulate, use_split_accumulator, result_infos
        lhs, _, rhs, *_ = arg_infos
        lhs_spec, rhs_spec = map(get_padded_spec, [lhs, rhs])

        lhs_inner_dim, rhs_inner_dim = map(sanitize_dims, contracting_dims, (lhs.ndim, rhs.ndim))
        lhs_outer_dim, rhs_outer_dim = map(
            mirror_dim,
            (lhs_inner_dim, rhs_inner_dim),
            (lhs.ndim, rhs.ndim),
        )

        # Modify operand specs:
        # - FSDP axes are all-gathered
        # - LHS operand outer dimension is all-gathered if RHS operand outer dimension is sharded
        # - LHS operand contracting dimension sharding is forced to match RHS contracting dimension
        lhs_spec_new = remove_fsdp_specs(lhs_spec)
        rhs_spec_new = remove_fsdp_specs(rhs_spec)
        if lhs_spec_new[lhs_inner_dim] != rhs_spec_new[rhs_inner_dim] and not grad:
            warnings.warn(
                "Forcing the inner dimension of LHS to match the sharding of inner "
                + "dimension of RHS. This can trigger additional communication if LHS is "
                + "not already partitioned correctly."
            )
        rhs_outer_spec = rhs_spec_new[rhs_outer_dim]
        if rhs_outer_spec is not None:
            lhs_spec_new[lhs_outer_dim] = None
        lhs_spec_new[lhs_inner_dim] = rhs_spec_new[rhs_inner_dim]

        # Output sharding is conditional on output shape
        lhs_bdims = [dim for dim in range(lhs.ndim) if dim not in [lhs_inner_dim, lhs_outer_dim]]
        batch_spec = [lhs_spec_new[dim] for dim in lhs_bdims]
        lhs_outer_spec = lhs_spec_new[lhs_outer_dim]
        out_spec = [lhs_outer_spec, rhs_outer_spec]
        if batched_output:
            out_spec = batch_spec + out_spec
        out_sharding = NamedSharding(mesh, PartitionSpec(*out_spec))

        # FP8 metas are always unsharded
        fp8_meta_sharding = NamedSharding(mesh, PartitionSpec(None))

        # Pre-GELU output is always 2D if GELU fusion is turned on, otherwise unsharded
        gelu_spec = [lhs_outer_spec, rhs_outer_spec] if fuse_gelu else [None]
        gelu_sharding = NamedSharding(mesh, PartitionSpec(*gelu_spec))

        # Bias gradient spec matches outer dimension of output if bias fusion is turned on
        bias_sharding = NamedSharding(mesh, PartitionSpec(rhs_outer_spec if fuse_bias else None))
        return (out_sharding, fp8_meta_sharding, fp8_meta_sharding, gelu_sharding, bias_sharding)

    @staticmethod
    def partition(
        out_dtype,
        batched_output,
        contracting_dims,
        fuse_gelu,
        fuse_bias,
        grad,
        accumulate,
        use_split_accumulator,
        mesh,
        arg_infos,
        result_infos,
    ):
        del result_infos
        lhs, _, rhs, *_ = arg_infos
        lhs_spec, rhs_spec = map(get_padded_spec, [lhs, rhs])

        lhs_inner_dim, rhs_inner_dim = map(sanitize_dims, contracting_dims, (lhs.ndim, rhs.ndim))
        lhs_outer_dim, rhs_outer_dim = map(
            mirror_dim,
            (lhs_inner_dim, rhs_inner_dim),
            (lhs.ndim, rhs.ndim),
        )

        # Modify operand specs:
        # - FSDP axes are all-gathered
        # - LHS operand outer dimension is all-gathered if RHS operand outer dimension is sharded
        # - LHS operand contracting dimension sharding is forced to match RHS contracting dimension
        lhs_spec_new = remove_fsdp_specs(lhs_spec)
        rhs_spec_new = remove_fsdp_specs(rhs_spec)
        rhs_outer_spec = rhs_spec_new[rhs_outer_dim]
        if rhs_outer_spec is not None:
            lhs_spec_new[lhs_outer_dim] = None
        lhs_spec_new[lhs_inner_dim] = rhs_spec_new[rhs_inner_dim]
        lhs_sharding = NamedSharding(mesh, PartitionSpec(*lhs_spec_new))
        rhs_sharding = NamedSharding(mesh, PartitionSpec(*rhs_spec_new))

        # Bias is sharded to match outer dimension spec of the RHS operand (also the output)
        bias_sharding = NamedSharding(mesh, PartitionSpec(rhs_outer_spec if fuse_bias else None))

        # FP8 metas are always unsharded
        fp8_meta_sharding = NamedSharding(mesh, PartitionSpec(None))

        # Output sharding is conditional on output shape
        lhs_bdims = [dim for dim in range(lhs.ndim) if dim not in [lhs_inner_dim, lhs_outer_dim]]
        batch_spec = [lhs_spec_new[dim] for dim in lhs_bdims]
        lhs_outer_spec = lhs_spec_new[lhs_outer_dim]
        out_spec = [lhs_outer_spec, rhs_outer_spec]
        if batched_output:
            out_spec = batch_spec + out_spec
        out_sharding = NamedSharding(mesh, PartitionSpec(*out_spec))

        # Pre-GELU output is always 2D if GELU fusion is turned on, otherwise unsharded
        gelu_spec = [lhs_outer_spec, rhs_outer_spec] if fuse_gelu else [None]
        gelu_sharding = NamedSharding(mesh, PartitionSpec(*gelu_spec))

        arg_shardings = (
            lhs_sharding,
            fp8_meta_sharding,
            rhs_sharding,
            fp8_meta_sharding,
            bias_sharding,
            gelu_sharding,
            fp8_meta_sharding,
            fp8_meta_sharding,
        )
        out_shardings = (
            out_sharding,
            fp8_meta_sharding,
            fp8_meta_sharding,
            gelu_sharding,
            bias_sharding,
        )

        def sharded_impl(
            lhs, lhs_scale_inv, rhs, rhs_scale_inv, bias, gelu_input, out_amax, out_scale
        ):
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
                batched_output=batched_output,
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

            # GEMM output needs to be all-reduced when the contracting dimension is sharded.
            if rhs_spec_new[rhs_inner_dim] is not None:
                out = lax_paral_op(out, jax.lax.psum, global_mesh_resource().tp_resource, mesh)
                if fuse_gelu:
                    pre_gelu_out = lax_paral_op(
                        pre_gelu_out, jax.lax.psum, global_mesh_resource().tp_resource, mesh
                    )

            return out, out_amax_updated, out_scale_updated, pre_gelu_out, bias_grad

        return mesh, sharded_impl, out_shardings, arg_shardings


register_primitive(CollectiveGemmPrimitive)


def fp8_gemm_impl(
    lhs: ArrayLike,
    lhs_scale_inv: ArrayLike,
    rhs_t: ArrayLike,
    rhs_scale_inv: ArrayLike,
    bias: Optional[ArrayLike] = None,
    gelu_input: Optional[ArrayLike] = None,
    out_amax: Optional[ArrayLike] = None,
    out_scale: Optional[ArrayLike] = None,
    out_dtype: jnp.dtype = jnp.bfloat16,
    batched_output: bool = False,
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
        assert bias is not None, "Missing bias in forward GEMM when bias epilogue is enabled."

    if not fuse_gelu:
        gelu_input = jnp.zeros(0, dtype=bias.dtype)
    elif gelu_input is None:
        gelu_shape = (reduce(operator.mul, lhs.shape[:-1]), rhs_t.shape[-1])
        gelu_input = jnp.zeros(gelu_shape, dtype=bias.dtype)

    out, out_amax, out_scale, pre_gelu_out, _ = CollectiveGemmPrimitive.outer_primitive.bind(
        lhs,
        lhs_scale_inv,
        rhs_t,
        rhs_scale_inv,
        bias,
        gelu_input,
        out_amax,
        out_scale,
        out_dtype=out_dtype,
        batched_output=batched_output,
        contracting_dims=(-1, -1),
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
    bias: Optional[ArrayLike] = None,
    gelu_input: Optional[ArrayLike] = None,
    batched_output: bool = False,
    contracting_dims: Tuple[int, int] = (-1, -2),
    fuse_gelu: bool = False,
    fuse_bias: bool = False,
    grad: bool = False,
    accumulate: bool = False,
    use_split_accumulator: bool = False,
) -> Tuple[ArrayLike, ...]:
    """Non-FP8 mat-mul with `nvte_cublas_gemm()` custom op."""
    lhs_inner_dim, rhs_inner_dim = map(sanitize_dims, contracting_dims, (lhs.ndim, rhs.ndim))
    lhs_outer_dim, rhs_outer_dim = map(
        mirror_dim,
        (lhs_inner_dim, rhs_inner_dim),
        (lhs.ndim, rhs.ndim),
    )

    if not fuse_bias:
        bias = jnp.zeros(0, dtype=lhs.dtype)
    elif grad:
        bias = jnp.zeros(rhs.shape[rhs_outer_dim], dtype=lhs.dtype)
    else:
        assert bias is not None, "Missing bias in forward GEMM when bias epilogue is enabled."

    if not fuse_gelu:
        gelu_input = jnp.zeros(0, dtype=lhs.dtype)
    elif grad:
        assert (
            gelu_input is not None
        ), "Backward GEMM with dGELU epilogue requires pre-GELU output from forward GEMM."
    elif gelu_input is None:
        bdims = [dim for dim in range(lhs.ndim) if dim not in [lhs_inner_dim, lhs_outer_dim]]
        batch_size = reduce(operator.mul, [lhs.shape[dim] for dim in bdims], 1)
        gelu_shape = (batch_size * lhs.shape[lhs_outer_dim], rhs.shape[rhs_outer_dim])
        gelu_input = jnp.zeros(gelu_shape, dtype=lhs.dtypes)

    dummy_fp8_meta = jnp.zeros(0, dtype=jnp.float32)
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
        batched_output=batched_output,
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
