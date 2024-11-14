# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
import warnings
import operator
from functools import reduce, partial
from typing import Optional, Tuple

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
    get_mesh_axis_size,
)


__all__ = [
    "fp8_gemm_impl",
    "gemm_impl",
]

_COMM_GEMM_OVERLAP_LAYERS = ["qkv", "proj", "fc1", "fc2"]
_COMM_GEMM_OVERLAP_NAMES = (
    [layer + "_fprop" for layer in _COMM_GEMM_OVERLAP_LAYERS]
    + [layer + "_dgrad" for layer in _COMM_GEMM_OVERLAP_LAYERS]
    + [layer + "_wgrad" for layer in _COMM_GEMM_OVERLAP_LAYERS if layer != "fc2"]
    + ["generic_ag", "generic_rs"]
)


def sanitize_dims(dim, ndims):
    return (ndims + dim) if dim < 0 else dim


def mirror_dim(dim, ndims):
    return ndims - 2 if dim == ndims - 1 else ndims - 1


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
    impl_static_args = (8, 9, 10, 11, 12, 13, 14, 15, 16)
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
        comm_overlap_config,
    ):
        """
        cuBlasLt GEMM abstract
        """
        del grad, accumulate, use_split_accumulator

        assert tex.ubuf_built_with_mpi(), (
            "Comm+GEMM overlap in TE/JAX requires Transformer Engine to be compiled with "
            + "`NVTE_UB_WITH_MPI=1` and `MPI_HOME=/path/to/mpi` options."
        )

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

        # Validate operand layouts, adjusted for comm-overlap if necessary
        lhs_inner_dim, rhs_inner_dim = map(
            sanitize_dims, contracting_dims, (lhs_aval.ndim, rhs_aval.ndim)
        )
        assert (
            lhs_aval.shape[lhs_inner_dim] == rhs_aval.shape[rhs_inner_dim]
        ), f"Incompatible contracting dimensions: {lhs_aval.shape} x {rhs_aval.shape}."

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

        if rhs_aval.ndim > 2:
            rhs_bdims = [
                dim for dim in range(rhs_aval.ndim) if dim not in [rhs_outer_dim, rhs_inner_dim]
            ]
            rhs_batch_shape = [rhs_aval.shape[dim] for dim in rhs_bdims]
            rhs_batch_size = reduce(operator.mul, rhs_bdims, 1)
            if rhs_batch_size > 1:
                assert lhs_batch_size == rhs_batch_size, (
                    f"Leading dimensins of RHS ({rhs_batch_shape=}) is not broadcast-compatible "
                    + f"with the leading dimensions of LHS ({lhs_batch_shape=})."
                )

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

        # Adjust output sizes for comm-overlap
        extra_out_shape = (0,)
        extra_out_dtype = jnp.bfloat16
        if comm_overlap_config is not None:
            comm_overlap_type = comm_overlap_config.get("comm_type", None)
            assert comm_overlap_type is not None, "Missing comm type for comm+GEMM overlap."
            comm_overlap_name = comm_overlap_config.get("name", None)
            assert (
                comm_overlap_name in _COMM_GEMM_OVERLAP_NAMES
            ), f"Unrecognized comm+GEMM overlap name: {comm_overlap_name=}"

            mesh = comm_overlap_config.get("mesh", None)
            tp_resource = comm_overlap_config.get("tp_resource", global_mesh_resource().tp_resource)
            tp_size = get_mesh_axis_size(tp_resource, mesh=mesh)

            match comm_overlap_type:
                case tex.CommOverlapType.AG:
                    # Extra output is all-gathered LHS copy
                    extra_out_shape = list(lhs_aval.shape).copy()
                    extra_out_shape[lhs_outer_dim] *= tp_size
                    extra_out_dtype = lhs_dtype

                case tex.CommOverlapType.RS:
                    # FP8 GEMM output for RS overlap is always FP8
                    if jax_dtype_is_fp8(lhs_dtype):
                        assert jax_dtype_is_fp8(
                            out_dtype
                        ), "FP8 GEMM with reduce-scatter overlap requires FP8 output."
                    # Extra output is reduce-scattered GEMM output
                    extra_out_shape = list(out_shape).copy()
                    extra_out_shape[-2] /= tp_size

                case _:
                    raise RuntimeError(
                        f"Unrecognized comm type for comm+GEMM overlap: {comm_overlap_type=}"
                    )

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
        extra_out_aval = jax.core.ShapedArray(shape=extra_out_shape, dtype=extra_out_dtype)
        workspace_aval = jax.core.ShapedArray(
            shape=(get_cublas_workspace_size_bytes(),), dtype=jnp.uint8
        )

        return (
            out_aval,
            out_amax_updated_aval,
            out_scale_updated_aval,
            pre_gelu_out_aval,
            bias_grad_aval,
            extra_out_aval,  # global LHS for AG overlap, or sharded output for RS overlap
            workspace_aval,
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
            extra_out_aval,
            *_,
        ) = CollectiveGemmPrimitive.abstract(*args, **kwargs)
        return (
            out_aval,
            out_amax_aval,
            out_scale_aval,
            pre_gelu_out_aval,
            bias_grad_aval,
            extra_out_aval,
        )

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
        comm_overlap_config,
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
            ffi_args = (
                ctx,
                lhs,
                lhs_scale_inv,
                rhs,
                rhs_scale_inv,
                bias,
                gelu_input,
                out_amax,
                out_scale,
            )
            ffi_kwargs = dict(
                lhs_trans=lhs_trans,
                rhs_trans=rhs_trans,
                fuse_gelu=fuse_gelu,
                fuse_bias=fuse_bias,
                grad=grad,
                accumulate=accumulate,
                use_split_accumulator=use_split_accumulator,
            )

            if comm_overlap_config is not None:
                name = "te_comm_gemm_overlap_ffi"
                ffi_kwargs["comm_type"] = int(comm_overlap_config["comm_type"])
                ffi_kwargs["name"] = comm_overlap_config["name"]

            return ffi.ffi_lowering(name, operand_output_aliases=operand_output_aliases)(
                *ffi_args, **ffi_kwargs
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

            descriptor_packer_fn = tex.pack_gemm_decriptor
            descriptor_args = (
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

            comm_overlap_type = comm_overlap_config.get("comm_type", None)
            if comm_overlap_type is not None:
                name = "te_comm_gemm_overlap"
                descriptor_packer_fn = tex.pack_overlap_descriptor
                descriptor_args += (
                    comm_overlap_type,
                    comm_overlap_config.get("name", None),
                )
            opaque = descriptor_packer_fn(*descriptor_args)

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
        comm_overlap_config,
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
        if lhs.ndim > 2 and rhs.ndim > 2:
            # If both LHS and RHS are batched, the batch dimensions collapse into the
            # contracting dimensions for both operands
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
        elif lhs.ndim > 2:
            # If only the LHS is batched,the batch dimension collapses into the outer dimension
            lhs_2d_shape = (lhs_batch_size * lhs.shape[lhs_outer_dim], lhs.shape[lhs_inner_dim])
            lhs_layout = (*lhs_batch_dims, lhs_outer_dim, lhs_inner_dim)
            contracting_dims_2d[0] = 1

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
            extra_out,
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
            comm_overlap_config=comm_overlap_config,
        )

        # Recover batched dimensions in the output
        if batched_output:
            out_shape = (*lhs_batch_shape, out.shape[-2] // lhs_batch_size, out.shape[-1])
            out = jax.lax.reshape(out, out_shape)

        return out, out_amax_updated, out_scale_updated, pre_gelu_out, bias_grad, extra_out

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
        comm_overlap_config,
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
        comm_overlap_config,
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

        # Modify operand specs
        lhs_spec_new = [spec for spec in lhs_spec]
        rhs_spec_new = [spec for spec in rhs_spec]
        reduce_output = False
        if comm_overlap_config is None:
            # When comm overlap is not enabled:
            # - Always all-gather the outer dimension of LHS.
            # - If contracting dims of both operands are sharded, all-gather RHS outer dim.
            # - If contracting dim of only one operand is sharded, all-gather the sharded operand.
            # - Never scatter any operand.
            lhs_spec_new[lhs_outer_dim] = None
            if lhs_spec_new[lhs_inner_dim] is not None and rhs_spec_new[rhs_inner_dim] is not None:
                assert (
                    lhs_spec_new[lhs_inner_dim] == rhs_spec_new[rhs_inner_dim]
                ), "Contracting dimensions of LHS and RHS operands must have the same sharding."
                if lhs_spec_new[lhs_outer_dim] is not None:
                    warnings.warn(
                        "Outer dimension of the LHS operand must be all-gathered when both "
                        + "contracting dimensions are sharded. This will cause additional "
                        + "communication overhead."
                    )

                if rhs_spec_new[rhs_outer_dim] is not None:
                    warnings.warn(
                        "Outer dimension of the RHS operand must be all-gathered when both "
                        + "contracting dimensions are sharded. This will cause additional "
                        + "communication overhead."
                    )
                rhs_spec_new[rhs_outer_dim] = None
                reduce_output = True
            else:
                if lhs_spec_new[lhs_inner_dim] is None and rhs_spec_new[rhs_inner_dim] is not None:
                    warnings.warn(
                        "Contracting dimension of the RHS operand must be all-gathered when the "
                        + "contracting dimension of the LHS operand is unsharded. This will cause "
                        + "additional communication overhead."
                    )
                if lhs_spec_new[lhs_inner_dim] is not None and rhs_spec_new[rhs_inner_dim] is None:
                    if not grad:
                        # This is expected for sequence/context-parallel gradient in BWD (DGRAD) GEMM.
                        warnings.warn(
                            "Contracting dimension of the LHS operand must be all-gathered when "
                            + "the contracting dimension of the RHS operand is unsharded. This "
                            + "will cause additional communication overhead."
                        )
                lhs_spec_new[lhs_inner_dim] = None
                rhs_spec_new[rhs_inner_dim] = None
        else:
            # When comm overlap is enabled, make sure both contracting dims are unsharded if one
            # of them is unsharded.
            if lhs_spec_new[lhs_inner_dim] is None or rhs_spec_new[rhs_inner_dim] is None:
                lhs_spec_new[lhs_inner_dim] = None
                rhs_spec_new[rhs_inner_dim] = None
        out_col_spec = rhs_spec_new[rhs_outer_dim]

        # Output sharding is conditional on output shape
        lhs_bdims = [dim for dim in range(lhs.ndim) if dim not in [lhs_inner_dim, lhs_outer_dim]]
        batch_spec = [lhs_spec_new[dim] for dim in lhs_bdims]
        out_spec = [None, out_col_spec]
        if batched_output:
            out_spec = batch_spec + out_spec
        out_sharding = NamedSharding(mesh, PartitionSpec(*out_spec))

        # FP8 metas are always unsharded
        fp8_meta_sharding = NamedSharding(mesh, PartitionSpec(None))

        # Pre-GELU output is always 2D if GELU fusion is turned on, otherwise unsharded
        gelu_spec = [None, out_col_spec] if fuse_gelu else [None]
        gelu_sharding = NamedSharding(mesh, PartitionSpec(*gelu_spec))

        # Bias gradient spec matches outer dimension of output if bias fusion is turned on
        bias_sharding = NamedSharding(mesh, PartitionSpec(out_col_spec if fuse_bias else None))

        # Validate operand sharding for comm+GEMM overlap and adust extra output sharding
        extra_out_spec = [None]
        if comm_overlap_config is not None:
            mesh = comm_overlap_config.get("mesh", None)
            tp_resource = comm_overlap_config.get("tp_resource", global_mesh_resource().tp_resource)
            match comm_overlap_config.get("comm_type", None):
                case tex.CommOverlapType.AG:
                    # AG overlap requires the outer dimension of LHS to be sharded
                    # over the TP resource
                    assert lhs_spec[lhs_outer_dim] == tp_resource, (
                        "AG+GEMM overlap requires the outer (sequence) dimension of the LHS "
                        + f"operand to be sharded over the TP resource (mesh axis: {tp_resource=})."
                    )
                    extra_out_spec = list(lhs_spec).copy()
                    extra_out_spec[lhs_outer_dim] = None

                case tex.CommOverlapType.RS:
                    # RS overlap requires the contracting dimensions of both LHS and RHS to be
                    # sharded over the TP resource, and the outer dimension of LHS to be unsharded
                    assert lhs_spec[lhs_outer_dim] is None, (
                        "GEMM+RS overlap requires the outer (sequence) dimension of the LHS "
                        + "operand to be un-sharded."
                    )
                    assert lhs_spec[lhs_inner_dim] == tp_resource, (
                        "GEMM+RS overlap requires the contracting dimension of the LHS operand "
                        + f"to be sharded over the TP resource (mesh axis: {tp_resource=})."
                    )
                    assert rhs_spec[rhs_inner_dim] == tp_resource, (
                        "GEMM+RS overlap requires the contracting dimension of the RHS operand "
                        + f"to be sharded over the TP resource (mesh axis: {tp_resource=})."
                    )
                    extra_out_spec = out_spec.copy()
                    extra_out_spec[-2] = tp_resource

        extra_out_sharding = NamedSharding(mesh, PartitionSpec(*extra_out_spec))

        return (
            out_sharding,
            fp8_meta_sharding,
            fp8_meta_sharding,
            gelu_sharding,
            bias_sharding,
            extra_out_sharding,
        )

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
        comm_overlap_config,
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

        # Modify operand specs
        lhs_spec_new = [spec for spec in lhs_spec]
        rhs_spec_new = [spec for spec in rhs_spec]
        reduce_output = False
        if comm_overlap_config is None:
            # When comm overlap is not enabled:
            # - Always all-gather the outer dimension of LHS.
            # - If contracting dims of both operands are sharded, all-gather RHS outer dim.
            # - If contracting dim of only one operand is sharded, all-gather the sharded operand.
            # - Never scatter any operand.
            lhs_spec_new[lhs_outer_dim] = None
            if lhs_spec_new[lhs_inner_dim] is not None and rhs_spec_new[rhs_inner_dim] is not None:
                rhs_spec_new[rhs_outer_dim] = None
                reduce_output = True
            else:
                lhs_spec_new[lhs_inner_dim] = None
                rhs_spec_new[rhs_inner_dim] = None
        else:
            # When comm overlap is enabled, make sure both contracting dims are unsharded if one
            # of them is unsharded.
            if lhs_spec_new[lhs_inner_dim] is None or rhs_spec_new[rhs_inner_dim] is None:
                lhs_spec_new[lhs_inner_dim] = None
                rhs_spec_new[rhs_inner_dim] = None
        out_col_spec = rhs_spec_new[rhs_outer_dim]

        lhs_sharding = NamedSharding(mesh, PartitionSpec(*lhs_spec_new))
        rhs_sharding = NamedSharding(mesh, PartitionSpec(*rhs_spec_new))

        # Bias is sharded to match outer dimension spec of the RHS operand (also the output)
        bias_sharding = NamedSharding(mesh, PartitionSpec(out_col_spec if fuse_bias else None))

        # FP8 metas are always unsharded
        fp8_meta_sharding = NamedSharding(mesh, PartitionSpec(None))

        # Output sharding is conditional on output shape
        lhs_bdims = [dim for dim in range(lhs.ndim) if dim not in [lhs_inner_dim, lhs_outer_dim]]
        batch_spec = [lhs_spec_new[dim] for dim in lhs_bdims]
        out_spec = [None, out_col_spec]
        if batched_output:
            out_spec = batch_spec + out_spec
        out_sharding = NamedSharding(mesh, PartitionSpec(*out_spec))

        # Pre-GELU output is always 2D if GELU fusion is turned on, otherwise unsharded
        gelu_spec = [None, out_col_spec] if fuse_gelu else [None]
        gelu_sharding = NamedSharding(mesh, PartitionSpec(*gelu_spec))

        # Adjust extra output sharding for comm+GEMM overlap
        extra_out_spec = [None]
        if comm_overlap_config is not None:
            mesh = comm_overlap_config.get("mesh", None)
            tp_resource = comm_overlap_config.get("tp_resource", global_mesh_resource().tp_resource)
            match comm_overlap_config.get("comm_type", None):
                case tex.CommOverlapType.AG:
                    extra_out_spec = list(lhs_spec).copy()
                    extra_out_spec[lhs_outer_dim] = None

                case tex.CommOverlapType.RS:
                    extra_out_spec = out_spec.copy()
                    extra_out_spec[-2] = tp_resource

        extra_out_sharding = NamedSharding(mesh, PartitionSpec(*extra_out_spec))

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
            extra_out_sharding,
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
                extra_out,
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
                comm_overlap_config=comm_overlap_config,
            )

            # FP8 amax reduction
            if jax_dtype_is_fp8(lhs.dtype):
                out_amax_updated = all_reduce_max_along_all_axes_except_PP(out_amax_updated, mesh)

            # All-reduce sum GEMM output when contracting dimensions are sharded
            if comm_overlap_config is None:
                if reduce_output:
                    out = jax.lax.psum(out, global_mesh_resource().tp_resource)
                    if fuse_gelu:
                        pre_gelu_out = jax.lax.psum(
                            pre_gelu_out, global_mesh_resource().tp_resource
                        )

            return out, out_amax_updated, out_scale_updated, pre_gelu_out, bias_grad, extra_out

        return mesh, sharded_impl, out_shardings, arg_shardings


register_primitive(CollectiveGemmPrimitive)


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
    comm_overlap_config: Optional[dict] = None,
) -> Tuple[ArrayLike, ...]:
    """Non-FP8 mat-mul with `nvte_cublas_gemm()` custom op."""
    dummy_fp8_meta = jnp.zeros(0, dtype=jnp.float32)
    lhs_inner_dim, rhs_inner_dim = map(sanitize_dims, contracting_dims, (lhs.ndim, rhs.ndim))
    lhs_outer_dim = lhs.ndim - 1 if lhs_inner_dim != lhs.ndim - 1 else lhs.ndim - 2
    rhs_outer_dim = rhs.ndim - 2 if rhs_inner_dim == rhs.ndim - 1 else rhs.ndim - 1
    out_shape = (*lhs.shape[:-2], lhs.shape[lhs_outer_dim], rhs.shape[rhs_outer_dim])

    if not fuse_bias:
        bias = jnp.zeros(0, dtype=lhs.dtype)
    elif grad:
        bias = jnp.zeros(out_shape[-1], dtype=lhs.dtype)
    else:
        assert bias is not None, "Missing bias in forward GEMM when bias epilogue is enabled."

    if not fuse_gelu:
        gelu_input = jnp.zeros(0, dtype=lhs.dtype)
    elif grad:
        assert (
            gelu_input is not None
        ), "Backward GEMM with dGELU epilogue requires pre-GELU output from forward GEMM."
    elif gelu_input is None:
        gelu_input = jnp.zeros(out_shape, dtype=lhs.dtypes)

    (
        out,
        _,  # out_amax in FP8 GEMM
        _,  # out_scale in FP8 GEMM
        pre_gelu_out,
        bias_grad,
        extra_out,
    ) = CollectiveGemmPrimitive.outer_primitive.bind(
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
        comm_overlap_config=comm_overlap_config,
    )

    if grad:
        return out, pre_gelu_out, bias_grad, extra_out
    else:
        return out, pre_gelu_out, extra_out


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
    comm_overlap_config: Optional[dict] = None,
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

    (out, out_amax, out_scale, pre_gelu_out, _, extra_out) = (  # bias_grad in non-FP8 GEMM
        CollectiveGemmPrimitive.outer_primitive.bind(
            rhs_t,
            rhs_scale_inv,
            lhs,
            lhs_scale_inv,
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
            comm_overlap_config=comm_overlap_config,
        )
    )

    return out, out_amax, out_scale, pre_gelu_out, extra_out


class CopyIntoOverlapBufferPrimitive(BasePrimitive):
    """
    Copy JAX array data into comm+GEMM overlap buffer
    """

    name = "te_copy_into_overlap_buffer"
    impl_static_args = (1, 2)
    multiple_results = False
    inner_primitive = None
    outer_primitive = None

    @staticmethod
    def abstract(inp_aval, name, comm_type):
        assert name in _COMM_GEMM_OVERLAP_NAMES, f"Unrecognized comm+GEMM overlap name: {name=}"
        assert comm_type in [
            tex.CommOverlapType.AG,
            tex.CommOverlapType.RS,
        ], "Invalid comm+GEMM overlap type."
        assert inp_aval.size > 0, "Cannot copy a zero-size array into overlap buffer."
        assert inp_aval.ndim == 2, "Cannot copy more than 2 dimensions!"
        return jax.core.ShapedArray(shape=(0,), dtype=dtypes.canonicalize_dtype(inp_aval.dtype))

    @staticmethod
    def lowering(ctx, inp, *, name, comm_type):
        if is_ffi_enabled():
            name = "te_copy_into_overlap_buffer_ffi"
            return ffi.ffi_lowering(name)(
                ctx,
                inp,
                name=name,
                comm_type=int(comm_type),
            )
        else:
            operands = [inp]
            operand_shapes = [ir.RankedTensorType(inp.type).shape]
            out_types = []
            args = CustomCallArgsWrapper(out_types, operands, operand_shapes)
            opaque = tex.pack_buffer_descriptor(
                name, inp.shape, jax_dtype_to_te_dtype(inp.dtype), comm_type
            )
            return custom_caller(CopyIntoOverlapBufferPrimitive.name, args, opaque, False)

    @staticmethod
    def impl(inp, name, comm_type):
        assert CopyIntoOverlapBufferPrimitive.inner_primitive is not None
        return CopyIntoOverlapBufferPrimitive.inner_primitive.bind(
            inp, name=name, comm_type=comm_type
        )

    @staticmethod
    def batcher(batched_args, batch_dims, *, name, comm_type):
        assert CopyIntoOverlapBufferPrimitive.inner_primitive is not None
        check_valid_batch_dims(batch_dims)
        return (
            CopyIntoOverlapBufferPrimitive.inner_primitive.bind(
                *batched_args, name=name, comm_type=comm_type
            ),
            None,
        )

    @staticmethod
    def infer_sharding_from_operands(name, comm_type, mesh, arg_infos, result_infos):
        del name, comm_type, arg_infos, result_infos
        return NamedSharding(mesh, PartitionSpec(None))

    @staticmethod
    def partition(name, comm_type, mesh, arg_infos, result_infos):
        del name, comm_type, result_infos
        inp_spec = arg_infos[0]
        arg_shardings = (NamedSharding(mesh, PartitionSpec(*inp_spec)),)
        out_sharding = NamedSharding(mesh, PartitionSpec(None))
        return (
            mesh,
            partial(CopyIntoOverlapBufferPrimitive.impl, name=name, comm_type=comm_type),
            out_sharding,
            arg_shardings,
        )


register_primitive(CopyIntoOverlapBufferPrimitive)


def copy_into_overlap_buffer(inp: ArrayLike, name: str, comm_type: tex.CommOverlapType) -> None:
    _ = CollectiveGemmPrimitive.outer_primitive.bind(inp, name=name, comm_type=comm_type)
