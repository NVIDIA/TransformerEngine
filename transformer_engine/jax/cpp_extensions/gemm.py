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
    all_reduce_max_along_all_axes_except_PP,
)


__all__ = [
    "fp8_gemm_impl",
    "gemm_impl",
    "copy_into_overlap_buffer",
    "bootstrap_comm_gemm_overlap",
]

_COMM_GEMM_OVERLAP_LAYERS = ["qkv", "proj", "fc1", "fc2"]
_COMM_GEMM_OVERLAP_NAMES = (
    [layer + "_fprop" for layer in _COMM_GEMM_OVERLAP_LAYERS]
    + [layer + "_dgrad" for layer in _COMM_GEMM_OVERLAP_LAYERS]
    + [layer + "_wgrad" for layer in _COMM_GEMM_OVERLAP_LAYERS if layer != "fc2"]
    + ["ag_gemm", "gemm_rs"]
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
    impl_static_args = (10, 11, 12, 13, 14, 15, 16, 17, 18)
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
        out_aval,
        out_amax_aval,
        out_scale_aval,
        extra_out_aval,
        batched_output,
        contracting_dims,
        fuse_gelu,
        fuse_bias,
        grad,
        accumulate,
        use_split_accumulator,
        comm_overlap_config,
        sharded_abstract,
    ):
        """
        cuBlasLt GEMM abstract
        """
        if comm_overlap_config is not None:
            assert tex.ubuf_built_with_mpi(), (
                "Comm+GEMM overlap in TE/JAX requires Transformer Engine to be compiled with "
                + "`NVTE_UB_WITH_MPI=1` and `MPI_HOME=/path/to/mpi` options."
            )
            assert is_ffi_enabled(), "Comm+GEMM overlap is supported only via XLA FFI."

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
        assert lhs_aval.shape[lhs_inner_dim] == rhs_aval.shape[rhs_inner_dim], (
            "Incompatible operand sizes: "
            + f"{lhs_aval.shape} @ idx {lhs_inner_dim} X {rhs_aval.shape} @ idx {rhs_inner_dim}."
        )

        lhs_trans = lhs_inner_dim != lhs_aval.ndim - 1
        rhs_trans = rhs_inner_dim == rhs_aval.ndim - 1
        assert not (
            lhs_trans and rhs_trans
        ), "GEMM does not support transposed LHS and transposed RHS at the same time."
        if is_fp8:
            assert not lhs_trans, "FP8 GEMM does not support transposed LHS."
            assert rhs_trans, "FP8 GEMM requires transposed RHS."

        # Make sure leading dimensions of RHS is broadcast-compatible with LHS
        lhs_outer_dim, rhs_outer_dim = map(
            mirror_dim,
            (lhs_inner_dim, rhs_inner_dim),
            (lhs_aval.ndim, rhs_aval.ndim),
        )
        if lhs_aval.ndim > 2 and rhs_aval.ndim > 2:
            assert (
                not batched_output
            ), "Batched output requires batched LHS and non-batched RHS operands."
            lhs_bdims = [
                dim for dim in range(lhs_aval.ndim) if dim not in [lhs_outer_dim, lhs_inner_dim]
            ]
            lhs_batch_shape = [lhs_aval.shape[dim] for dim in lhs_bdims]
            lhs_batch_size = reduce(operator.mul, lhs_batch_shape, 1)
            rhs_bdims = [
                dim for dim in range(rhs_aval.ndim) if dim not in [rhs_outer_dim, rhs_inner_dim]
            ]
            rhs_batch_shape = [rhs_aval.shape[dim] for dim in rhs_bdims]
            rhs_batch_size = reduce(operator.mul, rhs_batch_shape, 1)
            assert lhs_batch_size == rhs_batch_size, (
                "Leading dimensions of LHS and RHS are not broadcast-compatible: "
                + f"{lhs_aval.shape} @ idx {lhs_inner_dim} X {rhs_aval.shape} @ idx {rhs_inner_dim}"
            )

        # Validate output dtypes
        out_dtype = dtypes.canonicalize_dtype(out_aval.dtype)
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
            assert out_dtype == lhs_dtype, (
                "Output buffer has incorrect dtype: "
                + f"expected {lhs_dtype} but found {out_dtype}"
            )
            out_amax_updated_dtype = jnp.float32
            out_scale_updated_dtype = jnp.float32

        # Validate output buffers
        out_shape = out_aval.shape
        expected_out_shape = [
            *lhs_aval.shape[:-2],
            lhs_aval.shape[lhs_outer_dim],
            rhs_aval.shape[rhs_outer_dim],
        ]
        extra_out_shape = extra_out_aval.shape
        expected_extra_out_shape = [0]
        extra_out_dtype = dtypes.canonicalize_dtype(extra_out_aval.dtype)
        expected_extra_out_dtype = jnp.bfloat16
        if batched_output:
            assert out_aval.ndim > 2, "Batched output buffer is missing batch dimensions."
        else:
            expected_out_shape = [
                reduce(operator.mul, expected_out_shape[:-1], 1),
                expected_out_shape[-1],
            ]

        if comm_overlap_config is not None and comm_overlap_config["method"] != "bulk":
            comm_type = comm_overlap_config.get("comm_type", None)
            assert comm_type is not None, "Missing comm type for comm+GEMM overlap."

            tp_size = comm_overlap_config.get("tp_size", 1)
            assert (
                tp_size > 1
            ), "Comm+GEMM overlap requires tensor-parallel mesh axis size greater than 1."

            if comm_type == tex.CommOverlapType.AG:
                expected_extra_out_shape = list(lhs_aval.shape).copy()
            elif comm_type == tex.CommOverlapType.RS:
                expected_extra_out_shape = list(expected_out_shape).copy()
                expected_extra_out_dtype = lhs_dtype

            if sharded_abstract:
                if comm_type == tex.CommOverlapType.AG:
                    expected_out_shape[-2] *= tp_size
                    expected_extra_out_shape[-2] *= tp_size
                else:
                    expected_extra_out_shape[-2] = expected_extra_out_shape[-2] // tp_size

        assert out_aval.ndim == len(expected_out_shape), (
            "Output buffer has incorrect number of dimensions: "
            + f"expected {len(expected_out_shape)} but found {out_aval.ndim}"
        )
        assert all([out_aval.shape[i] == expected_out_shape[i] for i in range(out_aval.ndim)]), (
            "Output buffer has incorrect shape: "
            + f"expected {expected_out_shape=} but found {out_aval.shape=}"
        )

        assert extra_out_dtype == expected_extra_out_dtype, (
            "Extra output has incorrect dtype: "
            + f"expected {expected_extra_out_dtype} but found {extra_out_dtype}"
        )
        assert extra_out_aval.ndim == len(expected_extra_out_shape), (
            "Extra output buffer has incorrect number of dimensions: "
            + f"expected {len(expected_extra_out_shape)} but found {extra_out_aval.ndim}"
        )
        assert all(
            [
                extra_out_aval.shape[i] == expected_extra_out_shape[i]
                for i in range(extra_out_aval.ndim)
            ]
        ), (
            "Extra output buffer has incorrect shape: "
            + f"expected {expected_extra_out_shape=} but found {extra_out_aval.shape=}"
        )

        # Validate bias/bias_grad shape against output bufer
        bias_dtype = jnp.bfloat16 if jax_dtype_is_fp8(out_dtype) else out_dtype
        if fuse_bias:
            assert (
                bias_aval.size > 0 and bias_aval.ndim == 1 and bias_aval.shape[0] == out_shape[-1]
            ), (
                "Incorrect bias shape: "
                + f"expected ({out_shape[-1]}, ) but found ({bias_aval.shape[0]}, )"
            )
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
        out_updated_aval = out_aval.update(shape=out_shape, dtype=out_dtype)
        out_amax_updated_aval = out_amax_aval.update(
            shape=out_amax_aval.shape, dtype=out_amax_updated_dtype
        )
        out_scale_updated_aval = out_scale_aval.update(
            shape=out_scale_aval.shape, dtype=out_scale_updated_dtype
        )
        pre_gelu_out_aval = gelu_input_aval.update(shape=gelu_shape, dtype=bias_dtype)
        bias_grad_aval = bias_aval.update(shape=bias_aval.shape, dtype=bias_dtype)
        extra_out_updated_aval = extra_out_aval.update(shape=extra_out_shape, dtype=extra_out_dtype)
        workspace_aval = jax.core.ShapedArray(
            shape=(get_cublas_workspace_size_bytes(),), dtype=jnp.uint8
        )

        return (
            out_updated_aval,
            out_amax_updated_aval,
            out_scale_updated_aval,
            pre_gelu_out_aval,
            bias_grad_aval,
            extra_out_updated_aval,  # global LHS for AG overlap, or sharded output for RS overlap
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
        out,
        out_amax,
        out_scale,
        extra_out,
        *,
        batched_output,
        contracting_dims,
        fuse_gelu,
        fuse_bias,
        grad,
        accumulate,
        use_split_accumulator,
        comm_overlap_config,
        sharded_abstract,
    ):
        """
        Fused attention fwd lowering rules
        """
        del batched_output, sharded_abstract
        lhs_aval, _, rhs_aval, _, bias_aval, *_ = ctx.avals_in
        lhs_inner_dim, rhs_inner_dim = map(
            sanitize_dims, contracting_dims, (lhs_aval.ndim, rhs_aval.ndim)
        )
        lhs_trans = lhs_inner_dim != lhs_aval.ndim - 1
        rhs_trans = rhs_inner_dim == rhs_aval.ndim - 1

        operands = [
            lhs,
            lhs_scale_inv,
            rhs,
            rhs_scale_inv,
            bias,
            gelu_input,
            out,
            out_amax,
            out_scale,
            extra_out,
        ]

        operand_output_aliases = {
            4: 4,  # bias        <-->  bias_grad
            5: 3,  # gelu_input  <-->  pre_gelu_out
            6: 0,  # out         <-->  out_updated
            7: 1,  # out_amax    <-->  out_amax_updated
            8: 2,  # out_scale   <-->  out_scale_updated
            9: 5,  # extra_out   <-->  extra_out_updated
        }

        if is_ffi_enabled():
            name = "te_gemm_ffi"
            ffi_args = (ctx, *operands)
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
                ffi_kwargs["comm_type_flag"] = int(comm_overlap_config["comm_type"])
                ffi_kwargs["name"] = comm_overlap_config["name"]

            return ffi.ffi_lowering(name, operand_output_aliases=operand_output_aliases)(
                *ffi_args, **ffi_kwargs
            )

        else:
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
                jax_dtype_to_te_dtype(dtypes.canonicalize_dtype(ctx.avals_out[0].dtype)),
                bias_dtype,
                lhs_trans,
                rhs_trans,
                fuse_gelu,
                fuse_bias,
                grad,
                accumulate,
                use_split_accumulator,
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
        out,
        out_amax,
        out_scale,
        extra_out,
        batched_output,
        contracting_dims,
        fuse_gelu,
        fuse_bias,
        grad,
        accumulate,
        use_split_accumulator,
        comm_overlap_config,
        sharded_abstract,
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

        # Reshape output and extra output buffers into 2D as well
        if out.ndim > 2:
            out = jax.lax.reshape(out, (reduce(operator.mul, out.shape[:-1], 1), out.shape[-1]))
        if extra_out.size > 0 and extra_out.ndim > 2:
            extra_out = jax.lax.reshape(
                extra_out, (reduce(operator.mul, extra_out.shape[:-1], 1), extra_out.shape[-1])
            )

        batched_extra_out = False
        if comm_overlap_config is not None and comm_overlap_config["method"] != "bulk":
            comm_type = comm_overlap_config["comm_type"]
            if comm_type == tex.CommOverlapType.AG:
                # Extra output is global LHS, we can collapse but need to recover batches later
                batched_extra_out = len(lhs_batch_dims) > 0
            elif comm_type == tex.CommOverlapType.RS:
                # Extra output is scattered GEMM output, so we recover batches only if the output is
                # batched
                batched_extra_out = batched_output

        # Invoke GEMM with guaranteed 2D inputs, so batched_output=False
        (
            out_updated,
            out_amax_updated,
            out_scale_updated,
            pre_gelu_out,
            bias_grad,
            extra_out_updated,
            _,
        ) = CollectiveGemmPrimitive.inner_primitive.bind(
            lhs,
            lhs_scale_inv,
            rhs,
            rhs_scale_inv,
            bias,
            gelu_input,
            out,
            out_amax,
            out_scale,
            extra_out,
            batched_output=False,
            contracting_dims=contracting_dims_2d,
            fuse_gelu=fuse_gelu,
            fuse_bias=fuse_bias,
            grad=grad,
            accumulate=accumulate,
            use_split_accumulator=use_split_accumulator,
            comm_overlap_config=comm_overlap_config,
            sharded_abstract=sharded_abstract,
        )

        # Recover batched dimensions in the output
        if batched_output:
            out_shape = (
                *lhs_batch_shape,
                out_updated.shape[-2] // lhs_batch_size,
                out_updated.shape[-1],
            )
            out_updated = jax.lax.reshape(out_updated, out_shape)

        if batched_extra_out:
            extra_out_shape = (
                *lhs_batch_shape,
                extra_out_updated.shape[-2] // lhs_batch_size,
                extra_out_updated.shape[-1],
            )
            extra_out_updated = jax.lax.reshape(extra_out_updated, extra_out_shape)

        return (
            out_updated,
            out_amax_updated,
            out_scale_updated,
            pre_gelu_out,
            bias_grad,
            extra_out_updated,
        )

    @staticmethod
    def batcher(
        batched_args,
        batch_dims,
        *,
        batched_output,
        contracting_dims,
        fuse_gelu,
        fuse_bias,
        grad,
        accumulate,
        use_split_accumulator,
        comm_overlap_config,
        sharded_abstract,
    ):
        assert CollectiveGemmPrimitive.outer_primitive is not None
        check_valid_batch_dims(batch_dims)
        (
            *_,
            bias_bdims,
            gelu_input_bdims,
            out_bdims,
            out_amax_bdims,
            out_scale_bdims,
            extra_out_bdims,
        ) = batch_dims

        return (
            CollectiveGemmPrimitive.outer_primitive.bind(
                *batched_args,
                batched_output=batched_output,
                contracting_dims=contracting_dims,
                fuse_gelu=fuse_gelu,
                fuse_bias=fuse_bias,
                grad=grad,
                accumulate=accumulate,
                use_split_accumulator=use_split_accumulator,
                comm_overlap_config=comm_overlap_config,
                sharded_abstract=sharded_abstract,
            ),
            (
                out_bdims,
                out_amax_bdims,
                out_scale_bdims,
                gelu_input_bdims,
                bias_bdims,
                extra_out_bdims,
            ),
        )

    @staticmethod
    def infer_sharding_from_operands(
        batched_output,
        contracting_dims,
        fuse_gelu,
        fuse_bias,
        grad,
        accumulate,
        use_split_accumulator,
        comm_overlap_config,
        sharded_abstract,
        mesh,
        arg_infos,
        result_infos,
    ):
        del accumulate, use_split_accumulator, sharded_abstract, result_infos
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
            comm_type = comm_overlap_config.get("comm_type", None)
            tp_resource = comm_overlap_config.get("tp_resource", global_mesh_resource().tp_resource)
            if comm_type == tex.CommOverlapType.AG:
                # AG overlap requires the outer dimension of LHS to be sharded
                # over the TP resource
                assert lhs_spec[lhs_outer_dim] == tp_resource, (
                    "AG+GEMM overlap requires the outer (sequence) dimension of the LHS "
                    + f"operand to be sharded over the TP resource '{tp_resource=}'."
                )
                assert lhs_spec[lhs_inner_dim] is None, (
                    "AG+GEMM overlap requires the contracting dimension of the LHS operand "
                    + "to be unsharded."
                )
                assert rhs_spec[rhs_inner_dim] is None, (
                    "AG+GEMM overlap requires the contracting dimension of the RHS operand "
                    + "to be unsharded."
                )
                extra_out_spec = list(lhs_spec).copy()
                extra_out_spec[lhs_outer_dim] = None

            elif comm_type == tex.CommOverlapType.RS:
                # RS overlap requires the contracting dimensions of both LHS and RHS to be
                # sharded over the TP resource, and the outer dimensions of LHS and RHS to be
                # unsharded.
                assert lhs_spec[lhs_outer_dim] is None, (
                    "GEMM+RS overlap requires the outer (sequence) dimension of the LHS "
                    + "operand to be unsharded."
                )
                assert lhs_spec[lhs_inner_dim] == tp_resource, (
                    "GEMM+RS overlap requires the contracting dimension of the LHS operand "
                    + f"to be sharded over the TP resource '{tp_resource=}'."
                )
                assert rhs_spec[rhs_inner_dim] == tp_resource, (
                    "GEMM+RS overlap requires the contracting dimension of the RHS operand "
                    + f"to be sharded over the TP resource '{tp_resource=}'."
                )
                assert rhs_spec[rhs_outer_dim] is None, (
                    "GEMM+RS overlap requires the outer dimension of the RHS operand to be "
                    + "unsharded."
                )
                extra_out_spec = list(out_spec).copy()
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
        batched_output,
        contracting_dims,
        fuse_gelu,
        fuse_bias,
        grad,
        accumulate,
        use_split_accumulator,
        comm_overlap_config,
        sharded_abstract,
        mesh,
        arg_infos,
        result_infos,
    ):
        del sharded_abstract, result_infos
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

        # Extra output sharding for comm+GEMM overlap
        extra_out_spec = [None]
        if comm_overlap_config is not None:
            comm_type = comm_overlap_config.get("comm_type", None)
            if comm_type == tex.CommOverlapType.AG:
                extra_out_spec = list(lhs_spec).copy()
                extra_out_spec[lhs_outer_dim] = None
            elif comm_type == tex.CommOverlapType.RS:
                extra_out_spec = list(out_spec).copy()
                extra_out_spec[-2] = comm_overlap_config.get(
                    "tp_resource", global_mesh_resource().tp_resource
                )
        extra_out_sharding = NamedSharding(mesh, PartitionSpec(*extra_out_spec))

        arg_shardings = (
            lhs_sharding,
            fp8_meta_sharding,
            rhs_sharding,
            fp8_meta_sharding,
            bias_sharding,
            gelu_sharding,
            out_sharding,
            fp8_meta_sharding,
            fp8_meta_sharding,
            extra_out_sharding,
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
            lhs,
            lhs_scale_inv,
            rhs,
            rhs_scale_inv,
            bias,
            gelu_input,
            out,
            out_amax,
            out_scale,
            extra_out,
        ):
            (
                out_updated,
                out_amax_updated,
                out_scale_updated,
                pre_gelu_out,
                bias_grad,
                extra_out_updated,
            ) = CollectiveGemmPrimitive.impl(
                lhs,
                lhs_scale_inv,
                rhs,
                rhs_scale_inv,
                bias,
                gelu_input,
                out,
                out_amax,
                out_scale,
                extra_out,
                batched_output=batched_output,
                contracting_dims=contracting_dims,
                fuse_gelu=fuse_gelu,
                fuse_bias=fuse_bias,
                grad=grad,
                accumulate=accumulate,
                use_split_accumulator=use_split_accumulator,
                comm_overlap_config=comm_overlap_config,
                sharded_abstract=True,
            )

            # FP8 amax reduction
            if jax_dtype_is_fp8(lhs.dtype):
                out_amax_updated = all_reduce_max_along_all_axes_except_PP(out_amax_updated, mesh)

            # All-reduce sum GEMM output when contracting dimensions are sharded
            if comm_overlap_config is None and reduce_output:
                out_updated = jax.lax.psum(out_updated, global_mesh_resource().tp_resource)
                if fuse_gelu:
                    pre_gelu_out = jax.lax.psum(pre_gelu_out, global_mesh_resource().tp_resource)

            return (
                out_updated,
                out_amax_updated,
                out_scale_updated,
                pre_gelu_out,
                bias_grad,
                extra_out_updated,
            )

        return mesh, sharded_impl, out_shardings, arg_shardings


register_primitive(CollectiveGemmPrimitive)


def gemm_impl(
    lhs: ArrayLike,
    rhs: ArrayLike,
    bias: Optional[ArrayLike] = None,
    gelu_input: Optional[ArrayLike] = None,
    out: Optional[ArrayLike] = None,
    extra_out: Optional[ArrayLike] = None,
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

    out_shape_batched = (*lhs.shape[:-2], lhs.shape[lhs_outer_dim], rhs.shape[rhs_outer_dim])
    out_shape_2d = (reduce(operator.mul, out_shape_batched[:-1], 1), out_shape_batched[-1])
    out_shape = out_shape_batched if batched_output else out_shape_2d

    if out is None:
        out = jnp.zeros(out_shape, dtype=lhs.dtype)

    if extra_out is None:
        extra_out_shape = 0
        if comm_overlap_config is not None and comm_overlap_config["method"] != "bulk":
            comm_type = comm_overlap_config["comm_type"]
            if comm_type == tex.CommOverlapType.AG:
                extra_out_shape = list(lhs.shape).copy()
            elif comm_type == tex.CommOverlapType.RS:
                extra_out_shape = list(out_shape).copy()
        extra_out = jnp.zeros(extra_out_shape, dtype=lhs.dtype)

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
        gelu_input = jnp.zeros(out_shape_2d, dtype=lhs.dtype)

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
        out,
        dummy_fp8_meta,
        dummy_fp8_meta,
        extra_out,
        batched_output=batched_output,
        contracting_dims=contracting_dims,
        fuse_gelu=fuse_gelu,
        fuse_bias=fuse_bias,
        grad=grad,
        accumulate=accumulate,
        use_split_accumulator=use_split_accumulator,
        comm_overlap_config=comm_overlap_config,
        sharded_abstract=False,
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
    out: Optional[ArrayLike] = None,
    extra_out: Optional[ArrayLike] = None,
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
    out_shape_batched = (*lhs.shape[:-2], lhs.shape[-2], rhs_t.shape[-2])
    out_shape_2d = (reduce(operator.mul, out_shape_batched[:-1], 1), out_shape_batched[-1])
    out_shape = out_shape_batched if batched_output else out_shape_2d

    if out is None:
        out = jnp.zeros(out_shape, dtype=out_dtype)
    else:
        out_dtype = out.dtype

    if extra_out is None:
        extra_out_shape = 0
        if comm_overlap_config is not None and comm_overlap_config["method"] != "bulk":
            comm_type = comm_overlap_config["comm_type"]
            if comm_type == tex.CommOverlapType.AG:
                extra_out_shape = list(lhs.shape).copy()
            elif comm_type == tex.CommOverlapType.RS:
                extra_out_shape = list(out_shape).copy()
        extra_out = jnp.zeros(extra_out_shape, dtype=jnp.bfloat16)

    if jax_dtype_is_fp8(out_dtype):
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
        gelu_input = jnp.zeros(out_shape_2d, dtype=bias.dtype)

    (out, out_amax, out_scale, pre_gelu_out, _, extra_out) = (  # bias_grad in non-FP8 GEMM
        CollectiveGemmPrimitive.outer_primitive.bind(
            lhs,
            lhs_scale_inv,
            rhs_t,
            rhs_scale_inv,
            bias,
            gelu_input,
            out,
            out_amax,
            out_scale,
            extra_out,
            batched_output=batched_output,
            contracting_dims=(-1, -1),
            fuse_gelu=fuse_gelu,
            fuse_bias=fuse_bias,
            grad=False,
            accumulate=accumulate,
            use_split_accumulator=use_split_accumulator,
            comm_overlap_config=comm_overlap_config,
            sharded_abstract=False,
        )
    )

    return out, out_amax, out_scale, pre_gelu_out, extra_out


class BootstrapCommGemmOverlapPrimitive(BasePrimitive):
    """
    Initialize Comm+GEMM overlap communicators and buffers
    """

    name = "te_bootstrap_comm_gemm_overlap_ffi"
    impl_static_args = (1,)
    multiple_results = False
    inner_primitive = None
    outer_primitive = None

    @staticmethod
    def abstract(buffer_aval, myrank, numranks, comm_overlap_config):
        del myrank, numranks
        assert is_ffi_enabled(), "Comm+GEMM overlap is supported only via XLA FFI."
        overlap_name = comm_overlap_config.get("name", None)
        assert (
            overlap_name in _COMM_GEMM_OVERLAP_NAMES
        ), f"Unrecognized comm+GEMM overlap name: {overlap_name=}"
        assert buffer_aval.size > 0, "Cannot initialize a zero-size communication buffer."
        return jax.core.ShapedArray(shape=(0,), dtype=dtypes.canonicalize_dtype(buffer_aval.dtype))

    @staticmethod
    def lowering(ctx, buffer, *, myrank, numranks, comm_overlap_config):
        return ffi.ffi_lowering(BootstrapCommGemmOverlapPrimitive.name)(
            ctx,
            buffer,
            name=comm_overlap_config["name"],
            method=comm_overlap_config["method"],
            myrank=myrank,
            numranks=numranks,
            tp_size=comm_overlap_config["tp_size"],
            num_splits=comm_overlap_config["num_splits"],
            num_max_streams=comm_overlap_config["num_max_streams"],
            cga_size=comm_overlap_config["cga_size"],
            num_comm_sm=comm_overlap_config["num_sm"],
            set_sm_margin=comm_overlap_config["set_sm_margin"],
            use_ce=comm_overlap_config["use_ce"],
            atomic_gemm=comm_overlap_config["atomic_gemm"],
            aggregate=comm_overlap_config["aggregate"],
            pipeline_rs_overlap_first_gemm=comm_overlap_config["pipeline_rs_overlap_first_gemm"],
        )

    @staticmethod
    def impl(buffer, myrank, numranks, comm_overlap_config):
        assert BootstrapCommGemmOverlapPrimitive.inner_primitive is not None
        buffer = jax.lax.reshape(
            buffer, (reduce(operator.mul, buffer.shape[:-1], 1), buffer.shape[-1])
        )
        return BootstrapCommGemmOverlapPrimitive.inner_primitive.bind(
            buffer,
            myrank=myrank,
            numranks=numranks,
            comm_overlap_config=comm_overlap_config,
        )

    @staticmethod
    def batcher(batched_args, batch_dims, *, myrank, numranks, comm_overlap_config):
        assert BootstrapCommGemmOverlapPrimitive.inner_primitive is not None
        check_valid_batch_dims(batch_dims)
        return (
            BootstrapCommGemmOverlapPrimitive.inner_primitive.bind(
                *batched_args,
                myrank=myrank,
                numranks=numranks,
                comm_overlap_config=comm_overlap_config,
            ),
            None,
        )

    @staticmethod
    def infer_sharding_from_operands(
        myrank, numranks, comm_overlap_config, mesh, arg_infos, result_infos
    ):
        del myrank, numranks, comm_overlap_config, result_infos
        buffer_spec = get_padded_spec(arg_infos[0])
        assert all([spec is None for spec in buffer_spec]), "Sample buffer must be unsharded."
        return NamedSharding(mesh, PartitionSpec(None))

    @staticmethod
    def partition(myrank, numranks, comm_overlap_config, mesh, arg_infos, result_infos):
        del arg_infos, result_infos
        arg_shardings = (NamedSharding(mesh, PartitionSpec(None)),)
        out_sharding = NamedSharding(mesh, PartitionSpec(None))
        return (
            mesh,
            partial(
                BootstrapCommGemmOverlapPrimitive.impl,
                myrank=myrank,
                numranks=numranks,
                comm_overlap_config=comm_overlap_config,
            ),
            out_sharding,
            arg_shardings,
        )


register_primitive(BootstrapCommGemmOverlapPrimitive)


def bootstrap_comm_gemm_overlap(
    buffer: ArrayLike, myrank: int, numranks: int, comm_overlap_config: dict
):
    _ = BootstrapCommGemmOverlapPrimitive.outer_primitive.bind(
        buffer, myrank=myrank, numranks=numranks, comm_overlap_config=comm_overlap_config
    )


class CopyIntoOverlapBufferPrimitive(BasePrimitive):
    """
    Copy JAX array data into comm+GEMM overlap buffer
    """

    name = "te_copy_into_overlap_buffer_ffi"
    impl_static_args = (1, 2)
    multiple_results = False
    inner_primitive = None
    outer_primitive = None

    @staticmethod
    def abstract(inp_aval, name, sharded):
        del sharded
        assert is_ffi_enabled(), "Comm+GEMM overlap is supported only via XLA FFI."
        assert name in _COMM_GEMM_OVERLAP_NAMES, f"Unrecognized comm+GEMM overlap name: {name=}"
        assert inp_aval.size > 0, "Cannot copy a zero-size array into overlap buffer."
        return jax.core.ShapedArray(shape=(0,), dtype=dtypes.canonicalize_dtype(inp_aval.dtype))

    @staticmethod
    def lowering(ctx, inp, *, name, sharded):
        return ffi.ffi_lowering(name)(
            ctx,
            inp,
            name=name,
            sharded=sharded,
        )

    @staticmethod
    def impl(inp, name, sharded):
        assert CopyIntoOverlapBufferPrimitive.inner_primitive is not None
        inp_2d = jax.lax.reshape(inp, (reduce(operator.mul, inp.shape[:-1], 1), inp.shape[-1]))
        return CopyIntoOverlapBufferPrimitive.inner_primitive.bind(
            inp_2d, name=name, sharded=sharded
        )

    @staticmethod
    def batcher(batched_args, batch_dims, *, name, sharded):
        assert CopyIntoOverlapBufferPrimitive.inner_primitive is not None
        check_valid_batch_dims(batch_dims)
        return (
            CopyIntoOverlapBufferPrimitive.inner_primitive.bind(
                *batched_args, name=name, sharded=sharded
            ),
            None,
        )

    @staticmethod
    def infer_sharding_from_operands(name, sharded, mesh, arg_infos, result_infos):
        del name, result_infos
        inp_spec = get_padded_spec(arg_infos[0])
        if sharded:
            assert inp_spec[-2] is not None, (
                "Leading dimension of input tensor must be sharded in order to copy into a "
                + "sharded communication tensor (e.g. preparing for bulk all-gather overlap)."
            )
        else:
            assert inp_spec[-2] is None, (
                "Leading dimension of input tensor cannot be sharded when copying into an "
                + "unsharded communication tensor (e.g. preparing for bulk reduce-scatter overlap)."
            )
        return NamedSharding(mesh, PartitionSpec(None))

    @staticmethod
    def partition(name, sharded, mesh, arg_infos, result_infos):
        del name, sharded, result_infos
        inp_spec = get_padded_spec(arg_infos[0])
        arg_shardings = (NamedSharding(mesh, PartitionSpec(*inp_spec)),)
        out_sharding = NamedSharding(mesh, PartitionSpec(None))
        return (
            mesh,
            partial(CopyIntoOverlapBufferPrimitive.impl, name=name, sharded=sharded),
            out_sharding,
            arg_shardings,
        )


register_primitive(CopyIntoOverlapBufferPrimitive)


def copy_into_overlap_buffer(inp: ArrayLike, name: str, sharded: bool) -> None:
    _ = CopyIntoOverlapBufferPrimitive.outer_primitive.bind(inp, name=name, sharded=sharded)
