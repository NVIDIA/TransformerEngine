# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""JAX te modules"""

import operator
from typing import Tuple, Sequence, Union, Dict
from functools import partial, reduce

import jax
from jax import numpy as jnp
from jax import dtypes
from jax.sharding import PartitionSpec, NamedSharding

import transformer_engine_jax as tex

from .misc import (
    get_padded_spec,
    jax_dtype_to_te_dtype,
    check_valid_batch_dims,
)

from .base import BasePrimitive, register_primitive

from ..quantize import (
    ScaledTensor,
    ScalingMode,
    Quantizer,
    QuantizeConfig,
    noop_quantizer_set,
)

from ..sharding import (
    global_mesh_resource,
    get_mesh_axis_size,
)


__all__ = [
    "gemm",
    "collective_gemm_impl",
    "get_default_comm_overlap_config",
    "initialize_comm_overlap",
]

min_stream_priority = None
max_stream_priority = None
num_max_comm_overlap_streams = 3
num_cublas_streams = 4


def get_cublas_workspace_size_bytes() -> None:
    """Return 32 MiB if using hopper, 4 MiB for all other architectures."""
    if tex.get_device_compute_capability(0) >= 90:
        return 33_554_432
    return 4_194_304


def is_non_tn_fp8_gemm_supported() -> bool:
    """Check if the device supports cuBLAS FP8 GEMM with non-TN layouts."""
    device_capability = tex.get_device_compute_capability(0)
    return (100 <= device_capability < 120) or (device_capability >= 130)


def sanitize_dim(dim, ndims):
    """Convert relative (negative) dimension numbers to absolute dimensions."""
    return (ndims + dim) if dim < 0 else dim


def mirror_dim(dim, ndims):
    """Return the contracting or leading dimension of a GEMM operand when given the opposite."""
    return ndims - 2 if dim == ndims - 1 else ndims - 1


def get_gemm_layout(
    contracting_dims: Tuple[int, int],
    operand_ndims: Tuple[int, int]
) -> Tuple[bool, bool]:
    """Convert JAX-style contracting dimensions to cuBLAS-style transpose flags."""
    lhs_inner, rhs_inner = map(sanitize_dim, contracting_dims, operand_ndims)
    lhs_trans = lhs_inner != operand_ndims[0] - 1
    rhs_trans = rhs_inner == operand_ndims[1] - 1
    return lhs_trans, rhs_trans


class CollectiveGemmPrimitive(BasePrimitive):
    """
    cuBlasLt GEMM Primitive w/ support for distributed inputs and communication overlap
    """

    name = "te_collective_gemm_ffi"
    impl_static_args = (7, 8, 9, 10, 11, 12, 13, 14, 15)
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
        pre_gelu_in_aval,
        aux_in_aval,
        scaling_mode,
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

        del grad, accumulate, use_split_accumulator

        # Validate operand dtypes
        lhs_dtype = dtypes.canonicalize_dtype(lhs_aval.dtype)
        rhs_dtype = dtypes.canonicalize_dtype(rhs_aval.dtype)
        assert lhs_dtype == rhs_dtype, "Mismatched matrix dtypes for GEMM."
        if scaling_mode != ScalingMode.NO_SCALING:
            assert lhs_scale_inv_aval.size > 0, "Missing LHS scale inverse in quantized GEMM."
            assert rhs_scale_inv_aval.size > 0, "Missing RHS scale inverse in quantized GEMM."

        # Validate operand layouts
        lhs_inner_dim, rhs_inner_dim = map(sanitize_dim, contracting_dims,
                                           (lhs_aval.ndim, rhs_aval.ndim))
        lhs_outer_dim, rhs_outer_dim = map(mirror_dim, (lhs_inner_dim, rhs_inner_dim),
                                           (lhs_aval.ndim, rhs_aval.ndim))
        lhs_trans = lhs_inner_dim != lhs_aval.ndim - 1
        rhs_trans = rhs_inner_dim == rhs_aval.ndim - 1
        assert not (
            lhs_trans and rhs_trans
        ), "GEMM does not support transposed LHS and transposed RHS at the same time."
        lhs_2d_shape = (reduce(operator.mul, lhs_aval.shape[:-1]), lhs_aval.shape[-1])
        rhs_2d_shape = (reduce(operator.mul, rhs_aval.shape[:-1]), rhs_aval.shape[-1])
        assert lhs_2d_shape[int(not lhs_trans)] == rhs_2d_shape[rhs_inner_dim], (
            "Incompatible operand dimensions: "
            + f"{lhs_aval.shape} @ idx {lhs_inner_dim} X {rhs_aval.shape} @ idx {rhs_inner_dim}."
        )

        if scaling_mode != ScalingMode.NO_SCALING and not is_non_tn_fp8_gemm_supported():
            assert not lhs_trans, "FP8 GEMM does not support transposed LHS."
            assert rhs_trans, "FP8 GEMM requires transposed RHS."

        # Determine output shape and dtype
        out_dtype = (
            dtypes.canonicalize_dtype(jnp.bfloat16)
            if scaling_mode != ScalingMode.NO_SCALING
            else lhs_dtype
        )
        out_shape = [
            *lhs_aval.shape[:-2],
            lhs_aval.shape[lhs_outer_dim],
            rhs_aval.shape[rhs_outer_dim]
        ]
        if comm_overlap_config is not None and sharded_abstract:
            # Modify output shape only for AG overlap. RS overlap output goes into the auxiliary
            # buffer, not this one.
            if comm_overlap_config["comm_type"] == tex.CommOverlapType.AG:
                out_shape[-2] *= comm_overlap_config["tp_size"]

        # Set auxiliary output and workspace shape
        aux_out_shape = [ 0 ]
        aux_out_dtype = jnp.bfloat16
        workspace_size = get_cublas_workspace_size_bytes()
        if comm_overlap_config is not None:
            comm_type = comm_overlap_config.get("comm_type", None)
            assert comm_type is not None, "Missing comm type for comm+GEMM overlap."

            tp_size = comm_overlap_config.get("tp_size", 1)
            assert (
                tp_size > 1
            ), "Comm+GEMM overlap requires tensor-parallel mesh axis size greater than 1."

            if comm_overlap_config["method"] == tex.CommOverlapMethod.BULK:
                # Auxiliary output shape is the gathered or scattered shape of the auxiliary input
                aux_out_shape = list(aux_in_aval.shape).copy()
                aux_out_dtype = dtypes.canonicalize_dtype(aux_in_aval.dtype)
                if comm_type == tex.CommOverlapType.AG:
                    aux_out_shape[-2] *= tp_size
                else:
                    aux_out_shape[-2] = aux_out_shape[-2] // tp_size
            else:
                # Increase workspace size to ensure every GEMM chunk has an independent workspace
                # of the appropriate size
                workspace_size *= num_max_comm_overlap_streams

                if comm_type == tex.CommOverlapType.AG and aux_in_aval.size > 0:
                    # Auxiliary output shape is the gathered shape of the LHS operand
                    aux_out_shape = list(lhs_aval.shape).copy()
                    aux_out_dtype = lhs_dtype
                    if sharded_abstract:
                        aux_out_shape[-2] *= tp_size
                    assert (
                        all([aux_in_aval.shape[i] == aux_out_shape[i]
                             for i in range(len(aux_out_shape))])
                    ), (
                        "Auxiliary buffer for all-gathered LHS copy has incorrect shape, "
                        f"expected {aux_out_shape} but found {aux_in_aval.shape}."
                    )
                    aux_in_dtype = dtypes.canonicalize_dtype(aux_in_aval.dtype)
                    assert aux_in_dtype == aux_out_dtype, (
                        "Auxiliary buffer for all-gathered LHS copy has incorrect data type, "
                        f"expected {aux_out_dtype} but found {aux_in_dtype}"
                    )

                elif comm_type == tex.CommOverlapType.RS:
                    # Auxiliary output shape is the scattered shape of the GEMM output
                    aux_out_shape = list(out_shape).copy()
                    if sharded_abstract:
                        aux_out_shape[-2] = aux_out_shape[-2] // tp_size

        # Validate bias shape
        bias_shape = (0, )
        if fuse_bias:
            if not grad:
                assert bias_aval.ndim == 1 and bias_aval.size == out_shape[-1], (
                    f"Incorrect bias size, expected {out_shape[-1]} but found {bias_aval.size}."
                )
                bias_aval_dtype = dtypes.canonicalize_dtype(bias_aval.dtype)
                assert bias_aval_dtype == out_dtype, (
                    f"Incorrect bias data dtype, expected {out_dtype} but found {bias_aval_dtype}."
                )
            else:
                bias_shape = (out_shape[-1], )

        # Validate pre-GELU output
        gelu_shape = (0, )
        if fuse_gelu:
            if grad:
                assert (
                    all([pre_gelu_in_aval.shape[i] == out_shape[i]
                         for i in range(len(gelu_shape))])
                ), (
                    "Incorrect pre-GeLU output shape, "
                    f"expected {out_shape} but found {pre_gelu_in_aval.shape}"
                )
                pre_gelu_out_dtype = dtypes.canonicalize_dtype(pre_gelu_in_aval.dtype)
                assert pre_gelu_out_dtype == out_dtype, (
                    "Incorrect pre-GeLU output data type, "
                    f"expected {out_dtype} but found {pre_gelu_out_dtype}."
                )
            else:
                gelu_shape = list(out_shape).copy()

        # Create abstract arrays for all outputs
        out_aval = jax.core.ShapedArray(shape=out_shape, dtype=out_dtype)
        bias_grad_aval = jax.core.ShapedArray(shape=bias_shape, dtype=out_dtype)
        pre_gelu_out_aval = jax.core.ShapedArray(shape=gelu_shape, dtype=out_dtype)
        aux_out_aval = jax.core.ShapedArray(shape=aux_out_shape, dtype=aux_out_dtype)
        workspace_aval = jax.core.ShapedArray(shape=(workspace_size,), dtype=jnp.uint8)

        return (
            out_aval,
            bias_grad_aval,
            pre_gelu_out_aval,
            aux_out_aval,
            workspace_aval,
        )

    @staticmethod
    def outer_abstract(*args, **kwargs):
        """
        cuBlasLt GEMM outer abstract
        """
        outputs = CollectiveGemmPrimitive.abstract(*args, **kwargs)
        return outputs[:-1]  # discard cuBLAS workspace

    @staticmethod
    def lowering(
        ctx,
        lhs,
        lhs_scale_inv,
        rhs,
        rhs_scale_inv,
        bias,
        pre_gelu_in,
        aux_in,
        scaling_mode,
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
        del sharded_abstract
        lhs_aval, _, rhs_aval, *_ = ctx.avals_in
        lhs_trans, rhs_trans = get_gemm_layout(contracting_dims, (lhs_aval.ndim, rhs_aval.ndim))

        args = (
            lhs,
            lhs_scale_inv,
            rhs,
            rhs_scale_inv,
            bias,
            pre_gelu_in,
            aux_in,
        )

        kwargs = {
            "scaling_mode" : int(scaling_mode.value),
            "lhs_trans" : lhs_trans,
            "rhs_trans" : rhs_trans,
            "fuse_gelu" : fuse_gelu,
            "fuse_bias" : fuse_bias,
            "grad" : grad,
            "accumulate" : accumulate,
            "use_split_accumulator" : use_split_accumulator,
            "comm_overlap_id" : -1,
            "comm_overlap_method" : int(tex.CommOverlapMethod.NONE),
            "comm_type" : int(tex.CommOverlapType.NONE),
        }

        operand_output_aliases = {}
        if fuse_bias and not grad:
            operand_output_aliases.update({ 4 : 1 })  # bias <-> bias_grad
        if fuse_gelu and grad:
            operand_output_aliases.update({ 5 : 2 })  # pre_gelu_in <-> pre_gelu_out

        if comm_overlap_config is not None:
            kwargs["comm_overlap_id"] = comm_overlap_config["unique_id"]
            kwargs["comm_overlap_method"] = int(comm_overlap_config["method"])
            kwargs["comm_type"] = int(comm_overlap_config["comm_type"])

        return jax.ffi.ffi_lowering(
            CollectiveGemmPrimitive.name,
            operand_output_aliases=operand_output_aliases,
        )(ctx, *args, **kwargs)

    @staticmethod
    def impl(
        lhs,
        lhs_scale_inv,
        rhs,
        rhs_scale_inv,
        bias,
        pre_gelu_in,
        aux_in,
        scaling_mode,
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

        outputs = CollectiveGemmPrimitive.inner_primitive.bind(
            lhs,
            lhs_scale_inv,
            rhs,
            rhs_scale_inv,
            bias,
            pre_gelu_in,
            aux_in,
            scaling_mode=scaling_mode,
            contracting_dims=contracting_dims,
            fuse_gelu=fuse_gelu,
            fuse_bias=fuse_bias,
            grad=grad,
            accumulate=accumulate,
            use_split_accumulator=use_split_accumulator,
            comm_overlap_config=comm_overlap_config,
            sharded_abstract=sharded_abstract,
        )

        return outputs[:-1]  # discard cuBLAS workspace

    @staticmethod
    def batcher(
        batched_args,
        batch_dims,
        *,
        scaling_mode,
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
        lhs_bdims, *_, aux_in_bdims = batch_dims
        lhs, _, rhs, *_ = batched_args
        lhs_trans, rhs_trans = get_gemm_layout(contracting_dims, (lhs.ndim, rhs.ndim))

        out_bdims = (None, ) if lhs.ndim > 2 and lhs_trans and not rhs_trans else lhs_bdims
        bias_grad_bdims = (None, )
        pre_gelu_out_bdims = (None, )
        if fuse_gelu and grad:
            pre_gelu_out_bdims = out_bdims
        aux_out_bdims = aux_in_bdims
        if (
            comm_overlap_config is not None
            and comm_overlap_config["method"] != tex.CommOverlapMethod.BULK
        ):
            aux_out_bdims = lhs_bdims

        return (
            CollectiveGemmPrimitive.outer_primitive.bind(
                *batched_args,
                scaling_mode=scaling_mode,
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
                bias_grad_bdims,
                pre_gelu_out_bdims,
                aux_out_bdims,
            ),
        )

    @staticmethod
    def infer_sharding_from_operands(
        scaling_mode,
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
        del scaling_mode, accumulate, use_split_accumulator, sharded_abstract, result_infos
        lhs, _, rhs, *_, aux_in = arg_infos
        lhs_spec, rhs_spec = map(get_padded_spec, [lhs, rhs])

        lhs_inner_dim, rhs_inner_dim = map(sanitize_dim, contracting_dims, (lhs.ndim, rhs.ndim))
        lhs_outer_dim, rhs_outer_dim = map(
            mirror_dim,
            (lhs_inner_dim, rhs_inner_dim),
            (lhs.ndim, rhs.ndim),
        )

        # Check operand shardings
        assert not (lhs_spec[lhs_inner_dim] is not None and lhs_spec[lhs_outer_dim] is not None), (
            "LHS operand cannot be sharded in both leading and contracting dimensions."
        )
        assert not (rhs_spec[rhs_inner_dim] is not None and rhs_spec[rhs_outer_dim] is not None), (
            "RHS operand cannot be sharded in both leading and contracting dimensions."
        )
        assert lhs_spec[lhs_inner_dim] == rhs_spec[rhs_inner_dim], (
            "Contracting dimensions of GEMM operands must have the same sharding."
        )

        # Output sharding
        out_spec = (*lhs_spec[:-2], None, rhs_spec[rhs_outer_dim])
        out_sharding = NamedSharding(mesh, PartitionSpec(*out_spec))

        # Bias sharding
        bias_spec = (None, )
        if fuse_bias and grad:
            bias_spec = (out_spec[-1], )
        bias_sharding = NamedSharding(mesh, PartitionSpec(*bias_spec))

        # Pre-GELU output is always 2D if GELU fusion is turned on, otherwise unsharded
        pre_gelu_spec = (None, )
        if fuse_gelu and not grad:
            pre_gelu_spec = out_spec
        pre_gelu_sharding = NamedSharding(mesh, PartitionSpec(*pre_gelu_spec))

        # Auxiliary output sharding
        aux_in_spec = get_padded_spec(aux_in)
        aux_out_spec = (None, )
        if comm_overlap_config is not None:
            if comm_overlap_config["method"] == tex.CommOverlapMethod.BULK:
                aux_out_spec = aux_in_spec
            elif comm_overlap_config["comm_type"] == tex.CommOverlapType.RS:
                aux_out_spec = list(out_spec).copy()
                aux_out_spec[-2] = comm_overlap_config["tp_resource"]
            elif aux_in.size > 0:
                aux_out_spec = list(lhs_spec).copy()
                aux_out_spec[-2] = None
        aux_out_sharding = NamedSharding(mesh, PartitionSpec(*aux_out_spec))

        return [
            out_sharding,
            bias_sharding,
            pre_gelu_sharding,
            aux_out_sharding,
        ]

    @staticmethod
    def partition(
        scaling_mode,
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
        lhs, lhs_scale_inv, rhs, rhs_scale_inv, *_, aux_in = arg_infos
        lhs_inner_dim = sanitize_dim(contracting_dims[0], lhs.ndim)
        lhs_outer_dim = mirror_dim(lhs_inner_dim, lhs.ndim)

        # First get the output shardings
        out_shardings = CollectiveGemmPrimitive.infer_sharding_from_operands(
            scaling_mode, contracting_dims, fuse_gelu, fuse_bias, grad, accumulate,
            use_split_accumulator, comm_overlap_config, sharded_abstract, mesh, arg_infos,
            result_infos
        )
        out_spec = out_shardings[0].spec

        # Operand specs
        lhs_spec = list(get_padded_spec(lhs)).copy()
        rhs_spec = get_padded_spec(rhs)
        if comm_overlap_config is None:
            # Always all-gather the outer dimension of LHS when there is no comm+GEMM overlap
            lhs_spec[lhs_outer_dim] = None
        lhs_sharding = NamedSharding(mesh, PartitionSpec(*lhs_spec))
        rhs_sharding = NamedSharding(mesh, PartitionSpec(*rhs_spec))

        # Quantization scales are always unsharded
        lhs_scale_inv_spec = len(get_padded_spec(lhs_scale_inv)) * [ None ]
        lhs_scale_inv_sharding = NamedSharding(mesh, PartitionSpec(*lhs_scale_inv_spec))
        rhs_scale_inv_spec = len(get_padded_spec(rhs_scale_inv)) * [ None ]
        rhs_scale_inv_sharding = NamedSharding(mesh, PartitionSpec(*rhs_scale_inv_spec))

        # Bias is sharded to match trailing dimension of the GEMM output
        bias_spec = (None, )
        if fuse_bias and not grad:
            bias_spec = (out_spec[-1], )
        bias_sharding = NamedSharding(mesh, PartitionSpec(*bias_spec))

        # Pre-GELU output is always 2D if GELU fusion is turned on, otherwise unsharded
        pre_gelu_spec = (None, )
        if fuse_gelu and grad:
            pre_gelu_spec = out_spec
        pre_gelu_sharding = NamedSharding(mesh, PartitionSpec(*pre_gelu_spec))

        # Auxiliary input sharding
        aux_in_spec = get_padded_spec(aux_in)
        aux_in_sharding = NamedSharding(mesh, PartitionSpec(*aux_in_spec))

        arg_shardings = (
            lhs_sharding,
            lhs_scale_inv_sharding,
            rhs_sharding,
            rhs_scale_inv_sharding,
            bias_sharding,
            pre_gelu_sharding,
            aux_in_sharding,
        )

        def sharded_impl(
            lhs,
            lhs_scale_inv,
            rhs,
            rhs_scale_inv,
            bias,
            pre_gelu_in,
            aux_in,
        ):
            (
                output,
                bias_grad,
                pre_gelu_out,
                aux_out,
            ) = CollectiveGemmPrimitive.impl(
                lhs,
                lhs_scale_inv,
                rhs,
                rhs_scale_inv,
                bias,
                pre_gelu_in,
                aux_in,
                scaling_mode=scaling_mode,
                contracting_dims=contracting_dims,
                fuse_gelu=fuse_gelu,
                fuse_bias=fuse_bias,
                grad=grad,
                accumulate=accumulate,
                use_split_accumulator=use_split_accumulator,
                comm_overlap_config=comm_overlap_config,
                sharded_abstract=True,
            )

            # All-reduce sum GEMM output when contracting dimensions are sharded
            rhs_inner_dim = sanitize_dim(contracting_dims[1], rhs.ndim)
            if comm_overlap_config is None and rhs_spec[rhs_inner_dim] is not None:
                output = jax.lax.psum(output, global_mesh_resource().tp_resource)
                if fuse_gelu and not grad:
                    pre_gelu_out = jax.lax.psum(pre_gelu_out, global_mesh_resource().tp_resource)

            return output, bias_grad, pre_gelu_out, aux_out

        return mesh, sharded_impl, out_shardings, arg_shardings


register_primitive(CollectiveGemmPrimitive)


@partial(jax.jit, static_argnums=(7, 8, 9, 10, 11, 12, 13, 14, 15))
def _collective_gemm_custom_call(lhs, lhs_scale_inv, rhs, rhs_scale_inv, bias, pre_gelu_out, aux_in,
                                 scaling_mode, contracting_dims, fuse_bias, fuse_gelu, grad,
                                 accumulate, use_split_accumulator, comm_overlap_config_keys,
                                 comm_overlap_config_values):
    # Reconstruct the comm+GEMM overlap config dictionary from its key/value pairs
    comm_overlap_config = None
    if comm_overlap_config_keys is not None and comm_overlap_config_values is not None:
        comm_overlap_config = dict(zip(comm_overlap_config_keys, comm_overlap_config_values))

    return CollectiveGemmPrimitive.outer_primitive.bind(
        lhs,
        lhs_scale_inv,
        rhs,
        rhs_scale_inv,
        bias,
        pre_gelu_out,
        aux_in,
        scaling_mode=scaling_mode,
        contracting_dims=contracting_dims,
        fuse_bias=fuse_bias,
        fuse_gelu=fuse_gelu,
        grad=grad,
        accumulate=accumulate,
        use_split_accumulator=use_split_accumulator,
        comm_overlap_config=comm_overlap_config,
        sharded_abstract=False,
    )


def collective_gemm_impl(
    lhs: jax.Array,
    rhs: jax.Array,
    contracting_dims: Tuple[int, int] = (-1, 0),
    scaling_mode: ScalingMode = ScalingMode.NO_SCALING,
    lhs_scale_inv: jax.Array = None,
    rhs_scale_inv: jax.Array = None,
    grad: bool = False,
    fuse_bias: bool = False,
    bias: jax.Array = None,
    fuse_gelu: bool = False,
    pre_gelu_out: jax.Array = None,
    accumulate: bool = False,
    use_split_accumulator: bool = False,
    comm_overlap_config: dict = None,
    aux_in: jax.Array = None,
) -> Tuple[jax.Array, ...]:
    r"""
    cuBlasLt GEMM w/ support for distributed inputs and communication overlap

    Parameters
    ----------
    lhs: jax.Array
        Left-hand side operand.
    rhs: jax.Array
        Right-hand side operand.
    contracting_dims: Tuple[int, int], default = (-1, 0)
        Inner dimensions in the matrix multiplication. FP8 operands on Hopper are only supported
        with `(0, 0)` contracting dimensions.
    scaling_mode: ScalingMode, default = ScalingMode.NO_SCALING
        Scaling type for quantized GEMM.
    lhs_scale_inv: jax.Array, default = None
        Inverse scale for quantized LHS operand.
    rhs_scale_inv: jax.Array, default = None
        Inverse scale for quantized RHS operand.
    grad: bool, default = False
        Flag for switching bias and/or GeLU fusions in the prologue to backward mode.
    fuse_bias: bool, default = False
        Fuse bias addition or bias gradient into the GEMM prologue.
    bias: jax.Array, default = None
        Additive bias term, required when `fuse_bias=True` and `grad=False`.
    fuse_gelu: bool, default = False
        Fuse GeLU activation into the GEMM prologue.
    pre_gelu_out: jax.Array, default = None
        Pre-GeLU GEMM output, required when `fuse_gelu=True` and `grad=True`.
    accumulate: bool, default = False
        Accumulate the result directly into the output buffer.
    use_split_accumulator: bool, default = False
        Use split accumulator for FP8 GEMM.
    comm_overlap_config: dict, default = None
        Communication overlap options. If the operands are distributed but overlap config is `None`,
        XLA schedules blocking collectives before or after the GEMM custom call.
    aux_in: jax.Array, default = None
        Auxiliary input for BULK comm+GEMM overlap.
    """
    # Replace missing arrays with empties
    if lhs_scale_inv is None:
        lhs_scale_inv = jnp.empty(0, dtype=jnp.float32)

    if rhs_scale_inv is None:
        rhs_scale_inv = jnp.empty(0, dtype=jnp.float32)

    if bias is None:
        bias = jnp.empty(0, dtype=jnp.bfloat16)

    if pre_gelu_out is None:
        pre_gelu_out = jnp.empty(0, dtype=jnp.bfloat16)

    if aux_in is None:
        aux_in = jnp.empty(0, dtype=jnp.bfloat16)

    # Split comm+GEMM overlap dictionary into hashable tuples of keys and values
    # so that JAX can JIT the underlying custom call.
    comm_overlap_keys = None
    comm_overlap_values = None
    if comm_overlap_config is not None:
        comm_overlap_keys = tuple(comm_overlap_config.keys())
        comm_overlap_values = tuple(comm_overlap_config.values())

    return _collective_gemm_custom_call(
        lhs,
        lhs_scale_inv,
        rhs,
        rhs_scale_inv,
        bias,
        pre_gelu_out,
        aux_in,
        scaling_mode,
        contracting_dims,
        fuse_bias,
        fuse_gelu,
        grad,
        accumulate,
        use_split_accumulator,
        comm_overlap_keys,
        comm_overlap_values,
    )


def get_default_comm_overlap_config(
    method: tex.CommOverlapMethod,
    comm_type: tex.CommOverlapType,
    tp_size: int,
) -> dict:
    """Returns a config dictionary with default options for the given overlap method."""
    if comm_type == tex.CommOverlapType.AG:
        assert method == tex.CommOverlapMethod.RING_EXCHANGE, (
            "All-gather overlap is only supported with the ring-exchange method."
        )

    global min_stream_priority, max_stream_priority
    if min_stream_priority is None or max_stream_priority is None:
        min_stream_priority, max_stream_priority = tex.get_stream_priority_range()

    return {
        "num_splits": tp_size if method == tex.CommOverlapMethod.RING_EXCHANGE else 4,
        "num_max_streams": num_max_comm_overlap_streams,
        "comm_cga_size": 1 if method == tex.CommOverlapMethod.RING_EXCHANGE else 2,
        "comm_priority": max_stream_priority,
        "gemm_priority": min_stream_priority,
        "num_comm_sm": 1 if method == tex.CommOverlapMethod.RING_EXCHANGE else 16,
        "set_sm_margin": not method == tex.CommOverlapMethod.RING_EXCHANGE,
        "use_ce": True,
        "atomic_gemm": False,
        "rs_overlap_first_gemm": False,
        "aggregate_ag": False,
    }


def initialize_comm_overlap(
    buffer_shape: Sequence[int],
    buffer_dtype: jnp.dtype,
    mesh: jax.sharding.Mesh,
    tp_resource: str,
    comm_type: tex.CommOverlapType,
    method: tex.CommOverlapMethod,
    lhs_grad: Union[bool, dict] = True,
    rhs_grad: Union[bool, dict] = True,
    save_gathered_lhs_for_backward: bool = False,
    **kwargs: dict,
) -> dict:
    r"""
    Initializes a comm+GEMM overlap buffer and returns an identifier based on a hash of the
    buffer's shape, data type and the overlap configuration options. Buffer creation is skipped if
    a buffer with the same hashed identifier already exists.

    Parameters
    ----------
    buffer_shape: Sequence[int]
        Communication buffer shape. For all-gather overlap, this should be the LHS operand's global
        shape. For reduce-scatter overlaps, this should be the GEMM output's global shape. For bulk
        overlaps, this should be the global shape of the auxiliary input to collective GEMM.
    buffer_dtype: jnp.dtype
        Transformer Engine data type for the communication buffer.
    mesh: jax.sharding.Mesh
        JAX Mesh with a `tp_resource` axis.
    tp_resource: str,
        Name of the mesh axis used for tensor-parallelism.
    comm_type: tex.CommOverlapType
        Collective communication type to overlap with compute.
    method: tex.CommOverlapMethod
        Implementation method for the communication overlap algorithms.
    lhs_grad: Union[bool, dict], default = True
        Flag for controlling whether this call also allocated the backward-pass buffer for
        communication overlap with the LHS operand gradient. The buffer config options can be
        controlled by passing a dictionary into this option instead of a boolean flag.
    rhs_grad: Union[bool, dict], default = True
        Flag for controlling whether this call also allocated the backward-pass buffer for
        communication overlap with the RHS operand gradient. The buffer config options can be
        controlled by passing a dictionary into this option instead of a boolean flag.
    save_gathered_lhs_for_backward: bool, default = False
        Optional optimization for saving the gathered LHS operand during the all-gather overlap
        in the forward pass, in order to re-use it in the RHS gradient computation in the backward
        pass. This avoids an all-gather in the backward pass at the expense of storing the global
        global LHS operand in the autograd context.
    kwargs: dict, default = {}
        Communication overlap configuration options. Any option not defined here falls back on
        default values set by `get_default_comm_overlap_config()`.
    """
    if method == tex.CommOverlapMethod.BULK:
        assert not lhs_grad and not rhs_grad, (
            "Bulk-overlap in the forward pass does not have matching overlaps in the backward pass."
        )

    if save_gathered_lhs_for_backward:
        assert (
            comm_type == tex.CommOverlapType.AG
            and method == tex.CommOverlapMethod.RING_EXCHANGE
        ), (
            "Saving gathered LHS operand for the backward pass is only supported for ring-exchange "
            "all-gather overlap."
        )

    global num_max_comm_overlap_streams
    num_max_comm_overlap_streams = max(num_max_comm_overlap_streams,
                                       kwargs.get("num_max_streams", num_max_comm_overlap_streams))
    tp_size = get_mesh_axis_size(tp_resource, mesh=mesh)
    config = get_default_comm_overlap_config(method, comm_type, tp_size)
    config.update((k, kwargs[k]) for k in config.keys() & kwargs.keys())

    buffer_te_dtype = jax_dtype_to_te_dtype(buffer_dtype)
    buffer_2d_shape = (reduce(operator.mul, buffer_shape[:-1]), buffer_shape[-1])
    config["unique_id"] = tex.create_comm_overlap_buffer(
        comm_type, method, buffer_2d_shape, buffer_te_dtype, tp_size, **config
    )
    config["mesh"] = mesh
    config["tp_resource"] = tp_resource
    config["tp_size"] = tp_size
    config["save_gathered_lhs"] = save_gathered_lhs_for_backward
    config["comm_type"] = comm_type
    config["method"] = method

    config["lhs_grad"] = None
    if lhs_grad:
        lhs_grad_comm_type = tex.CommOverlapType.AG
        lhs_grad_method = (
            tex.CommOverlapMethod.RING_EXCHANGE
            if lhs_grad_comm_type == tex.CommOverlapType.RS
            else tex.CommOverlapMethod.BULK
        )

        # Override default method/comm with user selection
        if isinstance(lhs_grad, dict):
            user_lhs_grad_comm_type = lhs_grad.get("comm_type", lhs_grad_comm_type)
            user_lhs_grad_method = lhs_grad.get("method", lhs_grad_method)
            if comm_type == tex.CommOverlapType.AG:
                if user_lhs_grad_method == tex.CommOverlapMethod.BULK:
                    assert user_lhs_grad_comm_type == tex.CommOverlapType.AG, (
                        "Bulk-overlapped collective type with LHS_GRAD for AG+FPROP must be an "
                        "all-gather."
                    )
                else:
                    assert user_lhs_grad_comm_type == tex.CommOverlapType.RS, (
                        "Overlapped collective typo with LHS_GRAD for AG+FPROP must be a "
                        "reduce-scatter."
                    )
            else:
                assert (
                    user_lhs_grad_method == tex.CommOverlapMethod.RING_EXCHANGE
                    and user_lhs_grad_comm_type == tex.CommOverlapType.AG
                ), (
                    "LHS_GRAD for FPROP+RS can only overlap with a ring-exchange all-gather."
                )
            lhs_grad_method = user_lhs_grad_method
            lhs_grad_comm_type = user_lhs_grad_comm_type

        lhs_grad_config = get_default_comm_overlap_config(lhs_grad_comm_type, lhs_grad_method,
                                                          tp_size)

        if isinstance(lhs_grad, dict):
            lhs_grad_config.update(
                (k, lhs_grad[k]) for k in lhs_grad_config.keys() & lhs_grad.keys()
            )

        lhs_grad_config["unique_id"] = tex.create_comm_overlap_buffer(
            lhs_grad_comm_type, lhs_grad_method, buffer_2d_shape, buffer_te_dtype, tp_size,
            lhs_grad = False, rhs_grad = False, **lhs_grad_config
        )
        lhs_grad_config["mesh"] = mesh
        lhs_grad_config["tp_resource"] = tp_resource
        lhs_grad_config["tp_size"] = tp_size
        lhs_grad_config["save_gathered_lhs"] = False
        lhs_grad_config["lhs_grad"] = None
        lhs_grad_config["rhs_grad"] = None
        lhs_grad_config["method"] = lhs_grad_method
        lhs_grad_config["comm_type"] = lhs_grad_comm_type

        config["lhs_grad"] = lhs_grad_config

    config["rhs_grad"] = None
    if (
        rhs_grad
        and comm_type == tex.CommOverlapType.AG
        and method != tex.CommOverlapMethod.BULK
        and lhs_grad
        and lhs_grad_config["method"] == tex.CommOverlapMethod.BULK
    ):
        rhs_grad_comm_type = tex.CommOverlapType.RS
        rhs_grad_method = tex.CommOverlapMethod.BULK

        rhs_grad_config = get_default_comm_overlap_config(rhs_grad_comm_type, rhs_grad_method,
                                                          tp_size)

        if isinstance(rhs_grad, dict):
            assert (
                rhs_grad.get("method", rhs_grad_method) == tex.CommOverlapMethod.BULK
                and rhs_grad.get("comm_type", rhs_grad_comm_type) == tex.CommOverlapType.RS
            ), (
                "RHS_GRAD for AG+FPROP can overlap only with bulk reduce-scatter, and only when "
                "LHS_GRAD overlaps with bulk all-gather."
            )

            rhs_grad_config.update(
                (k, rhs_grad[k]) for k in rhs_grad_config.keys() & rhs_grad.keys()
            )

        rhs_grad_config["unique_id"] = tex.create_comm_overlap_buffer(
            rhs_grad_comm_type, rhs_grad_method, buffer_2d_shape, buffer_te_dtype, tp_size,
            lhs_grad = False, rhs_grad = False, **rhs_grad_config
        )
        rhs_grad_config["mesh"] = mesh
        rhs_grad_config["tp_resource"] = tp_resource
        rhs_grad_config["tp_size"] = tp_size
        rhs_grad_config["save_gathered_lhs"] = False
        rhs_grad_config["lhs_grad"] = None
        rhs_grad_config["rhs_grad"] = None
        rhs_grad_config["comm_type"] = rhs_grad_comm_type
        rhs_grad_config["method"] = rhs_grad_method

        config["rhs_grad"] = rhs_grad_config

    return config


class GroupedGemmPrimitive(BasePrimitive):
    """
    Primitive for grouped GEMM
    """

    name = "te_grouped_gemm_ffi"
    multiple_results = True
    impl_static_args = ()
    inner_primitive = None
    outer_primitive = None

    @staticmethod
    def abstract(*args, num_gemms, scaling_mode, out_dtype, has_bias):
        """
        Args:
            *args: Size num_gemms * 4 or num_gemms * 5 depending on has_bias:
                args[  0         :   num_gemms] are the lhs tensors,
                args[  num_gemms : 2*num_gemms] are the rhs tensors,
                args[2*num_gemms : 3*num_gemms] are the lhs scale_inv tensors,
                args[3*num_gemms : 4*num_gemms] are the rhs scale_inv tensors,
                args[4*num_gemms : 5*num_gemms] are the bias tensors if has_bias is True.
            num_gemms: Number of GEMM operations to perform.
            scaling_mode: Scaling mode for the GEMM operations.
            out_dtype: Data type of the output tensors.
            has_bias: Boolean indicating if bias tensors are provided.

        Returns:
           A tuple of ShapedArray objects of size num_gemms+1:
               ret[0 : num_gemms]: GEMM output tensors,
               ret[num_gemms]:workspace tensor.
        """
        del scaling_mode
        expected_num_args = 5 * num_gemms if has_bias else 4 * num_gemms
        assert (
            len(args) == expected_num_args
        ), f"Expected {expected_num_args} input arguments, but got {len(args)}"
        A_list = args[0:num_gemms]
        B_list = args[num_gemms : 2 * num_gemms]
        # A and B have shapes [1, m, k] and [1, n, k]
        out_list_aval = tuple(
            jax.core.ShapedArray((A.shape[1], B.shape[1]), dtype=out_dtype)
            for A, B in zip(A_list, B_list)
        )
        workspace_size = get_cublas_workspace_size_bytes() * num_cublas_streams
        workspace_aval = jax.core.ShapedArray(shape=(workspace_size,), dtype=jnp.uint8)
        return (*out_list_aval, workspace_aval)

    @staticmethod
    def outer_abstract(*args, **kwargs):
        (out_aval, _) = GroupedGemmPrimitive.abstract(*args, **kwargs)
        return out_aval

    @staticmethod
    def lowering(ctx, *args, num_gemms, scaling_mode, out_dtype, has_bias):
        del out_dtype
        return jax.ffi.ffi_lowering(GroupedGemmPrimitive.name)(
            ctx,
            *args,
            num_gemms=num_gemms,
            scaling_mode=int(scaling_mode),
            has_bias=has_bias,
        )

    @staticmethod
    def impl(*args, num_gemms, scaling_mode, out_dtype, has_bias):
        assert GroupedGemmPrimitive.inner_primitive is not None
        out = GroupedGemmPrimitive.inner_primitive.bind(
            *args,
            num_gemms=num_gemms,
            scaling_mode=scaling_mode.value,
            out_dtype=out_dtype,
            has_bias=has_bias,
        )
        return out[:-1]  # out is [out_list, wkspace], only return out_list


register_primitive(GroupedGemmPrimitive)


def _shape_normalization(x, dimension_numbers, already_transposed: bool = False):
    orig_order = list(range(x.ndim))
    contracting_dims, batch_dims = dimension_numbers
    contracting_order = [d for d in orig_order if d in contracting_dims]
    batch_order = [d for d in orig_order if d in batch_dims]
    non_contracting_order = [
        d for d in orig_order if d not in contracting_dims and d not in batch_dims
    ]
    batch_shape = [x.shape[d] for d in batch_order]
    rows_shape = [x.shape[d] for d in non_contracting_order]
    cols_shape = [x.shape[d] for d in contracting_order]
    new_order = batch_order + non_contracting_order + contracting_order
    rows, cols, batches = (
        reduce(operator.mul, rows_shape, 1),
        reduce(operator.mul, cols_shape, 1),
        reduce(operator.mul, batch_shape, 1),
    )
    # Remove this transpose when non-TN dot is supported
    if not already_transposed:
        t = jnp.transpose(x, new_order)
    else:
        t = x
    return jnp.reshape(t, (batches, rows, cols))


def _calculate_remaining_shape(shape, contracting_dims):
    return tuple(shape[dim] for dim in range(len(shape)) if dim not in contracting_dims)


def _transpose_contract_dims(ndim, contracting_dims):
    return tuple(ndim - i - 1 for i in contracting_dims)[::-1]


# Apply jit to guarantee correctness of FP8 GEMM.
@partial(jax.jit, static_argnums=(2, 3))
def _jax_gemm_tensor_scaling_fp8(lhs, rhs, dim_nums, precision):
    (lhs_contract, rhs_contract), (lhs_batch, rhs_batch) = dim_nums
    if lhs.data_layout == "T":
        lhs_contract = _transpose_contract_dims(lhs.data.ndim, lhs_contract)
    if rhs.data_layout == "T":
        rhs_contract = _transpose_contract_dims(rhs.data.ndim, rhs_contract)

    dim_nums = (lhs_contract, rhs_contract), (lhs_batch, rhs_batch)

    out_fp8 = jax.lax.dot_general(
        lhs.data, rhs.data, dim_nums, precision=precision, preferred_element_type=jnp.float32
    )
    scale_inv = (lhs.scale_inv * rhs.scale_inv).astype(jnp.float32)

    return (out_fp8 * scale_inv).astype(lhs.dq_dtype)


@partial(jax.jit, static_argnums=(2,))
def _jax_gemm_mxfp8_1d(
    lhs: ScaledTensor, rhs: ScaledTensor, dim_nums: Tuple[Tuple[Sequence[int], Sequence[int]]]
):
    """
    JAX GEMM for MXFP8 via scaled_matmul
    """
    assert (
        rhs.scaling_mode == ScalingMode.MXFP8_1D_SCALING
    ), "rhs does not have MXFP8 1D scaling mode"

    (lhs_contract, rhs_contract), (lhs_batch, rhs_batch) = dim_nums

    expected_lhs_is_colwise = lhs_contract[-1] != lhs.data.ndim - 1
    expected_rhs_is_colwise = rhs_contract[-1] != rhs.data.ndim - 1
    assert lhs.is_colwise is expected_lhs_is_colwise, (
        f"LHS with unexpected quantize dimension.\nExpect is_colwise={expected_lhs_is_colwise}, got"
        f" {lhs.is_colwise}"
    )
    assert rhs.is_colwise is expected_rhs_is_colwise, (
        f"RHS with unexpected quantize dimension.\nExpect is_colwise={expected_rhs_is_colwise}, got"
        f" {rhs.is_colwise}"
    )

    # Reshape + Transpose (if needed)
    # [..., M, K] -> [1, reduce(..., M), K]
    # [..., K, M] -> [1, reduce(..., M), K]
    lhs_3d = _shape_normalization(lhs.data, (lhs_contract, lhs_batch))
    rhs_3d = _shape_normalization(rhs.data, (rhs_contract, rhs_batch))
    lhs_scale_3d = _shape_normalization(lhs.scale_inv, (lhs_contract, lhs_batch))
    rhs_scale_3d = _shape_normalization(rhs.scale_inv, (rhs_contract, rhs_batch))

    # Slice out the padding as scaled_matmul does not support padded scales yet
    lhs_scale_3d = jnp.asarray(lhs_scale_3d[:, : lhs_3d.shape[1], : int(lhs_3d.shape[2] / 32)])
    rhs_scale_3d = jnp.asarray(rhs_scale_3d[:, : rhs_3d.shape[1], : int(rhs_3d.shape[2] / 32)])

    # JAX scaled_matmul only supports NT now (TN-gemm)
    # * Expected shape:
    # * lhs_data  (B, M, K)           * rhs_data  (B, N, K)
    # * lhs_scale (B, M, K_block)     * rhs_scale (B, N, K_block)
    out_3d = jax.nn.scaled_matmul(
        lhs_3d, rhs_3d, lhs_scale_3d, rhs_scale_3d, preferred_element_type=lhs.dq_dtype
    )
    # Reshape [1, reduce(..., M), N] -> [..., M, N]
    lhs_remain_shape = tuple(
        lhs.data.shape[dim] for dim in range(len(lhs.data.shape)) if dim not in lhs_contract
    )
    rhs_remain_shape = tuple(
        rhs.data.shape[dim] for dim in range(len(rhs.data.shape)) if dim not in rhs_contract
    )
    out = out_3d.reshape(*lhs_remain_shape, *rhs_remain_shape)
    return out


def _jax_gemm(
    lhs: Union[jnp.ndarray, ScaledTensor],
    rhs: Union[jnp.ndarray, ScaledTensor],
    contracting_dims: Tuple[Sequence[int], Sequence[int]] = ((1,), (0,)),
    quantizer_set: Dict["str", Quantizer] = noop_quantizer_set,
) -> jnp.ndarray:
    """
    FP8 GEMM via JAX
    """

    dim_nums = (contracting_dims, ((), ()))

    def _jax_gemm_fp8_impl(lhs, rhs):
        if lhs.scaling_mode.is_tensor_scaling():
            assert (
                rhs.scaling_mode == lhs.scaling_mode
            ), f"rhs.scaling_mode={rhs.scaling_mode} != lhs.scaling_mode={lhs.scaling_mode}"
            precision = (
                jax.lax.Precision.HIGHEST
                if QuantizeConfig.FP8_2X_ACC_FPROP
                else jax.lax.Precision.DEFAULT
            )
            return _jax_gemm_tensor_scaling_fp8(lhs, rhs, dim_nums, precision)

        if lhs.scaling_mode == ScalingMode.MXFP8_1D_SCALING:
            return _jax_gemm_mxfp8_1d(lhs, rhs, dim_nums)

        raise NotImplementedError("Unsupported ScalingMode: {lhs.scaling_mode}")

    if isinstance(lhs, ScaledTensor) and isinstance(rhs, ScaledTensor):
        return _jax_gemm_fp8_impl(lhs, rhs)

    if not isinstance(lhs, ScaledTensor) and not isinstance(rhs, ScaledTensor):
        if quantizer_set != noop_quantizer_set:
            assert type(quantizer_set.x) is type(quantizer_set.kernel)
            (((lhs_contract_dim,), (rhs_contract_dim,)), _) = dim_nums
            lhs_is_rowwise = lhs_contract_dim == lhs.ndim - 1
            rhs_is_rowwise = rhs_contract_dim == rhs.ndim - 1
            # Call JAX quantization so that XLA can do pattern matching (QDQ --> FP8 gemm)
            lhs_q = quantizer_set.x.quantize(
                lhs,
                is_rowwise=lhs_is_rowwise,
                is_colwise=not lhs_is_rowwise,
            )
            rhs_q = quantizer_set.kernel.quantize(
                rhs,
                is_rowwise=rhs_is_rowwise,
                is_colwise=not rhs_is_rowwise,
            )
            return _jax_gemm_fp8_impl(lhs_q, rhs_q)

    if (
        isinstance(lhs, jnp.ndarray)
        and isinstance(rhs, jnp.ndarray)
        and quantizer_set == noop_quantizer_set
    ):
        return jax.lax.dot_general(lhs, rhs, dim_nums, preferred_element_type=lhs.dtype)

    raise NotImplementedError("Not supporting multiplication of ScaledTensor and jnp.array")


def gemm(
    lhs: Union[jnp.ndarray, ScaledTensor],
    rhs: Union[jnp.ndarray, ScaledTensor],
    contracting_dims: Tuple[Sequence[int], Sequence[int]] = ((1,), (0,)),
    quantizer_set: Dict["str", Quantizer] = noop_quantizer_set,
) -> jnp.ndarray:
    """General matrix multiplication with optional quantization.

    Args:
        lhs: First input matrix.
        rhs: Second input matrix.
        contracting_dims: Tuple of two sequences representing the contracting dimensions.
            The first sequence represents the contracting dimensions of the first matrix,
            and the second sequence represents the contracting dimensions of the second matrix.
        quantizer_set: Set of quantizers for FP8 quantization of the output.
            If None, no quantization is applied and the output has the same dtype as the inputs.

    Returns:
        If quantizer_set is None:
            The matrix multiplication result.
            Shape: (M, N)
            Dtype: Same as input dtype
          If quantizer_set is provided:
            A ScaledTensor containing the quantized matrix multiplication result.
    """

    return _jax_gemm(lhs, rhs, contracting_dims, quantizer_set)


"""
def swizzled_scale(scales):
    # Swizzle the scale tensor for FP8 GEMM
    assert scales.ndim == 2
    rows, cols = scales.shape
    scales = scales.reshape(rows // 128, 4, 32, cols // 4, 4)
    scales = jnp.transpose(scales, (0, 3, 2, 1, 4))
    scales = scales.reshape(rows, cols)
    return scales


def grouped_gemm(
    lhs_list: List[Union[jnp.ndarray, ScaledTensor]],
    rhs_list: List[Union[jnp.ndarray, ScaledTensor]],
    contracting_dims_list: List[Tuple[Sequence[int], Sequence[int]]],
    bias_list: List[jnp.ndarray] = None,
) -> List[jnp.ndarray]:
    # Grouped GEMM for multiple pairs of tensors.
    assert (
        len(lhs_list) == len(rhs_list) == len(contracting_dims_list)
    ), "lhs_list, rhs_list, contracting_dims_list must have the same length"

    num_gemms = len(lhs_list)
    lhs_list_ = []
    rhs_list_ = []
    lhs_sinv_list_ = []
    rhs_sinv_list_ = []
    bias_list_ = []
    for i in range(num_gemms):
        lhs = lhs_list[i]
        rhs = rhs_list[i]
        contracting_dims = contracting_dims_list[i]
        dim_nums = (contracting_dims, ((), ()))
        if isinstance(lhs, ScaledTensor) and isinstance(rhs, ScaledTensor):
            scaling_mode = lhs.scaling_mode
            lhs_shape = lhs.data.shape
            rhs_shape = rhs.data.shape
            out_dtype = lhs.dq_dtype
            # For ScaledTensors and DELAYED_TENSOR_SCALING, need to handle internal data_layout
            if lhs.scaling_mode.is_tensor_scaling():
                assert not (
                    lhs.data.dtype == jnp.float8_e5m2 and rhs.data.dtype == jnp.float8_e5m2
                ), "FP8 GEMM does not support E5M2 * E5M2"
                ((lhs_contract_dim,), (rhs_contract_dim,)) = contracting_dims
                if lhs.data_layout == "T":
                    lhs_contract_dim = (lhs_contract_dim - 1) % lhs.data.ndim
                if rhs.data_layout == "T":
                    rhs_contract_dim = (rhs_contract_dim - 1) % rhs.data.ndim
                dim_nums = ((lhs_contract_dim,), (rhs_contract_dim,)), ((), ())
        else:
            # For jnp.ndarray, only consider contracting_dims, data_layout is always NN
            scaling_mode = ScalingMode.NO_SCALING
            lhs_shape = lhs.shape
            rhs_shape = rhs.shape
            out_dtype = lhs.dtype

        (lhs_contract, rhs_contract), (lhs_batch, rhs_batch) = dim_nums
        lhs_dn = (lhs_contract, lhs_batch)
        rhs_dn = (rhs_contract, rhs_batch)

        lhs_remain_shape = _calculate_remaining_shape(lhs_shape, lhs_contract)
        rhs_remain_shape = _calculate_remaining_shape(rhs_shape, rhs_contract)

        # Note: do not squeeze() for {lhs, rhs}_3d, it will trigger a D2D memcpy
        if scaling_mode == ScalingMode.NO_SCALING:
            lhs_3d = _shape_normalization(lhs, lhs_dn)
            rhs_3d = _shape_normalization(rhs, rhs_dn)
        elif scaling_mode.is_tensor_scaling():
            lhs_3d = _shape_normalization(lhs.data, lhs_dn, lhs.data_layout == "N")
            rhs_3d = _shape_normalization(rhs.data, rhs_dn, rhs.data_layout == "T")
        elif scaling_mode == ScalingMode.MXFP8_1D_SCALING:
            lhs_3d = _shape_normalization(lhs.data, lhs_dn)
            rhs_3d = _shape_normalization(rhs.data, rhs_dn)
            lhs_scale_inv = _shape_normalization(lhs.scale_inv, lhs_dn)
            rhs_scale_inv = _shape_normalization(rhs.scale_inv, rhs_dn)
            # swizzled_scale requires a matrix
            lhs_scale_inv = swizzled_scale(lhs_scale_inv.squeeze())
            rhs_scale_inv = swizzled_scale(rhs_scale_inv.squeeze())
        else:
            raise NotImplementedError("Unsupported ScalingMode: {scaling_mode}")

        # Note: already_transposed doesn't matter for the output shape
        # x.shape = [B, D1, D2]
        # contracting_dims = (2, )    --> output.shape = [1, B * D1, D2]
        # contracting_dims = (0, 1, ) --> output.shape = [1, D2, B * D1]
        # x.shape = [D1, D2]
        # contracting_dims = (1, )    --> output.shape = [1, D1, D2]
        # contracting_dims = (0, )    --> output.shape = [1, D2, D1]
        bm = lhs_remain_shape[0]
        bn = rhs_remain_shape[0]
        kl = lhs_3d.shape[-1]
        kr = rhs_3d.shape[-1]
        assert kl == kr, f"After shape normalization, contracting dim size mismatch: {kl} != {kr}"
        if (bm % 16 != 0) or (bn % 16 != 0) or (kl % 16 != 0):
            print("grouped_gemm input pair {i} has invalid problem shape for lowering: ")
            print(f"m = {bm}, n = {bn}, k = {kl}; ")
            print("cuBLAS requires the problem shapes being multiples of 16")
            assert (bm % 16 == 0) and (bn % 16 == 0) and (kl % 16 == 0)

        lhs_list_.append(lhs_3d)
        rhs_list_.append(rhs_3d)
        if scaling_mode == ScalingMode.NO_SCALING:
            lhs_sinv_list_.append(jnp.ones(1, dtype=jnp.float32))
            rhs_sinv_list_.append(jnp.ones(1, dtype=jnp.float32))
        if scaling_mode.is_tensor_scaling():
            lhs_sinv_list_.append(lhs.scale_inv)
            rhs_sinv_list_.append(rhs.scale_inv)
        if scaling_mode == ScalingMode.MXFP8_1D_SCALING:
            lhs_sinv_list_.append(lhs_scale_inv)
            rhs_sinv_list_.append(rhs_scale_inv)
        if bias_list is not None:
            bias_list_.append(bias_list[i])

    out_list = GroupedGemmPrimitive.outer_primitive.bind(
        *lhs_list_,
        *rhs_list_,
        *lhs_sinv_list_,
        *rhs_sinv_list_,
        *bias_list_,
        num_gemms=num_gemms,
        scaling_mode=scaling_mode,
        out_dtype=out_dtype,
        has_bias=1 if bias_list is not None else 0,
    )

    return out_list
"""
