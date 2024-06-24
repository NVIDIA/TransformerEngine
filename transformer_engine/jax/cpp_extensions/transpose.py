# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""JAX/TE custom ops for transpose"""
from functools import partial, reduce
from typing import Tuple, Sequence, Union, Callable
import operator

import jax.numpy as jnp
from jax import dtypes
from jax.interpreters.mlir import ir
from jax.sharding import PartitionSpec, NamedSharding

from transformer_engine import transformer_engine_jax
from transformer_engine.transformer_engine_jax import DType as TEDType

from .base import BasePrimitive, register_primitive
from .custom_call import custom_caller, CustomCallArgsWrapper
from .misc import (
    check_valid_batch_dims,
    jax_dtype_to_te_dtype,
    jax_dtype_to_ir_dtype,
    te_dtype_to_jax_dtype,
    get_padded_spec,
    multidim_transpose,
    normalize_axis_boundary,
)
from .activation import ActivationEnum
from ..sharding import all_reduce_max_along_all_axes_except_PP, all_reduce_sum_along_dp_fsdp


__all__ = [
    "transpose",
    "cast_transpose",
    "dbias_cast_transpose",
    "dact_lu_dbias_cast_transpose",
    "dgated_act_lu_cast_transpose",
]


class TransposePrimitive(BasePrimitive):
    """
    Transpose Primitive
    """

    name = "te_transpose"
    multiple_results = False
    impl_static_args = (1, 2)
    inner_primitive = None
    outer_primitive = None

    @staticmethod
    def abstract(x_aval, *, static_axis_boundary, transpose_axis_boundary):
        """
        _transpose abstract
        """
        transposed_x_shape = multidim_transpose(
            x_aval.shape, static_axis_boundary, transpose_axis_boundary
        )
        xt_aval = x_aval.update(shape=transposed_x_shape, dtype=x_aval.dtype)

        return xt_aval

    @staticmethod
    def lowering(ctx, x, *, static_axis_boundary, transpose_axis_boundary):
        """
        _transpose cuda lowering
        """

        x_aval = ctx.avals_in[0]
        assert x_aval.dtype in [
            jnp.float32,
            jnp.float16,
            jnp.bfloat16,
            jnp.float8_e4m3fn,
            jnp.float8_e5m2,
        ]

        ir_x_type = ir.RankedTensorType(x.type)
        ir_x_shape = ir_x_type.shape
        ir_out_dtype = jax_dtype_to_ir_dtype(x_aval.dtype)
        if static_axis_boundary >= 0:
            for i in range(static_axis_boundary + 1):
                assert ir_x_shape[i] == 1

        transposed_x_shape = multidim_transpose(
            ir_x_shape, static_axis_boundary, transpose_axis_boundary
        )

        out_types = [ir.RankedTensorType.get(transposed_x_shape, ir_out_dtype)]
        operands = [x]
        operand_shapes = [ir_x_shape]
        args = CustomCallArgsWrapper(out_types, operands, operand_shapes)

        te_dtype = jax_dtype_to_te_dtype(x_aval.dtype)
        contracted_x_shape = (
            reduce(operator.mul, ir_x_shape[:transpose_axis_boundary]),
            reduce(operator.mul, ir_x_shape[transpose_axis_boundary:]),
        )
        opaque = transformer_engine_jax.pack_common_descriptor(
            contracted_x_shape, te_dtype, te_dtype
        )

        out = custom_caller(TransposePrimitive.name, args, opaque, False)

        return [out]

    @staticmethod
    def impl(x, static_axis_boundary, transpose_axis_boundary):
        """
        tcast_transpose implementation
        """
        assert TransposePrimitive.inner_primitive is not None
        transposed_x = TransposePrimitive.inner_primitive.bind(
            x,
            static_axis_boundary=static_axis_boundary,
            transpose_axis_boundary=transpose_axis_boundary,
        )
        return transposed_x

    @staticmethod
    def batcher(batched_args, batch_dims, *, static_axis_boundary, transpose_axis_boundary):
        check_valid_batch_dims(batch_dims)
        assert TransposePrimitive.outer_primitive is not None
        assert static_axis_boundary < 0

        (x,) = batched_args
        (x_bdim,) = batch_dims

        # Minus batch dim.
        transpose_axis_boundary = normalize_axis_boundary(transpose_axis_boundary, x.ndim - 1)
        transpose_axis_boundary += 1  # Plus batch dim

        out_bdims = x_bdim
        return (
            TransposePrimitive.outer_primitive.bind(
                x, static_axis_boundary=x_bdim, transpose_axis_boundary=transpose_axis_boundary
            ),
            out_bdims,
        )

    @staticmethod
    def infer_sharding_from_operands(
        static_axis_boundary, transpose_axis_boundary, mesh, arg_infos, result_infos
    ):
        del result_infos
        x_spec = get_padded_spec(arg_infos[0])
        xt_spec = multidim_transpose(x_spec, static_axis_boundary, transpose_axis_boundary)
        transposed_x_sharding = NamedSharding(mesh, PartitionSpec(*xt_spec))
        return transposed_x_sharding

    @staticmethod
    def partition(static_axis_boundary, transpose_axis_boundary, mesh, arg_infos, result_infos):
        del result_infos
        x_spec = get_padded_spec(arg_infos[0])
        xt_spec = multidim_transpose(x_spec, static_axis_boundary, transpose_axis_boundary)
        transposed_x_sharding = NamedSharding(mesh, PartitionSpec(*xt_spec))
        arg_shardings = tuple(arg_i.sharding for arg_i in arg_infos)
        out_shardings = transposed_x_sharding

        impl = partial(
            TransposePrimitive.impl,
            static_axis_boundary=static_axis_boundary,
            transpose_axis_boundary=transpose_axis_boundary,
        )

        return mesh, impl, out_shardings, arg_shardings


register_primitive(TransposePrimitive)


def transpose(
    x: jnp.ndarray, static_axis_boundary: int, transpose_axis_boundary: int
) -> jnp.ndarray:
    """
    transpose wrapper
    """
    return TransposePrimitive.outer_primitive.bind(
        x,
        static_axis_boundary=static_axis_boundary,
        transpose_axis_boundary=transpose_axis_boundary,
    )


class CastTransposePrimitive(BasePrimitive):
    """
    Cast Transpose Primitive
    """

    name = "te_cast_transpose"
    multiple_results = True
    impl_static_args = (4, 5, 6)
    inner_primitive = None
    outer_primitive = None

    @staticmethod
    def abstract(
        x_aval,
        amax_aval,
        scale_aval,
        scale_inv_aval,
        *,
        out_dtype,
        static_axis_boundary,
        transpose_axis_boundary
    ):
        """
        te_cast_transpose_p abstract
        """
        dtype = dtypes.canonicalize_dtype(x_aval.dtype)
        assert dtype in [jnp.float32, jnp.float16, jnp.bfloat16]
        assert amax_aval.dtype == jnp.float32
        assert scale_aval.dtype == jnp.float32
        assert scale_inv_aval.dtype == jnp.float32

        transposed_x_shape = multidim_transpose(
            x_aval.shape, static_axis_boundary, transpose_axis_boundary
        )

        casted_x_aval = x_aval.update(shape=x_aval.shape, dtype=out_dtype)
        casted_xt_aval = x_aval.update(shape=transposed_x_shape, dtype=out_dtype)
        updated_amax_aval = amax_aval.update(shape=amax_aval.shape, dtype=amax_aval.dtype)

        return casted_x_aval, casted_xt_aval, updated_amax_aval

    @staticmethod
    def lowering(
        ctx, x, amax, scale, scale_inv, *, out_dtype, static_axis_boundary, transpose_axis_boundary
    ):
        """
        te_cast_transpose_p lowering rules
        """
        x_aval, amax_aval, scale_aval, scale_inv_aval = ctx.avals_in
        assert x_aval.dtype in [jnp.float32, jnp.float16, jnp.bfloat16]
        assert amax_aval.dtype == jnp.float32
        assert scale_aval.dtype == jnp.float32
        assert scale_inv_aval.dtype == jnp.float32
        ir_x_type = ir.RankedTensorType(x.type)
        ir_x_shape = ir_x_type.shape
        if static_axis_boundary >= 0:
            for i in range(static_axis_boundary + 1):
                assert ir_x_shape[i] == 1
        ir_out_dtype = jax_dtype_to_ir_dtype(out_dtype)
        ir_amax_type = ir.RankedTensorType(amax.type)
        ir_amax_dtype = ir_amax_type.element_type
        ir_amax_shape = ir_amax_type.shape
        ir_scale_shape = ir_amax_shape
        ir_scale_inv_shape = ir_amax_shape

        transposed_x_shape = multidim_transpose(
            ir_x_shape, static_axis_boundary, transpose_axis_boundary
        )

        out_types = [
            ir.RankedTensorType.get(ir_x_shape, ir_out_dtype),
            ir.RankedTensorType.get(transposed_x_shape, ir_out_dtype),
            ir.RankedTensorType.get(ir_amax_shape, ir_amax_dtype),
        ]
        operands = [x, amax, scale, scale_inv]
        operand_shapes = [ir_x_shape, ir_amax_shape, ir_scale_shape, ir_scale_inv_shape]
        args = CustomCallArgsWrapper(out_types, operands, operand_shapes)

        contracted_x_shape = (
            reduce(operator.mul, ir_x_shape[:transpose_axis_boundary]),
            reduce(operator.mul, ir_x_shape[transpose_axis_boundary:]),
        )
        opaque = transformer_engine_jax.pack_common_descriptor(
            contracted_x_shape,
            jax_dtype_to_te_dtype(x_aval.dtype),
            jax_dtype_to_te_dtype(out_dtype),
        )

        out = custom_caller(
            CastTransposePrimitive.name, args, opaque, False, operand_output_aliases={1: 2}
        )

        return out

    @staticmethod
    def impl(x, amax, scale, scale_inv, out_dtype, static_axis_boundary, transpose_axis_boundary):
        """
        te_cast_transpose implementation
        """
        assert CastTransposePrimitive.inner_primitive is not None
        casted_x, casted_transposed_x, updated_amax = CastTransposePrimitive.inner_primitive.bind(
            x,
            amax,
            scale,
            scale_inv,
            out_dtype=out_dtype,
            static_axis_boundary=static_axis_boundary,
            transpose_axis_boundary=transpose_axis_boundary,
        )
        return casted_x, casted_transposed_x, updated_amax

    @staticmethod
    def batcher(
        batched_args, batch_dims, *, out_dtype, static_axis_boundary, transpose_axis_boundary
    ):
        check_valid_batch_dims(batch_dims)
        assert CastTransposePrimitive.outer_primitive is not None
        assert static_axis_boundary < 0

        x, amax, scale, scale_inv = batched_args
        x_bdim, amax_bdim, *_ = batch_dims

        # Minus batch dim.
        transpose_axis_boundary = normalize_axis_boundary(transpose_axis_boundary, x.ndim - 1)
        transpose_axis_boundary += 1  # Plus batch dim

        out_bdims = x_bdim, x_bdim, amax_bdim
        return (
            CastTransposePrimitive.outer_primitive.bind(
                x,
                amax,
                scale,
                scale_inv,
                out_dtype=out_dtype,
                static_axis_boundary=x_bdim,
                transpose_axis_boundary=transpose_axis_boundary,
            ),
            out_bdims,
        )

    @staticmethod
    def infer_sharding_from_operands(
        out_dtype, static_axis_boundary, transpose_axis_boundary, mesh, arg_infos, result_infos
    ):
        del out_dtype, result_infos
        x_spec = get_padded_spec(arg_infos[0])
        casted_x_sharding = NamedSharding(mesh, PartitionSpec(*x_spec))
        xt_spec = multidim_transpose(x_spec, static_axis_boundary, transpose_axis_boundary)
        casted_transposed_x_sharding = NamedSharding(mesh, PartitionSpec(*xt_spec))
        amax_sharding = NamedSharding(mesh, PartitionSpec(*get_padded_spec(arg_infos[1])))
        return (casted_x_sharding, casted_transposed_x_sharding, amax_sharding)

    @staticmethod
    def partition(
        out_dtype, static_axis_boundary, transpose_axis_boundary, mesh, arg_infos, result_infos
    ):
        del result_infos
        x_spec = get_padded_spec(arg_infos[0])
        casted_x_sharding = NamedSharding(mesh, PartitionSpec(*x_spec))
        xt_spec = multidim_transpose(x_spec, static_axis_boundary, transpose_axis_boundary)
        casted_transposed_x_sharding = NamedSharding(mesh, PartitionSpec(*xt_spec))
        amax_sharding = NamedSharding(mesh, PartitionSpec(*get_padded_spec(arg_infos[1])))
        arg_shardings = tuple(arg_i.sharding for arg_i in arg_infos)
        out_shardings = (casted_x_sharding, casted_transposed_x_sharding, amax_sharding)

        def sharded_impl(x, amax, scale, scale_inv):
            local_cx, local_cxt, local_updated_amax = CastTransposePrimitive.impl(
                x,
                amax,
                scale,
                scale_inv,
                out_dtype=out_dtype,
                static_axis_boundary=static_axis_boundary,
                transpose_axis_boundary=transpose_axis_boundary,
            )
            global_updated_amax = all_reduce_max_along_all_axes_except_PP(local_updated_amax)

            return local_cx, local_cxt, global_updated_amax

        return mesh, sharded_impl, out_shardings, arg_shardings


register_primitive(CastTransposePrimitive)


def cast_transpose(
    x: jnp.ndarray,
    amax: jnp.ndarray,
    scale: jnp.ndarray,
    scale_inv: jnp.ndarray,
    out_dtype: jnp.dtype,
    static_axis_boundary: int,
    transpose_axis_boundary: int,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    cast transpose wrapper
    Return two tensors, FP8(inputs) and FP8(inputs.T), which are scaled by `scale`
    """
    return CastTransposePrimitive.outer_primitive.bind(
        x,
        amax,
        scale,
        scale_inv,
        out_dtype=out_dtype,
        static_axis_boundary=static_axis_boundary,
        transpose_axis_boundary=transpose_axis_boundary,
    )


class DBiasCastTransposePrimitive(BasePrimitive):
    """
    DBias Cast Transpose Primitive
    """

    name = "te_dbias_cast_transpose"
    multiple_results = True
    # out_dtype, static_axis_boundary, transpose_axis_boundary
    impl_static_args = (4, 5, 6)
    inner_primitive = None
    outer_primitive = None

    @staticmethod
    def abstract(
        dz_aval,
        amax_aval,
        scale_aval,
        scale_inv_aval,
        *,
        out_dtype,
        static_axis_boundary,
        transpose_axis_boundary
    ):
        """
        te_dbias_cast_transpose_p abstract
        """
        dtype = dtypes.canonicalize_dtype(dz_aval.dtype)
        assert dtype in [jnp.float32, jnp.float16, jnp.bfloat16]
        assert amax_aval.dtype == jnp.float32
        assert scale_aval.dtype == jnp.float32
        assert scale_inv_aval.dtype == jnp.float32
        gi_hidden_size = reduce(operator.mul, dz_aval.shape[transpose_axis_boundary:])
        t_shape = multidim_transpose(dz_aval.shape, static_axis_boundary, transpose_axis_boundary)
        out = dz_aval.update(shape=dz_aval.shape, dtype=out_dtype)
        t_out = dz_aval.update(shape=t_shape, dtype=out_dtype)

        dbias_shape = (*dz_aval.shape[: static_axis_boundary + 1], gi_hidden_size)
        dbias = dz_aval.update(shape=dbias_shape, dtype=dtype)

        updated_amax_aval = amax_aval.update(shape=amax_aval.shape, dtype=amax_aval.dtype)
        (wkspace_info,) = transformer_engine_jax.get_dbias_ct_workspace_sizes(
            dz_aval.size // gi_hidden_size,
            gi_hidden_size,
            jax_dtype_to_te_dtype(dz_aval.dtype),
            jax_dtype_to_te_dtype(out_dtype),
        )
        wkspace_aval = dz_aval.update(
            shape=wkspace_info[0], dtype=te_dtype_to_jax_dtype(wkspace_info[1])
        )

        return out, t_out, dbias, updated_amax_aval, wkspace_aval

    @staticmethod
    def outer_abstract(*args, **kwargs):
        """
        te_dbias_cast_transpose_p outer abstract
        """

        out, t_out, dbias, updated_amax_aval, _ = DBiasCastTransposePrimitive.abstract(
            *args, **kwargs
        )
        return out, t_out, dbias, updated_amax_aval

    @staticmethod
    def lowering(
        ctx, dz, amax, scale, scale_inv, *, out_dtype, static_axis_boundary, transpose_axis_boundary
    ):
        """
        te_dbias_cast_transpose_p lowering rules
        """
        dz_aval, amax_aval, scale_aval, scale_inv_aval = ctx.avals_in
        assert dz_aval.dtype in [jnp.float32, jnp.float16, jnp.bfloat16]
        assert amax_aval.dtype == jnp.float32
        assert scale_aval.dtype == jnp.float32
        assert scale_inv_aval.dtype == jnp.float32
        ir_dz_type = ir.RankedTensorType(dz.type)
        ir_dz_shape = ir_dz_type.shape
        batch_size = reduce(operator.mul, ir_dz_shape[:transpose_axis_boundary])
        ir_hidden_size = reduce(operator.mul, ir_dz_shape[transpose_axis_boundary:])
        contracted_dz_shape = (batch_size, ir_hidden_size)
        ir_out_dtype = jax_dtype_to_ir_dtype(out_dtype)
        ir_amax_type = ir.RankedTensorType(amax.type)
        ir_amax_dtype = ir_amax_type.element_type
        ir_amax_shape = ir_amax_type.shape
        ir_scale_shape = ir_amax_shape
        ir_scale_inv_shape = ir_amax_shape
        transposed_dz_shape = multidim_transpose(
            ir_dz_shape, static_axis_boundary, transpose_axis_boundary
        )
        dbias_shape = (*ir_dz_shape[: static_axis_boundary + 1], ir_hidden_size)

        wkspace_aval = ctx.avals_out[-1]

        out_types = [
            ir.RankedTensorType.get(ir_dz_shape, ir_out_dtype),
            ir.RankedTensorType.get(transposed_dz_shape, ir_out_dtype),
            ir.RankedTensorType.get(dbias_shape, ir_dz_type.element_type),
            ir.RankedTensorType.get(ir_amax_shape, ir_amax_dtype),
            ir.RankedTensorType.get(wkspace_aval.shape, jax_dtype_to_ir_dtype(wkspace_aval.dtype)),
        ]
        operands = [dz, amax, scale, scale_inv]
        operand_shapes = [ir_dz_shape, ir_amax_shape, ir_scale_shape, ir_scale_inv_shape]
        args = CustomCallArgsWrapper(out_types, operands, operand_shapes)
        opaque = transformer_engine_jax.pack_common_wk_descriptor(
            contracted_dz_shape,
            wkspace_aval.shape,
            jax_dtype_to_te_dtype(dz_aval.dtype),
            jax_dtype_to_te_dtype(out_dtype),
            jax_dtype_to_te_dtype(wkspace_aval.dtype),
        )

        out = custom_caller(
            DBiasCastTransposePrimitive.name, args, opaque, False, operand_output_aliases={1: 3}
        )

        return out

    @staticmethod
    def impl(dz, amax, scale, scale_inv, out_dtype, static_axis_boundary, transpose_axis_boundary):
        """
        to describe implementation
        """
        assert DBiasCastTransposePrimitive.inner_primitive is not None
        out, t_out, dbias, updated_amax, _ = DBiasCastTransposePrimitive.inner_primitive.bind(
            dz,
            amax,
            scale,
            scale_inv,
            out_dtype=out_dtype,
            static_axis_boundary=static_axis_boundary,
            transpose_axis_boundary=transpose_axis_boundary,
        )
        return out, t_out, dbias, updated_amax

    @staticmethod
    def batcher(
        batched_args, batch_dims, *, out_dtype, static_axis_boundary, transpose_axis_boundary
    ):
        """
        to describe batch rules for vmap
        """
        del static_axis_boundary
        check_valid_batch_dims(batch_dims)
        assert DBiasCastTransposePrimitive.outer_primitive is not None
        dz, amax, scale, scale_inv = batched_args
        dz_bdim, amax_bdim, _, _ = batch_dims

        # Minus batch dim.
        transpose_axis_boundary = normalize_axis_boundary(transpose_axis_boundary, dz.ndim - 1)
        transpose_axis_boundary += 1  # Plus batch dim

        out_bdims = dz_bdim, dz_bdim, dz_bdim, amax_bdim
        return (
            DBiasCastTransposePrimitive.outer_primitive.bind(
                dz,
                amax,
                scale,
                scale_inv,
                out_dtype=out_dtype,
                static_axis_boundary=dz_bdim,
                transpose_axis_boundary=transpose_axis_boundary,
            ),
            out_bdims,
        )

    @staticmethod
    def infer_sharding_from_operands(
        out_dtype, static_axis_boundary, transpose_axis_boundary, mesh, arg_infos, result_infos
    ):
        del out_dtype, result_infos
        x_spec = get_padded_spec(arg_infos[0])
        out_sharding = NamedSharding(mesh, PartitionSpec(*x_spec))
        xt_spec = multidim_transpose(x_spec, static_axis_boundary, transpose_axis_boundary)
        tranposed_out_sharding = NamedSharding(mesh, PartitionSpec(*xt_spec))
        dbias_shaprding = NamedSharding(
            mesh, PartitionSpec(*x_spec[: static_axis_boundary + 1], x_spec[-1])
        )
        amax_sharding = NamedSharding(mesh, PartitionSpec(*get_padded_spec(arg_infos[1])))
        return (out_sharding, tranposed_out_sharding, dbias_shaprding, amax_sharding)

    @staticmethod
    def partition(
        out_dtype, static_axis_boundary, transpose_axis_boundary, mesh, arg_infos, result_infos
    ):
        del result_infos
        x_spec = get_padded_spec(arg_infos[0])
        casted_x_sharding = NamedSharding(mesh, PartitionSpec(*x_spec))
        xt_spec = multidim_transpose(x_spec, static_axis_boundary, transpose_axis_boundary)
        casted_transposed_x_sharding = NamedSharding(mesh, PartitionSpec(*xt_spec))

        dbias_shaprding = NamedSharding(
            mesh, PartitionSpec(*x_spec[: static_axis_boundary + 1], x_spec[-1])
        )

        amax_sharding = NamedSharding(mesh, PartitionSpec(*get_padded_spec(arg_infos[1])))
        arg_shardings = tuple(arg_i.sharding for arg_i in arg_infos)
        out_shardings = (
            casted_x_sharding,
            casted_transposed_x_sharding,
            dbias_shaprding,
            amax_sharding,
        )

        def sharded_impl(dz, amax, scale, scale_inv):
            local_out, local_t_out, local_dbias, local_amax = DBiasCastTransposePrimitive.impl(
                dz,
                amax,
                scale,
                scale_inv,
                out_dtype=out_dtype,
                static_axis_boundary=static_axis_boundary,
                transpose_axis_boundary=transpose_axis_boundary,
            )
            global_dbias = all_reduce_sum_along_dp_fsdp(local_dbias)
            global_updated_amax = all_reduce_max_along_all_axes_except_PP(local_amax)
            return local_out, local_t_out, global_dbias, global_updated_amax

        return mesh, sharded_impl, out_shardings, arg_shardings


register_primitive(DBiasCastTransposePrimitive)


def dbias_cast_transpose(
    dz: jnp.ndarray,
    amax: jnp.ndarray,
    scale: jnp.ndarray,
    scale_inv: jnp.ndarray,
    out_dtype: TEDType,
    static_axis_boundary: int,
    transpose_axis_boundary: int = -1,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    cast transpose dbias partial fusion wrapper
    Return FP8(inputs), dbias
    """
    if static_axis_boundary < 0:
        static_axis_boundary = -1  # means no static axes

    return DBiasCastTransposePrimitive.outer_primitive.bind(
        dz,
        amax,
        scale,
        scale_inv,
        out_dtype=out_dtype,
        static_axis_boundary=static_axis_boundary,
        transpose_axis_boundary=transpose_axis_boundary,
    )


class DActLuDBiasCastTransposePrimitive(BasePrimitive):
    """
    DActLu DBias Cast Transpose Primitive
    """

    name = "te_dact_lu_dbias_cast_transpose"
    multiple_results = True
    # out_dtype, static_axis_boundary, transpose_axis_boundary, act_enum
    impl_static_args = (5, 6, 7, 8)
    inner_primitive = None
    outer_primitive = None

    @staticmethod
    def abstract(
        dz_aval,
        x_aval,
        amax_aval,
        scale_aval,
        scale_inv_aval,
        *,
        out_dtype,
        static_axis_boundary,
        transpose_axis_boundary,
        act_enum
    ):  # pylint: disable=unused-argument
        """
        te_dact_lu_dbais_cast_transpose_p abstract
        """
        dtype = dtypes.canonicalize_dtype(dz_aval.dtype)
        assert dtype in [jnp.float32, jnp.float16, jnp.bfloat16]
        assert x_aval.dtype == dtype
        assert amax_aval.dtype == jnp.float32
        assert scale_aval.dtype == jnp.float32
        assert scale_inv_aval.dtype == jnp.float32
        ir_hidden_szie = dz_aval.shape[-1]
        gi_hidden_size = x_aval.shape[-1]
        assert ir_hidden_szie == gi_hidden_size
        t_shape = multidim_transpose(x_aval.shape, static_axis_boundary, transpose_axis_boundary)
        out = dz_aval.update(shape=x_aval.shape, dtype=out_dtype)
        t_out = dz_aval.update(shape=t_shape, dtype=out_dtype)

        dbias_shape = (*x_aval.shape[: static_axis_boundary + 1], gi_hidden_size)
        dbias = dz_aval.update(shape=dbias_shape, dtype=dtype)

        updated_amax_aval = amax_aval.update(shape=amax_aval.shape, dtype=amax_aval.dtype)

        (wkspace_info,) = transformer_engine_jax.get_dact_dbias_ct_workspace_sizes(
            x_aval.size // gi_hidden_size,
            gi_hidden_size,
            jax_dtype_to_te_dtype(x_aval.dtype),
            jax_dtype_to_te_dtype(out_dtype),
        )
        wkspace_aval = x_aval.update(
            shape=wkspace_info[0], dtype=te_dtype_to_jax_dtype(wkspace_info[1])
        )

        return out, t_out, dbias, updated_amax_aval, wkspace_aval

    @staticmethod
    def outer_abstract(*args, **kwargs):
        """
        te_dact_lu_dbais_cast_transpose_p outer abstract
        """

        out, t_out, dbias, updated_amax_aval, _ = DActLuDBiasCastTransposePrimitive.abstract(
            *args, **kwargs
        )
        return out, t_out, dbias, updated_amax_aval

    @staticmethod
    def lowering(
        ctx,
        dz,
        x,
        amax,
        scale,
        scale_inv,
        *,
        out_dtype,
        static_axis_boundary,
        transpose_axis_boundary,
        act_enum
    ):
        """
        te_dgated_act_lu_cast_transpose_p lowering rules
        """
        dz_aval, x_aval, amax_aval, scale_aval, scale_inv_aval = ctx.avals_in
        assert dz_aval.dtype in [jnp.float32, jnp.float16, jnp.bfloat16]
        assert x_aval.dtype == dz_aval.dtype
        assert amax_aval.dtype == jnp.float32
        assert scale_aval.dtype == jnp.float32
        assert scale_inv_aval.dtype == jnp.float32
        ir_dz_type = ir.RankedTensorType(dz.type)
        ir_dz_shape = ir_dz_type.shape
        x_type = ir.RankedTensorType(x.type)
        x_shape = x_type.shape
        dz_batch_szie = reduce(operator.mul, ir_dz_shape[:-1])
        x_batch_size = reduce(operator.mul, x_shape[:-2])
        assert dz_batch_szie == x_batch_size
        ir_hidden_szie = ir_dz_shape[-1]
        contracted_x_shape = (x_batch_size, ir_hidden_szie)

        ir_out_dtype = jax_dtype_to_ir_dtype(out_dtype)
        ir_amax_type = ir.RankedTensorType(amax.type)
        ir_amax_dtype = ir_amax_type.element_type
        ir_amax_shape = ir_amax_type.shape
        ir_scale_shape = ir_amax_shape
        ir_scale_inv_shape = ir_amax_shape
        transposed_x_shape = multidim_transpose(
            x_shape, static_axis_boundary, transpose_axis_boundary
        )
        dbias_shape = (*x_shape[: static_axis_boundary + 1], ir_hidden_szie)

        wkspace_aval = ctx.avals_out[-1]

        out_types = [
            ir.RankedTensorType.get(x_shape, ir_out_dtype),
            ir.RankedTensorType.get(transposed_x_shape, ir_out_dtype),
            ir.RankedTensorType.get(dbias_shape, ir_dz_type.element_type),
            ir.RankedTensorType.get(ir_amax_shape, ir_amax_dtype),
            ir.RankedTensorType.get(wkspace_aval.shape, jax_dtype_to_ir_dtype(wkspace_aval.dtype)),
        ]
        operands = [dz, x, amax, scale, scale_inv]
        operand_shapes = [ir_dz_shape, x_shape, ir_amax_shape, ir_scale_shape, ir_scale_inv_shape]
        args = CustomCallArgsWrapper(out_types, operands, operand_shapes)
        opaque = transformer_engine_jax.pack_common_wk_descriptor(
            contracted_x_shape,
            wkspace_aval.shape,
            jax_dtype_to_te_dtype(dz_aval.dtype),
            jax_dtype_to_te_dtype(out_dtype),
            jax_dtype_to_te_dtype(wkspace_aval.dtype),
            act_enum,
        )

        out = custom_caller(
            DActLuDBiasCastTransposePrimitive.name,
            args,
            opaque,
            False,
            operand_output_aliases={2: 3},
        )

        return out

    @staticmethod
    def impl(
        dz,
        x,
        amax,
        scale,
        scale_inv,
        out_dtype,
        static_axis_boundary,
        transpose_axis_boundary,
        act_enum,
    ):
        """
        to describe implementation
        """
        assert DActLuDBiasCastTransposePrimitive.inner_primitive is not None
        out, t_out, dbias, updated_amax, _ = DActLuDBiasCastTransposePrimitive.inner_primitive.bind(
            dz,
            x,
            amax,
            scale,
            scale_inv,
            out_dtype=out_dtype,
            static_axis_boundary=static_axis_boundary,
            transpose_axis_boundary=transpose_axis_boundary,
            act_enum=act_enum,
        )
        return out, t_out, dbias, updated_amax

    @staticmethod
    def batcher(
        batched_args,
        batch_dims,
        *,
        out_dtype,
        static_axis_boundary,
        transpose_axis_boundary,
        act_enum
    ):
        """
        to describe batch rules for vmap
        """
        del static_axis_boundary
        check_valid_batch_dims(batch_dims)
        assert DActLuDBiasCastTransposePrimitive.outer_primitive is not None
        dz, x, amax, scale, scale_inv = batched_args
        x_bdim, _, amax_bdim, _, _ = batch_dims

        # Minus batch dim.
        transpose_axis_boundary = normalize_axis_boundary(transpose_axis_boundary, x.ndim - 1)
        transpose_axis_boundary += 1  # Plus batch dim

        out_bdims = x_bdim, x_bdim, x_bdim, amax_bdim
        return (
            DActLuDBiasCastTransposePrimitive.outer_primitive.bind(
                dz,
                x,
                amax,
                scale,
                scale_inv,
                out_dtype=out_dtype,
                static_axis_boundary=x_bdim,
                transpose_axis_boundary=transpose_axis_boundary,
                act_enum=act_enum,
            ),
            out_bdims,
        )

    @staticmethod
    def infer_sharding_from_operands(
        out_dtype,
        static_axis_boundary,
        transpose_axis_boundary,
        act_enum,
        mesh,
        arg_infos,
        result_infos,
    ):
        del out_dtype, result_infos, act_enum
        x_spec = get_padded_spec(arg_infos[1])
        out_sharding = NamedSharding(mesh, PartitionSpec(*x_spec))
        xt_spec = multidim_transpose(x_spec, static_axis_boundary, transpose_axis_boundary)
        tranposed_out_sharding = NamedSharding(mesh, PartitionSpec(*xt_spec))
        dbias_shaprding = NamedSharding(
            mesh, PartitionSpec(*x_spec[: static_axis_boundary + 1], x_spec[-1])
        )
        amax_sharding = NamedSharding(mesh, PartitionSpec(*get_padded_spec(arg_infos[2])))
        return (out_sharding, tranposed_out_sharding, dbias_shaprding, amax_sharding)

    @staticmethod
    def partition(
        out_dtype,
        static_axis_boundary,
        transpose_axis_boundary,
        act_enum,
        mesh,
        arg_infos,
        result_infos,
    ):
        del result_infos
        x_spec = get_padded_spec(arg_infos[1])
        casted_x_sharding = NamedSharding(mesh, PartitionSpec(*x_spec))
        xt_spec = multidim_transpose(x_spec, static_axis_boundary, transpose_axis_boundary)
        casted_transposed_x_sharding = NamedSharding(mesh, PartitionSpec(*xt_spec))

        dbias_shaprding = NamedSharding(
            mesh, PartitionSpec(*x_spec[: static_axis_boundary + 1], x_spec[-1])
        )

        amax_sharding = NamedSharding(mesh, PartitionSpec(*get_padded_spec(arg_infos[2])))
        arg_shardings = tuple(arg_i.sharding for arg_i in arg_infos)
        out_shardings = (
            casted_x_sharding,
            casted_transposed_x_sharding,
            dbias_shaprding,
            amax_sharding,
        )

        def sharded_impl(dz, x, amax, scale, scale_inv):
            local_out, local_t_out, local_dbias, local_amax = (
                DActLuDBiasCastTransposePrimitive.impl(
                    dz,
                    x,
                    amax,
                    scale,
                    scale_inv,
                    out_dtype=out_dtype,
                    static_axis_boundary=static_axis_boundary,
                    transpose_axis_boundary=transpose_axis_boundary,
                    act_enum=act_enum,
                )
            )
            global_dbias = all_reduce_sum_along_dp_fsdp(local_dbias)
            global_updated_amax = all_reduce_max_along_all_axes_except_PP(local_amax)
            return local_out, local_t_out, global_dbias, global_updated_amax

        return mesh, sharded_impl, out_shardings, arg_shardings


register_primitive(DActLuDBiasCastTransposePrimitive)


def dact_lu_dbias_cast_transpose(
    dz: jnp.ndarray,
    x: jnp.ndarray,
    amax: jnp.ndarray,
    scale: jnp.ndarray,
    scale_inv: jnp.ndarray,
    out_dtype: TEDType,
    static_axis_boundary: int,
    transpose_axis_boundary: int = -1,
    activation_type: Sequence[Union[str, Callable]] = ("gelu",),
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    cast transpose dact_lu and dbias fusion wrapper
    Return FP8(dact_lu(inputs)), dbias
    ONLY support non-gated activation type
    """
    if static_axis_boundary < 0:
        static_axis_boundary = -1  # means no static axes

    act_type_id = ActivationEnum[activation_type]
    return DActLuDBiasCastTransposePrimitive.outer_primitive.bind(
        dz,
        x,
        amax,
        scale,
        scale_inv,
        out_dtype=out_dtype,
        static_axis_boundary=static_axis_boundary,
        transpose_axis_boundary=transpose_axis_boundary,
        act_enum=act_type_id,
    )


class DgatedActLuCastTransposePrimitive(BasePrimitive):
    """
    Dgated ActLu Cast Transpose Primitive
    """

    name = "te_dgated_act_lu_cast_transpose"
    multiple_results = True
    impl_static_args = (5, 6, 7)  # out_dtype, static_axis_boundary, act_enum
    inner_primitive = None
    outer_primitive = None

    @staticmethod
    def abstract(
        dz_aval,
        x_aval,
        amax_aval,
        scale_aval,
        scale_inv_aval,
        *,
        out_dtype,
        static_axis_boundary,
        act_enum
    ):  # pylint: disable=unused-argument
        """
        te_dgated_act_lu_cast_transpose_p abstract
        """
        dtype = dtypes.canonicalize_dtype(dz_aval.dtype)
        assert dtype in [jnp.float32, jnp.float16, jnp.bfloat16]
        assert x_aval.dtype == dtype
        assert x_aval.shape[-2] == 2  # Linear + GeLU
        assert amax_aval.dtype == jnp.float32
        assert scale_aval.dtype == jnp.float32
        assert scale_inv_aval.dtype == jnp.float32
        ir_hidden_szie = dz_aval.shape[-1]
        gi_hidden_size = x_aval.shape[-1]
        assert ir_hidden_szie == gi_hidden_size
        t_shape = multidim_transpose(x_aval.shape, static_axis_boundary, -2)
        out = dz_aval.update(shape=x_aval.shape, dtype=out_dtype)
        t_out = dz_aval.update(shape=t_shape, dtype=out_dtype)
        updated_amax_aval = amax_aval.update(shape=amax_aval.shape, dtype=amax_aval.dtype)
        return out, t_out, updated_amax_aval

    @staticmethod
    def lowering(ctx, dz, x, amax, scale, scale_inv, *, out_dtype, static_axis_boundary, act_enum):
        """
        te_dgated_act_lu_cast_transpose_p lowering rules
        """
        dz_aval, x_aval, amax_aval, scale_aval, scale_inv_aval = ctx.avals_in
        assert dz_aval.dtype in [jnp.float32, jnp.float16, jnp.bfloat16]
        assert x_aval.dtype == dz_aval.dtype
        assert amax_aval.dtype == jnp.float32
        assert scale_aval.dtype == jnp.float32
        assert scale_inv_aval.dtype == jnp.float32
        ir_dz_type = ir.RankedTensorType(dz.type)
        ir_dz_shape = ir_dz_type.shape
        x_type = ir.RankedTensorType(x.type)
        x_shape = x_type.shape
        dz_batch_szie = reduce(operator.mul, ir_dz_shape[:-1])
        x_batch_size = reduce(operator.mul, x_shape[:-2])
        assert dz_batch_szie == x_batch_size
        assert x_shape[-2] == 2  # Linear + GeLU
        ir_hidden_szie = ir_dz_shape[-1]
        gi_hidden_size = x_shape[-1]
        assert ir_hidden_szie == gi_hidden_size
        ir_out_dtype = jax_dtype_to_ir_dtype(out_dtype)
        ir_amax_type = ir.RankedTensorType(amax.type)
        ir_amax_dtype = ir_amax_type.element_type
        ir_amax_shape = ir_amax_type.shape
        ir_scale_shape = ir_amax_shape
        ir_scale_inv_shape = ir_amax_shape
        transposed_x_shape = multidim_transpose(x_shape, static_axis_boundary, -2)
        out_types = [
            ir.RankedTensorType.get(x_shape, ir_out_dtype),
            ir.RankedTensorType.get(transposed_x_shape, ir_out_dtype),
            ir.RankedTensorType.get(ir_amax_shape, ir_amax_dtype),
        ]
        operands = [dz, x, amax, scale, scale_inv]
        operand_shapes = [ir_dz_shape, x_shape, ir_amax_shape, ir_scale_shape, ir_scale_inv_shape]
        args = CustomCallArgsWrapper(out_types, operands, operand_shapes)
        contracted_x_shape = (x_batch_size, x_shape[-1])
        opaque = transformer_engine_jax.pack_common_descriptor(
            contracted_x_shape,
            jax_dtype_to_te_dtype(dz_aval.dtype),
            jax_dtype_to_te_dtype(out_dtype),
            act_enum,
        )

        out = custom_caller(
            DgatedActLuCastTransposePrimitive.name,
            args,
            opaque,
            False,
            operand_output_aliases={2: 2},
        )

        return out

    @staticmethod
    def impl(dz, x, amax, scale, scale_inv, out_dtype, static_axis_boundary, act_enum):
        """
        to describe implementation
        """
        assert DgatedActLuCastTransposePrimitive.inner_primitive is not None
        out, t_out, updated_amax = DgatedActLuCastTransposePrimitive.inner_primitive.bind(
            dz,
            x,
            amax,
            scale,
            scale_inv,
            out_dtype=out_dtype,
            static_axis_boundary=static_axis_boundary,
            act_enum=act_enum,
        )
        return out, t_out, updated_amax

    @staticmethod
    def batcher(batched_args, batch_dims, *, out_dtype, static_axis_boundary, act_enum):
        """
        to describe batch rules for vmap
        """
        del static_axis_boundary
        check_valid_batch_dims(batch_dims)
        assert DgatedActLuCastTransposePrimitive.outer_primitive is not None
        dz, x, amax, scale, scale_inv = batched_args
        x_bdim, _, amax_bdim, _, _ = batch_dims

        out_bdims = x_bdim, x_bdim, amax_bdim
        return (
            DgatedActLuCastTransposePrimitive.outer_primitive.bind(
                dz,
                x,
                amax,
                scale,
                scale_inv,
                out_dtype=out_dtype,
                static_axis_boundary=x_bdim,
                act_enum=act_enum,
            ),
            out_bdims,
        )

    @staticmethod
    def infer_sharding_from_operands(
        out_dtype, static_axis_boundary, act_enum, mesh, arg_infos, result_infos
    ):
        del out_dtype, result_infos, act_enum
        x_spec = get_padded_spec(arg_infos[1])
        out_sharding = NamedSharding(mesh, PartitionSpec(*x_spec))
        xt_spec = multidim_transpose(x_spec, static_axis_boundary, -2)
        tranposed_out_sharding = NamedSharding(mesh, PartitionSpec(*xt_spec))
        amax_sharding = NamedSharding(mesh, PartitionSpec(*get_padded_spec(arg_infos[2])))
        return (out_sharding, tranposed_out_sharding, amax_sharding)

    @staticmethod
    def partition(out_dtype, static_axis_boundary, act_enum, mesh, arg_infos, result_infos):
        del result_infos
        x_spec = get_padded_spec(arg_infos[1])
        casted_x_sharding = NamedSharding(mesh, PartitionSpec(*x_spec))
        xt_spec = multidim_transpose(x_spec, static_axis_boundary, -2)
        casted_transposed_x_sharding = NamedSharding(mesh, PartitionSpec(*xt_spec))

        amax_sharding = NamedSharding(mesh, PartitionSpec(*get_padded_spec(arg_infos[2])))
        arg_shardings = tuple(arg_i.sharding for arg_i in arg_infos)
        out_shardings = (casted_x_sharding, casted_transposed_x_sharding, amax_sharding)

        def sharded_impl(dz, x, amax, scale, scale_inv):
            local_out, local_t_out, local_amax = DgatedActLuCastTransposePrimitive.impl(
                dz,
                x,
                amax,
                scale,
                scale_inv,
                out_dtype=out_dtype,
                static_axis_boundary=static_axis_boundary,
                act_enum=act_enum,
            )
            global_updated_amax = all_reduce_max_along_all_axes_except_PP(local_amax)
            return local_out, local_t_out, global_updated_amax

        return mesh, sharded_impl, out_shardings, arg_shardings


register_primitive(DgatedActLuCastTransposePrimitive)


def dgated_act_lu_cast_transpose(
    dz: jnp.ndarray,
    x: jnp.ndarray,
    amax: jnp.ndarray,
    scale: jnp.ndarray,
    scale_inv: jnp.ndarray,
    out_dtype: TEDType,
    static_axis_boundary: int,
    activation_type: Sequence[Union[str, Callable]],
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    cast transpose d_gated_act_lu fusion wrapper
    Return FP8(dgated_act_lu(inputs))
    """
    act_type_id = ActivationEnum[activation_type]
    return DgatedActLuCastTransposePrimitive.outer_primitive.bind(
        dz,
        x,
        amax,
        scale,
        scale_inv,
        out_dtype=out_dtype,
        static_axis_boundary=static_axis_boundary,
        act_enum=act_type_id,
    )
