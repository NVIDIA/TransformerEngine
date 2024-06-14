# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""JAX/TE custom ops for quantization"""
from typing import Tuple

import jax.numpy as jnp
from jax import dtypes
from jax.interpreters.mlir import ir
from jax.sharding import PartitionSpec, NamedSharding

from transformer_engine import transformer_engine_jax
from transformer_engine.transformer_engine_jax import DType as TEDType

from .base import BasePrimitive, register_primitive
from .custom_call import custom_caller, CustomCallArgsWrapper
from .misc import (
    get_padded_spec,
    check_valid_batch_dims,
    jax_dtype_to_te_dtype,
    jax_dtype_to_ir_dtype,
)
from ..sharding import all_reduce_max_along_all_axes_except_PP


__all__ = ["cast_fp8"]


class CastFP8Primitive(BasePrimitive):
    """
    Cast Primitive
    """

    name = "te_quantize"
    multiple_results = True
    impl_static_args = (4,)
    inner_primitive = None
    outer_primitive = None

    @staticmethod
    def abstract(x_aval, amax_aval, scale_aval, scale_inv_aval, *, out_dtype):
        """
        te_cast abstract
        """
        dtype = dtypes.canonicalize_dtype(x_aval.dtype)
        assert dtype in [jnp.float32, jnp.float16, jnp.bfloat16]
        assert amax_aval.dtype == jnp.float32
        assert scale_aval.dtype == jnp.float32
        assert scale_inv_aval.dtype == jnp.float32

        casted_x_aval = x_aval.update(shape=x_aval.shape, dtype=out_dtype)
        updated_amax_aval = amax_aval.update(shape=amax_aval.shape, dtype=amax_aval.dtype)

        return casted_x_aval, updated_amax_aval

    @staticmethod
    def lowering(ctx, x, amax, scale, scale_inv, *, out_dtype):
        """
        te_cast lowering rules
        """
        x_aval, amax_aval, scale_aval, scale_inv_aval = ctx.avals_in
        assert x_aval.dtype in [jnp.float32, jnp.float16, jnp.bfloat16]
        assert amax_aval.dtype == jnp.float32
        assert scale_aval.dtype == jnp.float32
        assert scale_inv_aval.dtype == jnp.float32
        ir_x_type = ir.RankedTensorType(x.type)
        ir_x_shape = ir_x_type.shape
        ir_out_dtype = jax_dtype_to_ir_dtype(out_dtype)
        ir_amax_type = ir.RankedTensorType(amax.type)
        ir_amax_dtype = ir_amax_type.element_type
        ir_amax_shape = ir_amax_type.shape
        ir_scale_shape = ir_amax_shape
        ir_scale_inv_shape = ir_amax_shape

        out_types = [
            ir.RankedTensorType.get(ir_x_shape, ir_out_dtype),
            ir.RankedTensorType.get(ir_amax_shape, ir_amax_dtype),
        ]
        operands = [x, amax, scale, scale_inv]
        operand_shapes = [ir_x_shape, ir_amax_shape, ir_scale_shape, ir_scale_inv_shape]
        args = CustomCallArgsWrapper(out_types, operands, operand_shapes)

        opaque = transformer_engine_jax.pack_common_descriptor(
            ir_x_shape, jax_dtype_to_te_dtype(x_aval.dtype), jax_dtype_to_te_dtype(out_dtype)
        )

        out = custom_caller(
            CastFP8Primitive.name, args, opaque, False, operand_output_aliases={1: 1}
        )

        return out

    @staticmethod
    def impl(x, amax, scale, scale_inv, out_dtype):
        """
        te_cast implementation
        """
        assert CastFP8Primitive.inner_primitive is not None
        casted_x, updated_amax = CastFP8Primitive.inner_primitive.bind(
            x, amax, scale, scale_inv, out_dtype=out_dtype
        )
        return casted_x, updated_amax

    @staticmethod
    def batcher(batched_args, batch_dims, *, out_dtype):
        check_valid_batch_dims(batch_dims)
        assert CastFP8Primitive.outer_primitive is not None

        x, amax, scale, scale_inv = batched_args
        x_bdim, amax_bdim, *_ = batch_dims

        out_bdims = x_bdim, amax_bdim
        return (
            CastFP8Primitive.outer_primitive.bind(x, amax, scale, scale_inv, out_dtype=out_dtype),
            out_bdims,
        )

    @staticmethod
    def infer_sharding_from_operands(out_dtype, mesh, arg_infos, result_infos):
        del out_dtype, result_infos
        x_spec = get_padded_spec(arg_infos[0])
        casted_x_sharding = NamedSharding(mesh, PartitionSpec(*x_spec))
        amax_sharding = NamedSharding(mesh, PartitionSpec(*get_padded_spec(arg_infos[1])))
        return (casted_x_sharding, amax_sharding)

    @staticmethod
    def partition(out_dtype, mesh, arg_infos, result_infos):
        del result_infos
        x_spec = get_padded_spec(arg_infos[0])
        casted_x_sharding = NamedSharding(mesh, PartitionSpec(*x_spec))
        amax_sharding = NamedSharding(mesh, PartitionSpec(*get_padded_spec(arg_infos[1])))
        arg_shardings = tuple(arg_i.sharding for arg_i in arg_infos)
        out_shardings = (casted_x_sharding, amax_sharding)

        def sharded_impl(x, amax, scale, scale_inv):
            local_cx, local_updated_amax = CastFP8Primitive.impl(
                x, amax, scale, scale_inv, out_dtype=out_dtype
            )
            global_updated_amax = all_reduce_max_along_all_axes_except_PP(local_updated_amax)

            return local_cx, global_updated_amax

        return mesh, sharded_impl, out_shardings, arg_shardings


register_primitive(CastFP8Primitive)


def cast_fp8(
    x: jnp.ndarray,
    amax: jnp.ndarray,
    scale: jnp.ndarray,
    scale_inv: jnp.ndarray,
    out_dtype: TEDType,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Cast wrapper
    Return FP8 tensor
    """
    return CastFP8Primitive.outer_primitive.bind(x, amax, scale, scale_inv, out_dtype=out_dtype)
