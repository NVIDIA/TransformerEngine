# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""JAX/TE custom ops for activation"""
from typing import Tuple, Sequence, Union, Callable
import operator
from functools import reduce

import jax.numpy as jnp
from jax import core, dtypes
from jax.interpreters.mlir import ir
from jax.sharding import PartitionSpec, NamedSharding

from transformer_engine import transformer_engine_jax
from transformer_engine.transformer_engine_jax import NVTE_Activation_Type

from .base import BasePrimitive, register_primitive
from .custom_call import custom_caller, CustomCallArgsWrapper
from .misc import (
    check_valid_batch_dims,
    jax_dtype_to_te_dtype,
    jax_dtype_to_ir_dtype,
    get_padded_spec,
)
from ..sharding import all_reduce_max_along_all_axes_except_PP


__all__ = ["act_lu", "dact_lu", "act_lu_fp8"]


ActivationEnum = {
    ("gelu",): NVTE_Activation_Type.GELU,
    ("gelu", "linear"): NVTE_Activation_Type.GEGLU,
    ("silu",): NVTE_Activation_Type.SILU,
    ("silu", "linear"): NVTE_Activation_Type.SWIGLU,
    ("relu",): NVTE_Activation_Type.RELU,
    ("relu", "linear"): NVTE_Activation_Type.REGLU,
    ("quick_gelu",): NVTE_Activation_Type.QGELU,
    ("quick_gelu", "linear"): NVTE_Activation_Type.QGEGLU,
    ("squared_relu",): NVTE_Activation_Type.SRELU,
    ("squared_relu", "linear"): NVTE_Activation_Type.SREGLU,
}


class ActLuPrimitive(BasePrimitive):
    """
    Activation Forward Primitive
    """

    name = "te_act_lu"
    multiple_results = False
    inner_primitive = None
    outer_primitive = None
    impl_static_args = (1,)

    @staticmethod
    def abstract(x_aval, *, act_enum):  # pylint: disable=unused-argument
        """
        act_lu abstract
        """
        dtype = dtypes.canonicalize_dtype(x_aval.dtype)
        assert dtype in [jnp.float32, jnp.float16, jnp.bfloat16]

        x_shape = x_aval.shape
        assert x_shape[-2] == 2 or x_shape[-2] == 1
        hidden_size = x_shape[-1]
        batch_shapes = x_shape[:-2]
        out_aval = core.raise_to_shaped(x_aval)
        out_shape = (batch_shapes) + (hidden_size,)
        out_aval = out_aval.update(shape=out_shape, dtype=dtype)

        return out_aval

    @staticmethod
    def lowering(ctx, x, *, act_enum):
        """
        act_lu lowering rules
        """
        (x_aval,) = ctx.avals_in
        assert x_aval.dtype in [jnp.float32, jnp.float16, jnp.bfloat16]
        ir_x_type = ir.RankedTensorType(x.type)
        ir_x_shape = ir_x_type.shape
        out_shape = ir_x_shape[:-2] + [ir_x_shape[-1]]

        out_types = [
            ir.RankedTensorType.get(out_shape, ir_x_type.element_type),
        ]
        operands = [x]
        operand_shapes = [ir_x_shape]
        args = CustomCallArgsWrapper(out_types, operands, operand_shapes)

        hidden_size = ir_x_shape[-1]
        batch_size = reduce(operator.mul, ir_x_shape[:-2])
        in_dtype = jax_dtype_to_te_dtype(x_aval.dtype)
        opaque = transformer_engine_jax.pack_common_descriptor(
            (batch_size, hidden_size), in_dtype, in_dtype, act_enum
        )

        out = custom_caller(ActLuPrimitive.name, args, opaque, False)

        return [out]

    @staticmethod
    def impl(x, act_enum):
        assert ActLuPrimitive.inner_primitive is not None
        out = ActLuPrimitive.inner_primitive.bind(x, act_enum=act_enum)
        return out

    @staticmethod
    def batcher(batched_args, batch_dims, *, act_enum):
        """
        act_lu batcher
        """
        check_valid_batch_dims(batch_dims)
        assert ActLuPrimitive.outer_primitive is not None
        (inputs,) = batched_args
        (inputs_bdim,) = batch_dims

        out_bdims = inputs_bdim
        return ActLuPrimitive.outer_primitive.bind(inputs, act_enum=act_enum), out_bdims

    @staticmethod
    def infer_sharding_from_operands(act_enum, mesh, arg_infos, result_infos):
        """
        act_lu infer_sharding_from_operands
        """
        del result_infos, act_enum  # Unused.
        x_spec = get_padded_spec(arg_infos[0])
        out_sharding = NamedSharding(mesh, PartitionSpec(*x_spec[:-2], x_spec[-1]))
        return out_sharding

    @staticmethod
    def partition(act_enum, mesh, arg_infos, result_infos):
        """
        act_lu partitioning
        """
        del result_infos
        x_spec = get_padded_spec(arg_infos[0])
        arg_shardings = tuple(arg_i.sharding for arg_i in arg_infos)
        out_sharding = NamedSharding(mesh, PartitionSpec(*x_spec[:-2], x_spec[-1]))

        def sharded_impl(x):
            return ActLuPrimitive.impl(x, act_enum=act_enum)

        return mesh, sharded_impl, out_sharding, arg_shardings


register_primitive(ActLuPrimitive)


def act_lu(inputs: jnp.ndarray, activation_type: Sequence[Union[str, Callable]]) -> jnp.ndarray:
    """
    act_lu wrapper
    Return act_lu(inputs)
    Input shape: (N, 1, H) for non-gated activations
                 (N, 2, H) for gated activations
    """
    act_type_id = ActivationEnum[activation_type]
    return ActLuPrimitive.outer_primitive.bind(inputs, act_enum=act_type_id)


class DActLuPrimitive(BasePrimitive):
    """
    Dgated ActLu Primitive
    """

    name = "te_dact_lu"
    multiple_results = False
    inner_primitive = None
    outer_primitive = None
    impl_static_args = (2,)

    @staticmethod
    def abstract(dz_aval, x_aval, *, act_enum):  # pylint: disable=unused-argument
        """
        dact_lu abstract
        """
        dtype = dtypes.canonicalize_dtype(dz_aval.dtype)
        assert dtype in [jnp.float32, jnp.float16, jnp.bfloat16]
        assert x_aval.dtype == dtype
        for axis in range(len(dz_aval.shape) - 1):
            assert dz_aval.shape[axis] == x_aval.shape[axis]
        assert x_aval.shape[-2] == 2 or x_aval.shape[-2] == 1

        i_hidden_size = dz_aval.shape[-1]
        g_hidden_size = x_aval.shape[-1]
        assert i_hidden_size == g_hidden_size
        out_aval = core.raise_to_shaped(x_aval)

        return out_aval

    @staticmethod
    def lowering(ctx, dz, x, *, act_enum):
        """
        dact_lu lowering rules
        """
        in_aval, gi_aval = ctx.avals_in
        assert in_aval.dtype in [jnp.float32, jnp.float16, jnp.bfloat16]
        assert gi_aval.dtype == in_aval.dtype
        ir_in_type = ir.RankedTensorType(dz.type)
        ir_in_shape = ir_in_type.shape
        gi_type = ir.RankedTensorType(x.type)
        gi_shape = gi_type.shape
        #        assert ir_in_shape == gi_shape
        for axis in range(len(ir_in_shape) - 1):
            assert ir_in_shape[axis] == gi_shape[axis]

        ir_batch_size = reduce(operator.mul, ir_in_shape[:-1])
        i_hidden_size = ir_in_shape[-1]
        g_hidden_size = gi_shape[-1]
        assert i_hidden_size == g_hidden_size
        out_dtype = ir_in_type.element_type
        out_shape = gi_shape

        out_types = [
            ir.RankedTensorType.get(out_shape, out_dtype),
        ]
        operands = [dz, x]
        operand_shapes = [ir_in_shape, gi_shape]
        args = CustomCallArgsWrapper(out_types, operands, operand_shapes)

        in_dtype = jax_dtype_to_te_dtype(in_aval.dtype)
        opaque = transformer_engine_jax.pack_common_descriptor(
            (ir_batch_size, i_hidden_size), in_dtype, in_dtype, act_enum
        )

        out = custom_caller(DActLuPrimitive.name, args, opaque, False)

        return [out]

    @staticmethod
    def impl(dz, x, act_enum):
        """
        dact_lu implementation
        """
        assert DActLuPrimitive.inner_primitive is not None
        dx = DActLuPrimitive.inner_primitive.bind(dz, x, act_enum=act_enum)
        return dx

    @staticmethod
    def batcher(batched_args, batch_dims, *, act_enum):
        """
        dact_lu batcher
        """
        check_valid_batch_dims(batch_dims)
        assert DActLuPrimitive.outer_primitive is not None
        dz, x = batched_args
        _, x_bdim = batch_dims

        out_bdims = x_bdim
        return DActLuPrimitive.outer_primitive.bind(dz, x, act_enum=act_enum), out_bdims

    @staticmethod
    def infer_sharding_from_operands(act_enum, mesh, arg_infos, result_infos):
        """
        dact_lu infer_sharding_from_operands
        """
        del result_infos, act_enum  # Unused.
        act_lu_out_spec = get_padded_spec(arg_infos[1])
        dx_sharding = NamedSharding(mesh, PartitionSpec(*act_lu_out_spec))
        return dx_sharding

    @staticmethod
    def partition(act_enum, mesh, arg_infos, result_infos):
        """
        dact_lu partition
        """
        del result_infos
        dx_sharding = NamedSharding(mesh, PartitionSpec(*get_padded_spec(arg_infos[1])))
        arg_shardings = tuple(arg_i.sharding for arg_i in arg_infos)
        out_shardings = dx_sharding

        def sharded_impl(dz, x):
            return DActLuPrimitive.impl(dz, x, act_enum=act_enum)

        return mesh, sharded_impl, out_shardings, arg_shardings


register_primitive(DActLuPrimitive)


def dact_lu(
    inputs: jnp.ndarray, act_lu_inputs: jnp.ndarray, activation_type: Sequence[Union[str, Callable]]
) -> jnp.ndarray:
    """
    dact_lu fusion wrapper
    Return dgated_act_lu(inputs)
    """
    act_type_id = ActivationEnum[activation_type]
    return DActLuPrimitive.outer_primitive.bind(inputs, act_lu_inputs, act_enum=act_type_id)


class ActLuFp8Primitive(BasePrimitive):
    """
    ActLu FP8 Primitive
    """

    name = "te_act_lu_fp8"
    multiple_results = True
    impl_static_args = (4, 5)  # out_dtype, act_enum
    inner_primitive = None
    outer_primitive = None

    @staticmethod
    def abstract(
        x_aval, amax_aval, scale_aval, scale_inv_aval, *, out_dtype, act_enum
    ):  # pylint: disable=unused-argument
        """
        te_act_lu_p abstract
        """
        dtype = dtypes.canonicalize_dtype(x_aval.dtype)
        # Currently only support casting to E4M3 only in C side.
        assert out_dtype == jnp.float8_e4m3fn
        assert dtype in [jnp.float32, jnp.float16, jnp.bfloat16]
        assert amax_aval.dtype == jnp.float32
        assert scale_aval.dtype == jnp.float32
        assert scale_inv_aval.dtype == jnp.float32

        assert x_aval.shape[-2] == 1 or x_aval.shape[-2] == 2
        hidden_size = x_aval.shape[-1]
        batch_shape = x_aval.shape[:-2]
        out_shape = (batch_shape) + (hidden_size,)
        out_aval = x_aval.update(shape=out_shape, dtype=out_dtype)
        updated_amax_aval = amax_aval.update(shape=amax_aval.shape, dtype=amax_aval.dtype)

        return out_aval, updated_amax_aval

    @staticmethod
    def lowering(ctx, x, amax, scale, scale_inv, *, out_dtype, act_enum):
        """
        te_gated_act_lu_p lowering rules
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

        hidden_size = ir_x_shape[-1]
        batch_shape = ir_x_shape[:-2]
        batch_size = reduce(operator.mul, batch_shape)
        out_shape = batch_shape + [hidden_size]
        out_types = [
            ir.RankedTensorType.get(out_shape, ir_out_dtype),
            ir.RankedTensorType.get(ir_amax_shape, ir_amax_dtype),
        ]
        operands = [x, amax, scale, scale_inv]
        operand_shapes = [ir_x_shape, ir_amax_shape, ir_scale_shape, ir_scale_inv_shape]
        args = CustomCallArgsWrapper(out_types, operands, operand_shapes)

        opaque = transformer_engine_jax.pack_common_descriptor(
            (batch_size, hidden_size),
            jax_dtype_to_te_dtype(x_aval.dtype),
            jax_dtype_to_te_dtype(out_dtype),
            act_enum,
        )

        out = custom_caller(
            ActLuFp8Primitive.name, args, opaque, False, operand_output_aliases={1: 1}
        )

        return out

    @staticmethod
    def impl(x, amax, scale, scale_inv, out_dtype, act_enum):
        """
        to describe implementation
        """
        assert ActLuFp8Primitive.inner_primitive is not None
        out, updated_amax = ActLuFp8Primitive.inner_primitive.bind(
            x, amax, scale, scale_inv, out_dtype=out_dtype, act_enum=act_enum
        )
        return out, updated_amax

    @staticmethod
    def batcher(batched_args, batch_dims, *, out_dtype, act_enum):
        """
        to describe batch rules for vmap
        """
        check_valid_batch_dims(batch_dims)
        assert ActLuFp8Primitive.outer_primitive is not None
        x, amax, scale, scale_inv = batched_args
        x_bdim, amax_bdim, _, _ = batch_dims

        out_bdims = x_bdim, amax_bdim
        return (
            ActLuFp8Primitive.outer_primitive.bind(
                x, amax, scale, scale_inv, out_dtype=out_dtype, act_enum=act_enum
            ),
            out_bdims,
        )

    @staticmethod
    def infer_sharding_from_operands(out_dtype, act_enum, mesh, arg_infos, result_infos):
        del out_dtype, result_infos, act_enum
        x_spec = get_padded_spec(arg_infos[0])
        out_sharding = NamedSharding(mesh, PartitionSpec(*x_spec[:-2], x_spec[-1]))
        amax_sharding = NamedSharding(mesh, PartitionSpec(*get_padded_spec(arg_infos[1])))
        return (out_sharding, amax_sharding)

    @staticmethod
    def partition(out_dtype, act_enum, mesh, arg_infos, result_infos):
        del result_infos
        x_spec = get_padded_spec(arg_infos[0])
        out_sharding = NamedSharding(mesh, PartitionSpec(*x_spec[:-2], x_spec[-1]))
        amax_sharding = NamedSharding(mesh, PartitionSpec(*get_padded_spec(arg_infos[1])))
        arg_shardings = tuple(arg_i.sharding for arg_i in arg_infos)
        out_shardings = (out_sharding, amax_sharding)

        def sharded_impl(x, amax, scale, scale_inv):
            local_x, local_amax = ActLuFp8Primitive.impl(
                x, amax, scale, scale_inv, out_dtype=out_dtype, act_enum=act_enum
            )
            global_updated_amax = all_reduce_max_along_all_axes_except_PP(local_amax)

            return local_x, global_updated_amax

        return mesh, sharded_impl, out_shardings, arg_shardings


register_primitive(ActLuFp8Primitive)


def act_lu_fp8(
    x: jnp.ndarray,
    amax: jnp.ndarray,
    scale: jnp.ndarray,
    scale_inv: jnp.ndarray,
    out_dtype: jnp.dtype,
    activation_type: Sequence[Union[str, Callable]],
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    act wrapper
    Return FP8(act_lu(x))
    Input shape: (N, 1, H) for non-gated activations
                 (N, 2, H) for gated activations
    """
    act_type_id = ActivationEnum[activation_type]
    return ActLuFp8Primitive.outer_primitive.bind(
        x, amax, scale, scale_inv, out_dtype=out_dtype, act_enum=act_type_id
    )
