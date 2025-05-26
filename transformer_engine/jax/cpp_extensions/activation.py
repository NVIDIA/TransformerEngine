# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""JAX/TE custom ops for activation"""
from typing import Sequence, Union, Callable, Optional, Tuple
import operator
from functools import reduce, partial
from packaging import version

import jax
import jax.numpy as jnp
from jax import dtypes
from jax.experimental.custom_partitioning import SdyShardingRule
from jax.sharding import PartitionSpec

import transformer_engine_jax
from transformer_engine_jax import NVTE_Activation_Type

from .base import BasePrimitive, register_primitive
from .misc import (
    jax_dtype_to_te_dtype,
    te_dtype_to_jax_dtype,
    get_padded_spec,
    check_valid_batch_dims,
    multidim_transpose,
    try_apply_delayed_scaling_2x_war,
    should_apply_1x_fused_dbias_war_for_arch_l_100,
    NamedSharding,
)
from .quantization import _jax_dbias, _quantize_dbias_impl
from ..sharding import all_reduce_max_along_all_axes_except_PP, all_reduce_sum_along_dp_fsdp
from ..quantize import ScaledTensor, ScaledTensorFactory
from ..quantize import (
    Quantizer,
    QuantizeLayout,
    DelayedScaleQuantizer,
    ScalingMode,
)

if version.parse(jax.__version__) >= version.parse("0.5.0"):
    from jax import ffi  # pylint: disable=ungrouped-imports
else:
    from jax.extend import ffi  # pylint: disable=ungrouped-imports

__all__ = ["act_lu", "dact_lu", "quantize_dact_dbias"]


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


def _convert_to_activation_function(fn_or_string):
    """Convert a string to an activation function."""
    if fn_or_string == "linear":
        return lambda x: x
    if fn_or_string == "quick_gelu":
        return lambda x: jax.nn.sigmoid(1.702 * x) * x
    if fn_or_string == "squared_relu":
        return lambda x: reduce(operator.mul, [jax.nn.relu(x), jax.nn.relu(x)])
    if isinstance(fn_or_string, str):
        return getattr(jax.nn, fn_or_string)
    if callable(fn_or_string):
        return fn_or_string
    raise ValueError(f"Unsupported {fn_or_string} to an activation function")


class ActLuPrimitive(BasePrimitive):
    """
    ActLu Primitive
    """

    name = "te_act_lu_ffi"
    multiple_results = True
    impl_static_args = (
        2,
        3,
        4,
        5,
        6,
        7,
        8,
    )  # out_dtype, act_enum, act_len, scaling_mode, is_2x, scale_dtype, is_outer
    inner_primitive = None
    outer_primitive = None

    @staticmethod
    def abstract(
        x_aval,
        scale_aval,
        *,
        out_dtype,
        act_enum,
        act_len,
        scaling_mode,
        is_2x,
        scale_dtype,
        is_outer,
    ):
        """
        te_act_lu_p abstract
        """
        del act_enum
        dtype = dtypes.canonicalize_dtype(x_aval.dtype)
        assert dtype in [jnp.float32, jnp.float16, jnp.bfloat16]
        assert scale_aval is None or scale_aval.dtype == jnp.float32
        assert x_aval.shape[-2] == act_len, (
            "activation input should be replicated by act_len in the -2 axis, got input shape"
            f" {x_aval.shape} and act_len {act_len}"
        )

        assert scaling_mode != ScalingMode.CURRENT_TENSOR_SCALING.value, (
            "Current tensor scaling is not yet supported for fused activation and quantization."
            " Please do activation in higher-precision then quantize with current tensor scaling."
        )

        out_shape = (*x_aval.shape[:-2], x_aval.shape[-1])  # Exclude act dim
        out_aval = x_aval.update(shape=out_shape, dtype=out_dtype)

        updated_amax_aval = jax.core.ShapedArray(shape=(1,), dtype=jnp.float32)

        rowwise_scale_inv_shape, colwise_scale_inv_shape = ScalingMode(
            scaling_mode
        ).get_scale_shape_2x(out_shape, is_padded=not is_outer, flatten_axis=-1)
        if not is_2x:
            out_shape = (1,)
            colwise_scale_inv_shape = (1,)
        colwise_out_aval = jax.core.ShapedArray(shape=out_shape, dtype=out_dtype)
        scale_inv_aval = jax.core.ShapedArray(shape=rowwise_scale_inv_shape, dtype=scale_dtype)
        colwise_scale_inv_aval = jax.core.ShapedArray(
            shape=colwise_scale_inv_shape, dtype=scale_dtype
        )

        return out_aval, colwise_out_aval, scale_inv_aval, colwise_scale_inv_aval, updated_amax_aval

    @staticmethod
    def lowering(
        ctx,
        x,
        scale,
        *,
        out_dtype,
        act_enum,
        act_len,
        scaling_mode,
        is_2x,
        scale_dtype,
        is_outer,
    ):
        """
        te_gated_act_lu_p lowering rules
        """
        del out_dtype, scale_dtype, act_len, is_outer
        x_aval, scale_aval = ctx.avals_in
        assert x_aval.dtype in [jnp.float32, jnp.float16, jnp.bfloat16]
        assert scale_aval is None or scale_aval.dtype == jnp.float32

        out = ffi.ffi_lowering(ActLuPrimitive.name)(
            ctx, x, scale, act_enum=act_enum, scaling_mode=scaling_mode.value, is_2x=is_2x
        )
        return out

    @staticmethod
    def impl(
        x,
        scale,
        out_dtype,
        act_enum,
        act_len,
        scaling_mode,
        is_2x,
        scale_dtype,
        is_outer,
    ):
        """
        to describe implementation
        """
        del is_outer
        assert ActLuPrimitive.inner_primitive is not None

        out, colwise_out, scale_inv, colwise_scale_inv, updated_amax = (
            ActLuPrimitive.inner_primitive.bind(
                x,
                scale,
                out_dtype=out_dtype,
                act_enum=act_enum,
                act_len=act_len,
                scaling_mode=scaling_mode,
                is_2x=is_2x,
                scale_dtype=scale_dtype,
                is_outer=False,
            )
        )
        rowwise_scale_inv_shape, colwise_scale_inv_shape = ScalingMode(
            scaling_mode
        ).get_scale_shape_2x(out.shape, is_padded=False, flatten_axis=-1)
        # Slice out padding for MXFP8, noop for DelayedScaling
        scale_inv = jax.lax.slice(
            scale_inv, [0] * len(rowwise_scale_inv_shape), rowwise_scale_inv_shape
        )
        if is_2x:
            colwise_scale_inv = jax.lax.slice(
                colwise_scale_inv, [0] * len(colwise_scale_inv_shape), colwise_scale_inv_shape
            )

        return out, colwise_out, scale_inv, colwise_scale_inv, updated_amax

    @staticmethod
    def batcher(
        batched_args,
        batch_dims,
        *,
        out_dtype,
        act_enum,
        act_len,
        scaling_mode,
        is_2x,
        scale_dtype,
        is_outer,
    ):
        """
        to describe batch rules for vmap
        """
        del act_len, is_outer
        check_valid_batch_dims(batch_dims)
        assert ActLuPrimitive.outer_primitive is not None
        x, scale = batched_args
        x_bdim, scale_bdim = batch_dims
        amax_bdim = scale_bdim

        out_bdims = x_bdim, x_bdim, scale_bdim, scale_bdim, amax_bdim
        return (
            ActLuPrimitive.outer_primitive.bind(
                x,
                scale,
                out_dtype=out_dtype,
                act_enum=act_enum,
                scaling_mode=scaling_mode,
                is_2x=is_2x,
                scale_dtype=scale_dtype,
            ),
            out_bdims,
        )

    @staticmethod
    def infer_sharding_from_operands(
        out_dtype,
        act_enum,
        act_len,
        scaling_mode,
        is_2x,
        scale_dtype,
        is_outer,
        mesh,
        arg_infos,
        result_infos,
    ):
        del (
            out_dtype,
            result_infos,
            act_enum,
            scale_dtype,
            act_len,
            is_outer,
        )  # Unused.
        x_spec = get_padded_spec(arg_infos[0])
        scale_spec = get_padded_spec(arg_infos[1])

        out_spec = (*x_spec[:-2], x_spec[-1])
        out_sharding = NamedSharding(mesh, PartitionSpec(*out_spec), desc="ActLuPrimitive.out")

        if is_2x:
            if scaling_mode == ScalingMode.DELAYED_TENSOR_SCALING.value:
                colwise_out_spec = multidim_transpose(out_spec, transpose_axis=-1)
            else:
                colwise_out_spec = out_spec
        else:
            colwise_out_spec = (None,)
        colwise_out_sharding = NamedSharding(
            mesh, PartitionSpec(*colwise_out_spec), desc="ActLuPrimitive.colwise_out"
        )

        scale_inv_spec = amax_spec = colwise_scale_inv_spec = (None,)
        if scaling_mode == ScalingMode.DELAYED_TENSOR_SCALING.value:
            scale_inv_spec = amax_spec = scale_spec
        elif scaling_mode == ScalingMode.MXFP8_1D_SCALING.value:
            scale_inv_spec = out_spec

        if is_2x:
            colwise_scale_inv_spec = scale_inv_spec

        scale_inv_sharding = NamedSharding(
            mesh, PartitionSpec(*scale_inv_spec), desc="ActLuPrimitive.scale_inv"
        )
        amax_sharding = NamedSharding(mesh, PartitionSpec(*amax_spec), desc="ActLuPrimitive.amax")
        colwise_scale_inv_sharding = NamedSharding(
            mesh, PartitionSpec(*colwise_scale_inv_spec), desc="ActLuPrimitive.colwise_scale_inv"
        )

        return (
            out_sharding,
            colwise_out_sharding,
            scale_inv_sharding,
            colwise_scale_inv_sharding,
            amax_sharding,
        )

    @staticmethod
    def partition(
        out_dtype,
        act_enum,
        act_len,
        scaling_mode,
        is_2x,
        scale_dtype,
        is_outer,
        mesh,
        arg_infos,
        result_infos,
    ):
        del result_infos, is_outer  # Unused.
        x_spec = get_padded_spec(arg_infos[0])
        scale_spec = get_padded_spec(arg_infos[1])

        out_spec = (*x_spec[:-2], x_spec[-1])
        out_sharding = NamedSharding(mesh, PartitionSpec(*out_spec), desc="ActLuPrimitive.out")

        if is_2x:
            if scaling_mode == ScalingMode.DELAYED_TENSOR_SCALING.value:
                colwise_out_spec = multidim_transpose(out_spec, transpose_axis=-1)
            else:
                colwise_out_spec = out_spec
        else:
            colwise_out_spec = (None,)
        colwise_out_sharding = NamedSharding(
            mesh, PartitionSpec(*colwise_out_spec), desc="ActLuPrimitive.colwise_out"
        )

        scale_inv_spec = amax_spec = colwise_scale_inv_spec = (None,)
        if scaling_mode == ScalingMode.DELAYED_TENSOR_SCALING.value:
            scale_inv_spec = amax_spec = scale_spec
        elif scaling_mode == ScalingMode.MXFP8_1D_SCALING.value:
            scale_inv_spec = out_spec

        if is_2x:
            colwise_scale_inv_spec = scale_inv_spec

        scale_inv_sharding = NamedSharding(
            mesh, PartitionSpec(*scale_inv_spec), desc="ActLuPrimitive.scale_inv"
        )
        amax_sharding = NamedSharding(mesh, PartitionSpec(*amax_spec), desc="ActLuPrimitive.amax")
        colwise_scale_inv_sharding = NamedSharding(
            mesh, PartitionSpec(*colwise_scale_inv_spec), desc="ActLuPrimitive.colwise_scale_inv"
        )

        arg_shardings = tuple(arg_i.sharding for arg_i in arg_infos)
        out_shardings = (
            out_sharding,
            colwise_out_sharding,
            scale_inv_sharding,
            colwise_scale_inv_sharding,
            amax_sharding,
        )

        def sharded_impl(x, scale):
            local_x, local_colwise_x, local_scale_inv, local_colwise_scale_inv, local_amax = (
                ActLuPrimitive.impl(
                    x,
                    scale,
                    out_dtype=out_dtype,
                    act_enum=act_enum,
                    act_len=act_len,
                    scaling_mode=scaling_mode,
                    is_2x=is_2x,
                    scale_dtype=scale_dtype,
                    is_outer=True,
                )
            )

            if scaling_mode == ScalingMode.DELAYED_TENSOR_SCALING.value:
                global_updated_amax = all_reduce_max_along_all_axes_except_PP(local_amax, mesh)
            else:
                global_updated_amax = local_amax

            return (
                local_x,
                local_colwise_x,
                local_scale_inv,
                local_colwise_scale_inv,
                global_updated_amax,
            )

        return mesh, sharded_impl, out_shardings, arg_shardings

    @staticmethod
    def shardy_sharding_rule(
        out_dtype,
        act_enum,
        act_len,
        scaling_mode,
        is_2x,
        scale_dtype,
        is_outer,
        mesh,
        value_types,
        result_types,
    ):
        del out_dtype, act_enum, act_len, scale_dtype, is_outer, mesh, result_types

        x_rank = len(value_types[0].shape)
        scale_rules = ScalingMode(scaling_mode).get_shardy_sharding_rules(
            x_rank - 1, unique_var="ActLuPrimitive_i", flatten_axis=-2
        )
        x_axes = scale_rules.input_spec + (f"x{x_rank-1}",)
        out = (*x_axes[:-2], x_axes[-1])
        scale_inv = scale_rules.rowwise_rule
        colwise_scale_inv = scale_rules.colwise_rule

        if is_2x:
            if scaling_mode == ScalingMode.DELAYED_TENSOR_SCALING.value:
                colwise_out = tuple(
                    multidim_transpose(x_axes, static_axis_boundary=-1, transpose_axis=-2)
                )
            else:
                colwise_out = out
        else:
            colwise_out = ("j",)
            colwise_scale_inv = ("k",)

        # amax is always a unit tensor.
        amax = ("l",)

        return SdyShardingRule(
            (
                x_axes,
                "…1",
            ),
            (out, colwise_out, scale_inv, colwise_scale_inv, amax),
            **scale_rules.factor_sizes,
        )


register_primitive(ActLuPrimitive)


# TODO(Jeremy): replace is_2x with q_layout
class BaseDActLuDBiasQuantizePrimitive(BasePrimitive):
    """
    DActLu DBias Cast Transpose Primitive
    """

    name = "te_dact_dbias_quantize_ffi"
    multiple_results = True
    # out_dtype, scaling_mode, is_2x, scale_dtype, is_dbias, act_enum, act_len, is_outer
    impl_static_args = (3, 4, 5, 6, 7, 8, 9, 10)
    inner_primitive = None
    outer_primitive = None

    @staticmethod
    def abstract(
        dz_aval,
        x_aval,
        scale_aval,
        *,
        out_dtype,
        scaling_mode,
        is_2x,
        scale_dtype,
        is_dbias,
        act_enum,
        act_len,
        is_outer,
    ):
        """
        te_dact_dbias_quantize_p abstract
        """
        del act_enum
        dz_dtype = dtypes.canonicalize_dtype(dz_aval.dtype)
        assert dz_dtype in [jnp.float32, jnp.float16, jnp.bfloat16]
        assert x_aval.dtype == dz_dtype
        assert x_aval.shape[-2] == act_len, (
            "activation input should be replicated by act_len in the -2 axis, got input shape"
            f" {x_aval.shape} and act_len {act_len}"
        )
        assert scale_aval.dtype == jnp.float32

        assert scaling_mode != ScalingMode.CURRENT_TENSOR_SCALING.value, (
            "Current tensor scaling is not supported for fused dact and quantization. Please do"
            " dact in higher-precision then quantize with current tensor scaling."
        )

        ir_hidden_size = dz_aval.shape[-1]
        gi_hidden_size = act_len * x_aval.shape[-1]
        assert act_len * ir_hidden_size == gi_hidden_size
        assert (
            x_aval.shape[:-2] == dz_aval.shape[:-1]
        ), "dz and x should have the same leading dimensions"
        out_shape = x_aval.shape
        out_aval = x_aval.update(shape=out_shape, dtype=out_dtype)

        updated_amax_aval = jax.core.ShapedArray(shape=(1,), dtype=jnp.float32)

        rowwise_scale_inv_shape, colwise_scale_inv_shape = ScalingMode(
            scaling_mode
        ).get_scale_shape_2x(x_aval.shape, is_padded=not is_outer, flatten_axis=-2)
        if is_2x:
            if ScalingMode(scaling_mode).is_tensor_scaling():
                colwise_out_shape = multidim_transpose(out_shape, transpose_axis=-2)
            else:
                colwise_out_shape = out_shape
        else:
            colwise_out_shape = (1,)
            colwise_scale_inv_shape = (1,)
        colwise_out_aval = jax.core.ShapedArray(shape=colwise_out_shape, dtype=out_dtype)
        scale_inv_aval = jax.core.ShapedArray(shape=rowwise_scale_inv_shape, dtype=scale_dtype)
        colwise_scale_inv_aval = jax.core.ShapedArray(
            shape=colwise_scale_inv_shape, dtype=scale_dtype
        )

        if is_dbias:
            dbias_shape = (act_len, ir_hidden_size)
            (wkspace_info,) = transformer_engine_jax.get_dact_dbias_quantize_workspace_sizes(
                x_aval.size // gi_hidden_size,
                gi_hidden_size,
                jax_dtype_to_te_dtype(x_aval.dtype),
                jax_dtype_to_te_dtype(out_dtype),
                scaling_mode,
                is_2x,
            )
            wkspace_shape = wkspace_info[0]
            wkspace_dtype = te_dtype_to_jax_dtype(wkspace_info[1])
        else:
            dbias_shape = (1,)
            wkspace_shape = (1,)
            wkspace_dtype = jnp.float32
        dbias_aval = jax.core.ShapedArray(shape=dbias_shape, dtype=dz_dtype)
        wkspace_aval = jax.core.ShapedArray(shape=wkspace_shape, dtype=wkspace_dtype)

        return (
            out_aval,
            colwise_out_aval,
            scale_inv_aval,
            colwise_scale_inv_aval,
            updated_amax_aval,
            dbias_aval,
            wkspace_aval,
        )

    @staticmethod
    def outer_abstract(*args, **kwargs):
        """
        te_dact_dbias_quantize_p outer abstract
        """
        (out, colwise_out, scale_inv, colwise_scale_inv, updated_amax, dbias, _) = (
            BaseDActLuDBiasQuantizePrimitive.abstract(*args, **kwargs)
        )
        return out, colwise_out, scale_inv, colwise_scale_inv, updated_amax, dbias

    @staticmethod
    def lowering(
        ctx,
        dz,
        x,
        scale,
        *,
        out_dtype,
        scaling_mode,
        is_2x,
        scale_dtype,
        is_dbias,
        act_enum,
        act_len,
        is_outer,
    ):
        """
        te_dact_dbias_quantize_p lowering rules
        """
        del out_dtype, scale_dtype, act_len, is_outer
        dz_aval, x_aval, scale_aval = ctx.avals_in
        assert dz_aval.dtype in [jnp.float32, jnp.float16, jnp.bfloat16]
        assert x_aval.dtype == dz_aval.dtype
        assert scale_aval.dtype == jnp.float32
        return ffi.ffi_lowering(BaseDActLuDBiasQuantizePrimitive.name)(
            ctx,
            dz,
            x,
            scale,
            scaling_mode=scaling_mode.value,
            is_2x=is_2x,
            is_dbias=is_dbias,
            act_enum=int(act_enum),
        )

    @staticmethod
    def impl(
        dz,
        x,
        scale,
        out_dtype,
        scaling_mode,
        is_2x,
        scale_dtype,
        is_dbias,
        act_enum,
        act_len,
        is_outer,
    ):
        """
        te_dact_dbias_quantize_p impl
        """
        del is_outer
        assert BaseDActLuDBiasQuantizePrimitive.inner_primitive is not None
        (out, colwise_out, scale_inv, colwise_scale_inv, updated_amax, dbias, _) = (
            BaseDActLuDBiasQuantizePrimitive.inner_primitive.bind(
                dz,
                x,
                scale,
                out_dtype=out_dtype,
                scaling_mode=scaling_mode,
                is_2x=is_2x,
                scale_dtype=scale_dtype,
                is_dbias=is_dbias,
                act_enum=act_enum,
                act_len=act_len,
                is_outer=False,
            )
        )
        rowwise_scale_inv_shape, colwise_scale_inv_shape = ScalingMode(
            scaling_mode
        ).get_scale_shape_2x(x.shape, is_padded=False, flatten_axis=-2)
        # Slice out padding for MXFP8, noop for DelayedScaling
        scale_inv = jax.lax.slice(
            scale_inv, [0] * len(rowwise_scale_inv_shape), rowwise_scale_inv_shape
        )
        if is_2x:
            colwise_scale_inv = jax.lax.slice(
                colwise_scale_inv, [0] * len(colwise_scale_inv_shape), colwise_scale_inv_shape
            )
        return out, colwise_out, scale_inv, colwise_scale_inv, updated_amax, dbias

    @staticmethod
    def batcher(
        batched_args,
        batch_dims,
        *,
        out_dtype,
        scaling_mode,
        is_2x,
        scale_dtype,
        is_dbias,
        act_enum,
        act_len,
        is_outer,
    ):
        """
        to describe batch rules for vmap
        """
        del is_outer
        check_valid_batch_dims(batch_dims)
        assert BaseDActLuDBiasQuantizePrimitive.outer_primitive is not None
        dz, x, scale = batched_args
        _, x_bdim, scale_bdim = batch_dims

        out_bdims = (
            x_bdim,  # rowwise output
            scale_bdim,  # rowwise scale_inv
            x_bdim,  # colwise output
            scale_bdim,  # colwise scale_inv
            scale_bdim,  # amax
            x_bdim,  # dbias
        )
        return (
            BaseDActLuDBiasQuantizePrimitive.outer_primitive.bind(
                dz,
                x,
                scale,
                out_dtype=out_dtype,
                scaling_mode=scaling_mode,
                is_2x=is_2x,
                scale_dtype=scale_dtype,
                is_dbias=is_dbias,
                act_enum=act_enum,
                act_len=act_len,
            ),
            out_bdims,
        )

    @staticmethod
    def infer_sharding_from_operands(
        out_dtype,
        scaling_mode,
        is_2x,
        scale_dtype,
        is_dbias,
        act_enum,
        act_len,
        is_outer,
        mesh,
        arg_infos,
        result_infos,
    ):
        del out_dtype, result_infos, act_enum
        del scale_dtype, act_len, is_outer
        x_spec = get_padded_spec(arg_infos[1])
        scale_spec = get_padded_spec(arg_infos[2])

        assert (
            scaling_mode != ScalingMode.CURRENT_TENSOR_SCALING.value
        ), "Partitioned current tensor scaling is not yet supported."

        out_sharding = NamedSharding(
            mesh, PartitionSpec(*x_spec), desc="BaseDActLuDBiasQuantizePrimitive.out"
        )
        if is_2x:
            if scaling_mode == ScalingMode.DELAYED_TENSOR_SCALING.value:
                colwise_x_spec = multidim_transpose(x_spec, transpose_axis=-2)
            else:
                colwise_x_spec = x_spec
        else:
            colwise_x_spec = (None,)
        colwise_out_sharding = NamedSharding(
            mesh,
            PartitionSpec(*colwise_x_spec),
            desc="BaseDActLuDBiasQuantizePrimitive.colwise_out",
        )

        dbias_spec = x_spec[-2:] if is_dbias else (None,)
        dbias_sharding = NamedSharding(
            mesh,
            PartitionSpec(*dbias_spec),
            desc="BaseDActLuDBiasQuantizePrimitive.dbias",
        )

        scale_inv_spec = amax_spec = colwise_scale_inv_spec = (None,)
        if scaling_mode == ScalingMode.DELAYED_TENSOR_SCALING.value:
            scale_inv_spec = amax_spec = scale_spec
        elif scaling_mode == ScalingMode.MXFP8_1D_SCALING.value:
            scale_inv_spec = x_spec

        if is_2x:
            colwise_scale_inv_spec = scale_inv_spec

        scale_inv_sharding = NamedSharding(
            mesh, PartitionSpec(*scale_inv_spec), desc="BaseDActLuDBiasQuantizePrimitive.scale_inv"
        )
        amax_sharding = NamedSharding(
            mesh, PartitionSpec(*amax_spec), desc="BaseDActLuDBiasQuantizePrimitive.amax"
        )
        colwise_scale_inv_sharding = NamedSharding(
            mesh,
            PartitionSpec(*colwise_scale_inv_spec),
            desc="BaseDActLuDBiasQuantizePrimitive.colwise_scale_inv",
        )
        return (
            out_sharding,
            colwise_out_sharding,
            scale_inv_sharding,
            colwise_scale_inv_sharding,
            amax_sharding,
            dbias_sharding,
        )

    @staticmethod
    def partition(
        out_dtype,
        scaling_mode,
        is_2x,
        scale_dtype,
        is_dbias,
        act_enum,
        act_len,
        is_outer,
        mesh,
        arg_infos,
        result_infos,
    ):
        del result_infos, is_outer
        x_spec = get_padded_spec(arg_infos[1])
        scale_spec = get_padded_spec(arg_infos[2])

        out_sharding = NamedSharding(
            mesh, PartitionSpec(*x_spec), desc="BaseDActLuDBiasQuantizePrimitive.out"
        )

        if is_2x:
            if scaling_mode == ScalingMode.DELAYED_TENSOR_SCALING.value:
                colwise_x_spec = multidim_transpose(x_spec, transpose_axis=-2)
            else:
                colwise_x_spec = x_spec
        else:
            colwise_x_spec = (None,)
        colwise_out_sharding = NamedSharding(
            mesh,
            PartitionSpec(*colwise_x_spec),
            desc="BaseDActLuDBiasQuantizePrimitive.colwise_out",
        )

        dbias_spec = x_spec[-2:] if is_dbias else (None,)
        dbias_sharding = NamedSharding(
            mesh,
            PartitionSpec(*dbias_spec),
            desc="BaseDActLuDBiasQuantizePrimitive.dbias",
        )

        scale_inv_spec = amax_spec = colwise_scale_inv_spec = (None,)
        if scaling_mode == ScalingMode.DELAYED_TENSOR_SCALING.value:
            scale_inv_spec = amax_spec = scale_spec
        elif scaling_mode == ScalingMode.MXFP8_1D_SCALING.value:
            scale_inv_spec = x_spec

        if is_2x:
            colwise_scale_inv_spec = scale_inv_spec

        scale_inv_sharding = NamedSharding(
            mesh, PartitionSpec(*scale_inv_spec), desc="ActLuPrimitive.scale_inv"
        )
        amax_sharding = NamedSharding(mesh, PartitionSpec(*amax_spec), desc="ActLuPrimitive.amax")
        colwise_scale_inv_sharding = NamedSharding(
            mesh, PartitionSpec(*colwise_scale_inv_spec), desc="ActLuPrimitive.colwise_scale_inv"
        )

        arg_shardings = list(arg_i.sharding for arg_i in arg_infos)
        # Ensure dz and x are partitioned the same way.
        arg_shardings[0] = NamedSharding(
            mesh,
            PartitionSpec(*x_spec[:-2], x_spec[-1]),
            desc="BaseDActLuDBiasQuantizePrimitive.dz",
        )
        arg_shardings = tuple(arg_shardings)
        out_shardings = (
            out_sharding,
            colwise_out_sharding,
            scale_inv_sharding,
            colwise_scale_inv_sharding,
            amax_sharding,
            dbias_sharding,
        )

        def sharded_impl(dz, x, scale):
            (out, colwise_out, scale_inv, colwise_scale_inv, local_amax, local_dbias) = (
                BaseDActLuDBiasQuantizePrimitive.impl(
                    dz,
                    x,
                    scale,
                    out_dtype=out_dtype,
                    scaling_mode=scaling_mode,
                    is_2x=is_2x,
                    scale_dtype=scale_dtype,
                    is_dbias=is_dbias,
                    act_enum=act_enum,
                    act_len=act_len,
                    is_outer=True,
                )
            )
            if is_dbias:
                global_dbias = all_reduce_sum_along_dp_fsdp(local_dbias, mesh)
            else:
                global_dbias = local_dbias

            if scaling_mode == ScalingMode.DELAYED_TENSOR_SCALING.value:
                global_updated_amax = all_reduce_max_along_all_axes_except_PP(local_amax, mesh)
            else:
                global_updated_amax = local_amax

            return out, colwise_out, scale_inv, colwise_scale_inv, global_updated_amax, global_dbias

        return mesh, sharded_impl, out_shardings, arg_shardings

    @staticmethod
    def shardy_sharding_rule(
        out_dtype,
        scaling_mode,
        is_2x,
        scale_dtype,
        is_dbias,
        act_enum,
        act_len,
        is_outer,
        mesh,
        value_types,
        result_types,
    ):
        del out_dtype, scale_dtype, act_enum, act_len, is_outer, mesh, result_types

        x_rank = len(value_types[1].shape)
        scale_rules = ScalingMode(scaling_mode).get_shardy_sharding_rules(
            x_rank, unique_var="BaseDActLuDBiasQuantizePrimitive_i", flatten_axis=-2
        )
        x_axes = scale_rules.input_spec
        out = x_axes
        if is_2x:
            if scaling_mode == ScalingMode.DELAYED_TENSOR_SCALING.value:
                colwise_out = tuple(multidim_transpose(x_axes, transpose_axis=-2))
            else:
                colwise_out = tuple(x_axes)
        else:
            colwise_out = ("j",)

        dbias = x_axes[-2:] if is_dbias else ("k",)
        amax = ("…4",)

        return SdyShardingRule(
            (("…0",), tuple(x_axes), ("…2",)),
            (out, colwise_out, scale_rules.rowwise_rule, scale_rules.colwise_rule, amax, dbias),
            **scale_rules.factor_sizes,
        )


register_primitive(BaseDActLuDBiasQuantizePrimitive)


class DActLuDBiasQuantizePrimitive(BaseDActLuDBiasQuantizePrimitive):
    """Subclass of BaseDActLuDBiasQuantizePrimitive for DBias and fused activation quantization. No change in functionality from the base primitive but named differently for use in more granular disabling of primitives via NVTE_JAX_CUSTOM_CALLS_RE."""


class DActLuQuantizePrimitive(BaseDActLuDBiasQuantizePrimitive):
    """Subclass of BaseDActLuDBiasQuantizePrimitive for fused activation quantization without dbias. No change in functionality from the base primitive but named differently for use in more granular disabling of primitives via NVTE_JAX_CUSTOM_CALLS_RE."""


def _jax_act_lu(inputs, activation_type, quantizer=None) -> Union[jnp.ndarray, ScaledTensor]:
    """
    JAX native activation implementation
    """
    act_len = len(activation_type)
    assert inputs.shape[-2] == act_len, (
        "activation input should be replicated by act_len in the -2 axis, got input shape"
        f" {inputs.shape} and act_len {act_len}"
    )

    x = jnp.split(inputs, act_len, axis=-2)
    acts = []
    for idx, act_fn in enumerate(activation_type):
        x_i = _convert_to_activation_function(act_fn)(x[idx])
        acts.append(x_i)
    x = reduce(operator.mul, acts)
    x = jnp.squeeze(x, axis=-2)
    if quantizer:
        return quantizer.quantize(x, flatten_axis=-1)
    return x


def _jax_quantize_dact_dbias(
    dz: jnp.ndarray,
    x: jnp.ndarray,
    activation_type: Sequence[Union[str, Callable]],
    is_dbias: bool = True,
    quantizer: Optional[Quantizer] = None,
):
    """
    JAX implementation of dact_lu and dbias with optional quantization
    """
    act_len = len(activation_type)
    assert x.shape[-2] == act_len, (
        "activation input should be replicated by act_len in the -2 axis, got input shape"
        f" {x.shape} and act_len {act_len}"
    )

    _, vjp_func = jax.vjp(
        partial(_jax_act_lu, activation_type=activation_type), x.astype(jnp.float32)
    )
    (dx,) = vjp_func(dz.astype(jnp.float32))

    dbias = None
    if is_dbias:
        dbias = _jax_dbias(dx, dtype=x.dtype, flatten_axis=-2)

    if quantizer is not None:
        dx = quantizer.quantize(dx, dq_dtype=x.dtype, flatten_axis=-2)
    else:
        dx = dx.astype(x.dtype)

    return dx, dbias


def act_lu(
    x: jnp.ndarray,
    activation_type: Sequence[Union[str, Callable]],
    quantizer: Optional[Quantizer] = None,
) -> Union[jnp.ndarray, ScaledTensor]:
    """Activation with optional quantization.

    Args:
        x: Input tensor to be processed.
            Shape: (..., ACT_DIM, K) where ACT_DIM is 1 for non-gated activations and 2 for gated activations
        activation_type: Type of activation function to apply.
        quantizer: Optional quantizer for FP8 quantization of the output.

    Returns:
        If quantizer is None:
            The activated input tensor with the same dtype as input.
        If quantizer is provided:
            A ScaledTensor containing the quantized activated input.
    """
    act_type_id = ActivationEnum[activation_type].value
    act_len = len(activation_type)
    assert x.shape[-2] == act_len, (
        "activation input should be replicated by act_len in the -2 axis, got input shape"
        f" {x.shape} and act_len {act_len}"
    )

    if not ActLuPrimitive.enabled():
        return _jax_act_lu(x, activation_type, quantizer)

    # TE/common does not support colwise-only quantization yet
    if quantizer is not None and quantizer.q_layout == QuantizeLayout.COLWISE:
        return _jax_act_lu(x, activation_type, quantizer)

    # TE/common does not support 2x quantization for DelayedScaling yet
    war_output = try_apply_delayed_scaling_2x_war(
        f=act_lu, x=x, activation_type=activation_type, quantizer=quantizer
    )
    if war_output is not None:
        return war_output

    scale = jnp.empty((1,), jnp.float32)
    output_shape = (*x.shape[:-2], x.shape[-1])

    if quantizer is None:
        out, _, _, _, _ = ActLuPrimitive.outer_primitive.bind(
            x,
            scale,
            out_dtype=x.dtype,
            act_enum=act_type_id,
            act_len=act_len,
            scaling_mode=ScalingMode.NO_SCALING.value,
            is_2x=False,
            scale_dtype=jnp.float32,
            is_outer=True,
        )
        out = out.reshape(output_shape)
        return out

    if quantizer.scaling_mode == ScalingMode.CURRENT_TENSOR_SCALING:
        # Current scaling does not support fused operations. Perform dact in higher precision then quantize after.
        out = act_lu(
            x=x.astype(jnp.float32),
            activation_type=activation_type,
            quantizer=None,
        )
        out, _ = _quantize_dbias_impl(out, is_dbias=False, quantizer=quantizer, dq_dtype=x.dtype)
        return out

    if isinstance(quantizer, DelayedScaleQuantizer):
        scale = quantizer.scale

    (
        rowwise_casted_output,
        colwise_casted_output,
        rowwise_scale_inv,
        colwise_scale_inv,
        updated_amax,
    ) = ActLuPrimitive.outer_primitive.bind(
        x,
        scale,
        out_dtype=quantizer.q_dtype,
        act_enum=act_type_id,
        act_len=act_len,
        scaling_mode=quantizer.scaling_mode.value,
        is_2x=quantizer.is_2x2x(),
        scale_dtype=quantizer.get_scale_dtype(),
        is_outer=True,
    )

    quantizer.update(updated_amax)

    return ScaledTensorFactory.create(
        data=rowwise_casted_output,
        scale_inv=rowwise_scale_inv,
        colwise_data=colwise_casted_output,
        colwise_scale_inv=colwise_scale_inv,
        scaling_mode=quantizer.scaling_mode,
        dq_dtype=x.dtype,
        q_layout=quantizer.q_layout,
        data_layout=quantizer.get_data_layout(),
    )


def quantize_dact_dbias(
    dz: jnp.ndarray,
    x: jnp.ndarray,
    activation_type: Sequence[Union[str, Callable]] = ("gelu",),
    is_dbias: bool = True,
    quantizer: Optional[Quantizer] = None,
) -> Tuple[ScaledTensor, jnp.ndarray]:
    """Compute gradients of activation and bias with optional quantization.

    Args:
        dz: Gradient of the output with respect to the activation output.
        x: Input tensor that was processed by the forward pass.
            Shape: (..., ACT_DIM, K) where ACT_DIM is 1 for non-gated activations and 2 for gated activations
        activation_type: Type of activation function used in the forward pass. Defaults to ("gelu",).
        is_dbias: If True, compute bias gradient. Defaults to True.
        quantizer: Optional quantizer for FP8 quantization of the output.

    Returns:
        Tuple[ScaledTensor, jnp.ndarray]: A tuple containing:
        - The gradient of the activation with respect to the input.
        - The gradient of the activation with respect to the bias.
    """

    act_len = len(activation_type)
    assert x.shape[-2] == act_len, (
        "activation input should be replicated by act_len in the -2 axis, got input shape"
        f" {x.shape} and act_len {act_len}"
    )

    PrimitiveClass = DActLuDBiasQuantizePrimitive if is_dbias else DActLuQuantizePrimitive
    if not PrimitiveClass.enabled():
        return _jax_quantize_dact_dbias(dz, x, activation_type, is_dbias, quantizer)

    # TE/common does not support colwise-only quantization yet
    if quantizer is not None and quantizer.q_layout == QuantizeLayout.COLWISE:
        return _jax_quantize_dact_dbias(dz, x, activation_type, is_dbias, quantizer)

    # TE/common does not support 1x dact_dbias_quantize on arch < 100 yet
    if should_apply_1x_fused_dbias_war_for_arch_l_100(is_dbias=is_dbias, quantizer=quantizer):
        out = dact_lu(
            dz.astype(jnp.float32), x.astype(jnp.float32), activation_type, quantizer=None
        )
        return _quantize_dbias_impl(
            out, quantizer, is_dbias=True, dq_dtype=x.dtype, flatten_axis=-2
        )

    is_gated = act_len == 2
    # TE/common does not support DelayedScaling2x for gated-act yet
    if is_gated:
        war_output = try_apply_delayed_scaling_2x_war(
            f=quantize_dact_dbias,
            dz=dz,
            x=x,
            activation_type=activation_type,
            is_dbias=is_dbias,
            quantizer=quantizer,
            flatten_axis=-2,
        )
        if war_output is not None:
            return war_output

    scale = jnp.empty((), jnp.float32)

    act_type_id = ActivationEnum[activation_type]

    if quantizer is None:
        output, _, _, _, _, _ = PrimitiveClass.outer_primitive.bind(
            dz,
            x,
            scale,
            # outputs float32 for dbias accumulation
            out_dtype=(jnp.float32 if is_dbias else x.dtype),
            # default value for no scaling, TE/common ignore this value when scale is unset
            scaling_mode=ScalingMode.NO_SCALING.value,
            is_2x=False,  # unused
            scale_dtype=jnp.float32,  # unused
            is_dbias=False,
            act_enum=act_type_id,
            act_len=act_len,
            is_outer=True,
        )
        dbias = None
        if is_dbias:
            dbias = _jax_dbias(output, dtype=x.dtype, flatten_axis=-2)
        return output.astype(x.dtype), dbias

    if quantizer.scaling_mode == ScalingMode.CURRENT_TENSOR_SCALING:
        # Current scaling does not support fused operations. Perform dact in higher precision then quantize after.
        out = dact_lu(
            dz=dz.astype(jnp.float32),
            x=x.astype(jnp.float32),
            activation_type=activation_type,
            quantizer=None,
        )
        out, dbias = _quantize_dbias_impl(
            out, is_dbias=is_dbias, quantizer=quantizer, dq_dtype=x.dtype, flatten_axis=-2
        )
        return out, dbias

    if isinstance(quantizer, DelayedScaleQuantizer):
        scale = quantizer.scale

    # TE/common dact_dbias_quantize does not support gated act yet
    if is_dbias and is_gated:
        dgated = dact_lu(
            dz.astype(jnp.float32), x.astype(jnp.float32), activation_type=activation_type
        )
        out, dbias = _quantize_dbias_impl(
            dgated, quantizer, is_dbias=True, dq_dtype=x.dtype, flatten_axis=-2
        )
        return out, dbias

    (
        rowwise_casted_output,
        colwise_casted_output,
        rowwise_scale_inv,
        colwise_scale_inv,
        updated_amax,
        dbias,
    ) = PrimitiveClass.outer_primitive.bind(
        dz,
        x,
        scale,
        out_dtype=quantizer.q_dtype,
        scaling_mode=quantizer.scaling_mode.value,
        is_2x=quantizer.is_2x2x(),
        scale_dtype=quantizer.get_scale_dtype(),
        is_dbias=is_dbias,
        act_enum=act_type_id,
        act_len=act_len,
        is_outer=True,
    )

    # For DelayedScaling transpose, the scale buffer is shared for both rowwise and colwise
    if quantizer.scaling_mode.is_tensor_scaling() and quantizer.is_2x2x():
        colwise_scale_inv = rowwise_scale_inv

    quantizer.update(updated_amax)

    out = ScaledTensorFactory.create(
        data=rowwise_casted_output,
        scale_inv=rowwise_scale_inv,
        colwise_data=colwise_casted_output,
        colwise_scale_inv=colwise_scale_inv,
        scaling_mode=quantizer.scaling_mode,
        dq_dtype=x.dtype,
        q_layout=quantizer.q_layout,
        data_layout=quantizer.get_data_layout(),
        flatten_axis=-2,  # as output has act axis
    )

    return out, dbias


def dact_lu(
    dz: jnp.ndarray,
    x: jnp.ndarray,
    activation_type: Sequence[Union[str, Callable]],
    quantizer: Optional[Quantizer] = None,
) -> Union[jnp.ndarray, ScaledTensor]:
    """
    Backward pass for activation with optional quantization.

    Args:
        dz: Gradient tensor from upstream.
        x: Input tensor that was used in forward pass.
        activation_type: Type of activation function that was applied.
        quantizer: Optional quantizer for FP8 quantization of the output gradient.

    Returns:
        The gradient of the activation with respect to the input.
    """
    output, _ = quantize_dact_dbias(
        dz=dz,
        x=x,
        activation_type=activation_type,
        is_dbias=False,
        quantizer=quantizer,
    )
    return output
