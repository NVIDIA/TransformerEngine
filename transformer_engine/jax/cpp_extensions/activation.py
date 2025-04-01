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
from .quantization import _jax_quantize_dbias, _jax_dbias, quantize_dbias
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
        9,
    )  # out_dtype, act_enum, act_len, scaling_mode, is_2x, scale_dtype, scale_shapes, is_outer
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
        scale_shapes,
        is_outer,
    ):
        """
        te_act_lu_p abstract
        """
        del act_enum, act_len, scale_shapes
        dtype = dtypes.canonicalize_dtype(x_aval.dtype)
        assert dtype in [jnp.float32, jnp.float16, jnp.bfloat16]
        assert scale_aval is None or scale_aval.dtype == jnp.float32

        out_shape = (
            *x_aval.shape[:-2],
            1,
            x_aval.shape[-1],
        )
        out_aval = x_aval.update(shape=out_shape, dtype=out_dtype)
        updated_amax_aval = jax.core.ShapedArray(shape=(1,), dtype=jnp.float32)

        rowwise_scale_inv_shape, colwise_scale_inv_shape = ScalingMode(
            scaling_mode
        ).get_scale_shape_2x(out_shape[:-2] + (out_shape[-1],), is_padded=not is_outer)

        if len(rowwise_scale_inv_shape) > 1:
            rowwise_scale_inv_shape = (
                rowwise_scale_inv_shape[:-1] + (1,) + rowwise_scale_inv_shape[-1:]
            )
        if len(colwise_scale_inv_shape) > 1:
            colwise_scale_inv_shape = (
                colwise_scale_inv_shape[:-1] + (1,) + colwise_scale_inv_shape[-1:]
            )

        scale_inv_aval = jax.core.ShapedArray(shape=rowwise_scale_inv_shape, dtype=scale_dtype)

        colwise_out_aval = jax.core.ShapedArray(shape=(1,), dtype=out_dtype)
        colwise_scale_inv_aval = jax.core.ShapedArray(shape=(1,), dtype=scale_dtype)
        if is_2x:
            colwise_out_aval = jax.core.ShapedArray(shape=out_shape, dtype=out_dtype)
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
        scale_shapes,
        is_outer,
    ):
        """
        te_gated_act_lu_p lowering rules
        """
        del out_dtype, scale_dtype, scale_shapes, act_len, is_outer
        x_aval, scale_aval = ctx.avals_in
        assert x_aval.dtype in [jnp.float32, jnp.float16, jnp.bfloat16]
        assert scale_aval is None or scale_aval.dtype == jnp.float32

        out = ffi.ffi_lowering(ActLuPrimitive.name)(
            ctx, x, scale, act_enum=act_enum, scaling_mode=scaling_mode, is_2x=is_2x
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
        scale_shapes,
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
                scale_shapes=scale_shapes,
                is_outer=False,
            )
        )
        rowwise_scale_inv_shape, colwise_scale_inv_shape = ScalingMode(
            scaling_mode
        ).get_scale_shape_2x(out.shape[:-2] + (out.shape[-1],), is_padded=False)
        if scaling_mode == ScalingMode.NVTE_MXFP8_1D_SCALING.value:
            rowwise_scale_inv_shape = (
                rowwise_scale_inv_shape[:-1] + (1,) + rowwise_scale_inv_shape[-1:]
            )
            if is_2x:
                colwise_scale_inv_shape = (
                    colwise_scale_inv_shape[:-1] + (1,) + colwise_scale_inv_shape[-1:]
                )
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
        scale_shapes,
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
                scale_shapes=scale_shapes,
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
        scale_shapes,
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
            scale_shapes,
            act_len,
            is_outer,
        )  # Unused.
        x_spec = get_padded_spec(arg_infos[0])
        out_spec = (*x_spec[:-2], None, x_spec[-2])
        out_sharding = NamedSharding(mesh, PartitionSpec(*out_spec), desc="ActLuPrimitive.out")
        if is_2x:
            if scaling_mode == ScalingMode.NVTE_DELAYED_TENSOR_SCALING.value:
                colwise_out_spec = multidim_transpose(out_spec)
            else:
                colwise_out_spec = out_spec
        else:
            colwise_out_spec = (None,)
        colwise_out_sharding = NamedSharding(
            mesh, PartitionSpec(*colwise_out_spec), desc="ActLuPrimitive.colwise_out"
        )
        scale_inv_sharding = NamedSharding(
            mesh, PartitionSpec(*get_padded_spec(arg_infos[1])), desc="ActLuPrimitive.scale_inv"
        )
        amax_sharding = scale_inv_sharding.duplicate_with_new_description("ActLuPrimitive.amax")

        if scaling_mode == ScalingMode.NVTE_MXFP8_1D_SCALING.value:
            scale_inv_sharding = NamedSharding(
                mesh, PartitionSpec(*out_spec), desc="ActLuPrimitive.scale_inv"
            )
        colwise_scale_inv_sharding = scale_inv_sharding.duplicate_with_new_description(
            "ActLuPrimitive.colwise_scale_inv"
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
        scale_shapes,
        is_outer,
        mesh,
        arg_infos,
        result_infos,
    ):
        del result_infos, is_outer  # Unused.
        x_spec = get_padded_spec(arg_infos[0])
        out_spec = (*x_spec[:-1], x_spec[-1])
        if act_len == 2 and x_spec[-1] is None:
            # Ensure last axis is partitioned and not the gating axis
            out_spec = (*x_spec[:-2], None, x_spec[-2])
        out_sharding = NamedSharding(mesh, PartitionSpec(*out_spec), desc="ActLuPrimitive.out")
        if is_2x:
            if scaling_mode == ScalingMode.NVTE_DELAYED_TENSOR_SCALING.value:
                colwise_out_spec = multidim_transpose(out_spec)
            else:
                colwise_out_spec = out_spec
        else:
            colwise_out_spec = (None,)
        colwise_out_sharding = NamedSharding(
            mesh, PartitionSpec(*colwise_out_spec), desc="ActLuPrimitive.colwise_out"
        )
        scale_inv_sharding = NamedSharding(
            mesh, PartitionSpec(*get_padded_spec(arg_infos[1])), desc="ActLuPrimitive.scale_inv"
        )
        amax_sharding = scale_inv_sharding.duplicate_with_new_description("ActLuPrimitive.amax")

        if scaling_mode == ScalingMode.NVTE_MXFP8_1D_SCALING.value:
            scale_inv_sharding = NamedSharding(
                mesh, PartitionSpec(*out_spec), desc="ActLuPrimitive.scale_inv"
            )
        colwise_scale_inv_sharding = scale_inv_sharding.duplicate_with_new_description(
            "ActLuPrimitive.colwise_scale_inv"
        )
        arg_shardings = list(arg_i.sharding for arg_i in arg_infos)
        arg_shardings[0] = NamedSharding(mesh, PartitionSpec(*out_spec))
        arg_shardings = tuple(arg_shardings)
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
                    scale_shapes=scale_shapes,
                    is_outer=True,
                )
            )

            if scaling_mode == ScalingMode.NVTE_DELAYED_TENSOR_SCALING.value:
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


register_primitive(ActLuPrimitive)


class DActLuDBiasQuantizePrimitive(BasePrimitive):
    """
    DActLu DBias Cast Transpose Primitive
    """

    name = "te_dact_dbias_quantize_ffi"
    multiple_results = True
    # out_dtype, scaling_mode, is_2x, scale_dtype, scale_shapes, is_dbias, act_enum, act_len, is_outer
    impl_static_args = (3, 4, 5, 6, 7, 8, 9, 10, 11)
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
        scale_shapes,
        is_dbias,
        act_enum,
        act_len,
        is_outer,
    ):
        """
        te_dact_dbias_quantize_p abstract
        """
        del act_enum, scale_shapes
        dtype = dtypes.canonicalize_dtype(dz_aval.dtype)
        assert dtype in [jnp.float32, jnp.float16, jnp.bfloat16]
        assert x_aval.dtype == dtype
        assert scale_aval.dtype == jnp.float32
        ir_hidden_size = dz_aval.shape[-1]
        gi_hidden_size = x_aval.shape[-1]
        assert act_len * ir_hidden_size == gi_hidden_size
        out_shape = x_aval.shape
        out_aval = x_aval.update(shape=out_shape, dtype=out_dtype)
        updated_amax_aval = jax.core.ShapedArray(shape=(1,), dtype=jnp.float32)

        rowwise_scale_inv_shape, colwise_scale_inv_shape = ScalingMode(
            scaling_mode
        ).get_scale_shape_2x(x_aval.shape, is_padded=not is_outer)

        scale_inv_aval = jax.core.ShapedArray(shape=rowwise_scale_inv_shape, dtype=scale_dtype)

        colwise_out_aval = jax.core.ShapedArray(shape=(1,), dtype=jnp.float32)
        colwise_scale_inv_aval = jax.core.ShapedArray(shape=(1,), dtype=scale_dtype)

        dbias_aval = jax.core.ShapedArray(shape=(1,), dtype=jnp.float32)
        wkspace_aval = jax.core.ShapedArray(shape=(1,), dtype=jnp.float32)
        if is_2x:
            # Don't transpose output for MXFP8
            if scaling_mode == ScalingMode.NVTE_MXFP8_1D_SCALING.value:
                t_shape = out_shape
            else:
                t_shape = multidim_transpose(out_shape)
            colwise_out_aval = x_aval.update(shape=t_shape, dtype=out_dtype)
            colwise_scale_inv_aval = jax.core.ShapedArray(
                shape=colwise_scale_inv_shape, dtype=scale_dtype
            )

        if is_dbias:
            dbias_shape = gi_hidden_size
            dbias_aval = x_aval.update(shape=dbias_shape, dtype=dtype)
            (wkspace_info,) = transformer_engine_jax.get_dact_dbias_quantize_workspace_sizes(
                x_aval.size // gi_hidden_size,
                gi_hidden_size,
                jax_dtype_to_te_dtype(x_aval.dtype),
                jax_dtype_to_te_dtype(out_dtype),
                scaling_mode,
                is_2x,
            )
            wkspace_aval = x_aval.update(
                shape=wkspace_info[0], dtype=te_dtype_to_jax_dtype(wkspace_info[1])
            )

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
            DActLuDBiasQuantizePrimitive.abstract(*args, **kwargs)
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
        scale_shapes,
        is_dbias,
        act_enum,
        act_len,
        is_outer,
    ):
        """
        te_dact_dbias_quantize_p lowering rules
        """
        del out_dtype, scale_dtype, scale_shapes, act_len, is_outer
        dz_aval, x_aval, scale_aval = ctx.avals_in
        assert dz_aval.dtype in [jnp.float32, jnp.float16, jnp.bfloat16]
        assert x_aval.dtype == dz_aval.dtype
        assert scale_aval.dtype == jnp.float32
        return ffi.ffi_lowering(DActLuDBiasQuantizePrimitive.name)(
            ctx,
            dz,
            x,
            scale,
            scaling_mode=scaling_mode,
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
        scale_shapes,
        is_dbias,
        act_enum,
        act_len,
        is_outer,
    ):
        """
        te_dact_dbias_quantize_p impl
        """
        del is_outer
        assert DActLuDBiasQuantizePrimitive.inner_primitive is not None
        (out, colwise_out, scale_inv, colwise_scale_inv, updated_amax, dbias, _) = (
            DActLuDBiasQuantizePrimitive.inner_primitive.bind(
                dz,
                x,
                scale,
                out_dtype=out_dtype,
                scaling_mode=scaling_mode,
                is_2x=is_2x,
                scale_dtype=scale_dtype,
                scale_shapes=scale_shapes,
                is_dbias=is_dbias,
                act_enum=act_enum,
                act_len=act_len,
                is_outer=False,
            )
        )
        rowwise_scale_inv_shape, colwise_scale_inv_shape = ScalingMode(
            scaling_mode
        ).get_scale_shape_2x(x.shape, is_padded=False)
        if scaling_mode == ScalingMode.NVTE_MXFP8_1D_SCALING.value:
            scale_inv = jax.lax.slice(
                scale_inv, [0] * len(rowwise_scale_inv_shape), rowwise_scale_inv_shape
            )
            if is_2x:
                colwise_scale_inv = jax.lax.slice(
                    colwise_scale_inv, [0] * len(colwise_scale_inv_shape), colwise_scale_inv_shape
                )
        return (
            out,
            colwise_out,
            scale_inv,
            colwise_scale_inv,
            updated_amax,
            dbias,
        )  # Exclude wkspace

    @staticmethod
    def batcher(
        batched_args,
        batch_dims,
        *,
        out_dtype,
        scaling_mode,
        is_2x,
        scale_dtype,
        scale_shapes,
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
        assert DActLuDBiasQuantizePrimitive.outer_primitive is not None
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
            DActLuDBiasQuantizePrimitive.outer_primitive.bind(
                dz,
                x,
                scale,
                out_dtype=out_dtype,
                scaling_mode=scaling_mode,
                is_2x=is_2x,
                scale_dtype=scale_dtype,
                scale_shapes=scale_shapes,
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
        scale_shapes,
        is_dbias,
        act_enum,
        act_len,
        is_outer,
        mesh,
        arg_infos,
        result_infos,
    ):
        del out_dtype, result_infos, act_enum
        del scale_dtype, scale_shapes, is_dbias, act_len, is_outer
        x_spec = get_padded_spec(arg_infos[1])

        out_sharding = NamedSharding(
            mesh, PartitionSpec(*x_spec), desc="DActLuDBiasQuantizePrimitive.out"
        )
        if is_2x:
            if scaling_mode == ScalingMode.NVTE_DELAYED_TENSOR_SCALING.value:
                colwise_x_spec = multidim_transpose(x_spec)
            else:
                colwise_x_spec = x_spec
        else:
            colwise_x_spec = (None,)
        colwise_out_sharding = NamedSharding(
            mesh, PartitionSpec(*colwise_x_spec), desc="DActLuDBiasQuantizePrimitive.colwise_out"
        )

        dbias_shaprding = NamedSharding(
            mesh,
            PartitionSpec(x_spec[-1]),
            desc="DActLuDBiasQuantizePrimitive.dbias",
        )
        scale_inv_sharding = NamedSharding(
            mesh, PartitionSpec(None), desc="DActLuDBiasQuantizePrimitive.scale_inv"
        )
        amax_sharding = NamedSharding(
            mesh, PartitionSpec(None), desc="DActLuDBiasQuantizePrimitive.amax"
        )
        if scaling_mode == ScalingMode.NVTE_MXFP8_1D_SCALING.value:
            scale_inv_sharding = NamedSharding(
                mesh, PartitionSpec(*x_spec), desc="DActLuDBiasQuantizePrimitive.scale_inv"
            )
        colwise_scale_inv_sharding = scale_inv_sharding.duplicate_with_new_description(
            "DActLuDBiasQuantizePrimitive.colwise_scale_inv"
        )
        return (
            out_sharding,
            colwise_out_sharding,
            scale_inv_sharding,
            colwise_scale_inv_sharding,
            amax_sharding,
            dbias_shaprding,
        )

    @staticmethod
    def partition(
        out_dtype,
        scaling_mode,
        is_2x,
        scale_dtype,
        scale_shapes,
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
        out_sharding = NamedSharding(mesh, PartitionSpec(*x_spec), desc="out")
        if is_2x:
            if scaling_mode == ScalingMode.NVTE_DELAYED_TENSOR_SCALING.value:
                colwise_x_spec = multidim_transpose(x_spec)
            else:
                colwise_x_spec = x_spec
        else:
            colwise_x_spec = (None,)
        colwise_out_sharding = NamedSharding(
            mesh, PartitionSpec(*colwise_x_spec), desc="DActLuDBiasQuantizePrimitive.colwise_out"
        )

        dbias_shaprding = NamedSharding(
            mesh,
            PartitionSpec(x_spec[-1]),
            desc="DActLuDBiasQuantizePrimitive.dbias",
        )
        scale_inv_sharding = NamedSharding(
            mesh, PartitionSpec(None), desc="DActLuDBiasQuantizePrimitive.scale_inv"
        )
        amax_sharding = NamedSharding(
            mesh, PartitionSpec(None), desc="DActLuDBiasQuantizePrimitive.amax"
        )
        if scaling_mode == ScalingMode.NVTE_MXFP8_1D_SCALING.value:
            scale_inv_sharding = NamedSharding(
                mesh, PartitionSpec(*x_spec), desc="DActLuDBiasQuantizePrimitive.scale_inv"
            )
        colwise_scale_inv_sharding = scale_inv_sharding.duplicate_with_new_description(
            "DActLuDBiasQuantizePrimitive.colwise_scale_inv"
        )

        arg_shardings = tuple(arg_i.sharding for arg_i in arg_infos)
        arg_shardings = (
            arg_shardings[1],
            arg_shardings[1],
            *arg_shardings[2:],
        )  # dz and x are the same
        out_shardings = (
            out_sharding,
            colwise_out_sharding,
            scale_inv_sharding,
            colwise_scale_inv_sharding,
            amax_sharding,
            dbias_shaprding,
        )

        def sharded_impl(dz, x, scale):
            (out, colwise_out, scale_inv, colwise_scale_inv, local_amax, local_dbias) = (
                DActLuDBiasQuantizePrimitive.impl(
                    dz,
                    x,
                    scale,
                    out_dtype=out_dtype,
                    scaling_mode=scaling_mode,
                    is_2x=is_2x,
                    scale_dtype=scale_dtype,
                    scale_shapes=scale_shapes,
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

            if scaling_mode == ScalingMode.NVTE_DELAYED_TENSOR_SCALING.value:
                global_updated_amax = all_reduce_max_along_all_axes_except_PP(local_amax, mesh)
            else:
                global_updated_amax = local_amax

            return out, colwise_out, scale_inv, colwise_scale_inv, global_updated_amax, global_dbias

        return mesh, sharded_impl, out_shardings, arg_shardings


register_primitive(DActLuDBiasQuantizePrimitive)


def _jax_act_lu(inputs, activation_type, quantizer=None) -> Union[jnp.ndarray, ScaledTensor]:
    """
    JAX native activation implementation
    """
    x = jnp.split(inputs, len(activation_type), axis=-1)
    acts = []
    for idx, act_fn in enumerate(activation_type):
        x_i = _convert_to_activation_function(act_fn)(x[idx])
        acts.append(x_i)
    x = reduce(operator.mul, acts)
    if quantizer:
        return quantizer.quantize(x)
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
    _, vjp_func = jax.vjp(
        partial(_jax_act_lu, activation_type=activation_type), x.astype(jnp.float32)
    )
    (dx,) = vjp_func(dz.astype(jnp.float32))

    dbias = None
    if is_dbias:
        dbias = _jax_dbias(dx).astype(x.dtype)

    if quantizer is not None:
        dx = quantizer.quantize(dx, dq_dtype=x.dtype)
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
        activation_type: Type of activation function to apply.
        quantizer: Optional quantizer for FP8 quantization of the output.

    Returns:
        If quantizer is None:
            The activated input tensor with the same dtype as input.
        If quantizer is provided:
            A ScaledTensor containing the quantized activated input.
    """
    act_type_id = ActivationEnum[activation_type].value

    if not ActLuPrimitive.enabled():
        return _jax_act_lu(x, activation_type, quantizer)

    # TE/common does not support colwise-only quantization yet
    if quantizer is not None and quantizer.q_axis == QuantizeLayout.COLWISE:
        return _jax_act_lu(x, activation_type, quantizer)

    # TE/common does not support 2x quantization for DelayedScaling yet
    war_output = try_apply_delayed_scaling_2x_war(
        f=act_lu, x=x, activation_type=activation_type, quantizer=quantizer
    )
    if war_output is not None:
        return war_output

    scale = jnp.empty((1,), jnp.float32)
    output_shape = (*x.shape[:-1], x.shape[-1] // len(activation_type))

    if quantizer is None:
        x = x.reshape((-1, len(activation_type), x.shape[-1] // len(activation_type)))
        out, _, _, _, _ = ActLuPrimitive.outer_primitive.bind(
            x,
            scale,
            out_dtype=x.dtype,
            act_enum=act_type_id,
            act_len=len(activation_type),
            scaling_mode=ScalingMode.NVTE_DELAYED_TENSOR_SCALING.value,
            is_2x=False,
            scale_dtype=jnp.float32,
            scale_shapes=((), ()),
            is_outer=True,
        )
        out = out.reshape(output_shape)
        return out

    if isinstance(quantizer, DelayedScaleQuantizer):
        scale = quantizer.scale

    x = x.reshape((*x.shape[:-1], len(activation_type), x.shape[-1] // len(activation_type)))
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
        act_len=len(activation_type),
        scaling_mode=quantizer.scaling_mode.value,
        is_2x=quantizer.is_2x2x(),
        scale_dtype=quantizer.get_scale_dtype(),
        scale_shapes=quantizer.get_scale_shapes(output_shape),
        is_outer=True,
    )

    rowwise_casted_output = rowwise_casted_output.reshape(output_shape)
    if len(rowwise_scale_inv.shape) > 1:
        rowwise_scale_inv = jnp.squeeze(rowwise_scale_inv, axis=-2)  # Remove act axis
    if quantizer.q_axis in (QuantizeLayout.COLWISE, QuantizeLayout.ROWWISE_COLWISE):
        colwise_output_shape = output_shape
        if quantizer.scaling_mode == ScalingMode.NVTE_DELAYED_TENSOR_SCALING:
            colwise_output_shape = multidim_transpose(output_shape)
        colwise_casted_output = colwise_casted_output.reshape(colwise_output_shape)
        if len(colwise_scale_inv.shape) > 1:
            colwise_scale_inv = jnp.squeeze(colwise_scale_inv, axis=-2)  # Remove act axis

    quantizer.update(updated_amax)

    return ScaledTensorFactory.create(
        data=rowwise_casted_output,
        scale_inv=rowwise_scale_inv,
        colwise_data=colwise_casted_output,
        colwise_scale_inv=colwise_scale_inv,
        scaling_mode=quantizer.scaling_mode,
        dq_dtype=x.dtype,
        q_axis=quantizer.q_axis,
        layout=quantizer.get_data_layout(),
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
            Shape: (..., ACT_DIM * K) where ACT_DIM is 1 for non-gated activations and 2 for gated activations
        activation_type: Type of activation function used in the forward pass. Defaults to ("gelu",).
        is_dbias: If True, compute bias gradient. Defaults to True.
        quantizer: Optional quantizer for FP8 quantization of the output.

    Returns:
        Tuple[ScaledTensor, jnp.ndarray]: A tuple containing:
        - The gradient of the activation with respect to the input.
        - The gradient of the activation with respect to the bias.
    """

    if not DActLuDBiasQuantizePrimitive.enabled():
        return _jax_quantize_dact_dbias(dz, x, activation_type, is_dbias, quantizer)

    # TE/common does not support colwise-only quantization yet
    if quantizer is not None and quantizer.q_axis == QuantizeLayout.COLWISE:
        return _jax_quantize_dact_dbias(dz, x, activation_type, is_dbias, quantizer)

    # TE/common does not support 1x dact_dbias_quantize on arch < 100 yet
    if should_apply_1x_fused_dbias_war_for_arch_l_100(is_dbias=is_dbias, quantizer=quantizer):
        out, _ = quantize_dact_dbias(
            dz=dz, x=x, activation_type=activation_type, is_dbias=False, quantizer=None
        )
        return quantize_dbias(out, is_dbias=True, quantizer=quantizer)

    is_gated = len(activation_type) == 2
    # TE/common does not support DelayedScaling2x for gated-act yet
    if is_gated:
        war_output = try_apply_delayed_scaling_2x_war(
            f=quantize_dact_dbias,
            dz=dz,
            x=x,
            activation_type=activation_type,
            is_dbias=is_dbias,
            quantizer=quantizer,
        )
        if war_output is not None:
            return war_output

    scale = jnp.empty((), jnp.float32)

    act_type_id = ActivationEnum[activation_type]

    if quantizer is None:
        output, _, _, _, _, _ = DActLuDBiasQuantizePrimitive.outer_primitive.bind(
            dz,
            x,
            scale,
            # outputs float32 for dbias accumulation
            out_dtype=(jnp.float32 if is_dbias else x.dtype),
            # default value for no scaling, TE/common ignore this value when scale is unset
            scaling_mode=ScalingMode.NVTE_DELAYED_TENSOR_SCALING.value,
            is_2x=False,  # unused
            scale_dtype=jnp.float32,  # unused
            scale_shapes=((), ()),  # unused
            is_dbias=False,
            act_enum=act_type_id,
            act_len=len(activation_type),
            is_outer=True,
        )
        dbias = None
        if is_dbias:
            dbias = _jax_dbias(output).astype(x.dtype)
        return output.astype(x.dtype), dbias

    if isinstance(quantizer, DelayedScaleQuantizer):
        scale = quantizer.scale

    # TE/common dact_dbias_quantize does not support gated act yet
    if is_dbias and is_gated:
        dgated = dact_lu(
            dz.astype(jnp.float32), x.astype(jnp.float32), activation_type=activation_type
        )
        # TODO(Jeremy): Debug - TE's quantize_dbias produced nans in this case for distributed layernorm_mlp tests
        if quantizer.scaling_mode == ScalingMode.NVTE_MXFP8_1D_SCALING:
            out, dbias = _jax_quantize_dbias(dgated, quantizer=quantizer, dq_dtype=x.dtype)
        else:
            out, dbias = quantize_dbias(
                dgated,
                quantizer=quantizer,
                is_dbias=True,
                dq_dtype=x.dtype,
            )
        return out, dbias

    out_shape = x.shape

    (
        rowwise_casted_output,
        colwise_casted_output,
        rowwise_scale_inv,
        colwise_scale_inv,
        updated_amax,
        dbias,
    ) = DActLuDBiasQuantizePrimitive.outer_primitive.bind(
        dz,
        x,
        scale,
        out_dtype=quantizer.q_dtype,
        scaling_mode=quantizer.scaling_mode.value,
        is_2x=quantizer.is_2x2x(),
        scale_dtype=quantizer.get_scale_dtype(),
        scale_shapes=quantizer.get_scale_shapes(out_shape),
        is_dbias=is_dbias,
        act_enum=act_type_id,
        act_len=len(activation_type),
        is_outer=True,
    )

    # For DelayedScaling transpose, the scale buffer is shared for both rowwise and colwise
    if quantizer.scaling_mode == ScalingMode.NVTE_DELAYED_TENSOR_SCALING and quantizer.is_2x2x():
        colwise_scale_inv = rowwise_scale_inv

    quantizer.update(updated_amax)

    out = ScaledTensorFactory.create(
        data=rowwise_casted_output,
        scale_inv=rowwise_scale_inv,
        colwise_data=colwise_casted_output,
        colwise_scale_inv=colwise_scale_inv,
        scaling_mode=quantizer.scaling_mode,
        dq_dtype=x.dtype,
        q_axis=quantizer.q_axis,
        layout=quantizer.get_data_layout(),
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
