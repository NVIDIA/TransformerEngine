# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""JAX/TE custom ops for quantization"""
import operator
from functools import reduce
from typing import Tuple, Optional, Union
import math


import jax
import jax.numpy as jnp
from jax import dtypes, ffi
from jax.experimental.custom_partitioning import SdyShardingRule, BATCHING
from jax.sharding import PartitionSpec

import transformer_engine_jax

from .amax import AmaxScope, calculate_amax, calculate_post_rht_amax
from .base import BasePrimitive, register_primitive
from .misc import (
    get_padded_spec,
    check_valid_batch_dims,
    te_dtype_to_jax_dtype,
    jax_dtype_to_te_dtype,
    multidim_transpose,
    should_apply_1x_fused_dbias_war_for_arch_l_100,
    get_min_device_compute_capability,
    NamedSharding,
)
from ..sharding import (
    all_reduce_max_along_all_axes_except_PP,
    all_reduce_sum_along_dp_fsdp,
    get_num_devices_in_mesh,
)
from ..quantize import (
    ScaledTensor2x,
    ScaledTensor,
    ScaledTensorFactory,
    GroupedScaledTensor1x,
    Quantizer,
    GroupedQuantizer,
    QuantizeLayout,
    ScalingMode,
    compute_scale_from_amax,
    NoScaleTensor,
    get_rht_matrix,
)


__all__ = ["quantize", "quantize_dbias", "grouped_quantize", "grouped_dbias"]


class BaseDBiasQuantizePrimitive(BasePrimitive):
    """
    Cast Primitive wrapping nvte_quantize and nvte_quantize_dbias
    """

    name = "te_dbias_quantize_ffi"
    multiple_results = True
    impl_static_args = (
        6,  # out_dtype
        7,  # scaling_mode
        8,  # q_layout
        9,  # flatten_axis
        10,  # scale_dtype
        11,  # is_dbias
        12,  # is_outer
        13,  # stochastic_rounding
        14,  # use_rht
    )
    inner_primitive = None
    outer_primitive = None

    @staticmethod
    def abstract(
        x_aval,
        scale_aval,
        amax_aval,
        sr_rng_state_aval,
        post_rht_amax_aval,
        rht_matrix_aval,
        *,
        out_dtype,
        scaling_mode,
        q_layout,
        flatten_axis,
        scale_dtype,
        is_dbias,
        is_outer,
        stochastic_rounding,
        use_rht,
    ):
        """
        te_dbias_quantize_p abstract
        """
        dtype = dtypes.canonicalize_dtype(x_aval.dtype)
        assert dtype in [jnp.float32, jnp.float16, jnp.bfloat16]
        out_shape = x_aval.shape
        assert scale_aval is None or scale_aval.dtype == jnp.float32
        if stochastic_rounding:
            assert ScalingMode(
                scaling_mode
            ).is_nvfp4_scaling, "stochastic_rounding can only be used with NVFP4 scaling modes"
            # JAX doesn't support 64-bit by default so use 4x uint32 instead of 2x int64
            assert sr_rng_state_aval is not None and sr_rng_state_aval.dtype == jnp.uint32, (
                "sr_rng_state must be a uint32 array when stochastic_rounding is True but"
                f" received {sr_rng_state_aval}"
            )
            if is_outer and get_num_devices_in_mesh() > 1:
                assert (
                    sr_rng_state_aval.shape[0] == get_num_devices_in_mesh()
                    and sr_rng_state_aval.shape[1] == 4
                ), (
                    "sr_rng_state must be of shape (num_devices, 4) when stochastic_rounding is"
                    f" True and is_outer is True but received {sr_rng_state_aval.shape}"
                )
            else:
                # We cannot assert the shape is exactly (4,) here because if the quantized data is not perfectly sharded across all devices then we will have extra rng state here. For example, this could occur when the weights are not sharded when using data parallelism. However, this is okay because the extra rng state will simply not be used and each device still has a unique rng state.
                assert sr_rng_state_aval.size >= 4, (
                    "Sharded sr_rng_state must have at least 4 elements per device when"
                    f" stochastic_rounding is True but received {sr_rng_state_aval.shape}"
                )

        if QuantizeLayout(q_layout).has_rowwise:
            rowwise_out_shape = out_shape
        else:
            rowwise_out_shape = (1,)
        rowwise_out_aval = jax.core.ShapedArray(shape=rowwise_out_shape, dtype=out_dtype)

        assert out_dtype in ScalingMode(scaling_mode).get_compatible_q_dtypes(), (
            f"out_dtype {out_dtype} not compatible with scaling_mode {scaling_mode}. out_dtype must"
            f" be one of {ScalingMode(scaling_mode).get_compatible_q_dtypes()}"
        )

        updated_amax_aval = amax_aval

        if use_rht:
            assert (
                x_aval.dtype == jnp.bfloat16
            ), "x must be of dtype bfloat16 to be eligible for RHT cast fusion."

            if flatten_axis < 0:
                flatten_axis += len(x_aval.shape)
            rows = reduce(operator.mul, x_aval.shape[:flatten_axis], 1)
            cols = reduce(operator.mul, x_aval.shape[flatten_axis:], 1)
            assert rows % 64 == 0 and cols % 128 == 0, (
                "Rows must be multiple of 64 and cols multiple of 128 when use_rht is True to be"
                f" eligible for RHT cast fusion. Received rows {rows} and cols {cols} of 2D shape"
                f" from original shape of {x_aval.shape} with flatten_axis {flatten_axis}."
            )

            assert (
                rht_matrix_aval is not None
                and rht_matrix_aval.dtype == jnp.bfloat16
                and rht_matrix_aval.shape == (16, 16)
            ), "rht_matrix must be of shape (16, 16) and dtype bfloat16"
            assert (
                post_rht_amax_aval is not None
                and post_rht_amax_aval.dtype == jnp.float32
                and post_rht_amax_aval.size == 1
            ), "post_rht_amax must be of dtype float32"

        rowwise_scale_inv_shape, colwise_scale_inv_shape = ScalingMode(
            scaling_mode
        ).get_scale_shape_2x(
            x_aval.shape,
            is_padded=not is_outer,
            flatten_axis=flatten_axis,
            broadcast_2d_scale_shape_to_1d=True,
        )

        if QuantizeLayout(q_layout).has_colwise:
            if ScalingMode(scaling_mode).is_colwise_transposed:
                colwise_out_shape = multidim_transpose(out_shape, transpose_axis=flatten_axis)
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
            dbias_shape = x_aval.shape[flatten_axis:]
            gi_hidden_size = reduce(operator.mul, x_aval.shape[flatten_axis:], 1)
            (wkspace_info,) = transformer_engine_jax.get_dbias_quantize_workspace_sizes(
                x_aval.size // gi_hidden_size,
                gi_hidden_size,
                jax_dtype_to_te_dtype(x_aval.dtype),
                jax_dtype_to_te_dtype(out_dtype),
                jax_dtype_to_te_dtype(scale_dtype),
                scaling_mode,
                q_layout.value,
            )
            wkspace_shape = wkspace_info[0]
            wkspace_dtype = te_dtype_to_jax_dtype(wkspace_info[1])
        else:
            dbias_shape = (1,)
            wkspace_shape = (1,)
            wkspace_dtype = jnp.float32
        dbias_aval = jax.core.ShapedArray(shape=dbias_shape, dtype=dtype)
        wkspace_aval = jax.core.ShapedArray(shape=wkspace_shape, dtype=wkspace_dtype)

        return (
            rowwise_out_aval,
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
        te_dbias_quantize_p outer primitive abstract
        """
        (
            out,
            colwise_out,
            scale_inv,
            colwise_scale_inv,
            updated_amax,
            dbias,
            _,
        ) = BaseDBiasQuantizePrimitive.abstract(*args, **kwargs)
        return out, colwise_out, scale_inv, colwise_scale_inv, updated_amax, dbias

    @staticmethod
    def lowering(
        ctx,
        x,
        scale,
        amax,
        sr_rng_state,
        post_rht_amax,
        rht_matrix,
        *,
        out_dtype,
        scaling_mode,
        q_layout,
        flatten_axis,
        scale_dtype,
        is_dbias,
        is_outer,
        stochastic_rounding,
        use_rht,
    ):
        """
        te_dbias_quantize_p lowering rules
        """
        del out_dtype, scale_dtype, is_outer
        x_aval, scale_aval, amax_aval, _, _, _ = ctx.avals_in
        assert x_aval.dtype in [jnp.float32, jnp.float16, jnp.bfloat16]
        assert scale_aval.dtype == amax_aval.dtype == jnp.float32
        return ffi.ffi_lowering(
            BaseDBiasQuantizePrimitive.name,
            operand_output_aliases={2: 4},  # donate amax buffer to updated_amax
        )(
            ctx,
            x,
            scale,
            amax,
            sr_rng_state,
            post_rht_amax,
            rht_matrix,
            scaling_mode=scaling_mode.value,
            q_layout=q_layout.value.value,
            flatten_axis=flatten_axis,
            is_dbias=is_dbias,
            stochastic_rounding=stochastic_rounding,
            use_rht=use_rht,
        )

    @staticmethod
    def impl(
        x,
        scale,
        amax,
        sr_rng_state,
        post_rht_amax,
        rht_matrix,
        out_dtype,
        scaling_mode,
        q_layout,
        flatten_axis,
        scale_dtype,
        is_dbias,
        is_outer,
        stochastic_rounding,
        use_rht,
    ):
        """
        te_dbias_quantize_p implementation
        """
        del is_outer
        assert BaseDBiasQuantizePrimitive.inner_primitive is not None
        (
            out,
            colwise_out,
            scale_inv,
            colwise_scale_inv,
            updated_amax,
            dbias,
            _,
        ) = BaseDBiasQuantizePrimitive.inner_primitive.bind(
            x,
            scale,
            amax,
            sr_rng_state,
            post_rht_amax,
            rht_matrix,
            out_dtype=out_dtype,
            scaling_mode=scaling_mode,
            q_layout=q_layout,
            flatten_axis=flatten_axis,
            scale_dtype=scale_dtype,
            is_dbias=is_dbias,
            is_outer=False,
            stochastic_rounding=stochastic_rounding,
            use_rht=use_rht,
        )
        rowwise_scale_inv_shape, colwise_scale_inv_shape = ScalingMode(
            scaling_mode
        ).get_scale_shape_2x(
            x.shape, is_padded=False, flatten_axis=flatten_axis, broadcast_2d_scale_shape_to_1d=True
        )
        scale_inv = jax.lax.slice(
            scale_inv, [0] * len(rowwise_scale_inv_shape), rowwise_scale_inv_shape
        )
        if q_layout.has_colwise:
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
        q_layout,
        flatten_axis,
        scale_dtype,
        is_dbias,
        is_outer,
        stochastic_rounding,
        use_rht,
    ):
        """
        to describe batch rules for vmap
        """
        del is_outer
        check_valid_batch_dims(batch_dims)
        assert BaseDBiasQuantizePrimitive.outer_primitive is not None
        x, scale, amax, sr_rng_state, post_rht_amax, rht_matrix = batched_args
        x_bdim, scale_bdim, amax_bdim, _, _, _ = batch_dims

        out_bdims = x_bdim, x_bdim, scale_bdim, scale_bdim, amax_bdim, x_bdim
        return (
            BaseDBiasQuantizePrimitive.outer_primitive.bind(
                x,
                scale,
                amax,
                sr_rng_state,
                post_rht_amax,
                rht_matrix,
                out_dtype=out_dtype,
                scaling_mode=scaling_mode,
                q_layout=q_layout,
                flatten_axis=flatten_axis,
                scale_dtype=scale_dtype,
                is_dbias=is_dbias,
                stochastic_rounding=stochastic_rounding,
                use_rht=use_rht,
            ),
            out_bdims,
        )

    @staticmethod
    def infer_sharding_from_operands(
        out_dtype,
        scaling_mode,
        q_layout,
        flatten_axis,
        scale_dtype,
        is_dbias,
        is_outer,
        stochastic_rounding,
        use_rht,
        mesh,
        arg_infos,
        result_infos,
    ):
        del (
            out_dtype,
            result_infos,
            scale_dtype,
            is_outer,
            stochastic_rounding,
            use_rht,
        )  # Unused.

        x_spec = get_padded_spec(arg_infos[0])
        amax_spec = get_padded_spec(arg_infos[2])
        out_sharding = NamedSharding(
            mesh,
            PartitionSpec(*x_spec),
            desc="BaseDBiasQuantizePrimitive.out_sharding",
        )
        if q_layout.has_colwise:
            if ScalingMode(scaling_mode).is_colwise_transposed:
                colwise_out_spec = multidim_transpose(x_spec, transpose_axis=flatten_axis)
            else:
                colwise_out_spec = x_spec
        else:
            colwise_out_spec = (None,)
        colwise_out_sharding = NamedSharding(
            mesh,
            PartitionSpec(*colwise_out_spec),
            desc="BaseDBiasQuantizePrimitive.colwise_out_sharding",
        )

        dbias_spec = x_spec[flatten_axis:] if is_dbias else (None,)
        dbias_sharding = NamedSharding(
            mesh,
            PartitionSpec(*dbias_spec),
            desc="BaseDBiasQuantizePrimitive.dbias_sharding",
        )

        scale_inv_spec = colwise_scale_inv_spec = (None,)
        if ScalingMode(scaling_mode).is_block_scaling:
            scale_inv_spec = x_spec

        if q_layout.has_colwise:
            if (
                ScalingMode(scaling_mode).is_block_scaling
                and ScalingMode(scaling_mode).is_colwise_transposed
            ):
                colwise_scale_inv_spec = multidim_transpose(
                    scale_inv_spec, transpose_axis=flatten_axis
                )
            else:
                colwise_scale_inv_spec = scale_inv_spec

        scale_inv_sharding = NamedSharding(
            mesh, PartitionSpec(*scale_inv_spec), desc="BaseDBiasQuantizePrimitive.scale_inv"
        )
        colwise_scale_inv_sharding = NamedSharding(
            mesh,
            PartitionSpec(*colwise_scale_inv_spec),
            desc="BaseDBiasQuantizePrimitive.colwise_scale_inv",
        )
        amax_sharding = NamedSharding(
            mesh, PartitionSpec(*amax_spec), desc="BaseDBiasQuantizePrimitive.amax"
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
        q_layout,
        flatten_axis,
        scale_dtype,
        is_dbias,
        is_outer,
        stochastic_rounding,
        use_rht,
        mesh,
        arg_infos,
        result_infos,
    ):
        del result_infos, is_outer  # Unused.

        x_spec = get_padded_spec(arg_infos[0])
        amax_spec = get_padded_spec(arg_infos[2])
        out_sharding = NamedSharding(
            mesh,
            PartitionSpec(*x_spec),
            desc="BaseDBiasQuantizePrimitive.out_sharding",
        )

        if q_layout.has_colwise:
            if ScalingMode(scaling_mode).is_colwise_transposed:
                colwise_out_spec = multidim_transpose(x_spec, transpose_axis=flatten_axis)
            else:
                colwise_out_spec = x_spec
        else:
            colwise_out_spec = (None,)
        colwise_out_sharding = NamedSharding(
            mesh,
            PartitionSpec(*colwise_out_spec),
            desc="BaseDBiasQuantizePrimitive.colwise_out_sharding",
        )

        dbias_spec = x_spec[flatten_axis:] if is_dbias else (None,)
        dbias_sharding = NamedSharding(
            mesh,
            PartitionSpec(*dbias_spec),
            desc="BaseDBiasQuantizePrimitive.dbias_sharding",
        )

        scale_inv_spec = colwise_scale_inv_spec = (None,)
        if ScalingMode(scaling_mode).is_block_scaling:
            scale_inv_spec = x_spec

        if q_layout.has_colwise:
            if (
                ScalingMode(scaling_mode).is_block_scaling
                and ScalingMode(scaling_mode).is_colwise_transposed
            ):
                colwise_scale_inv_spec = multidim_transpose(
                    scale_inv_spec, transpose_axis=flatten_axis
                )
            else:
                colwise_scale_inv_spec = scale_inv_spec

        scale_inv_sharding = NamedSharding(
            mesh, PartitionSpec(*scale_inv_spec), desc="BaseDBiasQuantizePrimitive.scale_inv"
        )
        amax_sharding = NamedSharding(
            mesh, PartitionSpec(*amax_spec), desc="BaseDBiasQuantizePrimitive.amax"
        )
        colwise_scale_inv_sharding = NamedSharding(
            mesh,
            PartitionSpec(*colwise_scale_inv_spec),
            desc="BaseDBiasQuantizePrimitive.colwise_scale_inv",
        )

        arg_shardings = list(arg_i.sharding for arg_i in arg_infos)
        arg_shardings[3] = NamedSharding(
            mesh,
            PartitionSpec(tuple(x for x in x_spec if x is not None), None),
            desc="BaseDBiasQuantizePrimitive.sr_rng_state",
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

        def sharded_impl(x, scale, amax, sr_rng_state, post_rht_amax, rht_matrix):
            if sr_rng_state.size > 4:
                # See comment in abstract method for explanation of why we cannot assert exact shape
                sr_rng_state = sr_rng_state.flatten()[:4]
            (
                local_x,
                local_colwise_x,
                local_scale_inv,
                local_colwise_scale_inv,
                local_amax,
                local_dbias,
            ) = BaseDBiasQuantizePrimitive.impl(
                x,
                scale,
                amax,
                sr_rng_state,
                post_rht_amax,
                rht_matrix,
                out_dtype=out_dtype,
                scaling_mode=scaling_mode,
                q_layout=q_layout,
                flatten_axis=flatten_axis,
                scale_dtype=scale_dtype,
                is_dbias=is_dbias,
                is_outer=True,
                stochastic_rounding=stochastic_rounding,
                use_rht=use_rht,
            )

            if scaling_mode == ScalingMode.DELAYED_TENSOR_SCALING.value:
                global_updated_amax = all_reduce_max_along_all_axes_except_PP(local_amax, mesh)
            else:
                global_updated_amax = local_amax

            if is_dbias:
                global_dbias = all_reduce_sum_along_dp_fsdp(local_dbias, mesh)
            else:
                global_dbias = local_dbias

            return (
                local_x,
                local_colwise_x,
                local_scale_inv,
                local_colwise_scale_inv,
                global_updated_amax,
                global_dbias,
            )

        return mesh, sharded_impl, out_shardings, arg_shardings

    @staticmethod
    def shardy_sharding_rule(
        out_dtype,
        scaling_mode,
        q_layout,
        flatten_axis,
        scale_dtype,
        is_dbias,
        is_outer,
        stochastic_rounding,
        use_rht,
        mesh,
        value_types,
        result_types,
    ):
        del (
            out_dtype,
            scale_dtype,
            is_outer,
            stochastic_rounding,
            use_rht,
            mesh,
            result_types,
        )

        prefix = "DBiasQuantize"
        scale_rules = ScalingMode(scaling_mode).get_shardy_sharding_rules(
            value_types[0].shape,
            unique_var=prefix,
            flatten_axis=flatten_axis,
            q_layout=q_layout,
            broadcast_2d_scale_shape_to_1d=True,
        )

        input_spec = scale_rules.input_spec
        dbias = input_spec[flatten_axis:] if is_dbias else (prefix + "_dbias",)
        amax = (BATCHING + prefix + "_amax",)
        scale = (BATCHING + prefix + "_scale",)
        sr_rng_state = (
            BATCHING + prefix + "_sr_rng_state_partition_axis",
            BATCHING + prefix + "sr_rng_state_data_axis",
        )

        post_rht_amax = (BATCHING + prefix + "_post_rht_amax",)
        rht_matrix = (BATCHING + prefix + "_rht_matrix_1", BATCHING + prefix + "_rht_matrix_2")

        return SdyShardingRule(
            (input_spec, scale, amax, sr_rng_state, post_rht_amax, rht_matrix),
            (
                scale_rules.rowwise_out_spec,
                scale_rules.colwise_out_spec,
                scale_rules.rowwise_scale_spec,
                scale_rules.colwise_scale_spec,
                amax,
                dbias,
            ),
            **scale_rules.factor_sizes,
        )


register_primitive(BaseDBiasQuantizePrimitive)


class DBiasQuantizePrimitive(BaseDBiasQuantizePrimitive):
    """Subclass of BaseDBiasQuantizePrimitive for DBias quantization. No change in functionality from the base primitive but named differently for use in more granular disabling of primitives via NVTE_JAX_CUSTOM_CALLS."""


class QuantizePrimitive(BaseDBiasQuantizePrimitive):
    """Subclass of BaseDBiasQuantizePrimitive for quantization without dbias. No change in functionality from the base primitive but named differently for use in more granular disabling of primitives via NVTE_JAX_CUSTOM_CALLS."""


def _jax_quantize(
    x, quantizer: Quantizer = None, dq_dtype: Optional[jnp.dtype] = None, flatten_axis: int = -1
):
    if quantizer is None:
        if isinstance(x, NoScaleTensor):
            return x
        return NoScaleTensor(data=x, amax=None)
    return quantizer.quantize(x, dq_dtype=dq_dtype, flatten_axis=flatten_axis)


def _jax_dbias(dx: Union[jnp.ndarray, NoScaleTensor], dtype=None, flatten_axis: int = -1):
    if isinstance(dx, NoScaleTensor):
        dx = dx.data
    sum_axis = dx.ndim + flatten_axis if flatten_axis < 0 else flatten_axis
    assert sum_axis < dx.ndim, "Flatten axis out of bounds!"
    dtype = dtype or dx.dtype
    dbias = jnp.sum(
        dx.astype(jnp.float32),
        axis=tuple(range(sum_axis)),
        keepdims=False,
    )
    return dbias.astype(dtype)


def _jax_quantize_dbias(
    x,
    quantizer: Quantizer = None,
    dq_dtype: Optional[jnp.dtype] = None,
    flatten_axis: int = -1,
):
    if quantizer is None:
        if isinstance(x, NoScaleTensor):
            return x, None
        return NoScaleTensor(data=x, amax=None), None
    return (
        quantizer.quantize(x, dq_dtype=dq_dtype, flatten_axis=flatten_axis),
        _jax_dbias(x, dtype=dq_dtype, flatten_axis=flatten_axis),
    )


def _quantize_dbias_impl(
    x: Union[jnp.ndarray, NoScaleTensor],
    quantizer: Quantizer,
    is_dbias: bool = False,
    dq_dtype: Optional[jnp.dtype] = None,
    flatten_axis: int = -1,
    amax_scope: AmaxScope = AmaxScope.LOCAL,  # Only works when using current-scaling
    transpose_batch_sequence: bool = False,
) -> Tuple[ScaledTensor2x, jnp.ndarray]:
    """
    Cast wrapper
    Return FP8 tensor
    """
    assert (dq_dtype is None) or (
        quantizer is not None
    ), "quantizer must be provided if dq_dtype is provided"

    if isinstance(x, jnp.ndarray):
        x = NoScaleTensor(data=x, amax=None)

    # Early-exit for non-quantized call
    dq_dtype = dq_dtype or x.data.dtype
    if quantizer is None:
        dbias = None
        if is_dbias:
            dbias = _jax_dbias(x.data, dtype=dq_dtype, flatten_axis=flatten_axis)
        return x, dbias

    # If TE/common custom quantize op is disabled, or if quantizer layout is COLWISE,
    # fall back on the native-JAX quantize implementation
    PrimitiveClass = DBiasQuantizePrimitive if is_dbias else QuantizePrimitive
    is_unsupported = quantizer.q_layout.is_colwise_only and not (
        quantizer.scaling_mode == ScalingMode.NVFP4_1D_SCALING
        and hasattr(quantizer, "use_rht")
        and quantizer.use_rht
    )
    if is_unsupported or not PrimitiveClass.enabled():
        if is_dbias:
            return _jax_quantize_dbias(
                x,
                quantizer=quantizer,
                dq_dtype=dq_dtype,
                flatten_axis=flatten_axis,
            )
        return (
            _jax_quantize(x, quantizer=quantizer, dq_dtype=dq_dtype, flatten_axis=flatten_axis),
            None,
        )

    # TE/common custom quantize op does not support dbias fusion with 1x quantization on arch < 100
    if should_apply_1x_fused_dbias_war_for_arch_l_100(is_dbias=is_dbias, quantizer=quantizer):
        out, _ = _quantize_dbias_impl(
            x=x,
            is_dbias=False,
            quantizer=quantizer,
            dq_dtype=dq_dtype,
            flatten_axis=flatten_axis,
            amax_scope=amax_scope,
            transpose_batch_sequence=transpose_batch_sequence,
        )
        dbias = _jax_dbias(x.data, dtype=dq_dtype, flatten_axis=flatten_axis)
        return out, dbias

    use_rht = False

    scale = jnp.empty((1,), jnp.float32)
    post_rht_amax = None
    rht_matrix = jnp.empty((1, 1), jnp.bfloat16)
    amax = x.amax

    if hasattr(quantizer, "use_rht") and quantizer.use_rht:
        use_rht = True
        rht_matrix = get_rht_matrix()

        new_amax, post_rht_amax = calculate_post_rht_amax(
            x.data,
            amax_scope=amax_scope,
            transpose_batch_sequence=transpose_batch_sequence,
            produce_regular_amax=amax is None,
            flatten_axis=flatten_axis,
        )
        if amax is None:
            # If amax is already calculated in a previous layer, we skip calculating it in the TE kernel
            # So here we only calculate and update amax when it is not provided from a previous layer (amax is None)
            amax = new_amax

    if quantizer.scaling_mode == ScalingMode.CURRENT_TENSOR_SCALING:
        if amax is None:
            amax = calculate_amax(
                x.data,
                amax_scope=amax_scope,
                transpose_batch_sequence=transpose_batch_sequence,
            )
        scale = compute_scale_from_amax(amax, quantizer.q_dtype)
    elif quantizer.scaling_mode == ScalingMode.DELAYED_TENSOR_SCALING:
        scale = quantizer.scale
        # Make sure to reset amax to zeros for DelayedScaling
        amax = jnp.zeros((1,), jnp.float32)
    elif quantizer.scaling_mode.is_nvfp4_scaling:
        if amax is None:
            amax = calculate_amax(
                x.data,
                amax_scope=amax_scope,
                transpose_batch_sequence=transpose_batch_sequence,
            )

    # Make sure amax is not None
    if amax is None:
        amax = jnp.zeros((1,), jnp.float32)

    # It is faster to use 1x quantization for tensor scaling
    is_1x_kernel_supported = not (is_dbias and get_min_device_compute_capability() < 100)
    force_1x_quantization = (
        quantizer.scaling_mode.is_tensor_scaling()
        and quantizer.q_layout.is_rowwise_colwise
        and is_1x_kernel_supported
    )
    q_layout = quantizer.q_layout

    if force_1x_quantization:
        q_layout = QuantizeLayout.ROWWISE

    sr_rng_state = None
    if quantizer.scaling_mode.is_nvfp4_scaling:
        # Only NVFP4 scaling modes support stochastic rounding
        if quantizer.stochastic_rounding_rng_state is not None:
            sr_rng_state = quantizer.stochastic_rounding_rng_state

    (
        rowwise_casted_output,
        colwise_casted_output,
        rowwise_scale_inv,
        colwise_scale_inv,
        updated_amax,
        dbias,
    ) = PrimitiveClass.outer_primitive.bind(
        x.data,
        scale,
        amax,
        (
            sr_rng_state
            if sr_rng_state is not None
            else jnp.empty((get_num_devices_in_mesh(), 1), jnp.uint32)
        ),
        post_rht_amax if post_rht_amax is not None else jnp.zeros((1,), jnp.float32),
        rht_matrix,
        out_dtype=quantizer.q_dtype,
        scaling_mode=quantizer.scaling_mode.value,
        q_layout=q_layout,
        flatten_axis=flatten_axis,
        scale_dtype=quantizer.get_scale_dtype(),
        is_dbias=is_dbias if not quantizer.scaling_mode.is_nvfp4_scaling else False,
        is_outer=True,
        stochastic_rounding=sr_rng_state is not None,
        use_rht=use_rht,
    )
    # For DelayedScaling2x, the scale buffer is shared between rowwise and colwise
    if quantizer.scaling_mode.is_tensor_scaling() and quantizer.q_layout.is_rowwise_colwise:
        colwise_scale_inv = rowwise_scale_inv

        if q_layout.is_rowwise_only:
            # Quantizer requires 2x quantization, but we are using 1x quantization
            # for performance reasons, so we need to generate the colwise data in JAX
            if flatten_axis < 0:
                flatten_axis += x.ndim
            colwise_casted_output = jnp.transpose(
                rowwise_casted_output, (*range(flatten_axis, x.ndim), *range(flatten_axis))
            )
    quantizer.update(updated_amax)
    if quantizer.scaling_mode.is_nvfp4_scaling and is_dbias:
        dbias = _jax_dbias(x, flatten_axis=flatten_axis)

    out = ScaledTensorFactory.create(
        data=rowwise_casted_output,
        scale_inv=rowwise_scale_inv,
        colwise_data=colwise_casted_output,
        colwise_scale_inv=colwise_scale_inv,
        amax=updated_amax,
        colwise_amax=post_rht_amax,
        scaling_mode=quantizer.scaling_mode,
        dq_dtype=dq_dtype,
        q_layout=quantizer.q_layout,
        data_layout=quantizer.get_data_layout(),
        flatten_axis=flatten_axis,
        colwise_has_rht_applied=use_rht,
    )
    return out, dbias.astype(dq_dtype)


def quantize(
    x: Union[jnp.ndarray, NoScaleTensor],
    quantizer: Quantizer,
    flatten_axis: int = -1,
    amax_scope: AmaxScope = AmaxScope.LOCAL,
    transpose_batch_sequence: bool = False,
) -> Tuple[ScaledTensor]:
    """Quantize input tensor according to the quantizer.

    Args:
        x: Input tensor to be quantized.
            Shape: (..., K) where K is the hidden size.
        quantizer: Quantizer for FP8 quantization of the output.
        flatten_axis: The quantization axis in which input data can be flattened to 2D for quantization.
            Defaults to -1.
            is None.
        amax_scope: Indicate the scope to run amax calculation. This only works when using current-scaling. Default is AmaxScope.LOCAL.

    Returns:
        A ScaledTensor containing the quantized input tensor.
    """
    out, _ = _quantize_dbias_impl(
        x,
        quantizer=quantizer,
        flatten_axis=flatten_axis,
        amax_scope=amax_scope,
        transpose_batch_sequence=transpose_batch_sequence,
    )
    return out


def quantize_dbias(
    dz: Union[jnp.ndarray, NoScaleTensor],
    quantizer: Quantizer,
    is_dbias: bool = True,
    flatten_axis: int = -1,
    amax_scope: AmaxScope = AmaxScope.LOCAL,
    transpose_batch_sequence: bool = False,
) -> Tuple[ScaledTensor2x, jnp.ndarray]:
    """Quantize input tensor and compute bias gradient.

    Args:
        dz: Input tensor to be quantized and used for bias gradient computation.
            Shape: (..., K) where K is the hidden size.
        quantizer: Quantizer for FP8 quantization of the output.
        is_dbias: If True, compute bias gradient. Defaults to True.
        flatten_axis: The quantization axis in which input data can be flattened to 2D for quantization.
            Defaults to -1.
        amax_scope: Indicate the scope to run amax calculation. This only works when using current-scaling. Default is AmaxScope.LOCAL.


    Returns:
        A tuple containing:
        - A ScaledTensor containing the quantized input tensor.
            The ScaledTensor includes both the quantized data and scaling factors.
        - The bias gradient tensor.
            Shape: (K,) or empty if is_dbias is False.
    """
    return _quantize_dbias_impl(
        dz,
        quantizer=quantizer,
        is_dbias=is_dbias,
        flatten_axis=flatten_axis,
        amax_scope=amax_scope,
        transpose_batch_sequence=transpose_batch_sequence,
    )


class GroupedQuantizePrimitive(BasePrimitive):
    """
    Cast Primitive wrapping nvte_quantize and nvte_quantize_dbias
    """

    name = "te_grouped_quantize_ffi"
    multiple_results = True
    impl_static_args = (
        3,
        4,
        5,
        6,
        7,
        8,
    )  # out_dtype, scaling_mode, q_layout, flatten_axis, scale_dtype
    inner_primitive = None
    outer_primitive = None

    @staticmethod
    def abstract(
        x_aval,
        scale_aval,
        group_sizes_aval,
        *,
        out_dtype,
        scaling_mode,
        q_layout,
        flatten_axis,
        group_axis,
        scale_dtype,
    ):
        """
        te_dbias_quantize_p abstract
        """
        dtype = dtypes.canonicalize_dtype(x_aval.dtype)
        assert dtype in [jnp.float32, jnp.float16, jnp.bfloat16]
        out_shape = math.prod(x_aval.shape)
        # TODO(Phuong): can scale_aval be None?
        assert scale_aval is None or scale_aval.dtype == jnp.float32

        assert out_dtype in ScalingMode(scaling_mode).get_compatible_q_dtypes(), (
            f"out_dtype {out_dtype} not compatible with scaling_mode {scaling_mode}. out_dtype must"
            f" be one of {ScalingMode(scaling_mode).get_compatible_q_dtypes()}"
        )

        rowwise_scale_inv_shape, colwise_scale_inv_shape = ScalingMode(
            scaling_mode
        ).get_grouped_scale_shape_2x(
            x_aval.shape,
            group_sizes_aval.size,
            group_axis,
            is_padded=True,
            flatten_axis=flatten_axis,
        )

        if q_layout.has_rowwise:
            rowwise_out_shape = out_shape
        else:
            rowwise_out_shape = (1,)
            rowwise_scale_inv_shape = (1,)
        rowwise_out_aval = jax.core.ShapedArray(shape=rowwise_out_shape, dtype=out_dtype)

        amax_aval = jax.core.ShapedArray(shape=(group_sizes_aval.size,), dtype=jnp.float32)

        if q_layout.has_colwise:
            colwise_out_shape = out_shape
        else:
            colwise_out_shape = (1,)
            colwise_scale_inv_shape = (1,)
        colwise_out_aval = jax.core.ShapedArray(shape=colwise_out_shape, dtype=out_dtype)
        rowwise_scale_inv_aval = jax.core.ShapedArray(
            shape=rowwise_scale_inv_shape, dtype=scale_dtype
        )
        colwise_scale_inv_aval = jax.core.ShapedArray(
            shape=colwise_scale_inv_shape, dtype=scale_dtype
        )

        return (
            rowwise_out_aval,
            colwise_out_aval,
            rowwise_scale_inv_aval,
            colwise_scale_inv_aval,
            amax_aval,
        )

    @staticmethod
    def outer_abstract(*args, **kwargs):
        """
        te_dbias_quantize_p outer primitive abstract
        """
        # Phuong: keeping outer abstract so that we can add fuse dbias later
        (
            rowwise_out,
            colwise_out,
            scale_inv,
            colwise_scale_inv,
            updated_amax,
        ) = GroupedQuantizePrimitive.abstract(*args, **kwargs)
        return rowwise_out, colwise_out, scale_inv, colwise_scale_inv, updated_amax

    @staticmethod
    def lowering(
        ctx,
        x,
        scale,
        group_sizes,
        *,
        out_dtype,
        scaling_mode,
        q_layout,
        flatten_axis,
        group_axis,
        scale_dtype,
    ):
        """
        te_dbias_quantize_p lowering rules
        """
        del out_dtype, scale_dtype
        x_aval, scale_aval, group_sizes_aval = ctx.avals_in
        assert x_aval.dtype in [jnp.float32, jnp.float16, jnp.bfloat16]
        assert scale_aval.dtype == jnp.float32
        assert group_sizes_aval.dtype == jnp.int32
        assert group_axis == 0
        return ffi.ffi_lowering(GroupedQuantizePrimitive.name)(
            ctx,
            x,
            scale,
            group_sizes,
            scaling_mode=scaling_mode.value,
            q_layout=q_layout.value.value,
            flatten_axis=flatten_axis,
        )

    @staticmethod
    def impl(
        x,
        scale,
        group_sizes,
        out_dtype,
        scaling_mode,
        q_layout,
        flatten_axis,
        group_axis,
        scale_dtype,
    ):
        """
        te_dbias_quantize_p implementation
        """
        assert GroupedQuantizePrimitive.inner_primitive is not None
        (
            rowwise_out,
            colwise_out,
            rowwise_scale_inv,
            colwise_scale_inv,
            updated_amax,
        ) = GroupedQuantizePrimitive.inner_primitive.bind(
            x,
            scale,
            group_sizes,
            out_dtype=out_dtype,
            scaling_mode=scaling_mode,
            q_layout=q_layout,
            flatten_axis=flatten_axis,
            group_axis=group_axis,
            scale_dtype=scale_dtype,
        )
        return (rowwise_out, colwise_out, rowwise_scale_inv, colwise_scale_inv, updated_amax)


register_primitive(GroupedQuantizePrimitive)


def grouped_quantize(
    x: jnp.ndarray,
    quantizer: GroupedQuantizer,
    group_sizes: jnp.ndarray = None,
    amax: jnp.ndarray = None,
    flatten_axis: int = -1,
) -> GroupedScaledTensor1x:
    """Quantize a tensor in grouped manner.

    This function quantizes a tensor by splitting it into groups along a specified axis
    and applying quantization to each group separately. The groups can be either specified
    explicitly through group_sizes or automatically split along the group_axis.

    Args:
        x: Input tensor to quantize
        quantizer: The quantizer to use for quantization
        group_sizes: Array of ints containing the size of each group (default: None)
        amax: The amax of x; if None, it is auto-generated. (default: None)
        flatten_axis: The axis along which the tensor could be flattened to 2D (default: -1)

    Returns:
        A GroupedScaledTensor1x containing the quantized data

    Note:
        - If group_sizes is not provided, the tensor will be split into equal-sized groups
          along the group_axis
        - The group_axis is currently fixed to 0
        - The quantizer's q_layout determines whether row-wise, column-wise, or both
          quantization is applied
    """

    if quantizer is None:
        if isinstance(x, NoScaleTensor):
            return x
        return NoScaleTensor(data=x, amax=None)

    # TODO(Phuong): add support for flatten_axis = -2
    assert flatten_axis in (
        -1,
        x.ndim - 1,
    ), f"Only flatten_axis = -1 is supported for now, got {flatten_axis}"
    group_axis = 0

    if group_sizes is None:
        group_sizes = jnp.ones(x.shape[group_axis], dtype=jnp.int32)

    if not GroupedQuantizePrimitive.enabled():
        return quantizer.quantize(
            x, flatten_axis=flatten_axis, group_sizes=group_sizes, group_axis=group_axis
        )
    n_groups = group_sizes.size
    original_shape = x.shape
    assert n_groups == len(
        quantizer.quantizers
    ), f"n_groups={n_groups} != n_quantizers = {len(quantizer.quantizers)}"
    scale = jnp.empty((n_groups,), jnp.float32)

    if quantizer.scaling_mode == ScalingMode.DELAYED_TENSOR_SCALING:
        for i, quantizer_i in enumerate(quantizer.quantizers):
            scale = scale.at[i].set(quantizer_i.scale[0])

    if quantizer.scaling_mode == ScalingMode.CURRENT_TENSOR_SCALING:
        if amax is not None:
            row_amax = amax
        else:
            row_amax = jnp.max(jnp.abs(x), axis=range(group_axis + 1, x.ndim))
        segment_ids = jnp.repeat(
            jnp.arange(n_groups), group_sizes, total_repeat_length=x.shape[group_axis]
        )
        grouped_amax = jax.ops.segment_max(row_amax, segment_ids, num_segments=n_groups)
        for i in range(n_groups):
            tmp_scale = compute_scale_from_amax(grouped_amax[i], quantizer.q_dtype)
            scale = scale.at[i].set(tmp_scale[0])

    is_tensor_scaling = quantizer.scaling_mode in (
        ScalingMode.DELAYED_TENSOR_SCALING,
        ScalingMode.CURRENT_TENSOR_SCALING,
    )
    # WAR for tensor_scaling as TE/Common does not support q_layout = COLWISE yet
    # So we performance ROWWISE_COLWISE and use the colwise_tensor_output
    apply_colwise_war = is_tensor_scaling and quantizer.q_layout.is_colwise_only
    q_layout = QuantizeLayout.ROWWISE_COLWISE if apply_colwise_war else quantizer.q_layout
    (
        rowwise_casted_output,
        colwise_casted_output,
        rowwise_scale_inv,
        colwise_scale_inv,
        updated_amax,
    ) = GroupedQuantizePrimitive.outer_primitive.bind(
        x,
        scale,
        group_sizes,
        out_dtype=quantizer.q_dtype,
        scaling_mode=quantizer.scaling_mode.value,
        q_layout=q_layout,
        flatten_axis=flatten_axis,
        group_axis=group_axis,
        scale_dtype=quantizer.get_scale_dtype(),
    )

    # For DelayedScaling2x and CurrentScaling2x, the scale buffer
    # is shared between rowwise and colwise
    if is_tensor_scaling and quantizer.q_layout.is_rowwise_colwise or apply_colwise_war:
        colwise_scale_inv = rowwise_scale_inv

    # TODO(Phuong): store the whole updated_amax in the grouped_quantize instead?
    if quantizer.scaling_mode == ScalingMode.DELAYED_TENSOR_SCALING:
        for i, quantizer_i in enumerate(quantizer.quantizers):
            quantizer_i.update(updated_amax[i].reshape((1,)))

    out = ScaledTensorFactory.create(
        data=rowwise_casted_output,
        scale_inv=rowwise_scale_inv,
        colwise_data=colwise_casted_output,
        colwise_scale_inv=colwise_scale_inv,
        scaling_mode=quantizer.scaling_mode,
        dq_dtype=x.dtype,
        q_layout=quantizer.q_layout,
        data_layout=quantizer.get_data_layout(),
        flatten_axis=flatten_axis,
        group_sizes=group_sizes,
        original_shape=original_shape,
        group_axis=group_axis,
    )
    return out


def grouped_dbias(grad: jnp.ndarray, group_sizes: jnp.ndarray) -> jnp.ndarray:
    """
    Compute the grouped bias gradient.

    Args:
        grad: jnp.ndarray of shape (M, N)
        group_sizes: jnp.ndarray of shape(num_groups,), sum(group_sizes) == M

    Returns:
        dbias: jnp.ndarray of shape (num_groups, N)
    """
    assert grad.ndim == 2, "Input grad must be a 2D tensor."
    assert group_sizes.ndim == 1, "group_sizes must be a 1D tensor."

    segment_ids = jnp.repeat(
        jnp.arange(group_sizes.size), group_sizes, total_repeat_length=grad.shape[0]
    )
    grad_fp32 = grad.astype(jnp.float32)
    dbias_fp32 = jax.ops.segment_sum(grad_fp32, segment_ids, num_segments=group_sizes.shape[0])
    dbias = dbias_fp32.astype(grad.dtype)
    return dbias
