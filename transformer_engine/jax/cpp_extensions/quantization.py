# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""JAX/TE custom ops for quantization"""
import operator
from functools import reduce
from typing import Tuple, Optional
import math
from packaging import version

import jax
import jax.numpy as jnp
from jax import dtypes
from jax.experimental.custom_partitioning import SdyShardingRule
from jax.sharding import PartitionSpec

import transformer_engine_jax

from .base import BasePrimitive, register_primitive
from .misc import (
    get_padded_spec,
    check_valid_batch_dims,
    te_dtype_to_jax_dtype,
    jax_dtype_to_te_dtype,
    multidim_transpose,
    should_apply_1x_fused_dbias_war_for_arch_l_100,
    NamedSharding,
)
from ..sharding import all_reduce_max_along_all_axes_except_PP, all_reduce_sum_along_dp_fsdp
from ..quantize import (
    ScaledTensor2x,
    ScaledTensor,
    ScaledTensorFactory,
    GroupedScaledTensor1x,
    Quantizer,
    GroupedQuantizer,
    QuantizeLayout,
    DelayedScaleQuantizer,
    ScalingMode,
    compute_scale_from_amax,
)

if version.parse(jax.__version__) >= version.parse("0.5.0"):
    from jax import ffi  # pylint: disable=ungrouped-imports
else:
    from jax.extend import ffi  # pylint: disable=ungrouped-imports


__all__ = ["quantize", "quantize_dbias", "grouped_quantize"]


class BaseDBiasQuantizePrimitive(BasePrimitive):
    """
    Cast Primitive wrapping nvte_quantize and nvte_quantize_dbias
    """

    name = "te_dbias_quantize_ffi"
    multiple_results = True
    impl_static_args = (
        2,
        3,
        4,
        5,
        6,
        7,
        8,
    )  # out_dtype, scaling_mode, q_layout, flatten_axis, scale_dtype, is_dbias, is_outer
    inner_primitive = None
    outer_primitive = None

    @staticmethod
    def abstract(
        x_aval,
        scale_aval,
        *,
        out_dtype,
        scaling_mode,
        q_layout,
        flatten_axis,
        scale_dtype,
        is_dbias,
        is_outer,
    ):
        """
        te_dbias_quantize_p abstract
        """
        dtype = dtypes.canonicalize_dtype(x_aval.dtype)
        assert dtype in [jnp.float32, jnp.float16, jnp.bfloat16]
        out_shape = x_aval.shape
        assert scale_aval is None or scale_aval.dtype == jnp.float32

        if q_layout in (QuantizeLayout.ROWWISE.value, QuantizeLayout.ROWWISE_COLWISE.value):
            rowwise_out_shape = out_shape
        else:
            rowwise_out_shape = (1,)
        rowwise_out_aval = jax.core.ShapedArray(shape=rowwise_out_shape, dtype=out_dtype)

        updated_amax_aval = jax.core.ShapedArray(shape=(1,), dtype=jnp.float32)

        rowwise_scale_inv_shape, colwise_scale_inv_shape = ScalingMode(
            scaling_mode
        ).get_scale_shape_2x(x_aval.shape, is_padded=not is_outer, flatten_axis=flatten_axis)

        if q_layout in (QuantizeLayout.COLWISE.value, QuantizeLayout.ROWWISE_COLWISE.value):
            if ScalingMode(scaling_mode).is_tensor_scaling():
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
                scaling_mode,
                QuantizeLayout(
                    q_layout
                ),  # For now until we have auto-decoding for QuantizeLayout enum
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
        *,
        out_dtype,
        scaling_mode,
        q_layout,
        flatten_axis,
        scale_dtype,
        is_dbias,
        is_outer,
    ):
        """
        te_dbias_quantize_p lowering rules
        """
        del out_dtype, scale_dtype, is_outer
        x_aval, scale_aval = ctx.avals_in
        assert x_aval.dtype in [jnp.float32, jnp.float16, jnp.bfloat16]
        assert scale_aval.dtype == jnp.float32
        return ffi.ffi_lowering(BaseDBiasQuantizePrimitive.name)(
            ctx,
            x,
            scale,
            scaling_mode=scaling_mode.value,
            q_layout=q_layout,
            flatten_axis=flatten_axis,
            is_dbias=is_dbias,
        )

    @staticmethod
    def impl(
        x,
        scale,
        out_dtype,
        scaling_mode,
        q_layout,
        flatten_axis,
        scale_dtype,
        is_dbias,
        is_outer,
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
            out_dtype=out_dtype,
            scaling_mode=scaling_mode,
            q_layout=q_layout,
            flatten_axis=flatten_axis,
            scale_dtype=scale_dtype,
            is_dbias=is_dbias,
            is_outer=False,
        )
        rowwise_scale_inv_shape, colwise_scale_inv_shape = ScalingMode(
            scaling_mode
        ).get_scale_shape_2x(x.shape, is_padded=False, flatten_axis=flatten_axis)
        scale_inv = jax.lax.slice(
            scale_inv, [0] * len(rowwise_scale_inv_shape), rowwise_scale_inv_shape
        )
        if q_layout in (QuantizeLayout.COLWISE.value, QuantizeLayout.ROWWISE_COLWISE.value):
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
    ):
        """
        to describe batch rules for vmap
        """
        del is_outer
        check_valid_batch_dims(batch_dims)
        assert BaseDBiasQuantizePrimitive.outer_primitive is not None
        x, scale = batched_args
        x_bdim, scale_bdim = batch_dims
        amax_bdim = scale_bdim

        out_bdims = x_bdim, x_bdim, scale_bdim, scale_bdim, amax_bdim, x_bdim
        return (
            BaseDBiasQuantizePrimitive.outer_primitive.bind(
                x,
                scale,
                out_dtype=out_dtype,
                scaling_mode=scaling_mode,
                q_layout=q_layout,
                flatten_axis=flatten_axis,
                scale_dtype=scale_dtype,
                is_dbias=is_dbias,
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
        mesh,
        arg_infos,
        result_infos,
    ):
        del (out_dtype, result_infos, scale_dtype, is_outer)  # Unused.

        x_spec = get_padded_spec(arg_infos[0])
        scale_spec = get_padded_spec(arg_infos[1])
        out_sharding = NamedSharding(
            mesh,
            PartitionSpec(*x_spec),
            desc="BaseDBiasQuantizePrimitive.out_sharding",
        )
        if q_layout in (QuantizeLayout.COLWISE.value, QuantizeLayout.ROWWISE_COLWISE.value):
            if ScalingMode(scaling_mode).is_tensor_scaling():
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

        scale_inv_spec = amax_spec = colwise_scale_inv_spec = (None,)
        if scaling_mode == ScalingMode.DELAYED_TENSOR_SCALING.value:
            scale_inv_spec = amax_spec = scale_spec
        elif scaling_mode == ScalingMode.MXFP8_1D_SCALING.value:
            scale_inv_spec = x_spec

        if q_layout in (QuantizeLayout.COLWISE.value, QuantizeLayout.ROWWISE_COLWISE.value):
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
        mesh,
        arg_infos,
        result_infos,
    ):
        del result_infos, is_outer

        x_spec = get_padded_spec(arg_infos[0])
        scale_spec = get_padded_spec(arg_infos[1])
        out_sharding = NamedSharding(
            mesh,
            PartitionSpec(*x_spec),
            desc="BaseDBiasQuantizePrimitive.out_sharding",
        )
        if q_layout in (QuantizeLayout.COLWISE.value, QuantizeLayout.ROWWISE_COLWISE.value):
            if ScalingMode(scaling_mode).is_tensor_scaling():
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

        scale_inv_spec = amax_spec = colwise_scale_inv_spec = (None,)
        if scaling_mode == ScalingMode.DELAYED_TENSOR_SCALING.value:
            scale_inv_spec = amax_spec = scale_spec
        elif scaling_mode == ScalingMode.MXFP8_1D_SCALING.value:
            scale_inv_spec = x_spec

        if q_layout in (QuantizeLayout.COLWISE.value, QuantizeLayout.ROWWISE_COLWISE.value):
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

        arg_shardings = tuple(arg_i.sharding for arg_i in arg_infos)
        out_shardings = (
            out_sharding,
            colwise_out_sharding,
            scale_inv_sharding,
            colwise_scale_inv_sharding,
            amax_sharding,
            dbias_sharding,
        )

        def sharded_impl(x, scale):
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
                out_dtype=out_dtype,
                scaling_mode=scaling_mode,
                q_layout=q_layout,
                flatten_axis=flatten_axis,
                scale_dtype=scale_dtype,
                is_dbias=is_dbias,
                is_outer=True,
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
        mesh,
        value_types,
        result_types,
    ):
        del out_dtype, scale_dtype, is_outer, mesh, result_types

        scale_rules = ScalingMode(scaling_mode).get_shardy_sharding_rules(
            len(value_types[0].shape),
            unique_var="BaseDBiasQuantizePrimitive_i",
            flatten_axis=flatten_axis,
        )

        x_axes = scale_rules.input_spec
        colwise_scale_inv = scale_rules.colwise_rule

        out = x_axes
        if q_layout in (QuantizeLayout.COLWISE.value, QuantizeLayout.ROWWISE_COLWISE.value):
            if ScalingMode(scaling_mode).is_tensor_scaling():
                colwise_out = tuple(multidim_transpose(x_axes, transpose_axis=flatten_axis))
            else:
                colwise_out = x_axes
        else:
            colwise_out = ("j",)
            colwise_scale_inv = ("k",)

        dbias = x_axes[flatten_axis:] if is_dbias else ("l",)
        amax = ("m",)

        return SdyShardingRule(
            (x_axes, ("…1",)),
            (out, colwise_out, scale_rules.rowwise_rule, colwise_scale_inv, amax, dbias),
            **scale_rules.factor_sizes,
        )


register_primitive(BaseDBiasQuantizePrimitive)


class DBiasQuantizePrimitive(BaseDBiasQuantizePrimitive):
    """Subclass of BaseDBiasQuantizePrimitive for DBias quantization. No change in functionality from the base primitive but named differently for use in more granular disabling of primitives via NVTE_JAX_CUSTOM_CALLS_RE."""


class QuantizePrimitive(BaseDBiasQuantizePrimitive):
    """Subclass of BaseDBiasQuantizePrimitive for quantization without dbias. No change in functionality from the base primitive but named differently for use in more granular disabling of primitives via NVTE_JAX_CUSTOM_CALLS_RE."""


def _jax_quantize(
    x, quantizer: Quantizer = None, dq_dtype: Optional[jnp.dtype] = None, flatten_axis: int = -1
):
    if quantizer is None:
        return x
    return quantizer.quantize(x, dq_dtype=dq_dtype, flatten_axis=flatten_axis)


def _jax_dbias(dx: jnp.ndarray, dtype=None, flatten_axis: int = -1):
    assert flatten_axis < 0
    dtype = dtype or dx.dtype
    dbias = jnp.sum(
        dx.astype(jnp.float32),
        axis=tuple(range(dx.ndim + flatten_axis)),
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
        return x, None
    return (
        quantizer.quantize(x, dq_dtype=dq_dtype, flatten_axis=flatten_axis),
        _jax_dbias(x, dtype=dq_dtype, flatten_axis=flatten_axis),
    )


def _quantize_dbias_impl(
    x: jnp.ndarray,
    quantizer: Quantizer,
    is_dbias: bool = False,
    dq_dtype: Optional[jnp.dtype] = None,
    flatten_axis: int = -1,
) -> Tuple[ScaledTensor2x, jnp.ndarray]:
    """
    Cast wrapper
    Return FP8 tensor
    """
    assert (dq_dtype is None) or (
        quantizer is not None
    ), "quantizer must be provided if dq_dtype is provided"

    dq_dtype = dq_dtype or x.dtype

    PrimitiveClass = DBiasQuantizePrimitive if is_dbias else QuantizePrimitive
    if not PrimitiveClass.enabled():
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

    # TE/common doesn't support colwise only quantization yet
    if quantizer is not None and quantizer.q_layout == QuantizeLayout.COLWISE:
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
    scale = jnp.empty((), jnp.float32)

    # TE/common dbias_quantize does not support 1x on arch < 100
    if should_apply_1x_fused_dbias_war_for_arch_l_100(is_dbias=is_dbias, quantizer=quantizer):
        out, _ = _quantize_dbias_impl(
            x=x,
            is_dbias=False,
            quantizer=quantizer,
            dq_dtype=dq_dtype,
            flatten_axis=flatten_axis,
        )
        dbias = _jax_dbias(x, dtype=dq_dtype, flatten_axis=flatten_axis)
        return out, dbias

    if quantizer is None:
        if is_dbias:
            return x, _jax_dbias(x, dtype=dq_dtype, flatten_axis=flatten_axis)
        return x, None

    if quantizer.scaling_mode == ScalingMode.CURRENT_TENSOR_SCALING:
        # Globally reduce amax across all devices for current scaling so we have a single global scale.
        # This differs from the PyTorch implementation which uses a local amax and scale per-device and persists this
        # until the tensor is dequantized (e.g. in the GEMM).
        amax = jnp.amax(jnp.abs(x), keepdims=True).astype(jnp.float32)
        scale = compute_scale_from_amax(amax, quantizer.q_dtype)

    if isinstance(quantizer, DelayedScaleQuantizer):
        scale = quantizer.scale

    (
        rowwise_casted_output,
        colwise_casted_output,
        rowwise_scale_inv,
        colwise_scale_inv,
        updated_amax,
        dbias,
    ) = PrimitiveClass.outer_primitive.bind(
        x,
        scale,
        out_dtype=quantizer.q_dtype,
        scaling_mode=quantizer.scaling_mode.value,
        q_layout=quantizer.q_layout.value,
        flatten_axis=flatten_axis,
        scale_dtype=quantizer.get_scale_dtype(),
        is_dbias=is_dbias,
        is_outer=True,
    )
    # For DelayedScaling2x, the scale buffer is shared between rowwise and colwise
    if quantizer.scaling_mode.is_tensor_scaling() and quantizer.is_2x2x():
        colwise_scale_inv = rowwise_scale_inv

    quantizer.update(updated_amax)

    out = ScaledTensorFactory.create(
        data=rowwise_casted_output,
        scale_inv=rowwise_scale_inv,
        colwise_data=colwise_casted_output,
        colwise_scale_inv=colwise_scale_inv,
        scaling_mode=quantizer.scaling_mode,
        dq_dtype=dq_dtype,
        q_layout=quantizer.q_layout,
        data_layout=quantizer.get_data_layout(),
        flatten_axis=flatten_axis,
    )
    return out, dbias.astype(dq_dtype)


def quantize(
    x: jnp.ndarray,
    quantizer: Quantizer,
    flatten_axis: int = -1,
) -> Tuple[ScaledTensor]:
    """Quantize input tensor according to the quantizer.

    Args:
        x: Input tensor to be quantized.
            Shape: (..., K) where K is the hidden size.
        quantizer: Quantizer for FP8 quantization of the output.
        flatten_axis: The quantization axis in which input data can be flattened to 2D for quantization.
            Defaults to -1.

    Returns:
        A ScaledTensor containing the quantized input tensor.
    """
    out, _ = _quantize_dbias_impl(
        x,
        quantizer=quantizer,
        flatten_axis=flatten_axis,
    )
    return out


def quantize_dbias(
    dz: jnp.ndarray,
    quantizer: Quantizer,
    is_dbias: bool = True,
    flatten_axis: int = -1,
) -> Tuple[ScaledTensor2x, jnp.ndarray]:
    """Quantize input tensor and compute bias gradient.

    Args:
        dz: Input tensor to be quantized and used for bias gradient computation.
            Shape: (..., K) where K is the hidden size.
        quantizer: Quantizer for FP8 quantization of the output.
        is_dbias: If True, compute bias gradient. Defaults to True.
        flatten_axis: The quantization axis in which input data can be flattened to 2D for quantization.
            Defaults to -1.

    Returns:
        A tuple containing:
        - A ScaledTensor containing the quantized input tensor.
            The ScaledTensor includes both the quantized data and scaling factors.
        - The bias gradient tensor.
            Shape: (K,) or empty if is_dbias is False.
    """
    return _quantize_dbias_impl(
        dz, quantizer=quantizer, is_dbias=is_dbias, flatten_axis=flatten_axis
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

        rowwise_scale_inv_shape, colwise_scale_inv_shape = ScalingMode(
            scaling_mode
        ).get_grouped_scale_shape_2x(
            x_aval.shape,
            group_sizes_aval.size,
            group_axis,
            is_padded=True,
            flatten_axis=flatten_axis,
        )

        if q_layout in (QuantizeLayout.ROWWISE.value, QuantizeLayout.ROWWISE_COLWISE.value):
            rowwise_out_shape = out_shape
        else:
            rowwise_out_shape = (1,)
            rowwise_scale_inv_shape = (1,)
        rowwise_out_aval = jax.core.ShapedArray(shape=rowwise_out_shape, dtype=out_dtype)

        amax_aval = jax.core.ShapedArray(shape=(group_sizes_aval.size,), dtype=jnp.float32)

        if q_layout in (QuantizeLayout.COLWISE.value, QuantizeLayout.ROWWISE_COLWISE.value):
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
            _dbias,
            _wkspace,
        ) = DBiasQuantizePrimitive.abstract(*args, **kwargs)
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
            q_layout=q_layout,
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
        return x

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

    # TODO(Phuong): for CurrentScaling, move this amax reduction to C++ so that the amax is
    # calculated for each individual matrix in the group
    if quantizer.scaling_mode == ScalingMode.CURRENT_TENSOR_SCALING:
        amax = jnp.amax(jnp.abs(x), keepdims=True).astype(jnp.float32)
        tmp_scale = compute_scale_from_amax(amax, quantizer.q_dtype)
        tmp_scale = tmp_scale.reshape(())
        for i in range(n_groups):
            scale = scale.at[i].set(tmp_scale)

    is_tensor_scaling = quantizer.scaling_mode in (
        ScalingMode.DELAYED_TENSOR_SCALING,
        ScalingMode.CURRENT_TENSOR_SCALING,
    )
    # WAR for tensor_scaling as TE/Common does not support q_layout = COLWISE yet
    # So we performance ROWWISE_COLWISE and use the colwise_tensor_output
    apply_colwise_war = is_tensor_scaling and quantizer.q_layout == QuantizeLayout.COLWISE
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
        q_layout=q_layout.value,
        flatten_axis=flatten_axis,
        group_axis=group_axis,
        scale_dtype=quantizer.get_scale_dtype(),
    )

    # For DelayedScaling2x and CurrentScaling2x, the scale buffer
    # is shared between rowwise and colwise
    if is_tensor_scaling and quantizer.is_2x2x() or apply_colwise_war:
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
