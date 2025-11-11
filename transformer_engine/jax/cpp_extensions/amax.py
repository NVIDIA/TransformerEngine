# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""JAX/TE custom ops for amax calculation"""
from enum import Enum


import jax
import jax.numpy as jnp
from jax import dtypes, ffi
from jax.experimental.custom_partitioning import SdyShardingRule
from jax.sharding import PartitionSpec

from .base import BasePrimitive, register_primitive
from .misc import (
    get_padded_spec,
    NamedSharding,
)
from ..sharding import (
    global_mesh_resource,
    lax_paral_op,
)
from ..quantize import (
    get_wgrad_sign_vector,
    get_sign_from_vector,
)


__all__ = ["AmaxScope", "calculate_amax", "calculate_post_rht_amax"]


class AmaxScope(Enum):
    """
    Amax Scope Enum
    """

    LOCAL = 1
    TPSP = 2
    FSDP = 3

    def all_reduce_amax_along_TPSP_and_FSDP(self, amax, data_spec, transpose_batch_sequence, mesh):
        """Reduce the amax based on its scope"""
        gmesh = global_mesh_resource()
        sequence_dim = 0 if transpose_batch_sequence else 1
        # Run AR across TPSP only when tensor-sequence is detected in the input spec
        if self is AmaxScope.TPSP and data_spec[sequence_dim] == gmesh.tpsp_resource:
            return lax_paral_op(amax, jax.lax.pmax, gmesh.tpsp_resource, mesh)
        # Run AR across FSDP
        if self is AmaxScope.FSDP:
            return lax_paral_op(amax, jax.lax.pmax, gmesh.fsdp_resource, mesh)
        return amax


class AmaxCalculationPrimitive(BasePrimitive):
    """
    Amax Calculation Primitive with custom_partitioning
    """

    name = "jax_local_amax"
    multiple_results = False
    impl_static_args = (
        1,
        2,
    )  # amax_scope, transpose_batch_sequence
    inner_primitive = None
    outer_primitive = None

    @staticmethod
    def abstract(
        x_aval,
        *,
        amax_scope,
        transpose_batch_sequence,
    ):
        """
        amax calcuation abstract
        """
        del amax_scope, transpose_batch_sequence

        dtype = dtypes.canonicalize_dtype(x_aval.dtype)
        assert dtype in [jnp.float32, jnp.float16, jnp.bfloat16]

        out_aval = jax.core.ShapedArray(shape=(1,), dtype=jnp.float32)
        return out_aval

    @staticmethod
    def impl(
        x,
        amax_scope,
        transpose_batch_sequence,
    ):
        """
        amax calcuation implementation
        """
        del amax_scope, transpose_batch_sequence
        amax = jnp.amax(jnp.abs(x), keepdims=True).astype(jnp.float32).reshape((1,))
        return amax

    @staticmethod
    def infer_sharding_from_operands(
        amax_scope,
        transpose_batch_sequence,
        mesh,
        arg_infos,
        result_infos,
    ):
        """
        amax calcuation infer_sharding_from_operands
        """
        del (amax_scope, transpose_batch_sequence, arg_infos, result_infos)  # Unused.
        amax_sharding = NamedSharding(
            mesh,
            PartitionSpec(None),
            desc="AmaxCalculationPrimitive.out_sharding",
        )
        return amax_sharding

    @staticmethod
    def partition(
        amax_scope,
        transpose_batch_sequence,
        mesh,
        arg_infos,
        result_infos,
    ):
        """
        amax calcuation partition
        """
        del result_infos
        x_spec = get_padded_spec(arg_infos[0])
        amax_sharding = NamedSharding(
            mesh,
            PartitionSpec(None),
            desc="AmaxCalculation.amax_sharding",
        )

        def sharded_impl(x):
            amax = AmaxCalculationPrimitive.impl(
                x,
                amax_scope=amax_scope,
                transpose_batch_sequence=transpose_batch_sequence,
            )
            amax = amax_scope.all_reduce_amax_along_TPSP_and_FSDP(
                amax, x_spec, transpose_batch_sequence, mesh
            )

            return amax

        arg_shardings = tuple(arg_i.sharding for arg_i in arg_infos)
        return mesh, sharded_impl, amax_sharding, arg_shardings

    @staticmethod
    def shardy_sharding_rule(amax_scope, transpose_batch_sequence, mesh, value_types, result_types):
        """
        amax calcuation shardy_sharding_rule
        """
        del amax_scope, transpose_batch_sequence, mesh, result_types
        prefix = "AmaxCal"
        input_spec = tuple(f"{prefix}_{i}" for i in range(len(value_types[0].shape)))
        output_spec = (f"{prefix}_amax",)
        return SdyShardingRule((input_spec,), (output_spec,))


register_primitive(AmaxCalculationPrimitive, outer_only=True)


class RHTAmaxCalculationPrimitive(BasePrimitive):
    """
    Amax Calculation Primitive with custom_partitioning for calculating regular and post-Random Hadamard Transform (RHT) amax using TE's fused kernels.
    """

    name = "te_rht_amax_ffi"
    multiple_results = True
    impl_static_args = (
        1,  # amax_scope
        2,  # transpose_batch_sequence
        3,  # rht_matrix_random_sign_mask_t
        4,  # produce_regular_amax
        5,  # flatten_axis
    )
    inner_primitive = None
    outer_primitive = None

    @staticmethod
    def abstract(
        x_aval,
        *,
        amax_scope,
        transpose_batch_sequence,
        rht_matrix_random_sign_mask_t,
        produce_regular_amax,
        flatten_axis,
    ):
        """
        amax calcuation abstract
        """
        del (
            amax_scope,
            transpose_batch_sequence,
            rht_matrix_random_sign_mask_t,
            produce_regular_amax,
            flatten_axis,
        )

        dtype = dtypes.canonicalize_dtype(x_aval.dtype)
        assert dtype in [jnp.bfloat16], f"RHT requires input to be bfloat16, but got {dtype}"

        amax_aval = jax.core.ShapedArray(shape=(1,), dtype=jnp.float32)
        post_rht_amax_aval = jax.core.ShapedArray(shape=(1,), dtype=jnp.float32)

        return amax_aval, post_rht_amax_aval

    @staticmethod
    def lowering(
        ctx,
        x,
        *,
        amax_scope,
        transpose_batch_sequence,
        rht_matrix_random_sign_mask_t,
        produce_regular_amax,
        flatten_axis,
    ):
        """
        te_dbias_quantize_p lowering rules
        """
        del amax_scope, transpose_batch_sequence
        (x_aval,) = ctx.avals_in
        assert x_aval.dtype in [jnp.float32, jnp.float16, jnp.bfloat16]

        flatten_axis = flatten_axis if flatten_axis >= 0 else flatten_axis + len(x_aval.shape)
        assert 0 < flatten_axis < len(x_aval.shape), "Flatten axis out of bounds!"

        return ffi.ffi_lowering(
            RHTAmaxCalculationPrimitive.name,
        )(
            ctx,
            x,
            rht_matrix_random_sign_mask_t=rht_matrix_random_sign_mask_t,
            produce_regular_amax=produce_regular_amax,
            flatten_axis=flatten_axis,
        )

    @staticmethod
    def impl(
        x,
        amax_scope,
        transpose_batch_sequence,
        rht_matrix_random_sign_mask_t,
        produce_regular_amax,
        flatten_axis,
    ):
        """
        amax calcuation implementation
        """
        assert RHTAmaxCalculationPrimitive.inner_primitive is not None
        (
            amax,
            post_rht_amax,
        ) = RHTAmaxCalculationPrimitive.inner_primitive.bind(
            x,
            amax_scope=amax_scope,
            transpose_batch_sequence=transpose_batch_sequence,
            rht_matrix_random_sign_mask_t=rht_matrix_random_sign_mask_t,
            produce_regular_amax=produce_regular_amax,
            flatten_axis=flatten_axis,
        )
        return amax, post_rht_amax

    @staticmethod
    def infer_sharding_from_operands(
        amax_scope,
        transpose_batch_sequence,
        rht_matrix_random_sign_mask_t,
        produce_regular_amax,
        flatten_axis,
        mesh,
        arg_infos,
        result_infos,
    ):
        """
        amax calcuation infer_sharding_from_operands
        """
        del (
            amax_scope,
            transpose_batch_sequence,
            rht_matrix_random_sign_mask_t,
            produce_regular_amax,
            flatten_axis,
            arg_infos,
            result_infos,
        )  # Unused.
        amax_sharding = NamedSharding(
            mesh,
            PartitionSpec(None),
            desc="RHTAmaxCalculationPrimitive.out_sharding",
        )
        return amax_sharding, amax_sharding

    @staticmethod
    def partition(
        amax_scope,
        transpose_batch_sequence,
        rht_matrix_random_sign_mask_t,
        produce_regular_amax,
        flatten_axis,
        mesh,
        arg_infos,
        result_infos,
    ):
        """
        amax calcuation partition
        """
        del result_infos
        x_spec = get_padded_spec(arg_infos[0])
        amax_sharding = NamedSharding(
            mesh,
            PartitionSpec(None),
            desc="RHTAmaxCalculationPrimitive.amax_sharding",
        )
        out_shardings = (amax_sharding, amax_sharding)

        def sharded_impl(x):
            amax, post_rht_amax = RHTAmaxCalculationPrimitive.impl(
                x,
                amax_scope=amax_scope,
                transpose_batch_sequence=transpose_batch_sequence,
                rht_matrix_random_sign_mask_t=rht_matrix_random_sign_mask_t,
                produce_regular_amax=produce_regular_amax,
                flatten_axis=flatten_axis,
            )
            amax = amax_scope.all_reduce_amax_along_TPSP_and_FSDP(
                amax, x_spec, transpose_batch_sequence, mesh
            )
            post_rht_amax = amax_scope.all_reduce_amax_along_TPSP_and_FSDP(
                post_rht_amax, x_spec, transpose_batch_sequence, mesh
            )

            return amax, post_rht_amax

        arg_shardings = tuple(arg_i.sharding for arg_i in arg_infos)
        return mesh, sharded_impl, out_shardings, arg_shardings

    @staticmethod
    def shardy_sharding_rule(
        amax_scope,
        transpose_batch_sequence,
        rht_matrix_random_sign_mask_t,
        produce_regular_amax,
        flatten_axis,
        mesh,
        value_types,
        result_types,
    ):
        """
        amax calcuation shardy_sharding_rule
        """
        del (
            amax_scope,
            transpose_batch_sequence,
            rht_matrix_random_sign_mask_t,
            produce_regular_amax,
            flatten_axis,
            mesh,
            result_types,
        )
        prefix = "RHTAmaxCal"
        input_spec = tuple(f"{prefix}_{i}" for i in range(len(value_types[0].shape)))
        output_amax_spec = (f"{prefix}_amax",)
        output_post_rht_amax_spec = (f"{prefix}_post_rht_amax",)
        return SdyShardingRule((input_spec,), (output_amax_spec, output_post_rht_amax_spec))


register_primitive(RHTAmaxCalculationPrimitive)


def calculate_amax(x: jnp.ndarray, amax_scope: AmaxScope, transpose_batch_sequence: bool):
    """
    Compute the maximum absolute value (amax) of the input tensor.
    """
    assert AmaxCalculationPrimitive.outer_primitive is not None
    return AmaxCalculationPrimitive.outer_primitive.bind(
        x,
        amax_scope=amax_scope,
        transpose_batch_sequence=transpose_batch_sequence,
    )


def calculate_post_rht_amax(
    x: jnp.ndarray,
    amax_scope: AmaxScope,
    transpose_batch_sequence: bool,
    produce_regular_amax: bool,
    flatten_axis: int,
):
    """Compute the post-Random Hadamard Transform (RHT) amax of the input tensor, and optionally the regular amax.

    Args:
        x: Input tensor.
        amax_scope: The scope for amax reduction (local, TPSP, or FSDP).
        transpose_batch_sequence: Whether the input tensor has its batch and sequence dimensions transposed.
        produce_regular_amax: Whether to compute and return the regular amax alongside the post-RHT amax.
        flatten_axis: The axis at which to flatten the input tensor before applying RHT.
    Returns:
        A tuple containing:
            - The regular amax if `produce_regular_amax` is True, otherwise None.
            - The post-RHT amax.
    """
    amax, post_rht_amax = RHTAmaxCalculationPrimitive.outer_primitive.bind(
        x,
        amax_scope=amax_scope,
        transpose_batch_sequence=transpose_batch_sequence,
        rht_matrix_random_sign_mask_t=get_sign_from_vector(get_wgrad_sign_vector()),
        produce_regular_amax=produce_regular_amax,
        flatten_axis=flatten_axis,
    )

    if produce_regular_amax:
        return amax, post_rht_amax
    return None, post_rht_amax
