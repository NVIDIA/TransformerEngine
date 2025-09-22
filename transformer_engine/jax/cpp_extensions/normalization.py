# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""JAX/TE custom ops for normalization"""
import os
import warnings
import operator
from functools import partial, cache, reduce
from typing import Optional, Union

import jax
import jax.numpy as jnp
from jax import dtypes, ffi
from jax.experimental.custom_partitioning import SdyShardingRule
from jax.interpreters.mlir import ir
from jax.sharding import PartitionSpec

import transformer_engine_jax
from transformer_engine_jax import NVTE_Norm_Type

from .base import BasePrimitive, register_primitive
from .misc import (
    get_padded_spec,
    check_valid_batch_dims,
    jax_dtype_to_te_dtype,
    te_dtype_to_jax_dtype,
    NamedSharding,
    get_cudnn_version,
)
from .quantization import _quantize_dbias_impl
from ..sharding import all_reduce_max_along_all_axes_except_PP, all_reduce_sum_along_dp_fsdp
from ..quantize import ScaledTensor, ScaledTensorFactory, NoScaleTensor
from ..quantize import (
    Quantizer,
    QuantizeLayout,
    DelayedScaleQuantizer,
    ScalingMode,
)


__all__ = [
    "layernorm_fwd",
    "layernorm_bwd",
    "rmsnorm_fwd",
    "rmsnorm_bwd",
    "normalization_fwd",
    "normalization_bwd",
]


@cache
def get_forward_sm_margin():
    """Retrieves the number of stream multiprocessors (SM) reserved for other kernels"""
    return int(os.getenv("NVTE_FWD_LAYERNORM_SM_MARGIN", "0"))


@cache
def get_backward_sm_margin():
    """Retrieves the number of stream multiprocessors (SM) reserved for other kernels"""
    return int(os.getenv("NVTE_BWD_LAYERNORM_SM_MARGIN", "0"))


@cache
def is_norm_fwd_cudnn_enabled(scaling_mode: ScalingMode) -> bool:
    """Retrieves whether CuDNN norm fwd is enabled."""
    # MXFP8_1D_SCALING always uses CuDNN currently
    return (
        int(os.getenv("NVTE_NORM_FWD_USE_CUDNN", "0")) == 1
        or scaling_mode == ScalingMode.MXFP8_1D_SCALING
    )


@cache
def is_norm_zero_centered_gamma_in_weight_dtype(scaling_mode: ScalingMode) -> bool:
    """Retrieves whether norm should compute `gamma += 1.0` for zero-centered gamma
    in weight dtype as opposed to compute dtype."""
    if not is_norm_fwd_cudnn_enabled(scaling_mode):
        # If CuDNN is not enabled, we use the TE backend which uses the compute dtype not weight dtype
        # Remove this when TE supports gamma += 1.0 in weight dtype
        return False
    return int(os.getenv("NVTE_ZERO_CENTERED_GAMMA_IN_WTYPE", "0")) == 1


# CuDNN version must be at least this to use MXFP8 fused normalization otherwise unfused norm and quantize will be used
FUSED_MXFP8_NORM_CUDNN_MIN_VERSION = (9, 10, 0)


class NormFwdPrimitive(BasePrimitive):
    """
    Layer Normalization Forward FP8 Primitive
    """

    name = "te_norm_forward_ffi"
    multiple_results = True
    impl_static_args = (4, 5, 6, 7, 8, 9, 10, 11)
    inner_primitive = None
    outer_primitive = None

    @staticmethod
    def abstract(
        x_aval,
        scale_aval,
        gamma_aval,
        beta_aval,
        *,
        norm_type,
        zero_centered_gamma,
        epsilon,
        out_dtype,
        scaling_mode,
        is_2x,
        scale_dtype,
        is_outer,
    ):
        """
        LayerNorm fwd inner primitive abstract
        """
        x_dtype = dtypes.canonicalize_dtype(x_aval.dtype)

        assert x_dtype in [jnp.float32, jnp.float16, jnp.bfloat16]
        assert scale_aval is None or scale_aval.dtype == jnp.float32

        assert (
            scaling_mode != ScalingMode.MXFP8_1D_SCALING.value
            or get_cudnn_version() >= FUSED_MXFP8_NORM_CUDNN_MIN_VERSION
        ), (
            "MXFP8 Fused Normalization is only supported in CuDNN version"
            f" {FUSED_MXFP8_NORM_CUDNN_MIN_VERSION} or higher"
        )

        assert scaling_mode != ScalingMode.CURRENT_TENSOR_SCALING.value, (
            "Current tensor scaling is not supported for fused norm and quantization. Please do"
            " norm in higher-precision then quantize with current tensor scaling."
        )

        mu_rsigama_dtype = jnp.float32

        if norm_type == NVTE_Norm_Type.LayerNorm:
            assert gamma_aval.size == beta_aval.size
            assert gamma_aval.dtype == beta_aval.dtype, (
                f"gamma and beta should have the same dtype, but got {gamma_aval.dtype} and "
                f"{beta_aval.dtype}"
            )

        out_aval = x_aval.update(shape=x_aval.shape, dtype=out_dtype)
        mu_aval = rsigma_aval = out_aval.update(shape=out_aval.shape[:-1], dtype=mu_rsigama_dtype)
        if norm_type == NVTE_Norm_Type.RMSNorm:
            mu_aval = mu_aval.update(shape=(1,))

        updated_amax_aval = jax.core.ShapedArray(shape=(1,), dtype=jnp.float32)

        colwise_out_shape = x_aval.shape if is_2x else (1,)
        colwise_out_aval = jax.core.ShapedArray(shape=colwise_out_shape, dtype=out_dtype)

        rowwise_scale_inv_shape, colwise_scale_inv_shape = ScalingMode(
            scaling_mode
        ).get_scale_shape_2x(x_aval.shape, is_padded=not is_outer)

        scale_inv_aval = jax.core.ShapedArray(shape=rowwise_scale_inv_shape, dtype=scale_dtype)
        colwise_scale_inv_shape = colwise_scale_inv_shape if is_2x else (1,)
        colwise_scale_inv_aval = jax.core.ShapedArray(
            shape=colwise_scale_inv_shape, dtype=scale_dtype
        )

        (wkspace_info,) = transformer_engine_jax.get_norm_fwd_workspace_sizes(
            x_aval.size // gamma_aval.size,  # batch size
            gamma_aval.size,  # hidden size
            jax_dtype_to_te_dtype(x_aval.dtype),  # itype
            jax_dtype_to_te_dtype(gamma_aval.dtype),  # wtype
            jax_dtype_to_te_dtype(out_dtype),
            norm_type,
            scaling_mode,
            zero_centered_gamma,
            epsilon,
            get_forward_sm_margin(),
            is_2x,
        )
        wkspace_aval = jax.core.ShapedArray(
            shape=wkspace_info[0], dtype=te_dtype_to_jax_dtype(wkspace_info[1])
        )

        return (
            out_aval,
            colwise_out_aval,
            scale_inv_aval,
            colwise_scale_inv_aval,
            updated_amax_aval,
            mu_aval,
            rsigma_aval,
            wkspace_aval,
        )

    @staticmethod
    def outer_abstract(*args, **kwargs):
        """
        LayerNorm fwd outer primitive abstract
        """
        (
            out_aval,
            colwise_out_aval,
            scale_inv_aval,
            colwise_scale_inv_aval,
            updated_amax_aval,
            mu_aval,
            rsigma_aval,
            _,
        ) = NormFwdPrimitive.abstract(*args, **kwargs)
        return (
            out_aval,
            colwise_out_aval,
            scale_inv_aval,
            colwise_scale_inv_aval,
            updated_amax_aval,
            mu_aval,
            rsigma_aval,
        )

    @staticmethod
    def lowering(
        ctx,
        x,
        scale,
        gamma,
        beta,
        *,
        norm_type,
        zero_centered_gamma,
        epsilon,
        out_dtype,
        scaling_mode,
        is_2x,
        scale_dtype,
        is_outer,
    ):
        """
        LayerNorm fwd lowering rules
        """
        del out_dtype, scale_dtype, is_outer
        x_aval, scale_aval, gamma_aval, beta_aval = ctx.avals_in

        assert x_aval.dtype in [jnp.float32, jnp.float16, jnp.bfloat16]
        assert scale_aval is None or scale_aval.dtype == jnp.float32

        g_type = ir.RankedTensorType(gamma.type)
        g_shape = g_type.shape
        if norm_type == NVTE_Norm_Type.LayerNorm:
            assert gamma_aval.dtype == beta_aval.dtype
            b_type = ir.RankedTensorType(beta.type)
            b_shape = b_type.shape
            assert g_type == b_type
            assert g_shape == b_shape

        sm_margin = get_forward_sm_margin()
        return ffi.ffi_lowering(NormFwdPrimitive.name)(
            ctx,
            x,
            scale,
            gamma,
            beta,
            norm_type=norm_type.value,
            zero_centered_gamma=zero_centered_gamma,
            epsilon=epsilon,
            sm_margin=sm_margin,
            scaling_mode=scaling_mode.value,
            is_2x=is_2x,
        )

    @staticmethod
    def impl(
        x,
        scale,
        gamma,
        beta,
        norm_type,
        zero_centered_gamma,
        epsilon,
        out_dtype,
        scaling_mode,
        is_2x,
        scale_dtype,
        is_outer,
    ):
        """
        to describe implementation
        """
        del is_outer
        assert NormFwdPrimitive.inner_primitive is not None
        (
            out,
            colwise_out,
            scale_inv,
            colwise_scale_inv,
            updated_amax,
            mu,
            rsigma,
            _,
        ) = NormFwdPrimitive.inner_primitive.bind(
            x,
            scale,
            gamma,
            beta,
            norm_type=norm_type,
            zero_centered_gamma=zero_centered_gamma,
            epsilon=epsilon,
            out_dtype=out_dtype,
            scaling_mode=scaling_mode,
            is_2x=is_2x,
            scale_dtype=scale_dtype,
            is_outer=False,
        )
        rowwise_scale_inv_shape, colwise_scale_inv_shape = ScalingMode(
            scaling_mode
        ).get_scale_shape_2x(x.shape, is_padded=False)
        # slice out padding for mxfp8, noop for DelayedScaling
        scale_inv = scale_inv.flatten()[: reduce(operator.mul, rowwise_scale_inv_shape, 1)].reshape(
            rowwise_scale_inv_shape
        )
        if is_2x:
            colwise_scale_inv = colwise_scale_inv.flatten()[
                : reduce(operator.mul, colwise_scale_inv_shape, 1)
            ].reshape(colwise_scale_inv_shape)
        return (
            out,
            colwise_out,
            scale_inv,
            colwise_scale_inv,
            updated_amax,
            mu,
            rsigma,
        )  # Exclude wkspace

    @staticmethod
    def batcher(
        batched_args,
        batch_dims,
        *,
        norm_type,
        zero_centered_gamma,
        epsilon,
        out_dtype,
        scaling_mode,
        is_2x,
        scale_dtype,
        is_outer,
    ):
        """
        to describe batch rules for vmap
        """
        del is_outer
        check_valid_batch_dims(batch_dims)
        assert NormFwdPrimitive.outer_primitive is not None
        x, scale, gamma, beta = batched_args
        x_bdim, scale_bdim, _, _ = batch_dims

        out_bdims = (
            x_bdim,  # rowwise output
            scale_bdim,  # rowwise scale_inv
            x_bdim,  # colwise output
            scale_bdim,  # colwise scale_inv
            scale_bdim,  # amax
            x_bdim,  # mu
            x_bdim,  # rsigma
        )
        return (
            NormFwdPrimitive.outer_primitive.bind(
                scale,
                x,
                gamma,
                beta,
                norm_type=norm_type,
                zero_centered_gamma=zero_centered_gamma,
                epsilon=epsilon,
                out_dtype=out_dtype,
                scaling_mode=scaling_mode,
                is_2x=is_2x,
                scale_dtype=scale_dtype,
            ),
            out_bdims,
        )

    @staticmethod
    def infer_sharding_from_operands(
        norm_type,
        zero_centered_gamma,
        epsilon,
        out_dtype,
        scaling_mode,
        is_2x,
        scale_dtype,
        is_outer,
        mesh,
        arg_infos,
        result_infos,
    ):
        del zero_centered_gamma, epsilon, out_dtype, result_infos
        del scale_dtype, is_outer
        x_spec = get_padded_spec(arg_infos[0])
        scale_spec = get_padded_spec(arg_infos[1])
        out_spec = (*x_spec[:-1], None)
        if x_spec[-1] is not None:
            warnings.warn(
                f"Does not support to shard hidden dim in {NormFwdPrimitive.name}! "
                "Force to not shard the hidden dim, which might introduce extra collective ops, "
                "and hurt performance."
            )

        out_sharding = NamedSharding(mesh, PartitionSpec(*out_spec), desc="NormFwdPrimitive.out")
        colwise_out_spec = out_spec if is_2x else (None,)
        colwise_out_sharding = NamedSharding(
            mesh, PartitionSpec(*colwise_out_spec), desc="NormFwdPrimitive.colwise_out"
        )
        rsigma_sharding = NamedSharding(
            mesh, PartitionSpec(*x_spec[:-1]), desc="NormFwdPrimitive.rsigma"
        )
        mu_spec = x_spec[:-1] if norm_type == NVTE_Norm_Type.LayerNorm else (None,)
        mu_sharding = NamedSharding(mesh, PartitionSpec(*mu_spec), desc="NormFwdPrimitive.mu")

        scale_inv_spec = amax_spec = (None,)
        if scaling_mode == ScalingMode.DELAYED_TENSOR_SCALING.value:
            scale_inv_spec = amax_spec = scale_spec
        elif scaling_mode == ScalingMode.MXFP8_1D_SCALING.value:
            scale_inv_spec = out_spec

        scale_inv_sharding = NamedSharding(
            mesh, PartitionSpec(*scale_inv_spec), desc="NormFwdPrimitive.scale_inv"
        )
        amax_sharding = NamedSharding(mesh, PartitionSpec(*amax_spec), desc="NormFwdPrimitive.amax")
        output = (
            out_sharding,
            colwise_out_sharding,
            scale_inv_sharding,  # rowwise
            scale_inv_sharding,  # colwise
            amax_sharding,
            mu_sharding,
            rsigma_sharding,
        )
        return output

    @staticmethod
    def partition(
        norm_type,
        zero_centered_gamma,
        epsilon,
        out_dtype,
        scaling_mode,
        is_2x,
        scale_dtype,
        is_outer,
        mesh,
        arg_infos,
        result_infos,
    ):
        del result_infos, is_outer
        x_spec = get_padded_spec(arg_infos[0])
        scale_spec = get_padded_spec(arg_infos[1])
        g_spec = get_padded_spec(arg_infos[2])
        b_spec = get_padded_spec(arg_infos[3])
        out_spec = (*x_spec[:-1], None)

        if x_spec[-1] is not None:
            warnings.warn(
                f"Does not support to shard hidden dim in {NormFwdPrimitive.name}! "
                "Force to not shard the hidden dim, which might introduce extra collective ops, "
                "and hurt performance."
            )
        if g_spec[-1] is not None:
            warnings.warn(
                f"{NormFwdPrimitive.name} does not support sharding of parameter gamma "
                "Enforcing no sharding of parameters hidden dim! "
            )
        if b_spec[-1] is not None:
            warnings.warn(
                f"{NormFwdPrimitive.name} does not support sharding of parameter beta "
                "Enforcing no sharding of parameters hidden dim! "
            )

        out_sharding = NamedSharding(mesh, PartitionSpec(*out_spec), desc="NormFwdPrimitive.out")
        colwise_out_spec = out_spec if is_2x else (None,)
        colwise_out_sharding = NamedSharding(
            mesh, PartitionSpec(*colwise_out_spec), desc="NormFwdPrimitive.colwise_out"
        )
        rsigma_sharding = NamedSharding(
            mesh, PartitionSpec(*x_spec[:-1]), desc="NormFwdPrimitive.rsigma"
        )
        mu_spec = x_spec[:-1] if norm_type == NVTE_Norm_Type.LayerNorm else (None,)
        mu_sharding = NamedSharding(mesh, PartitionSpec(*mu_spec), desc="NormFwdPrimitive.mu")

        scale_inv_spec = amax_spec = (None,)
        if scaling_mode == ScalingMode.DELAYED_TENSOR_SCALING.value:
            scale_inv_spec = amax_spec = scale_spec
        elif scaling_mode == ScalingMode.MXFP8_1D_SCALING.value:
            scale_inv_spec = out_spec

        scale_inv_sharding = NamedSharding(
            mesh, PartitionSpec(*scale_inv_spec), desc="NormFwdPrimitive.scale_inv"
        )
        amax_sharding = NamedSharding(mesh, PartitionSpec(*amax_spec), desc="NormFwdPrimitive.amax")

        arg_shardings = list(arg_i.sharding for arg_i in arg_infos)
        # Enforce no sharding of hidden dim for x, gamma and beta
        arg_shardings[0] = NamedSharding(mesh, PartitionSpec(*out_spec), desc="NormFwdPrimitive.x")
        arg_shardings[2] = NamedSharding(
            mesh, PartitionSpec(*g_spec[:-1], None), desc="NormFwdPrimitive.gamma"
        )
        arg_shardings[3] = NamedSharding(
            mesh, PartitionSpec(*b_spec[:-1], None), desc="NormFwdPrimitive.beta"
        )
        arg_shardings = tuple(arg_shardings)
        out_shardings = (
            out_sharding,
            colwise_out_sharding,
            scale_inv_sharding,  # rowwise
            scale_inv_sharding,  # colwise
            amax_sharding,
            mu_sharding,
            rsigma_sharding,
        )

        def sharded_impl(x, scale, gamma, beta):
            # expect tp and dp giving same shape, or tp being same shape as global
            (
                local_x,
                local_colwise_x,
                local_scale_inv,
                local_colwise_scale_inv,
                local_amax,
                local_mu,
                local_rsigma,
            ) = NormFwdPrimitive.impl(
                x,
                scale,
                gamma,
                beta,
                norm_type=norm_type,
                zero_centered_gamma=zero_centered_gamma,
                epsilon=epsilon,
                out_dtype=out_dtype,
                scaling_mode=scaling_mode,
                is_2x=is_2x,
                scale_dtype=scale_dtype,
                is_outer=True,
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
                local_mu,
                local_rsigma,
            )

        return mesh, sharded_impl, out_shardings, arg_shardings

    @staticmethod
    def shardy_sharding_rule(
        norm_type,
        zero_centered_gamma,
        epsilon,
        out_dtype,
        scaling_mode,
        is_2x,
        scale_dtype,
        is_outer,
        mesh,
        value_types,
        result_types,
    ):
        del (
            zero_centered_gamma,
            epsilon,
            out_dtype,
            scale_dtype,
            is_outer,
            mesh,
            result_types,
        )

        prefix = "NormFwdPrimitive_"
        scale_rules = ScalingMode(scaling_mode).get_shardy_sharding_rules(
            len(value_types[0].shape), unique_var=prefix + "x", flatten_axis=-1
        )
        x_axes = scale_rules.input_spec

        out = x_axes
        colwise_out = out if is_2x else (prefix + "out_colwise",)
        rsigma = x_axes[:-1]
        mu = (prefix + "mu",) if norm_type == NVTE_Norm_Type.RMSNorm else rsigma
        amax = (prefix + "amax",)

        return SdyShardingRule(
            (x_axes, ("…1",), ("…2",), ("…3",)),
            (
                out,
                colwise_out,
                scale_rules.rowwise_rule,
                scale_rules.colwise_rule,
                amax,
                mu,
                rsigma,
            ),
        )


register_primitive(NormFwdPrimitive)


class NormBwdPrimitive(BasePrimitive):
    """
    Layer Normalization Backward Primitive
    """

    name = "te_norm_backward_ffi"
    multiple_results = True
    impl_static_args = (5, 6)  # norm_type, zero_centered_gamma
    inner_primitive = None
    outer_primitive = None

    @staticmethod
    def abstract(dz_aval, x_aval, mu_aval, rsigma_aval, gamma_aval, norm_type, zero_centered_gamma):
        """
        bwd inner primitive abstract
        """
        w_dtype = dtypes.canonicalize_dtype(gamma_aval.dtype)
        rsigma_dtype = dtypes.canonicalize_dtype(rsigma_aval.dtype)

        assert dtypes.canonicalize_dtype(dz_aval.dtype) == w_dtype
        assert dz_aval.shape == x_aval.shape

        if norm_type == NVTE_Norm_Type.LayerNorm:
            mu_dtype = dtypes.canonicalize_dtype(mu_aval.dtype)
            assert mu_aval.shape == rsigma_aval.shape == x_aval.shape[:-1]
            assert mu_dtype == rsigma_dtype == jnp.float32

        dx_aval = dz_aval
        dgamma_aval = dbeta_aval = gamma_aval
        if norm_type != NVTE_Norm_Type.LayerNorm:
            dbeta_aval = dbeta_aval.update(shape=(1,))

        (wkspace_info,) = transformer_engine_jax.get_norm_bwd_workspace_sizes(
            x_aval.size // gamma_aval.size,  # batch size
            gamma_aval.size,  # hidden size
            jax_dtype_to_te_dtype(x_aval.dtype),  # input te_dtype
            jax_dtype_to_te_dtype(gamma_aval.dtype),  # weight te_dtype
            norm_type,
            zero_centered_gamma,
            get_backward_sm_margin(),
        )
        wkspace_aval = dx_aval.update(
            shape=wkspace_info[0], dtype=te_dtype_to_jax_dtype(wkspace_info[1])
        )

        return (
            dx_aval,
            dgamma_aval,
            dbeta_aval,
            wkspace_aval,
        )

    @staticmethod
    def outer_abstract(*args, **kwargs):
        """
        LayerNorm bwd outer primitive abstract
        """
        dx_aval, dgamma_aval, dbeta_aval, _ = NormBwdPrimitive.abstract(*args, **kwargs)
        return dx_aval, dgamma_aval, dbeta_aval

    @staticmethod
    def lowering(ctx, dz, x, mu, rsigma, gamma, *, norm_type, zero_centered_gamma):
        """
        bwd lowering rules
        """
        g_type = ir.RankedTensorType(gamma.type)
        g_shape = g_type.shape
        b_type = ir.RankedTensorType(gamma.type)
        b_shape = b_type.shape
        assert g_type == b_type
        assert g_shape == b_shape

        sm_margin = get_backward_sm_margin()
        return ffi.ffi_lowering(NormBwdPrimitive.name)(
            ctx,
            dz,
            x,
            mu,
            rsigma,
            gamma,
            norm_type=norm_type.value,
            zero_centered_gamma=zero_centered_gamma,
            sm_margin=sm_margin,
        )

    @staticmethod
    def impl(dz, x, mu, rsigma, gamma, norm_type, zero_centered_gamma):
        assert NormBwdPrimitive.inner_primitive is not None
        dx, dgamma, dbeta, _ = NormBwdPrimitive.inner_primitive.bind(
            dz, x, mu, rsigma, gamma, norm_type=norm_type, zero_centered_gamma=zero_centered_gamma
        )
        return dx, dgamma, dbeta

    @staticmethod
    def batcher(batched_args, batch_dims, *, norm_type, zero_centered_gamma):
        check_valid_batch_dims(batch_dims)
        assert NormBwdPrimitive.outer_primitive is not None
        dz, x, mu, rsigma, gamma = batched_args
        _, x_bdim, _, _, gamma_bdim = batch_dims

        out_bdims = x_bdim, gamma_bdim, gamma_bdim
        return (
            NormBwdPrimitive.outer_primitive.bind(
                dz,
                x,
                mu,
                rsigma,
                gamma,
                norm_type=norm_type,
                zero_centered_gamma=zero_centered_gamma,
            ),
            out_bdims,
        )

    @staticmethod
    def infer_sharding_from_operands(norm_type, zero_centered_gamma, mesh, arg_infos, result_infos):
        del norm_type, zero_centered_gamma, result_infos
        x_spec = get_padded_spec(arg_infos[1])
        if x_spec[-1] is not None:
            warnings.warn(
                f"Does not support to shard hidden dim in {NormBwdPrimitive.name}! "
                "Force to not shard the hidden dim, which might introduce extra collective ops, "
                "and hurt performance."
            )
        g_b_spec = get_padded_spec(arg_infos[4])
        if g_b_spec[-1] is not None:
            warnings.warn(
                f"{NormBwdPrimitive.name} does not support sharding of gradients "
                "of gamma and beta of  "
                "Enforcing no sharding of parameters hidden dim! "
            )

        dx_sharding = NamedSharding(
            mesh, PartitionSpec(*x_spec[:-1], None), desc="NormBwdPrimitive.dx"
        )
        dgamma_sharding = dbeta_sharding = NamedSharding(
            mesh, PartitionSpec(None), desc="NormBwdPrimitive.dgamma"
        )
        return dx_sharding, dgamma_sharding, dbeta_sharding

    @staticmethod
    def partition(norm_type, zero_centered_gamma, mesh, arg_infos, result_infos):
        del result_infos
        x_spec = get_padded_spec(arg_infos[1])
        if x_spec[-1] is not None:
            warnings.warn(
                f"Does not support to shard hidden dim in {NormBwdPrimitive.name}! "
                "Force to not shard the hidden dim, which might introduce extra collective ops, "
                "and hurt performance."
            )
        g_b_spec = get_padded_spec(arg_infos[4])
        if g_b_spec[-1] is not None:
            warnings.warn(
                f"{NormBwdPrimitive.name} does not support sharding of gradients "
                "of gamma and beta of  "
                "Enforcing no sharding of parameters hidden dim! "
            )

        dx_sharding = NamedSharding(
            mesh, PartitionSpec(*x_spec[:-1], None), desc="NormBwdPrimitive.dx"
        )
        dgamma_sharding = dbeta_sharding = NamedSharding(
            mesh, PartitionSpec(None), desc="NormBwdPrimitive.dgamma"
        )
        out_shardings = dx_sharding, dgamma_sharding, dbeta_sharding
        x_shardings = (dx_sharding,) * 2  # dz and x should have the same sharding.

        rsigma_sharding = NamedSharding(
            mesh, PartitionSpec(*x_spec[:-1]), desc="NormBwdPrimitive.rsigma"
        )
        mu_sharding = rsigma_sharding.duplicate_with_new_description("NormBwdPrimitive.mu")
        if norm_type == NVTE_Norm_Type.RMSNorm:
            mu_sharding = NamedSharding(mesh, PartitionSpec(None), desc="NormBwdPrimitive.mu")
        arg_shardings = (
            *x_shardings,
            mu_sharding,
            rsigma_sharding,
            NamedSharding(mesh, PartitionSpec(None), desc="NormBwdPrimitive.gamma"),
        )

        def sharded_impl(dz, x, mu, rsigma, gamma):
            local_dx, local_dgamma, local_dbeta = NormBwdPrimitive.impl(
                dz,
                x,
                mu,
                rsigma,
                gamma,
                norm_type=norm_type,
                zero_centered_gamma=zero_centered_gamma,
            )
            global_dgamma = all_reduce_sum_along_dp_fsdp(local_dgamma, mesh)
            if norm_type == NVTE_Norm_Type.LayerNorm:
                global_dbeta = all_reduce_sum_along_dp_fsdp(local_dbeta, mesh)
            else:
                global_dbeta = local_dbeta
            return local_dx, global_dgamma, global_dbeta

        return mesh, sharded_impl, out_shardings, arg_shardings

    @staticmethod
    def shardy_sharding_rule(*args):
        del args
        return "...0, ...1 i, ...2, ...3, ...4 -> ...1 j, k, l"


register_primitive(NormBwdPrimitive)


def _jax_layernorm(x, gamma, beta, zero_centered_gamma, epsilon, quantizer=None):
    """
    JAX native layernorm implementation
    """
    x_ = jnp.asarray(x, jnp.float32)
    if not is_norm_zero_centered_gamma_in_weight_dtype(
        quantizer.scaling_mode if quantizer else ScalingMode.NO_SCALING
    ):
        gamma = gamma.astype(jnp.float32)
    mean = jnp.mean(x_, axis=-1, keepdims=True)
    var = jnp.mean(jnp.square(x_ - mean), axis=-1, keepdims=True)
    rsigma = jax.lax.rsqrt(var + epsilon)
    normed_input = (x_ - mean) * rsigma
    if zero_centered_gamma:
        gamma += 1.0
    output = normed_input * gamma + beta

    if quantizer:
        if quantizer.scaling_mode == ScalingMode.CURRENT_TENSOR_SCALING:
            output = output.astype(x.dtype)
        ln_out = quantizer.quantize(output, dq_dtype=x.dtype)
    else:
        ln_out = jnp.asarray(output).astype(x.dtype)
        ln_out = NoScaleTensor(data=ln_out, amax=None)

    return ln_out, jnp.squeeze(mean, axis=-1), jnp.squeeze(rsigma, axis=-1)


def _jax_rmsnorm(x, gamma, zero_centered_gamma, epsilon, quantizer=None):
    """
    JAX native rmsnorm implementation
    """
    x_ = jnp.asarray(x, jnp.float32)
    if not is_norm_zero_centered_gamma_in_weight_dtype(
        quantizer.scaling_mode if quantizer else ScalingMode.NO_SCALING
    ):
        gamma = gamma.astype(jnp.float32)
    var = jnp.mean(jnp.square(x_), axis=-1, keepdims=True)
    rsigma = jax.lax.rsqrt(var + epsilon)
    normed_input = x_ * rsigma
    if zero_centered_gamma:
        gamma += 1.0
    output = normed_input * gamma

    if quantizer:
        if quantizer.scaling_mode == ScalingMode.CURRENT_TENSOR_SCALING:
            output = output.astype(x.dtype)
        ln_out = quantizer.quantize(output, dq_dtype=x.dtype)
    else:
        ln_out = jnp.asarray(output).astype(x.dtype)
        ln_out = NoScaleTensor(data=ln_out, amax=None)

    return ln_out, jnp.squeeze(rsigma, axis=-1)


def layernorm_fwd(
    x: jnp.ndarray,
    gamma: jnp.ndarray,
    beta: jnp.ndarray,
    zero_centered_gamma: bool,
    epsilon: float,
    quantizer: Optional[Quantizer],
) -> tuple[Union[jnp.ndarray, ScaledTensor], jnp.ndarray, jnp.ndarray]:
    """Layer normalization forward pass with optional quantization.

    Args:
        x: Input tensor to be normalized.
            Shape: (..., K) where K is the hidden size.
        gamma: Scale parameter for normalization.
            Shape: (K,)
        beta: Bias parameter for normalization.
            Shape: (K,)
        zero_centered_gamma: If True, gamma is zero-centered.
        epsilon: Small constant for numerical stability.
        quantizer: Optional quantizer for FP8 quantization of the output.

    Returns:
        A tuple containing:
        - If quantizer is None:
            The normalized input tensor. Shape: (..., K)
          If quantizer is provided:
            A ScaledTensor containing the quantized normalized input.
        - Mean of the input tensor. Shape: (..., 1)
        - Reciprocal of the standard deviation of the input tensor. Shape: (..., 1)
    """
    if not NormFwdPrimitive.enabled():
        return _jax_layernorm(x, gamma, beta, zero_centered_gamma, epsilon, quantizer)

    # TE/common does not support normalization with colwise only quantization yet
    if quantizer is not None and quantizer.q_layout == QuantizeLayout.COLWISE:
        return _jax_layernorm(x, gamma, beta, zero_centered_gamma, epsilon, quantizer)

    scale = (
        quantizer.scale
        if isinstance(quantizer, DelayedScaleQuantizer)
        else jnp.ones((1,), dtype=jnp.float32)
    )
    if quantizer is None:
        output, _, _, _, _, mu, rsigma = NormFwdPrimitive.outer_primitive.bind(
            x,
            scale,
            gamma,
            beta,
            norm_type=NVTE_Norm_Type.LayerNorm,
            zero_centered_gamma=zero_centered_gamma,
            epsilon=epsilon,
            out_dtype=x.dtype,
            scaling_mode=ScalingMode.NO_SCALING.value,
            is_2x=False,
            scale_dtype=jnp.float32,
            is_outer=True,
        )
        return NoScaleTensor(data=output, amax=None), mu, rsigma

    if (
        quantizer.scaling_mode == ScalingMode.MXFP8_1D_SCALING
        and get_cudnn_version() < FUSED_MXFP8_NORM_CUDNN_MIN_VERSION
    ):
        out, mu, rsigma = layernorm_fwd(
            x, gamma, beta, zero_centered_gamma, epsilon, quantizer=None
        )
        out, _ = _quantize_dbias_impl(out, quantizer)
        return out, mu, rsigma

    if quantizer.scaling_mode == ScalingMode.CURRENT_TENSOR_SCALING:
        # Current scaling does not support fused operations. Perform norm in higher precision then quantize after.
        out, mu, rsigma = layernorm_fwd(
            x=x,
            gamma=gamma,
            beta=beta,
            zero_centered_gamma=zero_centered_gamma,
            epsilon=epsilon,
            quantizer=None,
        )
        out, _ = _quantize_dbias_impl(out, is_dbias=False, quantizer=quantizer, dq_dtype=x.dtype)
        return out, mu, rsigma

    is_2x2x = quantizer.is_2x2x()
    # TE/common normalization doesn't support 2x delayed scaling
    if quantizer.is_2x2x() and quantizer.scaling_mode.is_tensor_scaling():
        is_2x2x = False
    (
        rowwise_casted_output,
        colwise_casted_output,
        rowwise_scale_inv,
        colwise_scale_inv,
        updated_amax,
        mu,
        rsigma,
    ) = NormFwdPrimitive.outer_primitive.bind(
        x,
        scale,
        gamma,
        beta,
        norm_type=NVTE_Norm_Type.LayerNorm,
        zero_centered_gamma=zero_centered_gamma,
        epsilon=epsilon,
        out_dtype=quantizer.q_dtype,
        scaling_mode=quantizer.scaling_mode.value,
        is_2x=is_2x2x,
        scale_dtype=quantizer.get_scale_dtype(),
        is_outer=True,
    )
    quantizer.update(updated_amax)

    # TE/common Norm doesn't support 2x delayed scaling so do 1x then JAX transpose
    if quantizer.is_2x2x() and quantizer.scaling_mode.is_tensor_scaling():
        colwise_casted_output = jnp.transpose(
            rowwise_casted_output, (-1, *range(rowwise_casted_output.ndim - 1))
        )
        colwise_scale_inv = rowwise_scale_inv

    # cuDNN MXFP8 Norm does not support padding but we enforced padded scale inputs for nvte APIs.
    # So here we need to slice out the zero tail and reshape it to the unpadded scale shape.
    # The ScaledTensorFactory takes care of padding when creating the ScaledTensor
    if quantizer.scaling_mode == ScalingMode.MXFP8_1D_SCALING:
        rowwise_unpadded_shape, colwise_unpadded_shape = quantizer.get_scale_shapes(
            x.shape, is_padded=False
        )
        rowwise_scale_inv = rowwise_scale_inv.flatten()[
            : reduce(operator.mul, rowwise_unpadded_shape)
        ].reshape(rowwise_unpadded_shape)
        colwise_scale_inv = colwise_scale_inv.flatten()[
            : reduce(operator.mul, colwise_unpadded_shape)
        ].reshape(colwise_unpadded_shape)

    scaled_tensor = ScaledTensorFactory.create(
        data=rowwise_casted_output,
        scale_inv=rowwise_scale_inv,
        colwise_data=colwise_casted_output,
        colwise_scale_inv=colwise_scale_inv,
        scaling_mode=quantizer.scaling_mode,
        dq_dtype=x.dtype,
        q_layout=quantizer.q_layout,
        data_layout=quantizer.get_data_layout(),
    )

    return scaled_tensor, mu, rsigma


def layernorm_bwd(
    dz: jnp.ndarray,
    x: jnp.ndarray,
    mu: jnp.ndarray,
    rsigma: jnp.ndarray,
    gamma: jnp.ndarray,
    beta: jnp.ndarray,
    zero_centered_gamma: bool,
    epsilon: float,
):
    """Layer normalization backward pass.

    Args:
        dz: Gradient of the output with respect to the normalized output.
            Shape: (..., K) where K is the hidden size.
        x: Input tensor that was normalized in the forward pass.
            Shape: (..., K)
        mu: Mean of the input tensor from the forward pass.
            Shape: (..., 1)
        rsigma: Reciprocal of the standard deviation from the forward pass.
            Shape: (..., 1)
        gamma: Scale parameter for normalization.
            Shape: (K,)
        beta: Bias parameter for normalization.
            Shape: (K,)
        zero_centered_gamma: If True, gamma is zero-centered.
        epsilon: Small constant for numerical stability.

    Returns:
        A tuple containing:
        - Gradient of the input tensor.
            Shape: (..., K)
        - Gradient of the scale parameter (gamma).
            Shape: (K,)
        - Gradient of the bias parameter (beta).
            Shape: (K,)
    """
    if not NormBwdPrimitive.enabled():
        _, vjp_func = jax.vjp(
            partial(_jax_layernorm, zero_centered_gamma=zero_centered_gamma, epsilon=epsilon),
            x,
            gamma,
            beta,
        )
        mu_empty = jnp.zeros(mu.shape, mu.dtype)
        rsigma_empty = jnp.zeros(rsigma.shape, rsigma.dtype)
        return vjp_func((NoScaleTensor(data=dz, amax=None), mu_empty, rsigma_empty))
    return NormBwdPrimitive.outer_primitive.bind(
        dz,
        x,
        mu,
        rsigma,
        gamma,
        norm_type=NVTE_Norm_Type.LayerNorm,
        zero_centered_gamma=zero_centered_gamma,
    )


def rmsnorm_fwd(
    x: jnp.ndarray,
    gamma: jnp.ndarray,
    zero_centered_gamma: bool,
    epsilon: float,
    quantizer: Optional[Quantizer],
) -> tuple[Union[jnp.ndarray, ScaledTensor], jnp.ndarray]:
    """Root mean square normalization forward pass with optional quantization.

    Args:
        x: Input tensor to be normalized.
            Shape: (..., K) where K is the hidden size.
        gamma: Scale parameter for normalization.
            Shape: (K,)
        zero_centered_gamma: If True, gamma is zero-centered.
        epsilon: Small constant for numerical stability.
        quantizer: Optional quantizer for FP8 quantization of the output.

    Returns:
        A tuple containing:
        - If quantizer is None:
            The normalized input tensor.
            Shape: (..., K)
          If quantizer is provided:
            A ScaledTensor containing the quantized normalized input.
        - Reciprocal of the root mean square of the input tensor.
            Shape: (..., 1)
    """
    if not NormFwdPrimitive.enabled():
        return _jax_rmsnorm(x, gamma, zero_centered_gamma, epsilon, quantizer)

    # TE/common does not support normalization with colwise only quantization yet
    if quantizer is not None and quantizer.q_layout == QuantizeLayout.COLWISE:
        return _jax_rmsnorm(x, gamma, zero_centered_gamma, epsilon, quantizer)

    scale = (
        quantizer.scale
        if isinstance(quantizer, DelayedScaleQuantizer)
        else jnp.ones((1,), dtype=jnp.float32)
    )
    beta = jnp.ones((1,), dtype=jnp.float32)

    if quantizer is None:
        output, _, _, _, _, _, rsigma = NormFwdPrimitive.outer_primitive.bind(
            x,
            scale,
            gamma,
            beta,
            norm_type=NVTE_Norm_Type.RMSNorm,
            zero_centered_gamma=zero_centered_gamma,
            epsilon=epsilon,
            out_dtype=x.dtype,
            scaling_mode=ScalingMode.NO_SCALING.value,
            is_2x=False,
            scale_dtype=jnp.float32,
            is_outer=True,
        )
        return NoScaleTensor(data=output, amax=None), rsigma

    if (
        quantizer.scaling_mode == ScalingMode.MXFP8_1D_SCALING
        and get_cudnn_version() < FUSED_MXFP8_NORM_CUDNN_MIN_VERSION
    ):
        out, rsigma = rmsnorm_fwd(x, gamma, zero_centered_gamma, epsilon, quantizer=None)
        out, _ = _quantize_dbias_impl(out.data, quantizer)
        return out, rsigma

    if quantizer.scaling_mode == ScalingMode.CURRENT_TENSOR_SCALING:
        # Current scaling does not support fused operations. Perform norm in higher precision then quantize after.
        out, rsigma = rmsnorm_fwd(
            x=x,
            gamma=gamma,
            zero_centered_gamma=zero_centered_gamma,
            epsilon=epsilon,
            quantizer=None,
        )
        out, _ = _quantize_dbias_impl(
            out.data, is_dbias=False, quantizer=quantizer, dq_dtype=x.dtype
        )
        return out, rsigma

    is_2x2x = quantizer.is_2x2x()
    # TE/common normalization doesn't support 2x delayed scaling
    if quantizer.is_2x2x() and quantizer.scaling_mode.is_tensor_scaling():
        is_2x2x = False
    (
        rowwise_casted_output,
        colwise_casted_output,
        rowwise_scale_inv,
        colwise_scale_inv,
        updated_amax,
        _,
        rsigma,
    ) = NormFwdPrimitive.outer_primitive.bind(
        x,
        scale,
        gamma,
        beta,
        norm_type=NVTE_Norm_Type.RMSNorm,
        zero_centered_gamma=zero_centered_gamma,
        epsilon=epsilon,
        out_dtype=quantizer.q_dtype,
        scaling_mode=quantizer.scaling_mode.value,
        is_2x=is_2x2x,
        scale_dtype=quantizer.get_scale_dtype(),
        is_outer=True,
    )
    quantizer.update(updated_amax)

    # TE/common Norm doesn't support 2x delayed scaling so do 1x then JAX transpose
    if quantizer.is_2x2x() and quantizer.scaling_mode.is_tensor_scaling():
        colwise_casted_output = jnp.transpose(
            rowwise_casted_output, (-1, *range(rowwise_casted_output.ndim - 1))
        )
        colwise_scale_inv = rowwise_scale_inv

    # cuDNN MXFP8 Norm does not support padding but we enforced padded scale inputs for nvte APIs.
    # So here we need to slice out the zero tail and reshape it to the unpadded scale shape.
    # The ScaledTensorFactory takes care of padding when creating the ScaledTensor
    if quantizer.scaling_mode == ScalingMode.MXFP8_1D_SCALING:
        rowwise_unpadded_shape, colwise_unpadded_shape = quantizer.get_scale_shapes(
            x.shape, is_padded=False
        )
        rowwise_scale_inv = rowwise_scale_inv.flatten()[
            : reduce(operator.mul, rowwise_unpadded_shape)
        ].reshape(rowwise_unpadded_shape)
        colwise_scale_inv = colwise_scale_inv.flatten()[
            : reduce(operator.mul, colwise_unpadded_shape)
        ].reshape(colwise_unpadded_shape)

    scaled_tensor = ScaledTensorFactory.create(
        data=rowwise_casted_output,
        scale_inv=rowwise_scale_inv,
        colwise_data=colwise_casted_output,
        colwise_scale_inv=colwise_scale_inv,
        scaling_mode=quantizer.scaling_mode,
        dq_dtype=x.dtype,
        q_layout=quantizer.q_layout,
        data_layout=quantizer.get_data_layout(),
    )

    return scaled_tensor, rsigma


def rmsnorm_bwd(
    dz: jnp.ndarray,
    x: jnp.ndarray,
    rsigma: jnp.ndarray,
    gamma: jnp.ndarray,
    zero_centered_gamma: bool,
    epsilon: float,
):
    """Root mean square normalization backward pass.

    Args:
        dz: Gradient of the output with respect to the normalized output.
            Shape: (..., K) where K is the hidden size.
        x: Input tensor that was normalized in the forward pass.
            Shape: (..., K)
        rsigma: Reciprocal of the root mean square from the forward pass.
            Shape: (..., 1)
        gamma: Scale parameter for normalization.
            Shape: (K,)
        zero_centered_gamma: If True, gamma is zero-centered.
        epsilon: Small constant for numerical stability.

    Returns:
        A tuple containing:
        - Gradient of the input tensor.
            Shape: (..., K)
        - Gradient of the scale parameter (gamma).
            Shape: (K,)
    """
    if not NormBwdPrimitive.enabled():
        _, vjp_func = jax.vjp(
            partial(_jax_rmsnorm, zero_centered_gamma=zero_centered_gamma, epsilon=epsilon),
            x,
            gamma,
        )
        rsigma_empty = jnp.zeros(rsigma.shape, rsigma.dtype)
        return vjp_func((NoScaleTensor(data=dz, amax=None), rsigma_empty))
    mu = jnp.empty(())
    dx, dgamma, _ = NormBwdPrimitive.outer_primitive.bind(
        dz,
        x,
        mu,
        rsigma,
        gamma,
        norm_type=NVTE_Norm_Type.RMSNorm,
        zero_centered_gamma=zero_centered_gamma,
    )
    return (dx, dgamma)


def normalization_fwd(
    x: jnp.ndarray,
    gamma: jnp.ndarray,
    beta: jnp.ndarray,
    zero_centered_gamma: bool,
    epsilon: float,
    norm_type: str,
    quantizer: Optional[Quantizer],
):
    """Common wrapper for normalization forward pass.

    Args:
        x: Input tensor to be normalized.
            Shape: (..., K) where K is the hidden size.
        gamma: Scale parameter for normalization.
            Shape: (K,)
        beta: Bias parameter for normalization.
            Shape: (K,)
        zero_centered_gamma: If True, gamma is zero-centered.
        epsilon: Small constant for numerical stability.
        norm_type: Type of normalization to apply. Must be one of:
            - 'layernorm': Layer normalization
            - 'rmsnorm': Root mean square normalization
        quantizer: Optional quantizer for FP8 quantization of the output.

    Returns:
        A tuple containing:
        - If quantizer is None:
            The normalized input tensor.
            Shape: (..., K)
          If quantizer is provided:
            A ScaledTensor containing the quantized normalized input.
        - Mean of the input tensor (None for RMSNorm).
            Shape: (..., 1)
        - Reciprocal of the standard deviation (or root mean square for RMSNorm).
            Shape: (..., 1)

    Note:
        zero_centered_gamma is not supported if norm_type is 'rmsnorm'.
    """
    if norm_type == "layernorm":
        output, mu, rsigma = layernorm_fwd(x, gamma, beta, zero_centered_gamma, epsilon, quantizer)
    elif norm_type == "rmsnorm":
        assert (
            not zero_centered_gamma
        ), "zero_centered_gamma is not supported if norm_type is 'rmsnorm'"
        output, rsigma = rmsnorm_fwd(x, gamma, zero_centered_gamma, epsilon, quantizer)
        mu = None
    else:
        raise ValueError(f"{norm_type=} is not supported.")

    return output, mu, rsigma


def normalization_bwd(
    dz: jnp.ndarray,
    x: jnp.ndarray,
    mu: jnp.ndarray,
    rsigma: jnp.ndarray,
    gamma: jnp.ndarray,
    beta: jnp.ndarray,
    zero_centered_gamma: bool,
    epsilon: float,
    norm_type: str,
):
    """Common wrapper for normalization backward pass.

    Args:
        dz: Gradient of the output with respect to the normalized output.
            Shape: (..., K) where K is the hidden size.
        x: Input tensor that was normalized in the forward pass.
            Shape: (..., K)
        mu: Mean of the input tensor from the forward pass (None for RMSNorm).
            Shape: (..., 1)
        rsigma: Reciprocal of the standard deviation (or root mean square) from the forward pass.
            Shape: (..., 1)
        gamma: Scale parameter for normalization.
            Shape: (K,)
        beta: Bias parameter for normalization.
            Shape: (K,)
        zero_centered_gamma: If True, gamma is zero-centered.
        epsilon: Small constant for numerical stability.
        norm_type: Type of normalization used in the forward pass. Must be one of:
            - 'layernorm': Layer normalization
            - 'rmsnorm': Root mean square normalization

    Returns:
        A tuple containing:
        - Gradient of the input tensor.
            Shape: (..., K)
        - Gradient of the scale parameter (gamma).
            Shape: (K,)
        - Gradient of the bias parameter (beta) (None for RMSNorm).
            Shape: (K,)

    Note:
        zero_centered_gamma is not supported if norm_type is 'rmsnorm'.
    """
    if norm_type == "layernorm":
        dx, dgamma, dbeta = layernorm_bwd(
            dz, x, mu, rsigma, gamma, beta, zero_centered_gamma, epsilon
        )
    elif norm_type == "rmsnorm":
        assert (
            not zero_centered_gamma
        ), "zero_centered_gamma is not supported if norm_type is 'rmsnorm'"
        dx, dgamma = rmsnorm_bwd(dz, x, rsigma, gamma, zero_centered_gamma, epsilon)
        dbeta = None
    else:
        raise ValueError(f"{norm_type=} is not supported.")

    return dx, dgamma, dbeta
