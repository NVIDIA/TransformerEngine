# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""JAX/TE custom ops for normalization"""
from functools import partial, reduce
import operator
import os
import warnings

import jax.numpy as jnp
from jax import core, dtypes
from jax.interpreters import mlir
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
    te_dtype_to_jax_dtype,
)
from ..sharding import all_reduce_max_along_all_axes_except_PP, all_reduce_sum_along_dp_fsdp


__all__ = [
    "layernorm_fwd",
    "layernorm_bwd",
    "rmsnorm_fwd",
    "rmsnorm_bwd",
    "layernorm_fwd_fp8",
    "rmsnorm_fwd_fp8",
]


class LayerNormFwdPrimitive(BasePrimitive):
    """
    Layer Normalization Forward Primitive
    """

    name = "te_layernorm_forward"
    multiple_results = True
    impl_static_args = (3, 4)  # zero_centered_gamma, epsilon
    inner_primitive = None
    outer_primitive = None

    @staticmethod
    def abstract(x_aval, gamma_aval, beta_aval, **kwargs):
        """
        LayerNorm fwd inner primitive abstract
        """
        x_dtype = dtypes.canonicalize_dtype(x_aval.dtype)
        assert x_dtype in [jnp.float32, jnp.float16, jnp.bfloat16]

        mu_rsigama_dtype = jnp.float32

        out_aval = core.raise_to_shaped(x_aval)
        mu_aval = rsigma_aval = out_aval.update(shape=out_aval.shape[:-1], dtype=mu_rsigama_dtype)

        assert gamma_aval.size == beta_aval.size
        hidden_size = gamma_aval.size
        assert x_aval.size % hidden_size == 0

        wkspace_info, barrier_info = transformer_engine_jax.get_layernorm_fwd_workspace_sizes(
            x_aval.size // hidden_size,  # batch size
            hidden_size,
            jax_dtype_to_te_dtype(x_aval.dtype),  # in te_dtype
            jax_dtype_to_te_dtype(gamma_aval.dtype),  # weight te_dtype
            jax_dtype_to_te_dtype(x_aval.dtype),  # out te_dtype (same as input for Fp16/Bf16)
            True,
            kwargs["zero_centered_gamma"],
            kwargs["epsilon"],
        )
        wkspace_aval = out_aval.update(
            shape=wkspace_info[0], dtype=te_dtype_to_jax_dtype(wkspace_info[1])
        )
        barrier_aval = out_aval.update(
            shape=barrier_info[0], dtype=te_dtype_to_jax_dtype(barrier_info[1])
        )

        return out_aval, mu_aval, rsigma_aval, wkspace_aval, barrier_aval

    @staticmethod
    def outer_abstract(*args, **kwargs):
        """
        LayerNorm fwd outer primitive abstract
        """
        out_aval, mu_aval, rsigma_aval, _, _ = LayerNormFwdPrimitive.abstract(*args, **kwargs)
        return out_aval, mu_aval, rsigma_aval

    @staticmethod
    def lowering(ctx, x, gamma, beta, *, zero_centered_gamma, epsilon):
        """
        LayerNorm fwd lowering rules
        """
        x_aval, gamma_aval, beta_aval = ctx.avals_in
        assert gamma_aval.dtype == beta_aval.dtype
        x_type = ir.RankedTensorType(x.type)
        x_shape = x_type.shape
        g_type = ir.RankedTensorType(gamma.type)
        g_shape = g_type.shape
        b_type = ir.RankedTensorType(beta.type)
        b_shape = b_type.shape

        assert g_type == b_type
        assert g_shape == b_shape

        # Output shape is same as the input shape, but the output type is same as the weight type.
        # See ln_api.cpp
        output_type = g_type.element_type
        ir_mu_dtype = ir.F32Type.get()
        ir_rsigma_dtype = ir.F32Type.get()

        out_shape = x_shape
        hidden_size = reduce(operator.mul, g_shape)
        batch_shape = out_shape[:-1]
        batch_size = reduce(operator.mul, x_shape) // hidden_size

        wkspace_aval, barrier_aval = ctx.avals_out[-2:]

        out_types = [
            ir.RankedTensorType.get(out_shape, output_type),
            ir.RankedTensorType.get(batch_shape, ir_mu_dtype),
            ir.RankedTensorType.get(batch_shape, ir_rsigma_dtype),
            ir.RankedTensorType.get(wkspace_aval.shape, jax_dtype_to_ir_dtype(wkspace_aval.dtype)),
            ir.RankedTensorType.get(barrier_aval.shape, jax_dtype_to_ir_dtype(barrier_aval.dtype)),
        ]
        operands = [x, gamma, beta]
        operand_shapes = [x_shape, g_shape, b_shape]
        args = CustomCallArgsWrapper(out_types, operands, operand_shapes)

        sm_margin = int(os.getenv("NVTE_FWD_LAYERNORM_SM_MARGIN", "0"))

        opaque = transformer_engine_jax.pack_norm_descriptor(
            batch_size,
            hidden_size,
            wkspace_aval.size,
            barrier_aval.size,
            (0,),  # no dgamma_part in FWD pass
            (0,),  # no dbeta_part in BWD pass
            jax_dtype_to_te_dtype(x_aval.dtype),
            jax_dtype_to_te_dtype(gamma_aval.dtype),
            jax_dtype_to_te_dtype(wkspace_aval.dtype),
            jax_dtype_to_te_dtype(barrier_aval.dtype),
            TEDType.kByte,  # dummy dgamma_part te_dtype
            TEDType.kByte,  # dummy dbeta_part te_dtype
            zero_centered_gamma,
            epsilon,
            sm_margin,
        )

        out = custom_caller(LayerNormFwdPrimitive.name, args, opaque, False)

        return out

    @staticmethod
    def impl(x, gamma, beta, zero_centered_gamma, epsilon):
        """
        to describe implementation
        """
        assert LayerNormFwdPrimitive.inner_primitive is not None
        out, mu, rsigma, _, _ = LayerNormFwdPrimitive.inner_primitive.bind(
            x, gamma, beta, zero_centered_gamma=zero_centered_gamma, epsilon=epsilon
        )
        return out, mu, rsigma

    @staticmethod
    def batcher(batched_args, batch_dims, *, zero_centered_gamma, epsilon):
        """
        to describe batch rules for vmap
        """
        check_valid_batch_dims(batch_dims)
        assert LayerNormFwdPrimitive.outer_primitive is not None
        x, gamma, beta = batched_args
        x_bdim, _, _ = batch_dims

        out_bdims = x_bdim, x_bdim, x_bdim
        return (
            LayerNormFwdPrimitive.outer_primitive.bind(
                x, gamma, beta, zero_centered_gamma=zero_centered_gamma, epsilon=epsilon
            ),
            out_bdims,
        )

    @staticmethod
    def infer_sharding_from_operands(zero_centered_gamma, epsilon, mesh, arg_infos, result_infos):
        del zero_centered_gamma, epsilon, result_infos
        x_spec = get_padded_spec(arg_infos[0])
        if x_spec[-1] is not None:
            warnings.warn(
                f"Does not support to shard hidden dim in {LayerNormFwdPrimitive.name}! "
                "Force to not shard the hidden dim, which might introduce extra collective ops, "
                "and hurt performance."
            )
        out_sharding = NamedSharding(mesh, PartitionSpec(*x_spec[:-1], None))
        mu_sharding = rsigma_sharding = NamedSharding(mesh, PartitionSpec(*x_spec[:-1]))
        return (out_sharding, mu_sharding, rsigma_sharding)

    @staticmethod
    def partition(zero_centered_gamma, epsilon, mesh, arg_infos, result_infos):
        del result_infos
        x_spec, g_spec, b_spec = map(get_padded_spec, arg_infos)
        if x_spec[-1] is not None:
            warnings.warn(
                f"Does not support to shard hidden dim in {LayerNormFwdPrimitive.name}! "
                "Force to not shard the hidden dim, which might introduce extra collective ops, "
                "and hurt performance."
            )
        if g_spec[-1] is not None:
            warnings.warn(
                f"{LayerNormFwdPrimitive.name} does not support sharding of parameter gamma "
                "Enforcing no sharding of parameters hidden dim! "
            )
        if b_spec[-1] is not None:
            warnings.warn(
                f"{LayerNormFwdPrimitive.name} does not support sharding of parameter beta "
                "Enforcing no sharding of parameters hidden dim! "
            )

        x_sharding = NamedSharding(mesh, PartitionSpec(*x_spec[:-1], None))
        g_sharding = NamedSharding(mesh, PartitionSpec(None))
        b_sharding = NamedSharding(mesh, PartitionSpec(None))
        out_sharding = x_sharding
        mu_sharding = rsigma_sharding = NamedSharding(mesh, PartitionSpec(*x_spec[:-1]))

        arg_shardings = (x_sharding, g_sharding, b_sharding)
        out_shardings = (out_sharding, mu_sharding, rsigma_sharding)
        impl = partial(
            LayerNormFwdPrimitive.impl, zero_centered_gamma=zero_centered_gamma, epsilon=epsilon
        )
        return mesh, impl, out_shardings, arg_shardings


register_primitive(LayerNormFwdPrimitive)


def layernorm_fwd(
    x: jnp.ndarray, gamma: jnp.ndarray, beta: jnp.ndarray, zero_centered_gamma: bool, epsilon: float
):
    """
    Wrapper for TE layernorm fwd
    """
    return LayerNormFwdPrimitive.outer_primitive.bind(
        x, gamma, beta, zero_centered_gamma=zero_centered_gamma, epsilon=epsilon
    )


class LayerNormBwdPrimitive(BasePrimitive):
    """
    Layer Normalization Backward Primitive
    """

    name = "te_layernorm_backward"
    multiple_results = True
    impl_static_args = (5, 6)  # zero_centered_gamma, epsilon
    inner_primitive = None
    outer_primitive = None

    @staticmethod
    def abstract(dz_aval, x_aval, mu_aval, rsigma_aval, gamma_aval, **kwargs):
        """
        Layernorm bwd inner primitive abstract
        """
        w_dtype = dtypes.canonicalize_dtype(gamma_aval.dtype)
        mu_dtype = dtypes.canonicalize_dtype(mu_aval.dtype)
        rsigma_dtype = dtypes.canonicalize_dtype(rsigma_aval.dtype)

        assert dtypes.canonicalize_dtype(dz_aval.dtype) == w_dtype
        assert dz_aval.shape == x_aval.shape
        assert mu_aval.shape == rsigma_aval.shape == x_aval.shape[:-1]
        assert mu_dtype == rsigma_dtype == jnp.float32

        dx_aval = core.raise_to_shaped(dz_aval)
        dgamma_aval = dbeta_aval = core.raise_to_shaped(gamma_aval)

        wkspace_info, barrier_info, dgamma_part_info, dbeta_part_info = (
            transformer_engine_jax.get_layernorm_bwd_workspace_sizes(
                x_aval.size // gamma_aval.size,  # batch size
                gamma_aval.size,  # hidden size
                jax_dtype_to_te_dtype(x_aval.dtype),  # input te_dtype
                jax_dtype_to_te_dtype(gamma_aval.dtype),  # weight te_dtype
                True,
                kwargs["zero_centered_gamma"],
                kwargs["epsilon"],
            )
        )
        wkspace_aval = dx_aval.update(
            shape=wkspace_info[0], dtype=te_dtype_to_jax_dtype(wkspace_info[1])
        )
        barrier_aval = dx_aval.update(
            shape=barrier_info[0], dtype=te_dtype_to_jax_dtype(barrier_info[1])
        )
        dgamma_part_aval = dgamma_aval.update(
            shape=dgamma_part_info[0], dtype=te_dtype_to_jax_dtype(dgamma_part_info[1])
        )
        dbeta_part_aval = dbeta_aval.update(
            shape=dbeta_part_info[0], dtype=te_dtype_to_jax_dtype(dbeta_part_info[1])
        )

        return (
            dx_aval,
            dgamma_aval,
            dbeta_aval,
            wkspace_aval,
            barrier_aval,
            dgamma_part_aval,
            dbeta_part_aval,
        )

    @staticmethod
    def outer_abstract(*args, **kwargs):
        """
        LayerNorm bwd outer primitive abstract
        """
        dx_aval, dgamma_aval, dbeta_aval, _, _, _, _ = LayerNormBwdPrimitive.abstract(
            *args, **kwargs
        )
        return dx_aval, dgamma_aval, dbeta_aval

    @staticmethod
    def lowering(ctx, dz, x, mu, rsigma, gamma, *, zero_centered_gamma, epsilon):
        """
        Layernorm bwd lowering rules
        """
        _, x_aval, _, _, gamma_aval = ctx.avals_in
        x_type = ir.RankedTensorType(x.type)
        x_shape = x_type.shape
        g_type = ir.RankedTensorType(gamma.type)
        g_shape = g_type.shape
        b_type = ir.RankedTensorType(gamma.type)
        b_shape = b_type.shape
        assert g_type == b_type
        assert g_shape == b_shape

        dz_shape = ir.RankedTensorType(dz.type).shape
        mu_shape = ir.RankedTensorType(mu.type).shape
        rsigma_shape = ir.RankedTensorType(rsigma.type).shape

        hidden_size = reduce(operator.mul, g_shape)
        batch_size = reduce(operator.mul, x_shape) // hidden_size

        out_types = [
            ir.RankedTensorType.get(output.shape, mlir.dtype_to_ir_type(output.dtype))
            for output in ctx.avals_out
        ]

        operands = [dz, mu, rsigma, x, gamma]
        operand_shapes = [dz_shape, mu_shape, rsigma_shape, x_shape, g_shape]
        args = CustomCallArgsWrapper(out_types, operands, operand_shapes)

        sm_margin = int(os.getenv("NVTE_BWD_LAYERNORM_SM_MARGIN", "0"))

        wkspace_aval, barrier_aval, dgamma_part_aval, dbeta_part_aval = ctx.avals_out[-4:]
        opaque = transformer_engine_jax.pack_norm_descriptor(
            batch_size,
            hidden_size,
            wkspace_aval.size,
            barrier_aval.size,
            dgamma_part_aval.shape,
            dbeta_part_aval.shape,
            jax_dtype_to_te_dtype(x_aval.dtype),
            jax_dtype_to_te_dtype(gamma_aval.dtype),
            jax_dtype_to_te_dtype(wkspace_aval.dtype),
            jax_dtype_to_te_dtype(barrier_aval.dtype),
            jax_dtype_to_te_dtype(dgamma_part_aval.dtype),
            jax_dtype_to_te_dtype(dbeta_part_aval.dtype),
            zero_centered_gamma,
            epsilon,
            sm_margin,
        )

        out = custom_caller(LayerNormBwdPrimitive.name, args, opaque, False)

        return out

    @staticmethod
    def impl(dz, x, mu, rsigma, gamma, zero_centered_gamma, epsilon):
        assert LayerNormBwdPrimitive.inner_primitive is not None
        dx, dgamma, dbeta, _, _, _, _ = LayerNormBwdPrimitive.inner_primitive.bind(
            dz, x, mu, rsigma, gamma, zero_centered_gamma=zero_centered_gamma, epsilon=epsilon
        )
        return dx, dgamma, dbeta

    @staticmethod
    def batcher(batched_args, batch_dims, *, zero_centered_gamma, epsilon):
        check_valid_batch_dims(batch_dims)
        assert LayerNormBwdPrimitive.outer_primitive is not None
        dz, x, mu, rsigma, gamma = batched_args
        _, x_bdim, _, _, gamma_bdim = batch_dims

        out_bdims = x_bdim, gamma_bdim, gamma_bdim
        return (
            LayerNormBwdPrimitive.outer_primitive.bind(
                dz, x, mu, rsigma, gamma, zero_centered_gamma=zero_centered_gamma, epsilon=epsilon
            ),
            out_bdims,
        )

    @staticmethod
    def infer_sharding_from_operands(zero_centered_gamma, epsilon, mesh, arg_infos, result_infos):
        del zero_centered_gamma, epsilon, result_infos
        x_spec = get_padded_spec(arg_infos[1])
        if x_spec[-1] is not None:
            warnings.warn(
                f"Does not support to shard hidden dim in {LayerNormBwdPrimitive.name}! "
                "Force to not shard the hidden dim, which might introduce extra collective ops, "
                "and hurt performance."
            )
        g_b_spec = get_padded_spec(arg_infos[4])
        if g_b_spec[-1] is not None:
            warnings.warn(
                f"{LayerNormBwdPrimitive.name} does not support sharding of gradients "
                "of gamma and beta of Layernorm "
                "Enforcing no sharding of parameters hidden dim! "
            )

        dx_sharding = NamedSharding(mesh, PartitionSpec(*x_spec[:-1], None))
        dgamma_sharding = dbeta_sharding = NamedSharding(mesh, PartitionSpec(None))
        return dx_sharding, dgamma_sharding, dbeta_sharding

    @staticmethod
    def partition(zero_centered_gamma, epsilon, mesh, arg_infos, result_infos):
        del result_infos
        x_spec = get_padded_spec(arg_infos[1])
        if x_spec[-1] is not None:
            warnings.warn(
                f"Does not support to shard hidden dim in {LayerNormBwdPrimitive.name}! "
                "Force to not shard the hidden dim, which might introduce extra collective ops, "
                "and hurt performance."
            )
        g_b_spec = get_padded_spec(arg_infos[4])
        if g_b_spec[-1] is not None:
            warnings.warn(
                f"{LayerNormBwdPrimitive.name} does not support sharding of gradients "
                "of gamma and beta of Layernorm "
                "Enforcing no sharding of parameters hidden dim! "
            )

        dx_sharding = NamedSharding(mesh, PartitionSpec(*x_spec[:-1], None))
        dgamma_sharding = dbeta_sharding = NamedSharding(mesh, PartitionSpec(None))
        out_shardings = dx_sharding, dgamma_sharding, dbeta_sharding
        x_shardings = (dx_sharding,) * 2  # dz and x should have the same sharding.
        mu_shardings = (NamedSharding(mesh, PartitionSpec(*x_spec[:-1])),) * 2
        arg_shardings = (*x_shardings, *mu_shardings, NamedSharding(mesh, PartitionSpec(None)))

        def sharded_impl(dz, x, mu, rsigma, gamma):
            local_dx, local_dgamma, local_dbeta = LayerNormBwdPrimitive.impl(
                dz, x, mu, rsigma, gamma, zero_centered_gamma=zero_centered_gamma, epsilon=epsilon
            )
            global_dgamma = all_reduce_sum_along_dp_fsdp(local_dgamma)
            global_dbeta = all_reduce_sum_along_dp_fsdp(local_dbeta)
            return local_dx, global_dgamma, global_dbeta

        return mesh, sharded_impl, out_shardings, arg_shardings


register_primitive(LayerNormBwdPrimitive)


def layernorm_bwd(
    dz: jnp.ndarray,
    x: jnp.ndarray,
    mu: jnp.ndarray,
    rsigma: jnp.ndarray,
    gamma: jnp.ndarray,
    zero_centered_gamma: bool,
    epsilon: float,
):
    """
    Wrapper for TE layernorm bwd
    """
    return LayerNormBwdPrimitive.outer_primitive.bind(
        dz, x, mu, rsigma, gamma, zero_centered_gamma=zero_centered_gamma, epsilon=epsilon
    )


class RmsNormFwdPrimitive(BasePrimitive):
    """
    RMS Normalization Forward Primitive
    """

    name = "te_rmsnorm_forward"
    multiple_results = True
    impl_static_args = (2,)  # epsilon
    inner_primitive = None
    outer_primitive = None

    @staticmethod
    def abstract(x_aval, gamma_aval, **kwargs):
        """
        RMSNorm fwd inner primitive abstract
        """
        x_dtype = dtypes.canonicalize_dtype(x_aval.dtype)
        assert x_dtype in [jnp.float32, jnp.float16, jnp.bfloat16]

        rsigama_dtype = jnp.float32

        out_aval = core.raise_to_shaped(x_aval)
        rsigma_aval = out_aval.update(shape=out_aval.shape[:-1], dtype=rsigama_dtype)

        hidden_size = gamma_aval.size
        assert x_aval.size % hidden_size == 0

        wkspace_info, barrier_info = transformer_engine_jax.get_layernorm_fwd_workspace_sizes(
            x_aval.size // hidden_size,  # batch size
            hidden_size,
            jax_dtype_to_te_dtype(x_aval.dtype),  # in te_dtype
            jax_dtype_to_te_dtype(gamma_aval.dtype),  # weight te_dtype
            jax_dtype_to_te_dtype(x_aval.dtype),  # out te_dtype (same as input for Fp16/Bf16)
            False,
            False,
            kwargs["epsilon"],
        )
        wkspace_aval = out_aval.update(
            shape=wkspace_info[0], dtype=te_dtype_to_jax_dtype(wkspace_info[1])
        )
        barrier_aval = out_aval.update(
            shape=barrier_info[0], dtype=te_dtype_to_jax_dtype(barrier_info[1])
        )

        return out_aval, rsigma_aval, wkspace_aval, barrier_aval

    @staticmethod
    def outer_abstract(*args, **kwargs):
        """
        RMSNorm fwd outer primitive abstract
        """
        out_aval, rsigma_aval, _, _ = RmsNormFwdPrimitive.abstract(*args, **kwargs)
        return out_aval, rsigma_aval

    @staticmethod
    def lowering(ctx, x, gamma, *, epsilon):
        """
        RMSNorm fwd lowering rules
        """
        x_aval, gamma_aval = ctx.avals_in
        x_type = ir.RankedTensorType(x.type)
        x_shape = x_type.shape
        g_type = ir.RankedTensorType(gamma.type)
        g_shape = g_type.shape
        rsigma_element_type = ir.F32Type.get()

        out_shape = x_shape
        hidden_size = reduce(operator.mul, g_shape)
        batch_shape = out_shape[:-1]
        batch_size = reduce(operator.mul, x_shape) // hidden_size

        wkspace_aval, barrier_aval = ctx.avals_out[-2:]

        out_types = [
            ir.RankedTensorType.get(out_shape, x_type.element_type),
            ir.RankedTensorType.get(batch_shape, rsigma_element_type),
            ir.RankedTensorType.get(wkspace_aval.shape, jax_dtype_to_ir_dtype(wkspace_aval.dtype)),
            ir.RankedTensorType.get(barrier_aval.shape, jax_dtype_to_ir_dtype(barrier_aval.dtype)),
        ]
        operands = [x, gamma]
        operand_shapes = [x_shape, g_shape]
        args = CustomCallArgsWrapper(out_types, operands, operand_shapes)

        sm_margin = int(os.getenv("NVTE_FWD_LAYERNORM_SM_MARGIN", "0"))

        opaque = transformer_engine_jax.pack_norm_descriptor(
            batch_size,
            hidden_size,
            wkspace_aval.size,
            barrier_aval.size,
            (0,),  # no dgamma_part in FWD pass
            (0,),  # no dbeta_part in BWD pass
            jax_dtype_to_te_dtype(x_aval.dtype),
            jax_dtype_to_te_dtype(gamma_aval.dtype),
            jax_dtype_to_te_dtype(wkspace_aval.dtype),
            jax_dtype_to_te_dtype(barrier_aval.dtype),
            TEDType.kByte,  # dummy dgamma_part te_dtype
            TEDType.kByte,  # dummy dbeta_part te_dtype
            False,  # RMSNorm doesn't support zero_centered_gamma
            epsilon,
            sm_margin,
        )

        out = custom_caller(RmsNormFwdPrimitive.name, args, opaque, False)

        return out

    @staticmethod
    def impl(x, gamma, epsilon):
        """
        to describe implementation
        """
        assert RmsNormFwdPrimitive.inner_primitive is not None
        out, rsigma, _, _ = RmsNormFwdPrimitive.inner_primitive.bind(x, gamma, epsilon=epsilon)
        return out, rsigma

    @staticmethod
    def batcher(batched_args, batch_dims, *, epsilon):
        """
        to describe batch rules for vmap
        """
        check_valid_batch_dims(batch_dims)
        assert RmsNormFwdPrimitive.outer_primitive is not None
        x, gamma = batched_args
        x_bdim, _ = batch_dims

        out_bdims = x_bdim, x_bdim
        return RmsNormFwdPrimitive.outer_primitive.bind(x, gamma, epsilon=epsilon), out_bdims

    @staticmethod
    def infer_sharding_from_operands(epsilon, mesh, arg_infos, result_infos):
        del epsilon, result_infos
        x_spec = get_padded_spec(arg_infos[0])
        if x_spec[-1] is not None:
            warnings.warn(
                f"Does not support to shard hidden dim in {RmsNormFwdPrimitive.name}! "
                "Force to not shard the hidden dim, which might introduce extra collective ops, "
                "and hurt performance."
            )
        out_sharding = NamedSharding(mesh, PartitionSpec(*x_spec[:-1], None))
        rsigma_sharding = NamedSharding(mesh, PartitionSpec(*x_spec[:-1]))
        return (out_sharding, rsigma_sharding)

    @staticmethod
    def partition(epsilon, mesh, arg_infos, result_infos):
        del result_infos
        x_spec, g_spec = map(get_padded_spec, arg_infos)
        if x_spec[-1] is not None:
            warnings.warn(
                f"Does not support to shard hidden dim in {RmsNormFwdPrimitive.name}! "
                "Force to not shard the hidden dim, which might introduce extra collective ops, "
                "and hurt performance."
            )
        if g_spec[-1] is not None:
            warnings.warn(
                f"{RmsNormFwdPrimitive.name} does not support sharding of parameter gamma "
                "Enforcing no sharding of parameters hidden dim! "
            )

        x_sharding = NamedSharding(mesh, PartitionSpec(*x_spec[:-1], None))
        g_sharding = NamedSharding(mesh, PartitionSpec(None))
        out_sharding = x_sharding
        rsigma_sharding = NamedSharding(mesh, PartitionSpec(*x_spec[:-1]))
        arg_shardings = (x_sharding, g_sharding)
        out_shardings = (out_sharding, rsigma_sharding)
        impl = partial(RmsNormFwdPrimitive.impl, epsilon=epsilon)
        return mesh, impl, out_shardings, arg_shardings


register_primitive(RmsNormFwdPrimitive)


def rmsnorm_fwd(x: jnp.ndarray, gamma: jnp.ndarray, epsilon: float):
    """
    Wrapper for TE rmsnorm fwd
    """
    return RmsNormFwdPrimitive.outer_primitive.bind(x, gamma, epsilon=epsilon)


class RmsNormBwdPrimitive(BasePrimitive):
    """
    RMS Normalization Backward Primitive
    """

    name = "te_rmsnorm_backward"
    multiple_results = True
    impl_static_args = (4,)  # epsilon
    inner_primitive = None
    outer_primitive = None

    @staticmethod
    def abstract(dz_aval, x_aval, rsigma_aval, gamma_aval, **kwargs):
        """
        RMSNorm bwd inner primitive abstract
        """
        w_dtype = dtypes.canonicalize_dtype(gamma_aval.dtype)
        rsigma_dtype = dtypes.canonicalize_dtype(rsigma_aval.dtype)

        assert dtypes.canonicalize_dtype(dz_aval.dtype) == w_dtype
        assert dz_aval.shape == x_aval.shape
        assert rsigma_aval.shape == x_aval.shape[:-1]
        assert rsigma_dtype == jnp.float32

        dx_aval = core.raise_to_shaped(dz_aval)
        dgamma_aval = core.raise_to_shaped(gamma_aval)

        wkspace_info, barrier_info, dgamma_part_info, _ = (
            transformer_engine_jax.get_layernorm_bwd_workspace_sizes(
                x_aval.size // gamma_aval.size,  # batch size
                gamma_aval.size,  # hidden size
                jax_dtype_to_te_dtype(x_aval.dtype),  # in te_dtype
                jax_dtype_to_te_dtype(gamma_aval.dtype),  # weight te_dtype
                False,
                False,
                kwargs["epsilon"],
            )
        )
        wkspace_aval = dx_aval.update(
            shape=wkspace_info[0], dtype=te_dtype_to_jax_dtype(wkspace_info[1])
        )
        barrier_aval = dx_aval.update(
            shape=barrier_info[0], dtype=te_dtype_to_jax_dtype(barrier_info[1])
        )
        dgamma_part_aval = dgamma_aval.update(
            shape=dgamma_part_info[0], dtype=te_dtype_to_jax_dtype(dgamma_part_info[1])
        )

        return dx_aval, dgamma_aval, wkspace_aval, barrier_aval, dgamma_part_aval

    @staticmethod
    def outer_abstract(*args, **kwargs):
        """
        RMSNorm bwd outer primitive abstract
        """
        dx_aval, dgamma_aval, _, _, _ = RmsNormBwdPrimitive.abstract(*args, **kwargs)
        return dx_aval, dgamma_aval

    @staticmethod
    def lowering(ctx, dz, x, rsigma, gamma, *, epsilon):
        """
        RMSNorm bwd lowering rules
        """
        _, x_aval, _, gamma_aval = ctx.avals_in
        x_type = ir.RankedTensorType(x.type)
        x_shape = x_type.shape
        g_type = ir.RankedTensorType(gamma.type)
        g_shape = g_type.shape
        dz_shape = ir.RankedTensorType(dz.type).shape
        rsigma_shape = ir.RankedTensorType(rsigma.type).shape

        hidden_size = reduce(operator.mul, g_shape)
        batch_size = reduce(operator.mul, x_shape) // hidden_size

        wkspace_aval, barrier_aval, dgamma_part_aval = ctx.avals_out[-3:]

        out_types = [
            ir.RankedTensorType.get(x_shape, x_type.element_type),
            ir.RankedTensorType.get(g_shape, g_type.element_type),
            ir.RankedTensorType.get(wkspace_aval.shape, jax_dtype_to_ir_dtype(wkspace_aval.dtype)),
            ir.RankedTensorType.get(barrier_aval.shape, jax_dtype_to_ir_dtype(barrier_aval.dtype)),
            ir.RankedTensorType.get(
                dgamma_part_aval.shape, jax_dtype_to_ir_dtype(dgamma_part_aval.dtype)
            ),
        ]
        operands = [dz, rsigma, x, gamma]
        operand_shapes = [dz_shape, rsigma_shape, x_shape, g_shape]
        args = CustomCallArgsWrapper(out_types, operands, operand_shapes)

        sm_margin = int(os.getenv("NVTE_BWD_LAYERNORM_SM_MARGIN", "0"))

        opaque = transformer_engine_jax.pack_norm_descriptor(
            batch_size,
            hidden_size,
            wkspace_aval.size,
            barrier_aval.size,
            dgamma_part_aval.shape,
            (0,),  # no dbeta_part for RMSnorm
            jax_dtype_to_te_dtype(x_aval.dtype),
            jax_dtype_to_te_dtype(gamma_aval.dtype),
            jax_dtype_to_te_dtype(wkspace_aval.dtype),
            jax_dtype_to_te_dtype(barrier_aval.dtype),
            jax_dtype_to_te_dtype(dgamma_part_aval.dtype),
            TEDType.kByte,  # dummy dbeta_part te_dtype
            False,  # RMSNorm doesn't support zero_centered_gamma
            epsilon,
            sm_margin,
        )

        out = custom_caller(RmsNormBwdPrimitive.name, args, opaque, False)

        return out

    @staticmethod
    def impl(dz, x, rsigma, gamma, epsilon):
        assert RmsNormBwdPrimitive.inner_primitive is not None
        dx, dgamma, _, _, _ = RmsNormBwdPrimitive.inner_primitive.bind(
            dz, x, rsigma, gamma, epsilon=epsilon
        )
        return dx, dgamma

    @staticmethod
    def batcher(batched_args, batch_dims, *, epsilon):
        check_valid_batch_dims(batch_dims)
        assert RmsNormBwdPrimitive.outer_primitive is not None
        dz, x, rsigma, gamma = batched_args
        _, x_bdim, _, gamma_bdim = batch_dims

        out_bdims = x_bdim, gamma_bdim
        return (
            RmsNormBwdPrimitive.outer_primitive.bind(dz, x, rsigma, gamma, epsilon=epsilon),
            out_bdims,
        )

    @staticmethod
    def infer_sharding_from_operands(epsilon, mesh, arg_infos, result_infos):
        del epsilon, result_infos
        x_spec = get_padded_spec(arg_infos[1])
        if x_spec[-1] is not None:
            warnings.warn(
                f"Does not support to shard hidden dim in {RmsNormBwdPrimitive.name}! "
                "Force to not shard the hidden dim, which might introduce extra collective ops, "
                "and hurt performance."
            )
        g_spec = get_padded_spec(arg_infos[3])
        if g_spec[-1] is not None:
            warnings.warn(
                f"{RmsNormBwdPrimitive.name} does not support sharding of parameter gamma "
                "Enforcing no sharding of parameters hidden dim! "
            )
        dx_sharding = NamedSharding(mesh, PartitionSpec(*x_spec[:-1], None))
        dgamma_sharding = NamedSharding(mesh, PartitionSpec(None))
        return dx_sharding, dgamma_sharding

    @staticmethod
    def partition(epsilon, mesh, arg_infos, result_infos):
        del result_infos
        x_spec = get_padded_spec(arg_infos[1])
        if x_spec[-1] is not None:
            warnings.warn(
                f"Does not support to shard hidden dim in {RmsNormBwdPrimitive.name}! "
                "Force to not shard the hidden dim, which might introduce extra collective ops, "
                "and hurt performance."
            )
        g_spec = get_padded_spec(arg_infos[3])
        if g_spec[-1] is not None:
            warnings.warn(
                f"{RmsNormBwdPrimitive.name} does not support sharding of parameter gamma "
                "Enforcing no sharding of parameters hidden dim! "
            )
        dx_sharding = NamedSharding(mesh, PartitionSpec(*x_spec[:-1], None))
        dgamma_sharding = NamedSharding(mesh, PartitionSpec(None))
        out_shardings = dx_sharding, dgamma_sharding
        x_shardings = (dx_sharding,) * 2  # dz and x should have the same sharding.
        rsigma_sharding = NamedSharding(mesh, PartitionSpec(*x_spec[:-1]))
        arg_shardings = (*x_shardings, rsigma_sharding, NamedSharding(mesh, PartitionSpec(None)))

        def sharded_impl(dz, x, rsigma, gamma):
            local_dx, local_dgamma = RmsNormBwdPrimitive.impl(dz, x, rsigma, gamma, epsilon=epsilon)
            global_dgamma = all_reduce_sum_along_dp_fsdp(local_dgamma)
            return local_dx, global_dgamma

        return mesh, sharded_impl, out_shardings, arg_shardings


register_primitive(RmsNormBwdPrimitive)


def rmsnorm_bwd(
    dz: jnp.ndarray, x: jnp.ndarray, rsigma: jnp.ndarray, gamma: jnp.ndarray, epsilon: float
):
    """
    Wrapper for TE layernorm bwd
    """
    return RmsNormBwdPrimitive.outer_primitive.bind(dz, x, rsigma, gamma, epsilon=epsilon)


class LayerNormFwdFp8Primitive(BasePrimitive):
    """
    Layer Normalization Forward FP8 Primitive
    """

    name = "te_layernorm_forward_fp8"
    multiple_results = True
    impl_static_args = (6, 7, 8)  # out_type, zero_centered_gamma, epsilon
    inner_primitive = None
    outer_primitive = None

    @staticmethod
    def abstract(
        x_aval,
        gamma_aval,
        beta_aval,
        amax_aval,
        scale_aval,
        scale_inv_aval,
        *,
        out_dtype,
        zero_centered_gamma,
        epsilon,
    ):
        """
        LayerNorm fwd (fp8 out) inner primitive abstract
        """
        x_dtype = dtypes.canonicalize_dtype(x_aval.dtype)

        assert x_dtype in [jnp.float32, jnp.float16, jnp.bfloat16]
        assert amax_aval.dtype == jnp.float32
        assert scale_aval.dtype == jnp.float32
        assert scale_inv_aval.dtype == jnp.float32

        mu_rsigama_dtype = jnp.float32

        assert gamma_aval.size == beta_aval.size

        wkspace_info, barrier_info = transformer_engine_jax.get_layernorm_fwd_workspace_sizes(
            x_aval.size // gamma_aval.size,  # batch size
            gamma_aval.size,  # hidden size
            jax_dtype_to_te_dtype(x_aval.dtype),  # in type
            jax_dtype_to_te_dtype(gamma_aval.dtype),  # weight type
            jax_dtype_to_te_dtype(out_dtype),
            True,
            zero_centered_gamma,
            epsilon,
        )

        out_aval = x_aval.update(shape=x_aval.shape, dtype=out_dtype)
        mu_aval = rsigma_aval = out_aval.update(shape=out_aval.shape[:-1], dtype=mu_rsigama_dtype)
        updated_amax_aval = amax_aval.update(shape=amax_aval.shape, dtype=amax_aval.dtype)
        wkspace_aval = x_aval.update(
            shape=wkspace_info[0], dtype=te_dtype_to_jax_dtype(wkspace_info[1])
        )
        barrier_aval = x_aval.update(
            shape=barrier_info[0], dtype=te_dtype_to_jax_dtype(barrier_info[1])
        )

        return out_aval, mu_aval, rsigma_aval, updated_amax_aval, wkspace_aval, barrier_aval

    @staticmethod
    def outer_abstract(*args, **kwargs):
        """
        LayerNorm fwd (fp8 out) outer primitive abstract
        """
        out_aval, mu_aval, rsigma_aval, updated_amax_aval, _, _ = LayerNormFwdFp8Primitive.abstract(
            *args, **kwargs
        )
        return out_aval, mu_aval, rsigma_aval, updated_amax_aval

    @staticmethod
    def lowering(
        ctx, x, gamma, beta, amax, scale, scale_inv, *, out_dtype, zero_centered_gamma, epsilon
    ):
        """
        LayerNorm fwd (fp8 out) lowering rules
        """
        x_aval, gamma_aval, beta_aval, amax_aval, scale_aval, scale_inv_aval = ctx.avals_in

        # Currently only support casting to E4M3 only in C side.
        assert out_dtype == jnp.float8_e4m3fn

        assert x_aval.dtype in [jnp.float32, jnp.float16, jnp.bfloat16]
        assert gamma_aval.dtype == beta_aval.dtype
        assert amax_aval.dtype == jnp.float32
        assert scale_aval.dtype == jnp.float32
        assert scale_inv_aval.dtype == jnp.float32

        x_type = ir.RankedTensorType(x.type)
        x_shape = x_type.shape
        g_type = ir.RankedTensorType(gamma.type)
        g_shape = g_type.shape
        b_type = ir.RankedTensorType(beta.type)
        b_shape = b_type.shape

        assert g_type == b_type
        assert g_shape == b_shape

        ir_out_dtype = jax_dtype_to_ir_dtype(out_dtype)
        ir_mu_dtype = ir.F32Type.get()
        ir_rsigma_dtype = ir.F32Type.get()
        ir_amax_type = ir.RankedTensorType(amax.type)
        ir_amax_dtype = ir_amax_type.element_type
        ir_amax_shape = ir_amax_type.shape
        ir_scale_shape = ir_amax_shape
        ir_scale_inv_shape = ir_amax_shape

        out_shape = x_shape
        hidden_size = reduce(operator.mul, g_shape)
        batch_shape = out_shape[:-1]
        batch_size = reduce(operator.mul, x_shape) // hidden_size

        wkspace_aval, barrier_aval = ctx.avals_out[-2:]

        out_types = [
            ir.RankedTensorType.get(out_shape, ir_out_dtype),
            ir.RankedTensorType.get(batch_shape, ir_mu_dtype),
            ir.RankedTensorType.get(batch_shape, ir_rsigma_dtype),
            ir.RankedTensorType.get(ir_amax_shape, ir_amax_dtype),
            ir.RankedTensorType.get(wkspace_aval.shape, jax_dtype_to_ir_dtype(wkspace_aval.dtype)),
            ir.RankedTensorType.get(barrier_aval.shape, jax_dtype_to_ir_dtype(barrier_aval.dtype)),
        ]
        operands = [x, gamma, beta, amax, scale, scale_inv]
        operand_shapes = [
            x_shape,
            g_shape,
            b_shape,
            ir_amax_shape,
            ir_scale_shape,
            ir_scale_inv_shape,
        ]
        args = CustomCallArgsWrapper(out_types, operands, operand_shapes)

        sm_margin = int(os.getenv("NVTE_FWD_LAYERNORM_SM_MARGIN", "0"))

        opaque = transformer_engine_jax.pack_norm_descriptor(
            batch_size,
            hidden_size,
            wkspace_aval.size,
            barrier_aval.size,
            (0,),  # no dgamma_part in FWD pass
            (0,),  # no dbeta_part in BWD pass
            jax_dtype_to_te_dtype(x_aval.dtype),
            jax_dtype_to_te_dtype(gamma_aval.dtype),
            jax_dtype_to_te_dtype(wkspace_aval.dtype),
            jax_dtype_to_te_dtype(barrier_aval.dtype),
            TEDType.kByte,  # dummy dgamma_part te_dtype
            TEDType.kByte,  # dummy dbeta_part te_dtype
            zero_centered_gamma,
            epsilon,
            sm_margin,
        )

        out = custom_caller(
            LayerNormFwdFp8Primitive.name, args, opaque, False, operand_output_aliases={3: 3}
        )

        return out

    @staticmethod
    def impl(x, gamma, beta, amax, scale, scale_inv, out_dtype, zero_centered_gamma, epsilon):
        """
        to describe implementation
        """
        assert LayerNormFwdFp8Primitive.inner_primitive is not None
        out, mu, rsigma, updated_amax, _, _ = LayerNormFwdFp8Primitive.inner_primitive.bind(
            x,
            gamma,
            beta,
            amax,
            scale,
            scale_inv,
            out_dtype=out_dtype,
            zero_centered_gamma=zero_centered_gamma,
            epsilon=epsilon,
        )
        return out, mu, rsigma, updated_amax

    @staticmethod
    def batcher(batched_args, batch_dims, *, out_dtype, zero_centered_gamma, epsilon):
        """
        to describe batch rules for vmap
        """
        check_valid_batch_dims(batch_dims)
        assert LayerNormFwdFp8Primitive.outer_primitive is not None
        x, gamma, beta, amax, scale, scale_inv = batched_args
        x_bdim, _, _, amax_bdim, _, _ = batch_dims

        out_bdims = x_bdim, x_bdim, x_bdim, amax_bdim
        return (
            LayerNormFwdFp8Primitive.outer_primitive.bind(
                x,
                gamma,
                beta,
                amax,
                scale,
                scale_inv,
                out_dtype=out_dtype,
                zero_centered_gamma=zero_centered_gamma,
                epsilon=epsilon,
            ),
            out_bdims,
        )

    @staticmethod
    def infer_sharding_from_operands(
        out_dtype, zero_centered_gamma, epsilon, mesh, arg_infos, result_infos
    ):
        del out_dtype, zero_centered_gamma, epsilon, result_infos
        x_spec = get_padded_spec(arg_infos[0])
        if x_spec[-1] is not None:
            warnings.warn(
                f"Does not support to shard hidden dim in {LayerNormFwdPrimitive.name}! "
                "Force to not shard the hidden dim, which might introduce extra collective ops, "
                "and hurt performance."
            )

        out_sharding = NamedSharding(mesh, PartitionSpec(*x_spec[:-1], None))
        mu_sharding = rsigma_sharding = NamedSharding(mesh, PartitionSpec(*x_spec[:-1]))
        amax_sharding = NamedSharding(mesh, PartitionSpec(*get_padded_spec(arg_infos[3])))
        return (out_sharding, mu_sharding, rsigma_sharding, amax_sharding)

    @staticmethod
    def partition(out_dtype, zero_centered_gamma, epsilon, mesh, arg_infos, result_infos):
        del result_infos
        x_spec = get_padded_spec(arg_infos[0])
        g_spec = get_padded_spec(arg_infos[1])
        b_spec = get_padded_spec(arg_infos[2])
        if x_spec[-1] is not None:
            warnings.warn(
                f"Does not support to shard hidden dim in {LayerNormFwdFp8Primitive.name}! "
                "Force to not shard the hidden dim, which might introduce extra collective ops, "
                "and hurt performance."
            )
        if g_spec[-1] is not None:
            warnings.warn(
                f"{LayerNormFwdFp8Primitive.name} does not support sharding of parameter gamma "
                "Enforcing no sharding of parameters hidden dim! "
            )
        if b_spec[-1] is not None:
            warnings.warn(
                f"{LayerNormFwdFp8Primitive.name} does not support sharding of parameter beta "
                "Enforcing no sharding of parameters hidden dim! "
            )
        x_sharding = NamedSharding(mesh, PartitionSpec(*x_spec[:-1], None))
        g_sharding = NamedSharding(mesh, PartitionSpec(None))
        b_sharding = NamedSharding(mesh, PartitionSpec(None))
        out_sharding = x_sharding
        mu_sharding = rsigma_sharding = NamedSharding(
            mesh, PartitionSpec(*get_padded_spec(arg_infos[0])[:-1])
        )
        amax_sharding = NamedSharding(mesh, PartitionSpec(*get_padded_spec(arg_infos[3])))
        fp8_meta_sharding = amax_sharding
        arg_shardings = (x_sharding, g_sharding, b_sharding) + (fp8_meta_sharding,) * 3
        out_shardings = (out_sharding, mu_sharding, rsigma_sharding, amax_sharding)

        def sharded_impl(x, gamma, beta, amax, scale, scale_inv):
            local_x, local_mu, local_rsigma, local_amax = LayerNormFwdFp8Primitive.impl(
                x,
                gamma,
                beta,
                amax,
                scale,
                scale_inv,
                out_dtype=out_dtype,
                zero_centered_gamma=zero_centered_gamma,
                epsilon=epsilon,
            )
            global_updated_amax = all_reduce_max_along_all_axes_except_PP(local_amax)

            return local_x, local_mu, local_rsigma, global_updated_amax

        return mesh, sharded_impl, out_shardings, arg_shardings


register_primitive(LayerNormFwdFp8Primitive)


def layernorm_fwd_fp8(
    x: jnp.ndarray,
    gamma: jnp.ndarray,
    beta: jnp.ndarray,
    amax: jnp.ndarray,
    scale: jnp.ndarray,
    scale_inv: jnp.ndarray,
    out_dtype: jnp.dtype,
    zero_centered_gamma: bool,
    epsilon: float,
):
    """
    Wrapper for TE layernorm fwd (fp8 out)
    """
    return LayerNormFwdFp8Primitive.outer_primitive.bind(
        x,
        gamma,
        beta,
        amax,
        scale,
        scale_inv,
        out_dtype=out_dtype,
        zero_centered_gamma=zero_centered_gamma,
        epsilon=epsilon,
    )


class RmsNormFwdFp8Primitive(BasePrimitive):
    """
    RMS Normalization Forward FP8 Primitive
    """

    name = "te_rmsnorm_forward_fp8"
    multiple_results = True
    impl_static_args = (5, 6)  # out_dtype, epsilon
    inner_primitive = None
    outer_primitive = None

    @staticmethod
    def abstract(x_aval, gamma_aval, amax_aval, scale_aval, scale_inv_aval, out_dtype, epsilon):
        """
        RMSNorm fwd (fp8 out) inner primitive abstract
        """
        x_dtype = dtypes.canonicalize_dtype(x_aval.dtype)

        assert x_dtype in [jnp.float32, jnp.float16, jnp.bfloat16]
        assert amax_aval.dtype == jnp.float32
        assert scale_aval.dtype == jnp.float32
        assert scale_inv_aval.dtype == jnp.float32

        hidden_size = gamma_aval.size
        assert x_aval.size % hidden_size == 0

        rsigama_dtype = jnp.float32

        wkspace_info, barrier_info = transformer_engine_jax.get_layernorm_fwd_workspace_sizes(
            x_aval.size // hidden_size,  # batch_size
            hidden_size,
            jax_dtype_to_te_dtype(x_aval.dtype),  # in te_dtype
            jax_dtype_to_te_dtype(gamma_aval.dtype),  # weight te_dtype
            jax_dtype_to_te_dtype(out_dtype),  # out te_dtype
            False,
            False,
            epsilon,
        )

        out_aval = x_aval.update(shape=x_aval.shape, dtype=out_dtype)
        rsigma_aval = out_aval.update(shape=out_aval.shape[:-1], dtype=rsigama_dtype)
        amax_aval = out_aval.update(shape=amax_aval.shape, dtype=amax_aval.dtype)
        wkspace_aval = x_aval.update(
            shape=wkspace_info[0], dtype=te_dtype_to_jax_dtype(wkspace_info[1])
        )
        barrier_aval = x_aval.update(
            shape=barrier_info[0], dtype=te_dtype_to_jax_dtype(barrier_info[1])
        )

        return out_aval, rsigma_aval, amax_aval, wkspace_aval, barrier_aval

    @staticmethod
    def outer_abstract(*args, **kwargs):
        """
        RMSNorm fwd (fp8 out) outer primitive abstract
        """
        out_aval, rsigma_aval, amax_aval, _, _ = RmsNormFwdFp8Primitive.abstract(*args, **kwargs)
        return out_aval, rsigma_aval, amax_aval

    @staticmethod
    def lowering(ctx, x, gamma, amax, scale, scale_inv, *, out_dtype, epsilon):
        """
        RMSNorm fwd (fp8 out) lowering rules
        """

        # Currently only support casting to E4M3 only in C side.
        assert out_dtype == jnp.float8_e4m3fn

        x_aval, gamma_aval, amax_aval, scale_aval, scale_inv_aval = ctx.avals_in

        assert x_aval.dtype in [jnp.float32, jnp.float16, jnp.bfloat16]
        assert amax_aval.dtype == jnp.float32
        assert scale_aval.dtype == jnp.float32
        assert scale_inv_aval.dtype == jnp.float32

        x_type = ir.RankedTensorType(x.type)
        x_shape = x_type.shape
        g_type = ir.RankedTensorType(gamma.type)
        g_shape = g_type.shape

        ir_out_dtype = jax_dtype_to_ir_dtype(out_dtype)
        ir_rsigma_dtype = ir.F32Type.get()
        ir_amax_type = ir.RankedTensorType(amax.type)
        ir_amax_dtype = ir_amax_type.element_type
        ir_amax_shape = ir_amax_type.shape
        ir_scale_shape = ir_amax_shape
        ir_scale_inv_shape = ir_amax_shape

        out_shape = x_shape
        hidden_size = reduce(operator.mul, g_shape)
        batch_shape = out_shape[:-1]
        batch_size = reduce(operator.mul, x_shape) // hidden_size

        wkspace_aval, barrier_aval = ctx.avals_out[-2:]

        out_types = [
            ir.RankedTensorType.get(out_shape, ir_out_dtype),
            ir.RankedTensorType.get(batch_shape, ir_rsigma_dtype),
            ir.RankedTensorType.get(ir_amax_shape, ir_amax_dtype),
            ir.RankedTensorType.get(wkspace_aval.shape, jax_dtype_to_ir_dtype(wkspace_aval.dtype)),
            ir.RankedTensorType.get(barrier_aval.shape, jax_dtype_to_ir_dtype(barrier_aval.dtype)),
        ]
        operands = [x, gamma, amax, scale, scale_inv]
        operand_shapes = [x_shape, g_shape, ir_amax_shape, ir_scale_shape, ir_scale_inv_shape]
        args = CustomCallArgsWrapper(out_types, operands, operand_shapes)

        sm_margin = int(os.getenv("NVTE_FWD_LAYERNORM_SM_MARGIN", "0"))

        opaque = transformer_engine_jax.pack_norm_descriptor(
            batch_size,
            hidden_size,
            wkspace_aval.size,
            barrier_aval.size,
            (0,),  # no dgamma_part in FWD pass
            (0,),  # no dbeta_part in BWD pass
            jax_dtype_to_te_dtype(x_aval.dtype),
            jax_dtype_to_te_dtype(gamma_aval.dtype),
            jax_dtype_to_te_dtype(wkspace_aval.dtype),
            jax_dtype_to_te_dtype(barrier_aval.dtype),
            TEDType.kByte,  # dummy dgamma_part te_dtype
            TEDType.kByte,  # dummy dbeta_part te_dtype
            False,  # RMSNorm doesn't support zero_centered_gamma
            epsilon,
            sm_margin,
        )

        out = custom_caller(
            RmsNormFwdFp8Primitive.name, args, opaque, False, operand_output_aliases={2: 2}
        )

        return out

    @staticmethod
    def impl(x, gamma, amax, scale, scale_inv, out_dtype, epsilon):
        """
        to describe implementation
        """
        assert RmsNormFwdFp8Primitive.inner_primitive is not None
        out, rsigma, amax, _, _ = RmsNormFwdFp8Primitive.inner_primitive.bind(
            x, gamma, amax, scale, scale_inv, out_dtype=out_dtype, epsilon=epsilon
        )
        return out, rsigma, amax

    @staticmethod
    def batcher(batched_args, batch_dims, *, out_dtype, epsilon):
        """
        to describe batch rules for vmap
        """
        check_valid_batch_dims(batch_dims)
        assert RmsNormFwdFp8Primitive.outer_primitive is not None
        x, gamma, amax, scale, scale_inv = batched_args
        x_bdim, _, amax_bdim, _, _ = batch_dims
        out_bdims = x_bdim, x_bdim, amax_bdim
        return (
            RmsNormFwdFp8Primitive.outer_primitive.bind(
                x, gamma, amax, scale, scale_inv, out_dtype=out_dtype, epsilon=epsilon
            ),
            out_bdims,
        )

    @staticmethod
    def infer_sharding_from_operands(out_dtype, epsilon, mesh, arg_infos, result_infos):
        del out_dtype, epsilon, result_infos
        x_spec = get_padded_spec(arg_infos[0])
        if x_spec[-1] is not None:
            warnings.warn(
                f"Does not support to shard hidden dim in {RmsNormFwdFp8Primitive.name}! "
                "Force to not shard the hidden dim, which might introduce extra collective ops, "
                "and hurt performance."
            )
        out_sharding = NamedSharding(mesh, PartitionSpec(*x_spec[:-1], None))
        rsigma_sharding = NamedSharding(mesh, PartitionSpec(*x_spec[:-1]))
        amax_sharding = NamedSharding(mesh, PartitionSpec(*get_padded_spec(arg_infos[2])))
        return (out_sharding, rsigma_sharding, amax_sharding)

    @staticmethod
    def partition(out_dtype, epsilon, mesh, arg_infos, result_infos):
        del result_infos
        x_spec = get_padded_spec(arg_infos[0])
        g_spec = get_padded_spec(arg_infos[1])
        if x_spec[-1] is not None:
            warnings.warn(
                f"Does not support to shard hidden dim in {RmsNormFwdFp8Primitive.name}! "
                "Force to not shard the hidden dim, which might introduce extra collective ops, "
                "and hurt performance."
            )
        if g_spec[-1] is not None:
            warnings.warn(
                f"{RmsNormFwdFp8Primitive.name} does not support sharding of parameter gamma "
                "Enforcing no sharding of parameters hidden dim! "
            )
        x_sharding = NamedSharding(mesh, PartitionSpec(*x_spec[:-1], None))
        g_sharding = NamedSharding(mesh, PartitionSpec(None))
        out_sharding = x_sharding
        rsigma_sharding = NamedSharding(mesh, PartitionSpec(*get_padded_spec(arg_infos[0])[:-1]))
        amax_sharding = NamedSharding(mesh, PartitionSpec(*get_padded_spec(arg_infos[2])))
        fp8_meta_sharding = amax_sharding
        arg_shardings = (x_sharding, g_sharding) + (fp8_meta_sharding,) * 3
        out_shardings = (out_sharding, rsigma_sharding, amax_sharding)

        def sharded_impl(x, gamma, amax, scale, scale_inv):
            local_x, local_rsigma, local_amax = RmsNormFwdFp8Primitive.impl(
                x, gamma, amax, scale, scale_inv, out_dtype=out_dtype, epsilon=epsilon
            )
            global_updated_amax = all_reduce_max_along_all_axes_except_PP(local_amax)

            return local_x, local_rsigma, global_updated_amax

        return mesh, sharded_impl, out_shardings, arg_shardings


register_primitive(RmsNormFwdFp8Primitive)


def rmsnorm_fwd_fp8(
    x: jnp.ndarray,
    gamma: jnp.ndarray,
    amax: jnp.ndarray,
    scale: jnp.ndarray,
    scale_inv: jnp.ndarray,
    out_dtype: jnp.dtype,
    epsilon: float,
):
    """
    Wrapper for TE rmsnorm fwd (fp8 out)
    """
    return RmsNormFwdFp8Primitive.outer_primitive.bind(
        x, gamma, amax, scale, scale_inv, out_dtype=out_dtype, epsilon=epsilon
    )
