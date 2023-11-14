# Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""JAX te custom call"""

from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import Tuple
from functools import partial, reduce
import operator
import warnings

import numpy as np
import jax.numpy as jnp
import jax.random as jax_random
from jax.lib import xla_client
from jax import core, dtypes
from jax.interpreters import xla, mlir
from jax.experimental.custom_partitioning import custom_partitioning
from jax.interpreters.mlir import ir, dtype_to_ir_type
from jax.sharding import PartitionSpec, NamedSharding
from jax._src.interpreters import batching

try:
    from jaxlib.hlo_helpers import custom_call
except ImportError:
    # Newer JAX changed its API. But we want to support a few JAX
    # version, so we still need this import.
    pass

import transformer_engine_jax
from transformer_engine_jax import DType as TEDType
from transformer_engine_jax import NVTE_Bias_Type
from transformer_engine_jax import NVTE_Mask_Type
from transformer_engine_jax import NVTE_QKV_Layout
from transformer_engine_jax import NVTE_Fused_Attn_Backend

from .sharding import all_reduce_max_along_all_axes_except_PP
from .sharding import all_reduce_sum_along_dp_fsdp
from .sharding import get_all_mesh_axes, num_of_devices
from .sharding import get_padded_spec as te_get_padded_spec

for _name, _value in transformer_engine_jax.registrations().items():
    xla_client.register_custom_call_target(_name, _value, platform="CUDA")


def te_dtype_to_jax_dtype(te_dtype):    # pylint: disable=too-many-return-statements
    """
    convert TE dtype to jax dtype
    """
    assert isinstance(te_dtype, TEDType)
    if te_dtype == TEDType.kFloat32:
        return jnp.float32
    if te_dtype == TEDType.kFloat16:
        return jnp.float16
    if te_dtype == TEDType.kBFloat16:
        return jnp.bfloat16
    if te_dtype == TEDType.kInt32:
        return jnp.int32
    if te_dtype == TEDType.kInt64:
        return jnp.int64
    if te_dtype == TEDType.kFloat8E4M3:
        return jnp.float8_e4m3fn
    if te_dtype == TEDType.kFloat8E5M2:
        return jnp.float8_e5m2
    raise ValueError(f"Not support the {te_dtype=}")


def te_dtype_to_ir_dtype(te_dtype):
    """
    convert TE dtype to MLIR dtype
    """
    return dtype_to_ir_type(np.dtype(te_dtype_to_jax_dtype(te_dtype)))


def jax_dtype_to_ir_dtype(jax_dtype):
    """
    convert Jax dtype to MLIR dtype
    """
    return dtype_to_ir_type(np.dtype(jax_dtype))


def jax_dtype_to_te_dtype(jax_dtype):
    """
    convert jax dtype to TE dtype
    """
    if jax_dtype == jnp.float32:
        return TEDType.kFloat32
    if jax_dtype == jnp.float16:
        return TEDType.kFloat16
    if jax_dtype == jnp.bfloat16:
        return TEDType.kBFloat16
    if jax_dtype == jnp.float8_e4m3fn:
        return TEDType.kFloat8E4M3
    if jax_dtype == jnp.float8_e5m2:
        return TEDType.kFloat8E5M2
    raise ValueError(f"Not support the {jax_dtype=}")


def get_padded_spec(arg_info):
    """
    Get padded spec for partitioning from arguments' information
    """
    if arg_info.sharding is None:
        return te_get_padded_spec(None, arg_info.ndim)
    ndim, spec = arg_info.ndim, arg_info.sharding.spec
    return te_get_padded_spec(spec, ndim)


class BasePrimitive(metaclass=ABCMeta):
    """
    jax premitive
    """

    @staticmethod
    @abstractmethod
    def abstract():
        """
        to describe computing graph
        """
        return NotImplemented

    @staticmethod
    @abstractmethod
    def lowering():
        """
        to describe MLIR
        """
        return NotImplemented

    @staticmethod
    @abstractmethod
    def impl():
        """
        to describe implementation
        """
        return NotImplemented

    @staticmethod
    @abstractmethod
    def batcher():
        """
        to describe batch rules for vmap
        """
        return NotImplemented

    @staticmethod
    @abstractmethod
    def infer_sharding_from_operands():
        """
        to describe infer_sharding_from_operands for custom_partitioning
        """
        return NotImplemented

    @staticmethod
    @abstractmethod
    def partition():
        """
        to describe partition for custom_partitioning
        """
        return NotImplemented


def register_primitive(cls):
    """
    register jax primitive
    """

    def name_of_wrapper_p():
        return cls.name + "_wrapper"

    inner_p = core.Primitive(cls.name)
    inner_p.multiple_results = cls.multiple_results
    inner_p.def_impl(partial(xla.apply_primitive, inner_p))
    inner_p.def_abstract_eval(cls.abstract)
    mlir.register_lowering(inner_p, cls.lowering, platform='cuda')
    cls.inner_primitive = inner_p

    outer_p = core.Primitive(name_of_wrapper_p())
    outer_p.multiple_results = cls.multiple_results
    outer_p.def_impl(cls.impl)
    outer_p.def_abstract_eval(cls.abstract)
    batching.primitive_batchers[outer_p] = cls.batcher
    outer_p_lower = custom_partitioning(cls.impl, static_argnums=cls.impl_static_args)
    outer_p_lower.def_partition(infer_sharding_from_operands=cls.infer_sharding_from_operands,
                                partition=cls.partition)
    mlir.register_lowering(outer_p,
                           mlir.lower_fun(outer_p_lower, multiple_results=cls.multiple_results))
    cls.outer_primitive = outer_p


@dataclass
class CustomCallArgsWrapper:
    """
    wrapper of XLA custom call args
    """

    def __init__(self,
                 output_types,
                 operands,
                 operand_shapes,
                 operand_specific_layouts=None,
                 output_specific_layouts=None):
        self.output_types = output_types
        self.operands = operands
        self.operand_layouts = CustomCallArgsWrapper.generate_layouts(operand_shapes,
                                                                      operand_specific_layouts)
        output_shapes = [x.shape for x in output_types]
        self.output_layouts = CustomCallArgsWrapper.generate_layouts(output_shapes,
                                                                     output_specific_layouts)

    @staticmethod
    def generate_layouts(shapes, specific_layouts):
        """
        setup layouts for XLA custom call
        """

        def default_layout(shape):
            return range(len(shape) - 1, -1, -1)

        if specific_layouts is None:
            specific_layouts = {}

        layouts = []
        for idx, shape in enumerate(shapes):
            if idx in specific_layouts:
                layouts.append(specific_layouts[idx])
            else:
                layouts.append(default_layout(shape))
        return layouts


def custom_caller(name, args, opaque, has_side_effect, **kwargs):
    """
    XLA custom call warpper
    """
    if hasattr(mlir, "custom_call"):
        out = mlir.custom_call(name,
                               result_types=args.output_types,
                               operands=args.operands,
                               operand_layouts=args.operand_layouts,
                               result_layouts=args.output_layouts,
                               backend_config=opaque,
                               has_side_effect=has_side_effect,
                               **kwargs).results
    else:
        # Need to disable one pylint error as the second function
        # parameter name recenctly in JAX. Otherwise we won't be
        # compatible with multiple JAX version.
        out = custom_call(    # pylint: disable=too-many-function-args
            name,
            args.output_types,
            operands=args.operands,
            operand_layouts=args.operand_layouts,
            result_layouts=args.output_layouts,
            backend_config=opaque,
            has_side_effect=has_side_effect,
            **kwargs)
    return out


class LayerNormFwdPrimitive(BasePrimitive):
    """
    Layer Normalization Forward Primitive
    """
    name = "te_layernorm_forward"
    multiple_results = True
    impl_static_args = (3, 4)    # zero_centered_gamma, epsilon
    inner_primitive = None
    outer_primitive = None

    @staticmethod
    def abstract(x_aval, gamma_aval, beta_aval, **kwargs):    # pylint: disable=unused-argument
        """
        LayerNorm fwd abstract
        """
        x_dtype = dtypes.canonicalize_dtype(x_aval.dtype)
        assert x_dtype in [jnp.float32, jnp.float16, jnp.bfloat16]

        mu_rsigama_dtype = jnp.float32

        out_aval = core.raise_to_shaped(x_aval)
        mu_aval = rsigma_aval = out_aval.update(shape=out_aval.shape[:-1], dtype=mu_rsigama_dtype)

        assert gamma_aval.size == beta_aval.size
        hidden_size = gamma_aval.size
        assert x_aval.size % hidden_size == 0

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

        out_types = [
            ir.RankedTensorType.get(out_shape, output_type),
            ir.RankedTensorType.get(batch_shape, ir_mu_dtype),
            ir.RankedTensorType.get(batch_shape, ir_rsigma_dtype),
        ]
        operands = [x, gamma, beta]
        operand_shapes = [x_shape, g_shape, b_shape]
        args = CustomCallArgsWrapper(out_types, operands, operand_shapes)

        opaque = transformer_engine_jax.pack_norm_descriptor(
            batch_size,
            hidden_size,
            jax_dtype_to_te_dtype(x_aval.dtype),
            jax_dtype_to_te_dtype(gamma_aval.dtype),
            zero_centered_gamma,
            epsilon,
        )

        out = custom_caller(LayerNormFwdPrimitive.name, args, opaque, False)

        return out

    @staticmethod
    def impl(x, gamma, beta, zero_centered_gamma, epsilon):
        """
        to describe implementation
        """
        assert LayerNormFwdPrimitive.inner_primitive is not None
        out, mu, rsigma = LayerNormFwdPrimitive.inner_primitive.bind(
            x, gamma, beta, zero_centered_gamma=zero_centered_gamma, epsilon=epsilon)
        return out, mu, rsigma

    @staticmethod
    def batcher(batched_args, batch_dims, *, zero_centered_gamma, epsilon):
        """
        to describe batch rules for vmap
        """
        assert LayerNormFwdPrimitive.outer_primitive is not None
        x, gamma, beta = batched_args
        x_bdim, _, _ = batch_dims

        out_bdims = x_bdim, x_bdim, x_bdim
        return LayerNormFwdPrimitive.outer_primitive.bind(x,
                                                          gamma,
                                                          beta,
                                                          zero_centered_gamma=zero_centered_gamma,
                                                          epsilon=epsilon), out_bdims

    @staticmethod
    def infer_sharding_from_operands(zero_centered_gamma, epsilon, mesh, arg_infos, result_infos):
        del zero_centered_gamma, epsilon, result_infos
        x_spec = get_padded_spec(arg_infos[0])
        if x_spec[-1] is not None:
            warnings.warn(
                f"Does not support to shard hidden dim in {LayerNormFwdPrimitive.name}! " \
                f"Force to not shard the hidden dim, which might introduce extra collective ops, " \
                f"and hurt performance."
            )
        out_sharding = NamedSharding(mesh, PartitionSpec(*x_spec[:-1], None))
        mu_sharding = rsigma_sharding = NamedSharding(mesh, PartitionSpec(*x_spec[:-1]))
        return (out_sharding, mu_sharding, rsigma_sharding)

    @staticmethod
    def partition(zero_centered_gamma, epsilon, mesh, arg_infos, result_infos):
        del result_infos
        x_spec = get_padded_spec(arg_infos[0])
        if x_spec[-1] is not None:
            warnings.warn(
                f"Does not support to shard hidden dim in {LayerNormFwdPrimitive.name}! " \
                f"Force to not shard the hidden dim, which might introduce extra collective ops, " \
                f"and hurt performance."
            )
        x_sharding = NamedSharding(mesh, PartitionSpec(*x_spec[:-1], None))
        g_sharding = NamedSharding(mesh, PartitionSpec(*get_padded_spec(arg_infos[1])))
        b_sharding = NamedSharding(mesh, PartitionSpec(*get_padded_spec(arg_infos[2])))
        out_sharding = x_sharding
        mu_sharding = rsigma_sharding = NamedSharding(
            mesh, PartitionSpec(*get_padded_spec(arg_infos[0])[:-1]))
        arg_shardings = (x_sharding, g_sharding, b_sharding)
        out_shardings = (out_sharding, mu_sharding, rsigma_sharding)
        impl = partial(LayerNormFwdPrimitive.impl,
                       zero_centered_gamma=zero_centered_gamma,
                       epsilon=epsilon)
        return mesh, impl, out_shardings, arg_shardings


register_primitive(LayerNormFwdPrimitive)


def layernorm_fwd(x: jnp.ndarray, gamma: jnp.ndarray, beta: jnp.ndarray, zero_centered_gamma: bool,
                  epsilon: float):
    """
    Wrapper for TE layernorm fwd
    """
    return LayerNormFwdPrimitive.outer_primitive.bind(x,
                                                      gamma,
                                                      beta,
                                                      zero_centered_gamma=zero_centered_gamma,
                                                      epsilon=epsilon)


class LayerNormBwdPrimitive(BasePrimitive):
    """
    Layer Normalization Backward Primitive
    """
    name = "te_layernorm_backward"
    multiple_results = True
    impl_static_args = (5, 6)    # zero_centered_gamma, epsilon
    inner_primitive = None
    outer_primitive = None

    @staticmethod
    def abstract(dz_aval, x_aval, mu_aval, rsigma_aval, gamma_aval, **kwargs):    # pylint: disable=unused-argument
        """
        Layernorm bwd abstract
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
            ir.RankedTensorType.get(x_shape, x_type.element_type),
            ir.RankedTensorType.get(g_shape, g_type.element_type),
            ir.RankedTensorType.get(b_shape, b_type.element_type),
        ]
        operands = [dz, mu, rsigma, x, gamma]
        operand_shapes = [dz_shape, mu_shape, rsigma_shape, x_shape, g_shape]
        args = CustomCallArgsWrapper(out_types, operands, operand_shapes)

        opaque = transformer_engine_jax.pack_norm_descriptor(
            batch_size,
            hidden_size,
            jax_dtype_to_te_dtype(x_aval.dtype),
            jax_dtype_to_te_dtype(gamma_aval.dtype),
            zero_centered_gamma,
            epsilon,
        )

        out = custom_caller(LayerNormBwdPrimitive.name, args, opaque, False)

        return out

    @staticmethod
    def impl(dz, x, mu, rsigma, gamma, zero_centered_gamma, epsilon):
        assert LayerNormBwdPrimitive.inner_primitive is not None
        dx, dgamma, dbeta = LayerNormBwdPrimitive.inner_primitive.bind(
            dz, x, mu, rsigma, gamma, zero_centered_gamma=zero_centered_gamma, epsilon=epsilon)
        return dx, dgamma, dbeta

    @staticmethod
    def batcher(batched_args, batch_dims, *, zero_centered_gamma, epsilon):
        assert LayerNormBwdPrimitive.outer_primitive is not None
        dz, x, mu, rsigma, gamma = batched_args
        _, x_bdim, _, _, gamma_bdim = batch_dims

        out_bdims = x_bdim, gamma_bdim, gamma_bdim
        return LayerNormBwdPrimitive.outer_primitive.bind(dz,
                                                          x,
                                                          mu,
                                                          rsigma,
                                                          gamma,
                                                          zero_centered_gamma=zero_centered_gamma,
                                                          epsilon=epsilon), out_bdims

    @staticmethod
    def infer_sharding_from_operands(zero_centered_gamma, epsilon, mesh, arg_infos, result_infos):
        del zero_centered_gamma, epsilon, result_infos
        x_spec = get_padded_spec(arg_infos[1])
        if x_spec[-1] is not None:
            warnings.warn(
                f"Does not support to shard hidden dim in {LayerNormBwdPrimitive.name}! " \
                f"Force to not shard the hidden dim, which might introduce extra collective ops, " \
                f"and hurt performance."
            )
        g_b_spec = get_padded_spec(arg_infos[4])
        dx_sharding = NamedSharding(mesh, PartitionSpec(*x_spec[:-1], None))
        dgamma_sharding = dbeta_sharding = NamedSharding(mesh, PartitionSpec(*g_b_spec))
        return dx_sharding, dgamma_sharding, dbeta_sharding

    @staticmethod
    def partition(zero_centered_gamma, epsilon, mesh, arg_infos, result_infos):
        del result_infos
        x_spec = get_padded_spec(arg_infos[1])
        if x_spec[-1] is not None:
            warnings.warn(
                f"Does not support to shard hidden dim in {LayerNormBwdPrimitive.name}! " \
                f"Force to not shard the hidden dim, which might introduce extra collective ops, " \
                f"and hurt performance."
            )
        g_b_spec = get_padded_spec(arg_infos[4])
        dx_sharding = NamedSharding(mesh, PartitionSpec(*x_spec[:-1], None))
        dgamma_sharding = dbeta_sharding = NamedSharding(mesh, PartitionSpec(*g_b_spec))
        out_shardings = dx_sharding, dgamma_sharding, dbeta_sharding
        x_shardings = (dx_sharding,) * 2
        mu_shardings = (NamedSharding(mesh, PartitionSpec(*x_spec[:-1])),) * 2
        arg_shardings = (*x_shardings, *mu_shardings, NamedSharding(mesh, PartitionSpec(*g_b_spec)))

        def sharded_impl(dz, x, mu, rsigma, gamma):
            local_dx, local_dgamma, local_dbeta = \
                LayerNormBwdPrimitive.impl(dz, x, mu, rsigma, gamma,
                     zero_centered_gamma=zero_centered_gamma,
                     epsilon=epsilon)
            global_dgamma = all_reduce_sum_along_dp_fsdp(local_dgamma)
            global_dbeta = all_reduce_sum_along_dp_fsdp(local_dbeta)
            return local_dx, global_dgamma, global_dbeta

        return mesh, sharded_impl, out_shardings, arg_shardings


register_primitive(LayerNormBwdPrimitive)


def layernorm_bwd(dz: jnp.ndarray, x: jnp.ndarray, mu: jnp.ndarray, rsigma: jnp.ndarray,
                  gamma: jnp.ndarray, zero_centered_gamma: bool, epsilon: float):
    """
    Wrapper for TE layernorm bwd
    """
    return LayerNormBwdPrimitive.outer_primitive.bind(dz,
                                                      x,
                                                      mu,
                                                      rsigma,
                                                      gamma,
                                                      zero_centered_gamma=zero_centered_gamma,
                                                      epsilon=epsilon)


class RmsNormFwdPrimitive(BasePrimitive):
    """
    RMS Normalization Forward Primitive
    """
    name = "te_rmsnorm_forward"
    multiple_results = True
    impl_static_args = (2,)    # epsilon
    inner_primitive = None
    outer_primitive = None

    @staticmethod
    def abstract(x_aval, gamma_aval, **kwargs):    # pylint: disable=unused-argument
        """
        RMSNorm fwd abstract
        """
        x_dtype = dtypes.canonicalize_dtype(x_aval.dtype)
        assert x_dtype in [jnp.float32, jnp.float16, jnp.bfloat16]

        rsigama_dtype = jnp.float32

        out_aval = core.raise_to_shaped(x_aval)
        rsigma_aval = out_aval.update(shape=out_aval.shape[:-1], dtype=rsigama_dtype)

        hidden_size = gamma_aval.size
        assert x_aval.size % hidden_size == 0

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

        out_types = [
            ir.RankedTensorType.get(out_shape, x_type.element_type),
            ir.RankedTensorType.get(batch_shape, rsigma_element_type),
        ]
        operands = [x, gamma]
        operand_shapes = [x_shape, g_shape]
        args = CustomCallArgsWrapper(out_types, operands, operand_shapes)

        opaque = transformer_engine_jax.pack_norm_descriptor(
            batch_size,
            hidden_size,
            jax_dtype_to_te_dtype(x_aval.dtype),
            jax_dtype_to_te_dtype(gamma_aval.dtype),
            False,    # RMSNorm doesn't support zero_centered_gamma
            epsilon,
        )

        out = custom_caller(RmsNormFwdPrimitive.name, args, opaque, False)

        return out

    @staticmethod
    def impl(x, gamma, epsilon):
        """
        to describe implementation
        """
        assert RmsNormFwdPrimitive.inner_primitive is not None
        out, rsigma = RmsNormFwdPrimitive.inner_primitive.bind(x, gamma, epsilon=epsilon)
        return out, rsigma

    @staticmethod
    def batcher(batched_args, batch_dims, *, epsilon):
        """
        to describe batch rules for vmap
        """
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
                f"Does not support to shard hidden dim in {RmsNormFwdPrimitive.name}! " \
                f"Force to not shard the hidden dim, which might introduce extra collective ops, " \
                f"and hurt performance."
            )
        out_sharding = NamedSharding(mesh, PartitionSpec(*x_spec[:-1], None))
        rsigma_sharding = NamedSharding(mesh, PartitionSpec(*x_spec[:-1]))
        return (out_sharding, rsigma_sharding)

    @staticmethod
    def partition(epsilon, mesh, arg_infos, result_infos):
        del result_infos
        x_spec = get_padded_spec(arg_infos[0])
        if x_spec[-1] is not None:
            warnings.warn(
                f"Does not support to shard hidden dim in {RmsNormFwdPrimitive.name}! " \
                f"Force to not shard the hidden dim, which might introduce extra collective ops, " \
                f"and hurt performance."
            )
        x_sharding = NamedSharding(mesh, PartitionSpec(*x_spec[:-1], None))
        g_sharding = NamedSharding(mesh, PartitionSpec(*get_padded_spec(arg_infos[1])))
        out_sharding = x_sharding
        rsigma_sharding = NamedSharding(mesh, PartitionSpec(*get_padded_spec(arg_infos[0])[:-1]))
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
    impl_static_args = (4,)    # epsilon
    inner_primitive = None
    outer_primitive = None

    @staticmethod
    def abstract(
            dz_aval,
            x_aval,
            rsigma_aval,
            gamma_aval,
            **kwargs    # pylint: disable=unused-argument
    ):
        """
        RMSNorm bwd abstract
        """
        w_dtype = dtypes.canonicalize_dtype(gamma_aval.dtype)
        rsigma_dtype = dtypes.canonicalize_dtype(rsigma_aval.dtype)

        assert dtypes.canonicalize_dtype(dz_aval.dtype) == w_dtype
        assert dz_aval.shape == x_aval.shape
        assert rsigma_aval.shape == x_aval.shape[:-1]
        assert rsigma_dtype == jnp.float32

        dx_aval = core.raise_to_shaped(dz_aval)
        dgamma_aval = core.raise_to_shaped(gamma_aval)
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
        go_shape = ir.RankedTensorType(dz.type).shape
        rsigma_shape = ir.RankedTensorType(rsigma.type).shape

        hidden_size = reduce(operator.mul, g_shape)
        batch_size = reduce(operator.mul, x_shape) // hidden_size

        out_types = [
            ir.RankedTensorType.get(x_shape, x_type.element_type),
            ir.RankedTensorType.get(g_shape, g_type.element_type),
        ]
        operands = [dz, rsigma, x, gamma]
        operand_shapes = [go_shape, rsigma_shape, x_shape, g_shape]
        args = CustomCallArgsWrapper(out_types, operands, operand_shapes)

        opaque = transformer_engine_jax.pack_norm_descriptor(
            batch_size,
            hidden_size,
            jax_dtype_to_te_dtype(x_aval.dtype),
            jax_dtype_to_te_dtype(gamma_aval.dtype),
            False,    # RMSNorm doesn't support zero_centered_gamma
            epsilon,
        )

        out = custom_caller(RmsNormBwdPrimitive.name, args, opaque, False)

        return out

    @staticmethod
    def impl(dz, x, rsigma, gamma, epsilon):
        assert RmsNormBwdPrimitive.inner_primitive is not None
        dx, dgamma = RmsNormBwdPrimitive.inner_primitive.bind(dz, x, rsigma, gamma, epsilon=epsilon)
        return dx, dgamma

    @staticmethod
    def batcher(batched_args, batch_dims, *, epsilon):
        assert RmsNormBwdPrimitive.outer_primitive is not None
        dz, x, rsigma, gamma = batched_args
        _, x_bdim, _, gamma_bdim = batch_dims

        out_bdims = x_bdim, gamma_bdim
        return RmsNormBwdPrimitive.outer_primitive.bind(dz, x, rsigma, gamma,
                                                        epsilon=epsilon), out_bdims

    @staticmethod
    def infer_sharding_from_operands(epsilon, mesh, arg_infos, result_infos):
        del epsilon, result_infos
        x_spec = get_padded_spec(arg_infos[1])
        if x_spec[-1] is not None:
            warnings.warn(
                f"Does not support to shard hidden dim in {RmsNormBwdPrimitive.name}! " \
                f"Force to not shard the hidden dim, which might introduce extra collective ops, " \
                f"and hurt performance."
            )
        g_spec = get_padded_spec(arg_infos[3])
        dx_sharding = NamedSharding(mesh, PartitionSpec(*x_spec[:-1], None))
        dgamma_sharding = NamedSharding(mesh, PartitionSpec(*g_spec))
        return dx_sharding, dgamma_sharding

    @staticmethod
    def partition(epsilon, mesh, arg_infos, result_infos):
        del result_infos
        x_spec = get_padded_spec(arg_infos[1])
        if x_spec[-1] is not None:
            warnings.warn(
                f"Does not support to shard hidden dim in {RmsNormBwdPrimitive.name}! " \
                f"Force to not shard the hidden dim, which might introduce extra collective ops, " \
                f"and hurt performance."
            )
        g_spec = get_padded_spec(arg_infos[3])
        dx_sharding = NamedSharding(mesh, PartitionSpec(*x_spec[:-1], None))
        dgamma_sharding = NamedSharding(mesh, PartitionSpec(*g_spec))
        out_shardings = dx_sharding, dgamma_sharding
        x_shardings = (dx_sharding,) * 2
        rsigma_sharding = NamedSharding(mesh, PartitionSpec(*x_spec[:-1]))
        arg_shardings = (*x_shardings, rsigma_sharding, NamedSharding(mesh, PartitionSpec(*g_spec)))

        def sharded_impl(dz, x, rsigma, gamma):
            local_dx, local_dgamma = \
                RmsNormBwdPrimitive.impl(dz, x, rsigma, gamma, epsilon=epsilon)
            global_dgamma = all_reduce_sum_along_dp_fsdp(local_dgamma)
            return local_dx, global_dgamma

        return mesh, sharded_impl, out_shardings, arg_shardings


register_primitive(RmsNormBwdPrimitive)


def rmsnorm_bwd(dz: jnp.ndarray, x: jnp.ndarray, rsigma: jnp.ndarray, gamma: jnp.ndarray,
                epsilon: float):
    """
    Wrapper for TE layernorm bwd
    """
    return RmsNormBwdPrimitive.outer_primitive.bind(dz, x, rsigma, gamma, epsilon=epsilon)


class SoftmaxPrimitive(BasePrimitive):
    """
    Softmax Primitive
    """
    max_k_seqlen_supported = 4096

    @staticmethod
    @abstractmethod
    def is_kernel_available(batch: int, heads: int, q_seqlen: int, k_seqlen: int,
                            dtype: jnp.dtype) -> bool:
        """Check Softmax kernel availability based on size"""
        raise NotImplementedError

    @staticmethod
    def get_batch_per_block(k_seqlen: int) -> int:
        """Get batch per CTA in Softmax kernels"""
        threads_per_warp = 32
        threads_per_block = 128    # Depends on the kernel implmentation

        pow2 = 1 << (k_seqlen - 1).bit_length()
        warp_size = pow2 if pow2 < threads_per_warp else threads_per_warp
        batches_per_warp = 2 if pow2 <= 128 else 1
        warps_per_block = threads_per_block // warp_size
        batches_per_block = warps_per_block * batches_per_warp
        return batches_per_block

    @staticmethod
    def forward_abstract(logits_aval, scale_factor):    # pylint: disable=unused-argument
        """
        softmax_forward abstract
        """
        del scale_factor
        i_dtype = dtypes.canonicalize_dtype(logits_aval.dtype)
        assert i_dtype in [jnp.float16, jnp.bfloat16]
        i_shape = logits_aval.shape
        # Assume [...Batch, Head, Q_Seqlen, K_Seqlen]
        q_seqlen = i_shape[-2]
        k_seqlen = i_shape[-1]
        assert k_seqlen <= SoftmaxPrimitive.max_k_seqlen_supported
        assert q_seqlen > 1

        out_aval = core.raise_to_shaped(logits_aval)
        return out_aval

    @staticmethod
    def forward_lowering(name, ctx, logits, *, scale_factor):
        """
        softmax_forward lowering rules
        """
        i_aval, = ctx.avals_in
        i_type = ir.RankedTensorType(logits.type)
        i_shape = i_type.shape
        # Assume [...Batch, Head, Q_Seqlen, K_Seqlen]
        batch = reduce(operator.mul, i_shape[:-3])
        pad_batch = batch
        heads = i_shape[-3]
        q_seqlen = i_shape[-2]
        k_seqlen = i_shape[-1]

        out_types = [ir.RankedTensorType.get(i_shape, i_type.element_type)]
        operands = [logits]
        operand_shapes = [i_shape]
        args = CustomCallArgsWrapper(out_types, operands, operand_shapes)

        opaque = transformer_engine_jax.pack_softmax_descriptor(batch, pad_batch, heads, q_seqlen,
                                                                k_seqlen,
                                                                jax_dtype_to_te_dtype(i_aval.dtype),
                                                                scale_factor)

        out = custom_caller(name, args, opaque, False)

        return [out]

    @staticmethod
    def forward_impl(primitive, logits, scale_factor):
        """
        softmax_forward implementation
        """
        assert primitive is not None
        output = primitive.bind(logits, scale_factor=scale_factor)
        return output

    @staticmethod
    def forward_batcher(primitive, batched_args, batch_dims, *, scale_factor):
        """
        softmax_forward batcher
        """
        assert primitive is not None
        logits, = batched_args
        logits_bdim, = batch_dims

        out_bdims = logits_bdim
        return primitive.bind(logits, scale_factor=scale_factor), out_bdims

    @staticmethod
    def forward_infer_sharding_from_operands(scale_factor, mesh, arg_infos, result_infos):
        """
        softmax_forward infer_sharding_from_operands
        """
        del scale_factor, result_infos    # Unused.
        logits_spec = get_padded_spec(arg_infos[0])
        out_sharding = NamedSharding(mesh, PartitionSpec(*logits_spec))
        return out_sharding

    @staticmethod
    def forward_partition(impl, scale_factor, mesh, arg_infos, result_infos):
        """
        softmax_forward partitioning
        """
        del result_infos
        logits_spec = NamedSharding(mesh, PartitionSpec(*get_padded_spec(arg_infos[0])))
        out_spec = logits_spec
        arg_shardings = (logits_spec,)
        out_shardings = out_spec
        impl = partial(impl, scale_factor=scale_factor)
        return mesh, impl, out_shardings, arg_shardings

    @staticmethod
    def backward_abstract(dz_aval, softmax_out_aval, scale_factor=None):    # pylint: disable=unused-argument
        """
        softmax_backward infer_sharding_from_operands
        """
        dz_dtype = dtypes.canonicalize_dtype(dz_aval.dtype)
        softmax_out_dtype = dtypes.canonicalize_dtype(softmax_out_aval.dtype)
        assert dz_dtype == softmax_out_dtype
        assert dz_dtype in [jnp.float16, jnp.bfloat16]
        assert softmax_out_dtype in [jnp.float16, jnp.bfloat16]

        assert dz_aval.shape == softmax_out_aval.shape

        dx_aval = core.raise_to_shaped(softmax_out_aval)
        return dx_aval

    @staticmethod
    def backward_lowering(name, ctx, dz, softmax_out, *, scale_factor):
        """
        softmax_backward lowering rules
        """
        dz_aval, _ = ctx.avals_in

        dz_type = ir.RankedTensorType(dz.type)
        dz_shape = dz_type.shape

        # Assume [...Batch, Head, Q_Seqlen, K_Seqlen]
        batch = reduce(operator.mul, dz_shape[:-3])
        pad_batch = batch    # unused
        heads = dz_shape[-3]
        q_seqlen = dz_shape[-2]
        k_seqlen = dz_shape[-1]

        softmax_out_type = ir.RankedTensorType(softmax_out.type)
        softmax_out_shape = softmax_out_type.shape

        out_types = [ir.RankedTensorType.get(softmax_out_shape, softmax_out_type.element_type)]
        operands = [dz, softmax_out]
        operand_shapes = [dz_shape, softmax_out_shape]
        args = CustomCallArgsWrapper(out_types, operands, operand_shapes)

        opaque = transformer_engine_jax.pack_softmax_descriptor(
            batch, pad_batch, heads, q_seqlen, k_seqlen, jax_dtype_to_te_dtype(dz_aval.dtype),
            scale_factor)

        out = custom_caller(name, args, opaque, False)

        return [out]

    @staticmethod
    def backward_impl(primitive, dz, softmax_out, scale_factor):
        """
        softmax_backward implementation
        """
        assert primitive is not None
        dx = primitive.bind(dz, softmax_out, scale_factor=scale_factor)
        return dx

    @staticmethod
    def backward_batcher(primitive, batched_args, batch_dims, *, scale_factor):
        """
        softmax_backward batcher
        """
        assert primitive is not None
        dz, softmax_out = batched_args
        _, softmax_out_bdim = batch_dims

        out_bdims = softmax_out_bdim
        return primitive.bind(dz, softmax_out, scale_factor=scale_factor), out_bdims

    @staticmethod
    def backward_infer_sharding_from_operands(scale_factor, mesh, arg_infos, result_infos):
        """
        softmax_backward infer_sharding_from_operands
        """
        del scale_factor, result_infos    # Unused.
        softmax_out_spec = get_padded_spec(arg_infos[1])
        dx_sharding = NamedSharding(mesh, PartitionSpec(*softmax_out_spec))
        return dx_sharding

    @staticmethod
    def backward_partition(impl, scale_factor, mesh, arg_infos, result_infos):
        """
        softmax_backward partition
        """
        del result_infos
        dz_spec = NamedSharding(mesh, PartitionSpec(*get_padded_spec(arg_infos[0])))
        softmax_out_spec = NamedSharding(mesh, PartitionSpec(*get_padded_spec(arg_infos[1])))
        dx_spec = softmax_out_spec
        arg_shardings = (dz_spec, softmax_out_spec)
        out_shardings = dx_spec
        impl = partial(impl, scale_factor=scale_factor)
        return mesh, impl, out_shardings, arg_shardings


class ScaledSoftmaxFwdPrimitive(SoftmaxPrimitive):
    """
    Scaled Softmax Fwd Primitive
    """
    name = "te_scaled_softmax_forward"
    multiple_results = False
    impl_static_args = (1,)    # scale_factor
    inner_primitive = None
    outer_primitive = None

    @staticmethod
    def is_kernel_available(batch: int, heads: int, q_seqlen: int, k_seqlen: int,
                            dtype: jnp.dtype) -> bool:
        """Check Softmax kernel availability based on size"""
        attn_batches = batch * heads

        if (dtype in [jnp.float16, jnp.bfloat16]
                and 16 < k_seqlen <= SoftmaxPrimitive.max_k_seqlen_supported
        # k_seqlen must be 16 ~ 4096
                and q_seqlen % 4 == 0    # q_seqlen must be divisor of 4
                and attn_batches % 4 == 0    # batch * heads must be divisor of 4
           ):
            if 0 <= k_seqlen <= SoftmaxPrimitive.max_k_seqlen_supported:
                batch_per_block = SoftmaxPrimitive.get_batch_per_block(k_seqlen)
                return q_seqlen % batch_per_block == 0
        return False

    @staticmethod
    def abstract(logits_aval, scale_factor):    # pylint: disable=unused-argument
        """
        te_scaled_softmax_forward abstract
        """
        return SoftmaxPrimitive.forward_abstract(logits_aval, scale_factor)

    @staticmethod
    def lowering(ctx, logits, *, scale_factor):
        """
        te_scaled_softmax_forward lowering rules
        """
        return SoftmaxPrimitive.forward_lowering(ScaledSoftmaxFwdPrimitive.name,
                                                 ctx,
                                                 logits,
                                                 scale_factor=scale_factor)

    @staticmethod
    def impl(logits, scale_factor):
        return SoftmaxPrimitive.forward_impl(ScaledSoftmaxFwdPrimitive.inner_primitive, logits,
                                             scale_factor)

    @staticmethod
    def batcher(batched_args, batch_dims, *, scale_factor):
        return SoftmaxPrimitive.forward_batcher(ScaledSoftmaxFwdPrimitive.outer_primitive,
                                                batched_args,
                                                batch_dims,
                                                scale_factor=scale_factor)

    @staticmethod
    def infer_sharding_from_operands(scale_factor, mesh, arg_infos, result_infos):
        return SoftmaxPrimitive.forward_infer_sharding_from_operands(scale_factor, mesh, arg_infos,
                                                                     result_infos)

    @staticmethod
    def partition(scale_factor, mesh, arg_infos, result_infos):
        return SoftmaxPrimitive.forward_partition(ScaledSoftmaxFwdPrimitive.impl, scale_factor,
                                                  mesh, arg_infos, result_infos)


register_primitive(ScaledSoftmaxFwdPrimitive)


def scaled_softmax_fwd(logits: jnp.ndarray, scale_factor: float) -> jnp.ndarray:
    """
    scaled_softmax_forward wrapper
    Return FP16/BF16 tensor
    """
    return ScaledSoftmaxFwdPrimitive.outer_primitive.bind(logits, scale_factor=scale_factor)


class ScaledSoftmaxBwdPrimitive(SoftmaxPrimitive):
    """
    Scaled Softmax Bwd Primitive
    """
    name = "te_scaled_softmax_backward"
    multiple_results = False
    impl_static_args = (2,)    # scale_factor
    inner_primitive = None
    outer_primitive = None

    @staticmethod
    def is_kernel_available(batch: int, heads: int, q_seqlen: int, k_seqlen: int,
                            dtype: jnp.dtype) -> bool:
        """Check Softmax kernel availability based on size"""
        return ScaledSoftmaxFwdPrimitive.is_kernel_available(batch, heads, q_seqlen, k_seqlen,
                                                             dtype)

    @staticmethod
    def abstract(dz_aval, softmax_out_aval, scale_factor):
        """
        te_scaled_softmax_backward abstract
        """
        return SoftmaxPrimitive.backward_abstract(dz_aval, softmax_out_aval, scale_factor)

    @staticmethod
    def lowering(ctx, dz, softmax_out, *, scale_factor):
        """
        te_scaled_softmax_backward lowering rules
        """
        out = SoftmaxPrimitive.backward_lowering(ScaledSoftmaxBwdPrimitive.name,
                                                 ctx,
                                                 dz,
                                                 softmax_out,
                                                 scale_factor=scale_factor)

        return out

    @staticmethod
    def impl(dz, softmax_out, scale_factor):
        return SoftmaxPrimitive.backward_impl(ScaledSoftmaxBwdPrimitive.inner_primitive,
                                              dz,
                                              softmax_out,
                                              scale_factor=scale_factor)

    @staticmethod
    def batcher(batched_args, batch_dims, *, scale_factor):
        return SoftmaxPrimitive.backward_batcher(ScaledSoftmaxBwdPrimitive.outer_primitive,
                                                 batched_args,
                                                 batch_dims,
                                                 scale_factor=scale_factor)

    @staticmethod
    def infer_sharding_from_operands(scale_factor, mesh, arg_infos, result_infos):
        return SoftmaxPrimitive.backward_infer_sharding_from_operands(scale_factor, mesh, arg_infos,
                                                                      result_infos)

    @staticmethod
    def partition(scale_factor, mesh, arg_infos, result_infos):
        return SoftmaxPrimitive.backward_partition(ScaledSoftmaxBwdPrimitive.impl, scale_factor,
                                                   mesh, arg_infos, result_infos)


register_primitive(ScaledSoftmaxBwdPrimitive)


def scaled_softmax_bwd(dz: jnp.ndarray, softmax_out: jnp.ndarray,
                       scale_factor: float) -> jnp.ndarray:
    """
    scaled_backward wrapper
    Return FP16/BF16 tensor
    """
    return ScaledSoftmaxBwdPrimitive.outer_primitive.bind(dz,
                                                          softmax_out,
                                                          scale_factor=scale_factor)


class ScaledMaskedSoftmaxFwdPrimitive(SoftmaxPrimitive):
    """
    Scaled Masked Softmax Fwd Primitive
    """
    name = "te_scaled_masked_softmax_forward"
    multiple_results = False
    impl_static_args = (2,)    # scale_factor
    inner_primitive = None
    outer_primitive = None

    @staticmethod
    def is_kernel_available(batch: int, heads: int, q_seqlen: int, k_seqlen: int,
                            dtype: jnp.dtype) -> bool:
        """Check Softmax kernel availability based on size"""
        attn_batches = batch * heads

        if (dtype in [jnp.float16, jnp.bfloat16]
                and 16 < k_seqlen <= SoftmaxPrimitive.max_k_seqlen_supported
        # k_seqlen must be 16 ~ 4096
                and q_seqlen % 4 == 0    # q_seqlen must be divisor of 4
                and attn_batches % 4 == 0    # batch * heads must be divisor of 4
           ):
            if 0 <= k_seqlen <= SoftmaxPrimitive.max_k_seqlen_supported:
                batch_per_block = SoftmaxPrimitive.get_batch_per_block(k_seqlen)
                return q_seqlen % batch_per_block == 0
        return False

    @staticmethod
    def abstract(logits_aval, mask_aval, scale_factor):    # pylint: disable=unused-argument
        """
        te_scaled_masked_softmax_forward abstract
        """

        i_dtype = dtypes.canonicalize_dtype(logits_aval.dtype)
        assert i_dtype in [jnp.float16, jnp.bfloat16]
        i_shape = logits_aval.shape

        # Assume [...Batch, Head, Q_Seqlen, K_Seqlen]
        batch = reduce(operator.mul, i_shape[:-3])
        q_seqlen = i_shape[-2]
        k_seqlen = i_shape[-1]
        assert k_seqlen <= SoftmaxPrimitive.max_k_seqlen_supported
        assert q_seqlen > 1

        mask_dtype = dtypes.canonicalize_dtype(mask_aval.dtype)
        assert mask_dtype in [
            jnp.uint8,
        ]
        mask_shape = mask_aval.shape
        pad_batch = batch = reduce(operator.mul, mask_shape[:-3])
        assert pad_batch in (1, batch)    # 1 means broadcast
        assert mask_shape[-3] == 1    # 1 means broadcast
        assert mask_shape[-2] == q_seqlen
        assert mask_shape[-1] == k_seqlen

        out_aval = core.raise_to_shaped(logits_aval)
        return out_aval

    @staticmethod
    def lowering(ctx, logits, mask, *, scale_factor):
        """
        te_scaled_masked_softmax_forward lowering rules
        """

        logits_aval, _ = ctx.avals_in
        i_type = ir.RankedTensorType(logits.type)
        i_shape = i_type.shape
        # Assume [...Batch, Head, Q_Seqlen, K_Seqlen]
        batch = reduce(operator.mul, i_shape[:-3])
        heads = i_shape[-3]
        q_seqlen = i_shape[-2]
        k_seqlen = i_shape[-1]

        mask_type = ir.RankedTensorType(mask.type)
        mask_shape = mask_type.shape
        pad_batch = reduce(operator.mul, mask_shape[:-3])

        out_types = [ir.RankedTensorType.get(i_shape, i_type.element_type)]
        operands = [logits, mask]
        operand_shapes = [i_shape, mask_shape]
        args = CustomCallArgsWrapper(out_types, operands, operand_shapes)

        opaque = transformer_engine_jax.pack_softmax_descriptor(
            batch, pad_batch, heads, q_seqlen, k_seqlen, jax_dtype_to_te_dtype(logits_aval.dtype),
            scale_factor)

        out = custom_caller(ScaledMaskedSoftmaxFwdPrimitive.name, args, opaque, False)

        return [out]

    @staticmethod
    def impl(logits, mask, scale_factor):
        assert ScaledMaskedSoftmaxFwdPrimitive.inner_primitive is not None
        output = ScaledMaskedSoftmaxFwdPrimitive.inner_primitive.bind(logits,
                                                                      mask,
                                                                      scale_factor=scale_factor)
        return output

    @staticmethod
    def batcher(batched_args, batch_dims, *, scale_factor):
        assert ScaledMaskedSoftmaxFwdPrimitive.outer_primitive is not None
        logits, mask = batched_args
        logits_bdim, _ = batch_dims

        out_bdims = logits_bdim
        return ScaledMaskedSoftmaxFwdPrimitive.outer_primitive.bind(
            logits, mask, scale_factor=scale_factor), out_bdims

    @staticmethod
    def infer_sharding_from_operands(scale_factor, mesh, arg_infos, result_infos):
        del scale_factor, result_infos    # Unused.
        logits_spec = get_padded_spec(arg_infos[0])
        out_sharding = NamedSharding(mesh, PartitionSpec(*logits_spec))
        return out_sharding

    @staticmethod
    def partition(scale_factor, mesh, arg_infos, result_infos):
        del result_infos
        logits_spec = NamedSharding(mesh, PartitionSpec(*get_padded_spec(arg_infos[0])))
        mask_spec = NamedSharding(mesh, PartitionSpec(*get_padded_spec(arg_infos[1])))
        out_spec = logits_spec
        arg_shardings = (logits_spec, mask_spec)
        out_shardings = out_spec
        impl = partial(ScaledMaskedSoftmaxFwdPrimitive.impl, scale_factor=scale_factor)
        return mesh, impl, out_shardings, arg_shardings


register_primitive(ScaledMaskedSoftmaxFwdPrimitive)


def scaled_masked_softmax_fwd(logits: jnp.ndarray, mask: jnp.ndarray,
                              scale_factor: float) -> jnp.ndarray:
    """
    scaled_masked_softmax_forward wrapper
    Return FP16/BF16 tensor
    """
    return ScaledMaskedSoftmaxFwdPrimitive.outer_primitive.bind(logits,
                                                                mask,
                                                                scale_factor=scale_factor)


class ScaledMaskedSoftmaxBwdPrimitive(SoftmaxPrimitive):
    """
    Scaled Masked Softmax Bwd Primitive
    """
    name = "te_scaled_masked_softmax_backward"
    multiple_results = False
    impl_static_args = (2,)    # scale_factor
    inner_primitive = None
    outer_primitive = None

    @staticmethod
    def is_kernel_available(batch: int, heads: int, q_seqlen: int, k_seqlen: int,
                            dtype: jnp.dtype) -> bool:
        """Check Softmax kernel availability based on size"""
        return ScaledSoftmaxFwdPrimitive.is_kernel_available(batch, heads, q_seqlen, k_seqlen,
                                                             dtype)

    @staticmethod
    def abstract(dz_aval, softmax_out_aval, *, scale_factor):
        """
        te_scaled_upper_triang_masked_backward abstract
        """
        return SoftmaxPrimitive.backward_abstract(dz_aval, softmax_out_aval, scale_factor)

    @staticmethod
    def lowering(ctx, dz, softmax_out, *, scale_factor):
        """
        te_scaled_upper_triang_masked_backward lowering rules
        """
        out = SoftmaxPrimitive.backward_lowering(ScaledMaskedSoftmaxBwdPrimitive.name,
                                                 ctx,
                                                 dz,
                                                 softmax_out,
                                                 scale_factor=scale_factor)

        return out

    @staticmethod
    def impl(dz, softmax_out, scale_factor):
        return SoftmaxPrimitive.backward_impl(ScaledMaskedSoftmaxBwdPrimitive.inner_primitive,
                                              dz,
                                              softmax_out,
                                              scale_factor=scale_factor)

    @staticmethod
    def batcher(batched_args, batch_dims, *, scale_factor):
        return SoftmaxPrimitive.backward_batcher(ScaledMaskedSoftmaxBwdPrimitive.outer_primitive,
                                                 batched_args,
                                                 batch_dims,
                                                 scale_factor=scale_factor)

    @staticmethod
    def infer_sharding_from_operands(scale_factor, mesh, arg_infos, result_infos):
        return SoftmaxPrimitive.backward_infer_sharding_from_operands(scale_factor, mesh, arg_infos,
                                                                      result_infos)

    @staticmethod
    def partition(scale_factor, mesh, arg_infos, result_infos):
        return SoftmaxPrimitive.backward_partition(ScaledMaskedSoftmaxBwdPrimitive.impl,
                                                   scale_factor, mesh, arg_infos, result_infos)


register_primitive(ScaledMaskedSoftmaxBwdPrimitive)


def scaled_masked_softmax_bwd(dz: jnp.ndarray, softmax_out: jnp.ndarray,
                              scale_factor: float) -> jnp.ndarray:
    """
    scaled_masked_backward wrapper
    Return FP16/BF16 tensor
    """
    return ScaledMaskedSoftmaxBwdPrimitive.outer_primitive.bind(dz,
                                                                softmax_out,
                                                                scale_factor=scale_factor)


class ScaledUpperTriangMaskedSoftmaxFwdPrimitive(SoftmaxPrimitive):
    """
    Scaled Upper Triang Masked Softmax Fwd Primitive
    """
    name = "te_scaled_upper_triang_masked_softmax_forward"
    multiple_results = False
    impl_static_args = (1,)    # scale_factor
    inner_primitive = None
    outer_primitive = None

    @staticmethod
    def is_kernel_available(batch: int, heads: int, q_seqlen: int, k_seqlen: int,
                            dtype: jnp.dtype) -> bool:
        """Check Softmax kernel availability based on size"""
        attn_batches = batch * heads

        if (dtype in [jnp.float16, jnp.bfloat16]
                and 16 < k_seqlen <= SoftmaxPrimitive.max_k_seqlen_supported
        # k_seqlen must be 16 ~ 4096
                and q_seqlen % 4 == 0    # q_seqlen must be divisor of 4
                and attn_batches % 4 == 0    # batch * heads must be divisor of 4
           ):
            if 0 <= k_seqlen <= SoftmaxPrimitive.max_k_seqlen_supported:
                batch_per_block = SoftmaxPrimitive.get_batch_per_block(k_seqlen)
                return attn_batches % batch_per_block == 0
        return False

    @staticmethod
    def abstract(logits_aval, scale_factor):    # pylint: disable=unused-argument
        """
        te_scaled_upper_triang_masked_softmax_forward abstract
        """
        q_seqlen = logits_aval.shape[2]
        k_seqlen = logits_aval.shape[3]
        assert q_seqlen == k_seqlen
        return SoftmaxPrimitive.forward_abstract(logits_aval, scale_factor)

    @staticmethod
    def lowering(ctx, logits, *, scale_factor):
        """
        te_scaled_upper_triang_masked_softmax_forward lowering rules
        """
        return SoftmaxPrimitive.forward_lowering(ScaledUpperTriangMaskedSoftmaxFwdPrimitive.name,
                                                 ctx,
                                                 logits,
                                                 scale_factor=scale_factor)

    @staticmethod
    def impl(logits, scale_factor):
        return SoftmaxPrimitive.forward_impl(
            ScaledUpperTriangMaskedSoftmaxFwdPrimitive.inner_primitive, logits, scale_factor)

    @staticmethod
    def batcher(batched_args, batch_dims, *, scale_factor):
        return SoftmaxPrimitive.forward_batcher(
            ScaledUpperTriangMaskedSoftmaxFwdPrimitive.outer_primitive,
            batched_args,
            batch_dims,
            scale_factor=scale_factor)

    @staticmethod
    def infer_sharding_from_operands(scale_factor, mesh, arg_infos, result_infos):
        return SoftmaxPrimitive.forward_infer_sharding_from_operands(scale_factor, mesh, arg_infos,
                                                                     result_infos)

    @staticmethod
    def partition(scale_factor, mesh, arg_infos, result_infos):
        return SoftmaxPrimitive.forward_partition(ScaledUpperTriangMaskedSoftmaxFwdPrimitive.impl,
                                                  scale_factor, mesh, arg_infos, result_infos)


register_primitive(ScaledUpperTriangMaskedSoftmaxFwdPrimitive)


def scaled_upper_triang_masked_softmax_fwd(logits: jnp.ndarray, scale_factor: float) -> jnp.ndarray:
    """
    scaled_upper_triang_masked_softmax_forward wrapper
    Return FP16/BF16 tensor
    """
    return ScaledUpperTriangMaskedSoftmaxFwdPrimitive.outer_primitive.bind(
        logits, scale_factor=scale_factor)


class ScaledUpperTriangMaskedSoftmaxBwdPrimitive(SoftmaxPrimitive):
    """
    Scaled Upper Triang Masked Softmax Bwd Primitive
    """
    name = "te_scaled_upper_triang_masked_softmax_backward"
    multiple_results = False
    impl_static_args = (2,)    # scale_factor
    inner_primitive = None
    outer_primitive = None

    @staticmethod
    def is_kernel_available(batch: int, heads: int, q_seqlen: int, k_seqlen: int,
                            dtype: jnp.dtype) -> bool:
        """Check Softmax kernel availability based on size"""
        return ScaledUpperTriangMaskedSoftmaxFwdPrimitive.is_kernel_available(
            batch, heads, q_seqlen, k_seqlen, dtype)

    @staticmethod
    def abstract(dz_aval, softmax_out_aval, *, scale_factor):
        """
        te_scaled_upper_triang_masked_backward abstract
        """
        return SoftmaxPrimitive.backward_abstract(dz_aval, softmax_out_aval, scale_factor)

    @staticmethod
    def lowering(ctx, dz, softmax_out, *, scale_factor):
        """
        te_scaled_upper_triang_masked_backward lowering rules
        """
        out = SoftmaxPrimitive.backward_lowering(ScaledUpperTriangMaskedSoftmaxBwdPrimitive.name,
                                                 ctx,
                                                 dz,
                                                 softmax_out,
                                                 scale_factor=scale_factor)

        return out

    @staticmethod
    def impl(dz, softmax_out, scale_factor):
        return SoftmaxPrimitive.backward_impl(
            ScaledUpperTriangMaskedSoftmaxBwdPrimitive.inner_primitive,
            dz,
            softmax_out,
            scale_factor=scale_factor)

    @staticmethod
    def batcher(batched_args, batch_dims, *, scale_factor):
        return SoftmaxPrimitive.backward_batcher(
            ScaledUpperTriangMaskedSoftmaxBwdPrimitive.outer_primitive,
            batched_args,
            batch_dims,
            scale_factor=scale_factor)

    @staticmethod
    def infer_sharding_from_operands(scale_factor, mesh, arg_infos, result_infos):
        return SoftmaxPrimitive.backward_infer_sharding_from_operands(scale_factor, mesh, arg_infos,
                                                                      result_infos)

    @staticmethod
    def partition(scale_factor, mesh, arg_infos, result_infos):
        return SoftmaxPrimitive.backward_partition(ScaledUpperTriangMaskedSoftmaxBwdPrimitive.impl,
                                                   scale_factor, mesh, arg_infos, result_infos)


register_primitive(ScaledUpperTriangMaskedSoftmaxBwdPrimitive)


def scaled_upper_triang_masked_softmax_bwd(dz: jnp.ndarray, softmax_out: jnp.ndarray,
                                           scale_factor: float) -> jnp.ndarray:
    """
    scaled_upper_triang_masked_backward wrapper
    Return FP16/BF16 tensor
    """
    return ScaledUpperTriangMaskedSoftmaxBwdPrimitive.outer_primitive.bind(
        dz, softmax_out, scale_factor=scale_factor)


@dataclass(frozen=True)
class FusedAttnHelper:
    """
    Helper for the fused attention backend
    """

    q_type: jnp.dtype
    kv_type: jnp.dtype
    attn_bias_type: NVTE_Bias_Type
    attn_mask_type: NVTE_Mask_Type
    dropout_probability: float
    max_seqlen_q: int
    max_seqlen_kv: int
    head_dim: int

    def is_fused_attn_kernel_available(self):
        """Check if there is available fused attention kernel"""
        return self.get_fused_attn_backend() != NVTE_Fused_Attn_Backend.NVTE_No_Backend

    def get_fused_attn_backend(self):
        """Get the fused attention kernel backend"""
        return transformer_engine_jax.get_fused_attn_backend(
            jax_dtype_to_te_dtype(self.q_type), jax_dtype_to_te_dtype(self.kv_type),
            NVTE_QKV_Layout.NVTE_QKV_INTERLEAVED, self.attn_bias_type, self.attn_mask_type,
            self.dropout_probability, self.max_seqlen_q, self.max_seqlen_kv, self.head_dim)


@dataclass(frozen=True)
class _FusedAttnRNGStateChecker:
    """
    Checker for guarding the fused attention rng state.
    The fused attention backend requires a 64 bits seed and a 64 bits offset.
    However, JAX doesn't enable 64 bits by default,
    so we have to emulate seed as two 32 bits array.
    The offset calculation is maintained in the backend.
    """
    rng_state_dtype: jnp.dtype = jnp.uint32
    # (seed,) with internal dtype int64
    seed_size: int = 2
    # (seed, offset) with internal dtype int64
    rng_state_size: int = 2 * 2

    def check_seed(self, seed, dropout_probability, is_training):
        """
        Check the seed and convert the data type of seed if possible.
        """
        # Jax can't bind None, create a dummy tensor for None
        if seed is None:
            dropout_enabled = dropout_probability > 0 and is_training
            assert not dropout_enabled, "seed is not allowed to be None when dropout is enabled."
            seed = jnp.zeros(2, dtype=self.rng_state_dtype)
            seed = jax_random.split(seed, num_of_devices())

        if seed.dtype != self.rng_state_dtype:
            warnings.warn(
                f"Requested {seed.dtype=} is not available, and will be "
                f"casted to dtype {self.rng_state_dtype}. "
                f"Please use threefry/rbg/unsafe_rbg PRNG implementations to remove this warning.")
            seed = seed.astype(self.rng_state_dtype)

        assert seed.dtype == self.rng_state_dtype
        # Backend takes an int64_t seed, so only the first two u32 elements are taken
        assert seed.size >= self.seed_size

        return seed


class SelfFusedAttnFwdPrimitive(BasePrimitive):
    """
    Self Fused Attention Forward Primitive
    """
    name = "te_self_fused_attn_forward"
    multiple_results = True
    impl_static_args = (4, 5, 6, 7, 8)
    inner_primitive = None
    outer_primitive = None

    @staticmethod
    def abstract(qkv_aval, bias_aval, cu_seqlen_aval, seed_aval, *, attn_bias_type, attn_mask_type,
                 scaling_factor, dropout_probability, is_training):
        """
        Self fused attention fwd abstract
        """
        del cu_seqlen_aval, scaling_factor, is_training
        qkv_dtype = dtypes.canonicalize_dtype(qkv_aval.dtype)
        batch_shape = qkv_aval.shape[:-4]
        max_seqlen = qkv_aval.shape[-4]
        nqkv = qkv_aval.shape[-3]
        num_head = qkv_aval.shape[-2]
        head_dim = qkv_aval.shape[-1]
        assert nqkv == 3
        assert qkv_aval.dtype == bias_aval.dtype

        output_shape = (*batch_shape, max_seqlen, num_head, head_dim)
        output_dtype = qkv_dtype

        backend = FusedAttnHelper(qkv_dtype, qkv_dtype, attn_bias_type, attn_mask_type,
                                  dropout_probability, max_seqlen, max_seqlen,
                                  head_dim).get_fused_attn_backend()

        if backend == NVTE_Fused_Attn_Backend.NVTE_F16_max512_seqlen:
            softmax_aux_shape = (*batch_shape, num_head, max_seqlen, max_seqlen)
            softmax_dtype = qkv_dtype
        elif backend == NVTE_Fused_Attn_Backend.NVTE_F16_arbitrary_seqlen:
            softmax_aux_shape = (*batch_shape, num_head, max_seqlen, 1)
            softmax_dtype = dtypes.canonicalize_dtype(jnp.float32)
        else:
            raise ValueError(f'Not supported {backend=}')

        checker = _FusedAttnRNGStateChecker()
        seed_dtype = dtypes.canonicalize_dtype(seed_aval.dtype)
        assert seed_dtype == checker.rng_state_dtype
        rng_state_shape = (seed_aval.shape[0], checker.rng_state_size)
        rng_state_dtype = seed_dtype

        out_aval = qkv_aval.update(shape=output_shape, dtype=output_dtype)
        softmax_aux_aval = qkv_aval.update(shape=softmax_aux_shape, dtype=softmax_dtype)
        rng_state_aval = qkv_aval.update(shape=rng_state_shape, dtype=rng_state_dtype)
        return out_aval, softmax_aux_aval, rng_state_aval

    @staticmethod
    def lowering(ctx, qkv, bias, cu_seqlen, seed, *, attn_bias_type, attn_mask_type, scaling_factor,
                 dropout_probability, is_training):
        """
        Self fused attention fwd lowering rules
        """
        qkv_aval, _, _, _ = ctx.avals_in

        batch = reduce(operator.mul, qkv_aval.shape[:-4])
        max_seqlen = qkv_aval.shape[-4]
        num_head = qkv_aval.shape[-2]
        head_dim = qkv_aval.shape[-1]

        operands = [qkv, bias, cu_seqlen, seed]
        operand_shapes = map(lambda x: x.type.shape, operands)
        out_types = [
            ir.RankedTensorType.get(output.shape, mlir.dtype_to_ir_type(output.dtype))
            for output in ctx.avals_out
        ]

        args = CustomCallArgsWrapper(out_types, operands, operand_shapes)
        opaque = transformer_engine_jax.pack_fused_attn_descriptor(
            batch, num_head, max_seqlen, max_seqlen, head_dim, scaling_factor, dropout_probability,
            attn_bias_type, attn_mask_type, jax_dtype_to_te_dtype(qkv_aval.dtype), is_training)

        out = custom_caller(SelfFusedAttnFwdPrimitive.name, args, opaque, has_side_effect=False)

        return out

    @staticmethod
    def impl(qkv, bias, cu_seqlen, seed, attn_bias_type, attn_mask_type, scaling_factor,
             dropout_probability, is_training):
        assert SelfFusedAttnFwdPrimitive.inner_primitive is not None
        output, softmax_aux, rng_state = SelfFusedAttnFwdPrimitive.inner_primitive.bind(
            qkv,
            bias,
            cu_seqlen,
            seed,
            attn_bias_type=attn_bias_type,
            attn_mask_type=attn_mask_type,
            scaling_factor=scaling_factor,
            dropout_probability=dropout_probability,
            is_training=is_training)
        return output, softmax_aux, rng_state

    @staticmethod
    def batcher(batched_args, batch_dims, *, attn_bias_type, attn_mask_type, scaling_factor,
                dropout_probability, is_training):
        assert SelfFusedAttnFwdPrimitive.outer_primitive is not None
        qkv, bias, cu_seqlen, seed = batched_args
        qkv_bdim, _, _, seed_bdim = batch_dims

        out_bdims = qkv_bdim, qkv_bdim, seed_bdim
        return SelfFusedAttnFwdPrimitive.outer_primitive.bind(
            qkv,
            bias,
            cu_seqlen,
            seed,
            attn_bias_type=attn_bias_type,
            attn_mask_type=attn_mask_type,
            scaling_factor=scaling_factor,
            dropout_probability=dropout_probability,
            is_training=is_training), out_bdims

    @staticmethod
    def infer_sharding_from_operands(attn_bias_type, attn_mask_type, scaling_factor,
                                     dropout_probability, is_training, mesh, arg_infos,
                                     result_infos):
        del attn_bias_type, attn_mask_type, scaling_factor
        del dropout_probability, is_training, result_infos
        x_spec = get_padded_spec(arg_infos[0])    # (...batch, seqlen, 3, head, hidden)
        out_sharding = NamedSharding(mesh, PartitionSpec(*x_spec[:-3], *x_spec[-2:]))
        softmax_aux_sharding = NamedSharding(
            mesh, PartitionSpec(*x_spec[:-4], x_spec[-2], x_spec[-4], None))
        rng_state_sharding = NamedSharding(mesh, PartitionSpec(get_all_mesh_axes(), None))
        return (out_sharding, softmax_aux_sharding, rng_state_sharding)

    @staticmethod
    def partition(attn_bias_type, attn_mask_type, scaling_factor, dropout_probability, is_training,
                  mesh, arg_infos, result_infos):
        del result_infos
        x_spec = get_padded_spec(arg_infos[0])    # (...batch, seqlen, 3, head, hidden)
        out_sharding = NamedSharding(mesh, PartitionSpec(*x_spec[:-3], *x_spec[-2:]))
        softmax_aux_sharding = NamedSharding(
            mesh, PartitionSpec(*x_spec[:-4], x_spec[-2], x_spec[-4], None))
        rng_state_sharding = NamedSharding(mesh, PartitionSpec(get_all_mesh_axes(), None))
        arg_shardings = tuple([arg_i.sharding for arg_i in arg_infos[:-1]] + [rng_state_sharding])
        out_shardings = (out_sharding, softmax_aux_sharding, rng_state_sharding)
        impl = partial(SelfFusedAttnFwdPrimitive.impl,
                       attn_bias_type=attn_bias_type,
                       attn_mask_type=attn_mask_type,
                       scaling_factor=scaling_factor,
                       dropout_probability=dropout_probability,
                       is_training=is_training)
        return mesh, impl, out_shardings, arg_shardings


register_primitive(SelfFusedAttnFwdPrimitive)


def self_fused_attn_fwd(qkv: jnp.ndarray, bias: jnp.ndarray, cu_seqlen: jnp.ndarray,
                        seed: jnp.ndarray, attn_bias_type: NVTE_Bias_Type,
                        attn_mask_type: NVTE_Mask_Type, scaling_factor: float,
                        dropout_probability: float, is_training: bool):
    """
    Wrapper for TE self fused attention fwd
    Return BMM1 -> (PreBias) -> ScaleMaskSoftmax -> (PostBias) -> (Dropout) -> BMM2
    """
    checker = _FusedAttnRNGStateChecker()
    seed = checker.check_seed(seed, dropout_probability, is_training)

    if attn_bias_type == NVTE_Bias_Type.NVTE_NO_BIAS:
        assert bias is None
        bias = jnp.zeros(0, dtype=qkv.dtype)
    return SelfFusedAttnFwdPrimitive.outer_primitive.bind(qkv,
                                                          bias,
                                                          cu_seqlen,
                                                          seed,
                                                          attn_bias_type=attn_bias_type,
                                                          attn_mask_type=attn_mask_type,
                                                          scaling_factor=scaling_factor,
                                                          dropout_probability=dropout_probability,
                                                          is_training=is_training)


class SelfFusedAttnBwdPrimitive(BasePrimitive):
    """
    Self Fused Attention Backward Primitive
    """
    name = "te_self_fused_attn_backward"
    multiple_results = True
    impl_static_args = (6, 7, 8, 9, 10)
    inner_primitive = None
    outer_primitive = None

    @staticmethod
    def abstract(qkv_aval, softmax_aux_aval, rng_state_aval, output_aval, doutput_aval,
                 cu_seqlen_aval, *, attn_bias_type, attn_mask_type, scaling_factor,
                 dropout_probability, is_training):
        """
        Self fused attention bwd abstract
        """
        del softmax_aux_aval, rng_state_aval
        del cu_seqlen_aval, output_aval, attn_mask_type
        del scaling_factor, dropout_probability, is_training
        qkv_dtype = dtypes.canonicalize_dtype(qkv_aval.dtype)
        assert qkv_aval.dtype == doutput_aval.dtype
        batch_shape = qkv_aval.shape[:-4]
        max_seqlen = qkv_aval.shape[-4]
        num_head = qkv_aval.shape[-2]

        if attn_bias_type == NVTE_Bias_Type.NVTE_NO_BIAS:
            bias_shape = (0,)
        else:
            bias_shape = (*batch_shape[:-1], 1, num_head, max_seqlen, max_seqlen)
        bias_dtype = qkv_dtype

        dqkv_aval = qkv_aval.update(shape=qkv_aval.shape, dtype=qkv_dtype)
        dbias = qkv_aval.update(shape=bias_shape, dtype=bias_dtype)
        return dqkv_aval, dbias

    @staticmethod
    def lowering(ctx, qkv, softmax_aux, rng_state, output, doutput, cu_seqlen, *, attn_bias_type,
                 attn_mask_type, scaling_factor, dropout_probability, is_training):
        """
        Self fused attention bwd lowering rules
        """
        qkv_aval, _, _, _, _, _ = ctx.avals_in

        batch = reduce(operator.mul, qkv_aval.shape[:-4])
        max_seqlen = qkv_aval.shape[-4]
        num_head = qkv_aval.shape[-2]
        head_dim = qkv_aval.shape[-1]

        operands = [qkv, softmax_aux, rng_state, output, doutput, cu_seqlen]
        operand_shapes = map(lambda x: x.type.shape, operands)
        out_types = [
            ir.RankedTensorType.get(output.shape, mlir.dtype_to_ir_type(output.dtype))
            for output in ctx.avals_out
        ]

        args = CustomCallArgsWrapper(out_types, operands, operand_shapes)

        opaque = transformer_engine_jax.pack_fused_attn_descriptor(
            batch, num_head, max_seqlen, max_seqlen, head_dim, scaling_factor, dropout_probability,
            attn_bias_type, attn_mask_type, jax_dtype_to_te_dtype(qkv_aval.dtype), is_training)

        out = custom_caller(SelfFusedAttnBwdPrimitive.name, args, opaque, has_side_effect=False)

        return out

    @staticmethod
    def impl(qkv, softmax_aux, rng_state, output, doutput, cu_seqlen, attn_bias_type,
             attn_mask_type, scaling_factor, dropout_probability, is_training):
        assert SelfFusedAttnBwdPrimitive.inner_primitive is not None
        dqkv, dbias = SelfFusedAttnBwdPrimitive.inner_primitive.bind(
            qkv,
            softmax_aux,
            rng_state,
            output,
            doutput,
            cu_seqlen,
            attn_bias_type=attn_bias_type,
            attn_mask_type=attn_mask_type,
            scaling_factor=scaling_factor,
            dropout_probability=dropout_probability,
            is_training=is_training)
        return dqkv, dbias

    @staticmethod
    def batcher(batched_args, batch_dims, *, attn_bias_type, attn_mask_type, scaling_factor,
                dropout_probability, is_training):
        assert SelfFusedAttnBwdPrimitive.outer_primitive is not None
        qkv, softmax_aux, rng_state, output, doutput, cu_seqlen = batched_args
        qkv_bdim, *_ = batch_dims

        out_bdims = qkv_bdim, qkv_bdim
        return SelfFusedAttnBwdPrimitive.outer_primitive.bind(
            qkv,
            softmax_aux,
            rng_state,
            output,
            doutput,
            cu_seqlen,
            attn_bias_type=attn_bias_type,
            attn_mask_type=attn_mask_type,
            scaling_factor=scaling_factor,
            dropout_probability=dropout_probability,
            is_training=is_training), out_bdims

    @staticmethod
    def infer_sharding_from_operands(attn_bias_type, attn_mask_type, scaling_factor,
                                     dropout_probability, is_training, mesh, arg_infos,
                                     result_infos):
        del attn_mask_type, scaling_factor, dropout_probability,
        del is_training, result_infos
        x_spec = get_padded_spec(arg_infos[0])
        dx_sharding = NamedSharding(mesh, PartitionSpec(*x_spec))
        dbias_spec = [None]
        if attn_bias_type is not NVTE_Bias_Type.NVTE_NO_BIAS:
            dbias_spec = [*x_spec[:-5], None, x_spec[-4], *x_spec[-2:]]
        dbias_sharding = NamedSharding(mesh, PartitionSpec(*dbias_spec))
        return (dx_sharding, dbias_sharding)

    @staticmethod
    def partition(attn_bias_type, attn_mask_type, scaling_factor, dropout_probability, is_training,
                  mesh, arg_infos, result_infos):
        del result_infos
        x_spec = get_padded_spec(arg_infos[0])
        dx_sharding = NamedSharding(mesh, PartitionSpec(*x_spec))
        dbias_spec = [None]
        if attn_bias_type is not NVTE_Bias_Type.NVTE_NO_BIAS:
            dbias_spec = [*x_spec[:-5], None, x_spec[-4], *x_spec[-2:]]
        dbias_sharding = NamedSharding(mesh, PartitionSpec(*dbias_spec))
        arg_shardings = tuple(arg_i.sharding for arg_i in arg_infos)
        out_shardings = (dx_sharding, dbias_sharding)

        def sharded_impl(qkv, softmax_aux, rng_state, output, doutput, cu_seqlen):
            local_dx, local_dbias = SelfFusedAttnBwdPrimitive.impl(
                qkv,
                softmax_aux,
                rng_state,
                output,
                doutput,
                cu_seqlen,
                attn_bias_type=attn_bias_type,
                attn_mask_type=attn_mask_type,
                scaling_factor=scaling_factor,
                dropout_probability=dropout_probability,
                is_training=is_training)
            global_dbias = local_dbias
            if attn_bias_type is not NVTE_Bias_Type.NVTE_NO_BIAS:
                global_dbias = all_reduce_sum_along_dp_fsdp(local_dbias)
            return local_dx, global_dbias

        return mesh, sharded_impl, out_shardings, arg_shardings


register_primitive(SelfFusedAttnBwdPrimitive)


def self_fused_attn_bwd(qkv: jnp.ndarray, softmax_aux: jnp.ndarray, rng_state: jnp.ndarray,
                        output: jnp.ndarray, doutput: jnp.ndarray, cu_seqlen: jnp.ndarray,
                        attn_bias_type: NVTE_Bias_Type, attn_mask_type: NVTE_Mask_Type,
                        scaling_factor: float, dropout_probability: float, is_training: bool):
    """
    Wrapper for TE self fused attention bwd
    Return the gradients of self fused attention with packed qkv input
    """
    return SelfFusedAttnBwdPrimitive.outer_primitive.bind(qkv,
                                                          softmax_aux,
                                                          rng_state,
                                                          output,
                                                          doutput,
                                                          cu_seqlen,
                                                          attn_bias_type=attn_bias_type,
                                                          attn_mask_type=attn_mask_type,
                                                          scaling_factor=scaling_factor,
                                                          dropout_probability=dropout_probability,
                                                          is_training=is_training)


class CrossFusedAttnFwdPrimitive(BasePrimitive):
    """
    Cross Fused Attention Forward Primitive
    """
    name = "te_cross_fused_attn_forward"
    multiple_results = True
    impl_static_args = (5, 6, 7, 8, 9)
    inner_primitive = None
    outer_primitive = None

    @staticmethod
    def abstract(q_aval, kv_aval, q_cu_seqlen_aval, kv_cu_seqlen_aval, seed_aval, *, attn_bias_type,
                 attn_mask_type, scaling_factor, dropout_probability, is_training):
        """
        Cross fused attention fwd abstract
        """
        del seed_aval, attn_bias_type, attn_mask_type
        del scaling_factor, dropout_probability, is_training

        q_dtype = dtypes.canonicalize_dtype(q_aval.dtype)
        q_batch_shape = q_aval.shape[:-3]
        q_max_seqlen = q_aval.shape[-3]
        q_num_head = q_aval.shape[-2]
        q_head_dim = q_aval.shape[-1]

        kv_dtype = dtypes.canonicalize_dtype(kv_aval.dtype)
        kv_batch_shape = kv_aval.shape[:-4]
        kv_max_seqlen = kv_aval.shape[-4]
        nkv = kv_aval.shape[-3]
        kv_num_head = kv_aval.shape[-2]
        kv_head_dim = kv_aval.shape[-1]

        assert q_dtype == kv_dtype
        assert q_batch_shape == kv_batch_shape
        assert q_num_head == kv_num_head
        assert q_head_dim == kv_head_dim
        assert nkv == 2
        assert q_cu_seqlen_aval.dtype == kv_cu_seqlen_aval.dtype

        output_shape = q_aval.shape
        output_dtype = q_dtype
        softmax_aux_shape = (*q_batch_shape, q_num_head, q_max_seqlen, kv_max_seqlen)
        softmax_aux_dtype = q_dtype

        out_aval = q_aval.update(shape=output_shape, dtype=output_dtype)
        softmax_aux_aval = q_aval.update(shape=softmax_aux_shape, dtype=softmax_aux_dtype)
        return out_aval, softmax_aux_aval

    @staticmethod
    def lowering(ctx, q, kv, q_cu_seqlen, kv_cu_seqlen, seed, *, attn_bias_type, attn_mask_type,
                 scaling_factor, dropout_probability, is_training):
        """
        Cross fused attention fwd lowering rules
        """
        q_aval, kv_aval, _, _, _ = ctx.avals_in
        assert q_aval.dtype == kv_aval.dtype

        batch = reduce(operator.mul, q_aval.shape[:-3])
        q_max_seqlen = q_aval.shape[-3]
        num_head = q_aval.shape[-2]
        head_dim = q_aval.shape[-1]
        kv_max_seqlen = kv_aval.shape[-4]

        operands = [q, kv, q_cu_seqlen, kv_cu_seqlen, seed]
        operand_shapes = map(lambda x: x.type.shape, operands)
        out_types = [
            ir.RankedTensorType.get(output.shape, mlir.dtype_to_ir_type(output.dtype))
            for output in ctx.avals_out
        ]

        args = CustomCallArgsWrapper(out_types, operands, operand_shapes)
        opaque = transformer_engine_jax.pack_fused_attn_descriptor(
            batch, num_head, q_max_seqlen, kv_max_seqlen, head_dim,
            scaling_factor, dropout_probability, attn_bias_type, attn_mask_type,
            jax_dtype_to_te_dtype(q_aval.dtype), is_training)

        out = custom_caller(CrossFusedAttnFwdPrimitive.name, args, opaque, has_side_effect=False)

        return out

    @staticmethod
    def impl(q, kv, q_cu_seqlen, kv_cu_seqlen, seed, attn_bias_type, attn_mask_type, scaling_factor,
             dropout_probability, is_training):
        assert CrossFusedAttnFwdPrimitive.inner_primitive is not None
        output, softmax_aux = CrossFusedAttnFwdPrimitive.inner_primitive.bind(
            q,
            kv,
            q_cu_seqlen,
            kv_cu_seqlen,
            seed,
            attn_bias_type=attn_bias_type,
            attn_mask_type=attn_mask_type,
            scaling_factor=scaling_factor,
            dropout_probability=dropout_probability,
            is_training=is_training)
        return output, softmax_aux

    @staticmethod
    def batcher(batched_args, batch_dims, *, attn_bias_type, attn_mask_type, scaling_factor,
                dropout_probability, is_training):
        assert CrossFusedAttnFwdPrimitive.outer_primitive is not None
        q, kv, q_cu_seqlen, kv_cu_seqlen, seed = batched_args
        q_bdim, *_ = batch_dims

        out_bdims = q_bdim, q_bdim
        return CrossFusedAttnFwdPrimitive.outer_primitive.bind(
            q,
            kv,
            q_cu_seqlen,
            kv_cu_seqlen,
            seed,
            attn_bias_type=attn_bias_type,
            attn_mask_type=attn_mask_type,
            scaling_factor=scaling_factor,
            dropout_probability=dropout_probability,
            is_training=is_training), out_bdims

    @staticmethod
    def infer_sharding_from_operands(attn_bias_type, attn_mask_type, scaling_factor,
                                     dropout_probability, is_training, mesh, arg_infos,
                                     result_infos):
        del attn_bias_type, attn_mask_type, scaling_factor
        del dropout_probability, is_training, result_infos
        q_spec = get_padded_spec(arg_infos[0])    # (...batch, q_seqlen, head, hidden)
        kv_spec = get_padded_spec(arg_infos[1])    # (...batch, kv_seqlen, 2, head, hidden)
        out_sharding = NamedSharding(mesh, PartitionSpec(*q_spec))
        softmax_aux_sharding = NamedSharding(
            mesh, PartitionSpec(*q_spec[:-3], q_spec[-2], q_spec[-3], kv_spec[-4]))
        return (out_sharding, softmax_aux_sharding)

    @staticmethod
    def partition(attn_bias_type, attn_mask_type, scaling_factor, dropout_probability, is_training,
                  mesh, arg_infos, result_infos):
        del result_infos
        q_spec = get_padded_spec(arg_infos[0])    # (...batch, q_seqlen, head, hidden)
        kv_spec = get_padded_spec(arg_infos[1])    # (...batch, kv_seqlen, 2, head, hidden)
        out_sharding = NamedSharding(mesh, PartitionSpec(*q_spec))
        softmax_aux_sharding = NamedSharding(
            mesh, PartitionSpec(*q_spec[:-3], q_spec[-2], q_spec[-3], kv_spec[-4]))
        seed_sharding = NamedSharding(mesh, PartitionSpec(get_all_mesh_axes(), None))
        arg_shardings = tuple([arg_i.sharding for arg_i in arg_infos[:-1]] + [seed_sharding])
        out_shardings = (out_sharding, softmax_aux_sharding)
        impl = partial(CrossFusedAttnFwdPrimitive.impl,
                       attn_bias_type=attn_bias_type,
                       attn_mask_type=attn_mask_type,
                       scaling_factor=scaling_factor,
                       dropout_probability=dropout_probability,
                       is_training=is_training)
        return mesh, impl, out_shardings, arg_shardings


register_primitive(CrossFusedAttnFwdPrimitive)


def cross_fused_attn_fwd(q: jnp.ndarray, kv: jnp.ndarray, q_cu_seqlen: jnp.ndarray,
                         kv_cu_seqlen: jnp.ndarray, seed: jnp.ndarray,
                         attn_bias_type: NVTE_Bias_Type, attn_mask_type: NVTE_Mask_Type,
                         scaling_factor: float, dropout_probability: float, is_training: bool):
    """
    Wrapper for TE cross fused attention fwd
    Return BMM1 -> (PreBias) -> ScaleMaskSoftmax -> (PostBias) -> (Dropout) -> BMM2
    """
    checker = _FusedAttnRNGStateChecker()
    seed = checker.check_seed(seed, dropout_probability, is_training)

    return CrossFusedAttnFwdPrimitive.outer_primitive.bind(q,
                                                           kv,
                                                           q_cu_seqlen,
                                                           kv_cu_seqlen,
                                                           seed,
                                                           attn_bias_type=attn_bias_type,
                                                           attn_mask_type=attn_mask_type,
                                                           scaling_factor=scaling_factor,
                                                           dropout_probability=dropout_probability,
                                                           is_training=is_training)


class CrossFusedAttnBwdPrimitive(BasePrimitive):
    """
    Cross Fused Attention Backward Primitive
    """
    name = "te_cross_fused_attn_backward"
    multiple_results = True
    impl_static_args = (6, 7, 8, 9, 10)
    inner_primitive = None
    outer_primitive = None

    @staticmethod
    def abstract(q_aval, kv_aval, softmax_aux_aval, doutput_aval, q_cu_seqlen_aval,
                 kv_cu_seqlen_aval, *, attn_bias_type, attn_mask_type, scaling_factor,
                 dropout_probability, is_training):
        """
        Cross fused attention bwd abstract
        """
        del attn_bias_type, attn_mask_type
        del scaling_factor, dropout_probability, is_training
        q_dtype = dtypes.canonicalize_dtype(q_aval.dtype)
        kv_dtype = dtypes.canonicalize_dtype(kv_aval.dtype)
        softmax_aux_dtype = dtypes.canonicalize_dtype(softmax_aux_aval.dtype)
        doutput_dtype = dtypes.canonicalize_dtype(doutput_aval.dtype)
        assert q_dtype == kv_dtype == softmax_aux_dtype == doutput_dtype
        assert q_cu_seqlen_aval.dtype == kv_cu_seqlen_aval.dtype

        dq_aval = q_aval.update(shape=q_aval.shape, dtype=q_dtype)
        dkv_aval = kv_aval.update(shape=kv_aval.shape, dtype=kv_dtype)
        return dq_aval, dkv_aval

    @staticmethod
    def lowering(ctx, q, kv, softmax_aux, doutput, q_cu_seqlen, kv_cu_seqlen, *, attn_bias_type,
                 attn_mask_type, scaling_factor, dropout_probability, is_training):
        """
        Cross fused attention bwd lowering rules
        """
        q_aval, kv_aval, _, _, _, _ = ctx.avals_in
        assert q_aval.dtype == kv_aval.dtype

        batch = reduce(operator.mul, q_aval.shape[:-3])
        q_max_seqlen = q_aval.shape[-3]
        num_head = q_aval.shape[-2]
        head_dim = q_aval.shape[-1]
        kv_max_seqlen = kv_aval.shape[-4]

        operands = [q, kv, softmax_aux, doutput, q_cu_seqlen, kv_cu_seqlen]
        operand_shapes = map(lambda x: x.type.shape, operands)
        out_types = [
            ir.RankedTensorType.get(output.shape, mlir.dtype_to_ir_type(output.dtype))
            for output in ctx.avals_out
        ]

        args = CustomCallArgsWrapper(out_types, operands, operand_shapes)

        # the dropout elements are encoded in the forward auxiliary tensor
        # so seed is not needed in backward
        opaque = transformer_engine_jax.pack_fused_attn_descriptor(
            batch, num_head, q_max_seqlen, kv_max_seqlen, head_dim,
            scaling_factor, dropout_probability, attn_bias_type, attn_mask_type,
            jax_dtype_to_te_dtype(q_aval.dtype), is_training)

        out = custom_caller(CrossFusedAttnBwdPrimitive.name, args, opaque, has_side_effect=False)

        return out

    @staticmethod
    def impl(q, kv, softmax_aux, doutput, q_cu_seqlen, kv_cu_seqlen, attn_bias_type, attn_mask_type,
             scaling_factor, dropout_probability, is_training):
        assert CrossFusedAttnBwdPrimitive.inner_primitive is not None
        dq, dkv = CrossFusedAttnBwdPrimitive.inner_primitive.bind(
            q,
            kv,
            softmax_aux,
            doutput,
            q_cu_seqlen,
            kv_cu_seqlen,
            attn_bias_type=attn_bias_type,
            attn_mask_type=attn_mask_type,
            scaling_factor=scaling_factor,
            dropout_probability=dropout_probability,
            is_training=is_training)
        return dq, dkv

    @staticmethod
    def batcher(batched_args, batch_dims, *, attn_bias_type, attn_mask_type, scaling_factor,
                dropout_probability, is_training):
        assert CrossFusedAttnBwdPrimitive.outer_primitive is not None
        q, kv, softmax_aux, doutput, q_cu_seqlen, kv_cu_seqlen = batched_args
        q_bdim, kv_bdim, *_ = batch_dims

        out_bdims = q_bdim, kv_bdim
        return CrossFusedAttnBwdPrimitive.outer_primitive.bind(
            q,
            kv,
            softmax_aux,
            doutput,
            q_cu_seqlen,
            kv_cu_seqlen,
            attn_bias_type=attn_bias_type,
            attn_mask_type=attn_mask_type,
            scaling_factor=scaling_factor,
            dropout_probability=dropout_probability,
            is_training=is_training), out_bdims

    @staticmethod
    def infer_sharding_from_operands(attn_bias_type, attn_mask_type, scaling_factor,
                                     dropout_probability, is_training, mesh, arg_infos,
                                     result_infos):
        del attn_bias_type, attn_mask_type, scaling_factor
        del dropout_probability, is_training, result_infos
        q_spec = get_padded_spec(arg_infos[0])
        kv_spec = get_padded_spec(arg_infos[1])
        dq_sharding = NamedSharding(mesh, PartitionSpec(*q_spec))
        dkv_sharding = NamedSharding(mesh, PartitionSpec(*kv_spec))
        return (dq_sharding, dkv_sharding)

    @staticmethod
    def partition(attn_bias_type, attn_mask_type, scaling_factor, dropout_probability, is_training,
                  mesh, arg_infos, result_infos):
        del result_infos
        q_spec = get_padded_spec(arg_infos[0])
        kv_spec = get_padded_spec(arg_infos[1])
        dq_sharding = NamedSharding(mesh, PartitionSpec(*q_spec))
        dkv_sharding = NamedSharding(mesh, PartitionSpec(*kv_spec))
        arg_shardings = tuple(arg_i.sharding for arg_i in arg_infos)
        out_shardings = (dq_sharding, dkv_sharding)

        impl = partial(CrossFusedAttnBwdPrimitive.impl,
                       attn_bias_type=attn_bias_type,
                       attn_mask_type=attn_mask_type,
                       scaling_factor=scaling_factor,
                       dropout_probability=dropout_probability,
                       is_training=is_training)

        return mesh, impl, out_shardings, arg_shardings


register_primitive(CrossFusedAttnBwdPrimitive)


def cross_fused_attn_bwd(q: jnp.ndarray, kv: jnp.ndarray, softmax_aux: jnp.ndarray,
                         doutput: jnp.ndarray, q_cu_seqlen: jnp.ndarray, kv_cu_seqlen: jnp.ndarray,
                         attn_bias_type: NVTE_Bias_Type, attn_mask_type: NVTE_Mask_Type,
                         scaling_factor: float, dropout_probability: float, is_training: bool):
    """
    Wrapper for TE cross fused attention bwd
    Return the gradients of cross fused attention with packed kv input
    """
    return CrossFusedAttnBwdPrimitive.outer_primitive.bind(q,
                                                           kv,
                                                           softmax_aux,
                                                           doutput,
                                                           q_cu_seqlen,
                                                           kv_cu_seqlen,
                                                           attn_bias_type=attn_bias_type,
                                                           attn_mask_type=attn_mask_type,
                                                           scaling_factor=scaling_factor,
                                                           dropout_probability=dropout_probability,
                                                           is_training=is_training)


class GatedGeluPrimitive(BasePrimitive):
    """
    Gated Gelu Froward Primitive
    """
    name = "te_gated_gelu"
    multiple_results = False
    inner_primitive = None
    outer_primitive = None
    impl_static_args = ()

    @staticmethod
    def abstract(x_aval):
        """
        gated_gelu abstract
        """
        dtype = dtypes.canonicalize_dtype(x_aval.dtype)
        assert dtype in [jnp.float32, jnp.float16, jnp.bfloat16]
        x_shape = x_aval.shape
        assert x_shape[-2] == 2    # Assume x in (....., 2, hidden)
        hidden_size = x_shape[-1]
        batch_shapes = x_shape[:-2]
        x_shape = x_aval.shape
        out_aval = core.raise_to_shaped(x_aval)
        out_shape = (batch_shapes) + (hidden_size,)
        out_aval = out_aval.update(shape=out_shape, dtype=dtype)

        return out_aval

    @staticmethod
    def lowering(ctx, x):
        """
        gated_gelu lowering rules
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
        opaque = transformer_engine_jax.pack_common_descriptor((batch_size, hidden_size), in_dtype,
                                                               in_dtype)

        out = custom_caller(GatedGeluPrimitive.name, args, opaque, False)

        return [out]

    @staticmethod
    def impl(x):
        assert GatedGeluPrimitive.inner_primitive is not None
        out = GatedGeluPrimitive.inner_primitive.bind(x)
        return out

    @staticmethod
    def batcher(batched_args, batch_dims):
        """
        gated_gelu batcher
        """
        assert GatedGeluPrimitive.outer_primitive is not None
        inputs, = batched_args
        inputs_bdim, = batch_dims

        out_bdims = inputs_bdim
        return GatedGeluPrimitive.outer_primitive.bind(inputs), out_bdims

    @staticmethod
    def infer_sharding_from_operands(mesh, arg_infos, result_infos):
        """
        gated_gelu infer_sharding_from_operands
        """
        del result_infos    # Unused.
        x_spec = get_padded_spec(arg_infos[0])
        out_sharding = NamedSharding(mesh, PartitionSpec(*x_spec[:-2], x_spec[-1]))
        return out_sharding

    @staticmethod
    def partition(mesh, arg_infos, result_infos):
        """
        gated_gelu partitioning
        """
        del result_infos
        x_spec = get_padded_spec(arg_infos[0])
        arg_shardings = tuple(arg_i.sharding for arg_i in arg_infos)
        out_sharding = NamedSharding(mesh, PartitionSpec(*x_spec[:-2], x_spec[-1]))
        impl = GatedGeluPrimitive.impl
        return mesh, impl, out_sharding, arg_shardings


register_primitive(GatedGeluPrimitive)


def gated_gelu(inputs: jnp.ndarray) -> jnp.ndarray:
    """
    gated gelu wrapper
    Return FP8(geglu(inputs))
    Assume inputs has two dimensions shape and the memory layout is (N, 2, H)
    """
    return GatedGeluPrimitive.outer_primitive.bind(inputs)


class DgatedGeluPrimitive(BasePrimitive):
    """
    Dgated Gelu Primitive
    """
    name = "te_dgated_gelu"
    multiple_results = False
    inner_primitive = None
    outer_primitive = None
    impl_static_args = ()

    @staticmethod
    def abstract(dz_aval, x_aval):
        """
        dgated_gelu abstract
        """
        dtype = dtypes.canonicalize_dtype(dz_aval.dtype)
        assert dtype in [jnp.float32, jnp.float16, jnp.bfloat16]
        assert x_aval.dtype == dtype
        for axis in range(len(dz_aval.shape) - 1):
            assert dz_aval.shape[axis] == x_aval.shape[axis]

        assert x_aval.shape[-2] == 2    # Assume x in (....., 2, hidden)

        i_hidden_size = dz_aval.shape[-1]
        g_hidden_size = x_aval.shape[-1]
        assert i_hidden_size == g_hidden_size
        out_aval = core.raise_to_shaped(x_aval)
        return out_aval

    @staticmethod
    def lowering(ctx, dz, x):
        """
        dgated_gelu lowering rules
        """
        in_aval, gi_aval = ctx.avals_in
        assert in_aval.dtype in [jnp.float32, jnp.float16, jnp.bfloat16]
        assert gi_aval.dtype == in_aval.dtype
        ir_in_type = ir.RankedTensorType(dz.type)
        ir_in_shape = ir_in_type.shape
        gi_type = ir.RankedTensorType(x.type)
        gi_shape = gi_type.shape
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
        opaque = transformer_engine_jax.pack_common_descriptor((ir_batch_size, i_hidden_size),
                                                               in_dtype, in_dtype)

        out = custom_caller(DgatedGeluPrimitive.name, args, opaque, False)

        return [out]

    @staticmethod
    def impl(dz, x):
        """
        dgated_gelu implementation
        """
        assert DgatedGeluPrimitive.inner_primitive is not None
        dx = DgatedGeluPrimitive.inner_primitive.bind(dz, x)
        return dx

    @staticmethod
    def batcher(batched_args, batch_dims):
        """
        dgated_gelu batcher
        """
        assert DgatedGeluPrimitive.outer_primitive is not None
        dz, x = batched_args
        _, x_bdim = batch_dims

        out_bdims = x_bdim
        return DgatedGeluPrimitive.outer_primitive.bind(dz, x), out_bdims

    @staticmethod
    def infer_sharding_from_operands(mesh, arg_infos, result_infos):
        """
        dgated_gelu infer_sharding_from_operands
        """
        del result_infos    # Unused.
        gelu_out_spec = get_padded_spec(arg_infos[1])
        dx_sharding = NamedSharding(mesh, PartitionSpec(*gelu_out_spec))
        return dx_sharding

    @staticmethod
    def partition(mesh, arg_infos, result_infos):
        """
        dgated_gelu partition
        """
        del result_infos
        dx_sharding = NamedSharding(mesh, PartitionSpec(*get_padded_spec(arg_infos[1])))
        arg_shardings = tuple(arg_i.sharding for arg_i in arg_infos)
        out_shardings = dx_sharding
        impl = DgatedGeluPrimitive.impl
        return mesh, impl, out_shardings, arg_shardings


register_primitive(DgatedGeluPrimitive)


def dgated_gelu(inputs: jnp.ndarray, gelu_inputs: jnp.ndarray) -> jnp.ndarray:
    """
    dgated_gelu fusion wrapper
    Return dgeglu(inputs)
    """
    return DgatedGeluPrimitive.outer_primitive.bind(inputs, gelu_inputs)


def _normalize_axis_boundary(axis, ndim):
    return axis if axis >= 0 else ndim + axis


def _multidim_transpose(shape, static_axis_boundary, transpose_axis_boundary):
    """
    te_cast_transpose_p multi-dims transpose
    """
    if static_axis_boundary < 0:
        static_axis_boundary = -1    # means no static axes
    assert static_axis_boundary < len(shape) - 2    # at least 2 remaining for transpose.
    transpose_start_idx = static_axis_boundary + 1
    transpose_axis_boundary = _normalize_axis_boundary(transpose_axis_boundary, len(shape))
    assert transpose_start_idx < transpose_axis_boundary
    return (*shape[:transpose_start_idx], *shape[transpose_axis_boundary:],
            *shape[transpose_start_idx:transpose_axis_boundary])


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
    def abstract(x_aval, amax_aval, scale_aval, scale_inv_aval, *, out_dtype, static_axis_boundary,
                 transpose_axis_boundary):
        """
        te_cast_transpose_p abstract
        """
        dtype = dtypes.canonicalize_dtype(x_aval.dtype)
        assert dtype in [jnp.float32, jnp.float16, jnp.bfloat16]
        assert amax_aval.dtype == jnp.float32
        assert scale_aval.dtype == jnp.float32
        assert scale_inv_aval.dtype == jnp.float32

        transposed_x_shape = _multidim_transpose(x_aval.shape, static_axis_boundary,
                                                 transpose_axis_boundary)

        casted_x_aval = x_aval.update(shape=x_aval.shape, dtype=out_dtype)
        casted_xt_aval = x_aval.update(shape=transposed_x_shape, dtype=out_dtype)
        updated_amax_aval = amax_aval.update(shape=amax_aval.shape, dtype=amax_aval.dtype)

        return casted_x_aval, casted_xt_aval, updated_amax_aval

    @staticmethod
    def lowering(ctx, x, amax, scale, scale_inv, *, out_dtype, static_axis_boundary,
                 transpose_axis_boundary):
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

        transposed_x_shape = _multidim_transpose(ir_x_shape, static_axis_boundary,
                                                 transpose_axis_boundary)

        out_types = [
            ir.RankedTensorType.get(ir_x_shape, ir_out_dtype),
            ir.RankedTensorType.get(transposed_x_shape, ir_out_dtype),
            ir.RankedTensorType.get(ir_amax_shape, ir_amax_dtype),
        ]
        operands = [x, amax, scale, scale_inv]
        operand_shapes = [ir_x_shape, ir_amax_shape, ir_scale_shape, ir_scale_inv_shape]
        args = CustomCallArgsWrapper(out_types, operands, operand_shapes)

        contracted_x_shape = (reduce(operator.mul, ir_x_shape[:transpose_axis_boundary]),
                              reduce(operator.mul, ir_x_shape[transpose_axis_boundary:]))
        opaque = transformer_engine_jax.pack_common_descriptor(contracted_x_shape,
                                                               jax_dtype_to_te_dtype(x_aval.dtype),
                                                               jax_dtype_to_te_dtype(out_dtype))

        out = custom_caller(CastTransposePrimitive.name,
                            args,
                            opaque,
                            False,
                            operand_output_aliases={1: 2})

        return out

    @staticmethod
    def impl(x, amax, scale, scale_inv, out_dtype, static_axis_boundary, transpose_axis_boundary):
        """
        tcast_transpose implementation
        """
        assert CastTransposePrimitive.inner_primitive is not None
        casted_x, casted_transposed_x, updated_amax = \
            CastTransposePrimitive.inner_primitive.bind(
                x, amax, scale, scale_inv, out_dtype=out_dtype,
                static_axis_boundary=static_axis_boundary,
                transpose_axis_boundary=transpose_axis_boundary)
        return casted_x, casted_transposed_x, updated_amax

    @staticmethod
    def batcher(batched_args, batch_dims, *, out_dtype, static_axis_boundary,
                transpose_axis_boundary):
        assert CastTransposePrimitive.outer_primitive is not None
        assert static_axis_boundary < 0

        x, amax, scale, scale_inv = batched_args
        x_bdim, amax_bdim, *_ = batch_dims

        # Minus batch dim.
        transpose_axis_boundary = _normalize_axis_boundary(transpose_axis_boundary, x.ndim - 1)
        transpose_axis_boundary += 1    # Plus batch dim

        out_bdims = x_bdim, x_bdim, amax_bdim
        return CastTransposePrimitive.outer_primitive.bind(
            x,
            amax,
            scale,
            scale_inv,
            out_dtype=out_dtype,
            static_axis_boundary=x_bdim,
            transpose_axis_boundary=transpose_axis_boundary), out_bdims

    @staticmethod
    def infer_sharding_from_operands(out_dtype, static_axis_boundary, transpose_axis_boundary, mesh,
                                     arg_infos, result_infos):
        del out_dtype, result_infos
        x_spec = get_padded_spec(arg_infos[0])
        casted_x_sharding = NamedSharding(mesh, PartitionSpec(*x_spec))
        xt_spec = _multidim_transpose(x_spec, static_axis_boundary, transpose_axis_boundary)
        casted_transposed_x_sharding = NamedSharding(mesh, PartitionSpec(*xt_spec))
        amax_sharding = NamedSharding(mesh, PartitionSpec(*get_padded_spec(arg_infos[1])))
        return (casted_x_sharding, casted_transposed_x_sharding, amax_sharding)

    @staticmethod
    def partition(out_dtype, static_axis_boundary, transpose_axis_boundary, mesh, arg_infos,
                  result_infos):
        del result_infos
        x_spec = get_padded_spec(arg_infos[0])
        casted_x_sharding = NamedSharding(mesh, PartitionSpec(*x_spec))
        xt_spec = _multidim_transpose(x_spec, static_axis_boundary, transpose_axis_boundary)
        casted_transposed_x_sharding = NamedSharding(mesh, PartitionSpec(*xt_spec))
        amax_sharding = NamedSharding(mesh, PartitionSpec(*get_padded_spec(arg_infos[1])))
        arg_shardings = tuple(arg_i.sharding for arg_i in arg_infos)
        out_shardings = (casted_x_sharding, casted_transposed_x_sharding, amax_sharding)

        def sharded_impl(x, amax, scale, scale_inv):
            local_cx, local_cxt, local_updated_amax = \
                CastTransposePrimitive.impl(x, amax, scale, scale_inv,
                     out_dtype=out_dtype,
                       static_axis_boundary=static_axis_boundary,
                       transpose_axis_boundary=transpose_axis_boundary)
            global_updated_amax = all_reduce_max_along_all_axes_except_PP(local_updated_amax)

            return local_cx, local_cxt, global_updated_amax

        return mesh, sharded_impl, out_shardings, arg_shardings


register_primitive(CastTransposePrimitive)


def cast_transpose(x: jnp.ndarray, amax: jnp.ndarray, scale: jnp.ndarray, scale_inv: jnp.ndarray,
                   out_dtype: jnp.dtype, static_axis_boundary: int,
                   transpose_axis_boundary: int) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
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
        transpose_axis_boundary=transpose_axis_boundary)


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
        transposed_x_shape = _multidim_transpose(x_aval.shape, static_axis_boundary,
                                                 transpose_axis_boundary)
        xt_aval = x_aval.update(shape=transposed_x_shape, dtype=x_aval.dtype)

        return xt_aval

    @staticmethod
    def lowering(ctx, x, *, static_axis_boundary, transpose_axis_boundary):
        """
        _transpose cuda lowering
        """

        x_aval = ctx.avals_in[0]
        assert x_aval.dtype in [
            jnp.float32, jnp.float16, jnp.bfloat16, jnp.float8_e4m3fn, jnp.float8_e5m2
        ]

        ir_x_type = ir.RankedTensorType(x.type)
        ir_x_shape = ir_x_type.shape
        ir_out_dtype = jax_dtype_to_ir_dtype(x_aval.dtype)
        if static_axis_boundary >= 0:
            for i in range(static_axis_boundary + 1):
                assert ir_x_shape[i] == 1

        transposed_x_shape = _multidim_transpose(ir_x_shape, static_axis_boundary,
                                                 transpose_axis_boundary)

        out_types = [ir.RankedTensorType.get(transposed_x_shape, ir_out_dtype)]
        operands = [x]
        operand_shapes = [ir_x_shape]
        args = CustomCallArgsWrapper(out_types, operands, operand_shapes)

        te_dtype = jax_dtype_to_te_dtype(x_aval.dtype)
        contracted_x_shape = (reduce(operator.mul, ir_x_shape[:transpose_axis_boundary]),
                              reduce(operator.mul, ir_x_shape[transpose_axis_boundary:]))
        opaque = transformer_engine_jax.pack_common_descriptor(contracted_x_shape, te_dtype,
                                                               te_dtype)

        out = custom_caller(TransposePrimitive.name, args, opaque, False)

        return [out]

    @staticmethod
    def impl(x, static_axis_boundary, transpose_axis_boundary):
        """
        tcast_transpose implementation
        """
        assert TransposePrimitive.inner_primitive is not None
        transposed_x = \
            TransposePrimitive.inner_primitive.bind(x,
                                                    static_axis_boundary=static_axis_boundary,
                                                    transpose_axis_boundary=transpose_axis_boundary)
        return transposed_x

    @staticmethod
    def batcher(batched_args, batch_dims, *, static_axis_boundary, transpose_axis_boundary):
        assert TransposePrimitive.outer_primitive is not None
        assert static_axis_boundary < 0

        x, = batched_args
        x_bdim, = batch_dims

        # Minus batch dim.
        transpose_axis_boundary = _normalize_axis_boundary(transpose_axis_boundary, x.ndim - 1)
        transpose_axis_boundary += 1    # Plus batch dim

        out_bdims = x_bdim
        return TransposePrimitive.outer_primitive.bind(
            x, static_axis_boundary=x_bdim,
            transpose_axis_boundary=transpose_axis_boundary), out_bdims

    @staticmethod
    def infer_sharding_from_operands(static_axis_boundary, transpose_axis_boundary, mesh, arg_infos,
                                     result_infos):
        del result_infos
        x_spec = get_padded_spec(arg_infos[0])
        xt_spec = _multidim_transpose(x_spec, static_axis_boundary, transpose_axis_boundary)
        transposed_x_sharding = NamedSharding(mesh, PartitionSpec(*xt_spec))
        return transposed_x_sharding

    @staticmethod
    def partition(static_axis_boundary, transpose_axis_boundary, mesh, arg_infos, result_infos):
        del result_infos
        x_spec = get_padded_spec(arg_infos[0])
        xt_spec = _multidim_transpose(x_spec, static_axis_boundary, transpose_axis_boundary)
        transposed_x_sharding = NamedSharding(mesh, PartitionSpec(*xt_spec))
        arg_shardings = tuple(arg_i.sharding for arg_i in arg_infos)
        out_shardings = transposed_x_sharding

        impl = partial(TransposePrimitive.impl,
                       static_axis_boundary=static_axis_boundary,
                       transpose_axis_boundary=transpose_axis_boundary)

        return mesh, impl, out_shardings, arg_shardings


register_primitive(TransposePrimitive)


def transpose(x: jnp.ndarray, static_axis_boundary: int,
              transpose_axis_boundary: int) -> jnp.ndarray:
    """
    transpose wrapper
    """
    return TransposePrimitive.outer_primitive.bind(x,
                                                   static_axis_boundary=static_axis_boundary,
                                                   transpose_axis_boundary=transpose_axis_boundary)


class LayerNormFwdFp8Primitive(BasePrimitive):
    """
    Layer Normalization Forward FP8 Primitive
    """
    name = "te_layernorm_forward_fp8"
    multiple_results = True
    impl_static_args = (6, 7, 8)    # out_type, zero_centered_gamma, epsilon
    inner_primitive = None
    outer_primitive = None

    @staticmethod
    def abstract(x_aval, gamma_aval, beta_aval, amax_aval, scale_aval, scale_inv_aval, *, out_dtype,
                 zero_centered_gamma, epsilon):
        """
        LayerNorm fwd (fp8 out) abstract
        """
        del zero_centered_gamma, epsilon
        x_dtype = dtypes.canonicalize_dtype(x_aval.dtype)

        assert x_dtype in [jnp.float32, jnp.float16, jnp.bfloat16]
        assert amax_aval.dtype == jnp.float32
        assert scale_aval.dtype == jnp.float32
        assert scale_inv_aval.dtype == jnp.float32

        mu_rsigama_dtype = jnp.float32

        assert gamma_aval.size == beta_aval.size

        out_aval = x_aval.update(shape=x_aval.shape, dtype=out_dtype)
        mu_aval = rsigma_aval = out_aval.update(shape=out_aval.shape[:-1], dtype=mu_rsigama_dtype)
        updated_amax_aval = amax_aval.update(shape=amax_aval.shape, dtype=amax_aval.dtype)

        return out_aval, mu_aval, rsigma_aval, updated_amax_aval

    @staticmethod
    def lowering(ctx, x, gamma, beta, amax, scale, scale_inv, *, out_dtype, zero_centered_gamma,
                 epsilon):
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

        out_types = [
            ir.RankedTensorType.get(out_shape, ir_out_dtype),
            ir.RankedTensorType.get(batch_shape, ir_mu_dtype),
            ir.RankedTensorType.get(batch_shape, ir_rsigma_dtype),
            ir.RankedTensorType.get(ir_amax_shape, ir_amax_dtype),
        ]
        operands = [x, gamma, beta, amax, scale, scale_inv]
        operand_shapes = [
            x_shape, g_shape, b_shape, ir_amax_shape, ir_scale_shape, ir_scale_inv_shape
        ]
        args = CustomCallArgsWrapper(out_types, operands, operand_shapes)

        opaque = transformer_engine_jax.pack_norm_descriptor(
            batch_size,
            hidden_size,
            jax_dtype_to_te_dtype(x_aval.dtype),
            jax_dtype_to_te_dtype(gamma_aval.dtype),
            zero_centered_gamma,
            epsilon,
        )

        out = custom_caller(LayerNormFwdFp8Primitive.name,
                            args,
                            opaque,
                            False,
                            operand_output_aliases={3: 3})

        return out

    @staticmethod
    def impl(x, gamma, beta, amax, scale, scale_inv, out_dtype, zero_centered_gamma, epsilon):
        """
        to describe implementation
        """
        assert LayerNormFwdFp8Primitive.inner_primitive is not None
        out, mu, rsigma, updated_amax = LayerNormFwdFp8Primitive.inner_primitive.bind(
            x,
            gamma,
            beta,
            amax,
            scale,
            scale_inv,
            out_dtype=out_dtype,
            zero_centered_gamma=zero_centered_gamma,
            epsilon=epsilon)
        return out, mu, rsigma, updated_amax

    @staticmethod
    def batcher(batched_args, batch_dims, *, out_dtype, zero_centered_gamma, epsilon):
        """
        to describe batch rules for vmap
        """
        assert LayerNormFwdFp8Primitive.outer_primitive is not None
        x, gamma, beta, amax, scale, scale_inv = batched_args
        x_bdim, _, _, amax_bdim, _, _ = batch_dims

        out_bdims = x_bdim, x_bdim, x_bdim, amax_bdim
        return LayerNormFwdFp8Primitive.outer_primitive.bind(
            x,
            gamma,
            beta,
            amax,
            scale,
            scale_inv,
            out_dtype=out_dtype,
            zero_centered_gamma=zero_centered_gamma,
            epsilon=epsilon), out_bdims

    @staticmethod
    def infer_sharding_from_operands(out_dtype, zero_centered_gamma, epsilon, mesh, arg_infos,
                                     result_infos):
        del out_dtype, zero_centered_gamma, epsilon, result_infos
        x_spec = get_padded_spec(arg_infos[0])
        if x_spec[-1] is not None:
            warnings.warn(
                f"Does not support to shard hidden dim in {LayerNormFwdPrimitive.name}! " \
                f"Force to not shard the hidden dim, which might introduce extra collective ops, " \
                f"and hurt performance.")

        out_sharding = NamedSharding(mesh, PartitionSpec(*x_spec[:-1], None))
        mu_sharding = rsigma_sharding = NamedSharding(mesh, PartitionSpec(*x_spec[:-1]))
        amax_sharding = NamedSharding(mesh, PartitionSpec(*get_padded_spec(arg_infos[3])))
        return (out_sharding, mu_sharding, rsigma_sharding, amax_sharding)

    @staticmethod
    def partition(out_dtype, zero_centered_gamma, epsilon, mesh, arg_infos, result_infos):
        del result_infos
        x_spec = get_padded_spec(arg_infos[0])
        if x_spec[-1] is not None:
            warnings.warn(
                f"Does not support to shard hidden dim in {LayerNormFwdPrimitive.name}! " \
                f"Force to not shard the hidden dim, which might introduce extra collective ops, " \
                f"and hurt performance."
            )
        x_sharding = NamedSharding(mesh, PartitionSpec(*x_spec[:-1], None))
        g_sharding = NamedSharding(mesh, PartitionSpec(*get_padded_spec(arg_infos[1])))
        b_sharding = NamedSharding(mesh, PartitionSpec(*get_padded_spec(arg_infos[2])))
        out_sharding = x_sharding
        mu_sharding = rsigma_sharding = NamedSharding(
            mesh, PartitionSpec(*get_padded_spec(arg_infos[0])[:-1]))
        amax_sharding = NamedSharding(mesh, PartitionSpec(*get_padded_spec(arg_infos[3])))
        fp8_meta_sharding = amax_sharding
        arg_shardings = (x_sharding, g_sharding, b_sharding) + (fp8_meta_sharding,) * 3
        out_shardings = (out_sharding, mu_sharding, rsigma_sharding, amax_sharding)

        def sharded_impl(x, gamma, beta, amax, scale, scale_inv):
            local_x, local_mu, local_rsigma, local_amax = \
                LayerNormFwdFp8Primitive.impl(x, gamma, beta, amax, scale, scale_inv,
                                            out_dtype=out_dtype,
                                            zero_centered_gamma=zero_centered_gamma,
                                            epsilon=epsilon)
            global_updated_amax = all_reduce_max_along_all_axes_except_PP(local_amax)

            return local_x, local_mu, local_rsigma, global_updated_amax

        return mesh, sharded_impl, out_shardings, arg_shardings


register_primitive(LayerNormFwdFp8Primitive)


def layernorm_fwd_fp8(x: jnp.ndarray, gamma: jnp.ndarray, beta: jnp.ndarray, amax: jnp.ndarray,
                      scale: jnp.ndarray, scale_inv: jnp.ndarray, out_dtype: jnp.dtype,
                      zero_centered_gamma: bool, epsilon: float):
    """
    Wrapper for TE layernorm fwd (fp8 out)
    """
    return LayerNormFwdFp8Primitive.outer_primitive.bind(x,
                                                         gamma,
                                                         beta,
                                                         amax,
                                                         scale,
                                                         scale_inv,
                                                         out_dtype=out_dtype,
                                                         zero_centered_gamma=zero_centered_gamma,
                                                         epsilon=epsilon)


class RmsNormFwdFp8Primitive(BasePrimitive):
    """
    RMS Normalization Forward FP8 Primitive
    """
    name = "te_rmsnorm_forward_fp8"
    multiple_results = True
    impl_static_args = (5, 6)    # out_dtype, epsilon
    inner_primitive = None
    outer_primitive = None

    @staticmethod
    def abstract(x_aval, gamma_aval, amax_aval, scale_aval, scale_inv_aval, out_dtype, epsilon):
        """
        RMSNorm fwd (fp8 out) abstract
        """
        del epsilon
        x_dtype = dtypes.canonicalize_dtype(x_aval.dtype)

        assert x_dtype in [jnp.float32, jnp.float16, jnp.bfloat16]
        assert amax_aval.dtype == jnp.float32
        assert scale_aval.dtype == jnp.float32
        assert scale_inv_aval.dtype == jnp.float32

        hidden_size = gamma_aval.size
        assert x_aval.size % hidden_size == 0

        rsigama_dtype = jnp.float32

        out_aval = x_aval.update(shape=x_aval.shape, dtype=out_dtype)
        rsigma_aval = out_aval.update(shape=out_aval.shape[:-1], dtype=rsigama_dtype)
        amax_aval = out_aval.update(shape=amax_aval.shape, dtype=amax_aval.dtype)

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

        out_types = [
            ir.RankedTensorType.get(out_shape, ir_out_dtype),
            ir.RankedTensorType.get(batch_shape, ir_rsigma_dtype),
            ir.RankedTensorType.get(ir_amax_shape, ir_amax_dtype),
        ]
        operands = [x, gamma, amax, scale, scale_inv]
        operand_shapes = [x_shape, g_shape, ir_amax_shape, ir_scale_shape, ir_scale_inv_shape]
        args = CustomCallArgsWrapper(out_types, operands, operand_shapes)

        opaque = transformer_engine_jax.pack_norm_descriptor(
            batch_size,
            hidden_size,
            jax_dtype_to_te_dtype(x_aval.dtype),
            jax_dtype_to_te_dtype(gamma_aval.dtype),
            False,    # RMSNorm doesn't support zero_centered_gamma
            epsilon,
        )

        out = custom_caller(RmsNormFwdFp8Primitive.name,
                            args,
                            opaque,
                            False,
                            operand_output_aliases={2: 2})

        return out

    @staticmethod
    def impl(x, gamma, amax, scale, scale_inv, out_dtype, epsilon):
        """
        to describe implementation
        """
        assert RmsNormFwdFp8Primitive.inner_primitive is not None
        out, rsigma, amax = RmsNormFwdFp8Primitive.inner_primitive.bind(x,
                                                                        gamma,
                                                                        amax,
                                                                        scale,
                                                                        scale_inv,
                                                                        out_dtype=out_dtype,
                                                                        epsilon=epsilon)
        return out, rsigma, amax

    @staticmethod
    def batcher(batched_args, batch_dims, *, out_dtype, epsilon):
        """
        to describe batch rules for vmap
        """
        assert RmsNormFwdFp8Primitive.outer_primitive is not None
        x, gamma, amax, scale, scale_inv = batched_args
        x_bdim, _, amax_bdim, _, _ = batch_dims
        out_bdims = x_bdim, x_bdim, amax_bdim
        return RmsNormFwdFp8Primitive.outer_primitive.bind(x,
                                                           gamma,
                                                           amax,
                                                           scale,
                                                           scale_inv,
                                                           out_dtype=out_dtype,
                                                           epsilon=epsilon), out_bdims

    @staticmethod
    def infer_sharding_from_operands(out_dtype, epsilon, mesh, arg_infos, result_infos):
        del out_dtype, epsilon, result_infos
        x_spec = get_padded_spec(arg_infos[0])
        if x_spec[-1] is not None:
            warnings.warn(
                f"Does not support to shard hidden dim in {RmsNormFwdFp8Primitive.name}! " \
                f"Force to not shard the hidden dim, which might introduce extra collective ops, " \
                f"and hurt performance."
            )
        out_sharding = NamedSharding(mesh, PartitionSpec(*x_spec[:-1], None))
        rsigma_sharding = NamedSharding(mesh, PartitionSpec(*x_spec[:-1]))
        amax_sharding = NamedSharding(mesh, PartitionSpec(*get_padded_spec(arg_infos[2])))
        return (out_sharding, rsigma_sharding, amax_sharding)

    @staticmethod
    def partition(out_dtype, epsilon, mesh, arg_infos, result_infos):
        del result_infos
        x_spec = get_padded_spec(arg_infos[0])
        if x_spec[-1] is not None:
            warnings.warn(
                f"Does not support to shard hidden dim in {RmsNormFwdFp8Primitive.name}! " \
                f"Force to not shard the hidden dim, which might introduce extra collective ops, " \
                f"and hurt performance."
            )
        x_sharding = NamedSharding(mesh, PartitionSpec(*x_spec[:-1], None))
        g_sharding = NamedSharding(mesh, PartitionSpec(*get_padded_spec(arg_infos[1])))
        out_sharding = x_sharding
        rsigma_sharding = NamedSharding(mesh, PartitionSpec(*get_padded_spec(arg_infos[0])[:-1]))
        amax_sharding = NamedSharding(mesh, PartitionSpec(*get_padded_spec(arg_infos[2])))
        fp8_meta_sharding = amax_sharding
        arg_shardings = (x_sharding, g_sharding) + (fp8_meta_sharding,) * 3
        out_shardings = (out_sharding, rsigma_sharding, amax_sharding)

        def sharded_impl(x, gamma, amax, scale, scale_inv):
            local_x, local_rsigma, local_amax= \
                RmsNormFwdFp8Primitive.impl(x, gamma, amax, scale, scale_inv,
                                            out_dtype=out_dtype, epsilon=epsilon)
            global_updated_amax = all_reduce_max_along_all_axes_except_PP(local_amax)

            return local_x, local_rsigma, global_updated_amax

        return mesh, sharded_impl, out_shardings, arg_shardings


register_primitive(RmsNormFwdFp8Primitive)


def rmsnorm_fwd_fp8(x: jnp.ndarray, gamma: jnp.ndarray, amax: jnp.ndarray, scale: jnp.ndarray,
                    scale_inv: jnp.ndarray, out_dtype: jnp.dtype, epsilon: float):
    """
    Wrapper for TE rmsnorm fwd (fp8 out)
    """
    return RmsNormFwdFp8Primitive.outer_primitive.bind(x,
                                                       gamma,
                                                       amax,
                                                       scale,
                                                       scale_inv,
                                                       out_dtype=out_dtype,
                                                       epsilon=epsilon)


class GatedGeluFp8Primitive(BasePrimitive):
    """
    Gated Gelu FP8 Primitive
    """
    name = "te_gated_gelu_fp8"
    multiple_results = True
    impl_static_args = (4,)    #out_dtype
    inner_primitive = None
    outer_primitive = None

    @staticmethod
    def abstract(x_aval, amax_aval, scale_aval, scale_inv_aval, *, out_dtype):
        """
        te_gated_gelu_p abstract
        """
        dtype = dtypes.canonicalize_dtype(x_aval.dtype)
        # Currently only support casting to E4M3 only in C side.
        assert out_dtype == jnp.float8_e4m3fn
        assert dtype in [jnp.float32, jnp.float16, jnp.bfloat16]
        assert amax_aval.dtype == jnp.float32
        assert scale_aval.dtype == jnp.float32
        assert scale_inv_aval.dtype == jnp.float32

        assert x_aval.shape[-2] == 2    # Assume x in (....., 2, hidden)
        hidden_size = x_aval.shape[-1]
        batch_shape = x_aval.shape[:-2]
        out_shape = (batch_shape) + (hidden_size,)
        out_aval = x_aval.update(shape=out_shape, dtype=out_dtype)
        updated_amax_aval = amax_aval.update(shape=amax_aval.shape, dtype=amax_aval.dtype)

        return out_aval, updated_amax_aval

    @staticmethod
    def lowering(ctx, x, amax, scale, scale_inv, *, out_dtype):
        """
        te_gated_gelu_p lowering rules
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

        opaque = transformer_engine_jax.pack_common_descriptor((batch_size, out_shape[-1]),
                                                               jax_dtype_to_te_dtype(x_aval.dtype),
                                                               jax_dtype_to_te_dtype(out_dtype))

        out = custom_caller(GatedGeluFp8Primitive.name,
                            args,
                            opaque,
                            False,
                            operand_output_aliases={1: 1})

        return out

    @staticmethod
    def impl(x, amax, scale, scale_inv, out_dtype):
        """
        to describe implementation
        """
        assert GatedGeluFp8Primitive.inner_primitive is not None
        out, updated_amax = GatedGeluFp8Primitive.inner_primitive.bind(x,
                                                                       amax,
                                                                       scale,
                                                                       scale_inv,
                                                                       out_dtype=out_dtype)
        return out, updated_amax

    @staticmethod
    def batcher(batched_args, batch_dims, *, out_dtype):
        """
        to describe batch rules for vmap
        """
        assert GatedGeluFp8Primitive.outer_primitive is not None
        x, amax, scale, scale_inv = batched_args
        x_bdim, amax_bdim, _, _ = batch_dims

        out_bdims = x_bdim, amax_bdim
        return GatedGeluFp8Primitive.outer_primitive.bind(x,
                                                          amax,
                                                          scale,
                                                          scale_inv,
                                                          out_dtype=out_dtype), out_bdims

    @staticmethod
    def infer_sharding_from_operands(out_dtype, mesh, arg_infos, result_infos):
        del out_dtype, result_infos
        x_spec = get_padded_spec(arg_infos[0])
        out_sharding = NamedSharding(mesh, PartitionSpec(*x_spec[:-2], x_spec[-1]))
        amax_sharding = NamedSharding(mesh, PartitionSpec(*get_padded_spec(arg_infos[1])))
        return (out_sharding, amax_sharding)

    @staticmethod
    def partition(out_dtype, mesh, arg_infos, result_infos):
        del result_infos
        x_spec = get_padded_spec(arg_infos[0])
        out_sharding = NamedSharding(mesh, PartitionSpec(*x_spec[:-2], x_spec[-1]))
        amax_sharding = NamedSharding(mesh, PartitionSpec(*get_padded_spec(arg_infos[1])))
        arg_shardings = tuple(arg_i.sharding for arg_i in arg_infos)
        out_shardings = (out_sharding, amax_sharding)

        def sharded_impl(x, amax, scale, scale_inv):
            local_x, local_amax = GatedGeluFp8Primitive.impl(x,
                                                             amax,
                                                             scale,
                                                             scale_inv,
                                                             out_dtype=out_dtype)
            global_updated_amax = all_reduce_max_along_all_axes_except_PP(local_amax)

            return local_x, global_updated_amax

        return mesh, sharded_impl, out_shardings, arg_shardings


register_primitive(GatedGeluFp8Primitive)


def gated_gelu_fp8(x: jnp.ndarray, amax: jnp.ndarray, scale: jnp.ndarray, scale_inv: jnp.ndarray,
                   out_dtype: jnp.dtype) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    gated gelu wrapper
    Return FP8(geglu(x))
    """
    return GatedGeluFp8Primitive.outer_primitive.bind(x,
                                                      amax,
                                                      scale,
                                                      scale_inv,
                                                      out_dtype=out_dtype)


class DgatedGeluCastTransposePrimitive(BasePrimitive):
    """
    Dgated Gelu Cast Transpose Primitive
    """
    name = "te_dgated_gelu_cast_transpose"
    multiple_results = True
    impl_static_args = (5, 6)    # out_dtype, static_axis_boundary
    inner_primitive = None
    outer_primitive = None

    @staticmethod
    def abstract(dz_aval, x_aval, amax_aval, scale_aval, scale_inv_aval, *, out_dtype,
                 static_axis_boundary):
        """
        te_dgated_gelu_cast_transpose_p abstract
        """
        dtype = dtypes.canonicalize_dtype(dz_aval.dtype)
        assert dtype in [jnp.float32, jnp.float16, jnp.bfloat16]
        assert x_aval.dtype == dtype
        assert x_aval.shape[-2] == 2    # Linear + GeLU
        assert amax_aval.dtype == jnp.float32
        assert scale_aval.dtype == jnp.float32
        assert scale_inv_aval.dtype == jnp.float32
        ir_hidden_szie = dz_aval.shape[-1]
        gi_hidden_size = x_aval.shape[-1]
        assert ir_hidden_szie == gi_hidden_size
        t_shape = _multidim_transpose(x_aval.shape, static_axis_boundary, -2)
        out = dz_aval.update(shape=x_aval.shape, dtype=out_dtype)
        t_out = dz_aval.update(shape=t_shape, dtype=out_dtype)
        updated_amax_aval = amax_aval.update(shape=amax_aval.shape, dtype=amax_aval.dtype)
        return out, t_out, updated_amax_aval

    @staticmethod
    def lowering(ctx, dz, x, amax, scale, scale_inv, *, out_dtype, static_axis_boundary):
        """
        te_dgated_gelu_cast_transpose_p lowering rules
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
        assert x_shape[-2] == 2    # Linear + GeLU
        ir_hidden_szie = ir_dz_shape[-1]
        gi_hidden_size = x_shape[-1]
        assert ir_hidden_szie == gi_hidden_size
        ir_out_dtype = jax_dtype_to_ir_dtype(out_dtype)
        ir_amax_type = ir.RankedTensorType(amax.type)
        ir_amax_dtype = ir_amax_type.element_type
        ir_amax_shape = ir_amax_type.shape
        ir_scale_shape = ir_amax_shape
        ir_scale_inv_shape = ir_amax_shape
        transposed_x_shape = _multidim_transpose(x_shape, static_axis_boundary, -2)
        out_types = [
            ir.RankedTensorType.get(x_shape, ir_out_dtype),
            ir.RankedTensorType.get(transposed_x_shape, ir_out_dtype),
            ir.RankedTensorType.get(ir_amax_shape, ir_amax_dtype),
        ]
        operands = [dz, x, amax, scale, scale_inv]
        operand_shapes = [ir_dz_shape, x_shape, ir_amax_shape, ir_scale_shape, ir_scale_inv_shape]
        args = CustomCallArgsWrapper(out_types, operands, operand_shapes)
        contracted_x_shape = (x_batch_size, x_shape[-1])
        opaque = transformer_engine_jax.pack_common_descriptor(contracted_x_shape,
                                                               jax_dtype_to_te_dtype(dz_aval.dtype),
                                                               jax_dtype_to_te_dtype(out_dtype))

        out = custom_caller(DgatedGeluCastTransposePrimitive.name,
                            args,
                            opaque,
                            False,
                            operand_output_aliases={2: 2})

        return out

    @staticmethod
    def impl(dz, x, amax, scale, scale_inv, out_dtype, static_axis_boundary):
        """
        to describe implementation
        """
        assert DgatedGeluCastTransposePrimitive.inner_primitive is not None
        out, t_out, updated_amax = DgatedGeluCastTransposePrimitive.inner_primitive.bind(
            dz,
            x,
            amax,
            scale,
            scale_inv,
            out_dtype=out_dtype,
            static_axis_boundary=static_axis_boundary)
        return out, t_out, updated_amax

    @staticmethod
    def batcher(batched_args, batch_dims, *, out_dtype, static_axis_boundary):
        """
        to describe batch rules for vmap
        """
        del static_axis_boundary
        assert DgatedGeluCastTransposePrimitive.outer_primitive is not None
        dz, x, amax, scale, scale_inv = batched_args
        x_bdim, _, amax_bdim, _, _ = batch_dims

        out_bdims = x_bdim, x_bdim, amax_bdim
        return DgatedGeluCastTransposePrimitive.outer_primitive.bind(
            dz, x, amax, scale, scale_inv, out_dtype=out_dtype,
            static_axis_boundary=x_bdim), out_bdims

    @staticmethod
    def infer_sharding_from_operands(out_dtype, static_axis_boundary, mesh, arg_infos,
                                     result_infos):
        del out_dtype, result_infos
        x_spec = get_padded_spec(arg_infos[1])
        out_sharding = NamedSharding(mesh, PartitionSpec(*x_spec))
        xt_spec = _multidim_transpose(x_spec, static_axis_boundary, -2)
        tranposed_out_sharding = NamedSharding(mesh, PartitionSpec(*xt_spec))
        amax_sharding = NamedSharding(mesh, PartitionSpec(*get_padded_spec(arg_infos[2])))
        return (out_sharding, tranposed_out_sharding, amax_sharding)

    @staticmethod
    def partition(out_dtype, static_axis_boundary, mesh, arg_infos, result_infos):
        del result_infos
        x_spec = get_padded_spec(arg_infos[1])
        casted_x_sharding = NamedSharding(mesh, PartitionSpec(*x_spec))
        xt_spec = _multidim_transpose(x_spec, static_axis_boundary, -2)
        casted_transposed_x_sharding = NamedSharding(mesh, PartitionSpec(*xt_spec))

        amax_sharding = NamedSharding(mesh, PartitionSpec(*get_padded_spec(arg_infos[2])))
        arg_shardings = tuple(arg_i.sharding for arg_i in arg_infos)
        out_shardings = (casted_x_sharding, casted_transposed_x_sharding, amax_sharding)

        def sharded_impl(dz, x, amax, scale, scale_inv):
            local_out, local_t_out, local_amax = DgatedGeluCastTransposePrimitive.impl(
                dz,
                x,
                amax,
                scale,
                scale_inv,
                out_dtype=out_dtype,
                static_axis_boundary=static_axis_boundary)
            global_updated_amax = all_reduce_max_along_all_axes_except_PP(local_amax)
            return local_out, local_t_out, global_updated_amax

        return mesh, sharded_impl, out_shardings, arg_shardings


register_primitive(DgatedGeluCastTransposePrimitive)


def dgated_gelu_cast_transpose(
        dz: jnp.ndarray, x: jnp.ndarray, amax: jnp.ndarray, scale: jnp.ndarray,
        scale_inv: jnp.ndarray, out_dtype: TEDType,
        static_axis_boundary: int) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    cast transpose d_gated_gelu fusion wrapper
    Return FP8(dgeglu(inputs))
    """
    return DgatedGeluCastTransposePrimitive.outer_primitive.bind(
        dz,
        x,
        amax,
        scale,
        scale_inv,
        out_dtype=out_dtype,
        static_axis_boundary=static_axis_boundary)
