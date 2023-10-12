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
from jax.lib import xla_client
from jax import core, dtypes
from jax.core import ShapedArray
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

from .sharding import all_reduce_sum_along_dp_fsdp

for _name, _value in transformer_engine_jax.registrations().items():
    xla_client.register_custom_call_target(_name, _value, platform="CUDA")


def te_dtype_to_jax_dtype(te_dtype):
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
    return jnp.int8


def te_dtype_to_ir_dtype(te_dtype):
    """
    convert TE dtype to MLIR dtype
    """
    return dtype_to_ir_type(np.dtype(te_dtype_to_jax_dtype(te_dtype)))


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
    raise ValueError(f"Not support the {jax_dtype=}")


def get_padded_spec(arg_info):
    """
    Get padded spec for partitioning from arguments' information
    """
    if arg_info.sharding is None:
        return (None,) * arg_info.ndim
    ndim, spec = arg_info.ndim, arg_info.sharding.spec
    assert len(spec) <= ndim
    return spec + (None,) * (ndim - len(spec))


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
    impl_static_args = (3, 4)
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
        w_type = ir.RankedTensorType(gamma.type)
        w_shape = w_type.shape
        b_type = ir.RankedTensorType(beta.type)
        b_shape = b_type.shape

        assert w_type == b_type
        assert w_shape == b_shape

        # Output shape is same as the input shape, but the output type is same as the weight type.
        # See ln_api.cpp
        output_type = w_type.element_type
        ir_mu_dtype = ir.F32Type.get()
        ir_rsigma_dtype = ir.F32Type.get()

        out_shape = x_shape
        hidden_size = reduce(operator.mul, w_shape)
        batch_shape = out_shape[:-1]
        batch_size = reduce(operator.mul, x_shape) // hidden_size

        out_types = [
            ir.RankedTensorType.get(out_shape, output_type),
            ir.RankedTensorType.get(batch_shape, ir_mu_dtype),
            ir.RankedTensorType.get(batch_shape, ir_rsigma_dtype),
        ]
        operands = [x, gamma, beta]
        operand_shapes = [x_shape, w_shape, b_shape]
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
        assert x_spec[-1] is None, \
            f"Does not support to shard hidden dim in {LayerNormFwdPrimitive.name}"
        out_sharding = NamedSharding(mesh, PartitionSpec(*x_spec))
        mu_sharding = rsigma_sharding = NamedSharding(mesh, PartitionSpec(*x_spec[:-1]))
        return (out_sharding, mu_sharding, rsigma_sharding)

    @staticmethod
    def partition(zero_centered_gamma, epsilon, mesh, arg_infos, result_infos):
        del result_infos
        x_spec = NamedSharding(mesh, PartitionSpec(*get_padded_spec(arg_infos[0])))
        assert x_spec.spec[-1] is None, \
            f"Does not support to shard hidden dim in {LayerNormFwdPrimitive.name}"
        g_spec = NamedSharding(mesh, PartitionSpec(*get_padded_spec(arg_infos[1])))
        b_spec = NamedSharding(mesh, PartitionSpec(*get_padded_spec(arg_infos[2])))
        out_spec = x_spec
        mu_spec = rsigma_spec = NamedSharding(mesh,
                                              PartitionSpec(*get_padded_spec(arg_infos[0])[:-1]))
        arg_shardings = (x_spec, g_spec, b_spec)
        out_shardings = (out_spec, mu_spec, rsigma_spec)
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
    impl_static_args = (5, 6)
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
        w_type = ir.RankedTensorType(gamma.type)
        w_shape = w_type.shape
        b_type = ir.RankedTensorType(gamma.type)
        b_shape = b_type.shape
        assert w_type == b_type
        assert w_shape == b_shape

        dz_shape = ir.RankedTensorType(dz.type).shape
        mu_shape = ir.RankedTensorType(mu.type).shape
        rsigma_shape = ir.RankedTensorType(rsigma.type).shape

        hidden_size = reduce(operator.mul, w_shape)
        batch_size = reduce(operator.mul, x_shape) // hidden_size

        out_types = [
            ir.RankedTensorType.get(x_shape, x_type.element_type),
            ir.RankedTensorType.get(w_shape, w_type.element_type),
            ir.RankedTensorType.get(b_shape, b_type.element_type),
        ]
        operands = [dz, mu, rsigma, x, gamma]
        operand_shapes = [dz_shape, mu_shape, rsigma_shape, x_shape, w_shape]
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
        assert x_spec[-1] is None, \
            f"Does not support to shard hidden dim in {LayerNormBwdPrimitive.name}"
        g_b_spec = get_padded_spec(arg_infos[4])
        dx_sharding = NamedSharding(mesh, PartitionSpec(*x_spec))
        dgamma_sharding = dbeta_sharding = NamedSharding(mesh, PartitionSpec(*g_b_spec))
        return dx_sharding, dgamma_sharding, dbeta_sharding

    @staticmethod
    def partition(zero_centered_gamma, epsilon, mesh, arg_infos, result_infos):
        del result_infos
        x_spec = get_padded_spec(arg_infos[1])
        assert x_spec[-1] is None, \
            f"Does not support to shard hidden dim in {LayerNormBwdPrimitive.name}"
        g_b_spec = get_padded_spec(arg_infos[4])
        dx_sharding = NamedSharding(mesh, PartitionSpec(*x_spec))
        dgamma_sharding = dbeta_sharding = NamedSharding(mesh, PartitionSpec(*g_b_spec))
        out_shardings = dx_sharding, dgamma_sharding, dbeta_sharding
        x_shardings = (NamedSharding(mesh, PartitionSpec(*x_spec)),) * 2
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
    impl_static_args = (2,)
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
        w_type = ir.RankedTensorType(gamma.type)
        w_shape = w_type.shape
        rsigma_element_type = ir.F32Type.get()

        out_shape = x_shape
        hidden_size = reduce(operator.mul, w_shape)
        batch_shape = out_shape[:-1]
        batch_size = reduce(operator.mul, x_shape) // hidden_size

        out_types = [
            ir.RankedTensorType.get(out_shape, x_type.element_type),
            ir.RankedTensorType.get(batch_shape, rsigma_element_type),
        ]
        operands = [x, gamma]
        operand_shapes = [x_shape, w_shape]
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
        assert x_spec[-1] is None, \
            f"Does not support to shard hidden dim in {RmsNormFwdPrimitive.name}"
        out_sharding = NamedSharding(mesh, PartitionSpec(*x_spec))
        rsigma_sharding = NamedSharding(mesh, PartitionSpec(*x_spec[:-1]))
        return (out_sharding, rsigma_sharding)

    @staticmethod
    def partition(epsilon, mesh, arg_infos, result_infos):
        del result_infos
        x_spec = NamedSharding(mesh, PartitionSpec(*get_padded_spec(arg_infos[0])))
        assert x_spec.spec[-1] is None, \
            f"Does not support to shard hidden dim in {RmsNormFwdPrimitive.name}"
        g_spec = NamedSharding(mesh, PartitionSpec(*get_padded_spec(arg_infos[1])))
        out_spec = x_spec
        rsigma_spec = NamedSharding(mesh, PartitionSpec(*get_padded_spec(arg_infos[0])[:-1]))
        arg_shardings = (x_spec, g_spec)
        out_shardings = (out_spec, rsigma_spec)
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
    impl_static_args = (4,)
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
        w_type = ir.RankedTensorType(gamma.type)
        w_shape = w_type.shape
        go_shape = ir.RankedTensorType(dz.type).shape
        rsigma_shape = ir.RankedTensorType(rsigma.type).shape

        hidden_size = reduce(operator.mul, w_shape)
        batch_size = reduce(operator.mul, x_shape) // hidden_size

        out_types = [
            ir.RankedTensorType.get(x_shape, x_type.element_type),
            ir.RankedTensorType.get(w_shape, w_type.element_type),
        ]
        operands = [dz, rsigma, x, gamma]
        operand_shapes = [go_shape, rsigma_shape, x_shape, w_shape]
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
        assert x_spec[-1] is None, \
            f"Does not support to shard hidden dim in {RmsNormBwdPrimitive.name}"
        g_spec = get_padded_spec(arg_infos[3])
        dx_sharding = NamedSharding(mesh, PartitionSpec(*x_spec))
        dgamma_sharding = NamedSharding(mesh, PartitionSpec(*g_spec))
        return dx_sharding, dgamma_sharding

    @staticmethod
    def partition(epsilon, mesh, arg_infos, result_infos):
        del result_infos
        x_spec = get_padded_spec(arg_infos[1])
        assert x_spec[-1] is None, \
            f"Does not support to shard hidden dim in {RmsNormBwdPrimitive.name}"
        g_spec = get_padded_spec(arg_infos[3])
        dx_sharding = NamedSharding(mesh, PartitionSpec(*x_spec))
        dgamma_sharding = NamedSharding(mesh, PartitionSpec(*g_spec))
        out_shardings = dx_sharding, dgamma_sharding
        x_shardings = (NamedSharding(mesh, PartitionSpec(*x_spec)),) * 2
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

<<<<<<< HEAD
    q_type: jnp.dtype
    kv_type: jnp.dtype
    qkv_layout: NVTE_QKV_Layout
    attn_bias_type: NVTE_Bias_Type
    attn_mask_type: NVTE_Mask_Type
    dropout_probability: float
    max_seqlen_q: int
    max_seqlen_kv: int
    head_dim: int
=======
    @staticmethod
    @abstractmethod
    def is_kernel_available(batch: int, heads: int, q_seqlen: int, k_seqlen: int,
                            dtype: jnp.dtype) -> bool:
        """Check Softmax kernel availability based on size"""
        raise NotImplementedError
>>>>>>> Migrating both FWD and BWD of all kinds of softmax from xmap to custom_partitioning.

    @staticmethod
    def get_batch_per_block(k_seqlen: int) -> int:
        """Get batch per CTA in Softmax kernels"""
        threads_per_warp = 32
        threads_per_block = 128    # Depends on the kernel implmentation

<<<<<<< HEAD
    def get_fused_attn_backend(self):
        """Get the fused attention kernel backend"""
        return transformer_engine_jax.get_fused_attn_backend(jax_dtype_to_te_dtype(self.q_type),
                                                             jax_dtype_to_te_dtype(self.kv_type),
                                                             self.qkv_layout, self.attn_bias_type,
                                                             self.attn_mask_type,
                                                             self.dropout_probability,
                                                             self.max_seqlen_q, self.max_seqlen_kv,
                                                             self.head_dim)
=======
        pow2 = 1 << (k_seqlen - 1).bit_length()
        warp_size = pow2 if pow2 < threads_per_warp else threads_per_warp
        batches_per_warp = 2 if pow2 <= 128 else 1
        warps_per_block = threads_per_block // warp_size
        batches_per_block = warps_per_block * batches_per_warp
        return batches_per_block
>>>>>>> Migrating both FWD and BWD of all kinds of softmax from xmap to custom_partitioning.

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
    impl_static_args = (1,)
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
    impl_static_args = (2,)
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
    impl_static_args = (2,)
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
    impl_static_args = (2,)
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
    impl_static_args = (1,)
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
    impl_static_args = (2,)
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


# Deprecating Items ---------------------------------------------------------------
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


def merge_named_shape(base, new):
    """
    merge named shape(ie, dict), no key conflict
    """
    output_named_shape = {**base}
    for key in new:
        if key in output_named_shape:
            assert output_named_shape[key] == new[key], \
                f"The value of named shape with a same name should be equal between" \
                f" base and new in merge_named_shape, but got base[{key}]=" \
                f"{output_named_shape[key]} and {new[key]=}"
        else:
            output_named_shape[key] = new[key]
    return output_named_shape


class BasePrimitiveLegacy(metaclass=ABCMeta):
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


def register_primitive_legacy(cls):
    """
    register jax primitive
    """
    p = core.Primitive(cls.name)
    p.multiple_results = cls.multiple_results
    p.def_impl(partial(xla.apply_primitive, p))
    p.def_abstract_eval(cls.abstract)
    mlir.register_lowering(p, cls.lowering, platform='cuda')
    return p


class TransposePrimitive(BasePrimitiveLegacy):
    """
    Transpose Primitive
    """
    name = "te_transpose"
    multiple_results = False

    @staticmethod
    def abstract(inputs, *, dtype):
        """
        _transpose abstract
        """
        in_dtype = dtypes.canonicalize_dtype(inputs.dtype)
        out_dtype = te_dtype_to_jax_dtype(dtype)

        assert len(inputs.shape) == 2
        assert isinstance(dtype, TEDType)
        assert in_dtype == out_dtype

        return ShapedArray((inputs.shape[1], inputs.shape[0]),
                           in_dtype,
                           named_shape=inputs.named_shape)

    @staticmethod
    def lowering(ctx, inputs, *, dtype):
        """
        _transpose cuda lowering
        """

        in_aval = ctx.avals_in[0]
        assert in_aval.dtype in [jnp.float32, jnp.float16, jnp.bfloat16, jnp.int8]

        ir_in_type = ir.RankedTensorType(inputs.type)
        ir_in_shape = ir_in_type.shape
        ir_out_dtype = te_dtype_to_ir_dtype(dtype)

        out_types = [ir.RankedTensorType.get([ir_in_shape[1], ir_in_shape[0]], ir_out_dtype)]
        operands = [inputs]
        operand_shapes = [ir_in_shape]
        args = CustomCallArgsWrapper(out_types, operands, operand_shapes)

        assert len(ir_in_shape) == 2
        opaque = transformer_engine_jax.pack_common_descriptor(ir_in_shape, dtype, dtype)

        out = custom_caller(TransposePrimitive.name, args, opaque, False)

        return [out]


_transpose_p = register_primitive_legacy(TransposePrimitive)


def transpose(inputs: jnp.ndarray, dtype: TEDType) -> jnp.ndarray:
    """
    transpose wrapper
    Assume input has two dimension shape
    """
    return _transpose_p.bind(inputs, dtype=dtype)


class CastTransposePrimitive(BasePrimitiveLegacy):
    """
    Cast Transpose Primitive
    """
    name = "te_cast_transpose"
    multiple_results = True

    @staticmethod
    def abstract(inputs, amax, scale, scale_inv, *, out_dtype):
        """
        te_cast_transpose_p abstract
        """
        dtype = dtypes.canonicalize_dtype(inputs.dtype)
        assert len(inputs.shape) == 2
        assert dtype in [jnp.float32, jnp.float16, jnp.bfloat16]
        assert amax.dtype == jnp.float32
        assert scale.dtype == jnp.float32
        assert scale_inv.dtype == jnp.float32
        out_dtype = te_dtype_to_jax_dtype(out_dtype)
        # input_cast, input_cast_trans, amax
        return (ShapedArray((inputs.shape[0], inputs.shape[1]),
                            out_dtype,
                            named_shape=inputs.named_shape),
                ShapedArray((inputs.shape[1], inputs.shape[0]),
                            out_dtype,
                            named_shape=inputs.named_shape),
                ShapedArray((1,), amax.dtype, named_shape=amax.named_shape))

    @staticmethod
    def lowering(ctx, inputs, amax, scale, scale_inv, *, out_dtype):
        """
        te_cast_transpose_p lowering rules
        """
        in_aval, amax_aval, scale_aval, scale_inv_aval = ctx.avals_in
        assert in_aval.dtype in [jnp.float32, jnp.float16, jnp.bfloat16]
        assert amax_aval.dtype == jnp.float32
        assert scale_aval.dtype == jnp.float32
        assert scale_inv_aval.dtype == jnp.float32
        ir_in_type = ir.RankedTensorType(inputs.type)
        ir_in_shape = ir_in_type.shape
        ir_out_dtype = te_dtype_to_ir_dtype(out_dtype)
        ir_amax_type = ir.RankedTensorType(amax.type)
        ir_amax_dtype = ir_amax_type.element_type
        ir_amax_shape = ir_amax_type.shape
        ir_scale_shape = ir_amax_shape
        ir_scale_inv_shape = ir_amax_shape

        out_types = [
            ir.RankedTensorType.get([ir_in_shape[0], ir_in_shape[1]], ir_out_dtype),
            ir.RankedTensorType.get([ir_in_shape[1], ir_in_shape[0]], ir_out_dtype),
            ir.RankedTensorType.get(ir_amax_shape, ir_amax_dtype),
        ]
        operands = [inputs, amax, scale, scale_inv]
        operand_shapes = [ir_in_shape, ir_amax_shape, ir_scale_shape, ir_scale_inv_shape]
        args = CustomCallArgsWrapper(out_types, operands, operand_shapes)

        assert len(ir_in_shape) == 2
        opaque = transformer_engine_jax.pack_common_descriptor(ir_in_shape,
                                                               jax_dtype_to_te_dtype(in_aval.dtype),
                                                               out_dtype)

        out = custom_caller(CastTransposePrimitive.name,
                            args,
                            opaque,
                            False,
                            operand_output_aliases={1: 2})

        return out


_cast_transpose_p = register_primitive_legacy(CastTransposePrimitive)


def cast_transpose(inputs: jnp.ndarray, amax: jnp.ndarray, scale: jnp.ndarray,
                   scale_inv: jnp.ndarray,
                   out_dtype: TEDType) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    cast transpose wrapper
    Return two tensors, FP8(inputs) and FP8(inputs.T), which are scaled by `scale`
    """
    return _cast_transpose_p.bind(inputs, amax, scale, scale_inv, out_dtype=out_dtype)


class GatedGeluPrimitive(BasePrimitiveLegacy):
    """
    Gated Gelu Primitive
    """
    name = "te_gated_gelu"
    multiple_results = False

    @staticmethod
    def abstract(inputs):
        """
        te_gated_gelu_p abstract
        """
        dtype = dtypes.canonicalize_dtype(inputs.dtype)
        assert dtype in [jnp.float32, jnp.float16, jnp.bfloat16]
        inputs_shape = inputs.shape
        hidden_size = inputs_shape[-1]
        # In Transformer, batch_shape = (batch,  seqlen, )
        batch_shapes = inputs_shape[:-1]
        assert hidden_size % 2 == 0
        inputs_shape = inputs.shape
        out_shape = (batch_shapes) + (hidden_size // 2,)

        return ShapedArray(out_shape, dtype, named_shape=inputs.named_shape)

    @staticmethod
    def lowering(ctx, inputs):
        """
        te_gated_gelu_p lowering rules
        """
        (in_aval,) = ctx.avals_in
        assert in_aval.dtype in [jnp.float32, jnp.float16, jnp.bfloat16]
        ir_in_type = ir.RankedTensorType(inputs.type)
        ir_in_shape = ir_in_type.shape
        out_shape = ir_in_shape[:-1] + [ir_in_shape[-1] // 2]

        out_types = [
            ir.RankedTensorType.get(out_shape, ir_in_type.element_type),
        ]
        operands = [inputs]
        operand_shapes = [ir_in_shape]
        args = CustomCallArgsWrapper(out_types, operands, operand_shapes)

        hidden_size = ir_in_shape[-1]
        # In Transformer, batch_size = batch x seqlen
        batch_size = reduce(operator.mul, ir_in_shape[:-1])
        in_dtype = jax_dtype_to_te_dtype(in_aval.dtype)
        opaque = transformer_engine_jax.pack_common_descriptor((batch_size, hidden_size // 2),
                                                               in_dtype, in_dtype)

        out = custom_caller(GatedGeluPrimitive.name, args, opaque, False)

        return [out]


_gated_gelu_p = register_primitive_legacy(GatedGeluPrimitive)


def gated_gelu(inputs: jnp.ndarray) -> jnp.ndarray:
    """
    gated gelu wrapper
    Return FP8(geglu(inputs))
    Assume inputs has two dimensions shape and the memory layout is (N, 2, H)
    """
    return _gated_gelu_p.bind(inputs)


class GatedGeluFp8Primitive(BasePrimitiveLegacy):
    """
    Gated Gelu FP8 Primitive
    """
    name = "te_gated_gelu_fp8"
    multiple_results = True

    @staticmethod
    def abstract(inputs, amax, scale, scale_inv, *, out_dtype):
        """
        te_gated_gelu_p abstract
        """
        dtype = dtypes.canonicalize_dtype(inputs.dtype)
        assert dtype in [jnp.float32, jnp.float16, jnp.bfloat16]
        assert amax.dtype == jnp.float32
        assert scale.dtype == jnp.float32
        assert scale_inv.dtype == jnp.float32
        out_dtype = te_dtype_to_jax_dtype(out_dtype)

        assert len(inputs.shape) == 2
        hidden_size = inputs.shape[1]
        batch_size = inputs.shape[0]    # In Transformer, batch_size = batch x seqlen

        # input_cast, input_cast_trans, amax
        return (ShapedArray((batch_size, hidden_size // 2),
                            out_dtype,
                            named_shape=inputs.named_shape),
                ShapedArray((1,), amax.dtype, named_shape=amax.named_shape))

    @staticmethod
    def lowering(ctx, inputs, amax, scale, scale_inv, *, out_dtype):
        """
        te_gated_gelu_p lowering rules
        """
        in_aval, amax_aval, scale_aval, scale_inv_aval = ctx.avals_in
        assert in_aval.dtype in [jnp.float32, jnp.float16, jnp.bfloat16]
        assert amax_aval.dtype == jnp.float32
        assert scale_aval.dtype == jnp.float32
        assert scale_inv_aval.dtype == jnp.float32
        ir_in_type = ir.RankedTensorType(inputs.type)
        ir_in_shape = ir_in_type.shape
        ir_out_dtype = te_dtype_to_ir_dtype(out_dtype)
        ir_amax_type = ir.RankedTensorType(amax.type)
        ir_amax_dtype = ir_amax_type.element_type
        ir_amax_shape = ir_amax_type.shape
        ir_scale_shape = ir_amax_shape
        ir_scale_inv_shape = ir_amax_shape

        hidden_size = ir_in_shape[1]
        batch_size = ir_in_shape[0]    # In Transformer, batch_size = batch x seqlen
        out_types = [
            ir.RankedTensorType.get([batch_size, hidden_size // 2], ir_out_dtype),
            ir.RankedTensorType.get(ir_amax_shape, ir_amax_dtype),
        ]
        operands = [inputs, amax, scale, scale_inv]
        operand_shapes = [ir_in_shape, ir_amax_shape, ir_scale_shape, ir_scale_inv_shape]
        args = CustomCallArgsWrapper(out_types, operands, operand_shapes)

        opaque = transformer_engine_jax.pack_common_descriptor(
            (ir_in_shape[0], ir_in_shape[1] // 2), jax_dtype_to_te_dtype(in_aval.dtype), out_dtype)

        out = custom_caller(GatedGeluFp8Primitive.name,
                            args,
                            opaque,
                            False,
                            operand_output_aliases={1: 1})

        return out


_gated_gelu_fp8_p = register_primitive_legacy(GatedGeluFp8Primitive)


def gated_gelu_fp8(inputs: jnp.ndarray, amax: jnp.ndarray, scale: jnp.ndarray,
                   scale_inv: jnp.ndarray,
                   out_dtype: TEDType) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    cast gated gelu wrapper
    Return FP8(geglu(inputs))
    Assume inputs has two dimensions shape and the memory layout is (N, 2, H)
    """
    return _gated_gelu_fp8_p.bind(inputs, amax, scale, scale_inv, out_dtype=out_dtype)


class DgatedGeluPrimitive(BasePrimitiveLegacy):
    """
    Dgated Gelu Primitive
    """
    name = "te_dgated_gelu"
    multiple_results = False

    @staticmethod
    def abstract(inputs, gelu_inputs):
        """
        te_dgated_gelu_p abstract
        """
        dtype = dtypes.canonicalize_dtype(inputs.dtype)
        assert dtype in [jnp.float32, jnp.float16, jnp.bfloat16]
        assert gelu_inputs.dtype == dtype
        for axis in range(len(inputs.shape) - 1):
            assert inputs.shape[axis] == gelu_inputs.shape[axis]

        i_hidden_size = inputs.shape[-1]
        g_hidden_szie = gelu_inputs.shape[-1]
        assert i_hidden_size * 2 == g_hidden_szie
        return ShapedArray(gelu_inputs.shape, dtype, named_shape=inputs.named_shape)

    @staticmethod
    def lowering(ctx, inputs, gelu_inputs):
        """
        te_dgated_gelu_p lowering rules
        """
        in_aval, gi_aval = ctx.avals_in
        assert in_aval.dtype in [jnp.float32, jnp.float16, jnp.bfloat16]
        assert gi_aval.dtype == in_aval.dtype
        ir_in_type = ir.RankedTensorType(inputs.type)
        ir_in_shape = ir_in_type.shape
        gi_type = ir.RankedTensorType(gelu_inputs.type)
        gi_shape = gi_type.shape
        for axis in range(len(ir_in_shape) - 1):
            assert ir_in_shape[axis] == gi_shape[axis]

        # In Transformer, batch_size = batch x seqlen
        ir_batch_szie = reduce(operator.mul, ir_in_shape[:-1])
        i_hidden_size = ir_in_shape[-1]
        g_hidden_szie = gi_shape[-1]
        assert i_hidden_size * 2 == g_hidden_szie
        out_dtype = ir_in_type.element_type
        out_shape = gi_shape

        out_types = [
            ir.RankedTensorType.get(out_shape, out_dtype),
        ]
        operands = [inputs, gelu_inputs]
        operand_shapes = [ir_in_shape, gi_shape]
        args = CustomCallArgsWrapper(out_types, operands, operand_shapes)

        in_dtype = jax_dtype_to_te_dtype(in_aval.dtype)
        opaque = transformer_engine_jax.pack_common_descriptor((ir_batch_szie, i_hidden_size),
                                                               in_dtype, in_dtype)

        out = custom_caller(DgatedGeluPrimitive.name, args, opaque, False)

        return [out]


_dgated_gelu_p = register_primitive_legacy(DgatedGeluPrimitive)


def dgated_gelu(inputs: jnp.ndarray, gelu_inputs: jnp.ndarray) -> jnp.ndarray:
    """
    dgated_gelu fusion wrapper
    Return dgeglu(inputs)
    """
    return _dgated_gelu_p.bind(inputs, gelu_inputs)


class DgatedGeluCastTransposePrimitive(BasePrimitiveLegacy):
    """
    Dgated Gelu Cast Transpose Primitive
    """
    name = "te_dgated_gelu_cast_transpose"
    multiple_results = True

    @staticmethod
    def abstract(inputs, gelu_inputs, amax, scale, scale_inv, *, out_dtype):
        """
        te_dgated_gelu_cast_transpose_p abstract
        """
        dtype = dtypes.canonicalize_dtype(inputs.dtype)
        assert dtype in [jnp.float32, jnp.float16, jnp.bfloat16]
        assert gelu_inputs.dtype == dtype
        assert len(inputs.shape) == 2
        assert len(gelu_inputs.shape) == 2
        ir_batch_szie = inputs.shape[0]
        gi_batch_size = gelu_inputs.shape[0]
        assert ir_batch_szie == gi_batch_size
        ir_hidden_szie = inputs.shape[1]
        gi_hidden_size = gelu_inputs.shape[1]
        assert ir_hidden_szie * 2 == gi_hidden_size
        assert amax.dtype == jnp.float32
        assert scale.dtype == jnp.float32
        assert scale_inv.dtype == jnp.float32
        out_dtype = te_dtype_to_jax_dtype(out_dtype)
        # input_cast, input_cast_trans, amax
        return (ShapedArray((gi_batch_size, gi_hidden_size),
                            out_dtype,
                            named_shape=inputs.named_shape),
                ShapedArray((gi_hidden_size, gi_batch_size),
                            out_dtype,
                            named_shape=inputs.named_shape),
                ShapedArray((1,), amax.dtype, named_shape=amax.named_shape))

    @staticmethod
    def lowering(ctx, inputs, gelu_inputs, amax, scale, scale_inv, *, out_dtype):
        """
        te_dgated_gelu_cast_transpose_p lowering rules
        """
        in_aval, gi_aval, amax_aval, scale_aval, scale_inv_aval = ctx.avals_in
        assert in_aval.dtype in [jnp.float32, jnp.float16, jnp.bfloat16]
        assert gi_aval.dtype == in_aval.dtype
        assert amax_aval.dtype == jnp.float32
        assert scale_aval.dtype == jnp.float32
        assert scale_inv_aval.dtype == jnp.float32
        ir_in_type = ir.RankedTensorType(inputs.type)
        ir_in_shape = ir_in_type.shape
        gi_type = ir.RankedTensorType(gelu_inputs.type)
        gi_shape = gi_type.shape
        ir_batch_szie = ir_in_shape[0]
        gi_batch_size = gi_shape[0]
        assert ir_batch_szie == gi_batch_size
        ir_hidden_szie = ir_in_shape[1]
        gi_hidden_size = gi_shape[1]
        assert ir_hidden_szie * 2 == gi_hidden_size
        ir_out_dtype = te_dtype_to_ir_dtype(out_dtype)
        ir_amax_type = ir.RankedTensorType(amax.type)
        ir_amax_dtype = ir_amax_type.element_type
        ir_amax_shape = ir_amax_type.shape
        ir_scale_shape = ir_amax_shape
        ir_scale_inv_shape = ir_amax_shape

        out_types = [
            ir.RankedTensorType.get([gi_batch_size, gi_hidden_size], ir_out_dtype),
            ir.RankedTensorType.get([gi_hidden_size, gi_batch_size], ir_out_dtype),
            ir.RankedTensorType.get(ir_amax_shape, ir_amax_dtype),
        ]
        operands = [inputs, gelu_inputs, amax, scale, scale_inv]
        operand_shapes = [ir_in_shape, gi_shape, ir_amax_shape, ir_scale_shape, ir_scale_inv_shape]
        args = CustomCallArgsWrapper(out_types, operands, operand_shapes)

        opaque = transformer_engine_jax.pack_common_descriptor((ir_batch_szie, ir_hidden_szie),
                                                               jax_dtype_to_te_dtype(in_aval.dtype),
                                                               out_dtype)

        out = custom_caller(DgatedGeluCastTransposePrimitive.name,
                            args,
                            opaque,
                            False,
                            operand_output_aliases={2: 2})

        return out


_dgated_gelu_cast_transpose_p = register_primitive_legacy(DgatedGeluCastTransposePrimitive)


def dgated_gelu_cast_transpose(inputs: jnp.ndarray, gelu_inputs: jnp.ndarray, amax: jnp.ndarray,
                               scale: jnp.ndarray, scale_inv: jnp.ndarray,
                               out_dtype: TEDType) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    cast transpose d_gated_gelu fusion wrapper
    Return FP8(dgeglu(inputs))
    """
    return _dgated_gelu_cast_transpose_p.bind(inputs,
                                              gelu_inputs,
                                              amax,
                                              scale,
                                              scale_inv,
                                              out_dtype=out_dtype)


class GemmPrimitive(BasePrimitiveLegacy):
    """
    Gemm Primitive
    """
    name = "te_gemm"
    multiple_results = False

    @staticmethod
    def abstract(A, B, A_scale_inv, B_scale_inv, *, A_dtype, B_dtype, D_dtype, transa, transb,
                 use_split_accumulator):    # pylint: disable=unused-argument
        """
        te_gemm_p abstract
        """
        atype = dtypes.canonicalize_dtype(A.dtype)
        btype = dtypes.canonicalize_dtype(B.dtype)
        assert atype == te_dtype_to_jax_dtype(A_dtype)
        assert btype == te_dtype_to_jax_dtype(B_dtype)
        assert A_scale_inv.dtype == jnp.float32
        assert B_scale_inv.dtype == jnp.float32

        m = A.shape[0] if transa else A.shape[1]
        k = A.shape[1] if transa else A.shape[0]
        n = B.shape[1] if transb else B.shape[0]
        assert (transb and k == B.shape[0]) or k == B.shape[1]

        out_dtype = te_dtype_to_jax_dtype(D_dtype)
        return ShapedArray((n, m),
                           out_dtype,
                           named_shape=merge_named_shape(A.named_shape, B.named_shape))

    @staticmethod
    def lowering(ctx, A, B, A_scale_inv, B_scale_inv, *, A_dtype, B_dtype, D_dtype, transa, transb,
                 use_split_accumulator):
        """
        te_gemm_p lowering rules
        """
        A_aval, B_aval, A_scale_inv_aval, B_scale_inv_aval = ctx.avals_in
        assert A_aval.dtype == te_dtype_to_jax_dtype(A_dtype)
        assert B_aval.dtype == te_dtype_to_jax_dtype(B_dtype)
        assert A_scale_inv_aval.dtype == jnp.float32
        assert B_scale_inv_aval.dtype == jnp.float32
        A_type = ir.RankedTensorType(A.type)
        B_type = ir.RankedTensorType(B.type)
        A_shape = A_type.shape
        B_shape = B_type.shape
        A_scale_inv_shape = ir.RankedTensorType(A_scale_inv.type).shape
        B_scale_inv_shape = ir.RankedTensorType(B_scale_inv.type).shape

        m = A_shape[0] if transa else A_shape[1]
        k = A_shape[1] if transa else A_shape[0]
        n = B_shape[1] if transb else B_shape[0]
        assert (transb and k == B_shape[0]) or k == B_shape[1]

        ir_out_dtype = dtype_to_ir_type(np.dtype(te_dtype_to_jax_dtype(D_dtype)))
        out_types = [
            ir.RankedTensorType.get([n, m], ir_out_dtype),
        ]
        operands = [A, B, A_scale_inv, B_scale_inv]
        operand_shapes = [A_shape, B_shape, A_scale_inv_shape, B_scale_inv_shape]
        args = CustomCallArgsWrapper(out_types, operands, operand_shapes)

        # m, n, k here should be equal to transa=False and transb=False,
        # due to te_gemm's implementation.
        # Therefore, m=A_shape[1], n=B_shape[0], k=A_shape[0]
        opaque = transformer_engine_jax.pack_gemm_descriptor(A_shape[1], B_shape[0], A_shape[0],
                                                             A_dtype, B_dtype, D_dtype, transa,
                                                             transb, use_split_accumulator)

        out = custom_caller(GemmPrimitive.name, args, opaque, False)

        return [out]


_gemm_p = register_primitive_legacy(GemmPrimitive)


def gemm(A: jnp.ndarray,
         A_scale_inv: jnp.ndarray,
         A_type: TEDType,
         transa: bool,
         B: jnp.ndarray,
         B_scale_inv: jnp.ndarray,
         B_type: TEDType,
         transb: bool,
         D_type: TEDType,
         use_split_accumulator: bool = False) -> jnp.ndarray:
    """
    gemm wrapper
    """
    return _gemm_p.bind(A,
                        B,
                        A_scale_inv,
                        B_scale_inv,
                        A_dtype=A_type,
                        B_dtype=B_type,
                        D_dtype=D_type,
                        transa=transa,
                        transb=transb,
                        use_split_accumulator=use_split_accumulator)


class LayerNormFwdFp8Primitive(BasePrimitiveLegacy):
    """
    Layer Normalization Forward FP8 Primitive
    """
    name = "te_layernorm_forward_fp8"
    multiple_results = True

    @staticmethod
    def abstract(
            x,
            gamma,
            beta,
            amax,
            scale,
            scale_inv,
            **kwargs    # pylint: disable=unused-argument
    ):
        """
        LayerNorm fwd (fp8 out) abstract
        """
        x_dtype = dtypes.canonicalize_dtype(x.dtype)

        assert x_dtype in [jnp.float32, jnp.float16, jnp.bfloat16]
        assert amax.dtype == jnp.float32
        assert scale.dtype == jnp.float32
        assert scale_inv.dtype == jnp.float32

        out_dtype = jnp.int8
        mu_dtype = jnp.float32
        rsigma_dtype = jnp.float32

        assert gamma.size == beta.size

        hidden_szie = gamma.size
        # In Transformer, batch_size = batch x seqlen
        batch_size = x.size // hidden_szie

        return (
            ShapedArray(x.shape, out_dtype, named_shape=x.named_shape),    # output
            ShapedArray((batch_size,), mu_dtype, named_shape=x.named_shape),    # mu
            ShapedArray((batch_size,), rsigma_dtype, named_shape=x.named_shape),    # rsigma
            ShapedArray((1,), amax.dtype, named_shape=amax.named_shape),    # amax
        )

    @staticmethod
    def lowering(ctx, x, gamma, beta, amax, scale, scale_inv, *, zero_centered_gamma, epsilon):
        """
        LayerNorm fwd (fp8 out) lowering rules
        """
        x_aval, gamma_aval, beta_aval, amax_aval, scale_aval, scale_inv_aval = ctx.avals_in

        assert x_aval.dtype in [jnp.float32, jnp.float16, jnp.bfloat16]
        assert gamma_aval.dtype == beta_aval.dtype
        assert amax_aval.dtype == jnp.float32
        assert scale_aval.dtype == jnp.float32
        assert scale_inv_aval.dtype == jnp.float32

        x_type = ir.RankedTensorType(x.type)
        x_shape = x_type.shape
        w_type = ir.RankedTensorType(gamma.type)
        w_shape = w_type.shape
        b_type = ir.RankedTensorType(beta.type)
        b_shape = b_type.shape

        ir_out_dtype = dtype_to_ir_type(np.dtype(np.int8))
        ir_mu_dtype = ir.F32Type.get()
        ir_rsigma_dtype = ir.F32Type.get()
        ir_amax_type = ir.RankedTensorType(amax.type)
        ir_amax_dtype = ir_amax_type.element_type
        ir_amax_shape = ir_amax_type.shape
        ir_scale_shape = ir_amax_shape
        ir_scale_inv_shape = ir_amax_shape

        hidden_size = reduce(operator.mul, w_shape)
        # In Transformer, batch_size = batch x seqlen
        batch_size = reduce(operator.mul, x_shape) // hidden_size

        out_types = [
            ir.RankedTensorType.get(x_shape, ir_out_dtype),
            ir.RankedTensorType.get((batch_size,), ir_mu_dtype),
            ir.RankedTensorType.get((batch_size,), ir_rsigma_dtype),
            ir.RankedTensorType.get(ir_amax_shape, ir_amax_dtype),
        ]
        operands = [x, gamma, beta, amax, scale, scale_inv]
        operand_shapes = [
            x_shape, w_shape, b_shape, ir_amax_shape, ir_scale_shape, ir_scale_inv_shape
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


_layernorm_fwd_fp8_p = register_primitive_legacy(LayerNormFwdFp8Primitive)


def layernorm_fwd_fp8(x: jnp.ndarray, gamma: jnp.ndarray, beta: jnp.ndarray, amax: jnp.ndarray,
                      scale: jnp.ndarray, scale_inv: jnp.ndarray, zero_centered_gamma: bool,
                      epsilon: float):
    """
    Wrapper for TE layernorm fwd (fp8 out)
    """
    return _layernorm_fwd_fp8_p.bind(x,
                                     gamma,
                                     beta,
                                     amax,
                                     scale,
                                     scale_inv,
                                     zero_centered_gamma=zero_centered_gamma,
                                     epsilon=epsilon)


class RmsNormFwdFp8Primitive(BasePrimitiveLegacy):
    """
    RMS Normalization Forward FP8 Primitive
    """
    name = "te_rmsnorm_forward_fp8"
    multiple_results = True

    @staticmethod
    def abstract(
            x,
            gamma,
            amax,
            scale,
            scale_inv,
            **kwargs    # pylint: disable=unused-argument
    ):
        """
        RMSNorm fwd (fp8 out) abstract
        """
        x_dtype = dtypes.canonicalize_dtype(x.dtype)

        assert x_dtype in [jnp.float32, jnp.float16, jnp.bfloat16]
        assert amax.dtype == jnp.float32
        assert scale.dtype == jnp.float32
        assert scale_inv.dtype == jnp.float32

        out_dtype = jnp.int8
        rsigma_dtype = jnp.float32

        hidden_size = gamma.size
        # In Transformer, batch_size = batch x seqlen
        batch_size = x.size // hidden_size

        return (
            ShapedArray(x.shape, out_dtype, named_shape=x.named_shape),    # output
            ShapedArray((batch_size,), rsigma_dtype, named_shape=x.named_shape),    # rsigma
            ShapedArray((1,), amax.dtype, named_shape=amax.named_shape),    # amax
        )

    @staticmethod
    def lowering(ctx, x, gamma, amax, scale, scale_inv, *, epsilon):
        """
        RMSNorm fwd (fp8 out) lowering rules
        """
        x_aval, gamma_aval, amax_aval, scale_aval, scale_inv_aval = ctx.avals_in

        assert x_aval.dtype in [jnp.float32, jnp.float16, jnp.bfloat16]
        assert amax_aval.dtype == jnp.float32
        assert scale_aval.dtype == jnp.float32
        assert scale_inv_aval.dtype == jnp.float32

        x_type = ir.RankedTensorType(x.type)
        x_shape = x_type.shape
        w_type = ir.RankedTensorType(gamma.type)
        w_shape = w_type.shape

        ir_out_dtype = dtype_to_ir_type(np.dtype(np.int8))
        ir_rsigma_dtype = ir.F32Type.get()
        ir_amax_type = ir.RankedTensorType(amax.type)
        ir_amax_dtype = ir_amax_type.element_type
        ir_amax_shape = ir_amax_type.shape
        ir_scale_shape = ir_amax_shape
        ir_scale_inv_shape = ir_amax_shape

        hidden_size = reduce(operator.mul, w_shape)
        # In Transformer, batch_size = batch x seqlen
        batch_size = reduce(operator.mul, x_shape) // hidden_size

        out_types = [
            ir.RankedTensorType.get(x_shape, ir_out_dtype),
            ir.RankedTensorType.get((batch_size,), ir_rsigma_dtype),
            ir.RankedTensorType.get(ir_amax_shape, ir_amax_dtype),
        ]
        operands = [x, gamma, amax, scale, scale_inv]
        operand_shapes = [x_shape, w_shape, ir_amax_shape, ir_scale_shape, ir_scale_inv_shape]
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


_rmsnorm_fwd_fp8_p = register_primitive_legacy(RmsNormFwdFp8Primitive)


def rmsnorm_fwd_fp8(x: jnp.ndarray, gamma: jnp.ndarray, amax: jnp.ndarray, scale: jnp.ndarray,
                    scale_inv: jnp.ndarray, epsilon: float):
    """
    Wrapper for TE rmsnorm fwd (fp8 out)
    """
    return _rmsnorm_fwd_fp8_p.bind(x, gamma, amax, scale, scale_inv, epsilon=epsilon)


class QuantizePrimitive(BasePrimitiveLegacy):
    """
    Quantize Primitive
    """
    name = "te_quantize"
    multiple_results = True

    @staticmethod
    def abstract(inputs, amax, scale, scale_inv, *, out_dtype):
        """
        te_quantize abstract
        """
        in_dtype = dtypes.canonicalize_dtype(inputs.dtype)
        assert in_dtype in [jnp.float32, jnp.float16, jnp.bfloat16]

        assert isinstance(out_dtype, TEDType)
        out_dtype = te_dtype_to_jax_dtype(out_dtype)

        assert amax.dtype == jnp.float32
        assert scale.dtype == jnp.float32
        assert scale_inv.dtype == jnp.float32

        return (ShapedArray(inputs.shape, out_dtype, named_shape=inputs.named_shape),
                ShapedArray((1,), amax.dtype, named_shape=amax.named_shape))

    @staticmethod
    def lowering(ctx, inputs, amax, scale, scale_inv, *, out_dtype):
        """
        te_quantize lowering rules
        """
        in_aval, amax_aval, scale_aval, scale_inv_aval = ctx.avals_in

        assert in_aval.dtype in [jnp.float32, jnp.float16, jnp.bfloat16]
        assert amax_aval.dtype == jnp.float32
        assert scale_aval.dtype == jnp.float32
        assert scale_inv_aval.dtype == jnp.float32

        ir_in_type = ir.RankedTensorType(inputs.type)
        ir_in_shape = ir_in_type.shape

        ir_out_dtype = te_dtype_to_ir_dtype(out_dtype)
        ir_out_shape = ir_in_shape

        ir_amax_type = ir.RankedTensorType(amax.type)
        ir_amax_shape = ir_amax_type.shape
        ir_amax_dtype = ir_amax_type.element_type

        ir_scale_shape = ir_amax_shape
        ir_scale_inv_shape = ir_amax_shape

        out_types = [
            ir.RankedTensorType.get(ir_out_shape, ir_out_dtype),
            ir.RankedTensorType.get(ir_amax_shape, ir_amax_dtype),
        ]
        operands = [inputs, amax, scale, scale_inv]
        operand_shapes = [ir_in_shape, ir_amax_shape, ir_scale_shape, ir_scale_inv_shape]
        args = CustomCallArgsWrapper(out_types, operands, operand_shapes)

        opaque = transformer_engine_jax.pack_common_descriptor(in_aval.shape,
                                                               jax_dtype_to_te_dtype(in_aval.dtype),
                                                               out_dtype)

        out = custom_caller(QuantizePrimitive.name,
                            args,
                            opaque,
                            False,
                            operand_output_aliases={1: 1})

        return out


_quantize_p = register_primitive_legacy(QuantizePrimitive)


def quantize(inputs: jnp.ndarray, amax: jnp.ndarray, scale: jnp.ndarray, scale_inv: jnp.ndarray,
             out_dtype: TEDType) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    quantize wrapper
    Return FP8 tensor
    """
    return _quantize_p.bind(inputs, amax, scale, scale_inv, out_dtype=out_dtype)


class DequantizePrimitive(BasePrimitiveLegacy):
    """
    Dequantize Primitive
    """
    name = "te_dequantize"
    multiple_results = False

    @staticmethod
    def abstract(inputs, amax, scale, scale_inv, *, fp8_dtype, out_dtype):
        """
        te_dquantize abstract
        """
        in_dtype = dtypes.canonicalize_dtype(inputs.dtype)
        assert in_dtype == jnp.int8
        assert isinstance(fp8_dtype, TEDType)

        assert isinstance(out_dtype, TEDType)
        out_dtype = te_dtype_to_jax_dtype(out_dtype)
        assert out_dtype in [jnp.float32, jnp.float16, jnp.bfloat16]

        assert amax.dtype == jnp.float32
        assert scale.dtype == jnp.float32
        assert scale_inv.dtype == jnp.float32

        return ShapedArray(inputs.shape, out_dtype, named_shape=inputs.named_shape)

    @staticmethod
    def lowering(ctx, inputs, amax, scale, scale_inv, *, fp8_dtype, out_dtype):
        """
        te_dquantize lowering rules
        """
        in_aval, amax_aval, scale_aval, scale_inv_aval = ctx.avals_in

        assert in_aval.dtype == jnp.int8
        assert amax_aval.dtype == jnp.float32
        assert scale_aval.dtype == jnp.float32
        assert scale_inv_aval.dtype == jnp.float32

        ir_in_type = ir.RankedTensorType(inputs.type)
        ir_in_shape = ir_in_type.shape

        ir_out_dtype = te_dtype_to_ir_dtype(out_dtype)
        ir_out_shape = ir_in_shape

        ir_amax_type = ir.RankedTensorType(amax.type)
        ir_amax_shape = ir_amax_type.shape

        ir_scale_shape = ir_amax_shape
        ir_scale_inv_shape = ir_amax_shape

        out_types = [ir.RankedTensorType.get(ir_out_shape, ir_out_dtype)]
        operands = [inputs, amax, scale, scale_inv]
        operand_shapes = [ir_in_shape, ir_amax_shape, ir_scale_shape, ir_scale_inv_shape]
        args = CustomCallArgsWrapper(out_types, operands, operand_shapes)

        opaque = transformer_engine_jax.pack_common_descriptor(in_aval.shape, fp8_dtype, out_dtype)

        out = custom_caller(DequantizePrimitive.name, args, opaque, False)

        return [out]


_dequantize_p = register_primitive_legacy(DequantizePrimitive)


def dequantize(inputs: jnp.ndarray, amax: jnp.ndarray, scale: jnp.ndarray, scale_inv: jnp.ndarray,
               fp8_dtype: TEDType, out_dtype: TEDType) -> jnp.ndarray:
    """
    dequantize wrapper
    Return FP16/BF16/FP32 tensor
    """
    return _dequantize_p.bind(inputs,
                              amax,
                              scale,
                              scale_inv,
                              fp8_dtype=fp8_dtype,
                              out_dtype=out_dtype)


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


class SelfFusedAttnFwdPrimitive(BasePrimitiveLegacy):
    """
    Self Fused Attention Forward Primitive
    """
    name = "te_self_fused_attn_forward"
    multiple_results = True

    @staticmethod
    def abstract(
            qkv,
            bias,
            cu_seqlen,    # pylint: disable=unused-argument
            seed,    # pylint: disable=unused-argument
            *,
            attn_bias_type,    # pylint: disable=unused-argument
            attn_mask_type,    # pylint: disable=unused-argument
            scaling_factor,    # pylint: disable=unused-argument
            dropout_probability,    # pylint: disable=unused-argument
            is_training    # pylint: disable=unused-argument
    ):
        """
        Self fused attention fwd abstract
        """
        qkv_dtype = dtypes.canonicalize_dtype(qkv.dtype)
        batch, max_seqlen, nqkv, num_head, head_dim = qkv.shape
        assert nqkv == 3
        assert qkv.dtype == bias.dtype

        output_shape = (batch, max_seqlen, num_head, head_dim)
        output_dtype = qkv_dtype

        backend = FusedAttnHelper(qkv_dtype, qkv_dtype, NVTE_QKV_Layout.NVTE_BS3HD, attn_bias_type,
                                  attn_mask_type, dropout_probability, max_seqlen, max_seqlen,
                                  head_dim).get_fused_attn_backend()

        if backend == NVTE_Fused_Attn_Backend.NVTE_F16_max512_seqlen:
            softmax_aux_shape = (batch, num_head, max_seqlen, max_seqlen)
            softmax_dtype = qkv_dtype
        elif backend == NVTE_Fused_Attn_Backend.NVTE_F16_arbitrary_seqlen:
            softmax_aux_shape = (batch, num_head, max_seqlen, 1)
            softmax_dtype = dtypes.canonicalize_dtype(jnp.float32)
        else:
            raise ValueError(f'Not supported {backend=}')

        checker = _FusedAttnRNGStateChecker()
        seed_dtype = dtypes.canonicalize_dtype(seed.dtype)
        assert seed_dtype == checker.rng_state_dtype
        rng_state_shape = (checker.rng_state_size,)
        rng_state_dtype = seed_dtype

        return (
            ShapedArray(output_shape, output_dtype, named_shape=qkv.named_shape),    # output
            ShapedArray(softmax_aux_shape, softmax_dtype,
                        named_shape=qkv.named_shape),    # softmax_aux
            ShapedArray(rng_state_shape, rng_state_dtype,
                        named_shape=seed.named_shape),    # rng_state
        )

    @staticmethod
    def lowering(ctx, qkv, bias, cu_seqlen, seed, *, attn_bias_type, attn_mask_type, scaling_factor,
                 dropout_probability, is_training):
        """
        Self fused attention fwd lowering rules
        """
        qkv_aval, _, _, _ = ctx.avals_in

        batch, max_seqlen, _, num_head, head_dim = qkv_aval.shape

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


_self_fused_attn_fwd_p = register_primitive_legacy(SelfFusedAttnFwdPrimitive)


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
    return _self_fused_attn_fwd_p.bind(qkv,
                                       bias,
                                       cu_seqlen,
                                       seed,
                                       attn_bias_type=attn_bias_type,
                                       attn_mask_type=attn_mask_type,
                                       scaling_factor=scaling_factor,
                                       dropout_probability=dropout_probability,
                                       is_training=is_training)


class SelfFusedAttnBwdPrimitive(BasePrimitiveLegacy):
    """
    Self Fused Attention Backward Primitive
    """
    name = "te_self_fused_attn_backward"
    multiple_results = True

    @staticmethod
    def abstract(
            qkv,
            softmax_aux,    # pylint: disable=unused-argument
            rng_state,    # pylint: disable=unused-argument
            output,    # pylint: disable=unused-argument
            doutput,
            cu_seqlen,    # pylint: disable=unused-argument
            *,
            attn_bias_type,    # pylint: disable=unused-argument
            attn_mask_type,    # pylint: disable=unused-argument
            scaling_factor,    # pylint: disable=unused-argument
            dropout_probability,    # pylint: disable=unused-argument
            is_training    # pylint: disable=unused-argument
    ):
        """
        Self fused attention bwd abstract
        """
        qkv_dtype = dtypes.canonicalize_dtype(qkv.dtype)
        assert qkv.dtype == doutput.dtype

        _, seqlen, _, num_head, _ = qkv.shape

        if attn_bias_type == NVTE_Bias_Type.NVTE_NO_BIAS:
            bias_shape = (0,)
        else:
            bias_shape = (1, num_head, seqlen, seqlen)
        bias_dtype = qkv_dtype

        return (
            ShapedArray(qkv.shape, qkv_dtype, named_shape=qkv.named_shape),    # dqkv
            ShapedArray(bias_shape, bias_dtype, named_shape=qkv.named_shape))

    @staticmethod
    def lowering(ctx, qkv, softmax_aux, rng_state, output, doutput, cu_seqlen, *, attn_bias_type,
                 attn_mask_type, scaling_factor, dropout_probability, is_training):
        """
        Self fused attention bwd lowering rules
        """
        qkv_aval, _, _, _, _, _ = ctx.avals_in

        batch, max_seqlen, _, num_head, head_dim = qkv_aval.shape

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


_self_fused_attn_bwd_p = register_primitive_legacy(SelfFusedAttnBwdPrimitive)


def self_fused_attn_bwd(qkv: jnp.ndarray, softmax_aux: jnp.ndarray, rng_state: jnp.ndarray,
                        output: jnp.ndarray, doutput: jnp.ndarray, cu_seqlen: jnp.ndarray,
                        attn_bias_type: NVTE_Bias_Type, attn_mask_type: NVTE_Mask_Type,
                        scaling_factor: float, dropout_probability: float, is_training: bool):
    """
    Wrapper for TE self fused attention bwd
    Return the gradients of self fused attention with packed qkv input
    """
    return _self_fused_attn_bwd_p.bind(qkv,
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


class CrossFusedAttnFwdPrimitive(BasePrimitiveLegacy):
    """
    Cross Fused Attention Forward Primitive
    """
    name = "te_cross_fused_attn_forward"
    multiple_results = True

    @staticmethod
    def abstract(
            q,
            kv,
            q_cu_seqlen,
            kv_cu_seqlen,
            seed,    # pylint: disable=unused-argument
            *,
            attn_bias_type,    # pylint: disable=unused-argument
            attn_mask_type,    # pylint: disable=unused-argument
            scaling_factor,    # pylint: disable=unused-argument
            dropout_probability,    # pylint: disable=unused-argument
            is_training    # pylint: disable=unused-argument
    ):
        """
        Cross fused attention fwd abstract
        """
        q_dtype = dtypes.canonicalize_dtype(q.dtype)
        batch_q, q_max_seqlen, num_head_q, head_dim_q = q.shape
        kv_dtype = dtypes.canonicalize_dtype(kv.dtype)
        batch_kv, kv_max_seqlen, nkv, num_head_kv, head_dim_kv = kv.shape

        assert q_dtype == kv_dtype
        assert batch_q == batch_kv
        assert num_head_q == num_head_kv
        assert head_dim_q == head_dim_kv
        assert nkv == 2
        assert q_cu_seqlen.dtype == kv_cu_seqlen.dtype

        output_shape = q.shape
        output_dtype = q_dtype
        softmax_aux_shape = (batch_q, num_head_q, q_max_seqlen, kv_max_seqlen)
        softmax_aux_dtype = q_dtype

        return (
            ShapedArray(output_shape, output_dtype, named_shape=q.named_shape),    # output
            ShapedArray(softmax_aux_shape, softmax_aux_dtype,
                        named_shape=q.named_shape),    # softmax_aux
        )

    @staticmethod
    def lowering(ctx, q, kv, q_cu_seqlen, kv_cu_seqlen, seed, *, attn_bias_type, attn_mask_type,
                 scaling_factor, dropout_probability, is_training):
        """
        Cross fused attention fwd lowering rules
        """
        q_aval, kv_aval, _, _, _ = ctx.avals_in
        assert q_aval.dtype == kv_aval.dtype

        batch, q_max_seqlen, num_head, head_dim = q_aval.shape
        kv_max_seqlen = kv_aval.shape[1]

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


_cross_fused_attn_fwd_p = register_primitive_legacy(CrossFusedAttnFwdPrimitive)


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

    return _cross_fused_attn_fwd_p.bind(q,
                                        kv,
                                        q_cu_seqlen,
                                        kv_cu_seqlen,
                                        seed,
                                        attn_bias_type=attn_bias_type,
                                        attn_mask_type=attn_mask_type,
                                        scaling_factor=scaling_factor,
                                        dropout_probability=dropout_probability,
                                        is_training=is_training)


class CrossFusedAttnBwdPrimitive(BasePrimitiveLegacy):
    """
    Cross Fused Attention Backward Primitive
    """
    name = "te_cross_fused_attn_backward"
    multiple_results = True

    @staticmethod
    def abstract(
            q,
            kv,
            softmax_aux,
            doutput,
            q_cu_seqlen,
            kv_cu_seqlen,
            *,
            attn_bias_type,    # pylint: disable=unused-argument
            attn_mask_type,    # pylint: disable=unused-argument
            scaling_factor,    # pylint: disable=unused-argument
            dropout_probability,    # pylint: disable=unused-argument
            is_training    # pylint: disable=unused-argument
    ):
        """
        Cross fused attention bwd abstract
        """
        q_dtype = dtypes.canonicalize_dtype(q.dtype)
        kv_dtype = dtypes.canonicalize_dtype(kv.dtype)
        softmax_aux_dtype = dtypes.canonicalize_dtype(softmax_aux.dtype)
        doutput_dtype = dtypes.canonicalize_dtype(doutput.dtype)
        assert q_dtype == kv_dtype == softmax_aux_dtype == doutput_dtype
        assert q_cu_seqlen.dtype == kv_cu_seqlen.dtype

        return (
            ShapedArray(q.shape, q_dtype, named_shape=q.named_shape),    # dq
            ShapedArray(kv.shape, kv_dtype, named_shape=kv.named_shape),    # dkv
        )

    @staticmethod
    def lowering(ctx, q, kv, softmax_aux, doutput, q_cu_seqlen, kv_cu_seqlen, *, attn_bias_type,
                 attn_mask_type, scaling_factor, dropout_probability, is_training):
        """
        Cross fused attention bwd lowering rules
        """
        q_aval, kv_aval, _, _, _, _ = ctx.avals_in
        assert q_aval.dtype == kv_aval.dtype

        batch, q_max_seqlen, num_head, head_dim = q_aval.shape
        kv_max_seqlen = kv_aval.shape[1]

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


_cross_fused_attn_bwd_p = register_primitive_legacy(CrossFusedAttnBwdPrimitive)


def cross_fused_attn_bwd(q: jnp.ndarray, kv: jnp.ndarray, softmax_aux: jnp.ndarray,
                         doutput: jnp.ndarray, q_cu_seqlen: jnp.ndarray, kv_cu_seqlen: jnp.ndarray,
                         attn_bias_type: NVTE_Bias_Type, attn_mask_type: NVTE_Mask_Type,
                         scaling_factor: float, dropout_probability: float, is_training: bool):
    """
    Wrapper for TE cross fused attention bwd
    Return the gradients of cross fused attention with packed kv input
    """
    return _cross_fused_attn_bwd_p.bind(q,
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
