# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""JAX/TE base custom ops"""
from abc import ABCMeta, abstractmethod
from functools import partial

from jax import core
from jax.interpreters import xla, mlir
from jax.experimental.custom_partitioning import custom_partitioning
from jax._src.interpreters import batching
from jax._src import dispatch


class BasePrimitive(metaclass=ABCMeta):
    """
    jax primitive
    """

    @staticmethod
    @abstractmethod
    def abstract():
        """
        to describe computing graph
        """
        return NotImplemented

    @classmethod
    def outer_abstract(cls, *args, **kwargs):
        """
        optional abstract wrapper to eliminate workspace tensors
        """
        return cls.abstract(*args, **kwargs)

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
    dispatch.prim_requires_devices_during_lowering.add(inner_p)
    inner_p.multiple_results = cls.multiple_results
    inner_p.def_impl(partial(xla.apply_primitive, inner_p))
    inner_p.def_abstract_eval(cls.abstract)
    mlir.register_lowering(inner_p, cls.lowering, platform="cuda")
    cls.inner_primitive = inner_p

    outer_p = core.Primitive(name_of_wrapper_p())
    dispatch.prim_requires_devices_during_lowering.add(outer_p)
    outer_p.multiple_results = cls.multiple_results
    outer_p.def_impl(cls.impl)
    outer_p.def_abstract_eval(cls.outer_abstract)
    batching.primitive_batchers[outer_p] = cls.batcher
    outer_p_lower = custom_partitioning(cls.impl, static_argnums=cls.impl_static_args)
    outer_p_lower.def_partition(
        infer_sharding_from_operands=cls.infer_sharding_from_operands, partition=cls.partition
    )
    mlir.register_lowering(
        outer_p, mlir.lower_fun(outer_p_lower, multiple_results=cls.multiple_results)
    )
    cls.outer_primitive = outer_p
