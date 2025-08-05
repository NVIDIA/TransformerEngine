# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""JAX/TE base custom ops"""
import os
import re
import warnings
from abc import ABCMeta, abstractmethod
from functools import partial
from packaging import version

from jax.extend import core
from jax.interpreters import xla, mlir
from jax.experimental.custom_partitioning import custom_partitioning
from jax._src.interpreters import batching
from jax._src import dispatch

import jax
import transformer_engine_jax

if version.parse(jax.__version__) >= version.parse("0.5.0"):
    from jax import ffi  # pylint: disable=ungrouped-imports
else:
    from jax.extend import ffi  # pylint: disable=ungrouped-imports


class BasePrimitive(metaclass=ABCMeta):
    """
    jax primitive
    """

    name = None

    _is_enabled = True

    # Default list of primitives to disable for all recipes
    _default_disable_names = ["GemmPrimitive", "NormFwdPrimitive", "NormBwdPrimitive"]

    @classmethod
    def enabled(cls):
        """
        Determines if a custom call is enabled based on a state variable and environment variables.
        Checks `NVTE_JAX_CUSTOM_CALLS` (key/value format) first, then falls back to the deprecated `NVTE_JAX_CUSTOM_CALLS_RE` (regex pattern),
        and finally to the internal state `_is_enabled` if neither is set.

        Environment Variables:
            1. `NVTE_JAX_CUSTOM_CALLS`: Preferred key/value format to enable/disable specific primitives or a single value 'true' or 'false' to enable/disable all primitives.
               - Example 1 (global enable): 'true' enables all primitives.
               - Example 2 (global disable): 'false' disables all primitives.
               - Example 3 (specific settings): 'DBiasQuantizePrimitive=false,GemmPrimitive=true' disables DBiasQuantizePrimitive and enables GemmPrimitive, leaving others at their default state.
                 Note that the default state is set at class level based on _default_disable_names.
            2. `NVTE_JAX_CUSTOM_CALLS_RE`: Deprecated regex pattern to match primitive names.
               - Example: 'DBiasQuantizePrimitive' or '^(?!DBiasQuantizePrimitive$).+$' to enable/disable DBiasQuantizePrimitive.
               - A deprecation warning is raised if used; it will be removed in future releases.

        Behavior:
            1. Checks if `NVTE_JAX_CUSTOM_CALLS` is set and parses key/value pairs or single true/false value.
            2. If not set, checks `NVTE_JAX_CUSTOM_CALLS_RE` (with deprecation warning) for regex matching.
            3. If neither is set, falls back to the internal state `_is_enabled`.
        """

        # Check new key/value environment variable first
        custom_calls_str = os.getenv("NVTE_JAX_CUSTOM_CALLS")
        if custom_calls_str is not None:
            custom_calls_str = custom_calls_str.strip()
            if custom_calls_str.lower() == "true":
                return True
            if custom_calls_str.lower() == "false":
                return False

            # Parse key=value pairs
            settings = {}
            for pair in custom_calls_str.split(","):
                pair = pair.strip()
                if "=" in pair:
                    key, value = pair.split("=", 1)
                    key = key.strip()
                    value = value.strip().lower()
                    settings[key] = value == "true"
            if cls.__name__ in settings:
                return settings[cls.__name__]

        # Check old regex environment variable (deprecated)
        pattern_str = os.getenv("NVTE_JAX_CUSTOM_CALLS_RE")
        if pattern_str is not None:
            warnings.warn(
                "NVTE_JAX_CUSTOM_CALLS_RE is deprecated and will be removed in future releases. Use"
                " NVTE_JAX_CUSTOM_CALLS with key=value format instead (e.g.,"
                " 'DBiasQuantizePrimitive=false').",
                DeprecationWarning,
            )
            pattern = re.compile(pattern_str)
            env_enabled = pattern.fullmatch(cls.__name__) is not None
            return env_enabled

        # If no environment variable is set, fall back to the internal state
        return cls._is_enabled

    @classmethod
    def set_enabled(cls, enabled: bool):
        """
        Sets the enabled state for this primitive.
        """
        cls._is_enabled = enabled

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

    @staticmethod
    @abstractmethod
    def shardy_sharding_rule(*args):
        """
        Returns the sharding rule for this primitive.
        """
        del args
        return "... -> ..."


# Registry to store all registered primitive classes
_primitive_registry = {}


def register_primitive(cls):
    """
    Register a JAX primitive and add it to the internal registry.
    """
    _primitive_registry[cls.__name__] = cls

    # Set default disabled state at class level based on _default_disable_names
    if cls.__name__ in BasePrimitive._default_disable_names:
        cls.set_enabled(False)

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
        infer_sharding_from_operands=cls.infer_sharding_from_operands,
        partition=cls.partition,
        sharding_rule=cls.shardy_sharding_rule,
    )
    mlir.register_lowering(
        outer_p, mlir.lower_fun(outer_p_lower, multiple_results=cls.multiple_results)
    )
    cls.outer_primitive = outer_p


for _name, _value in transformer_engine_jax.registrations().items():
    ffi.register_ffi_target(_name, _value, platform="CUDA")


def manage_primitives(enable_names=None, disable_names=None, disable_all_first=False):
    """
    Helper function to manage primitive states by name without modifying environment variables.
    Allows enabling specific primitives, disabling specific primitives, or disabling all primitives.
    This helper is used in the QuantizeConfig.initialize() methods.

    Args:
        enable_names: List of strings, each representing the name of a primitive class to enable. Defaults to None.
        disable_names: List of strings, each representing the name of a primitive class to disable. Defaults to None.
        disable_all_first: Boolean, if True, disables all primitives before applying enable/disable lists. Defaults to False.

    Note:
        1. If `disable_all_first` is True, all primitives are disabled first, then `enable_names` is applied.
        2. Conflicts (a primitive in both enable and disable lists) are resolved by applying disable last.
    """

    enable_set = set(enable_names or [])
    disable_set = set(disable_names or [])

    if disable_all_first:
        for name, cls in _primitive_registry.items():
            if (
                isinstance(cls, type)
                and issubclass(cls, BasePrimitive)
                and cls is not BasePrimitive
            ):
                cls.set_enabled(False)

    # Apply enables
    for name in enable_set:
        cls = _primitive_registry.get(name)
        if cls and isinstance(cls, type) and issubclass(cls, BasePrimitive):
            cls.set_enabled(True)
        else:
            raise ValueError(f"Primitive not found in registry: {name}")

    # Apply disables (overrides enables if there's a conflict)
    for name in disable_set:
        cls = _primitive_registry.get(name)
        if cls and isinstance(cls, type) and issubclass(cls, BasePrimitive):
            cls.set_enabled(False)
        else:
            raise ValueError(f"Primitive not found in registry: {name}")
