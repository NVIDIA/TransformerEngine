# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""
Config module for quantization metadata management

This module provides configuration and helper functions for managing quantization metadata
in JAX, including support for different scaling modes and datatypes.
"""
from contextlib import contextmanager
from enum import Enum
from typing import Optional, Tuple, Dict, Union

import jax
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict

from transformer_engine_jax import DType
from transformer_engine_jax import get_cublasLt_version
from transformer_engine_jax import (
    get_cuda_version,
    get_device_compute_capability,
)
from transformer_engine.common import recipe
from transformer_engine.jax.sharding import global_shard_guard, MeshResource

from .scaling_modes import ScalingMode
from .. import cpp_extensions as tex

__all__ = [
    "QuantizeConfig",
    "fp8_autocast",
    "is_fp8_available",
    "update_collections",
    "get_delayed_scaling",
    "NVTE_FP8_COLLECTION_NAME",
]

_is_fp8_available = None
_reason_for_no_fp8 = ""
Collection = Union[Dict, FrozenDict]


def _check_delayed_scaling_fp8_support(gpu_arch) -> Tuple[bool, str]:
    """Check if delayed scaling FP8 is supported on the given GPU architecture.

    Args:
        gpu_arch: The GPU architecture version

    Returns:
        A tuple of (bool, str) indicating support and any error message
    """
    if gpu_arch >= 90:  # hopper and above
        return True, ""
    if gpu_arch < 89:  # pre-ada
        return False, "Device compute capability 8.9 or higher required for FP8 execution."
    if get_cublasLt_version() < 120103:
        return False, "CublasLt version 12.1.3.x or higher required for FP8 execution on Ada."
    if get_cuda_version() < 12010:
        return False, "Cuda version 12.1 or higher required for FP8 execution on Ada."
    return True, ""


def _check_block_scaling_fp8_support(gpu_arch) -> Tuple[bool, str]:
    """Check if block scaling FP8 is supported on the given GPU architecture.

    Args:
        gpu_arch: The GPU architecture version

    Returns:
        A tuple of (bool, str) indicating support and any error message
    """
    if gpu_arch >= 100:  # blackwell and above
        return True, ""
    if gpu_arch < 99:  # pre-blackwell
        return False, "Device compute capability 9.9 or higher required for MXFP8 execution."
    if get_cublasLt_version() < 120800:
        return False, "CublasLt version 12.8.0 or higher required for MXFP8 execution."
    if get_cuda_version() < 12010:
        return False, "Cuda version 12.8 or higher required for MXFP8 execution."
    if not tex.jax_version_meet_requirement("0.5.3"):
        return False, "Jax version 0.5.3 or higher required for MXFP8 execution."
    return True, ""


def _check_fp8_support(scaling_mode, gpu_id) -> Tuple[bool, str]:
    """Check if FP8 is supported for the given scaling mode and GPU.

    Args:
        scaling_mode: The scaling mode to check support for
        gpu_id: The ID of the GPU to check

    Returns:
        A tuple of (bool, str) indicating support and any error message
    """
    gpu_arch = get_device_compute_capability(gpu_id)
    if scaling_mode.is_tensor_scaling():
        return _check_delayed_scaling_fp8_support(gpu_arch)
    if scaling_mode == ScalingMode.MXFP8_1D_SCALING:
        return _check_block_scaling_fp8_support(gpu_arch)
    return (False, "Unsupported scaling_mode!")


def is_fp8_available(
    scaling_mode=ScalingMode.DELAYED_TENSOR_SCALING,
    gpu_id=None,
) -> Tuple[bool, str]:
    """Check if FP8 is available for the given scaling mode and GPU.

    Args:
        scaling_mode: The scaling mode to check availability for (default: DELAYED_TENSOR_SCALING)
        gpu_id: Optional GPU ID to check specific device (default: None)

    Returns:
        A tuple of (bool, str) indicating availability and any error message
    """
    if gpu_id is not None:
        return _check_fp8_support(scaling_mode, gpu_id)

    global _is_fp8_available, _reason_for_no_fp8
    if _is_fp8_available is None:
        _is_fp8_available = {}
        _reason_for_no_fp8 = {}

    if scaling_mode not in _is_fp8_available:
        _is_fp8_available[scaling_mode] = True
        _reason_for_no_fp8[scaling_mode] = ""
        # JAX doesn't provide the local GPU id.
        for local_gpu_id in range(len(jax.local_devices())):
            ret, msg = _check_fp8_support(scaling_mode, local_gpu_id)
            if ret is False:
                _is_fp8_available[scaling_mode] = ret
                _reason_for_no_fp8[scaling_mode] = msg
                return ret, msg

    return _is_fp8_available[scaling_mode], _reason_for_no_fp8[scaling_mode]


def _format2dtypes(format_: recipe.Format):
    """Convert recipe.Format.dtype to corresponding JAX dtypes.

    Args:
        format_: The FP8 format to convert

    Returns:
        A tuple of (forward_dtype, backward_dtype) for the given format
    """
    if format_ == recipe.Format.E4M3:
        return jnp.float8_e4m3fn, jnp.float8_e4m3fn
    if format_ == recipe.Format.E5M2:
        return jnp.float8_e5m2, jnp.float8_e5m2
    if format_ == recipe.Format.HYBRID:
        return jnp.float8_e4m3fn, jnp.float8_e5m2
    return jnp.bfloat16, jnp.bfloat16


class AmaxComputeAlgo(Enum):
    """Enumeration for AMAX computation algorithms.

    Attributes:
        MAX: Use maximum value for AMAX computation
        MOST_RECENT: Use most recent value for AMAX computation
    """

    MAX = "max"
    MOST_RECENT = "most_recent"


def _get_scaling_mode(fp8_recipe: recipe.Recipe) -> ScalingMode:
    """Convert recipe.Recipe to ScalingMode.

    Args:
        fp8_recipe: The FP8 recipe to convert

    Returns:
        The corresponding ScalingMode

    Raises:
        ValueError: If the recipe type is not supported
    """
    if isinstance(fp8_recipe, recipe.DelayedScaling):
        return ScalingMode.DELAYED_TENSOR_SCALING
    if isinstance(fp8_recipe, recipe.MXFP8BlockScaling):
        return ScalingMode.MXFP8_1D_SCALING
    if isinstance(fp8_recipe, recipe.Float8CurrentScaling):
        return ScalingMode.CURRENT_TENSOR_SCALING
    raise ValueError("Invalid fp8_recipe!")


class QuantizeConfig:
    """Configuration class for quantization settings.

    This class manages global quantization settings including FP8 formats,
    scaling modes, and accumulation settings.

    Attributes:
        INITIALIZED: Whether the config has been initialized
        MARGIN: Margin value for quantization
        COLLECTION_NAME: Name of the collection for quantization metadata
        FP8_FORMAT: FP8 format to use
        FWD_DTYPE: Forward pass data type
        BWD_DTYPE: Backward pass data type
        FP8_2X_ACC_FPROP: Whether to use 2x accumulation for forward pass
        FP8_2X_ACC_DGRAD: Whether to use 2x accumulation for data gradients
        FP8_2X_ACC_WGRAD: Whether to use 2x accumulation for weight gradients
        IF_QUANTIZE_2X: Whether 2x quantization is enabled
        SCALING_MODE: Scaling mode
        AMAX_HISTORY_LEN: Length of AMAX history for delayed scaling
        AMAX_COMPUTE_ALGO: Algorithm for AMAX computation
    """

    INITIALIZED = False
    MARGIN: float = 0.0
    COLLECTION_NAME: str = "fp8_metas"
    FP8_FORMAT: recipe.Format = recipe.Format.HYBRID
    FWD_DTYPE: DType = _format2dtypes(recipe.Format.HYBRID)[0]
    BWD_DTYPE: DType = _format2dtypes(recipe.Format.HYBRID)[1]
    FP8_2X_ACC_FPROP: bool = False
    FP8_2X_ACC_DGRAD: bool = False
    FP8_2X_ACC_WGRAD: bool = False
    IF_QUANTIZE_2X: bool = False
    SCALING_MODE: ScalingMode = ScalingMode.NO_SCALING

    # DelayedScaling
    AMAX_HISTORY_LEN: int = 1024
    AMAX_COMPUTE_ALGO: AmaxComputeAlgo = AmaxComputeAlgo.MAX

    @staticmethod
    def is_fp8_enabled():
        """Check if FP8 quantization is enabled.

        Returns:
            bool: True if quantization is enabled, False otherwise
        """
        return QuantizeConfig.INITIALIZED

    @classmethod
    def initialize(cls, fp8_recipe: recipe.Recipe) -> None:
        """Initialize the quantization configuration.

        Args:
            fp8_recipe: The FP8 recipe to use for initialization
        """
        cls.INITIALIZED = True
        cls.MARGIN = fp8_recipe.margin if "margin" in dir(fp8_recipe) else 0.0
        cls.FP8_FORMAT = fp8_recipe.fp8_format
        cls.FWD_DTYPE, cls.BWD_DTYPE = _format2dtypes(cls.FP8_FORMAT)
        cls.SCALING_MODE = _get_scaling_mode(fp8_recipe)
        cls.IF_QUANTIZE_2X = True

    @classmethod
    def finalize(cls) -> None:
        """Reset the quantization configuration to default values."""
        cls.INITIALIZED = False
        cls.MARGIN = 0.0
        cls.FP8_FORMAT = recipe.Format.HYBRID
        cls.FWD_DTYPE, cls.BWD_DTYPE = _format2dtypes(cls.FP8_FORMAT)
        cls.SCALING_MODE = ScalingMode.NO_SCALING
        cls.FP8_2X_ACC_FPROP = False
        cls.FP8_2X_ACC_DGRAD = False
        cls.FP8_2X_ACC_WGRAD = False
        cls.SCALING_MODE = ScalingMode.NO_SCALING
        cls.IF_QUANTIZE_2X = False
        # DelayedScaling
        cls.AMAX_HISTORY_LEN = 1024
        cls.AMAX_COMPUTE_ALGO = AmaxComputeAlgo.MAX


class DelayedScalingQuantizeConfig:
    """Configuration class for delayed scaling FP8 recipe.

    This class provides specific initialization and finalization for delayed scaling
    FP8 quantization mode.
    """

    @staticmethod
    def initialize(fp8_recipe: recipe.Recipe) -> None:
        """Initialize delayed scaling FP8 configuration.

        Args:
            fp8_recipe: The FP8 recipe to use for initialization

        Raises:
            AssertionError: If recipe parameters are not supported
        """
        assert fp8_recipe.amax_compute_algo in [
            "max",
            "most_recent",
        ], "DelayedScaling amax_compute_algo only supports max and most_recent with TE/JAX."
        assert (
            fp8_recipe.scaling_factor_compute_algo is None
        ), "DelayedScaling scaling_factor_compute_algo isn't supported by TE/JAX."
        assert fp8_recipe.reduce_amax, "DelayedScaling reduce_amax should be enabled for TE/JAX."

        cls = QuantizeConfig
        cls.initialize(fp8_recipe)

        cls.AMAX_HISTORY_LEN = fp8_recipe.amax_history_len
        string_to_amax_compute_algo = {
            "max": AmaxComputeAlgo.MAX,
            "most_recent": AmaxComputeAlgo.MOST_RECENT,
        }
        cls.AMAX_COMPUTE_ALGO = string_to_amax_compute_algo[fp8_recipe.amax_compute_algo]

        cls.FP8_2X_ACC_DGRAD = True
        cls.FP8_2X_ACC_WGRAD = True

    @staticmethod
    def finalize() -> None:
        """Reset the delayed scaling configuration."""
        QuantizeConfig.finalize()


class CurrentScalingQuantizeConfig:
    """Configuration class for current scaling FP8 recipe.

    This class provides specific initialization and finalization for current scaling
    FP8 quantization mode.
    """

    @staticmethod
    def initialize(fp8_recipe: recipe.Recipe) -> None:
        """Initialize current scaling FP8 configuration.

        Args:
            fp8_recipe: The FP8 recipe to use for initialization
        """
        cls = QuantizeConfig
        cls.initialize(fp8_recipe)
        cls.AMAX_HISTORY_LEN = 0

    @staticmethod
    def finalize() -> None:
        """Reset the current scaling configuration."""
        QuantizeConfig.finalize()


class BlockScalingQuantizeConfig:
    """Configuration class for block scaling FP8 recipe.

    This class provides specific initialization and finalization for block scaling
    FP8 quantization mode.
    """

    @staticmethod
    def initialize(fp8_recipe: recipe.Recipe) -> None:
        """Initialize block scaling FP8 configuration.

        Args:
            fp8_recipe: The FP8 recipe to use for initialization
        """
        cls = QuantizeConfig
        cls.initialize(fp8_recipe)
        cls.AMAX_HISTORY_LEN = 0

    @staticmethod
    def finalize() -> None:
        """Reset the block scaling configuration."""
        QuantizeConfig.finalize()


@contextmanager
def fp8_autocast(
    enabled: bool = False,
    fp8_recipe: Optional[recipe.Recipe] = None,
    mesh_resource: Optional[MeshResource] = None,
) -> None:
    r"""Context manager for FP8 automatic mixed precision.

    This context manager enables FP8 quantization for the duration of its context.
        .. code-block:: python

        mesh_shape = (4, 2)
        dp_mesh_axis_name = 'data_parallel'
        tp_mesh_axis_name = 'tensor_parallel'
        devices = np.asarray(jax.devices()).reshape(*mesh_shape)

        with maps.Mesh(devices, (dp_mesh_axis_name, tp_mesh_axis_name)):
            mesh_resource=MeshResource(dp_mesh_axis_name, tp_mesh_axis_name)

            with fp8_autocast(enabled=True, mesh_resource=mesh_resource):
                rules = extend_logical_axis_rules(tuple())
                transformer = TransformerLayer()

                with partitioning.axis_rules(rules):
                    pjit(transformer.init, ...)(...)

    .. note::
        We only support :attr:`margin`, :attr:`fp8_format`, :attr:`amax_history_len`,
        and :attr:`amax_compute_algo` (with value 'max' and 'most_recent') in
        recipe.DelayedScaling currently. Other parameters in recipe.DelayedScaling
        will trigger an assertion.

    Parameters
    ----------
    enabled: bool, default = False
        Whether or not to enable fp8
    fp8_recipe: recipe.DelayedScaling, default = None
        Recipe used for FP8 training.
    mesh_resource: MeshResource, default = None
        Specify the mesh axes for data and tensor parallelism to shard along.
        If set to None, then no data or tensor parallelism will be used.

    """
    if fp8_recipe is None:
        fp8_recipe = recipe.DelayedScaling()

    if mesh_resource is None:
        mesh_resource = MeshResource()

    Config = DelayedScalingQuantizeConfig
    if isinstance(fp8_recipe, recipe.MXFP8BlockScaling):
        Config = BlockScalingQuantizeConfig
    if isinstance(fp8_recipe, recipe.Float8CurrentScaling):
        Config = CurrentScalingQuantizeConfig

    try:
        with global_shard_guard(mesh_resource):
            if enabled:
                fp8_available, reason_for_no_fp8 = is_fp8_available(_get_scaling_mode(fp8_recipe))
                assert fp8_available, reason_for_no_fp8

                Config.initialize(fp8_recipe)
            yield
    finally:
        Config.finalize()


def get_delayed_scaling():
    r"""
    Obtain an instance of  DelayedScaling which is set via fp8_autocast.

    .. note::
        We only store :attr:`margin`, :attr:`fp8_format`, :attr:`amax_history_len`
        , and :attr:`amax_compute_algo` via fp8_autocast. Other parameters in
        recipe.DelayedScaling would be returned as the default values.

    Returns
    -------
    delay_scaling : DelayedScaling
        an instance of  DelayedScaling which is set via fp8_autocast.
    """
    amax_compute_algo = (
        "max" if QuantizeConfig.AMAX_COMPUTE_ALGO is AmaxComputeAlgo.MAX else "most_recent"
    )
    return recipe.DelayedScaling(
        margin=int(QuantizeConfig.MARGIN),
        fp8_format=QuantizeConfig.FP8_FORMAT,
        amax_history_len=QuantizeConfig.AMAX_HISTORY_LEN,
        amax_compute_algo=amax_compute_algo,
    )


def update_collections(new: Collection, original: Collection) -> Collection:
    r"""Update collections with new values while preserving original structure.

    Args:
        new: New collection of values to add/update
        original: Original collection to update

    Returns:
        Updated collection with new values merged with original

    Raises:
        AssertionError: If either collection is not a dict or FrozenDict
    """
    assert isinstance(original, (dict, FrozenDict))
    assert isinstance(new, (dict, FrozenDict))
    frozen_original = FrozenDict(original) if not isinstance(original, FrozenDict) else original
    for key in new:
        if key in frozen_original:
            frozen_original, _ = frozen_original.pop(key)
    new_coll = FrozenDict({**new, **frozen_original})
    if not isinstance(original, FrozenDict):
        new_coll = new_coll.unfreeze()
    return new_coll


NVTE_FP8_COLLECTION_NAME = QuantizeConfig.COLLECTION_NAME
