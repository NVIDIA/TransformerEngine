# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""
Config module for quantization metadata management

This module provides configuration and helper functions for managing quantization metadata
in JAX, including support for different scaling modes and datatypes.
"""

from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
import hashlib
from typing import Optional, Tuple, Dict, Union, Sequence, Type, List
from functools import reduce, lru_cache
import operator
from importlib.metadata import version as get_pkg_version
import warnings
from packaging.version import Version as PkgVersion

import jax
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict

from transformer_engine_jax import DType, get_cublasLt_version, get_cuda_version
from transformer_engine.common.recipe import (
    Recipe,
    DelayedScaling,
    Format,
    MXFP8BlockScaling,
    Float8CurrentScaling,
    NVFP4BlockScaling,
)
from transformer_engine.jax.sharding import (
    global_shard_guard,
    MeshResource,
    get_num_devices_in_mesh,
    get_all_mesh_axes,
    with_sharding_constraint,
)

from .metadata import QuantizeMeta
from .scaling_modes import ScalingMode
from .device_utils import get_device_compute_capability

__all__ = [
    "get_quantize_config",
    "get_quantize_config_with_recipe",
    "autocast",
    "fp8_autocast",
    "is_fp8_available",
    "is_scaling_mode_supported",
    "get_supported_scaling_modes",
    "get_supported_quantization_recipes",
    "update_collections",
    "apply_padding_to_scale_inv",
    "remove_padding_from_scale_inv",
    "NVTE_FP8_COLLECTION_NAME",
    "TensorSource",
]

_is_scaling_mode_supported = None
_reason_for_no_scaling_mode = ""
Collection = Union[Dict, FrozenDict]

NVTE_FP8_COLLECTION_NAME = "fp8_metas"


@lru_cache(maxsize=None)
def _jax_version_meet_requirement(version: str):
    """
    Helper function checking if required JAX version is available
    """
    jax_version = PkgVersion(get_pkg_version("jax"))
    jax_version_required = PkgVersion(version)
    return jax_version >= jax_version_required


def _check_delayed_scaling_fp8_support(gpu_arch) -> Tuple[bool, str]:
    """Check if delayed scaling FP8 is supported on the given GPU architecture.

    Args:
        gpu_arch: The GPU architecture version

    Returns:
        A tuple of (bool, str) indicating support and any error message
    """
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
    if gpu_arch < 99:  # pre-blackwell
        return False, "Device compute capability 9.9 or higher required for MXFP8 execution."
    if get_cublasLt_version() < 120800:
        return False, "CublasLt version 12.8.0 or higher required for MXFP8 execution."
    if get_cuda_version() < 12080:
        return False, "Cuda version 12.8 or higher required for MXFP8 execution."
    if not _jax_version_meet_requirement("0.5.3"):
        return False, "Jax version 0.5.3 or higher required for MXFP8 execution."
    return True, ""


def _check_fp4_support(gpu_arch) -> Tuple[bool, str]:
    """Check if FP4 is supported for the given GPU architecture."""
    if gpu_arch < 100:  # pre-blackwell
        return False, "Device compute capability 10.0 or higher required for NVFP4 execution."
    if get_cublasLt_version() < 120800:
        return False, "CublasLt version 12.8.0 or higher required for NVFP4 execution."
    if get_cuda_version() < 12080:
        return False, "Cuda version 12.8 or higher required for NVFP4 execution."
    if not _jax_version_meet_requirement("0.5.3"):
        return False, "Jax version 0.5.3 or higher required for NVFP4 execution."
    return True, ""


def _check_scaling_support(scaling_mode: ScalingMode, gpu_id: int) -> Tuple[bool, str]:
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
    if scaling_mode.is_mxfp8_scaling:
        return _check_block_scaling_fp8_support(gpu_arch)
    if scaling_mode.is_nvfp4_scaling:
        return _check_fp4_support(gpu_arch)
    return (True, "")  # NO_SCALING is always supported


def is_scaling_mode_supported(
    scaling_mode=ScalingMode.NO_SCALING,
    gpu_id=None,
) -> Tuple[bool, str]:
    """Check if the given scaling mode is available for the given GPU."""
    if gpu_id is not None:
        return _check_scaling_support(scaling_mode, gpu_id)

    global _is_scaling_mode_supported, _reason_for_no_scaling_mode
    if _is_scaling_mode_supported is None:
        _is_scaling_mode_supported = {}
        _reason_for_no_scaling_mode = {}
    if scaling_mode not in _is_scaling_mode_supported:
        _is_scaling_mode_supported[scaling_mode] = True
        _reason_for_no_scaling_mode[scaling_mode] = ""
        for local_gpu_id in range(len(jax.local_devices())):
            ret, msg = _check_scaling_support(scaling_mode, local_gpu_id)
            if ret is False:
                _is_scaling_mode_supported[scaling_mode] = ret
                _reason_for_no_scaling_mode[scaling_mode] = msg
                return ret, msg
    return _is_scaling_mode_supported[scaling_mode], _reason_for_no_scaling_mode[scaling_mode]


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
    warnings.warn(
        "is_fp8_available is deprecated. Use is_scaling_mode_supported instead.", DeprecationWarning
    )
    return is_scaling_mode_supported(scaling_mode=scaling_mode, gpu_id=gpu_id)


# TODO(Phuong): make the infrastruture to support NO_SCALING
def get_supported_scaling_modes() -> List[ScalingMode]:
    """Get all supported quantization scaling modes."""
    return [
        scaling_mode
        for scaling_mode in ScalingMode
        if is_scaling_mode_supported(scaling_mode=scaling_mode)[0]
        and scaling_mode != ScalingMode.NO_SCALING
    ]


def get_supported_quantization_recipes() -> List[Recipe]:
    """Get all supported quantization recipes."""
    # We don't support all the recipes TE/Common supports yet
    # return [get_quantize_config_class(recipe)() for recipe in recipe.Recipe.__subclasses__()]
    all_recipes = [
        DelayedScaling(),
        Float8CurrentScaling(),
        MXFP8BlockScaling(),
        NVFP4BlockScaling(),
    ]
    return [
        recipe for recipe in all_recipes if get_quantize_config_class(recipe)().is_supported()[0]
    ]


def _format2dtypes(format_: Format):
    """Convert recipe.Format.dtype to corresponding JAX dtypes.

    Args:
        format_: The FP8 format to convert

    Returns:
        A tuple of (forward_dtype, backward_dtype) for the given format
    """
    if format_ == Format.E4M3:
        return jnp.float8_e4m3fn, jnp.float8_e4m3fn
    if format_ == Format.E5M2:
        return jnp.float8_e5m2, jnp.float8_e5m2
    if format_ == Format.HYBRID:
        return jnp.float8_e4m3fn, jnp.float8_e5m2
    if format_ == Format.E2M1:
        return jnp.float4_e2m1fn, jnp.float4_e2m1fn
    return jnp.bfloat16, jnp.bfloat16


class TensorSource(Enum):
    """Enumeration for where a tensor's data comes from."""

    # Input data
    X = 0
    # Model parameters
    KERNEL = 1
    # Gradients in the backward pass
    DGRAD = 2


class AmaxComputeAlgo(Enum):
    """Enumeration for AMAX computation algorithms.

    Attributes:
        MAX: Use maximum value for AMAX computation
        MOST_RECENT: Use most recent value for AMAX computation
    """

    MAX = "max"
    MOST_RECENT = "most_recent"


@dataclass
class BaseQuantizeConfig(ABC):
    """Configuration class for quantization settings.

    This class manages global quantization settings including FP8 formats,
    scaling modes, and accumulation settings.

    Attributes:
        INITIALIZED: Whether the config has been initialized
        MARGIN: Margin value for quantization
        COLLECTION_NAME: Name of the collection for quantization metadata
        FWD_DTYPE: Forward pass data type
        BWD_DTYPE: Backward pass data type
        FP8_2X_ACC_FPROP: Whether to use 2x accumulation for forward pass
        FP8_2X_ACC_DGRAD: Whether to use 2x accumulation for data gradients
        FP8_2X_ACC_WGRAD: Whether to use 2x accumulation for weight gradients
        INFERENCE_MODE: Whether to enable optimization for inference
        AMAX_HISTORY_LEN: Length of AMAX history for delayed scaling
        AMAX_COMPUTE_ALGO: Algorithm for AMAX computation
    """

    INITIALIZED = False
    MARGIN: float = 0.0
    COLLECTION_NAME: str = NVTE_FP8_COLLECTION_NAME
    FWD_DTYPE: DType = None
    BWD_DTYPE: DType = None
    FP8_2X_ACC_FPROP: bool = False
    FP8_2X_ACC_DGRAD: bool = False
    FP8_2X_ACC_WGRAD: bool = False
    INFERENCE_MODE: bool = False

    # DelayedScaling
    # TODO(Phuong): move these two into DelayedScalingQuantizeConfig
    AMAX_HISTORY_LEN: int = 1024
    AMAX_COMPUTE_ALGO: AmaxComputeAlgo = AmaxComputeAlgo.MAX

    def initialize_from_recipe(self, fp8_recipe: Recipe) -> None:
        """Initialize the quantization configuration from a given recipe.

        Args:
            fp8_recipe: The FP8 recipe to use for initialization
        """
        self.INITIALIZED = True
        self.FWD_DTYPE, self.BWD_DTYPE = _format2dtypes(fp8_recipe.fp8_format)

    def is_fp8_enabled(self) -> bool:
        """Check if FP8 quantization is enabled.

        Returns:
            bool: True if quantization is enabled, False otherwise
        """
        return self.INITIALIZED

    @abstractmethod
    def get_scaling_mode(self, tensor_source: TensorSource) -> ScalingMode:
        """Gets the scaling mode for a specific tensor's usage type.

        Args:
            tensor_source: The usage type for which to get the scaling mode.

        Returns:
            The scaling mode for the specified usage type.
        """

    @abstractmethod
    def get_quantize_flax_meta(
        self,
        module,
        collection_name: str,
        postfix: str,
        tensor_source: TensorSource,
        quantizer_name: str,
    ) -> QuantizeMeta:
        """Get the quantization metadata for a given Flax module.

        Args:
            module: The Flax module to get metadata for
            collection_name: The name of the collection to store metadata in
            postfix: Postfix to append to metadata names
            tensor_source: The source type of the tensor (e.g., X, KERNEL, DGRAD)
            quantizer_name: The name of the quantizer within the module
        Returns:
            The quantization metadata for the specified module and tensor. It can be empty if no metadata is needed.
        """

    def is_supported(self) -> tuple[bool, str]:
        """Check if this QuantizeConfig class is supported on the available devices.

        Returns:
            bool: True if the class is supported, False otherwise
            str: Reason for being unsupported, if applicable.
        """

        x_scaling_mode = self.get_scaling_mode(TensorSource.X)
        kernel_scaling_mode = self.get_scaling_mode(TensorSource.KERNEL)
        grad_scaling_mode = self.get_scaling_mode(TensorSource.DGRAD)
        for scaling_mode in [x_scaling_mode, kernel_scaling_mode, grad_scaling_mode]:
            is_supported, reason = is_scaling_mode_supported(scaling_mode=scaling_mode)
            if not is_supported:
                return is_supported, reason
        return True, None


class NoOpQuantizeConfig(BaseQuantizeConfig):
    """Configuration class higher-precision non-quantized operation."""

    def initialize_from_recipe(self, fp8_recipe: Recipe) -> None:
        """Initialize no-op configuration."""
        raise NotImplementedError(
            "NoOpQuantizeConfig cannot be initialize from a recipe as it represents"
            " higher-precision when no quantized recipe is set."
        )

    def get_scaling_mode(self, tensor_source: TensorSource) -> ScalingMode:
        """Gets the scaling mode for a specific tensor's usage type."""
        return ScalingMode.NO_SCALING

    def get_quantize_flax_meta(
        self,
        module,
        collection_name: str,
        postfix: str,
        tensor_source: TensorSource,
        quantizer_name: str,
    ) -> QuantizeMeta:
        """Get the quantization metadata for a given Flax module.

        Args:
            module: The Flax module to get metadata for
            collection_name: The name of the collection to store metadata in
            postfix: Postfix to append to metadata names
            tensor_source: The source type of the tensor (e.g., X, KERNEL, DGRAD)
            quantizer_name: The name of the quantizer within the module
        Returns:
            The quantization metadata for the specified module and tensor. It can be empty if no metadata is needed.
        """
        return QuantizeMeta()


class DelayedScalingQuantizeConfig(BaseQuantizeConfig):
    """Configuration class for delayed scaling FP8 recipe.

    This class provides specific initialization and finalization for delayed scaling
    FP8 quantization mode.
    """

    def initialize_from_recipe(self, fp8_recipe: Recipe) -> None:
        """Initialize delayed scaling FP8 configuration.

        Args:
            fp8_recipe: The FP8 recipe to use for initialization

        Raises:
            AssertionError: If recipe parameters are not supported
        """
        super().initialize_from_recipe(fp8_recipe)
        self.MARGIN = fp8_recipe.margin if "margin" in dir(fp8_recipe) else 0.0

        assert fp8_recipe.amax_compute_algo in [
            "max",
            "most_recent",
        ], "DelayedScaling amax_compute_algo only supports max and most_recent with TE/JAX."
        assert (
            fp8_recipe.scaling_factor_compute_algo is None
        ), "DelayedScaling scaling_factor_compute_algo isn't supported by TE/JAX."
        assert fp8_recipe.reduce_amax, "DelayedScaling reduce_amax should be enabled for TE/JAX."

        self.AMAX_HISTORY_LEN = fp8_recipe.amax_history_len
        string_to_amax_compute_algo = {
            "max": AmaxComputeAlgo.MAX,
            "most_recent": AmaxComputeAlgo.MOST_RECENT,
        }
        self.AMAX_COMPUTE_ALGO = string_to_amax_compute_algo[fp8_recipe.amax_compute_algo]

        self.FP8_2X_ACC_DGRAD = True
        self.FP8_2X_ACC_WGRAD = True

    def get_scaling_mode(self, tensor_source: TensorSource) -> ScalingMode:
        """Gets the scaling mode for a specific tensor's usage type."""
        return ScalingMode.DELAYED_TENSOR_SCALING

    def get_quantize_flax_meta(
        self,
        module,
        collection_name: str,
        postfix: str,
        tensor_source: TensorSource,
        quantizer_name: str,
    ) -> QuantizeMeta:
        """Get the quantization metadata for a given Flax module.

        Args:
            module: The Flax module to get metadata for
            collection_name: The name of the collection to store metadata in
            postfix: Postfix to append to metadata names
            tensor_source: The source type of the tensor (e.g., X, KERNEL, DGRAD)
            quantizer_name: The name of the quantizer within the module
        Returns:
            The quantization metadata for the specified module and tensor. It can be empty if no metadata is needed.
        """
        scale = module.variable(
            collection_name,
            f"{quantizer_name}{postfix}_scale",
            jnp.ones,
            (1,),
            jnp.float32,
        ).value
        amax_history = module.variable(
            collection_name,
            f"{quantizer_name}{postfix}_amax_history",
            jnp.zeros,
            (self.AMAX_HISTORY_LEN,),
            jnp.float32,
        ).value
        return QuantizeMeta(scale=scale, amax_history=amax_history)


class CurrentScalingQuantizeConfig(BaseQuantizeConfig):
    """Configuration class for current scaling FP8 recipe.

    This class provides specific initialization and finalization for current scaling
    FP8 quantization mode.
    """

    def initialize_from_recipe(self, fp8_recipe: Recipe) -> None:
        """Initialize current scaling FP8 configuration.

        Args:
            fp8_recipe: The FP8 recipe to use for initialization
        """
        super().initialize_from_recipe(fp8_recipe)
        self.AMAX_HISTORY_LEN = 0

    def get_scaling_mode(self, tensor_source: TensorSource) -> ScalingMode:
        """Gets the scaling mode for a specific tensor's usage type."""
        return ScalingMode.CURRENT_TENSOR_SCALING

    def get_quantize_flax_meta(
        self,
        module,
        collection_name: str,
        postfix: str,
        tensor_source: TensorSource,
        quantizer_name: str,
    ) -> QuantizeMeta:
        """Get the quantization metadata for a given Flax module.

        Args:
            module: The Flax module to get metadata for
            collection_name: The name of the collection to store metadata in
            postfix: Postfix to append to metadata names
            tensor_source: The source type of the tensor (e.g., X, KERNEL, DGRAD)
            quantizer_name: The name of the quantizer within the module
        Returns:
            The quantization metadata for the specified module and tensor. It can be empty if no metadata is needed.
        """
        return QuantizeMeta()


class BlockScalingQuantizeConfig(BaseQuantizeConfig):
    """Configuration class for block scaling FP8 recipe.

    This class provides specific initialization and finalization for block scaling
    FP8 quantization mode.
    """

    def initialize_from_recipe(self, fp8_recipe: Recipe) -> None:
        """Initialize block scaling FP8 configuration.

        Args:
            fp8_recipe: The FP8 recipe to use for initialization
        """
        super().initialize_from_recipe(fp8_recipe)
        self.AMAX_HISTORY_LEN = 0

    def get_scaling_mode(self, tensor_source: TensorSource) -> ScalingMode:
        """Gets the scaling mode for a specific tensor's usage type."""
        return ScalingMode.MXFP8_1D_SCALING

    def get_quantize_flax_meta(
        self,
        module,
        collection_name: str,
        postfix: str,
        tensor_source: TensorSource,
        quantizer_name: str,
    ) -> QuantizeMeta:
        """Get the quantization metadata for a given Flax module.

        Args:
            module: The Flax module to get metadata for
            collection_name: The name of the collection to store metadata in
            postfix: Postfix to append to metadata names
            tensor_source: The source type of the tensor (e.g., X, KERNEL, DGRAD)
            quantizer_name: The name of the quantizer within the module
        Returns:
            The quantization metadata for the specified module and tensor. It can be empty if no metadata is needed.
        """
        return QuantizeMeta()


@dataclass
class NVFP4ScalingQuantizeConfig(BaseQuantizeConfig):
    """Configuration class for NVFP4 scaling recipe.

    This class provides specific initialization and finalization for NVFP4 scaling quantization mode.
    """

    DISABLE_STOCHASTIC_ROUNDING: bool = False
    DISABLE_RHT: bool = False
    DISABLE_2D_QUANTIZATION: bool = False

    def initialize_from_recipe(self, fp8_recipe: Recipe) -> None:
        """Initialize block scaling NVFP4 configuration.

        Args:
            fp8_recipe: The quantization recipe to use for initialization
        """
        assert isinstance(fp8_recipe, NVFP4BlockScaling)

        self.INITIALIZED = True
        self.FWD_DTYPE, self.BWD_DTYPE = _format2dtypes(fp8_recipe.fp4_format)
        self.AMAX_HISTORY_LEN = 0

        self.DISABLE_STOCHASTIC_ROUNDING = fp8_recipe.disable_stochastic_rounding
        self.DISABLE_RHT = fp8_recipe.disable_rht
        self.DISABLE_2D_QUANTIZATION = fp8_recipe.disable_2d_quantization

    def get_scaling_mode(self, tensor_source: TensorSource) -> ScalingMode:
        """Gets the scaling mode for a specific tensor's usage type."""
        if (not self.DISABLE_2D_QUANTIZATION) and tensor_source == TensorSource.KERNEL:
            return ScalingMode.NVFP4_2D_SCALING
        # for x and grad
        return ScalingMode.NVFP4_1D_SCALING

    def get_quantize_flax_meta(
        self,
        module,
        collection_name: str,
        postfix: str,
        tensor_source: TensorSource,
        quantizer_name: str,
    ) -> QuantizeMeta:
        """Get the quantization metadata for a given Flax module.

        Args:
            module: The Flax module to get metadata for
            collection_name: The name of the collection to store metadata in
            postfix: Postfix to append to metadata names
            tensor_source: The source type of the tensor (e.g., X, KERNEL, DGRAD)
            quantizer_name: The name of the quantizer within the module
        Returns:
            The quantization metadata for the specified module and tensor. It can be empty if no metadata is needed.
        """
        if self.DISABLE_STOCHASTIC_ROUNDING:
            return QuantizeMeta()

        if tensor_source != TensorSource.DGRAD:
            # Only DGRAD uses stochastic rounding
            return QuantizeMeta()

        sr_jax_rng = module.make_rng("sr_rng")
        # Get a unique key for this quantizer
        # Use hashlib to get a deterministic hash value for quantizer_name
        quantizer_hash = (
            int(hashlib.sha256(quantizer_name.encode("utf-8")).hexdigest(), 16)
            % jnp.iinfo(jnp.int32).max
        )
        sr_jax_rng = jax.jit(jax.random.fold_in)(sr_jax_rng, quantizer_hash)

        # Generate 4 random uint32 values from the JAX PRNG key
        shape = (4,)
        if get_num_devices_in_mesh() > 1:
            shape = (get_num_devices_in_mesh(), 4)
        sr_jax_rng_state = jax.random.randint(
            sr_jax_rng, shape, 0, jnp.iinfo(jnp.int32).max, dtype=jnp.int32
        ).view(jnp.uint32)
        sr_jax_rng_state = with_sharding_constraint(
            sr_jax_rng_state, jax.sharding.PartitionSpec(get_all_mesh_axes(), None)
        )

        return QuantizeMeta(stochastic_rounding_rng_state=sr_jax_rng_state)


_QUANTIZE_CONFIG = NoOpQuantizeConfig()


def get_quantize_config():
    """Global instance of BaseQuantizeConfig set by autocast context."""
    return _QUANTIZE_CONFIG


def get_quantize_config_class(
    fp8_recipe: Recipe,
) -> Type[BaseQuantizeConfig]:
    """Get the quantization configuration class based on the FP8 recipe.

    Args:
        fp8_recipe: The FP8 recipe to use for initialization
    Returns:
        The quantization config class corresponding to the given recipe.
    """
    if isinstance(fp8_recipe, DelayedScaling):
        return DelayedScalingQuantizeConfig
    if isinstance(fp8_recipe, MXFP8BlockScaling):
        return BlockScalingQuantizeConfig
    if isinstance(fp8_recipe, Float8CurrentScaling):
        return CurrentScalingQuantizeConfig
    if isinstance(fp8_recipe, NVFP4BlockScaling):
        return NVFP4ScalingQuantizeConfig
    raise ValueError(f"Unsupported recipe type: {type(fp8_recipe)}")


def get_quantize_config_with_recipe(fp8_recipe: Recipe):
    """Get the quantization configuration object based on the FP8 recipe."""
    config = get_quantize_config_class(fp8_recipe)()
    config.initialize_from_recipe(fp8_recipe)
    return config


@contextmanager
def autocast(
    enabled: bool = False,
    recipe: Optional[Recipe] = None,
    mesh_resource: Optional[MeshResource] = None,
) -> None:
    r"""Context manager for FP8 or FP4 usage.

    This context manager enables quantization for the duration of its context.
        .. code-block:: python

            mesh_shape = (4, 2)
            dp_mesh_axis_name = 'data_parallel'
            tp_mesh_axis_name = 'tensor_parallel'
            devices = np.asarray(jax.devices()).reshape(*mesh_shape)

            with maps.Mesh(devices, (dp_mesh_axis_name, tp_mesh_axis_name)):
                mesh_resource=MeshResource(dp_mesh_axis_name, tp_mesh_axis_name)

                with autocast(enabled=True, mesh_resource=mesh_resource):
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
    recipe: recipe.DelayedScaling, default = None
            recipe used for low precision quantization.
    mesh_resource: MeshResource, default = None
        Specify the mesh axes for data and tensor parallelism to shard along.
        If set to None, then no data or tensor parallelism will be used.

    """
    if recipe is None:
        recipe = DelayedScaling()

    global _QUANTIZE_CONFIG

    old_quantize_config = _QUANTIZE_CONFIG

    _QUANTIZE_CONFIG = NoOpQuantizeConfig()

    try:
        with global_shard_guard(mesh_resource):
            if enabled:
                _QUANTIZE_CONFIG = get_quantize_config_class(recipe)()
                is_supported, reason = _QUANTIZE_CONFIG.is_supported()
                assert is_supported, reason
                _QUANTIZE_CONFIG.initialize_from_recipe(recipe)
            yield
    finally:
        _QUANTIZE_CONFIG = old_quantize_config


@contextmanager
def fp8_autocast(
    enabled: bool = False,
    fp8_recipe: Optional[Recipe] = None,
    mesh_resource: Optional[MeshResource] = None,
) -> None:
    """
    .. warning::

       fp8_autocast is deprecated and will be removed in a future release.
       Use autocast(enabled=..., recipe=..., mesh_resource=...) instead.

    """

    warnings.warn(
        "fp8_autocast is deprecated and will be removed in a future release. "
        "Use autocast(enabled=..., recipe=..., mesh_resource=...) instead.",
        category=DeprecationWarning,
        stacklevel=2,
    )

    # Call new implementation.
    with autocast(
        enabled=enabled,
        recipe=fp8_recipe,
        mesh_resource=mesh_resource,
    ):
        yield


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


def remove_padding_from_scale_inv(
    scale_inv: jax.Array,
    scaling_mode: ScalingMode,
    data_shape: Sequence[int],
    is_colwise: bool = False,
    flatten_axis: int = -1,
):
    """
    Slice padding out of padded inverse scale factors.

    Args:
        scale_inv: Inverse scale factor.
        data_shape: Shape of the quantized data the inverse scale belongs to.
        scaling_mode: ScalingMode representing the quantization method.
        is_colwise: Whether the data was quantized column-wise.
        flatten_axis: The axis along with the data could be flattened to 2D.

    Returns:
        Inverse scale factor without padding.
    """
    # Get expected unpadded scale shape and check if inverse scale already matches
    unpadded_scale_shape = scaling_mode.get_scale_shape(
        data_shape, is_colwise=is_colwise, is_padded=False, flatten_axis=flatten_axis
    )
    if scaling_mode == ScalingMode.NO_SCALING or scale_inv.shape == unpadded_scale_shape:
        return scale_inv

    # Get the padded scale shape and make sure inverse scale matches
    padded_scale_shape = scaling_mode.get_scale_shape(
        data_shape,
        is_colwise=is_colwise,
        is_padded=True,
        flatten_axis=flatten_axis,
    )
    assert scale_inv.shape == padded_scale_shape, (
        f"Padded inverse scale factor has wrong shape, expected {padded_scale_shape} but got "
        f"{scale_inv.shape} instead."
    )

    # Reshape scale inverse to 2D in two stages to preserve the flatten axis
    padded_scale_shape_2d = (
        reduce(operator.mul, padded_scale_shape[:flatten_axis]),
        reduce(operator.mul, padded_scale_shape[flatten_axis:]),
    )
    scale_inv_2d = jnp.reshape(
        jnp.reshape(scale_inv, (padded_scale_shape_2d[0], *scale_inv.shape[flatten_axis:])),
        padded_scale_shape_2d,
    )

    # Slice reshaped 2D scale inverse using collapsed 2D unpadded_scale_shape
    unpadded_scale_shape_2d = (
        reduce(operator.mul, unpadded_scale_shape[:flatten_axis]),
        reduce(operator.mul, unpadded_scale_shape[flatten_axis:]),
    )
    scale_inv_2d_unpadded = jnp.asarray(
        scale_inv_2d[: unpadded_scale_shape_2d[0], : unpadded_scale_shape_2d[1]]
    )

    # Reshape 2D scale inverse back in two stages in order to preserve the flatten axis
    scale_inv_unpadded = jnp.reshape(
        jnp.reshape(
            scale_inv_2d_unpadded,
            (*unpadded_scale_shape[:flatten_axis], scale_inv_2d_unpadded.shape[1]),
        ),
        unpadded_scale_shape,
    )
    return scale_inv_unpadded


def apply_padding_to_scale_inv(
    scale_inv: jax.Array,
    scaling_mode: ScalingMode,
    data_shape: Sequence[int],
    is_colwise: bool = False,
    flatten_axis: int = -1,
):
    """
    Pad the scale inverse with zeros to match the necessary padded shape for this scaling
    mode.

    Args:
        scale_inv: Inverse scale factor.
        data_shape: Shape of the quantized data the inverse scale belongs to.
        scaling_mode: ScalingMode representing the quantization method.
        is_colwise: Whether the data was quantized column-wise.
        flatten_axis: The axis along with the data could be flattened to 2D.

    Returns:
        Padded inverse scale factor.
    """
    # Get the expected padded scale shape and check if inverse scale already matches
    padded_scale_shape = scaling_mode.get_scale_shape(
        data_shape, is_colwise=is_colwise, is_padded=True, flatten_axis=flatten_axis
    )
    if scaling_mode == ScalingMode.NO_SCALING or scale_inv.shape == padded_scale_shape:
        return scale_inv

    # Get the expected unpadded scale shape and make sure inverse scales match
    unpadded_scale_shape = scaling_mode.get_scale_shape(
        data_shape, is_colwise=is_colwise, is_padded=False, flatten_axis=flatten_axis
    )
    assert scale_inv.shape == unpadded_scale_shape, (
        f"Unpadded inverse scale factor has wrong shape, expected {unpadded_scale_shape} but got "
        f"{scale_inv.shape}."
    )

    # Pad the scales with the lowest representable value (2^-127) and return
    pad_width = tuple((0, a - b) for a, b in zip(padded_scale_shape, unpadded_scale_shape))
    return jnp.pad(scale_inv, pad_width=pad_width, mode="constant", constant_values=2**-127)
