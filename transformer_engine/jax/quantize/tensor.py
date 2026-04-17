# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""
Tensor classes for TE/JAX

This module provides tensor classes for handling quantized tensors in JAX, including
both single-scale (1x) and double-scale (2x) quantization schemes. It supports
rowwise and colwise quantization modes with proper scaling and dequantization.
"""
from dataclasses import dataclass
from typing import Callable, Optional, Tuple
from abc import ABC, abstractmethod

import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class
from jax.ad_checkpoint import checkpoint_name as jax_checkpoint_name


from .scaling_modes import ScalingMode, TensorUsage
from .dequantizer import ScalingModeToDequantizerMap
from .misc import QuantizeLayout
from ..sharding import (
    with_sharding_constraint_by_logical_axes as original_with_sharding_constraint_by_logical_axes,
)

__all__ = [
    "TensorUsage",
    "AbstractBaseTensor",
    "NoScaleTensor",
    "ScaledTensor",
    "ScaledTensor1x",
    "ScaledTensor2x",
    "GroupedScaledTensor1x",
    "GroupedNoScaleTensor",
    "ScaledTensorFactory",
    "with_sharding_constraint_by_logical_axes",
]


@dataclass
class AbstractBaseTensor(ABC):
    """Abstract base class for all tensor types."""

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """Reconstructs the tensor from its flattened representation.

        Args:
            aux_data: Auxiliary data needed for reconstruction
            children: The flattened tensor components

        Returns:
            A reconstructed tensor instance
        """
        return cls(*children, *aux_data)

    @property
    @abstractmethod
    def ndim(self):
        """Number of dimensions of the underlying quantized array."""

    @abstractmethod
    def dequantize(self):
        """Dequantizes the tensor back to its original precision.

        Returns:
            The dequantized tensor
        """

    @abstractmethod
    def get_tensor(self, usage: TensorUsage):
        """Returns the appropriate tensor based on the tensor usage and the scaling mode.
        If the tensor usage is not valid for the scaling mode, an error is raised.

        Args:
            usage: The usage of the tensor

        Returns:
            The tensor based on the usage
        """

    @abstractmethod
    def apply_sharding_constraint_by_logical_axes(self, logical_axis_names: Tuple[str, ...]):
        """Applies sharding constraints to a tensor based on logical axis names.

        Args:
            logical_axis_names: Tuple of logical axis names for sharding

        Returns:
            The tensor with applied sharding constraints
        """

    @abstractmethod
    def checkpoint(self, quantizer):
        """Checkpoints the tensor with the given quantizer's checkpoint name if available.

        Args:
            quantizer: The quantizer to use for checkpointing. If None, no checkpointing is applied.

        Returns:
            The checkpointed tensor
        """


@dataclass
class AbstractBaseTensor1x(AbstractBaseTensor):
    """Abstract base class for single layout tensors."""

    data: jnp.ndarray
    amax: jnp.ndarray


@register_pytree_node_class
@dataclass
class NoScaleTensor(AbstractBaseTensor1x):
    """Higher-precision tensor."""

    def __post_init__(self):
        assert isinstance(self.data, jnp.ndarray), "NoScaleTensor's data must be a jnp.ndarray."

    def tree_flatten(self):
        """Flattens the tensor for JAX tree operations.

        Returns:
            A tuple containing (children, aux_data) for tree operations
        """
        children = (self.data, self.amax)
        aux_data = ()
        return (children, aux_data)

    @property
    def ndim(self):
        """Number of dimensions of the underlying array."""
        return self.data.ndim

    def dequantize(self):
        """This is a no-op for a higher-precision tensor so this simply returns the tensor's data."""
        return self.data

    def get_tensor(self, usage: TensorUsage):
        """Returns the tensor based on the tensor usage."""
        q_layout = ScalingMode.NO_SCALING.get_quantize_layout(usage)
        assert q_layout.is_rowwise_only, "Only ROWWISE layout is supported for NoScaleTensor"
        return self

    def apply_sharding_constraint_by_logical_axes(self, logical_axis_names: Tuple[str, ...]):
        """Applies sharding constraints to a tensor based on logical axis names.

        Args:
            logical_axis_names: Tuple of logical axis names for sharding

        Returns:
            The tensor with applied sharding constraints
        """
        if not logical_axis_names:
            return self

        data = with_sharding_constraint_by_logical_axes(self.data, logical_axis_names)

        return NoScaleTensor(
            data=data,
            amax=self.amax,
        )

    def checkpoint(self, quantizer):
        """Checkpoints the tensor with the given quantizer's checkpoint name if available.

        Args:
            quantizer: The quantizer to use for checkpointing. If None, no checkpointing is applied.

        Returns:
            The checkpointed tensor
        """
        assert quantizer is None, "NoScaleTensor does not support quantization."
        return self


class ScaledTensor(ABC):
    """Abstract base class for scaled tensors."""


@register_pytree_node_class
@dataclass
class ScaledTensor1x(AbstractBaseTensor1x, ScaledTensor):
    """Single-scale quantized tensor implementation.

    This class represents a tensor quantized with a single scaling factor,
    supporting both row-wise and column-wise quantization modes.

    Attributes:
        data: The quantized tensor data
        scale_inv: The inverse scaling factors
        amax: The maximum absolute value of the tensor
        scaling_mode: The scaling mode used for quantization
        dq_dtype: The data type for dequantized values
        _dq_func: The dequantization function
        is_colwise: Whether the tensor uses column-wise quantization
        data_layout: The data_layout specification for the tensor
        flatten_axis: The quantization axis for the tensor
        has_rht_applied: Whether the tensor had the Randomized Hadamard Transform (RHT) applied during quantization
    """

    scale_inv: jnp.ndarray
    scaling_mode: ScalingMode
    dq_dtype: jnp.dtype
    _dq_func: Callable
    is_colwise: bool
    data_layout: str
    flatten_axis: int
    has_rht_applied: bool

    def __post_init__(self):
        """Validates and adjusts the scale_inv shape after initialization.

        Ensures the scale_inv shape matches the expected shape based on the scaling mode
        and quantization direction. Pads the scale_inv if necessary.
        """
        assert self.flatten_axis > 0
        assert (
            0 < self.flatten_axis < len(self.data.shape)
        ), f"flatten_axis {self.flatten_axis} is out of bounds for shape {self.data.shape}"

        if self.scaling_mode == ScalingMode.NO_SCALING:
            self.scale_inv = jnp.empty((0,), dtype=jnp.float32)
        else:
            unpadded_scale_shape = self.scaling_mode.get_scale_shape(
                self.data.shape,
                data_layout=self.data_layout,
                is_colwise=self.is_colwise,
                is_padded=False,
                # expect the flatten_axis wrt the N layout
                flatten_axis=(
                    self.flatten_axis
                    if self.data_layout == "N"
                    else self.data.ndim - self.flatten_axis
                ),
            )
            unpadded_scale_shape_broadcast = self.scaling_mode.get_scale_shape(
                self.data.shape,
                data_layout=self.data_layout,
                is_colwise=self.is_colwise,
                is_padded=False,
                # expect the flatten_axis wrt the N layout
                flatten_axis=(
                    self.flatten_axis
                    if self.data_layout == "N"
                    else self.data.ndim - self.flatten_axis
                ),
                broadcast_2d_scale_shape_to_1d=True,
            )
            assert self.scale_inv.shape in (unpadded_scale_shape, unpadded_scale_shape_broadcast), (
                f"Unpadded inverse scale factor has wrong shape, expected {unpadded_scale_shape} or"
                f" {unpadded_scale_shape_broadcast} but got {self.scale_inv.shape}."
            )

    def tree_flatten(self):
        """Flattens the tensor for JAX tree operations.

        Returns:
            A tuple containing (children, aux_data) for tree operations
        """
        children = (self.data, self.amax, self.scale_inv)
        aux_data = (
            self.scaling_mode,
            self.dq_dtype,
            self._dq_func,
            self.is_colwise,
            self.data_layout,
            self.flatten_axis,
            self.has_rht_applied,
        )
        return (children, aux_data)

    @property
    def ndim(self):
        return self.data.ndim

    def dequantize(self):
        """Dequantizes the tensor using the stored dequantization function.

        Returns:
            The dequantized tensor
        """
        return self._dq_func(self)

    def get_tensor(self, usage: TensorUsage):
        """Returns the tensor based on the tensor usage."""
        q_layout = self.scaling_mode.get_quantize_layout(usage)
        colwise_usage_valid = q_layout.is_colwise_only and self.is_colwise
        rowwise_usage_valid = q_layout.is_rowwise_only and not self.is_colwise

        if colwise_usage_valid or rowwise_usage_valid:
            return self

        raise ValueError(
            f"Calling get_tensor() with usage {usage} is not valid for this tensor as"
            f" self.is_colwise={self.is_colwise}!"
        )

    def apply_sharding_constraint_by_logical_axes(self, logical_axis_names: Tuple[str, ...]):
        """Applies sharding constraints to a tensor based on logical axis names.

        Args:
            logical_axis_names: Tuple of logical axis names for sharding

        Returns:
            The tensor with applied sharding constraints
        """
        if not logical_axis_names:
            return self

        # axis_names were given for N layout, so needs to be transpose for T layout
        if self.data_layout == "T":
            assert self.flatten_axis > 0
            assert len(logical_axis_names) == self.data.ndim
            flatten_axis = self.data.ndim - self.flatten_axis
            axis_names = (
                *logical_axis_names[flatten_axis:],
                *logical_axis_names[:flatten_axis],
            )
        else:
            axis_names = logical_axis_names

        data = with_sharding_constraint_by_logical_axes(self.data, axis_names)

        if self.scaling_mode.is_block_scaling:  # Both MXFP8 and NVFP4
            scale_inv = with_sharding_constraint_by_logical_axes(self.scale_inv, axis_names)
        else:
            scale_inv = self.scale_inv

        return ScaledTensor1x(
            data=data,
            amax=self.amax,
            scale_inv=scale_inv,
            scaling_mode=self.scaling_mode,
            dq_dtype=self.dq_dtype,
            _dq_func=self._dq_func,
            is_colwise=self.is_colwise,
            data_layout=self.data_layout,
            flatten_axis=self.flatten_axis,
            has_rht_applied=self.has_rht_applied,
        )

    def checkpoint(self, quantizer):
        """Checkpoints the tensor with the given quantizer's checkpoint name if available.

        Args:
            quantizer: The quantizer to use for checkpointing. If None, no checkpointing is applied.

        Returns:
            The checkpointed tensor
        """
        if quantizer is None or quantizer.checkpoint_name is None:
            return self

        return jax_checkpoint_name(self, name=quantizer.checkpoint_name)


@register_pytree_node_class
@dataclass
class GroupedScaledTensor1x(ScaledTensor1x):
    """Grouped Quantizer for an array.

    This class extends ScaledTensor1x to support quantization of an array in grouped manner,
    where elements are grouped along a specified axis.

    Attributes:
        first_dims: Per-group sizes of the first (row) 2D dim, or None if not ragged
        last_dims: Per-group sizes of the last (col) 2D dim, or None if not ragged
        original_shape: The original shape of the tensor before grouping
        pre_swizzled: Whether the scale_inv is already swizzled for GEMM. True when produced
            by V2 grouped quantize (nvte_group_quantize fuses the swizzle). The V2 grouped
            GEMM FFI requires pre_swizzled=True for MXFP8 inputs and will not re-swizzle.
    """

    first_dims: Optional[jnp.ndarray]
    last_dims: Optional[jnp.ndarray]
    original_shape: Tuple
    pre_swizzled: bool = False

    def __init__(
        self,
        data,
        scale_inv,
        amax,
        first_dims,
        last_dims,
        scaling_mode,
        dq_dtype,
        _dq_func,
        is_colwise,
        data_layout,
        flatten_axis,
        original_shape,
        pre_swizzled=False,
    ):
        self.flatten_axis = flatten_axis
        self.first_dims = first_dims
        self.last_dims = last_dims
        self.original_shape = original_shape
        self.pre_swizzled = pre_swizzled
        # TODO(Phuong):Handle RHT for grouped quantization once grouped quantization supports NVFP4
        super().__init__(
            data=data,
            scale_inv=scale_inv,
            amax=amax,
            scaling_mode=scaling_mode,
            dq_dtype=dq_dtype,
            _dq_func=_dq_func,
            is_colwise=is_colwise,
            data_layout=data_layout,
            flatten_axis=flatten_axis,
            has_rht_applied=False,
        )

    @property
    def group_sizes(self) -> jnp.ndarray:
        """Per-group sizes along the group axis.

        When first_dims is set (ragged groups), returns first_dims.
        When first_dims is None (equal-sized groups), returns an array of ones with
        length equal to the number of groups.
        """
        if self.first_dims is not None and self.first_dims.size > 0:
            return self.first_dims
        return jnp.ones((self.original_shape[0],), dtype=jnp.int32)

    def __post_init__(self):
        assert self.scale_inv.ndim == 1, "Only support flattened scale_inv"
        assert self.data.ndim == 1, "Only support flattened data"
        assert self.flatten_axis > 0

        data_ndim = len(self.original_shape)
        assert (
            0 < self.flatten_axis < data_ndim
        ), f"flatten_axis {self.flatten_axis} is out of bounds for data.ndim = {data_ndim}"

        active_dims = (
            self.first_dims
            if self.first_dims is not None and self.first_dims.size > 0
            else self.last_dims
        )
        if active_dims is not None:
            num_groups = active_dims.size
        else:
            num_groups = self.original_shape[0]

        expected_scale_shape = self.scaling_mode.get_grouped_scale_shape(
            self.original_shape,
            num_groups,
            self.is_colwise,
            is_padded=True,
            flatten_axis=self.flatten_axis,
        )

        assert self.scale_inv.shape == expected_scale_shape, (
            f"Unexpected scale_inv shape! \nExpect {expected_scale_shape} for padded"
            f" scale_inv, got {self.scale_inv.shape}"
        )

    def tree_flatten(self):
        """Flattens the tensor for JAX tree operations.

        Returns:
            A tuple containing (children, aux_data) for tree operations
        """
        children = (self.data, self.scale_inv, self.amax, self.first_dims, self.last_dims)
        aux_data = (
            self.scaling_mode,
            self.dq_dtype,
            self._dq_func,
            self.is_colwise,
            self.data_layout,
            self.flatten_axis,
            self.original_shape,
            self.pre_swizzled,
        )
        return (children, aux_data)

    def apply_sharding_constraint_by_logical_axes(self, logical_axis_names: Tuple[str, ...]):
        raise NotImplementedError

    def checkpoint(self, quantizer):
        """Checkpoints the tensor with the given quantizer's checkpoint name if available.

        Args:
            quantizer: The quantizer to use for checkpointing. If None, no checkpointing is applied.

        Returns:
            The checkpointed tensor
        """
        if quantizer is None or quantizer.checkpoint_name is None:
            return self

        return jax_checkpoint_name(self, name=quantizer.checkpoint_name)


@register_pytree_node_class
@dataclass
class GroupedNoScaleTensor(AbstractBaseTensor1x):
    """Unquantized grouped tensor.

    Stores N-D data with per-group dimension sizes so that grouped_gemm()
    can extract first/last dims automatically without explicit parameters.

    Attributes:
        data: The raw (unquantized) tensor data in N-D layout
        first_dims: Per-group sizes of the first (row) 2D dim, or None if not ragged
        last_dims: Per-group sizes of the last (col) 2D dim, or None if not ragged
        original_shape: Shape of data (same as data.shape for N-D unquantized)
    """

    first_dims: Optional[jnp.ndarray]
    last_dims: Optional[jnp.ndarray]
    original_shape: Tuple

    def tree_flatten(self):
        """Flattens the tensor for JAX tree operations."""
        children = (self.data, self.amax, self.first_dims, self.last_dims)
        aux_data = (self.original_shape,)
        return (children, aux_data)

    @property
    def ndim(self):
        """Number of dimensions of the underlying array."""
        return self.data.ndim

    def dequantize(self):
        """This is a no-op for a higher-precision tensor so this simply returns the tensor's data."""
        return self.data

    def get_tensor(self, usage: TensorUsage):
        """Returns the tensor based on the tensor usage."""
        q_layout = ScalingMode.NO_SCALING.get_quantize_layout(usage)
        assert q_layout.is_rowwise_only, "Only ROWWISE layout is supported for NoScaleTensor"
        return self

    def apply_sharding_constraint_by_logical_axes(self, logical_axis_names: Tuple[str, ...]):
        """Applies sharding constraints to a tensor based on logical axis names.

        Args:
            logical_axis_names: Tuple of logical axis names for sharding

        Returns:
            The tensor with applied sharding constraints
        """
        if not logical_axis_names:
            return self

        data = with_sharding_constraint_by_logical_axes(self.data, logical_axis_names)

        return GroupedNoScaleTensor(
            data=data,
            amax=self.amax,
            first_dims=self.first_dims,
            last_dims=self.last_dims,
            original_shape=self.original_shape,
        )

    def checkpoint(self, quantizer):
        """Checkpoints the tensor with the given quantizer's checkpoint name if available.

        Args:
            quantizer: The quantizer to use for checkpointing. If None, no checkpointing is applied.

        Returns:
            The checkpointed tensor
        """
        assert quantizer is None, "NoScaleTensor does not support quantization."
        return self


@register_pytree_node_class
@dataclass
class ScaledTensor2x(AbstractBaseTensor, ScaledTensor):
    """Double-scale quantized tensor implementation.

    This class represents a tensor quantized with both row-wise and column-wise scaling factors.

    Attributes:
        rowwise_tensor: The row-wise quantized component
        colwise_tensor: The column-wise quantized component
    """

    rowwise_tensor: ScaledTensor1x
    colwise_tensor: ScaledTensor1x

    def tree_flatten(self):
        """Flattens the tensor for JAX tree operations.

        Returns:
            A tuple containing (children, aux_data) for tree operations
        """
        children = (self.rowwise_tensor, self.colwise_tensor)
        aux_data = ()
        return (children, aux_data)

    @property
    def ndim(self):
        """Number of dimensions of the underlying row-wise tensor."""
        return self.rowwise_tensor.ndim

    def dequantize(self):
        """Dequantizes the tensor using the row-wise component's dequantization.

        Returns:
            The dequantized tensor
        """
        return self.rowwise_tensor.dequantize()

    def get_tensor(self, usage: TensorUsage):
        """Returns the tensor based on the tensor usage."""
        q_layout_rowwise = self.rowwise_tensor.scaling_mode.get_quantize_layout(usage)
        q_layout_colwise = self.colwise_tensor.scaling_mode.get_quantize_layout(usage)

        if q_layout_rowwise.is_rowwise_only:
            return self.rowwise_tensor

        if q_layout_colwise.is_colwise_only:
            return self.colwise_tensor

        raise ValueError(
            f"Calling get_tensor() with usage {usage} is not valid for this tensor as"
            f" q_layout_rowwise={q_layout_rowwise} and q_layout_colwise={q_layout_colwise}!"
        )

    def apply_sharding_constraint_by_logical_axes(self, logical_axis_names: Tuple[str, ...]):
        """Applies sharding constraints to a tensor based on logical axis names.

        Args:
            logical_axis_names: Tuple of logical axis names for sharding

        Returns:
            The tensor with applied sharding constraints
        """
        if not logical_axis_names:
            return self

        rowwise_tensor = self.rowwise_tensor.apply_sharding_constraint_by_logical_axes(
            logical_axis_names
        )
        colwise_tensor = self.colwise_tensor.apply_sharding_constraint_by_logical_axes(
            logical_axis_names
        )

        return ScaledTensor2x(rowwise_tensor, colwise_tensor)

    def checkpoint(self, quantizer):
        raise NotImplementedError


@dataclass
class ScaledTensorFactory:
    """Factory class for creating scaled tensor instances.

    Provides static methods to create both single-scale (1x) and double-scale (2x)
    quantized tensors with various configurations.
    """

    @staticmethod
    def create_1x(
        data,
        scale_inv,
        amax=None,
        scaling_mode=ScalingMode.NO_SCALING,
        dq_dtype=jnp.bfloat16,
        is_colwise=False,
        data_layout="N",
        flatten_axis=-1,
        first_dims=None,
        last_dims=None,
        original_shape=None,
        has_rht_applied=False,
        pre_swizzled=False,
    ):
        """Creates a single-scale quantized tensor.

        Args:
            data: The quantized tensor data
            scale_inv: The inverse scaling factors
            amax: The maximum absolute value of the tensor
            scaling_mode: The scaling mode for quantization
            dq_dtype: The data type for dequantized values (default: bfloat16)
            is_colwise: Whether to use column-wise quantization (default: False)
            data_layout: The data_layout specification (default: "N")
            flatten_axis: The quantization axis for the tensor
            first_dims: Per-group sizes of the first (row) 2D dim (default: None)
            last_dims: Per-group sizes of the last (col) 2D dim (default: None)
            original_shape: The original shape of the tensor before grouping (default: None)
            has_rht_applied: Whether the tensor had the Randomized Hadamard Transform (RHT) applied during quantization (default: False)

        Returns:
            A ScaledTensor1x or GroupedScaledTensor1x instance depending on whether first_dims or last_dims is provided
        """
        if amax is None:
            amax = jnp.empty((1,), dtype=jnp.float32)

        dequantizer = ScalingModeToDequantizerMap.get(scaling_mode)

        if first_dims is not None or last_dims is not None or original_shape is not None:
            assert (
                original_shape is not None
            ), "original_shape is not given for GroupedScaledTensor1x"
            flatten_axis = (len(original_shape) + flatten_axis) % len(original_shape)

            # Determine num_groups from whichever dims array is provided, or from original_shape
            active_dims = (
                first_dims if first_dims is not None and first_dims.size > 0 else last_dims
            )
            if active_dims is not None:
                num_groups = active_dims.size
            else:
                num_groups = original_shape[0]

            # Handling attrs of transposed tensors
            if data_layout == "T":
                if original_shape[0] == num_groups:
                    original_shape = (
                        original_shape[0],
                        *original_shape[flatten_axis:],
                        *original_shape[1:flatten_axis],
                    )
                    flatten_axis = len(original_shape) - flatten_axis + 1
                else:
                    original_shape = (
                        *original_shape[flatten_axis:],
                        *original_shape[:flatten_axis],
                    )
                    flatten_axis = len(original_shape) - flatten_axis

            return GroupedScaledTensor1x(
                data=data,
                scale_inv=scale_inv,
                amax=amax,
                scaling_mode=scaling_mode,
                dq_dtype=dq_dtype,
                _dq_func=dequantizer.grouped_dequantize,
                is_colwise=is_colwise,
                data_layout=data_layout,
                flatten_axis=flatten_axis,
                first_dims=first_dims,
                last_dims=last_dims,
                original_shape=original_shape,
                pre_swizzled=pre_swizzled,
            )

        # Handling attrs of transposed tensors
        flatten_axis = (data.ndim + flatten_axis) % data.ndim
        if data_layout == "T":
            flatten_axis = data.ndim - flatten_axis

        return ScaledTensor1x(
            data=data,
            scale_inv=scale_inv,
            amax=amax,
            scaling_mode=scaling_mode,
            dq_dtype=dq_dtype,
            _dq_func=dequantizer.dequantize,
            is_colwise=is_colwise,
            data_layout=data_layout,
            flatten_axis=flatten_axis,
            has_rht_applied=has_rht_applied,
        )

    @staticmethod
    def create_2x(
        data,
        scale_inv,
        colwise_data,
        colwise_scale_inv,
        amax=None,
        colwise_amax=None,
        scaling_mode=ScalingMode.NO_SCALING,
        dq_dtype=jnp.bfloat16,
        data_layout="NN",
        flatten_axis=-1,
        first_dims=None,
        last_dims=None,
        original_shape=None,
        rowwise_has_rht_applied=False,
        colwise_has_rht_applied=False,
        pre_swizzled=False,
    ):
        """Creates a double-scale quantized tensor.

        Args:
            data: The row-wise quantized data
            scale_inv: The row-wise inverse scaling factors
            colwise_data: The column-wise quantized data
            colwise_scale_inv: The column-wise inverse scaling factors
            amax: The maximum absolute value of the tensor
            scaling_mode: The scaling mode for quantization
            dq_dtype: The data type for dequantized values (default: bfloat16)
            data_layout: The data_layout specification (default: "NN")
            flatten_axis: The quantization axis for the tensor
            first_dims: Per-group sizes of the first (row) 2D dim (default: None)
            last_dims: Per-group sizes of the last (col) 2D dim (default: None)
            original_shape: The original shape of the tensor before grouping (default: None)
            rowwise_has_rht_applied: Whether the row-wise tensor uses the Randomized Hadamard Transform (RHT) (default: False)
            colwise_has_rht_applied: Whether the column-wise tensor uses the Randomized Hadamard Transform (RHT) (default: False)

        Returns:
            A ScaledTensor2x instance
        """
        if amax is None:
            amax = jnp.empty((1,), dtype=jnp.float32)
        if colwise_amax is None:
            colwise_amax = amax

        assert len(data_layout) == 2, f"Expect 2 layouts, got {data_layout}"
        rowwise_tensor = ScaledTensorFactory.create_1x(
            data,
            scale_inv,
            amax,
            scaling_mode,
            dq_dtype,
            is_colwise=False,
            data_layout=data_layout[0],
            flatten_axis=flatten_axis,
            first_dims=first_dims,
            last_dims=last_dims,
            original_shape=original_shape,
            has_rht_applied=rowwise_has_rht_applied,
            pre_swizzled=pre_swizzled,
        )
        colwise_tensor = ScaledTensorFactory.create_1x(
            colwise_data,
            colwise_scale_inv,
            colwise_amax,
            scaling_mode,
            dq_dtype,
            is_colwise=True,
            data_layout=data_layout[1],
            flatten_axis=flatten_axis,
            first_dims=first_dims,
            last_dims=last_dims,
            original_shape=original_shape,
            has_rht_applied=colwise_has_rht_applied,
            pre_swizzled=pre_swizzled,
        )
        return ScaledTensor2x(rowwise_tensor, colwise_tensor)

    @staticmethod
    def create(
        data: jnp.ndarray,
        scale_inv: jnp.ndarray,
        colwise_data: jnp.ndarray,
        colwise_scale_inv: jnp.ndarray,
        amax=None,
        colwise_amax=None,
        scaling_mode: ScalingMode = ScalingMode.NO_SCALING,
        dq_dtype: jnp.dtype = jnp.bfloat16,
        data_layout: str = "NN",
        q_layout: QuantizeLayout = QuantizeLayout.ROWWISE,
        flatten_axis: int = -1,
        first_dims: jnp.ndarray = None,
        last_dims: jnp.ndarray = None,
        original_shape: Tuple[int] = None,
        rowwise_has_rht_applied: bool = False,
        colwise_has_rht_applied: bool = False,
        pre_swizzled: bool = False,
    ):
        """Creates a scaled tensor based on the quantization axis.

        Args:
            data: The quantized tensor data
            scale_inv: The inverse scaling factors
            colwise_data: The column-wise quantized data
            colwise_scale_inv: The column-wise inverse scaling factors
            scaling_mode: The scaling mode for quantization
            dq_dtype: The data type for dequantized values (default: bfloat16)
            data_layout: The data_layout specification (default: "NN")
            q_layout: The quantization axis (default: ROWWISE)
            flatten_axis: The axis along which the tensor could be flattened to 2D (default: -1)
            first_dims: Per-group sizes of the first (row) 2D dim (default: None)
            last_dims: Per-group sizes of the last (col) 2D dim (default: None)
            original_shape: The original shape of the tensor before grouping (default: None)
            rowwise_has_rht_applied: Whether the row-wise tensor uses the Randomized Hadamard Transform (RHT) (default: False)
            colwise_has_rht_applied: Whether the col-wise tensor uses the Randomized Hadamard Transform (RHT) (default: False)
            pre_swizzled: Whether scale_inv is already swizzled (produced by V2 grouped quantize).

        Returns:
            Either a ScaledTensor1x or ScaledTensor2x instance depending on q_layout
        """
        assert not rowwise_has_rht_applied, "RHT is not supported for rowwise quantization yet"

        if q_layout.is_rowwise_colwise:
            return ScaledTensorFactory.create_2x(
                data,
                scale_inv,
                colwise_data,
                colwise_scale_inv,
                amax,
                colwise_amax,
                scaling_mode,
                dq_dtype,
                data_layout=data_layout,
                flatten_axis=flatten_axis,
                first_dims=first_dims,
                last_dims=last_dims,
                original_shape=original_shape,
                rowwise_has_rht_applied=rowwise_has_rht_applied,
                colwise_has_rht_applied=colwise_has_rht_applied,
                pre_swizzled=pre_swizzled,
            )

        if q_layout.is_colwise_only:
            return ScaledTensorFactory.create_1x(
                colwise_data,
                colwise_scale_inv,
                colwise_amax if colwise_amax is not None else amax,
                scaling_mode,
                dq_dtype,
                is_colwise=True,
                data_layout=data_layout[0],
                flatten_axis=flatten_axis,
                first_dims=first_dims,
                last_dims=last_dims,
                original_shape=original_shape,
                has_rht_applied=colwise_has_rht_applied,
                pre_swizzled=pre_swizzled,
            )

        return ScaledTensorFactory.create_1x(
            data,
            scale_inv,
            amax,
            scaling_mode,
            dq_dtype,
            is_colwise=False,
            data_layout=data_layout[0],
            flatten_axis=flatten_axis,
            first_dims=first_dims,
            last_dims=last_dims,
            original_shape=original_shape,
            has_rht_applied=rowwise_has_rht_applied,
            pre_swizzled=pre_swizzled,
        )


def with_sharding_constraint_by_logical_axes(x, logical_axis_names: Tuple[str, ...]):
    """Applies sharding constraints to a tensor based on logical axis names.

    Args:
        x: The tensor to apply sharding constraints to
        logical_axis_names: Tuple of logical axis names for sharding

    Returns:
        The tensor with applied sharding constraints
    """
    if isinstance(x, GroupedScaledTensor1x):
        raise NotImplementedError

    if isinstance(x, AbstractBaseTensor):
        return x.apply_sharding_constraint_by_logical_axes(logical_axis_names)

    return original_with_sharding_constraint_by_logical_axes(x, logical_axis_names)
