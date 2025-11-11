# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""
Tensor classes for TE/JAX

This module provides tensor classes for handling quantized tensors in JAX, including
both single-scale (1x) and double-scale (2x) quantization schemes. It supports
rowwise and colwise quantization modes with proper scaling and dequantization.
"""
from dataclasses import dataclass
from typing import Callable, Tuple
from abc import ABC, abstractmethod

import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class

from transformer_engine_jax import QuantizeLayout

from .scaling_modes import ScalingMode, TensorUsage
from .dequantizer import ScalingModeToDequantizerMap
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
        assert (
            q_layout == QuantizeLayout.ROWWISE
        ), "Only ROWWISE layout is supported for NoScaleTensor"
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
    """

    scale_inv: jnp.ndarray
    scaling_mode: ScalingMode
    dq_dtype: jnp.dtype
    _dq_func: Callable
    is_colwise: bool
    data_layout: str
    flatten_axis: int

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
                is_colwise=self.is_colwise,
                is_padded=False,
                flatten_axis=self.flatten_axis,
            )
            assert self.scale_inv.shape == unpadded_scale_shape, (
                "Unpadded inverse scale factor has wrong shape, expected"
                f" {unpadded_scale_shape} but got {self.scale_inv.shape}."
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
        colwise_usage_valid = q_layout == QuantizeLayout.COLWISE and self.is_colwise
        rowwise_usage_valid = q_layout == QuantizeLayout.ROWWISE and not self.is_colwise

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

        if self.scaling_mode == ScalingMode.MXFP8_1D_SCALING:
            # TODO(Phuong): Handle padding !?
            scale_inv = with_sharding_constraint_by_logical_axes(self.scale_inv, axis_names)
        else:
            scale_inv = self.scale_inv

        return ScaledTensor1x(
            data=data,
            scale_inv=scale_inv,
            amax=self.amax,
            scaling_mode=self.scaling_mode,
            dq_dtype=self.dq_dtype,
            _dq_func=self._dq_func,
            is_colwise=self.is_colwise,
            data_layout=self.data_layout,
            flatten_axis=self.flatten_axis,
        )


@register_pytree_node_class
@dataclass
class GroupedScaledTensor1x(ScaledTensor1x):
    """Grouped Quantizer for an array.

    This class extends ScaledTensor1x to support quantization of an array in grouped manner,
    where elements are grouped along a specified axis.

    Attributes:
        group_sizes: Array containing the size of each group
        original_shape: The original shape of the tensor before grouping
        group_axis: The axis along which grouping is performed (default: 0)
    """

    group_sizes: jnp.ndarray
    original_shape: Tuple
    group_axis: int

    def __init__(
        self,
        data,
        scale_inv,
        amax,
        group_sizes,
        scaling_mode,
        dq_dtype,
        _dq_func,
        is_colwise,
        data_layout,
        flatten_axis,
        original_shape,
        group_axis=0,
    ):
        self.flatten_axis = flatten_axis
        self.group_sizes = group_sizes
        self.original_shape = original_shape
        self.group_axis = group_axis
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
        )

    def __post_init__(self):
        assert self.scale_inv.ndim == 1, "Only support flattened scale_inv"
        assert self.data.ndim == 1, "Only support flattened data"
        assert self.group_axis >= 0
        assert self.flatten_axis > 0

        data_ndim = len(self.original_shape)
        assert (
            0 < self.flatten_axis < data_ndim
        ), f"flatten_axis {self.flatten_axis} is out of bounds for data.ndim = {data_ndim}"

        assert (
            0 <= self.group_axis < data_ndim
        ), f"group_axis {self.group_axis} is out of bounds for shape {self.original_shape}"

        expected_scale_shape = self.scaling_mode.get_grouped_scale_shape(
            self.original_shape,
            self.group_sizes.size,
            self.group_axis,
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
        children = (self.data, self.scale_inv, self.amax, self.group_sizes)
        aux_data = (
            self.scaling_mode,
            self.dq_dtype,
            self._dq_func,
            self.is_colwise,
            self.data_layout,
            self.flatten_axis,
            self.original_shape,
            self.group_axis,
        )
        return (children, aux_data)

    def apply_sharding_constraint_by_logical_axes(self, logical_axis_names: Tuple[str, ...]):
        raise NotImplementedError


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

        if q_layout_rowwise == QuantizeLayout.ROWWISE:
            return self.rowwise_tensor

        if q_layout_colwise == QuantizeLayout.COLWISE:
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
        group_sizes=None,
        original_shape=None,
        group_axis=0,
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
            group_sizes: Array of ints containing the size of each group (default: None)
            original_shape: The original shape of the tensor before grouping (default: None)
            group_axis: The axis along which grouping is performed (default: 0)

        Returns:
            A ScaledTensor1x or GroupedScaledTensor1x instance depending on whether group_sizes is provided
        """
        if amax is None:
            amax = jnp.empty((1,), dtype=jnp.float32)

        dequantizer = ScalingModeToDequantizerMap.get(scaling_mode)

        if group_sizes is not None:
            flatten_axis = len(original_shape) + flatten_axis if flatten_axis < 0 else flatten_axis
            assert (
                original_shape is not None
            ), "original_shape is not given for GroupedScaledTensor1x"

            # Handling attrs of transposed tensors
            group_axis = len(original_shape) + group_axis if group_axis < 0 else group_axis
            if data_layout == "T":
                if original_shape[0] == group_sizes.size:
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
                    group_axis = flatten_axis
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
                group_sizes=group_sizes,
                original_shape=original_shape,
                group_axis=group_axis,
            )

        # Handling attrs of transposed tensors
        flatten_axis = data.ndim + flatten_axis if flatten_axis < 0 else flatten_axis
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
        )

    @staticmethod
    def create_2x(
        data,
        scale_inv,
        colwise_data,
        colwise_scale_inv,
        amax=None,
        scaling_mode=ScalingMode.NO_SCALING,
        dq_dtype=jnp.bfloat16,
        data_layout="NN",
        flatten_axis=-1,
        group_sizes=None,
        original_shape=None,
        group_axis=0,
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
            group_sizes: Array containing the size of each group (default: None)
            original_shape: The original shape of the tensor before grouping (default: None)
            group_axis: The axis along which grouping is performed (default: 0)

        Returns:
            A ScaledTensor2x instance
        """
        if amax is None:
            amax = jnp.empty((1,), dtype=jnp.float32)

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
            group_sizes=group_sizes,
            original_shape=original_shape,
            group_axis=group_axis,
        )
        colwise_tensor = ScaledTensorFactory.create_1x(
            colwise_data,
            colwise_scale_inv,
            amax,
            scaling_mode,
            dq_dtype,
            is_colwise=True,
            data_layout=data_layout[1],
            flatten_axis=flatten_axis,
            group_sizes=group_sizes,
            original_shape=original_shape,
            group_axis=group_axis,
        )
        return ScaledTensor2x(rowwise_tensor, colwise_tensor)

    @staticmethod
    def create(
        data: jnp.ndarray,
        scale_inv: jnp.ndarray,
        colwise_data: jnp.ndarray,
        colwise_scale_inv: jnp.ndarray,
        amax=None,
        scaling_mode: ScalingMode = ScalingMode.NO_SCALING,
        dq_dtype: jnp.dtype = jnp.bfloat16,
        data_layout: str = "NN",
        q_layout: QuantizeLayout = QuantizeLayout.ROWWISE,
        flatten_axis: int = -1,
        group_sizes: jnp.ndarray = None,
        original_shape: Tuple[int] = None,
        group_axis: int = 0,
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
            group_sizes: Array containing the size of each group (default: None)
            original_shape: The original shape of the tensor before grouping (default: None)
            group_axis: The axis along which grouping is performed (default: 0)

        Returns:
            Either a ScaledTensor1x or ScaledTensor2x instance depending on q_layout
        """
        if q_layout == QuantizeLayout.ROWWISE_COLWISE:
            return ScaledTensorFactory.create_2x(
                data,
                scale_inv,
                colwise_data,
                colwise_scale_inv,
                amax,
                scaling_mode,
                dq_dtype,
                data_layout=data_layout,
                flatten_axis=flatten_axis,
                group_sizes=group_sizes,
                original_shape=original_shape,
                group_axis=group_axis,
            )

        is_colwise = q_layout == QuantizeLayout.COLWISE
        if is_colwise:
            return ScaledTensorFactory.create_1x(
                colwise_data,
                colwise_scale_inv,
                amax,
                scaling_mode,
                dq_dtype,
                is_colwise=is_colwise,
                data_layout=data_layout[0],
                flatten_axis=flatten_axis,
                group_sizes=group_sizes,
                original_shape=original_shape,
                group_axis=group_axis,
            )

        return ScaledTensorFactory.create_1x(
            data,
            scale_inv,
            amax,
            scaling_mode,
            dq_dtype,
            is_colwise=is_colwise,
            data_layout=data_layout[0],
            flatten_axis=flatten_axis,
            group_sizes=group_sizes,
            original_shape=original_shape,
            group_axis=group_axis,
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
