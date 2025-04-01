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

from transformer_engine_jax import QuantizeAxis

from .scaling_modes import ScalingMode
from .dequantizer import Dequantizer
from ..sharding import (
    with_sharding_constraint_by_logical_axes as original_with_sharding_constraint_by_logical_axes,
)

__all__ = [
    "ScaledTensor",
    "ScaledTensor1x",
    "ScaledTensor2x",
    "ScaledTensorFactory",
    "with_sharding_constraint_by_logical_axes",
]


@register_pytree_node_class
@dataclass
class ScaledTensor(ABC):
    """Abstract base class for scaled tensors.

    This class defines the interface for all scaled tensor implementations,
    providing methods for dequantization and accessing row/column-wise components.
    """

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

    @abstractmethod
    def dequantize(self):
        """Dequantizes the tensor back to its original precision.

        Returns:
            The dequantized tensor
        """

    @abstractmethod
    def get_rowwise_tensor(self):
        """Returns the row-wise component of the tensor.

        Returns:
            The row-wise tensor component

        Raises:
            ValueError: If called on a tensor that doesn't support row-wise access
        """

    @abstractmethod
    def get_colwise_tensor(self):
        """Returns the column-wise component of the tensor.

        Returns:
            The column-wise tensor component

        Raises:
            ValueError: If called on a tensor that doesn't support column-wise access
        """


@register_pytree_node_class
@dataclass
class ScaledTensor1x(ScaledTensor):
    """Single-scale quantized tensor implementation.

    This class represents a tensor quantized with a single scaling factor,
    supporting both row-wise and column-wise quantization modes.

    Attributes:
        data: The quantized tensor data
        scale_inv: The inverse scaling factors
        scaling_mode: The scaling mode used for quantization
        dq_dtype: The data type for dequantized values
        _dq_func: The dequantization function
        is_colwise: Whether the tensor uses column-wise quantization
        layout: The layout specification for the tensor
    """

    data: jnp.ndarray
    scale_inv: jnp.ndarray
    scaling_mode: ScalingMode
    dq_dtype: jnp.dtype
    _dq_func: Callable
    is_colwise: bool
    layout: str

    def __post_init__(self):
        """Validates and adjusts the scale_inv shape after initialization.

        Ensures the scale_inv shape matches the expected shape based on the scaling mode
        and quantization direction. Pads the scale_inv if necessary.
        """
        expected_scale_shape = self.scaling_mode.get_scale_shape(
            self.data.shape, self.is_colwise, is_padded=True
        )
        expected_unpadded_scale_shape = self.scaling_mode.get_scale_shape(
            self.data.shape, self.is_colwise, is_padded=False
        )
        if self.scale_inv.shape != expected_scale_shape:
            assert self.scale_inv.shape == expected_unpadded_scale_shape, (
                f"Unexpected scale_inv shape! \nExpect {expected_scale_shape} for padded"
                f" scale_inv or {expected_unpadded_scale_shape} for unpadded scale_inv, got"
                f" {self.scale_inv.shape}"
            )
            pad_width = tuple(
                (0, a - b) for a, b in zip(expected_scale_shape, expected_unpadded_scale_shape)
            )
            # This actually pad scale_inv with nan, should we pad it with 127 directly instead?
            self.scale_inv = jnp.pad(
                self.scale_inv, pad_width=pad_width, mode="constant", constant_values=0
            )

    def tree_flatten(self):
        """Flattens the tensor for JAX tree operations.

        Returns:
            A tuple containing (children, aux_data) for tree operations
        """
        children = (self.data, self.scale_inv)
        aux_data = (self.scaling_mode, self.dq_dtype, self._dq_func, self.is_colwise, self.layout)
        return (children, aux_data)

    def dequantize(self):
        """Dequantizes the tensor using the stored dequantization function.

        Returns:
            The dequantized tensor
        """
        return self._dq_func(self)

    def get_rowwise_tensor(self):
        """Returns the tensor if it's row-wise quantized.

        Returns:
            The row-wise tensor

        Raises:
            ValueError: If called on a column-wise quantized tensor
        """
        if not self.is_colwise:
            return self

        raise ValueError("Calling get_rowwise_tensor() from a colwise ScaledTensor1x!")

    def get_colwise_tensor(self):
        """Returns the tensor if it's column-wise quantized.

        Returns:
            The column-wise tensor

        Raises:
            ValueError: If called on a row-wise quantized tensor
        """
        if self.is_colwise:
            return self

        raise ValueError("Calling get_colwise_tensor() from a rowwise ScaledTensor1x!")


@register_pytree_node_class
@dataclass
class ScaledTensor2x(ScaledTensor):
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

    def dequantize(self):
        """Dequantizes the tensor using the row-wise component's dequantization.

        Returns:
            The dequantized tensor
        """
        return self.rowwise_tensor.dequantize()

    def get_rowwise_tensor(self):
        """Returns the row-wise quantized component.

        Returns:
            The row-wise tensor component
        """
        return self.rowwise_tensor

    def get_colwise_tensor(self):
        """Returns the column-wise quantized component.

        Returns:
            The column-wise tensor component
        """
        return self.colwise_tensor


@dataclass
class ScaledTensorFactory:
    """Factory class for creating scaled tensor instances.

    Provides static methods to create both single-scale (1x) and double-scale (2x)
    quantized tensors with various configurations.
    """

    @staticmethod
    def create_1x(
        data, scale_inv, scaling_mode, dq_dtype=jnp.bfloat16, is_colwise=False, layout="N"
    ):
        """Creates a single-scale quantized tensor.

        Args:
            data: The quantized tensor data
            scale_inv: The inverse scaling factors
            scaling_mode: The scaling mode for quantization
            dq_dtype: The data type for dequantized values (default: bfloat16)
            is_colwise: Whether to use column-wise quantization (default: False)
            layout: The layout specification (default: "N")

        Returns:
            A ScaledTensor1x instance
        """
        dq_func = Dequantizer.funcs.get(scaling_mode)
        return ScaledTensor1x(data, scale_inv, scaling_mode, dq_dtype, dq_func, is_colwise, layout)

    @staticmethod
    def create_2x(
        data,
        scale_inv,
        colwise_data,
        colwise_scale_inv,
        scaling_mode,
        dq_dtype=jnp.bfloat16,
        layout="NN",
    ):
        """Creates a double-scale quantized tensor.

        Args:
            data: The row-wise quantized data
            scale_inv: The row-wise inverse scaling factors
            colwise_data: The column-wise quantized data
            colwise_scale_inv: The column-wise inverse scaling factors
            scaling_mode: The scaling mode for quantization
            dq_dtype: The data type for dequantized values (default: bfloat16)
            layout: The layout specification (default: "NN")

        Returns:
            A ScaledTensor2x instance
        """
        dq_func = Dequantizer.funcs.get(scaling_mode)
        rowwise_tensor = ScaledTensor1x(
            data,
            scale_inv,
            scaling_mode,
            dq_dtype,
            dq_func,
            is_colwise=False,
            layout=layout[0],
        )
        colwise_tensor = ScaledTensor1x(
            colwise_data,
            colwise_scale_inv,
            scaling_mode,
            dq_dtype,
            dq_func,
            is_colwise=True,
            layout=layout[1],
        )
        return ScaledTensor2x(rowwise_tensor, colwise_tensor)

    @staticmethod
    def create(
        data: jnp.ndarray,
        scale_inv: jnp.ndarray,
        colwise_data: jnp.ndarray,
        colwise_scale_inv: jnp.ndarray,
        scaling_mode: ScalingMode,
        dq_dtype: jnp.dtype = jnp.bfloat16,
        layout: str = "NN",
        q_axis: QuantizeAxis = QuantizeAxis.ROWWISE,
    ):
        """Creates a scaled tensor based on the quantization axis.

        Args:
            data: The quantized tensor data
            scale_inv: The inverse scaling factors
            colwise_data: The column-wise quantized data
            colwise_scale_inv: The column-wise inverse scaling factors
            scaling_mode: The scaling mode for quantization
            dq_dtype: The data type for dequantized values (default: bfloat16)
            layout: The layout specification (default: "NN")
            q_axis: The quantization axis (default: ROWWISE)

        Returns:
            Either a ScaledTensor1x or ScaledTensor2x instance depending on q_axis
        """
        if q_axis == QuantizeAxis.ROWWISE_COLWISE:
            return ScaledTensorFactory.create_2x(
                data,
                scale_inv,
                colwise_data,
                colwise_scale_inv,
                scaling_mode,
                dq_dtype,
                layout=layout,
            )

        is_colwise = q_axis == QuantizeAxis.COLWISE
        return ScaledTensorFactory.create_1x(
            data, scale_inv, scaling_mode, dq_dtype, is_colwise=is_colwise, layout=layout[0]
        )


def with_sharding_constraint_by_logical_axes(x, logical_axis_names: Tuple[str, ...]):
    """Applies sharding constraints to a tensor based on logical axis names.

    Args:
        x: The tensor to apply sharding constraints to
        logical_axis_names: Tuple of logical axis names for sharding

    Returns:
        The tensor with applied sharding constraints
    """
    if isinstance(x, ScaledTensor1x):
        return ScaledTensor1x(
            data=with_sharding_constraint_by_logical_axes(x.data, logical_axis_names),
            scale_inv=x.scale_inv,
            scaling_mode=x.scaling_mode,
            dq_dtype=x.dq_dtype,
            _dq_func=x._dq_func,
            is_colwise=x.is_colwise,
            layout=x.layout,
        )
    if isinstance(x, ScaledTensor2x):
        return ScaledTensor2x(
            rowwise_tensor=with_sharding_constraint_by_logical_axes(
                x.rowwise_tensor, logical_axis_names
            ),
            colwise_tensor=with_sharding_constraint_by_logical_axes(
                x.colwise_tensor, logical_axis_names
            ),
        )

    return original_with_sharding_constraint_by_logical_axes(x, logical_axis_names)
