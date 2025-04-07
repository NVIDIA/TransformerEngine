# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""
Scaling mode implementations for quantization in JAX.

This module provides implementations of different scaling modes for tensor quantization,
including delayed scaling and block scaling strategies.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Tuple, Dict
from functools import reduce
import operator

from jax.tree_util import register_pytree_node_class
import jax.numpy as jnp

__all__ = ["ScalingMode"]


class ScalingModeMetadataImpl(ABC):
    """Base class for scaling mode implementations.

    This abstract class defines the interface for different scaling mode implementations,
    providing methods to get scale data types and shapes.
    """

    @abstractmethod
    def get_scale_dtype(self) -> jnp.dtype:
        """Get the data type for scale tensors.

        Returns:
            The data type used for scale tensors
        """

    @abstractmethod
    def get_scale_shape(
        self,
        data_shape: Tuple[int, ...],
        is_colwise: bool = False,
        is_padded: bool = True,
        flatten_axis: int = -1,
    ) -> Tuple[int, ...]:
        """Get the shape for scale tensors.

        Args:
            data_shape: The shape of the tensor being quantized
            is_colwise: Whether the scaling is column-wise
            is_padded: Whether to return padded shape
            flatten_axis: Axis along which data can be flattened to 2D for quantization. Defaults to -1.
        Returns:
            The shape for scale tensors
        """


class DelayedScalingModeMetadataImpl(ScalingModeMetadataImpl):
    """Implementation for delayed scaling mode.

    This implementation provides metadata for delayed scaling mode, including scale data type and shape.
    """

    def get_scale_dtype(self) -> jnp.dtype:
        """Get the data type for scale tensors in delayed scaling.

        Returns:
            The data type used for scale tensors (float32)
        """
        return jnp.float32

    def get_scale_shape(
        self,
        data_shape: Tuple[int, ...],
        is_colwise: bool = False,
        is_padded: bool = True,
        flatten_axis: int = -1,
    ) -> Tuple[int, ...]:
        """Get the shape for scale tensors in delayed scaling.

        Args:
            data_shape: The shape of the tensor being scaled
            is_colwise: Whether the scaling is column-wise
            is_padded: Whether to return padded shape
            flatten_axis: Axis along which data can be flattened to 2D for quantization. Defaults to -1.

        Returns:
            The shape for scale tensors - (1,)
        """
        del data_shape, is_colwise
        return (1,)


class BlockScalingModeMetadataImpl(ScalingModeMetadataImpl):
    """Implementation for block scaling mode.

    This implementation provides metadata for block scaling mode, which uses
    block-based scaling with specific alignment requirements.

    Attributes:
        _block_dims: Dimensions of the scaling blocks
        _block_alignment: Alignment requirements for blocks
    """

    def __init__(self, block_dims: Tuple[int]):
        """Initialize block scaling mode implementation.

        Args:
            block_dims: Dimensions of the scaling blocks
        """
        self._block_dims = block_dims
        self._block_alignment = (128, 4)

    def get_scale_dtype(self) -> jnp.dtype:
        """Get the data type for scale tensors in block scaling.

        Returns:
            The data type used for scale tensors (float8_e8m0fnu)
        """
        return jnp.float8_e8m0fnu

    def _apply_scale_shape_correction(self, data_shape, n_scale_blocks, scale_block_dim):
        """Remove excess padding from the scale shape and return the shape with respect to the original data shape."""
        if len(data_shape) > 1:
            # handle last dim
            assert data_shape[-1] % scale_block_dim == 0
            last = data_shape[-1] // scale_block_dim
            scale_shape = (last,)
            assert n_scale_blocks % last == 0
            n_scale_blocks //= last
            # handle middle dim, exclude first and last
            for mid in reversed(data_shape[1:-1]):
                scale_shape = (mid,) + scale_shape
                assert n_scale_blocks % mid == 0
                n_scale_blocks //= mid
            scale_shape = (n_scale_blocks,) + scale_shape
        else:
            scale_shape = (n_scale_blocks,)

        assert len(scale_shape) == len(
            data_shape
        ), f"scale_shape {scale_shape}, data_shape {data_shape}"
        return scale_shape

    def get_scale_shape(
        self,
        data_shape: Tuple[int, ...],
        is_colwise: bool = False,
        is_padded: bool = True,
        flatten_axis: int = -1,
    ) -> Tuple[int, ...]:
        """Get the shape for scale tensors in block scaling.

        Args:
            data_shape: The shape of the tensor being quantized
            is_colwise: Whether the scaling is column-wise
            is_padded: Whether to return padded shape
            flatten_axis: Axis along which data can be flattened to 2D for quantization. Defaults to -1.

        Returns:
            The shape for scale tensors
        """
        block_alignment = self._block_alignment if is_padded else (1, 1)

        if is_colwise:
            block_y, block_x = self._block_dims
            alignment_y, alignment_x = block_alignment
        else:
            block_x, block_y = self._block_dims
            alignment_x, alignment_y = block_alignment

        if flatten_axis < 0:
            flatten_axis = len(data_shape) + flatten_axis
        assert (
            0 < flatten_axis < len(data_shape)
        ), f"flatten_axis {flatten_axis} is out of bounds for shape {data_shape}"

        assert data_shape[flatten_axis - 1] % block_x == 0, (
            f"Data shape {data_shape} should be divisible by block_x {block_x} in axis"
            f" {flatten_axis - 1}"
        )
        assert (
            data_shape[-1] % block_y == 0
        ), f"Data shape {data_shape} should be divisible by block_y {block_y} in axis -1"

        flattened_first_dim = reduce(operator.mul, data_shape[:flatten_axis], 1)
        flattened_last_dim = reduce(operator.mul, data_shape[flatten_axis:], 1)

        assert flattened_first_dim % block_x == 0, (
            f"Flattened first dim - mutiplication of axes={tuple(range(0, flatten_axis))} of shape"
            f" {data_shape} - should be divisible by block_x {block_x}"
        )
        assert flattened_last_dim % block_y == 0, (
            "Flattened last dim - mutiplication of"
            f" axes={tuple(range(flatten_axis, len(data_shape)))} of shape {data_shape} - should be"
            f" divisible by block_y {block_y}"
        )

        n_block_x = int(flattened_first_dim / block_x)
        n_block_y = int(flattened_last_dim / block_y)

        # padding
        n_block_x = int(((n_block_x + alignment_x - 1) // alignment_x) * alignment_x)
        n_block_y = int(((n_block_y + alignment_y - 1) // alignment_y) * alignment_y)

        first_dim_scale_shape = self._apply_scale_shape_correction(
            data_shape[:flatten_axis], n_block_x, block_x
        )
        last_dim_scale_shape = self._apply_scale_shape_correction(
            data_shape[flatten_axis:], n_block_y, block_y
        )

        return (*first_dim_scale_shape, *last_dim_scale_shape)


# (Phuong: Map the NVTEScalingMode value to the ScalingMode


@dataclass(frozen=True)
@register_pytree_node_class
class ScalingMode(Enum):
    """Enumeration of tensor scaling modes with their corresponding metadata implementations.

    This class defines the available scaling modes for tensor quantization:
    - NVTE_DELAYED_TENSOR_SCALING: Uses delayed scaling with FP8 data type and float32 scales
    - NVTE_MXFP8_1D_SCALING: Uses block-based scaling with FP8 data type and E8M0 scales
    - NVTE_INVALID_SCALING: Invalid scaling mode
    - NVTE_NO_SCALING: No scaling applied
    """

    NVTE_DELAYED_TENSOR_SCALING = 0
    NVTE_MXFP8_1D_SCALING = 1
    NVTE_INVALID_SCALING = 100
    NVTE_NO_SCALING = 1000

    def _get_impl(self) -> ScalingModeMetadataImpl:
        """Get the implementation for this scaling mode.

        Returns:
            The scaling mode implementation

        Raises:
            ValueError: If the scaling mode is invalid
        """
        impl = SCALING_MODES_TO_IMPL.get(self)
        if impl is None:
            raise ValueError("Invalid scaling mode")
        return impl

    def get_scale_dtype(self):
        """Get the data type for scale tensors in this mode.

        Returns:
            The data type for scale tensors
        """
        return self._get_impl().get_scale_dtype()

    def get_scale_shape_2x(self, data_shape, is_padded=True, flatten_axis=-1) -> Tuple[Tuple[int]]:
        """Get shapes for both row-wise and column-wise scaling.

        Args:
            data_shape: Shape of the data tensor
            is_padded: Whether to use padded shapes
            flatten_axis: Axis along which data can be flattened to 2D for quantization. Defaults to -1.

        Returns:
            Tuple of (rowwise_scale_shape, colwise_scale_shape)
        """
        rowwise_scale_shape = self.get_scale_shape(
            data_shape, is_colwise=False, is_padded=is_padded, flatten_axis=flatten_axis
        )
        colwise_scale_shape = self.get_scale_shape(
            data_shape, is_colwise=True, is_padded=is_padded, flatten_axis=flatten_axis
        )
        return (rowwise_scale_shape, colwise_scale_shape)

    def get_scale_shape(
        self, data_shape, is_colwise, is_padded=True, flatten_axis=-1
    ) -> Tuple[int]:
        """Get the shape for scale tensors in this mode.

        Args:
            data_shape: Shape of the data tensor
            is_colwise: Whether to use column-wise scaling
            is_padded: Whether to use padded shapes
            flatten_axis: Axis along which data can be flattened to 2D for quantization. Defaults to -1.

        Returns:
            The shape for scale tensors
        """
        return self._get_impl().get_scale_shape(data_shape, is_colwise, is_padded, flatten_axis)

    def __eq__(self, other):
        """Compare this scaling mode with another.

        Args:
            other: The other scaling mode to compare with

        Returns:
            True if the modes are equal, False otherwise
        """
        if not isinstance(other, ScalingMode):
            return False
        return self.value == other.value

    def tree_flatten(self):
        """Flatten this scaling mode for JAX tree operations.

        Returns:
            Tuple of (children, aux_data) for tree operations
        """
        return (), (self.value)

    @classmethod
    def tree_unflatten(cls, aux_data, _children):
        """Reconstruct a scaling mode from its flattened representation.

        Args:
            aux_data: Auxiliary data containing the mode value
            _children: Unused children data

        Returns:
            A reconstructed ScalingMode instance
        """
        return cls(aux_data)


SCALING_MODES_TO_IMPL: Dict[ScalingMode, ScalingModeMetadataImpl] = {
    ScalingMode.NVTE_DELAYED_TENSOR_SCALING: DelayedScalingModeMetadataImpl(),
    ScalingMode.NVTE_MXFP8_1D_SCALING: BlockScalingModeMetadataImpl(block_dims=(1, 32)),
    # WAR
    ScalingMode.NVTE_NO_SCALING: DelayedScalingModeMetadataImpl(),
}
