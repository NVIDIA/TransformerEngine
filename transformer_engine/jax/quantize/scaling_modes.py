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
        self, data_shape: Tuple[int, ...], is_colwise: bool = False, is_padded: bool = True
    ) -> Tuple[int, ...]:
        """Get the shape for scale tensors.

        Args:
            data_shape: The shape of the tensor being quantized
            is_colwise: Whether the scaling is column-wise
            is_padded: Whether to return padded shape

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
        self, data_shape: Tuple[int, ...], is_colwise: bool = False, is_padded: bool = True
    ) -> Tuple[int, ...]:
        """Get the shape for scale tensors in delayed scaling.

        Args:
            data_shape: The shape of the tensor being scaled
            is_colwise: Whether the scaling is column-wise
            is_padded: Whether to return padded shape

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

    def get_scale_shape(
        self, data_shape: Tuple[int, ...], is_colwise: bool = False, is_padded: bool = True
    ) -> Tuple[int, ...]:
        """Get the shape for scale tensors in block scaling.

        Args:
            data_shape: The shape of the tensor being quantized
            is_colwise: Whether the scaling is column-wise
            is_padded: Whether to return padded shape

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

        seq_layout = len(data_shape) - 2

        assert (
            data_shape[seq_layout] % block_x == 0
        ), f"Input data of shape {data_shape} should be padded by {block_x} in axes={seq_layout}"
        assert (
            data_shape[-1] % block_y == 0
        ), f"Input data of shape {data_shape} should be padded by {block_y} in axis -1"

        # NOTE: this overpads if dim > 2 and dims before seq_layout are greater than 1
        n_block_seq = data_shape[seq_layout] // block_x
        n_block_y = data_shape[-1] // block_y

        n_flat_first_dim = reduce(operator.mul, data_shape[:seq_layout], 1) * n_block_seq

        # Padding
        n_flat_first_dim = ((n_flat_first_dim + alignment_x - 1) // alignment_x) * alignment_x
        n_block_y = ((n_block_y + alignment_y - 1) // alignment_y) * alignment_y

        out_shape = ()
        for i in range(seq_layout):
            d = data_shape[i]
            out_shape += (d,)
            assert n_flat_first_dim % d == 0
            n_flat_first_dim //= d

        out_shape += (n_flat_first_dim, n_block_y)

        return out_shape


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
    NVTE_INVALID_SCALING = 2
    NVTE_NO_SCALING = 3

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

    def get_scale_shape_2x(self, data_shape, is_padded=True) -> Tuple[Tuple[int]]:
        """Get shapes for both row-wise and column-wise scaling.

        Args:
            data_shape: Shape of the data tensor
            is_padded: Whether to use padded shapes

        Returns:
            Tuple of (rowwise_scale_shape, colwise_scale_shape)
        """
        rowwise_scale_shape = self.get_scale_shape(
            data_shape, is_colwise=False, is_padded=is_padded
        )
        colwise_scale_shape = self.get_scale_shape(data_shape, is_colwise=True, is_padded=is_padded)
        return (rowwise_scale_shape, colwise_scale_shape)

    def get_scale_shape(self, data_shape, is_colwise, is_padded=True) -> Tuple[int]:
        """Get the shape for scale tensors in this mode.

        Args:
            data_shape: Shape of the data tensor
            is_colwise: Whether to use column-wise scaling
            is_padded: Whether to use padded shapes

        Returns:
            The shape for scale tensors
        """
        return self._get_impl().get_scale_shape(data_shape, is_colwise, is_padded)

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
