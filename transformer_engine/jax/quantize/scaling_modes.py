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
import math
from typing import Tuple, Dict
from functools import reduce
import operator

from jax.experimental.custom_partitioning import CompoundFactor
from jax.tree_util import register_pytree_node_class
import jax.numpy as jnp

from transformer_engine_jax import JAXX_Scaling_Mode


__all__ = ["QuantizeShardyRules", "ScalingMode"]


@dataclass
class QuantizeShardyRules:
    """Information necessary to shard scale tensors with Shardy.

    Attributes:
        input_spec: Specification for the input axes
        rowwise_rule: Sharding rule for the row-wise scale tensor, depends on
          the axes in `input_spec`
        colwise_rule: Likewise for the column-wise scale tensor.
        factor_sizes: For block scaling, contains the block size factor, which is
          used in `input_spec`.
    """

    input_spec: Tuple[str]
    rowwise_rule: Tuple[str]
    colwise_rule: Tuple[str]
    factor_sizes: Dict[str, int]


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

    @abstractmethod
    def get_shardy_sharding_rules(
        self, input_rank, unique_var, flatten_axis
    ) -> QuantizeShardyRules:
        """Sharding rules for the input and (row, col)wise scale tensors.

        Args:
            input_rank: The rank of the input tensor (for which we produce the scale tensor)
            unique_var: An otherwise unused Shardy variable name prefix
            flatten_axis: Axis along which data can be flattened to 2D for quantization.

        Returns:
            The Shardy rules for the scaling mode
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

    def get_shardy_sharding_rules(
        self, input_rank, unique_var, flatten_axis
    ) -> QuantizeShardyRules:
        """Sharding rules for the input and (row, col)wise scale tensors.

        Args:
            input_rank: The rank of the input tensor (for which we produce the scale tensor)
            unique_var: An otherwise unused Shardy variable name prefix
            flatten_axis: Axis along which data can be flattened to 2D for quantization.

        Returns:
            The Shardy rules for the scaling mode
        """
        del flatten_axis
        input_spec = tuple(f"x{i}" for i in range(input_rank))
        return QuantizeShardyRules(input_spec, (unique_var,), (unique_var,), {})


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

    def get_shardy_sharding_rules(
        self, input_rank, unique_var, flatten_axis
    ) -> QuantizeShardyRules:
        """Sharding rules for the input and (row, col)wise scale tensors.

        Args:
            input_rank: The rank of the input tensor (for which we produce the scale tensor)
            unique_var: An otherwise unused Shardy variable name prefix

        Returns:
            The Shardy rules for the scaling mode
        """
        input_spec = [f"x{i}" for i in range(input_rank)]

        # We have to use two different factors in the two CompoundFactors because of Shardy
        # verifier requirements, even though they are the same.
        rowwise_var = unique_var
        colwise_var = f"{unique_var}_"
        input_spec[flatten_axis - 1] = CompoundFactor(colwise_var, "block_size_colwise")
        input_spec[-1] = CompoundFactor(rowwise_var, "block_size_rowwise")

        # The rowwise and colwise scale tensors should be sharded the same way as the input.
        # However, we need to adjust the dimensions where the block scaling factor applies.
        rowwise = input_spec.copy()
        rowwise[-1] = rowwise_var

        colwise = input_spec.copy()
        colwise[flatten_axis - 1] = colwise_var

        # This implementation needs to be updated for different block dims.
        assert self._block_dims == (1, 32)

        return QuantizeShardyRules(
            tuple(input_spec),
            tuple(rowwise),
            tuple(colwise),
            {"block_size_rowwise": 32, "block_size_colwise": 32},
        )


@dataclass(frozen=True)
@register_pytree_node_class
class ScalingMode(Enum):
    """Enumeration of tensor scaling modes with their corresponding metadata implementations.

    This class defines the available scaling modes for tensor quantization:
    - DELAYED_TENSOR_SCALING: Uses delayed scaling with FP8 data type and float32 scales
    - MXFP8_1D_SCALING: Uses block-based scaling with FP8 data type and E8M0 scales
    - NO_SCALING: No scaling applied
    """

    NO_SCALING = JAXX_Scaling_Mode.NO_SCALING
    DELAYED_TENSOR_SCALING = JAXX_Scaling_Mode.DELAYED_TENSOR_SCALING
    MXFP8_1D_SCALING = JAXX_Scaling_Mode.MXFP8_1D_SCALING

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

    def get_shardy_sharding_rules(
        self, input_rank, unique_var, flatten_axis=-1
    ) -> Tuple[Tuple[str]]:
        """Sharding rules for the input and (row, col)wise scale tensors.

        Args:
            input_rank: The rank of the input tensor (for which we produce the scale tensor)
            unique_var: An otherwise unused Shardy variable name prefix

        Returns:
            The Shardy rules for the scaling mode
        """
        return self._get_impl().get_shardy_sharding_rules(input_rank, unique_var, flatten_axis)

    def get_grouped_scale_shape_2x(
        self, data_shape, group_sizes, is_padded=True, flatten_axis=-1
    ) -> Tuple[Tuple[int]]:
        """Get shapes for both row-wise and column-wise scaling.

        Args:
            data_shape: Shape of the data tensor
            is_padded: Whether to use padded shapes
            flatten_axis: Axis along which data can be flattened to 2D for quantization. Defaults to -1.

        Returns:
            Tuple of (rowwise_scale_shape, colwise_scale_shape)
        """
        rowwise_scale_shape = self.get_grouped_scale_shape(
            data_shape,
            group_sizes,
            is_colwise=False,
            is_padded=is_padded,
            flatten_axis=flatten_axis,
        )
        colwise_scale_shape = self.get_grouped_scale_shape(
            data_shape, group_sizes, is_colwise=True, is_padded=is_padded, flatten_axis=flatten_axis
        )
        return (rowwise_scale_shape, colwise_scale_shape)

    def get_grouped_scale_shape_from_original_data_shape(
        self, data_shape, group_sizes, is_colwise, is_padded=True, flatten_axis=-1
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
        assert jnp.sum(group_sizes) == data_shape[0]
        grouped_scale_size = 0
        for i in range(group_sizes.size):
            data_shape_i = (group_sizes[i], *data_shape[1:])
            scale_shape_i = self._get_impl().get_scale_shape(
                data_shape_i, is_colwise, is_padded, flatten_axis
            )
            grouped_scale_size += math.prod(scale_shape_i)
        return (grouped_scale_size,)

    def get_grouped_scale_shape_from_flattened_data_shape(
        self, data_shape, group_sizes, other_sizes, is_colwise, is_padded=True, flatten_axis=-1
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
        assert len(data_shape) == 1, f"Expect 1D flattened data_shape, got {data_shape}"
        assert jnp.sum(group_sizes) * math.prod(other_sizes) == data_shape[0]
        grouped_scale_size = 0
        for i in range(group_sizes.size):
            data_shape_i = (group_sizes[i], *other_sizes)
            scale_shape_i = self._get_impl().get_scale_shape(
                data_shape_i, is_colwise, is_padded, flatten_axis
            )
            grouped_scale_size += math.prod(scale_shape_i)
        return (grouped_scale_size,)

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
    ScalingMode.DELAYED_TENSOR_SCALING: DelayedScalingModeMetadataImpl(),
    ScalingMode.MXFP8_1D_SCALING: BlockScalingModeMetadataImpl(block_dims=(1, 32)),
    # WAR
    ScalingMode.NO_SCALING: DelayedScalingModeMetadataImpl(),
}
