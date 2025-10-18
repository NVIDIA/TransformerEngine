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
from functools import reduce, lru_cache
import operator
import numpy as np

from jax.experimental.custom_partitioning import BATCHING, CompoundFactor
from jax.tree_util import register_pytree_node_class
import jax.numpy as jnp

from transformer_engine_jax import JAXX_Scaling_Mode, QuantizeLayout
from .device_utils import is_fp8_gemm_with_all_layouts_supported


__all__ = [
    "QuantizeShardyRules",
    "ScalingMode",
    "TensorUsage",
]


class TensorUsage(Enum):
    """Enum indicating tensor usage in GEMM operations.

    Given a GEMM operation: C = A * B in which A and B can be in the normal or transposed form.
    The tensor usage can be:
    - LHS: A is in the normal form
    - LHS_TRANS: A is in the transposed form
    - RHS: B is in the normal form
    - RHS_TRANS: B is in the transposed form

    The tensor usage is used in the ScaledTensor.get_tensor() method.
    """

    # LHS: Left-hand side, RHS: Right-hand side
    # LHS_TRANS: Left-hand side transposed, RHS_TRANS: Right-hand side transposed
    LHS = 0
    LHS_TRANS = 1
    RHS = 2
    RHS_TRANS = 3

    def __eq__(self, other):
        if not isinstance(other, TensorUsage):
            return False
        return self.value == other.value

    def __hash__(self):
        return hash(self.value)


def DIVUP(a, b):
    "Divide a by b and then round up"
    return -(a // -b)


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
    def get_data_layout(self) -> str:
        """Get the data layout for rowwise and colwise scaling.

        Returns:
            The data layout, two characters, e.g. "NT", where each is either "N" (default) or "T" for transposed. The first character refers to the rowwise layout and the second refers to the colwise layout.
        """

    @abstractmethod
    def get_scale_shape(
        self,
        data_shape: Tuple[int, ...],
        data_layout: str = "N",
        is_colwise: bool = False,
        is_padded: bool = True,
        flatten_axis: int = -1,
    ) -> Tuple[int, ...]:
        """Get the shape for scale tensors.

        Args:
            data_shape: The shape of the tensor being quantized
            data_layout: Layout of the data shape, either "N" (default) or "T" for transposed.
            is_colwise: Whether the scaling is column-wise
            is_padded: Whether to return padded shape
            flatten_axis: The axis along which the tensor could be flattened to 2D (default: -1)

        Returns:
            The shape for scale tensors
        """

    @abstractmethod
    def get_grouped_scale_shape(
        self, data_shape, n_groups, group_axis, is_colwise, is_padded=True, flatten_axis=-1
    ) -> Tuple[int]:
        """Get the shape for scale tensors in this mode.

        Args:
            data_shape: Original shape of the data tensor
            n_groups: Number of groups in grouped quantization
            group_axis: The axis along which grouping is performed
            is_colwise: Whether to use column-wise scaling
            is_padded: Whether to use padded shapes
            flatten_axis: The axis along which the tensor could be flattened to 2D (default: -1)

        Returns:
            The shape for scale tensors
        """

    @lru_cache(maxsize=4)
    @abstractmethod
    def get_quantize_layout(self, usage: TensorUsage) -> QuantizeLayout:
        """Get the quantize layout for the tensor usage.

        Args:
            usage: The usage of the tensor

        Returns:
            The quantize layout for the tensor usage
        """

    @abstractmethod
    def get_shardy_sharding_rules(
        self,
        input_shape,
        unique_var,
        flatten_axis,
        broadcast_2d_scale_shape_to_1d,
    ) -> QuantizeShardyRules:
        """Sharding rules for the input and (row, col)wise scale tensors.

        Args:
            input_shape: The shape of the input tensor (for which we produce the scale tensor)
            unique_var: An otherwise unused Shardy variable name prefix
            flatten_axis: Axis along which data can be flattened to 2D for quantization
            broadcast_2d_scale_shape_to_1d: Whether to broadcast the 2D scale shape to 1D.

        Returns:
            The Shardy rules for the scaling mode
        """


class NoScalingModeMetadataImpl(ScalingModeMetadataImpl):
    """Implementation for no scaling mode.

    This implementation provides metadata for no scaling mode, for using non-quantized higher-precision datatypes such as bf16.
    """

    def get_scale_dtype(self) -> jnp.dtype:
        """Get the data type for scale tensors. This is a placeholder and won't be used for higher-precision values that don't have scaling.

        Returns:
            The data type used for scale tensors (float32)
        """
        return jnp.float32

    def get_data_layout(self) -> str:
        """Get the data layout for rowwise and colwise scaling.

        Returns:
            The data layout, two characters, e.g. "NT", where each is either "N" (default) or "T" for transposed. The first character refers to the rowwise layout and the second refers to the colwise layout.
        """
        return "NN"

    def get_scale_shape(
        self,
        data_shape: Tuple[int, ...],
        data_layout: str = "N",
        is_colwise: bool = False,
        is_padded: bool = True,
        flatten_axis: int = -1,
        broadcast_2d_scale_shape_to_1d: bool = True,
    ) -> Tuple[int, ...]:
        """Get the shape for scale tensors. This always returns an empty shape because this mode applies no scaling.

        Args:
            data_shape: The shape of the tensor being scaled
            is_colwise: Whether the scaling is column-wise
            is_padded: Whether to return padded shape
            flatten_axis: Axis along which data can be flattened to 2D for quantization. Defaults to -1.

        Returns:
            The shape for scale tensors - (1,)
        """
        del (
            data_shape,
            data_layout,
            is_colwise,
            is_padded,
            flatten_axis,
            broadcast_2d_scale_shape_to_1d,
        )
        return (0,)

    @lru_cache(maxsize=4)
    def get_quantize_layout(self, usage: TensorUsage) -> QuantizeLayout:
        """Get the quantize layout for the tensor usage.

        Args:
            usage: The usage of the tensor

        Returns:
            The quantize layout for the tensor usage
        """
        return QuantizeLayout.ROWWISE

    def get_grouped_scale_shape(
        self, data_shape, n_groups, group_axis, is_colwise, is_padded=True, flatten_axis=-1
    ) -> Tuple[int]:
        """Get the shape for scale tensors in this mode.

        Args:
            data_shape: Original shape of the data tensor
            is_colwise: Whether to use column-wise scaling
            is_padded: Whether to use padded shapes
            flatten_axis: Axis along which data can be flattened to 2D for quantization. Defaults to -1.

        Returns:
            The shape for scale tensors
        """
        del data_shape, group_axis, is_colwise
        assert isinstance(n_groups, int)
        return (n_groups,)

    def get_shardy_sharding_rules(
        self,
        input_shape,
        unique_var,
        flatten_axis,
        broadcast_2d_scale_shape_to_1d,
    ) -> QuantizeShardyRules:
        """Sharding rules for the input and (row, col)wise scale tensors.

        Args:
            input_shape: The shape of the input tensor (for which we produce the scale tensor)
            unique_var: An otherwise unused Shardy variable name prefix
            flatten_axis: Axis along which data can be flattened to 2D for quantization
            broadcast_2d_scale_shape_to_1d: Whether to broadcast the 2D scale shape to 1D.

        Returns:
            The Shardy rules for the scaling mode
        """
        del flatten_axis, broadcast_2d_scale_shape_to_1d
        input_spec = tuple(f"{unique_var}{i}" for i in range(len(input_shape)))
        scale_var = BATCHING + unique_var + "_scale_inv"
        return QuantizeShardyRules(input_spec, (scale_var,), (scale_var,), {})


class CurrentScalingModeMetadataImpl(ScalingModeMetadataImpl):
    """Implementation for current scaling mode.

    This implementation provides metadata for current scaling mode, including scale data type and shape.
    """

    def get_scale_dtype(self) -> jnp.dtype:
        """Get the data type for scale tensors in delayed scaling.

        Returns:
            The data type used for scale tensors (float32)
        """
        return jnp.float32

    def get_data_layout(self) -> str:
        """Get the data layout for rowwise and colwise scaling.

        Returns:
            The data layout, two characters, e.g. "NT", where each is either "N" (default) or "T" for transposed. The first character refers to the rowwise layout and the second refers to the colwise layout.
        """
        return "NT"

    def get_scale_shape(
        self,
        data_shape: Tuple[int, ...],
        data_layout: str = "N",
        is_colwise: bool = False,
        is_padded: bool = True,
        flatten_axis: int = -1,
        broadcast_2d_scale_shape_to_1d: bool = True,
    ) -> Tuple[int, ...]:
        """Get the shape for scale tensors in delayed scaling.

        Args:
            data_shape: The shape of the tensor being scaled
            data_layout: Layout of the data shape, either "N" (default) or "T" for transposed.
            is_colwise: Whether the scaling is column-wise
            is_padded: Whether to return padded shape
            flatten_axis: Axis along which data can be flattened to 2D for quantization. Defaults to -1.
            broadcast_2d_scale_shape_to_1d: Whether to broadcast the 2D scale shape to 1D. Defaults to True.

        Returns:
            The shape for scale tensors - (1,)
        """
        del data_layout, is_colwise, broadcast_2d_scale_shape_to_1d
        if np.prod(data_shape) == 0:
            return (0,)
        return (1,)

    @lru_cache(maxsize=4)
    def get_quantize_layout(self, usage: TensorUsage) -> QuantizeLayout:
        """Get the quantize layout for the tensor usage.

        Args:
            usage: The usage of the tensor

        Returns:
            The quantize layout for the tensor usage
        """
        if is_fp8_gemm_with_all_layouts_supported():
            return QuantizeLayout.ROWWISE

        if usage in (TensorUsage.LHS, TensorUsage.RHS_TRANS):
            return QuantizeLayout.ROWWISE
        return QuantizeLayout.COLWISE

    def get_grouped_scale_shape(
        self, data_shape, n_groups, group_axis, is_colwise, is_padded=True, flatten_axis=-1
    ) -> Tuple[int]:
        """Get the shape for scale tensors in this mode.

        Args:
            data_shape: Original shape of the data tensor
            is_colwise: Whether to use column-wise scaling
            is_padded: Whether to use padded shapes
            flatten_axis: Axis along which data can be flattened to 2D for quantization. Defaults to -1.

        Returns:
            The shape for scale tensors
        """
        del data_shape, group_axis, is_colwise
        assert isinstance(n_groups, int)
        return (n_groups,)

    def get_shardy_sharding_rules(
        self,
        input_shape,
        unique_var,
        flatten_axis,
        broadcast_2d_scale_shape_to_1d,
    ) -> QuantizeShardyRules:
        """Sharding rules for the input and (row, col)wise scale tensors.

        Args:
            input_shape: The shape of the input tensor (for which we produce the scale tensor)
            unique_var: An otherwise unused Shardy variable name prefix
            flatten_axis: Axis along which data can be flattened to 2D for quantization
            broadcast_2d_scale_shape_to_1d: Whether to broadcast the 2D scale shape to 1D.

        Returns:
            The Shardy rules for the scaling mode
        """
        del flatten_axis, broadcast_2d_scale_shape_to_1d
        input_spec = tuple(f"{unique_var}{i}" for i in range(len(input_shape)))
        scale_var = BATCHING + unique_var + "_scale_inv"
        return QuantizeShardyRules(input_spec, (scale_var,), (scale_var,), {})


class DelayedScalingModeMetadataImpl(CurrentScalingModeMetadataImpl):
    """Implementation for delayed scaling mode.

    This implementation provides metadata for delayed scaling mode, including scale data type and shape.
    """


class BlockScalingModeMetadataImpl(ScalingModeMetadataImpl):
    """Implementation for block scaling mode.

    This implementation provides metadata for block scaling mode, which uses
    block-based scaling with specific alignment requirements.

    Attributes:
        _block_dims: Dimensions of the scaling blocks
        _block_alignment: Alignment requirements for blocks
    """

    def __init__(self, block_dims: Tuple[int], scale_dtype: jnp.dtype, data_layout: str):
        """Initialize block scaling mode implementation.

        Args:
            block_dims: Dimensions of the scaling blocks
            scale_dtype: Data type of the scale tensor
            data_layout: Layout for rowwise and colwise scaling, two characters, e.g. "NT", where each is either "N" (default) or "T" for transposed. The first character refers to the rowwise layout and the second refers to the colwise layout.
        """
        self._block_dims = block_dims
        self._scale_dtype = scale_dtype
        self._block_alignment = (128, 4)
        self._data_layout = data_layout

    def get_scale_dtype(self) -> jnp.dtype:
        """Get the data type for scale tensors in block scaling.

        Returns:
            The data type used for scale tensors (float8_e8m0fnu)
        """
        return self._scale_dtype

    def get_data_layout(self) -> str:
        """Get the data layout for rowwise and colwise scaling.

        Returns:
            The data layout, two characters, e.g. "NT", where each is either "N" (default) or "T" for transposed. The first character refers to the rowwise layout and the second refers to the colwise layout.
        """
        return self._data_layout

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
        data_layout: str = "N",
        is_colwise: bool = False,
        is_padded: bool = True,
        flatten_axis: int = -1,
        broadcast_2d_scale_shape_to_1d: bool = False,
    ) -> Tuple[int, ...]:
        """Get the shape for scale tensors in block scaling.

        Args:
            data_shape: The shape of the tensor being quantized
            data_layout: Layout of the data shape, either "N" (default) or "T" for transposed.
            is_colwise: Whether the scaling is column-wise
            is_padded: Whether to return padded shape
            flatten_axis: Axis along which data can be flattened to 2D for quantization. Defaults to -1.
            broadcast_2d_scale_shape_to_1d: Whether to broadcast the 2D scale shape to 1D. Defaults to True.

        Returns:
            The shape for scale tensors
        """
        flatten_axis = (len(data_shape) + flatten_axis) % len(data_shape)
        assert (
            0 < flatten_axis < len(data_shape)
        ), f"flatten_axis {flatten_axis} is out of bounds for shape {data_shape}"

        block_alignment = self._block_alignment if is_padded else (1, 1)

        if is_colwise:
            assert data_layout == self._data_layout[1], (
                f"Data layout must match colwise layout, received {data_layout} but expected"
                f" {self._data_layout[1]}"
            )
        else:
            assert data_layout == self._data_layout[0], (
                f"Data layout must match rowwise layout, received {data_layout} but expected"
                f" {self._data_layout[0]}"
            )

        if is_colwise and self._data_layout[1] == "T":
            # TODO(Phuong): rework this hack so that we don't implicitly change is_colwise value
            is_colwise = False  # now rowwise in T is colwise in N
            if flatten_axis < 0:
                flatten_axis = len(data_shape) + flatten_axis
            # flatten_axis is given wrt N layout, convert to T layout
            flatten_axis = len(data_shape) - flatten_axis

        if is_colwise:
            block_y, block_x = self._block_dims
            alignment_y, alignment_x = block_alignment
        else:
            block_x, block_y = self._block_dims
            alignment_x, alignment_y = block_alignment

        is_block_2d = block_x > 1 and block_y > 1
        assert data_shape[flatten_axis - 1] % block_x == 0, (
            f"Data shape {data_shape} should be divisible by block_x {block_x} in axis"
            f" {flatten_axis - 1}"
        )
        assert (
            data_shape[-1] % block_y == 0
        ), f"Data shape {data_shape} should be divisible by block_y {block_y} in axis -1"

        if broadcast_2d_scale_shape_to_1d and is_block_2d:
            block_x = 1

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

    @lru_cache(maxsize=4)
    def get_quantize_layout(self, usage: TensorUsage) -> QuantizeLayout:
        """Get the quantize layout for the tensor usage.

        Args:
            usage: The usage of the tensor

        Returns:
            The quantize layout for the tensor usage
        """
        # If we need to support 1x1x for inference in the future
        # if get_quantize_config().INFERENCE_MODE:
        #     assert usage not in (TensorUsage.LHS_TRANS, TensorUsage.RHS_TRANS), (f"Invalid usage {usage} as we are in MXFP8_1D_SCALING 1x1x (FWD only) mode so no transposed usage is needed!")
        #     if usage == TensorUsage.LHS:
        #         return QuantizeLayout.ROWWISE
        #     return QuantizeLayout.COLWISE

        if usage in (TensorUsage.LHS, TensorUsage.RHS_TRANS):
            return QuantizeLayout.ROWWISE
        return QuantizeLayout.COLWISE

    def get_grouped_scale_shape(
        self, data_shape, n_groups, group_axis, is_colwise, is_padded=True, flatten_axis=-1
    ) -> Tuple[int]:
        """Get the shape for grouped scale tensors in this mode.
        If padded: The estimiated maximal possible shape for grouped scale tensor is return instead.

        Args:
            data_shape: Original shape of the data tensor
            is_colwise: Whether to use column-wise scaling
            is_padded: Whether to use padded shapes
            flatten_axis: Axis along which data can be flattened to 2D for quantization. Defaults to -1.

        Returns:
            The shape for scale tensors
        """
        assert isinstance(n_groups, int)
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

        n_block_x = int(flattened_first_dim // block_x)
        n_block_y = int(flattened_last_dim // block_y)

        """
            Given the scale shape of [M, N], and G groups, and padding alignment (128, 4),
            The worst scenario is when we have (G-1) groups with 1 rows and 1 group with (M-G+1) rows.
            Then:
                max_padded_rows = (G-1) * 128 + DIVUP(M-G+1, 128) * 128
                max_padded_cols = DIVUP(N, 4) * 4
                max_scale_size = max_padded_rows * max_padded_cols
        """
        if is_padded:
            n_block_x = (n_groups - 1) * alignment_x + DIVUP(
                n_block_x - n_groups + 1, alignment_x
            ) * alignment_x
            n_block_y = DIVUP(n_block_y, alignment_y) * alignment_y

        return (n_block_x * n_block_y,)

    def get_shardy_sharding_rules(
        self,
        input_shape,
        unique_var,
        flatten_axis,
        broadcast_2d_scale_shape_to_1d,
    ) -> QuantizeShardyRules:
        """Sharding rules for the input and (row, col)wise scale tensors.

        Args:
            input_shape: The shape of the input tensor (for which we produce the scale tensor)
            unique_var: An otherwise unused Shardy variable name prefix
            flatten_axis: Axis along which data can be flattened to 2D for quantization
            broadcast_2d_scale_shape_to_1d: Whether to broadcast the 2D scale shape to 1D.

        Returns:
            The Shardy rules for the scaling mode
        """
        # TODO(Phuong): to rework the shardy rule to handle transposes after NVFP4 is upstreamed
        input_rank = len(input_shape)
        input_spec = [f"{unique_var}_{i}" for i in range(input_rank)]
        flatten_axis = (flatten_axis + input_rank) % input_rank

        assert (
            self._block_dims[1] != 1
        ), f"Expect 1D rowwise or 2D block. Got _block_dims={self._block_dims}"
        # For 2D block scaling, only support when with broadcast_2d_scale_shape_to_1d
        if self._block_dims[0] != 1:
            assert self._block_dims[0] == self._block_dims[1] and broadcast_2d_scale_shape_to_1d, (
                f"Got broadcast_2d_scale_shape_to_1d={broadcast_2d_scale_shape_to_1d},"
                f" _block_dims={self._block_dims}"
            )

        block_size_1d = self._block_dims[1]

        # We have to use two different factors in the two CompoundFactors because of Shardy
        # verifier requirements, even though they are the same.
        blocksizes = {}
        colwise_var = f"{unique_var}_None"
        rowwise_var = f"{unique_var}_None"
        if not input_shape[-1] == block_size_1d:
            rowwise_var = input_spec[-1] + "_compound"
            input_spec[-1] = CompoundFactor(rowwise_var, "blocksize_x")
            blocksizes["blocksize_x"] = block_size_1d
        if not input_shape[flatten_axis - 1] == block_size_1d:
            colwise_var = input_spec[flatten_axis - 1] + "_compound"
            input_spec[flatten_axis - 1] = CompoundFactor(colwise_var, "blocksize_y")
            blocksizes["blocksize_y"] = block_size_1d

        # The rowwise and colwise scale tensors should be sharded the same way as the input.
        # However, we need to adjust the dimensions where the block scaling factor applies.
        rowwise = input_spec.copy()
        rowwise[-1] = rowwise_var

        colwise = input_spec.copy()
        colwise[flatten_axis - 1] = colwise_var

        return QuantizeShardyRules(
            tuple(input_spec),
            tuple(rowwise),
            tuple(colwise),
            blocksizes,
        )


@dataclass(frozen=True)
@register_pytree_node_class
class ScalingMode(Enum):
    """Enumeration of tensor scaling modes with their corresponding metadata implementations.

    This class defines the available scaling modes for tensor quantization:
    - DELAYED_TENSOR_SCALING: Uses delayed scaling with FP8 data type and float32 scales
    - MXFP8_1D_SCALING: Uses block-based scaling with FP8 data type and E8M0 scales
    - CURRENT_TENSOR_SCALING: Uses current scaling with FP8 data type and float32 scales
    - NVFP4_1D_SCALING: Uses block-based scaling with FP4 data type and E4M3 scales
    - NVFP4_2D_SCALING: Uses block-based scaling with FP4 data type and E4M3 scales
    - NO_SCALING: No scaling applied
    """

    NO_SCALING = JAXX_Scaling_Mode.NO_SCALING
    DELAYED_TENSOR_SCALING = JAXX_Scaling_Mode.DELAYED_TENSOR_SCALING
    MXFP8_1D_SCALING = JAXX_Scaling_Mode.MXFP8_1D_SCALING
    CURRENT_TENSOR_SCALING = JAXX_Scaling_Mode.CURRENT_TENSOR_SCALING
    NVFP4_1D_SCALING = JAXX_Scaling_Mode.NVFP4_1D_SCALING
    NVFP4_2D_SCALING = JAXX_Scaling_Mode.NVFP4_2D_SCALING

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

    def get_scale_shape_2x(
        self, data_shape, is_padded=True, flatten_axis=-1, broadcast_2d_scale_shape_to_1d=False
    ) -> Tuple[Tuple[int]]:
        """Get shapes for both row-wise and column-wise scaling.

        Args:
            data_shape: Shape of the data tensor
            is_padded: Whether to use padded shapes
            flatten_axis: Axis along which data can be flattened to 2D for quantization. Defaults to -1.
            broadcast_2d_scale_shape_to_1d: Whether to broadcast the 2D scale shape to 1D. Defaults to False.

        Returns:
            Tuple of (rowwise_scale_shape, colwise_scale_shape)
        """
        data_layout = self._get_impl().get_data_layout()
        rowwise_layout = data_layout[0]
        assert (
            rowwise_layout == "N"
        ), f"For rowwise layout only 'N' is supported, received {rowwise_layout}"
        colwise_layout = data_layout[1]

        rowwise_scale_shape = self.get_scale_shape(
            data_shape,
            data_layout=rowwise_layout,
            is_colwise=False,
            is_padded=is_padded,
            flatten_axis=flatten_axis,
            broadcast_2d_scale_shape_to_1d=broadcast_2d_scale_shape_to_1d,
        )

        colwise_data_shape = data_shape
        if colwise_layout == "T":
            colwise_data_shape = data_shape[flatten_axis:] + data_shape[:flatten_axis]
        colwise_scale_shape = self.get_scale_shape(
            colwise_data_shape,
            data_layout=colwise_layout,
            is_colwise=True,
            is_padded=is_padded,
            flatten_axis=flatten_axis,
            broadcast_2d_scale_shape_to_1d=broadcast_2d_scale_shape_to_1d,
        )
        return (rowwise_scale_shape, colwise_scale_shape)

    def get_scale_shape(
        self,
        data_shape,
        data_layout="N",
        is_colwise=False,
        is_padded=True,
        flatten_axis=-1,
        broadcast_2d_scale_shape_to_1d=False,
    ) -> Tuple[int]:
        """Get the shape for scale tensors in this mode.

        Args:
            data_shape: Shape of the data tensor
            data_layout: Layout of the data shape, either "N" (default) or "T" for transposed.
            is_colwise: Whether to use column-wise scaling
            is_padded: Whether to use padded shapes
            flatten_axis: Axis along which data can be flattened to 2D for quantization. Defaults to -1.
            broadcast_2d_scale_shape_to_1d: Whether to broadcast the 2D scale shape to 1D. Defaults to False.

        Returns:
            The shape for scale tensors
        """
        return self._get_impl().get_scale_shape(
            data_shape,
            data_layout=data_layout,
            is_colwise=is_colwise,
            is_padded=is_padded,
            flatten_axis=flatten_axis,
            broadcast_2d_scale_shape_to_1d=broadcast_2d_scale_shape_to_1d,
        )

    def get_quantize_layout(self, usage: TensorUsage) -> QuantizeLayout:
        """Get the quantize layout for the tensor usage.

        Args:
            usage: The usage of the tensor

        Returns:
            The quantize layout for the tensor usage
        """
        return self._get_impl().get_quantize_layout(usage)

    def get_shardy_sharding_rules(
        self,
        input_shape,
        unique_var,
        flatten_axis=-1,
        broadcast_2d_scale_shape_to_1d=False,
    ) -> Tuple[Tuple[str]]:
        """Sharding rules for the input and (row, col)wise scale tensors.

        Args:
            input_shape: The shape of the input tensor (for which we produce the scale tensor)
            unique_var: An otherwise unused Shardy variable name prefix
            flatten_axis: Axis along which data can be flattened to 2D for quantization.
            broadcast_2d_scale_shape_to_1d: Whether to broadcast the 2D scale shape to 1D. Defaults to False.

        Returns:
            The Shardy rules for the scaling mode
        """
        return self._get_impl().get_shardy_sharding_rules(
            input_shape, unique_var, flatten_axis, broadcast_2d_scale_shape_to_1d
        )

    def get_grouped_scale_shape_2x(
        self, data_shape, n_groups, group_axis, is_padded=True, flatten_axis=-1
    ) -> Tuple[Tuple[int]]:
        """Get shapes for both row-wise and column-wise scaling.

        Args:
            data_shape: Shape of the data tensor
            n_groups: Number of groups for grouped quantization
            group_axis: The axis along which grouping is performed
            is_padded: Whether to use padded shapes
            flatten_axis: The axis along which the tensor could be flattened to 2D (default: -1)

        Returns:
            Tuple of (rowwise_scale_shape, colwise_scale_shape)
        """
        rowwise_scale_shape = self.get_grouped_scale_shape(
            data_shape,
            n_groups,
            group_axis,
            is_colwise=False,
            is_padded=is_padded,
            flatten_axis=flatten_axis,
        )
        colwise_scale_shape = self.get_grouped_scale_shape(
            data_shape,
            n_groups,
            group_axis,
            is_colwise=True,
            is_padded=is_padded,
            flatten_axis=flatten_axis,
        )
        return (rowwise_scale_shape, colwise_scale_shape)

    def get_grouped_scale_shape(
        self, data_shape, n_groups, group_axis, is_colwise, is_padded=True, flatten_axis=-1
    ) -> Tuple[Tuple[int]]:
        """Get shapes for both row-wise and column-wise scaling.

        Args:
            data_shape: Shape of the data tensor
            is_padded: Whether to use padded shapes
            flatten_axis: Axis along which data can be flattened to 2D for quantization. Defaults to -1.

        Returns:
            Tuple of (rowwise_scale_shape, colwise_scale_shape)
        """
        return self._get_impl().get_grouped_scale_shape(
            data_shape,
            n_groups,
            group_axis,
            is_colwise=is_colwise,
            is_padded=is_padded,
            flatten_axis=flatten_axis,
        )

    def is_tensor_scaling(self) -> bool:
        """Check if this scaling mode is per-tensor scaling.

        Returns:
            True if the scaling mode is tensor scaling, False otherwise
        """
        return self in (
            ScalingMode.DELAYED_TENSOR_SCALING,
            ScalingMode.CURRENT_TENSOR_SCALING,
        )

    def is_1d_block_scaling(self) -> bool:
        """Check if this scaling mode is 1D block scaling.

        Returns:
            True if the scaling mode is 1D block scaling, False otherwise
        """
        # Both 1D and 2D NVFP4 scaling are treated as 1D block scaling since the 2D scales are broadcast to 1D because it is required for the GEMM.
        return self == ScalingMode.MXFP8_1D_SCALING or self.is_nvfp4_scaling

    @property
    def is_block_scaling(self) -> bool:
        """Check if this scaling mode is block scaling.

        Returns:
            True if the scaling mode is block scaling, False otherwise
        """
        # Currently we only have 1D block scaling modes
        return self.is_1d_block_scaling()

    def get_compatible_q_dtypes(self) -> set[jnp.dtype]:
        """Returns a set of compatible quantized data types for this scaling mode.

        Returns:
            A set of compatible quantized data types
        """
        if self in (
            ScalingMode.DELAYED_TENSOR_SCALING,
            ScalingMode.CURRENT_TENSOR_SCALING,
            ScalingMode.MXFP8_1D_SCALING,
        ):
            return {jnp.float8_e5m2, jnp.float8_e4m3fn}
        if self in (ScalingMode.NVFP4_1D_SCALING, ScalingMode.NVFP4_2D_SCALING):
            return {jnp.float4_e2m1fn}
        if self == ScalingMode.NO_SCALING:
            return {jnp.float16, jnp.bfloat16, jnp.float32}
        raise ValueError(f"Invalid scaling mode: {self}")

    @property
    def is_nvfp4_scaling(self) -> bool:
        """Check if this scaling mode is NVFP4 scaling.

        Returns:
            True if the scaling mode is NVFP4 scaling, False otherwise
        """
        return self in (ScalingMode.NVFP4_1D_SCALING, ScalingMode.NVFP4_2D_SCALING)

    @property
    def is_mxfp8_scaling(self) -> bool:
        """Check if this scaling mode is NVFP4 scaling.

        Returns:
            True if the scaling mode is NVFP4 scaling, False otherwise
        """
        return self == ScalingMode.MXFP8_1D_SCALING

    @property
    def is_colwise_transposed(self) -> bool:
        """Check if this scaling mode uses transposed layout for column-wise scaling.

        Returns:
            True if the scaling mode uses transposed layout for column-wise scaling, False otherwise
        """
        return self.is_tensor_scaling() or self.is_nvfp4_scaling

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
    ScalingMode.NO_SCALING: NoScalingModeMetadataImpl(),
    ScalingMode.DELAYED_TENSOR_SCALING: DelayedScalingModeMetadataImpl(),
    ScalingMode.MXFP8_1D_SCALING: BlockScalingModeMetadataImpl(
        block_dims=(1, 32),
        scale_dtype=jnp.float8_e8m0fnu,
        data_layout="NN",
    ),
    ScalingMode.CURRENT_TENSOR_SCALING: CurrentScalingModeMetadataImpl(),
    ScalingMode.NVFP4_1D_SCALING: BlockScalingModeMetadataImpl(
        block_dims=(1, 16),
        scale_dtype=jnp.float8_e4m3fn,
        data_layout="NT",
    ),
    ScalingMode.NVFP4_2D_SCALING: BlockScalingModeMetadataImpl(
        block_dims=(16, 16), scale_dtype=jnp.float8_e4m3fn, data_layout="NT"
    ),
}
