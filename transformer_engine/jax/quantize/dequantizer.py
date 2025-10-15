# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""
Dequantization utilities for TE/JAX.

This module provides utilities for dequantizing tensors that have been quantized
using various scaling modes, including delayed scaling and block scaling.
"""
import math
from dataclasses import dataclass
from abc import ABC, abstractmethod

import jax
import jax.numpy as jnp

from .scaling_modes import ScalingMode
from .hadamard import apply_rht


__all__ = ["ScalingModeToDequantizerMap"]


@dataclass
class Dequantizer(ABC):
    """
    Base Dequantizer Class
    """

    @staticmethod
    @abstractmethod
    def _dequantize_func(data, scale_inv, dq_dtype, **kwargs):
        pass

    @staticmethod
    @abstractmethod
    def dequantize(scaled_tensor):
        """Dequantizing given tensor to higher precision."""


@dataclass
class NoopDequantizer(Dequantizer):
    """No-op Dequantizer Class"""

    @staticmethod
    def _dequantize_func(data, *args, **kwargs):
        """A no-op dequantize function that returns the data without any changes."""
        del args, kwargs
        return data

    @staticmethod
    def dequantize(scaled_tensor):
        """A no-op dequantize function that simply returns the data array in the ScaledTensor."""
        return scaled_tensor.data


class TensorScaleDequantizer(Dequantizer):
    """
    TensorScaling Dequantizer Class

    This class provides static methods for dequantizing tensors that have been
    quantized using different tensor scaling modes. It supports both delayed scaling
    and current scaling modes.
    """

    @staticmethod
    def _dequantize_func(data, scale_inv, dq_dtype, **kwargs):
        del kwargs
        return jnp.asarray(
            data.astype(jnp.float32) * scale_inv.astype(jnp.float32),
            dq_dtype,
        )

    @staticmethod
    def dequantize(scaled_tensor):
        """Dequantize a tensor using delayed scaling.

        This function dequantizes a tensor that was quantized using delayed scaling
        by multiplying the quantized data with the inverse scaling factor.

        Args:
            scaled_tensor: The quantized tensor to dequantize

        Returns:
            The dequantized tensor in the specified data type
        """
        return TensorScaleDequantizer._dequantize_func(
            scaled_tensor.data, scaled_tensor.scale_inv, scaled_tensor.dq_dtype
        )


class BlockScaleDequantizer(Dequantizer):
    """BlockScaling Dequantizer Class.

    This class provides static methods for dequantizing tensors that have been
    quantized using block scaling modes.
    """

    @staticmethod
    def _dequantize_func(data, scale_inv, dq_dtype, scaling_mode, is_colwise, flatten_axis):
        """Dequantize a tensor using block scaling.

        Args:
            data: The quantized tensor data
            scale_inv: The inverse scaling factors
            dq_dtype: The data type for dequantized values
            scaling_mode: The scaling mode used for quantization
            is_colwise: Whether the scaling is column-wise
            flatten_axis: The axis along which the tensor could be flattened to 2D

        Returns:
            The dequantized tensor
        """

        data = data.astype(jnp.float32)
        scale_inv = scale_inv.view(jnp.uint8).astype(jnp.float32)

        data_shape = data.shape
        flatten_axis = len(data_shape) + flatten_axis if flatten_axis < 0 else flatten_axis
        assert (
            0 < flatten_axis < len(data_shape)
        ), f"flatten_axis {flatten_axis} is out of bounds for shape {data_shape}"
        scale_shape = scaling_mode.get_scale_shape(
            data_shape, is_colwise=is_colwise, is_padded=False, flatten_axis=flatten_axis
        )

        data = data.reshape(
            *data_shape[: flatten_axis - 1],
            scale_shape[flatten_axis - 1],
            int(data_shape[flatten_axis - 1] / scale_shape[flatten_axis - 1]),
            *data_shape[flatten_axis:-1],
            scale_shape[-1],
            int(data_shape[-1] / scale_shape[-1]),
        )

        scale_inv = jnp.expand_dims(scale_inv, axis=(flatten_axis + 2 - 2, -1))

        # E8M0 does not have a bit for sign. So 0 - 127 represent negative numbers.
        return jnp.asarray(data * jnp.power(2, scale_inv - 127), dq_dtype).reshape(data_shape)

    @staticmethod
    def dequantize(scaled_tensor):
        """Dequantize a tensor using block scaling.

        Args:
            data: The quantized tensor data
            scale_inv: The inverse scaling factors
            dq_dtype: The data type for dequantized values
            scaling_mode: The scaling mode used for quantization
            is_colwise: Whether the scaling is column-wise
            flatten_axis: The axis along which the tensor could be flattened to 2D

        Returns:
            The dequantized tensor
        """
        return BlockScaleDequantizer._dequantize_func(
            scaled_tensor.data,
            scaled_tensor.scale_inv,
            scaled_tensor.dq_dtype,
            scaled_tensor.scaling_mode,
            scaled_tensor.is_colwise,
            scaled_tensor.flatten_axis,
        )


class NVFP4Dequantizer(Dequantizer):
    """NVFP4 Dequantizer Class.

    This class provides static methods for dequantizing tensors that have been
    quantized using NVFP4 scaling modes.
    """

    @staticmethod
    def _dequantize_func(
        data, scale_inv, amax, dq_dtype, scaling_mode, is_colwise, flatten_axis, has_rht_applied
    ):
        """Dequantize a tensor using block scaling.

        Args:
            data: The quantized tensor data
            scale_inv: The inverse scaling factors
            amax: The maximum absolute value of the tensor
            dq_dtype: The data type for dequantized values
            scaling_mode: The scaling mode used for quantization
            is_colwise: Whether the scaling is column-wise
            flatten_axis: The axis along which the tensor could be flattened to 2D
            has_rht_applied: Whether the quantization has RHT applied and we need to apply the inverse RHT to dequantize

        Returns:
            The dequantized tensor
        """

        DATA_DTYPE_MAX = jnp.finfo(data.dtype).max.astype(jnp.float32)
        SCALE_DTYPE_MAX = jnp.finfo(scale_inv.dtype).max.astype(jnp.float32)
        tensor_scale_inv = amax / (DATA_DTYPE_MAX * SCALE_DTYPE_MAX)

        data = data.astype(jnp.float32)
        scale_inv = scale_inv.astype(jnp.float32) * tensor_scale_inv
        data_layout = "T" if is_colwise else "N"

        data_shape = data.shape
        flatten_axis = len(data_shape) + flatten_axis if flatten_axis < 0 else flatten_axis
        assert (
            0 < flatten_axis < len(data_shape)
        ), f"flatten_axis {flatten_axis} is out of bounds for shape {data_shape}"
        scale_shape = scaling_mode.get_scale_shape(
            data_shape,
            data_layout=data_layout,
            is_colwise=is_colwise,
            is_padded=False,
            # expect the flatten_axis wrt the N layout
            flatten_axis=flatten_axis if data_layout == "N" else len(data_shape) - flatten_axis,
            broadcast_2d_scale_shape_to_1d=True,
        )

        data = data.reshape(
            *data_shape[: flatten_axis - 1],
            scale_shape[flatten_axis - 1],
            int(data_shape[flatten_axis - 1] / scale_shape[flatten_axis - 1]),
            *data_shape[flatten_axis:-1],
            scale_shape[-1],
            int(data_shape[-1] / scale_shape[-1]),
        )

        scale_inv = jnp.expand_dims(scale_inv, axis=(flatten_axis + 2 - 2, -1))
        out = jnp.asarray(data * scale_inv, dq_dtype).reshape(data_shape)

        # Apply inverse of RHT if needed
        if has_rht_applied:
            out = apply_rht(out, inverse=True)

        return out

    @staticmethod
    def dequantize(scaled_tensor):
        """Dequantize a tensor using block scaling.

        Args:
            scaled_tensor: The quantized tensor to dequantize

        Returns:
            The dequantized tensor
        """
        return NVFP4Dequantizer._dequantize_func(
            scaled_tensor.data,
            scaled_tensor.scale_inv,
            scaled_tensor.amax,
            scaled_tensor.dq_dtype,
            scaled_tensor.scaling_mode,
            scaled_tensor.is_colwise,
            scaled_tensor.flatten_axis,
            scaled_tensor.has_rht_applied,
        )


ScalingModeToDequantizerMap = {
    ScalingMode.DELAYED_TENSOR_SCALING: TensorScaleDequantizer,
    ScalingMode.CURRENT_TENSOR_SCALING: TensorScaleDequantizer,
    ScalingMode.MXFP8_1D_SCALING: BlockScaleDequantizer,
    ScalingMode.NVFP4_1D_SCALING: NVFP4Dequantizer,
    ScalingMode.NVFP4_2D_SCALING: NVFP4Dequantizer,
    ScalingMode.NO_SCALING: NoopDequantizer,
}


@staticmethod
def _grouped_dequantize(grouped_scaled_tensor):
    """Dequantize a grouped tensor.

    Args:
        grouped_scaled_tensor: The grouped scaled tensor to dequantize

    Returns:
        List of dequantized tensors for each group
    """
    data = grouped_scaled_tensor.data
    scale_inv = grouped_scaled_tensor.scale_inv
    group_sizes = grouped_scaled_tensor.group_sizes
    flatten_axis = grouped_scaled_tensor.flatten_axis
    scaling_mode = grouped_scaled_tensor.scaling_mode
    original_shape = grouped_scaled_tensor.original_shape
    group_axis = grouped_scaled_tensor.group_axis

    flatten_axis = len(original_shape) + flatten_axis if flatten_axis < 0 else flatten_axis

    output = []
    non_group_shape = tuple(
        original_shape[i] for i in range(len(original_shape)) if i != group_axis
    )
    matrix_sizes = group_sizes * math.prod(non_group_shape)

    data = jnp.split(data, jnp.cumulative_sum(matrix_sizes)[:-1])

    scale_inv_ptr = 0
    for i, data_i in enumerate(data):
        data_shape_i = (
            *original_shape[:group_axis],
            group_sizes[i],
            *original_shape[group_axis + 1 :],
        )
        assert math.prod(data_shape_i) == data_i.size, (
            f"math.prod({data_shape_i}) = {math.prod(data_shape_i)} which is not equal to"
            f" {data_i.size}"
        )
        padded_scale_shape_i = scaling_mode.get_scale_shape(
            data_shape_i,
            is_colwise=grouped_scaled_tensor.is_colwise,
            is_padded=True,
            flatten_axis=flatten_axis,
        )
        unpadded_scale_shape_i = scaling_mode.get_scale_shape(
            data_shape_i,
            is_colwise=grouped_scaled_tensor.is_colwise,
            is_padded=False,
            flatten_axis=flatten_axis,
        )
        scale_inv_i = scale_inv[
            scale_inv_ptr : scale_inv_ptr + math.prod(padded_scale_shape_i)
        ].reshape(padded_scale_shape_i)
        scale_inv_i = jax.lax.slice(
            scale_inv_i, [0] * len(unpadded_scale_shape_i), unpadded_scale_shape_i
        )
        dequantizer_type = ScalingModeToDequantizerMap.get(grouped_scaled_tensor.scaling_mode)
        if len(data_i) == 0:
            out_i = []
        else:
            out_i = dequantizer_type._dequantize_func(
                data_i.reshape(data_shape_i),
                scale_inv_i,
                grouped_scaled_tensor.dq_dtype,
                scaling_mode=grouped_scaled_tensor.scaling_mode,
                is_colwise=grouped_scaled_tensor.is_colwise,
                flatten_axis=grouped_scaled_tensor.flatten_axis,
            )
        output.append(out_i)
        scale_inv_ptr += math.prod(padded_scale_shape_i)

    return output


Dequantizer.grouped_dequantize = _grouped_dequantize
