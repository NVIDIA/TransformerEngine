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
from typing import Callable, Tuple
from abc import ABC, abstractmethod

import jax
import jax.numpy as jnp

from .scaling_modes import ScalingMode

__all__ = ["ScalingModeToDequantizerMap"]


@dataclass
class Dequantizer(ABC):

    @abstractmethod
    @staticmethod
    def _dequantize_func(data, scale_inv, dq_dtype, **kwargs):
        pass

    @abstractmethod
    @staticmethod
    def dequantize(scaled_tensor):
        pass

    # TODO (Phuong): use grouped_dequantize with group_size = 1 replacing dequantize
    def grouped_dequantize(grouped_scaled_tensor):
        data = grouped_scaled_tensor.data
        scale_inv = grouped_scaled_tensor.scale_inv
        group_size = grouped_scaled_tensor.group_size
        flatten_axis = grouped_scaled_tensor.flatten_axis
        scaling_mode = grouped_scaled_tensor.scaling_mode

        flatten_axis = len(data.shape) + flatten_axis if flatten_axis < 0 else flatten_axis
        assert (
            0 < flatten_axis < len(data.shape)
        ), f"flatten_axis {flatten_axis} is out of bounds for shape {data.shape}"

        output = []
        data = jnp.split(data, group_size, axis=0)

        scale_inv_ptr = 0
        for data_i in data:
            scale_shape_i = scaling_mode.get_scale_shape(
                data_i.shape,
                grouped_scaled_tensor.is_colwise,
                is_padded=True,
                flatten_axis=flatten_axis,
            )
            scale_shape_i_size = math.prod(scale_shape_i)
            scale_inv_i = scale_inv[scale_inv_ptr : scale_inv_ptr + scale_shape_i_size].reshape(
                scale_shape_i
            )
            out_i = self._dequantize_func(
                data_i,
                scale_inv_i,
                grouped_scaled_tensor.dq_dtype,
                grouped_scaled_tensor.scaling_mode,
                grouped_scaled_tensor.is_colwise,
                grouped_scaled_tensor.flatten_axis,
            )
            output.append(out_i)
            scale_inv_ptr += scale_shape_i_size

        # TODO(Phuong): Stack the output to a single ndarray !?
        return output


class TensorScaleDequantizer(Dequantizer):
    """Encapsulation class for dequantization helpers.

    This class provides static methods for dequantizing tensors that have been
    quantized using different scaling modes. It supports both delayed scaling
    and block scaling modes.
    """

    @staticmethod
    def _dequantize_func(data, scale_inv, dq_dtype):
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
        return Dequantizer._dequantize_func(
            scaled_tensor.data, scaled_tensor.scale_inv, scaled_tensor.dq_dtype
        )


class BlockScaleDequantizer(Dequantizer):

    @staticmethod
    def _dequantize_func(data, scale_inv, dq_dtype, scaling_mode, is_colwise, flatten_axis):

        data = data.astype(jnp.float32)
        scale_inv = scale_inv.view(jnp.uint8).astype(jnp.float32)

        data_shape = data.shape
        flatten_axis = len(data_shape) + flatten_axis if flatten_axis < 0 else flatten_axis
        assert (
            0 < flatten_axis < len(data_shape)
        ), f"flatten_axis {flatten_axis} is out of bounds for shape {data_shape}"
        scale_shape = scaling_mode.get_scale_shape(
            data_shape, is_colwise, is_padded=False, flatten_axis=flatten_axis
        )
        scale_inv = jax.lax.slice(
            scale_inv, [0] * len(scale_shape), scale_shape
        )  # slice out the padding

        data = data.reshape(
            *data_shape[: flatten_axis - 1],
            scale_shape[flatten_axis - 1],
            int(data_shape[flatten_axis - 1] / scale_shape[flatten_axis - 1]),
            *data_shape[flatten_axis:-1],
            scale_shape[-1],
            int(data_shape[-1] / scale_shape[-1]),
        )

        # E8M0 does not have a bit for sign. So 0 - 127 represent negative numbers.
        scale_inv = jnp.expand_dims(scale_inv, axis=(flatten_axis + 2 - 2, -1))
        # E8M0 does not have a bit for sign. So 0 - 127 represent negative numbers.
        return jnp.asarray(data * jnp.power(2, scale_inv - 127), dq_dtype).reshape(data_shape)

    @staticmethod
    def dequantize(scaled_tensor):
        """Dequantize a tensor using block scaling.

        This function dequantizes a tensor that was quantized using block scaling
        by applying the inverse scaling factor to each block of data.

        Args:
            scaled_tensor: The quantized tensor to dequantize

        Returns:
            The dequantized tensor in the specified data type
        """
        return Dequantizer.__dq_func_block_scaling_impl(
            scaled_tensor.data,
            scaled_tensor.scale_inv,
            scaled_tensor.dq_dtype,
            scaled_tensor.scaling_mode,
            scaled_tensor.is_colwise,
            scaled_tensor.flatten_axis,
        )


ScalingModeToDequantizerMap = {
    ScalingMode.DELAYED_TENSOR_SCALING: TensorScaleDequantizer,
    ScalingMode.MXFP8_1D_SCALING: BlockScaleDequantizer,
}
