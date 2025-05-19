# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""
Dequantization utilities for TE/JAX.

This module provides utilities for dequantizing tensors that have been quantized
using various scaling modes, including delayed scaling and block scaling.
"""
import jax
import jax.numpy as jnp

from .scaling_modes import ScalingMode

__all__ = ["Dequantizer"]


class Dequantizer:
    """Encapsulation class for dequantization helpers.

    This class provides static methods for dequantizing tensors that have been
    quantized using different scaling modes. It supports both delayed scaling
    and block scaling modes.
    """

    @staticmethod
    def _dq_func_tensor_scaling(scaled_tensor):
        """Dequantize a tensor using delayed scaling.

        This function dequantizes a tensor that was quantized using delayed scaling
        by multiplying the quantized data with the inverse scaling factor.

        Args:
            scaled_tensor: The quantized tensor to dequantize

        Returns:
            The dequantized tensor in the specified data type
        """
        return jnp.asarray(
            scaled_tensor.data.astype(jnp.float32) * scaled_tensor.scale_inv.astype(jnp.float32),
            scaled_tensor.dq_dtype,
        )

    @staticmethod
    def _dq_func_block_scaling(scaled_tensor):
        """Dequantize a tensor using block scaling.

        This function dequantizes a tensor that was quantized using block scaling
        by applying the inverse scaling factor to each block of data.

        Args:
            scaled_tensor: The quantized tensor to dequantize

        Returns:
            The dequantized tensor in the specified data type
        """
        data = scaled_tensor.data.astype(jnp.float32)
        data_shape = data.shape
        scale = scaled_tensor.scale_inv.view(jnp.uint8).astype(jnp.float32)
        flatten_axis = scaled_tensor.flatten_axis
        flatten_axis = len(data_shape) + flatten_axis if flatten_axis < 0 else flatten_axis
        assert (
            0 < flatten_axis < len(data_shape)
        ), f"flatten_axis {flatten_axis} is out of bounds for shape {data_shape}"
        scale_shape = scaled_tensor.scaling_mode.get_scale_shape(
            data_shape, scaled_tensor.is_colwise, is_padded=False, flatten_axis=flatten_axis
        )
        scale = jax.lax.slice(scale, [0] * len(scale_shape), scale_shape)  # slice out the padding

        data = data.reshape(
            *data_shape[: flatten_axis - 1],
            scale_shape[flatten_axis - 1],
            int(data_shape[flatten_axis - 1] / scale_shape[flatten_axis - 1]),
            *data_shape[flatten_axis:-1],
            scale_shape[-1],
            int(data_shape[-1] / scale_shape[-1]),
        )

        # E8M0 does not have a bit for sign. So 0 - 127 represent negative numbers.
        scale = jnp.expand_dims(scale, axis=(flatten_axis + 2 - 2, -1))
        # E8M0 does not have a bit for sign. So 0 - 127 represent negative numbers.
        return jnp.asarray(data * jnp.power(2, scale - 127), scaled_tensor.dq_dtype).reshape(
            data_shape
        )

    funcs = {
        ScalingMode.DELAYED_TENSOR_SCALING: _dq_func_tensor_scaling,
        ScalingMode.CURRENT_TENSOR_SCALING: _dq_func_tensor_scaling,
        ScalingMode.MXFP8_1D_SCALING: _dq_func_block_scaling,
    }

    @staticmethod
    def dequantize(scaled_tensor):
        """Dequantize a scaled tensor using the appropriate scaling mode.

        This method selects the appropriate dequantization function based on the
        scaling mode used for quantization and applies it to the tensor.

        Args:
            scaled_tensor: The quantized tensor to dequantize

        Returns:
            The dequantized tensor in the specified data type
        """
        dq_func = Dequantizer.funcs[scaled_tensor.scaling_mode]
        return dq_func(scaled_tensor)
