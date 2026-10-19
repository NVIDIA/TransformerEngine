# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


def _unswizzle_mxfp8_grouped_scale(scale_inv_flat, padded_scale_2d, is_colwise):
    """Un-swizzle MXFP8 GEMM-swizzled scale_inv back to plain layout.

    Both V1 and V2 MXFP8 grouped quantize produce scale_inv in a GEMM-swizzled
    layout.  This is the inverse of ``swizzled_scale`` in ``gemm.py``.

    The swizzle pattern (for rowwise) is:
        reshape(R//128, 4, 32, C//4, 4) → transpose(0,3,2,1,4) → reshape(R, C)
    The inverse is:
        reshape(R//128, C//4, 32, 4, 4) → transpose(0,3,2,1,4) → reshape(R, C)

    For colwise the swizzle is applied to the transposed scale, so the inverse
    must un-transpose as well.
    """
    if is_colwise:
        # Colwise forward: reshape_2d → transpose → swizzle_5d → reshape_original
        # Inverse: reshape_to_5d → inverse_swizzle → reshape_to_transposed_2d → transpose
        cols, rows = padded_scale_2d
        scale_2d = scale_inv_flat.reshape(cols, rows)
        # The swizzled data lives in the transposed (rows, cols) domain
        reshaped = scale_2d.reshape(rows // 128, cols // 4, 32, 4, 4)
        unswizzled = jnp.transpose(reshaped, (0, 3, 2, 1, 4))
        # Back to transposed 2D, then un-transpose
        return jnp.transpose(unswizzled.reshape(rows, cols))

    rows, cols = padded_scale_2d
    reshaped = scale_inv_flat.reshape(rows // 128, cols // 4, 32, 4, 4)
    unswizzled = jnp.transpose(reshaped, (0, 3, 2, 1, 4))
    return unswizzled.reshape(rows, cols)


def _grouped_dequantize(grouped_scaled_tensor):
    """Dequantize a grouped tensor.

    Args:
        grouped_scaled_tensor: The grouped scaled tensor to dequantize

    Returns:
        List of dequantized tensors for each group
    """
    data = grouped_scaled_tensor.data
    scale_inv = grouped_scaled_tensor.scale_inv
    group_sizes = (
        grouped_scaled_tensor.first_dims
        if grouped_scaled_tensor.first_dims is not None
        and grouped_scaled_tensor.first_dims.size > 0
        else grouped_scaled_tensor.last_dims
    )
    # For non-ragged groups (kernel case), group_sizes is not stored; derive from original_shape
    if group_sizes is None:
        group_sizes = jnp.ones(grouped_scaled_tensor.original_shape[0], dtype=jnp.int32)
    flatten_axis = grouped_scaled_tensor.flatten_axis
    scaling_mode = grouped_scaled_tensor.scaling_mode
    original_shape = grouped_scaled_tensor.original_shape
    flatten_axis = len(original_shape) + flatten_axis if flatten_axis < 0 else flatten_axis

    output = []
    # When data_layout=="T" (colwise, transposed) and first_dims is set (ragged groups), the
    # original_shape is stored transposed: the group (variable-size) axis is the LAST dimension
    # rather than the first. Non-group dims are original_shape[:-1], not original_shape[1:].
    is_transposed_ragged = (
        grouped_scaled_tensor.data_layout == "T"
        and grouped_scaled_tensor.first_dims is not None
        and grouped_scaled_tensor.first_dims.size > 0
    )
    if is_transposed_ragged:
        non_group_shape = original_shape[:-1]
    else:
        non_group_shape = tuple(original_shape[i] for i in range(len(original_shape)) if i != 0)
    matrix_sizes = group_sizes * math.prod(non_group_shape)

    data = jnp.split(data, jnp.cumulative_sum(matrix_sizes)[:-1])

    scale_inv_ptr = 0
    for i, data_i in enumerate(data):
        if is_transposed_ragged:
            data_shape_i = (*non_group_shape, int(group_sizes[i]))
        else:
            data_shape_i = (
                group_sizes[i],
                *original_shape[1:],
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
        scale_inv_i = scale_inv[scale_inv_ptr : scale_inv_ptr + math.prod(padded_scale_shape_i)]
        # MXFP8 grouped quantize (both V1 and V2) always produces GEMM-swizzled
        # scales.  Detect by scaling_mode (not pre_swizzled, which is only set for V2
        # to maintain pytree compatibility with the GEMM path).
        is_colwise = grouped_scaled_tensor.is_colwise
        needs_unswizzle = scaling_mode == ScalingMode.MXFP8_1D_SCALING
        if needs_unswizzle:
            flat_data_2d = (
                math.prod(data_shape_i[:flatten_axis]),
                math.prod(data_shape_i[flatten_axis:]),
            )
            padded_2d = scaling_mode.get_scale_shape(
                flat_data_2d, is_colwise=is_colwise, is_padded=True, flatten_axis=1
            )
            unpadded_2d = scaling_mode.get_scale_shape(
                flat_data_2d, is_colwise=is_colwise, is_padded=False, flatten_axis=1
            )
            scale_inv_i = _unswizzle_mxfp8_grouped_scale(scale_inv_i, padded_2d, is_colwise)
            scale_inv_i = jax.lax.slice(scale_inv_i, [0, 0], list(unpadded_2d))
        else:
            scale_inv_i = scale_inv_i.reshape(padded_scale_shape_i)
            scale_inv_i = jax.lax.slice(
                scale_inv_i, [0] * len(unpadded_scale_shape_i), unpadded_scale_shape_i
            )
        dequantizer_type = ScalingModeToDequantizerMap.get(grouped_scaled_tensor.scaling_mode)
        if len(data_i) == 0:
            out_i = []
        else:
            # _dequantize_func is designed for 2D-flattened data.  Flatten the
            # per-group shape to 2D, dequantize, then reshape back.
            flat_shape_i = (
                math.prod(data_shape_i[:flatten_axis]),
                math.prod(data_shape_i[flatten_axis:]),
            )
            out_i = dequantizer_type._dequantize_func(
                data_i.reshape(flat_shape_i),
                scale_inv_i,
                grouped_scaled_tensor.dq_dtype,
                scaling_mode=grouped_scaled_tensor.scaling_mode,
                is_colwise=grouped_scaled_tensor.is_colwise,
                flatten_axis=1,
            )
            out_i = out_i.reshape(data_shape_i)
        output.append(out_i)
        scale_inv_ptr += math.prod(padded_scale_shape_i)

    return output


Dequantizer.grouped_dequantize = _grouped_dequantize
