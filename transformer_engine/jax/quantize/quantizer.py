# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""
Tensor quantization classes for TE/JAX.

This module provides classes and utilities for quantizing tensors in JAX.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import partial
from typing import Union, Optional, Tuple
import warnings

import jax
import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class
from transformer_engine_jax import QuantizeLayout
from transformer_engine.common import recipe

from .scaling_modes import ScalingMode
from .tensor import (
    ScaledTensor,
    ScaledTensor1x,
    ScaledTensor2x,
    ScaledTensorFactory,
    NoScaleTensor,
)
from .helper import (
    get_quantize_config,
    get_quantize_config_class,
    AmaxComputeAlgo,
    TensorSource,
)
from .device_utils import is_fp8_gemm_with_all_layouts_supported

__all__ = [
    "QuantizeLayout",
    "Quantizer",
    "QuantizerSet",
    "CurrentScaleQuantizer",
    "DelayedScaleQuantizer",
    "BlockScaleQuantizer",
    "GroupedQuantizer",
    "QuantizerFactory",
    "noop_quantizer_set",
    "compute_scale_from_amax",
]


def compute_scale_from_amax(
    amax: jnp.ndarray, q_dtype: jnp.dtype, scale: Optional[jnp.ndarray] = None
) -> jnp.ndarray:
    """Compute scale from amax value.

    Args:
        amax: Maximum absolute value of the tensor
        q_dtype: Quantization data type

    Returns:
        Scale value
    """
    fp8_max = jnp.astype(jnp.finfo(q_dtype).max, jnp.float32)
    if scale is None:
        scale = jnp.ones((1,))
    sf = (fp8_max / amax) / (2 ** get_quantize_config().MARGIN)
    sf = jnp.where(amax > 0.0, sf, scale)
    sf = jnp.where(jnp.isfinite(amax), sf, scale)
    return sf


@register_pytree_node_class
@dataclass
class Quantizer(ABC):
    """Base class for quantizers.

    This abstract class defines the interface for tensor quantization, providing
    methods for quantization and scale management.

    Attributes:
        q_dtype: The data type for quantized values
        scaling_mode: The scaling mode to use for quantization
        q_layout: The quantization axis (row-wise, column-wise, or both)
    """

    q_dtype: jnp.dtype
    scaling_mode: ScalingMode
    q_layout: QuantizeLayout
    data_layout: str

    def tree_flatten(self):
        """Flatten the quantizer for JAX tree operations.

        Returns:
            Tuple of (children, aux_data) for tree operations
        """
        children = ()
        aux_data = (self.q_dtype, self.scaling_mode, self.q_layout, self.data_layout)
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """Reconstruct a quantizer from its flattened representation.

        Args:
            aux_data: Auxiliary data containing quantizer parameters
            children: Unused children data

        Returns:
            A reconstructed Quantizer instance
        """
        return cls(*aux_data, *children)

    def update(self, *args, **kwargs):
        """Update quantizer state (no-op in base class)."""
        del args, kwargs

    def is_2x2x(self) -> bool:
        """Check if quantizer uses both row-wise and column-wise quantization.

        Returns:
            True if using both row-wise and column-wise quantization
        """
        return self.q_layout == QuantizeLayout.ROWWISE_COLWISE

    def get_data_layout(self) -> str:
        """Get the data data_layout string.

        Returns:
            Data data_layout in string format

        Raises:
            ValueError: If quantization axis is invalid
        """
        if self.q_layout == QuantizeLayout.ROWWISE_COLWISE:
            return self.data_layout
        if self.q_layout == QuantizeLayout.ROWWISE:
            return self.data_layout[0]
        if self.q_layout == QuantizeLayout.COLWISE:
            return self.data_layout[1]
        raise ValueError(f"Invalid q_layout: {self.q_layout}")

    @abstractmethod
    def _quantize_func(self, x, is_colwise=False, dq_dtype=None, flatten_axis=-1) -> ScaledTensor1x:
        """Core quantization function to be implemented by subclasses.

        Args:
            x: Input tensor to quantize
            is_colwise: Whether to use column-wise quantization
            dq_dtype: Data type for dequantized values, default is x.dtype
            flatten_axis: The quantization axis for the tensor

        Returns:
            A ScaledTensor1x containing the quantized data
        """

    def quantize(
        self, x, is_rowwise=False, is_colwise=False, dq_dtype=None, flatten_axis=-1, **kwargs
    ) -> ScaledTensor:
        """Quantize a tensor using the internal _quantize_func().

        Args:
            x: Input tensor to quantize
            is_rowwise: Whether to use row-wise quantization
            is_colwise: Whether to use column-wise quantization
            dq_dtype: Data type for dequantized values
            flatten_axis: The quantization axis for the tensor

        Returns:
            A ScaledTensor1x or ScaledTensor2x containing the quantized data
        """
        del kwargs
        if (is_rowwise and is_colwise) or self.is_2x2x():
            rowwise_tensor = self._quantize_func(x, dq_dtype=dq_dtype, flatten_axis=flatten_axis)
            colwise_tensor = self._quantize_func(
                x, is_colwise=True, dq_dtype=dq_dtype, flatten_axis=flatten_axis
            )
            return ScaledTensor2x(rowwise_tensor, colwise_tensor)

        if is_colwise:
            return self._quantize_func(
                x, is_colwise=True, dq_dtype=dq_dtype, flatten_axis=flatten_axis
            )

        return self._quantize_func(x, dq_dtype=dq_dtype, flatten_axis=flatten_axis)

    def get_scale_shapes(self, data_shape, is_padded=True, flatten_axis=-1, **kwargs):
        """Get shapes for scale tensors.

        Args:
            data_shape: Shape of the input tensor
            is_padded: Whether to use padded shapes

        Returns:
            Tuple of (rowwise_scale_shape, colwise_scale_shape)
        """
        del kwargs
        return self.scaling_mode.get_scale_shape_2x(data_shape, is_padded, flatten_axis)

    def get_scale_dtype(self):
        """Get the data type for scale tensors.

        Returns:
            The data type for scale tensors
        """
        return self.scaling_mode.get_scale_dtype()


@register_pytree_node_class
@dataclass
class CurrentScaleQuantizer(Quantizer):
    """Quantizer implementation using current scaling.

    This quantizer uses current scaling mode with float32 scales

    Attributes:
        scaling_mode: Set to NVTE_DELAYED_TENSOR_SCALING
        q_layout: Quantization axis (default: ROWWISE_COLWISE)
    """

    scaling_mode: ScalingMode = ScalingMode.CURRENT_TENSOR_SCALING
    q_layout: QuantizeLayout = QuantizeLayout.ROWWISE_COLWISE
    data_layout: str = "NT"

    def _quantize_func(
        self,
        x: Union[jnp.ndarray, NoScaleTensor],
        is_colwise=False,
        dq_dtype=None,
        flatten_axis=-1,
    ) -> ScaledTensor1x:
        """Quantize function helper for delayed scaling FP8.

        Args:
            x: Input tensor to quantize
            is_colwise: Whether to use column-wise quantization
            dq_dtype: Data type for dequantized values

        Returns:
            A ScaledTensor1x containing the quantized data
        """
        if isinstance(x, jnp.ndarray):
            x = NoScaleTensor(data=x, amax=None)

        dq_dtype = dq_dtype if dq_dtype is not None else x.data.dtype

        compute_dtype = jnp.float32
        dtype_max = (jnp.finfo(self.q_dtype).max).astype(compute_dtype)
        amax = x.amax or jnp.max(jnp.abs(x.data)).reshape((1,))
        fp8_max = jnp.astype(jnp.finfo(self.q_dtype).max, jnp.float32)
        scale = (fp8_max / amax) / (2 ** get_quantize_config().MARGIN)
        scaled_x = x.data.astype(compute_dtype) * scale

        clipped_scaled_x = jnp.clip(scaled_x, -dtype_max, dtype_max).astype(self.q_dtype)
        scale_inv = 1.0 / scale
        return ScaledTensorFactory.create_1x(
            data=clipped_scaled_x,
            scale_inv=scale_inv,
            scaling_mode=self.scaling_mode,
            dq_dtype=dq_dtype,
            flatten_axis=flatten_axis,
        )

    def quantize(
        self, x, is_rowwise: bool = None, is_colwise: bool = None, dq_dtype=None, flatten_axis=-1
    ):
        """Quantize a tensor using the internal _quantize_func().

        Args:
            x: Input tensor to quantize
            is_rowwise: Whether to use row-wise quantization
            is_colwise: Whether to use column-wise quantization
            dq_dtype: Data type for dequantized values
            flatten_axis: The quantization axis for the tensor

        Returns:
            A ScaledTensor1x or ScaledTensor2x containing the quantized data
        """
        if isinstance(x, jnp.ndarray):
            x = NoScaleTensor(data=x, amax=None)

        dq_dtype = dq_dtype if dq_dtype is not None else x.data.dtype
        if flatten_axis < 0:
            flatten_axis += x.ndim
        assert 0 < flatten_axis < x.ndim, "flatten_axis is out of bounds!"

        is_rowwise = (
            is_rowwise
            if is_rowwise is not None
            else (self.q_layout == QuantizeLayout.ROWWISE or self.is_2x2x())
        )
        is_colwise = (
            is_colwise
            if is_colwise is not None
            else (self.q_layout == QuantizeLayout.COLWISE or self.is_2x2x())
        )

        rowwise_tensor = self._quantize_func(x, dq_dtype=dq_dtype, flatten_axis=flatten_axis)
        colwise_tensor = None
        if is_colwise:
            colwise_tensor = ScaledTensorFactory.create_1x(
                data=jnp.transpose(
                    rowwise_tensor.data, (*range(flatten_axis, x.ndim), *range(flatten_axis))
                ),
                scale_inv=rowwise_tensor.scale_inv,
                scaling_mode=self.scaling_mode,
                dq_dtype=dq_dtype,
                is_colwise=True,
                data_layout="T",
                flatten_axis=flatten_axis,
            )

        if is_colwise and is_rowwise:
            return ScaledTensor2x(rowwise_tensor, colwise_tensor)
        if is_colwise:
            return colwise_tensor
        return rowwise_tensor


@register_pytree_node_class
@dataclass
class DelayedScaleQuantizer(CurrentScaleQuantizer):
    """Quantizer implementation using delayed scaling.

    This quantizer uses delayed scaling mode with float32 scales and maintains
    a history of maximum absolute values for dynamic scaling.

    Attributes:
        scaling_mode: Set to NVTE_DELAYED_TENSOR_SCALING
        q_layout: Quantization axis (default: ROWWISE_COLWISE)
        scale: Current scaling factor
        amax_history: History of maximum absolute values
    """

    scaling_mode: ScalingMode = ScalingMode.DELAYED_TENSOR_SCALING
    q_layout: QuantizeLayout = QuantizeLayout.ROWWISE_COLWISE

    scale: jnp.ndarray = field(default_factory=lambda: jnp.ones((1,), jnp.float32))
    amax_history: jnp.ndarray = field(
        default_factory=lambda: jnp.zeros((get_quantize_config().AMAX_HISTORY_LEN,), jnp.float32)
    )

    def tree_flatten(self):
        """Flatten the quantizer for JAX tree operations.

        Returns:
            Tuple of (children, aux_data) for tree operations
        """
        children = (self.scale, self.amax_history)
        aux_data = (self.q_dtype, self.scaling_mode, self.q_layout, self.data_layout)
        return (children, aux_data)

    def _quantize_func(
        self, x: jnp.ndarray, is_colwise=False, dq_dtype=None, flatten_axis=-1
    ) -> ScaledTensor1x:
        """Quantize function helper for delayed scaling FP8.

        Args:
            x: Input tensor to quantize
            is_colwise: Whether to use column-wise quantization
            dq_dtype: Data type for dequantized values
            flatten_axis: The quantization axis for the tensor
        Returns:
            A ScaledTensor1x containing the quantized data
        """
        if isinstance(x, jnp.ndarray):
            x = NoScaleTensor(data=x, amax=None)

        dq_dtype = dq_dtype if dq_dtype is not None else x.data.dtype

        compute_dtype = jnp.float32
        dtype_max = (jnp.finfo(self.q_dtype).max).astype(compute_dtype)
        scaled_x = x.data.astype(compute_dtype) * self.scale

        # quantize() in the old dot.py do this way, leave this code block here for future debugging
        # compute_dtype = x.dtype
        # dtype_max = (jnp.finfo(self.q_dtype).max).astype(compute_dtype)
        # scaled_x = x * self.scale.astype(compute_dtype)

        clipped_scaled_x = jnp.clip(scaled_x, -dtype_max, dtype_max).astype(self.q_dtype)
        scale_inv = 1.0 / self.scale
        amax = x.amax or jnp.max(jnp.abs(x.data)).reshape((1,))
        self.update(amax)
        return ScaledTensorFactory.create_1x(
            data=clipped_scaled_x,
            scale_inv=scale_inv,
            scaling_mode=self.scaling_mode,
            dq_dtype=dq_dtype,
            flatten_axis=flatten_axis,
        )

    @staticmethod
    @jax.jit
    def _update_amax_history(amax_history, new_amax):
        """Update AMAX history with new maximum value.

        Args:
            amax_history: Current AMAX history
            new_amax: New maximum value to add

        Returns:
            Updated AMAX history
        """
        amax_history = amax_history.at[0].set(new_amax[0])
        return amax_history

    @staticmethod
    @partial(jax.jit, static_argnums=(2,))
    def _compute_scale(amax_history, scale, q_dtype):
        """Compute new scale based on AMAX history.

        Args:
            amax_history: History of maximum absolute values
            scale: Current scale
            q_dtype: Quantization data type

        Returns:
            Updated scale value
        """
        # 2. Calculate the current scale
        if get_quantize_config().AMAX_COMPUTE_ALGO is AmaxComputeAlgo.MAX:
            amax = jnp.max(amax_history, axis=-1, keepdims=True)
        else:
            amax = amax_history[0:1]

        return compute_scale_from_amax(amax, q_dtype, scale=scale)

    @staticmethod
    @jax.jit
    def _roll_and_reset_amax_history(amax_history):
        """Roll AMAX history and reset first element.

        Args:
            amax_history: Current AMAX history

        Returns:
            Updated AMAX history
        """
        updated_amax_history = jnp.roll(amax_history, -1, -1)
        amax_history = updated_amax_history.at[0].set(0.0)
        return amax_history

    def update(self, new_amax: jnp.ndarray):
        """Update AMAX history and compute new scale.

        Args:
            new_amax: New maximum absolute value to add to history
        """
        amax_history = self._update_amax_history(self.amax_history, new_amax)
        self.scale = self._compute_scale(amax_history, self.scale, self.q_dtype)
        self.amax_history = self._roll_and_reset_amax_history(amax_history)


@register_pytree_node_class
@dataclass
class BlockScaleQuantizer(Quantizer):
    """Quantizer implementation using block-based scaling.

    This quantizer uses block scaling mode with FP8 scales and block-based
    quantization for improved efficiency.

    Attributes:
        scaling_mode: Set to NVTE_MXFP8_1D_SCALING
        q_layout: Quantization axis (default: ROWWISE_COLWISE)
    """

    scaling_mode: ScalingMode = ScalingMode.MXFP8_1D_SCALING
    q_layout: QuantizeLayout = QuantizeLayout.ROWWISE_COLWISE
    data_layout: str = "NN"

    def _quantize_func(self, x, is_colwise=False, dq_dtype=None, flatten_axis=-1) -> ScaledTensor1x:
        """Quantize function helper for block scaling FP8.

        Args:
            x: Input tensor to quantize
            is_colwise: Whether to use column-wise quantization
            dq_dtype: Data type for dequantized values
            flatten_axis: The quantization axis for the tensor

        Returns:
            A ScaledTensor1x containing the quantized data
        """
        if isinstance(x, NoScaleTensor):
            # No need for amax in MXFP8 block scaling, so simply extract the jnp.ndarray data tensor from the NoScaleTensor x.
            x = x.data

        # TODO(Phuong): use quantize_func from JAX
        if flatten_axis < 0:
            flatten_axis = x.ndim + flatten_axis
        assert (
            0 <= flatten_axis < x.ndim
        ), f"Invalid flatten_axis: {flatten_axis} for tensor of shape {x.shape}"

        dq_dtype = dq_dtype if dq_dtype is not None else x.dtype
        x_shape = x.shape
        scale_shape = self.scaling_mode.get_scale_shape(
            x_shape, is_colwise, is_padded=False, flatten_axis=flatten_axis
        )
        scale_dtype = self.scaling_mode.get_scale_dtype()
        x = x.reshape(
            *x_shape[: flatten_axis - 1],
            scale_shape[flatten_axis - 1],
            int(x_shape[flatten_axis - 1] / scale_shape[flatten_axis - 1]),
            *x_shape[flatten_axis:-1],
            scale_shape[-1],
            int(x_shape[-1] / scale_shape[-1]),
        )
        amax = jnp.max(jnp.abs(x), axis=(flatten_axis + 2 - 2, -1), keepdims=True)
        MAX = jnp.finfo(self.q_dtype).max.astype(jnp.float32)
        scales = amax.astype(jnp.float32) / MAX

        scales_q = self._cast_to_e8m0_with_rounding_up(scales)
        scaled_x = x / self._e8m0_to_dtype(scales_q, jnp.float32)

        clipped_x = jnp.clip(scaled_x, -MAX, MAX)
        x_q = clipped_x.astype(self.q_dtype).reshape(x_shape)
        scales_q = scales_q.reshape(scale_shape).view(scale_dtype)

        return ScaledTensorFactory.create_1x(
            x_q,
            scales_q,
            scaling_mode=self.scaling_mode,
            is_colwise=is_colwise,
            dq_dtype=dq_dtype,
            flatten_axis=flatten_axis,
        )

    def _cast_to_e8m0_with_rounding_up(self, scales):
        """Cast scales to E8M0 format with rounding up.

        Args:
            scales: Input scales to convert

        Returns:
            Scales in E8M0 format
        """
        temp = scales.astype(jnp.float32).view(jnp.uint32)
        exp = temp >> 23
        mant = temp & 0x7FFFFF
        is_ru = jnp.logical_and(
            jnp.logical_and((mant > 0), (exp != 0xFE)),
            ~jnp.logical_and((exp == 0), (mant <= 0x400000)),
        )
        exp = jnp.where(is_ru, exp + 1, exp)
        new_scales = exp.astype(jnp.uint8)
        return new_scales

    def _e8m0_to_dtype(self, x, dtype):
        """Convert E8M0 format to specified data type.

        Args:
            x: Input in E8M0 format
            dtype: Target data type

        Returns:
            Converted values in target data type
        """
        temp = x.astype(jnp.uint32)
        exp = temp << 23
        new_x = exp.view(jnp.float32)
        near_zero_value = 2**-15 if dtype == jnp.float16 else 2**-127
        new_x = jnp.where(new_x == 0, jnp.array(near_zero_value, jnp.float32), new_x)
        return new_x.astype(dtype)


@register_pytree_node_class
@dataclass
class QuantizerSet:
    """Set of quantizers for different tensor types.

    This class manages quantizers for input tensors, kernel tensors, and
    gradient tensors.

    Attributes:
        x: Quantizer for input tensors
        kernel: Quantizer for kernel tensors
        dgrad: Quantizer for gradient tensors
    """

    x: Optional[Quantizer]
    kernel: Optional[Quantizer]
    dgrad: Optional[Quantizer]

    def tree_flatten(self):
        """Flatten the quantizer set for JAX tree operations.

        Returns:
            Tuple of (children, aux_data) for tree operations
        """
        children = (self.x, self.kernel, self.dgrad)
        aux_data = ()
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """Reconstruct a quantizer set from its flattened representation.

        Args:
            aux_data: Unused auxiliary data
            children: Tuple of quantizers

        Returns:
            A reconstructed QuantizerSet instance
        """
        return cls(*aux_data, *children)


@register_pytree_node_class
@dataclass
class GroupedQuantizer(Quantizer):
    """Quantizer for grouped arrays.

    This class extends Quantizer to support quantization of arrays in grouped manner,
    where elements are grouped along a specified axis then quantized separately.

    Attributes:
        data_layout: The data layout specification
        n_groups: Number of groups for quantization
        quantizers: Tuple of quantizers for each group
    """

    data_layout: str = None
    n_groups: int = 1
    quantizers: Tuple[Quantizer] = field(default_factory=lambda: (None,))

    def tree_flatten(self):
        """Flatten the quantizer for JAX tree operations.

        Returns:
            Tuple of (children, aux_data) for tree operations
        """
        children = (self.quantizers,)
        aux_data = (self.q_dtype, self.scaling_mode, self.q_layout, self.data_layout, self.n_groups)
        return (children, aux_data)

    def __post_init__(self):
        if self.quantizers[0] is None:
            quantizers = QuantizerFactory.create(
                self.n_groups, self.scaling_mode, self.q_dtype, self.q_layout
            )
            self.quantizers = (quantizers,) if not isinstance(quantizers, tuple) else quantizers
        self.data_layout = self.quantizers[0].data_layout

    def _create_grouped_tensor_from_tensor_list(
        self, tensor_list, group_sizes, original_shape, group_axis, mode
    ):
        # mode 0 = concate, mode 1 = add
        # TODO(Ming Huang): Consider to apply Enum for mode.
        assert mode in [0, 1]
        grouped_data = (
            [] if mode == 0 else jnp.zeros(tensor_list[0].data.shape, tensor_list[0].data.dtype)
        )
        grouped_scale_inv = []

        for tensor in tensor_list:
            if mode == 0:
                grouped_data.append(tensor.data.flatten())
            else:
                grouped_data += tensor.data
            grouped_scale_inv.append(tensor.scale_inv.flatten())

        grouped_data = jnp.concatenate(grouped_data) if mode == 0 else grouped_data.flatten()
        grouped_scale_inv = jnp.concatenate(grouped_scale_inv)

        return ScaledTensorFactory.create_1x(
            grouped_data,
            grouped_scale_inv,
            scaling_mode=self.scaling_mode,
            dq_dtype=tensor_list[0].dq_dtype,
            is_colwise=tensor_list[0].is_colwise,
            data_layout=tensor_list[0].data_layout,
            flatten_axis=tensor_list[0].flatten_axis,
            group_sizes=group_sizes,
            original_shape=original_shape,
            group_axis=group_axis,
        )

    def _quantize_func(self, *args, **kwargs):
        pass

    def quantize(
        self,
        x,
        is_rowwise: bool = None,
        is_colwise: bool = None,
        dq_dtype=None,
        flatten_axis=-1,
        group_sizes=None,
        group_axis=0,
    ):
        """Quantize a tensor in grouped manner.

        Expected input shape: [M, K] or [G, K, N]
        Split to x.shape[group_axis] number of groups if group_sizes is not given

        Args:
            x: Input tensor to quantize
            is_rowwise: Whether to use row-wise quantization
            is_colwise: Whether to use column-wise quantization
            dq_dtype: Data type for dequantized values
            flatten_axis: The axis along which the tensor could be flattened to 2D (default: -1)
            group_sizes: Array of ints containing the size of each group (default: None)
            group_axis: The axis along which grouping is performed (default: 0)

        Returns:
            A ScaledTensor1x or ScaledTensor2x containing the quantized data
        """
        assert group_axis == 0, "Only group_axis == 0 is supported now!"

        dq_dtype = dq_dtype if dq_dtype is not None else x.dtype
        if flatten_axis < 0:
            flatten_axis += x.ndim
        assert 0 < flatten_axis < x.ndim, "flatten_axis is out of bounds!"

        is_rowwise = (
            is_rowwise
            if is_rowwise is not None
            else (self.q_layout == QuantizeLayout.ROWWISE or self.is_2x2x())
        )
        is_colwise = (
            is_colwise
            if is_colwise is not None
            else (self.q_layout == QuantizeLayout.COLWISE or self.is_2x2x())
        )
        assert is_rowwise or is_colwise, "No quantization layout is specified"

        original_shape = x.shape

        if group_sizes is not None:
            assert not is_colwise, "Not yet implememted!"
            assert group_sizes.ndim == 1, (
                "GroupedQuantizer only support 1D group_sizes, got group_sizes.ndim ="
                f" {group_sizes.ndim}"
            )

            _zeros = partial(jax.lax.full_like, fill_value=0)

            x_iota = jax.lax.broadcasted_iota(group_sizes.dtype, x.shape, 0)
            group_ends = jnp.cumulative_sum(group_sizes)
            group_starts = jax.lax.concatenate(
                [_zeros(group_sizes)[:1], group_ends[:-1]],
                dimension=0,
            )
            x_zero = _zeros(x)

            tensor_list = []
            for i in range(len(group_sizes)):
                mask = jax.lax.bitwise_and(group_starts[i] <= x_iota, x_iota < group_ends[i])
                x_selected = jax.lax.select(mask, x, x_zero)
                tensor = self.quantizers[i].quantize(
                    x_selected, is_rowwise, is_colwise, dq_dtype, flatten_axis
                )
                tensor_list.append(tensor)
            combine_mode = 1  # Add
        else:
            group_sizes = jnp.ones(x.shape[group_axis], dtype=jnp.int32)
            x = jnp.split(x, x.shape[group_axis], axis=group_axis)

            tensor_list = []
            for i in range(len(group_sizes)):
                tensor = self.quantizers[i].quantize(
                    x[i], is_rowwise, is_colwise, dq_dtype, flatten_axis
                )
                tensor_list.append(tensor)
            combine_mode = 0  # Concate

        grouped_rowwise_tensor = grouped_colwise_tensor = None
        if is_rowwise:
            rowwise_tensor_list = [tensor.get_rowwise_tensor() for tensor in tensor_list]
            grouped_rowwise_tensor = self._create_grouped_tensor_from_tensor_list(
                rowwise_tensor_list, group_sizes, original_shape, group_axis, combine_mode
            )
        if is_colwise:
            colwise_tensor_list = [tensor.get_colwise_tensor() for tensor in tensor_list]
            grouped_colwise_tensor = self._create_grouped_tensor_from_tensor_list(
                colwise_tensor_list, group_sizes, original_shape, group_axis, combine_mode
            )

        if is_colwise and is_rowwise:
            return ScaledTensor2x(grouped_rowwise_tensor, grouped_colwise_tensor)
        if is_colwise:
            return grouped_colwise_tensor
        return grouped_rowwise_tensor

    def get_scale_shapes(self, data_shape, is_padded=True, flatten_axis=-1, group_sizes=None):
        assert group_sizes, "Empty group_sizes was given!"
        return self.scaling_mode.get_grouped_scale_shape_2x(
            data_shape, group_sizes, is_padded, flatten_axis
        )


@dataclass
class QuantizerFactory:
    """Factory class for creating quantizers.

    This class provides static methods to create individual quantizers and
    sets of quantizers with various configurations.
    """

    quantizer_type_map = {
        ScalingMode.DELAYED_TENSOR_SCALING: DelayedScaleQuantizer,
        ScalingMode.CURRENT_TENSOR_SCALING: CurrentScaleQuantizer,
        ScalingMode.MXFP8_1D_SCALING: BlockScaleQuantizer,
    }

    @staticmethod
    def create(
        n_quantizers: int = 1,
        scaling_mode: ScalingMode = None,
        q_dtype: jnp.dtype = None,
        q_layout: QuantizeLayout = None,
        n_groups: int = None,
        **kwargs,
    ) -> Quantizer:
        """Create one or more quantizers with specified parameters.

        Args:
            n_quantizers: Number of quantizers to create
            scaling_mode: Scaling mode to use
            q_dtype: Quantization data type
            q_layout: Quantization axis
            flatten_axis: The quantization axis for the tensor
            n_groups: Number of quantizers if GroupedQuantizer
            **kwargs: Additional arguments for quantizer initialization

        Returns:
            A single quantizer or tuple of quantizers
        """
        # (Phuong): add this assert back when NVTE_NO_SCALING is fully implememted
        assert isinstance(scaling_mode, ScalingMode), "Invalid scaling_mode type"
        if n_groups:
            if n_quantizers != 1:
                warnings.warn(
                    "Using more than one GroupedQuantizer for a grouped input is not recommended"
                )
            quantizer_type = GroupedQuantizer
            kwargs["n_groups"] = n_groups
        else:
            quantizer_type = QuantizerFactory.quantizer_type_map.get(scaling_mode)

        if scaling_mode == ScalingMode.NO_SCALING:
            quantizers = [None] * n_quantizers
        else:
            quantizers = []
            for _ in range(n_quantizers):
                quantizers.append(
                    quantizer_type(
                        q_dtype=q_dtype, scaling_mode=scaling_mode, q_layout=q_layout, **kwargs
                    )
                )
        return quantizers[0] if len(quantizers) == 1 else tuple(quantizers)

    @staticmethod
    def _create_set(
        x_scaling_mode,
        kernel_scaling_mode,
        grad_scaling_mode,
        fwd_dtype,
        bwd_dtype,
        is_2x2x,
        n_groups,
        **kwargs,
    ) -> QuantizerSet:
        """Create a set of quantizers for forward and backward passes.

        Args:
            x_scaling_mode: Scaling mode to use for input tensor 'x'
            kernel_scaling_mode: Scaling mode to use for kernel tensor
            grad_scaling_mode: Scaling mode to use for gradient tensor
            fwd_dtype: Data type for forward pass
            bwd_dtype: Data type for backward pass
            is_2x2x: Whether to use 2x2x quantization
            n_groups
            **kwargs: Additional arguments for quantizer initialization

        Returns:
            A QuantizerSet instance
        """
        if is_2x2x:
            q_layout_x = q_layout_kernel = q_layout_dgrad = QuantizeLayout.ROWWISE_COLWISE
        else:
            q_layout_x = q_layout_kernel = q_layout_dgrad = QuantizeLayout.ROWWISE
            if kernel_scaling_mode.is_1d_block_scaling():
                q_layout_kernel = QuantizeLayout.COLWISE
            if get_quantize_config().INFERENCE_MODE:
                q_layout_dgrad = None

        if "quantize_meta_set" in kwargs:
            quantize_meta_set = kwargs.get("quantize_meta_set")
            args_x = {
                "scale": quantize_meta_set.x.scale,
                "amax_history": quantize_meta_set.x.amax_history,
            }
            args_kernel = {
                "scale": quantize_meta_set.kernel.scale,
                "amax_history": quantize_meta_set.kernel.amax_history,
            }
            args_grad = {
                "scale": quantize_meta_set.grad.scale,
                "amax_history": quantize_meta_set.grad.amax_history,
            }
        else:
            args_x = args_kernel = args_grad = {}

        q_x = QuantizerFactory.create(1, x_scaling_mode, fwd_dtype, q_layout_x, n_groups, **args_x)
        q_kernel = QuantizerFactory.create(
            1, kernel_scaling_mode, fwd_dtype, q_layout_kernel, n_groups, **args_kernel
        )
        q_dgrad = QuantizerFactory.create(
            1, grad_scaling_mode, bwd_dtype, q_layout_dgrad, n_groups, **args_grad
        )
        return QuantizerSet(x=q_x, kernel=q_kernel, dgrad=q_dgrad)

    @staticmethod
    def create_set(
        n_quantizer_sets: int = 1,
        scaling_mode: Optional[ScalingMode] = None,
        fwd_dtype: jnp.dtype = None,
        bwd_dtype: jnp.dtype = None,
        is_2x2x: bool = None,
        n_groups: int = None,
        fp8_recipe: Optional[recipe.Recipe] = None,
        **kwargs,
    ) -> tuple[Union[tuple[Quantizer], None]]:
        """Create one or more sets of quantizers.

        Args:
            n_quantizer_sets: Number of quantizer sets to create
            scaling_mode: Scaling mode to use, default is get_quantize_config().get_scaling_mode
            fwd_dtype: Data type for forward pass, default is get_quantize_config().FWD_DTYPE
            bwd_dtype: Data type for backward pass, default is get_quantize_config().BWD_DTYPE
            is_2x2x: Whether to use 2x2x quantization, default is get_quantize_config().IF_QUANTIZE_2X
            n_groups:
            fp8_recipe: Recipe to use for quantization. Scaling mode can be specified directly via the scaling_mode parameter or indirectly via recipe. Recipe is preferred as it will support additional recipes in future where scaling mode differs between x, kernel, and grad in the quantizer set.
            **kwargs: Additional arguments for quantizer initialization

        Returns:
            A single quantizer set or tuple of quantizer sets
        """

        assert scaling_mode is None or fp8_recipe is None, (
            "Cannot specify both scaling_mode and fp8_recipe when creating a quantizer set. Scaling"
            " mode can be specified directly via the scaling_mode parameter or indirectly via"
            " recipe. Recipe is preferred as it will support additional recipes in future where"
            " scaling mode differs between x, kernel, and grad in the quantizer set."
        )

        if fp8_recipe is not None:
            quantize_config = get_quantize_config_class(fp8_recipe)()
            x_scaling_mode = quantize_config.get_scaling_mode(TensorSource.X)
            kernel_scaling_mode = quantize_config.get_scaling_mode(TensorSource.KERNEL)
            grad_scaling_mode = quantize_config.get_scaling_mode(TensorSource.DGRAD)
        elif scaling_mode is not None:
            x_scaling_mode = scaling_mode
            kernel_scaling_mode = scaling_mode
            grad_scaling_mode = scaling_mode
        else:
            x_scaling_mode = get_quantize_config().get_scaling_mode(TensorSource.X)
            kernel_scaling_mode = get_quantize_config().get_scaling_mode(TensorSource.KERNEL)
            grad_scaling_mode = get_quantize_config().get_scaling_mode(TensorSource.DGRAD)

        fwd_dtype = fwd_dtype or get_quantize_config().FWD_DTYPE
        bwd_dtype = bwd_dtype or get_quantize_config().BWD_DTYPE
        if is_2x2x is None:
            # TODO(Jeremy): check x, kernel, grad separately for 2x
            if x_scaling_mode.is_1d_block_scaling():
                is_2x2x = True
            elif x_scaling_mode.is_tensor_scaling():
                is_2x2x = not is_fp8_gemm_with_all_layouts_supported()
            else:  # NO_SCALING ignores is_2x2x for now
                is_2x2x = False
        is_inference_mode = get_quantize_config().INFERENCE_MODE
        assert not is_inference_mode, "Inference mode is not supported yet!"

        q_set = []
        for _ in range(n_quantizer_sets):
            q_set.append(
                QuantizerFactory._create_set(
                    x_scaling_mode=x_scaling_mode,
                    kernel_scaling_mode=kernel_scaling_mode,
                    grad_scaling_mode=grad_scaling_mode,
                    fwd_dtype=fwd_dtype,
                    bwd_dtype=bwd_dtype,
                    is_2x2x=is_2x2x,
                    n_groups=n_groups,
                    **kwargs,
                )
            )

        return q_set[0] if len(q_set) == 1 else tuple(q_set)


noop_quantizer_set = QuantizerFactory.create_set(scaling_mode=ScalingMode.NO_SCALING, is_2x2x=False)
