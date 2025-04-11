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
from typing import Union, Optional

import jax
import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class
from transformer_engine_jax import QuantizeLayout

from .scaling_modes import ScalingMode
from .tensor import ScaledTensor1x, ScaledTensor2x, ScaledTensorFactory
from .helper import (
    QuantizeConfig,
    AmaxComputeAlgo,
)

__all__ = [
    "QuantizeLayout",
    "Quantizer",
    "QuantizerSet",
    "DelayedScaleQuantizer",
    "BlockScaleQuantizer",
    "QuantizerFactory",
    "noop_quantizer_set",
]


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

    def quantize(self, x, is_rowwise=False, is_colwise=False, dq_dtype=None, flatten_axis=-1)
    ->ScaledTensor:
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

    def get_scale_shapes(self, data_shape, is_padded=True, flatten_axis=-1):
        """Get shapes for scale tensors.

        Args:
            data_shape: Shape of the input tensor
            is_padded: Whether to use padded shapes

        Returns:
            Tuple of (rowwise_scale_shape, colwise_scale_shape)
        """
        return self.scaling_mode.get_scale_shape_2x(data_shape, is_padded, flatten_axis)

    def get_scale_dtype(self):
        """Get the data type for scale tensors.

        Returns:
            The data type for scale tensors
        """
        return self.scaling_mode.get_scale_dtype()


@register_pytree_node_class
@dataclass
class DelayedScaleQuantizer(Quantizer):
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
    data_layout: str = "NT"

    scale: jnp.ndarray = field(default_factory=lambda: jnp.ones((1,), jnp.float32))
    amax_history: jnp.ndarray = field(
        default_factory=lambda: jnp.zeros((QuantizeConfig.AMAX_HISTORY_LEN,), jnp.float32)
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
        dq_dtype = dq_dtype if dq_dtype is not None else x.dtype

        compute_dtype = self.scale.dtype
        dtype_max = (jnp.finfo(self.q_dtype).max).astype(compute_dtype)
        scaled_x = x.astype(compute_dtype) * self.scale

        clipped_scaled_x = jnp.clip(scaled_x, -dtype_max, dtype_max).astype(self.q_dtype)
        scale_inv = 1.0 / self.scale
        self.update(jnp.max(jnp.abs(x)).reshape((1,)))
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
        fp8_max = jnp.astype(jnp.finfo(q_dtype).max, jnp.float32)

        if QuantizeConfig.AMAX_COMPUTE_ALGO is AmaxComputeAlgo.MAX:
            amax = jnp.max(amax_history, axis=-1, keepdims=True)
        else:
            amax = amax_history[0:1]

        sf = (fp8_max / amax) / (2**QuantizeConfig.MARGIN)
        sf = jnp.where(amax > 0.0, sf, scale)
        sf = jnp.where(jnp.isfinite(amax), sf, scale)
        scale = scale.at[0].set(sf[0])
        return scale

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
            self.scaling_mode,
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


@dataclass
class QuantizerFactory:
    """Factory class for creating quantizers.

    This class provides static methods to create individual quantizers and
    sets of quantizers with various configurations.
    """

    quantizer_type_map = {
        ScalingMode.DELAYED_TENSOR_SCALING: DelayedScaleQuantizer,
        ScalingMode.MXFP8_1D_SCALING: BlockScaleQuantizer,
    }

    @staticmethod
    def create(
        n_quantizers: int = 1,
        scaling_mode: ScalingMode = None,
        q_dtype: jnp.dtype = None,
        q_layout: QuantizeLayout = None,
        **kwargs,
    ) -> Quantizer:
        """Create one or more quantizers with specified parameters.

        Args:
            n_quantizers: Number of quantizers to create
            scaling_mode: Scaling mode to use
            q_dtype: Quantization data type
            q_layout: Quantization axis
            flatten_axis: The quantization axis for the tensor
            **kwargs: Additional arguments for quantizer initialization

        Returns:
            A single quantizer or tuple of quantizers
        """
        # (Phuong): add this assert back when NVTE_NO_SCALING is fully implememted
        assert isinstance(scaling_mode, ScalingMode), "Invalid scaling_mode type"
        # import pdb; pdb.set_trace()
        if scaling_mode == ScalingMode.NO_SCALING:
            quantizers = [None] * n_quantizers
        else:
            quantizers = []
            for _ in range(n_quantizers):
                quantizer_type = QuantizerFactory.quantizer_type_map.get(scaling_mode)
                quantizers.append(
                    quantizer_type(
                        q_dtype=q_dtype, scaling_mode=scaling_mode, q_layout=q_layout, **kwargs
                    )
                )
        return quantizers[0] if len(quantizers) == 1 else tuple(quantizers)

    @staticmethod
    def _create_set(scaling_mode, fwd_dtype, bwd_dtype, is_2x2x, **kwargs) -> QuantizerSet:
        """Create a set of quantizers for forward and backward passes.

        Args:
            scaling_mode: Scaling mode to use
            fwd_dtype: Data type for forward pass
            bwd_dtype: Data type for backward pass
            is_2x2x: Whether to use 2x2x quantization
            **kwargs: Additional arguments for quantizer initialization

        Returns:
            A QuantizerSet instance
        """
        if is_2x2x:
            q_layout_x = q_layout_kernel = q_layout_dgrad = QuantizeLayout.ROWWISE_COLWISE
        else:
            q_layout_x = QuantizeLayout.ROWWISE
            q_layout_kernel = QuantizeLayout.COLWISE
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

        q_x = QuantizerFactory.create(1, scaling_mode, fwd_dtype, q_layout_x, **args_x)
        q_kernel = QuantizerFactory.create(
            1, scaling_mode, fwd_dtype, q_layout_kernel, **args_kernel
        )
        q_dgrad = QuantizerFactory.create(1, scaling_mode, bwd_dtype, q_layout_dgrad, **args_grad)
        return QuantizerSet(x=q_x, kernel=q_kernel, dgrad=q_dgrad)

    @staticmethod
    def create_set(
        n_quantizer_sets: int = 1,
        scaling_mode: ScalingMode = None,
        fwd_dtype: jnp.dtype = None,
        bwd_dtype: jnp.dtype = None,
        is_2x2x: bool = None,
        **kwargs,
    ) -> tuple[Union[tuple[Quantizer], None]]:
        """Create one or more sets of quantizers.

        Args:
            n_quantizer_sets: Number of quantizer sets to create
            scaling_mode: Scaling mode to use, default is QuantizeConfig.SCALING_MODE
            fwd_dtype: Data type for forward pass, default is QuantizeConfig.FWD_DTYPE
            bwd_dtype: Data type for backward pass, default is QuantizeConfig.BWD_DTYPE
            is_2x2x: Whether to use 2x2x quantization, default is QuantizeConfig.IF_QUANTIZE_2X
            **kwargs: Additional arguments for quantizer initialization

        Returns:
            A single quantizer set or tuple of quantizer sets
        """
        scaling_mode = scaling_mode or QuantizeConfig.SCALING_MODE
        fwd_dtype = fwd_dtype or QuantizeConfig.FWD_DTYPE
        bwd_dtype = bwd_dtype or QuantizeConfig.BWD_DTYPE
        is_2x2x = is_2x2x or QuantizeConfig.IF_QUANTIZE_2X

        q_set = []
        for _ in range(n_quantizer_sets):
            q_set.append(
                QuantizerFactory._create_set(scaling_mode, fwd_dtype, bwd_dtype, is_2x2x, **kwargs)
            )

        return q_set[0] if len(q_set) == 1 else tuple(q_set)


noop_quantizer_set = QuantizerFactory.create_set(scaling_mode=ScalingMode.NO_SCALING)


@register_pytree_node_class
@dataclass
class GroupedQuantizer(Quantizer):
    """Base class for quantizers.

    This abstract class defines the interface for tensor quantization, providing
    methods for quantization and scale management.

    Attributes:
        q_dtype: The data type for quantized values
        scaling_mode: The scaling mode to use for quantization
        q_layout: The quantization axis (row-wise, column-wise, or both)

    """
    num_groups: int
    q_dtype: jnp.dtype
    scaling_mode: ScalingMode
    q_layout: QuantizeLayout
    data_layout: str = None
    quantizers: List[Quantizer] = None

    def tree_flatten(self):
        """Flatten the quantizer for JAX tree operations.

        Returns:
            Tuple of (children, aux_data) for tree operations
        """
        children = (self.quantizers)
        aux_data = (self.num_groups, self.q_dtype, self.scaling_mode, self.q_layout, self.data_layout)
        return (aux_data, children)

    def __post_init__(self):
        if not self._internal_quantizers:
            self._internal_quantizer = QuantizerFactory.create(
                    self.num_groups, self.scaling_mode, self.q_dtype, self.q_layout
                    )
        self.data_layout = self.quantizers[0].data_layout

    def _create_grouped_tensor_from_tensor_list(tensor_list, group_sizes, other_sizes):
        grouped_data = []
        grouped_scale_inv = []

        for tensor in tensor_list:
            grouped_data.append(tensor_list.data.flatten_axis)
            grouped_scale_inv.append(tensor_list.scale_inv.flatten())

        grouped_data = jnp.concatenate(grouped_data)
        grouped_scale_inv = jnp.concatenate(grouped_scale_inv)

        return ScaledTensorFactory.create_1x(grouped_data, grouped_scale_inv,
                                             self.scaling_mode, dq_dtype, is_colwise,
                                             self.data_layout, flatten_axis
                                             group_size=group_size,
                                             other_sizes=other_sizes,
                                             )

    def _quantize_func(self, *args, **kwargs):
        pass

    def quantize(self, x, group_sizes, is_rowwise: bool = None, is_colwise: bool = None,
                 dq_dtype=None, flatten_axis=-1):
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

        assert flatten_axis == 1, f"GroupedQuantizer only support flatten_axis == 1, got {flatten_axis}"

        assert group_sizes.sum == x.shape[flatten_axis - 1], f"Unable to split x. x.shape[{flatten_axis - 1}] = {x.shape[flatten_axis -1]} while sum(group_size) = {group_size.sum}"

        other_sizes = x.shape[flatten_axis:]

        x = jnp.split(x, group_sizes, axis=flatten_axis - 1)
        tensor_list = []
        for i in range(len(group_sizes)):
            tensor = self.quantizers[i](x[i], is_rowwise, is_colwise, dq_dtype, flatten_axis)
            tensor_list.append[tensor]

        rowwise_grouped_tensor = colwise_grouped_tensor = None
        if is_rowwise:
            rowwise_tensor_list = [tensor.get_rowwise_tensor() for tensor in tensor_list]
            grouped_rowwise_tensor = _create_grouped_tensor_from_tensor_list(
                    rowwise_tensor_list, group_sizes, other_sizes
                    )
        if is_colwise:
            colwise_tensor_list = [tensor.get_colwise_tensor() for tensor in tensor_list]
            grouped_colwise_tensor = _create_grouped_tensor_from_tensor_list(
                    colwise_tensor_list, group_sizes, other_sizes
                    )

        if is_colwise and is_rowwise:
            return ScaledTensor2x(grouped_rowwise_tensor, grouped_colwise_tensor)
        if is_colwise:
            return grouped_colwise_tensor
        return grouped_rowwise_tensor

    def get_scale_shapes(self, data_shape, group_sizes, is_padded=True, flatten_axis=-1):
        return self.scaling_mode.get_grouped_scale_shape_2x(data_shape, group_sizes, is_padded, flatten_axis)
