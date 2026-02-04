# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Grouped tensor class for handling collections of tensors with different shapes"""
from __future__ import annotations
from typing import Optional, Tuple, List, Union
import math

import torch

from transformer_engine_torch import Float8BlockScaleTensorFormat

from ...quantized_tensor import QuantizedTensorStorage, Quantizer

from ..mxfp8_tensor import MXFP8Tensor
from ..nvfp4_tensor import NVFP4Tensor
from ..float8_tensor import Float8Tensor
from ..float8_blockwise_tensor import Float8BlockwiseQTensor
from .float8_tensor_storage import Float8TensorStorage
from .mxfp8_tensor_storage import MXFP8TensorStorage
from .float8_blockwise_tensor_storage import Float8BlockwiseQTensorStorage
from .nvfp4_tensor_storage import NVFP4TensorStorage


class GroupedTensor:
    """
    EXPERIMENTAL FEATURE AND SUBJECT TO CHANGE.

    Grouped tensor is a collection of tensors with different shapes but the same dtype and scaling mode.

    Shape Representation:
    - logical_shape: 2D shape representing the conceptual layout, i.e. the shape when member tensors
      are flattened to 2D and stacked together (REQUIRED)
        + When all_same_shape(): [num_tensors * M, N] where each tensor is (M, N)
        + When varying_first_dim(): [~sum_of_first_dims, N] where N is common
        + When varying_last_dim(): [M, ~sum_of_last_dims] where M is common
        + When varying_both_dims(): [1, total_elements] (fully flattened)

    - first_dims and last_dims are OPTIONAL (None if dimension is uniform)
        + None first_dims: all tensors have the same first dimension
        + None last_dims: all tensors have the same last dimension
        + Both None: all tensors have identical shapes
        + Both set: each tensor has unique shape (first_dims[i], last_dims[i])

    Data Layout:
    - ALL data fields are stored as 1D flattened arrays (data, columnwise_data, scale_inv, etc.)
    - logical_shape provides the conceptual 2D interpretation
    - All data is stored on device in contiguous layout

    Note: This structure is used only for combined storage of multiple tensors with the same dtype and scaling mode.
    """

    def __init__(
        self,
        num_tensors: int,
        shape: List[Tuple[int, int]],
        quantizers: List[Optional[Quantizer]] = None,
        dtype: Optional[torch.dtype] = None,
        data: Optional[torch.Tensor] = None,
        columnwise_data: Optional[torch.Tensor] = None,
        scale_inv: Optional[torch.Tensor] = None,
        columnwise_scale_inv: Optional[torch.Tensor] = None,
        amax: Optional[torch.Tensor] = None,
        columnwise_amax: Optional[torch.Tensor] = None,
        scale: Optional[torch.Tensor] = None,
        first_dims: Optional[torch.Tensor] = None,
        last_dims: Optional[torch.Tensor] = None,
        tensor_offsets: Optional[torch.Tensor] = None,
        offsets: Optional[List[int]] = None,
        scale_inv_offsets: Optional[List[int]] = None,
        columnwise_scale_inv_offsets: Optional[List[int]] = None,
        logical_shape: Optional[Tuple[int, int]] = None,
    ) -> None:
        """
        Initialize a GroupedTensor.

        Args:
            num_tensors: Number of tensors in the group
            shape: 2D shape of each tensor (len num_tensors)
            quantizers: Quantizers for the grouped tensor
            data: Row-wise data buffer (1D flattened)
            columnwise_data: Column-wise data buffer (1D flattened)
            scale_inv: Row-wise scale inverse buffer
            columnwise_scale_inv: Column-wise scale inverse buffer
            amax: Row-wise amax buffer
            columnwise_amax: Column-wise amax buffer
            scale: Scale buffer (for FP8-DS only)
            first_dims: Device tensor of int64 array of length num_tensors (or None if uniform)
            last_dims: Device tensor of int64 array of length num_tensors (or None if uniform)
            tensor_offsets: Device tensor of int64 array of length num_tensors (or None if uniform)
            offsets: Vector of integer offsets for each tensor.
            logical_shape: 2D tuple representing conceptual shape
        """
        self.num_tensors = num_tensors
        self.quantizers = quantizers
        self.shape = shape
        self.dtype = (
            dtype if dtype is not None else torch.float32
        )  # Default to float32 if not provided

        # Data buffers
        self.data = data
        self.columnwise_data = columnwise_data
        self.scale_inv = scale_inv
        self.columnwise_scale_inv = columnwise_scale_inv
        self.amax = amax
        self.columnwise_amax = columnwise_amax
        self.scale = scale

        # For convenient indexing for python GroupedTensor API.
        self.scale_inv_offsets = scale_inv_offsets
        self.columnwise_scale_inv_offsets = columnwise_scale_inv_offsets

        # Shape information (OPTIONAL - None if dimension is uniform across all tensors)
        # first_dims[i] = first dimension of tensor i (None if all tensors have same first dim)
        # last_dims[i] = last dimension of tensor i (None if all tensors have same last dim)
        self.first_dims = (
            first_dims  # Device pointer to int64_t array of length num_tensors (or None)
        )
        self.last_dims = (
            last_dims  # Device pointer to int64_t array of length num_tensors (or None)
        )

        # Offsets for indexing into contiguous 1D layout (OPTIONAL - not needed if all_same_shape())
        # tensor_offsets[i] = element offset to start of tensor i (cumulative sum of numel for tensors 0..i-1)
        # Usage: tensor_i_ptr = data.data_ptr() + tensor_offsets[i] * element_size
        # If None and all_same_shape(): offset[i] = i * M * N (where M, N are common dimensions)
        self.tensor_offsets = (
            tensor_offsets  # Device pointer to int64_t array of length num_tensors (or None)
        )
        self.offsets = offsets  # Vector of integer offsets for each tensor.

        # Logical shape: conceptual 2D shape of the grouped data (REQUIRED)
        # Represents how the 1D flattened data should be interpreted as 2D
        # Always 2D with positive dimensions
        self.logical_shape = logical_shape if logical_shape is not None else (0, 0)

        # Hold a reference to the quantized tensors that occupy same storage as the GroupedTensor.
        # Used as a convenience.
        self.quantized_tensors = None

    def has_data(self) -> bool:
        """
        Check if the tensor has row-wise data.

        Returns:
            True if data buffer is initialized, False otherwise
        """
        return self.data is not None

    def has_columnwise_data(self) -> bool:
        """
        Check if the tensor has column-wise data.

        Returns:
            True if columnwise_data buffer is initialized, False otherwise
        """
        return self.columnwise_data is not None

    def all_same_first_dim(self) -> bool:
        """
        Check if all tensors in the group have the same first dimension.

        Returns:
            True if first dimension is uniform across all tensors
        """
        return self.first_dims is None

    def all_same_last_dim(self) -> bool:
        """
        Check if all tensors in the group have the same last dimension.

        Returns:
            True if last dimension is uniform across all tensors
        """
        return self.last_dims is None

    def all_same_shape(self) -> bool:
        """
        Check if all tensors in the group have identical shapes.

        Returns:
            True if all tensors have the same shape
        """
        return self.first_dims is None and self.last_dims is None

    def varying_both_dims(self) -> bool:
        """
        Check if both dimensions vary across tensors.

        Returns:
            True if both first and last dimensions vary
        """
        return self.first_dims is not None and self.last_dims is not None

    def get_common_first_dim(self) -> int:
        """
        Get the common first dimension when all tensors share it.

        Returns:
            The common first dimension

        Raises:
            RuntimeError: If first dimension varies across tensors or logical_shape is not 2D
        """
        if not self.all_same_first_dim():
            raise RuntimeError("First dim varies across tensors")
        if len(self.logical_shape) != 2:
            raise RuntimeError("Logical shape must be 2D")

        if self.all_same_shape():
            # When both dims are uniform: logical_shape = [num_tensors * M, N]
            return self.logical_shape[0] // self.num_tensors
        # When varying last dims but not first dim: logical_shape = [M, sum_of_last_dims]
        return self.logical_shape[0]

    def get_common_last_dim(self) -> int:
        """
        Get the common last dimension when all tensors share it.

        Returns:
            The common last dimension

        Raises:
            RuntimeError: If last dimension varies across tensors or logical_shape is not 2D
        """
        if not self.all_same_last_dim():
            raise RuntimeError("Last dim varies across tensors")
        if len(self.logical_shape) != 2:
            raise RuntimeError("Logical shape must be 2D")

        # For both uniform and varying first dim cases: logical_shape[1] is the common last dim
        return self.logical_shape[1]

    def get_dtype(self) -> torch.dtype:
        """
        Get the high precision data type of the tensor.

        Returns:
            The high precision dtype of the data buffer
        """

        return self.dtype

    def clear(self) -> None:
        """
        Reset tensor data and clear all buffers.
        """
        self.data = None
        self.columnwise_data = None
        self.scale_inv = None
        self.columnwise_scale_inv = None
        self.amax = None
        self.columnwise_amax = None
        self.scale = None
        self.first_dims = None
        self.last_dims = None
        self.tensor_offsets = None
        self.logical_shape = (0, 0)
        self.num_tensors = 0
        self.quantizers = None
        self.quantized_tensors = None
        self.offsets = None
        self.scale_inv_offsets = None
        self.columnwise_scale_inv_offsets = None

    def __repr__(self) -> str:
        """String representation of the GroupedTensor."""
        return (
            f"GroupedTensor(num_tensors={self.num_tensors}, "
            f"shape={self.shape}, "
            f"logical_shape={self.logical_shape}, "
            f"dtype={self.get_dtype()})"
        )

    def __str__(self) -> str:
        """User-friendly string representation."""
        shape_info = []
        if self.all_same_shape():
            shape_info.append("uniform shape")
        else:
            if not self.all_same_first_dim():
                shape_info.append("varying first dim")
            if not self.all_same_last_dim():
                shape_info.append("varying last dim")

        return (
            f"GroupedTensor with {self.num_tensors} tensors "
            f"({', '.join(shape_info) if shape_info else 'uniform'}), "
            f"logical_shape={self.logical_shape}, "
            f"dtype={self.get_dtype()}"
        )

    @staticmethod
    def make_grouped_tensor(
        num_tensors: int,
        shape: List[Tuple[int, int]],
        quantizers: List[Optional[Quantizer]] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> GroupedTensor:
        """
        Create a GroupedTensor for storing multiple weight tensors of the same shape.

        Args:
            num_tensors: Number of tensors
            shape: 2D shape of each tensor (len num_tensors)
            quantizers: List of quantizers for each tensor (len num_tensors)
                        Used to figure out the recipe and what to allocate.
            device: Device to allocate tensors on, defaults to current cuda device
            dtype: Data type of the tensor (for high precision case)

        Returns:
            A GroupedTensor.
        """
        # Input validation
        assert (
            len(shape) == num_tensors
        ), f"Shape list length {len(shape)} must match num_tensors {num_tensors}"
        assert all(len(s) == 2 for s in shape), "All shapes must be 2D tuples"
        assert all(s[0] > 0 and s[1] > 0 for s in shape), "All dimensions must be positive"

        # Set device
        if device is None:
            device = torch.cuda.current_device()

        # Analyze shape patterns
        first_dims_list = [s[0] for s in shape]
        last_dims_list = [s[1] for s in shape]

        all_same_first = len(set(first_dims_list)) == 1
        all_same_last = len(set(last_dims_list)) == 1

        # Create dimension arrays if needed
        first_dims = (
            None
            if all_same_first
            else torch.tensor(first_dims_list, dtype=torch.int64, device=device)
        )
        last_dims = (
            None
            if all_same_last
            else torch.tensor(last_dims_list, dtype=torch.int64, device=device)
        )

        # Calculate tensor offsets (cumulative element offsets)
        tensor_offsets = None
        offsets = None
        if not (all_same_first and all_same_last):
            # Need explicit offsets for non-uniform shapes
            # Offsets are based on number of elements and not pointers.
            # Kernels need to calculate precise pointers based on size of elements.
            numels = [s[0] * s[1] for s in shape]
            offsets = [0]
            for i in range(num_tensors - 1):
                offsets.append(offsets[-1] + numels[i])
            tensor_offsets = torch.tensor(offsets, dtype=torch.int64, device=device)

        # Calculate logical shape based on shape pattern
        if all_same_first and all_same_last:
            # All same shape: [num_tensors * M, N]
            M, N = shape[0]
            logical_shape = (num_tensors * M, N)
        elif all_same_first and not all_same_last:
            # Varying last dim only: [M, sum_of_last_dims]
            M = first_dims_list[0]
            sum_last = sum(last_dims_list)
            logical_shape = (M, sum_last)
        elif not all_same_first and all_same_last:
            # Varying first dim only: [sum_of_first_dims, N]
            sum_first = sum(first_dims_list)
            N = last_dims_list[0]
            logical_shape = (sum_first, N)
        else:
            # Varying both dims: [1, total_elements]
            total_elements = sum(s[0] * s[1] for s in shape)
            logical_shape = (1, total_elements)

        no_quantization = quantizers is None or len(quantizers) == 0 or quantizers[0] is None

        # TODO(ksivaman): (Do we need multiple quantizers?)
        # Current implementation assumes all tensors have the different quantizers.
        # instances but effectively the same quantizer.
        rowwise_usage = quantizers[0].rowwise_usage if not no_quantization else True
        columnwise_usage = quantizers[0].columnwise_usage if not no_quantization else False

        # Calculate total elements across all tensors
        total_elements = sum(s[0] * s[1] for s in shape)

        data = None
        columnwise_data = None
        scale_inv = None
        columnwise_scale_inv = None
        amax = None
        columnwise_amax = None
        scale = None
        scale_inv_offsets = None
        columnwise_scale_inv_offsets = None
        if no_quantization:
            assert dtype is not None, "dtype must be provided for unquantized GroupedTensor"
            if rowwise_usage:
                # Allocate rowwise data buffer (1D flattened, uint8)
                data = torch.empty(total_elements, dtype=dtype, device=device)

            if columnwise_usage:
                # Allocate columnwise data buffer (1D flattened, uint8)
                columnwise_data = torch.empty(total_elements, dtype=dtype, device=device)
        elif quantizers[0]._get_compatible_recipe().mxfp8():
            if rowwise_usage:
                # Allocate rowwise data buffer (1D flattened, uint8)
                data = torch.empty(total_elements, dtype=torch.uint8, device=device)
                # Scale inverse buffer for MXFP8 - complex shape based on block scaling
                # For grouped tensors, we need to calculate scale_inv size for all tensors
                total_scale_elements = 0
                scale_inv_offsets = [0]
                for i, s in enumerate(shape):
                    scale_inv_shape = quantizers[i].get_scale_shape(s, False)
                    scale_elements = math.prod(scale_inv_shape)
                    total_scale_elements += scale_elements
                    if i < num_tensors - 1:
                        scale_inv_offsets.append(total_scale_elements)
                scale_inv = torch.empty(total_scale_elements, dtype=torch.uint8, device=device)

            if columnwise_usage:
                # Allocate columnwise data buffer (1D flattened, uint8)
                columnwise_data = torch.empty(total_elements, dtype=torch.uint8, device=device)
                # Columnwise scale inverse buffer
                total_columnwise_scale_elements = 0
                columnwise_scale_inv_offsets = [0]
                for i, s in enumerate(shape):
                    scale_inv_shape = quantizers[i].get_scale_shape(s, False)
                    columnwise_scale_elements = math.prod(scale_inv_shape)
                    total_columnwise_scale_elements += columnwise_scale_elements
                    if i < num_tensors - 1:
                        columnwise_scale_inv_offsets.append(total_columnwise_scale_elements)
                columnwise_scale_inv = torch.empty(
                    total_columnwise_scale_elements, dtype=torch.uint8, device=device
                )
        elif quantizers[0]._get_compatible_recipe().delayed():
            if rowwise_usage:
                # Allocate rowwise data buffer (1D flattened, uint8)
                data = torch.empty(total_elements, dtype=torch.uint8, device=device)
                # Scale inverse - one per tensor
                scale_inv = torch.empty(num_tensors, dtype=torch.float32, device=device)
                # One scale per tensor, so offsets are simply 0, 1, 2, ..., num_tensors-1
                scale_inv_offsets = list(range(num_tensors))

            if columnwise_usage:
                # Allocate columnwise data buffer (1D flattened, uint8)
                columnwise_data = torch.empty(total_elements, dtype=torch.uint8, device=device)
                # Columnwise scale inverse - one per tensor
                columnwise_scale_inv = torch.empty(num_tensors, dtype=torch.float32, device=device)
                # One scale per tensor, so offsets are simply 0, 1, 2, ..., num_tensors-1
                columnwise_scale_inv_offsets = list(range(num_tensors))

            # Amax buffer for delayed scaling - one per tensor
            amax = torch.empty(num_tensors, dtype=torch.float32, device=device)
        elif quantizers[0]._get_compatible_recipe().nvfp4():

            if rowwise_usage:
                # Allocate rowwise data buffer (1D flattened, uint8, but FP4 packs 2 values per byte)
                data = torch.empty((total_elements) // 2, dtype=torch.uint8, device=device)
                # Scale inverse buffer for NVFP4 - complex shape based on block scaling
                # For simplicity, calculate total scale elements needed
                total_scale_elements = 0
                scale_inv_offsets = [0]
                for i, s in enumerate(shape):
                    scale_inv_shape = quantizers[i].get_scale_shape(s, False)
                    total_scale_elements += math.prod(scale_inv_shape)
                    if i < num_tensors - 1:
                        scale_inv_offsets.append(total_scale_elements)
                scale_inv = torch.empty(total_scale_elements, dtype=torch.uint8, device=device)
                # Amax buffer - one per tensor
                amax = torch.empty(num_tensors, dtype=torch.float32, device=device)

            if columnwise_usage:
                # Allocate columnwise data buffer (1D flattened, uint8, FP4 packed)
                columnwise_data = torch.empty(
                    (total_elements) // 2, dtype=torch.uint8, device=device
                )
                # Columnwise scale inverse buffer
                total_columnwise_scale_elements = 0
                columnwise_scale_inv_offsets = [0]
                for i, s in enumerate(shape):
                    columnwise_scale_inv_shape = quantizers[i].get_scale_shape(s, True)
                    total_columnwise_scale_elements += math.prod(columnwise_scale_inv_shape)
                    if i < num_tensors - 1:
                        columnwise_scale_inv_offsets.append(total_columnwise_scale_elements)
                columnwise_scale_inv = torch.empty(
                    total_columnwise_scale_elements, dtype=torch.uint8, device=device
                )
                # Columnwise amax buffer - one per tensor
                columnwise_amax = torch.empty(num_tensors, dtype=torch.float32, device=device)
        elif quantizers[0]._get_compatible_recipe().float8_block_scaling():
            if rowwise_usage:
                # Allocate rowwise data buffer (1D flattened, uint8)
                data = torch.empty(total_elements, dtype=torch.uint8, device=device)
                # Scale inverse - size depends on block configuration
                # For simplicity, calculate total scale elements needed
                total_scale_elements = 0
                scale_inv_offsets = [0]
                for i, s in enumerate(shape):
                    scale_inv_shape = quantizers[i].get_scale_shape(s, False)
                    total_scale_elements += math.prod(scale_inv_shape)
                    if i < num_tensors - 1:
                        scale_inv_offsets.append(total_scale_elements)
                scale_inv = torch.empty(total_scale_elements, dtype=torch.float32, device=device)

            if columnwise_usage:
                # Allocate columnwise data buffer (1D flattened, uint8)
                columnwise_data = torch.empty(total_elements, dtype=torch.uint8, device=device)
                # Columnwise scale inverse
                total_columnwise_scale_elements = 0
                columnwise_scale_inv_offsets = [0]
                for i, s in enumerate(shape):
                    columnwise_scale_inv_shape = quantizers[i].get_scale_shape(s, True)
                    total_columnwise_scale_elements += math.prod(columnwise_scale_inv_shape)
                    if i < num_tensors - 1:
                        columnwise_scale_inv_offsets.append(total_columnwise_scale_elements)
                columnwise_scale_inv = torch.empty(
                    total_columnwise_scale_elements, dtype=torch.float32, device=device
                )
        elif quantizers[0]._get_compatible_recipe().float8_current_scaling():
            # Current scaling - per-tensor scaling computed on the fly
            if rowwise_usage:
                # Allocate rowwise data buffer (1D flattened, uint8)
                data = torch.empty(total_elements, dtype=torch.uint8, device=device)
                # Scale inverse - one per tensor
                scale_inv = torch.empty(num_tensors, dtype=torch.float32, device=device)
                # One scale per tensor, so offsets are simply 0, 1, 2, ..., num_tensors-1
                scale_inv_offsets = list(range(num_tensors))

            if columnwise_usage:
                # Allocate columnwise data buffer (1D flattened, uint8)
                columnwise_data = torch.empty(total_elements, dtype=torch.uint8, device=device)
                # Columnwise scale inverse - one per tensor
                columnwise_scale_inv = torch.empty(num_tensors, dtype=torch.float32, device=device)
                # One scale per tensor, so offsets are simply 0, 1, 2, ..., num_tensors-1
                columnwise_scale_inv_offsets = list(range(num_tensors))

            # Scale and amax buffers for current scaling - one per tensor
            scale = torch.empty(num_tensors, dtype=torch.float32, device=device)
            amax = torch.empty(num_tensors, dtype=torch.float32, device=device)
        else:
            raise ValueError(f"Unsupported quantizer for GroupedTensor: {quantizers[0]}")

        grouped_tensor = GroupedTensor(
            num_tensors=num_tensors,
            shape=shape,
            dtype=dtype,
            quantizers=quantizers,
            data=data,
            columnwise_data=columnwise_data,
            scale_inv=scale_inv,
            columnwise_scale_inv=columnwise_scale_inv,
            amax=amax,
            columnwise_amax=columnwise_amax,
            scale=scale,
            first_dims=first_dims,
            last_dims=last_dims,
            tensor_offsets=tensor_offsets,
            offsets=offsets,
            scale_inv_offsets=scale_inv_offsets,
            columnwise_scale_inv_offsets=columnwise_scale_inv_offsets,
            logical_shape=logical_shape,
        )

        grouped_tensor.quantized_tensors = grouped_tensor.split_into_quantized_tensors()
        return grouped_tensor

    def split_into_quantized_tensors(
        self,
    ) -> List[Union[QuantizedTensorStorage, torch.Tensor]]:
        """
        Split the GroupedTensor into a list of `num_tensors`
        quantized tensors based on the quantizer. No additional memory allocation is performed,
        so the tensors returned are the same as the ones used to create the GroupedTensor.

        If quantizer is None, returns normal torch tensors.
        If quantizer.internal is True, returns QuantizedTensorStorage.
        Otherwise, returns QuantizedTensor.
        """

        result = []

        no_quantization = (
            self.quantizers is None or len(self.quantizers) == 0 or self.quantizers[0] is None
        )

        # Case 1: No quantization - return regular torch tensors
        if no_quantization:
            for i in range(self.num_tensors):
                # Get tensor shape
                tensor_shape = self.shape[i]

                # Get tensor data slice
                if self.offsets is not None:
                    start_offset = self.offsets[i]
                    numel = tensor_shape[0] * tensor_shape[1]
                    end_offset = start_offset + numel

                    if self.has_data():
                        tensor_data = self.data[start_offset:end_offset].view(tensor_shape)
                        result.append(tensor_data)
                    elif self.has_columnwise_data():
                        tensor_data = self.columnwise_data[start_offset:end_offset].view(
                            tensor_shape
                        )
                        result.append(tensor_data)
                    else:
                        raise RuntimeError("GroupedTensor has no data to split")
                else:
                    # All same shape case
                    numel = tensor_shape[0] * tensor_shape[1]
                    start_offset = i * numel
                    end_offset = start_offset + numel

                    if self.has_data():
                        tensor_data = self.data[start_offset:end_offset].view(tensor_shape)
                        result.append(tensor_data)
                    elif self.has_columnwise_data():
                        tensor_data = self.columnwise_data[start_offset:end_offset].view(
                            tensor_shape
                        )
                        result.append(tensor_data)
                    else:
                        raise RuntimeError("GroupedTensor has no data to split")

            return result

        # Case 2: Quantized tensors
        recipe = self.quantizers[0]._get_compatible_recipe()

        for i in range(self.num_tensors):
            # Get tensor shape
            tensor_shape = self.shape[i]
            numel = tensor_shape[0] * tensor_shape[1]

            # Get data offsets
            if self.offsets is not None:
                data_start = self.offsets[i]
                data_end = data_start + numel
            else:
                # All same shape
                data_start = i * numel
                data_end = data_start + numel

            # Special shape handling for NVFP4.
            nvfp4 = self.quantizers[i]._get_compatible_recipe().nvfp4()
            if nvfp4:
                data_start = data_start // 2
                data_end = data_end // 2

            # Extract rowwise and columnwise data
            rowwise_data = None
            columnwise_data = None

            if self.has_data():
                if nvfp4:
                    rowwise_tensor_shape = self.quantizers[i].convert_shape_for_fp4(tensor_shape)
                else:
                    rowwise_tensor_shape = tensor_shape
                rowwise_data = self.data[data_start:data_end].view(rowwise_tensor_shape)

            if self.has_columnwise_data():
                columnwise_tensor_shape = self.quantizers[i].get_columnwise_shape(tensor_shape)
                if nvfp4:
                    columnwise_tensor_shape = self.quantizers[i].convert_shape_for_fp4(
                        columnwise_tensor_shape
                    )
                columnwise_data = self.columnwise_data[data_start:data_end].view(
                    columnwise_tensor_shape
                )

            # MXFP8 format
            if recipe.mxfp8():
                # Extract scale_inv data
                rowwise_scale_inv = None
                columnwise_scale_inv = None

                if self.scale_inv is not None and self.scale_inv_offsets is not None:
                    scale_start = self.scale_inv_offsets[i]
                    if i < self.num_tensors - 1:
                        scale_end = self.scale_inv_offsets[i + 1]
                    else:
                        scale_end = self.scale_inv.numel()

                    # Calculate expected scale shape for MXFP8
                    scale_shape = self.quantizers[i].get_scale_shape(tensor_shape, False)
                    rowwise_scale_inv = self.scale_inv[scale_start:scale_end].view(scale_shape)

                if (
                    self.columnwise_scale_inv is not None
                    and self.columnwise_scale_inv_offsets is not None
                ):
                    cscale_start = self.columnwise_scale_inv_offsets[i]
                    if i < self.num_tensors - 1:
                        cscale_end = self.columnwise_scale_inv_offsets[i + 1]
                    else:
                        cscale_end = self.columnwise_scale_inv.numel()

                    cscale_shape = self.quantizers[i].get_scale_shape(tensor_shape, True)
                    columnwise_scale_inv = self.columnwise_scale_inv[cscale_start:cscale_end].view(
                        cscale_shape
                    )

                if self.quantizers[i].internal:
                    mxfp8_tensor_class = MXFP8TensorStorage
                else:
                    mxfp8_tensor_class = MXFP8Tensor
                tensor = mxfp8_tensor_class(
                    shape=tensor_shape,
                    dtype=self.dtype,
                    rowwise_data=rowwise_data,
                    rowwise_scale_inv=rowwise_scale_inv,
                    columnwise_data=columnwise_data,
                    columnwise_scale_inv=columnwise_scale_inv,
                    fp8_dtype=self.quantizers[i].dtype,
                    quantizer=self.quantizers[i],
                    with_gemm_swizzled_scales=self.quantizers[i].optimize_for_gemm,
                )
                result.append(tensor)

            # Delayed scaling or current scaling (both use Float8TensorStorage)
            elif recipe.delayed() or recipe.float8_current_scaling():
                # Scale inverse - one per tensor
                scale_inv = None
                if self.scale_inv is not None:
                    scale_inv = self.scale_inv[i : i + 1]

                if self.quantizers[i].internal:
                    float8_tensor_class = Float8TensorStorage
                else:
                    float8_tensor_class = Float8Tensor

                tensor = float8_tensor_class(
                    shape=tensor_shape,
                    dtype=self.dtype,
                    data=rowwise_data,
                    fp8_scale_inv=scale_inv,
                    fp8_dtype=self.quantizers[i].dtype,
                    quantizer=self.quantizers[i],
                    data_transpose=columnwise_data,
                )
                result.append(tensor)

            # Float8 block scaling
            elif recipe.float8_block_scaling():
                # Extract scale_inv data
                rowwise_scale_inv = None
                columnwise_scale_inv = None

                if self.scale_inv is not None and self.scale_inv_offsets is not None:
                    scale_start = self.scale_inv_offsets[i]
                    if i < self.num_tensors - 1:
                        scale_end = self.scale_inv_offsets[i + 1]
                    else:
                        scale_end = self.scale_inv.numel()

                    # Get scale shape from quantizer
                    scale_shape = self.quantizers[i].get_scale_shape(tensor_shape, False)
                    rowwise_scale_inv = self.scale_inv[scale_start:scale_end].view(scale_shape)

                if (
                    self.columnwise_scale_inv is not None
                    and self.columnwise_scale_inv_offsets is not None
                ):
                    cscale_start = self.columnwise_scale_inv_offsets[i]
                    if i < self.num_tensors - 1:
                        cscale_end = self.columnwise_scale_inv_offsets[i + 1]
                    else:
                        cscale_end = self.columnwise_scale_inv.numel()

                    # Get columnwise scale shape from quantizer
                    cscale_shape = self.quantizers[i].get_scale_shape(tensor_shape, True)
                    columnwise_scale_inv = self.columnwise_scale_inv[cscale_start:cscale_end].view(
                        cscale_shape
                    )

                # Compute is_2D_scaled and data_format from quantizer attributes
                is_2D_scaled = self.quantizers[i].block_scaling_dim == 2

                if self.quantizers[i].internal:
                    float8_blockwise_q_tensor_class = Float8BlockwiseQTensorStorage
                else:
                    float8_blockwise_q_tensor_class = Float8BlockwiseQTensor

                tensor = float8_blockwise_q_tensor_class(
                    shape=tensor_shape,
                    dtype=self.dtype,
                    rowwise_data=rowwise_data,
                    rowwise_scale_inv=rowwise_scale_inv,
                    columnwise_data=columnwise_data,
                    columnwise_scale_inv=columnwise_scale_inv,
                    fp8_dtype=self.quantizers[i].dtype,
                    quantizer=self.quantizers[i],
                    is_2D_scaled=is_2D_scaled,
                )
                result.append(tensor)

            # NVFP4 format
            elif recipe.nvfp4():
                # Extract scale_inv data
                rowwise_scale_inv = None
                columnwise_scale_inv = None
                amax_rowwise = None
                amax_columnwise = None

                if self.scale_inv is not None and self.scale_inv_offsets is not None:
                    scale_start = self.scale_inv_offsets[i]
                    if i < self.num_tensors - 1:
                        scale_end = self.scale_inv_offsets[i + 1]
                    else:
                        scale_end = self.scale_inv.numel()

                    # Get scale shape from quantizer
                    scale_shape = self.quantizers[i].get_scale_shape(tensor_shape, False)
                    rowwise_scale_inv = self.scale_inv[scale_start:scale_end].view(scale_shape)

                if (
                    self.columnwise_scale_inv is not None
                    and self.columnwise_scale_inv_offsets is not None
                ):
                    cscale_start = self.columnwise_scale_inv_offsets[i]
                    if i < self.num_tensors - 1:
                        cscale_end = self.columnwise_scale_inv_offsets[i + 1]
                    else:
                        cscale_end = self.columnwise_scale_inv.numel()

                    # Get columnwise scale shape from quantizer
                    cscale_shape = self.quantizers[i].get_scale_shape(tensor_shape, True)
                    columnwise_scale_inv = self.columnwise_scale_inv[cscale_start:cscale_end].view(
                        cscale_shape
                    )

                # Extract amax - one per tensor
                if self.amax is not None:
                    amax_rowwise = self.amax[i : i + 1]

                if self.columnwise_amax is not None:
                    amax_columnwise = self.columnwise_amax[i : i + 1]

                if self.quantizers[i].internal:
                    nvfp4_tensor_class = NVFP4TensorStorage
                else:
                    nvfp4_tensor_class = NVFP4Tensor

                tensor = nvfp4_tensor_class(
                    shape=tensor_shape,
                    dtype=self.dtype,
                    rowwise_data=rowwise_data,
                    rowwise_scale_inv=rowwise_scale_inv,
                    columnwise_data=columnwise_data,
                    columnwise_scale_inv=columnwise_scale_inv,
                    amax_rowwise=amax_rowwise,
                    amax_columnwise=amax_columnwise,
                    fp4_dtype=self.quantizers[i].dtype,
                    quantizer=self.quantizers[i],
                    with_gemm_swizzled_scales=self.quantizers[i].optimize_for_gemm,
                )
                result.append(tensor)

            else:
                raise ValueError(f"Unsupported quantization recipe: {recipe}")

        return result

    @staticmethod
    def create_and_quantize(
        tensors: int,
        quantizers: None | List[Quantizer],
        *,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        noop_flag: Optional[torch.Tensor] = None,
    ) -> Tuple[QuantizedTensorStorage, ...]:
        """
        Quantize given tensors into quantized tensors with underlying
        storage allocated in a GroupedTensor.
        """

        num_tensors = len(tensors)

        if quantizers is not None:
            assert num_tensors == len(quantizers), "Number of tensors and quantizers must match"

        grouped_tensor = GroupedTensor.make_grouped_tensor(
            num_tensors=len(tensors),
            shape=[t.shape for t in tensors],
            quantizers=quantizers,
            device=device,
            dtype=dtype,
        )

        grouped_tensor.quantize(tensors, noop_flag=noop_flag)

        return grouped_tensor

    def quantize(
        self,
        tensors: List[torch.Tensor],
        noop_flag: Optional[torch.Tensor] = None,
    ) -> Tuple[QuantizedTensorStorage, ...]:
        """
        Quantize the GroupedTensor inplace.
        """

        quantized_tensors = self.split_into_quantized_tensors()
        for i in range(self.num_tensors):
            self.quantizers[i].update_quantized(
                tensors[i], quantized_tensors[i], noop_flag=noop_flag
            )
        return quantized_tensors
