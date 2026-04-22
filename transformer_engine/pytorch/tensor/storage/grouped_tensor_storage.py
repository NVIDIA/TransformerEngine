# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Grouped tensor storage class for handling collections of tensors with different shapes"""
from __future__ import annotations
from typing import Optional, Tuple, List, Union
import math

import torch
from ...quantized_tensor import QuantizedTensorStorage, Quantizer

from ..mxfp8_tensor import MXFP8Tensor
from ..nvfp4_tensor import NVFP4Tensor
from ..float8_tensor import Float8Tensor
from ..float8_blockwise_tensor import Float8BlockwiseQTensor
from .float8_tensor_storage import Float8TensorStorage
from .mxfp8_tensor_storage import MXFP8TensorStorage
from .float8_blockwise_tensor_storage import Float8BlockwiseQTensorStorage
from .nvfp4_tensor_storage import NVFP4TensorStorage


class GroupedTensorStorage:
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

    @staticmethod
    def _initialize_storage_fields(
        instance: "GroupedTensorStorage",
        shape: Tuple[int, int],
        dtype: torch.dtype,
        num_tensors: int,
        shapes: Optional[List[Tuple[int, ...]]] = None,
        quantizer: Optional[Quantizer] = None,
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
        requires_grad: bool = False,
        stride: Optional[List[int]] = None,
        with_gemm_swizzled_scales: bool = False,
    ) -> None:
        """
        Initialize a GroupedTensor.

        Args:
            shape: 2D tuple representing conceptual shape
            dtype: Data type of the grouped tensor
            num_tensors: Number of tensors in the group
            shapes: 2D shape of each tensor (len num_tensors)
            quantizer: Quantizer used for all tensors in the group
            data: Row-wise data buffer (1D flattened)
            columnwise_data: Column-wise data buffer (1D flattened)
            scale_inv: Row-wise scale inverse buffer
            columnwise_scale_inv: Column-wise scale inverse buffer
            amax: Row-wise amax buffer
            columnwise_amax: Column-wise amax buffer
            scale: Scale buffer (for FP8-DS only)
            first_dims: Device tensor of int64 array of length num_tensors (or None if uniform)
            last_dims: Device tensor of int64 array of length num_tensors (or None if uniform)
            tensor_offsets: Device tensor of int64 array of length num_tensors+1 (CSR-style,
                or None if uniform). offsets[i] = start of tensor i, offsets[num_tensors] = total.
            offsets: Vector of integer offsets for each tensor.
        """
        # `requires_grad` and `stride` are accepted for API symmetry with
        # GroupedTensor.__new__ but are not relevant for storage-only
        # initialization; they are intentionally ignored here.
        del requires_grad
        del stride

        instance.num_tensors = num_tensors
        instance.quantizer = quantizer
        instance.tensor_shapes = shapes
        instance.fake_dtype = dtype

        # Data buffers
        instance.rowwise_data = data
        instance.columnwise_data = columnwise_data
        instance.scale_inv = scale_inv
        instance.columnwise_scale_inv = columnwise_scale_inv
        instance.amax = amax
        instance.columnwise_amax = columnwise_amax
        instance.scale = scale

        # For convenient indexing for python GroupedTensor API.
        instance.scale_inv_offsets = scale_inv_offsets
        instance.columnwise_scale_inv_offsets = columnwise_scale_inv_offsets

        # Shape information (OPTIONAL - None if dimension is uniform across all tensors)
        # first_dims[i] = first dimension of tensor i (None if all tensors have same first dim)
        # last_dims[i] = last dimension of tensor i (None if all tensors have same last dim)
        instance.first_dims = (
            first_dims  # Device pointer to int64_t array of length num_tensors (or None)
        )
        instance.last_dims = (
            last_dims  # Device pointer to int64_t array of length num_tensors (or None)
        )

        # Offsets for indexing into contiguous 1D layout (OPTIONAL - not needed if all_same_shape())
        # tensor_offsets[i] = element offset to start of tensor i (cumulative sum of numel for tensors 0..i-1)
        # Usage: tensor_i_ptr = data.data_ptr() + tensor_offsets[i] * element_size
        # If None and all_same_shape(): offset[i] = i * M * N (where M, N are common dimensions)
        instance.tensor_offsets = (
            tensor_offsets  # Device pointer to int64_t array of length num_tensors (or None)
        )
        instance.offsets = offsets  # Vector of integer offsets for each tensor.

        # Logical shape: conceptual 2D shape of the grouped data (REQUIRED)
        # Represents how the 1D flattened data should be interpreted as 2D
        # Always 2D with positive dimensions
        instance.logical_shape = shape

        # Hold a reference to the quantized tensors that occupy same storage as the GroupedTensor.
        # Used as a convenience.
        instance.quantized_tensors = None
        instance._with_gemm_swizzled_scales = with_gemm_swizzled_scales

    def __new__(
        cls,
        shape: Tuple[int, int],
        dtype: torch.dtype,
        *,
        num_tensors: int,
        shapes: Optional[List[Tuple[int, ...]]] = None,
        quantizer: Optional[Quantizer] = None,
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
        requires_grad: bool = False,
        stride: Optional[List[int]] = None,
        with_gemm_swizzled_scales: bool = False,
    ):
        instance = object.__new__(cls)
        cls._initialize_storage_fields(
            instance=instance,
            shape=shape,
            dtype=dtype,
            num_tensors=num_tensors,
            shapes=shapes,
            quantizer=quantizer,
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
            requires_grad=requires_grad,
            stride=stride,
            with_gemm_swizzled_scales=with_gemm_swizzled_scales,
        )
        return instance

    def has_data(self) -> bool:
        """
        Check if the tensor has row-wise data.

        Returns:
            True if data buffer is initialized, False otherwise
        """
        return self.rowwise_data is not None

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

        return self.fake_dtype

    def clear(self) -> None:
        """
        Reset tensor data and clear all buffers.
        """
        self.rowwise_data = None
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
        self.quantizer = None
        self.quantized_tensors = None
        self.offsets = None
        self.scale_inv_offsets = None
        self.columnwise_scale_inv_offsets = None
        self.tensor_shapes = []
        self.fake_dtype = torch.float32

    def __repr__(self) -> str:
        """String representation of the GroupedTensorStorage."""
        return (
            f"GroupedTensorStorage(num_tensors={self.num_tensors}, "
            f"shapes={self.tensor_shapes}, "
            f"logical_shape={self.logical_shape}, "
            f"quantizer={self.quantizer}, "
            f"dtype={self.get_dtype()})"
        )

    @staticmethod
    def make_grouped_tensor_with_shapes(
        num_tensors: int,
        shapes: List[Tuple[int, int]],
        quantizer: Optional[Quantizer] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> GroupedTensorStorage:
        """
        Create a GroupedTensor for storing multiple weight tensors of the same shape.

        Args:
            num_tensors: Number of tensors
            shapes: 2D shape of each tensor (len num_tensors)
            quantizer: Quantizer used for all tensors
            device: Device to allocate tensors on, defaults to current cuda device
            dtype: Data type of the tensor (for high precision case)

        Returns:
            A GroupedTensor.
        """

        # First dim
        first_dim_list = [s[0] for s in shapes]
        uniform_first_dim = all(first_dim_list[0] == x for x in first_dim_list)
        logical_first_dim = sum(first_dim_list)
        if uniform_first_dim:
            first_dims = None
        else:
            first_dims = torch.tensor([s[0] for s in shapes], dtype=torch.int64, device=device)

        # Last dim
        last_dim_list = [s[1] for s in shapes]
        logical_last_dim = last_dim_list[0]
        assert all(logical_last_dim == x for x in last_dim_list), "Last dims should be uniform"

        return GroupedTensorStorage.make_grouped_tensor(
            num_tensors=num_tensors,
            first_dims=first_dims,
            last_dims=None,
            logical_first_dim=logical_first_dim,
            logical_last_dim=logical_last_dim,
            quantizer=quantizer,
            device=device,
            dtype=dtype,
        )

    @staticmethod
    def make_grouped_tensor_from_rowwise_data(
        *,
        num_tensors: int,
        tensor_shape: Tuple[int, ...],
        rowwise_data: torch.Tensor,
        dtype: Optional[torch.dtype] = None,
        internal: bool = False,
    ) -> GroupedTensorStorage:
        """Wrap pre-existing contiguous rowwise data as a grouped tensor.

        This helper does not allocate storage. It creates grouped metadata over
        `rowwise_data`, which is expected to contain `num_tensors` tensors of
        shape ``tensor_shape`` in packed contiguous layout.

        ``tensor_shape`` may be:

        * ``(rows, cols)`` — each member is a 2D matrix; wrapper shape
          ``(num_tensors, rows, cols)``.
        * ``(n,)`` — each member is a 1D vector of length ``n``; logical storage
          uses ``logical_shape = (num_tensors * n, 1)`` and the wrapper shape is
          ``(num_tensors, n)``.
        """
        if num_tensors <= 0:
            raise ValueError(f"num_tensors must be positive, got {num_tensors}")
        if rowwise_data is None:
            raise ValueError("rowwise_data must not be None")
        if not rowwise_data.is_contiguous():
            rowwise_data = rowwise_data.contiguous()

        if len(tensor_shape) == 2:
            rows, cols = tensor_shape
            expected_numel = num_tensors * rows * cols
            logical_shape = (num_tensors * rows, cols)
            shapes_list: List[Tuple[int, ...]] = [tensor_shape] * num_tensors
        elif len(tensor_shape) == 1:
            (n,) = tensor_shape
            expected_numel = num_tensors * n
            logical_shape = (num_tensors * n, 1)
            shapes_list = [tensor_shape] * num_tensors
        else:
            raise ValueError(
                "tensor_shape must be 1D (n,) or 2D (rows, cols), "
                f"got {tensor_shape!r} with length {len(tensor_shape)}"
            )

        if rowwise_data.numel() != expected_numel:
            raise ValueError(
                "Grouped rowwise buffer size mismatch: expected "
                f"{expected_numel} elements for {num_tensors}x{tensor_shape}, "
                f"but got {rowwise_data.numel()}"
            )
        if dtype is None:
            dtype = rowwise_data.dtype
        grouped_tensor_class = GroupedTensorStorage
        if not internal:
            from ..grouped_tensor import GroupedTensor

            grouped_tensor_class = GroupedTensor

        return grouped_tensor_class(
            shape=logical_shape,
            dtype=dtype,
            num_tensors=num_tensors,
            shapes=shapes_list,
            quantizer=None,
            data=rowwise_data.view(-1),
            columnwise_data=None,
            scale_inv=None,
            columnwise_scale_inv=None,
            amax=None,
            columnwise_amax=None,
            scale=None,
            first_dims=None,
            last_dims=None,
            tensor_offsets=None,
            offsets=None,
            scale_inv_offsets=None,
            columnwise_scale_inv_offsets=None,
            with_gemm_swizzled_scales=False,
            requires_grad=False,
        )

    def copy(self) -> "GroupedTensorStorage":
        """Create a shallow copy that shares all data buffers with *self*.
        No tensor data is copied; the returned object references the same
        underlying storage for every buffer (data, scales, offsets, etc.).
        This is useful when you need to mutate metadata (e.g. swizzle
        scales in-place) without affecting the original object.
        """
        return GroupedTensorStorage(
            shape=self.logical_shape,
            dtype=self.fake_dtype,
            num_tensors=self.num_tensors,
            shapes=self.tensor_shapes,
            quantizer=self.quantizer,
            data=self.rowwise_data,
            columnwise_data=self.columnwise_data,
            scale_inv=self.scale_inv,
            columnwise_scale_inv=self.columnwise_scale_inv,
            amax=self.amax,
            columnwise_amax=self.columnwise_amax,
            scale=self.scale,
            first_dims=self.first_dims,
            last_dims=self.last_dims,
            tensor_offsets=self.tensor_offsets,
            offsets=self.offsets,
            scale_inv_offsets=self.scale_inv_offsets,
            columnwise_scale_inv_offsets=self.columnwise_scale_inv_offsets,
            with_gemm_swizzled_scales=self._with_gemm_swizzled_scales,
        )

    @staticmethod
    def make_tensor_offsets(first_dims: torch.Tensor, logical_last_dim: int) -> torch.Tensor:
        """Calculate GPU offsets from first dim splits."""
        return torch.cat(
            [
                torch.zeros(1, device=first_dims.device, dtype=first_dims.dtype),
                torch.cumsum(first_dims * logical_last_dim, dim=0),
            ]
        )

    @staticmethod
    def make_grouped_tensor(
        num_tensors: int,
        first_dims: Optional[torch.Tensor],
        last_dims: Optional[torch.Tensor],
        logical_first_dim: int,
        logical_last_dim: int,
        quantizer: Optional[Quantizer] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> GroupedTensorStorage:
        """
        Create a GroupedTensor for storing multiple weight tensors of the same shape.

        Args:
            num_tensors: Number of tensors
            first_dims: Device tensor of int64 array of length num_tensors (or None if uniform)
            last_dims: Device tensor of int64 array of length num_tensors (or None if uniform)
            logical_first_dim: Logical first dimension
            logical_last_dim: Logical last dimension
            quantizer: Quantizer used for all tensors. Used to figure out recipe
                       and what to allocate.
            device: Device to allocate tensors on, defaults to current cuda device
            dtype: Data type of the tensor (for high precision case)

        Returns:
            A GroupedTensor.
        """

        # Set device
        if device is None:
            device = torch.cuda.current_device()

        # Shape patterns and validation.
        all_same_first = first_dims is None
        all_same_last = last_dims is None

        assert all_same_last, "Last dim must be uniform for GroupedTensor"
        assert logical_first_dim >= 0, "Logical first dim must be non-negative for GroupedTensor"
        assert logical_last_dim > 0, "Logical last dim must be positive for GroupedTensor"

        # assert (
        #     logical_first_dim % 128 == 0
        # ), "Logical first dim must be divisible by 128"
        # assert logical_last_dim % 128 == 0, "Logical last dim must be divisible by 128"

        # Calculate tensor offsets (cumulative element offsets)
        tensor_offsets = None
        offsets = None
        shape = []
        if not all_same_first:
            # Need explicit offsets for non-uniform shapes
            # Offsets are based on number of elements and not pointers.
            # Kernels need to calculate precise pointers based on size of elements.

            # TODO(ksivaman): Single kernel + remove the host offset calculation.
            tensor_offsets = GroupedTensorStorage.make_tensor_offsets(first_dims, logical_last_dim)
            if (
                first_dims.device.type == "cuda"
                and torch.cuda.is_available()
                and torch.cuda.is_current_stream_capturing()
            ):
                # Avoid host sync during CUDA graph capture.
                offsets = None
                shape = None
            else:
                offsets = tensor_offsets.tolist()
                first_dims_list = first_dims.tolist()
                for i in range(num_tensors):
                    shape.append((first_dims_list[i], logical_last_dim))
        else:
            offsets = [
                i * logical_first_dim * logical_last_dim // num_tensors
                for i in range(num_tensors + 1)
            ]
            for i in range(num_tensors):
                shape.append((logical_first_dim // num_tensors, logical_last_dim))

        # Calculate logical shape based
        logical_shape = (logical_first_dim, logical_last_dim)

        no_quantization = quantizer is None

        rowwise_usage = quantizer.rowwise_usage if not no_quantization else True
        columnwise_usage = quantizer.columnwise_usage if not no_quantization else False

        # Calculate total elements across all tensors
        total_elements = logical_first_dim * logical_last_dim

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
        elif quantizer._get_compatible_recipe().mxfp8():
            if rowwise_usage:
                # Allocate rowwise data buffer (1D flattened, uint8)
                data = torch.empty(total_elements, dtype=torch.uint8, device=device)
                # Scale inverse buffer for MXFP8 - complex shape based on block scaling
                # For grouped tensors, we need to calculate scale_inv size for all tensors
                total_scale_elements = 0
                scale_inv_offsets = [0]
                for i, s in enumerate(shape):
                    scale_inv_shape = quantizer.get_scale_shape(s, False)
                    scale_elements = math.prod(scale_inv_shape)
                    total_scale_elements += scale_elements
                    scale_inv_offsets.append(total_scale_elements)
                scale_inv = torch.empty(total_scale_elements, dtype=torch.uint8, device=device)

            if columnwise_usage:
                # Allocate columnwise data buffer (1D flattened, uint8)
                columnwise_data = torch.empty(total_elements, dtype=torch.uint8, device=device)
                # Columnwise scale inverse buffer
                total_columnwise_scale_elements = 0
                columnwise_scale_inv_offsets = [0]
                for i, s in enumerate(shape):
                    scale_inv_shape = quantizer.get_scale_shape(s, True)
                    columnwise_scale_elements = math.prod(scale_inv_shape)
                    total_columnwise_scale_elements += columnwise_scale_elements
                    columnwise_scale_inv_offsets.append(total_columnwise_scale_elements)
                columnwise_scale_inv = torch.empty(
                    total_columnwise_scale_elements, dtype=torch.uint8, device=device
                )
        elif quantizer._get_compatible_recipe().delayed():
            if rowwise_usage:
                # Allocate rowwise data buffer (1D flattened, uint8)
                data = torch.empty(total_elements, dtype=torch.uint8, device=device)
                # Scale inverse - one per tensor
                scale_inv = torch.empty(num_tensors, dtype=torch.float32, device=device)
                # One scale per tensor, so offsets are simply 0, 1, 2, ..., num_tensors
                scale_inv_offsets = list(range(num_tensors + 1))

            if columnwise_usage:
                # Allocate columnwise data buffer (1D flattened, uint8)
                columnwise_data = torch.empty(total_elements, dtype=torch.uint8, device=device)
                # Columnwise scale inverse - one per tensor
                columnwise_scale_inv = torch.empty(num_tensors, dtype=torch.float32, device=device)
                # One scale per tensor, so offsets are simply 0, 1, 2, ..., num_tensors
                columnwise_scale_inv_offsets = list(range(num_tensors + 1))

            # Amax buffer for delayed scaling - one per tensor
            amax = torch.empty(num_tensors, dtype=torch.float32, device=device)
        elif quantizer._get_compatible_recipe().nvfp4():

            if rowwise_usage:
                # Allocate rowwise data buffer (1D flattened, uint8, but FP4 packs 2 values per byte)
                data = torch.empty((total_elements) // 2, dtype=torch.uint8, device=device)
                # Scale inverse buffer for NVFP4 - complex shape based on block scaling
                # For simplicity, calculate total scale elements needed
                total_scale_elements = 0
                scale_inv_offsets = [0]
                for i, s in enumerate(shape):
                    scale_inv_shape = quantizer.get_scale_shape(s, False)
                    total_scale_elements += math.prod(scale_inv_shape)
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
                    columnwise_scale_inv_shape = quantizer.get_scale_shape(s, True)
                    total_columnwise_scale_elements += math.prod(columnwise_scale_inv_shape)
                    columnwise_scale_inv_offsets.append(total_columnwise_scale_elements)
                columnwise_scale_inv = torch.empty(
                    total_columnwise_scale_elements, dtype=torch.uint8, device=device
                )
                # Columnwise amax buffer - one per tensor
                columnwise_amax = torch.empty(num_tensors, dtype=torch.float32, device=device)
        elif quantizer._get_compatible_recipe().float8_block_scaling():
            if rowwise_usage:
                # Allocate rowwise data buffer (1D flattened, uint8)
                data = torch.empty(total_elements, dtype=torch.uint8, device=device)
                # Scale inverse - size depends on block configuration
                # For simplicity, calculate total scale elements needed
                total_scale_elements = 0
                scale_inv_offsets = [0]
                for i, s in enumerate(shape):
                    scale_inv_shape = quantizer.get_scale_shape(s, False)
                    total_scale_elements += math.prod(scale_inv_shape)
                    scale_inv_offsets.append(total_scale_elements)
                scale_inv = torch.empty(total_scale_elements, dtype=torch.float32, device=device)

            if columnwise_usage:
                # Allocate columnwise data buffer (1D flattened, uint8)
                columnwise_data = torch.empty(total_elements, dtype=torch.uint8, device=device)
                # Columnwise scale inverse
                total_columnwise_scale_elements = 0
                columnwise_scale_inv_offsets = [0]
                for i, s in enumerate(shape):
                    columnwise_scale_inv_shape = quantizer.get_scale_shape(s, True)
                    total_columnwise_scale_elements += math.prod(columnwise_scale_inv_shape)
                    columnwise_scale_inv_offsets.append(total_columnwise_scale_elements)
                columnwise_scale_inv = torch.empty(
                    total_columnwise_scale_elements, dtype=torch.float32, device=device
                )
        elif quantizer._get_compatible_recipe().float8_current_scaling():
            # Current scaling - per-tensor scaling computed on the fly
            if rowwise_usage:
                # Allocate rowwise data buffer (1D flattened, uint8)
                data = torch.empty(total_elements, dtype=torch.uint8, device=device)
                # Scale inverse - one per tensor
                scale_inv = torch.empty(num_tensors, dtype=torch.float32, device=device)
                # One scale per tensor, so offsets are simply 0, 1, 2, ..., num_tensors
                scale_inv_offsets = list(range(num_tensors + 1))

            if columnwise_usage:
                # Allocate columnwise data buffer (1D flattened, uint8)
                columnwise_data = torch.empty(total_elements, dtype=torch.uint8, device=device)
                # Columnwise scale inverse - one per tensor
                columnwise_scale_inv = torch.empty(num_tensors, dtype=torch.float32, device=device)
                # One scale per tensor, so offsets are simply 0, 1, 2, ..., num_tensors
                columnwise_scale_inv_offsets = list(range(num_tensors + 1))

            # Scale and amax buffers for current scaling - one per tensor
            scale = torch.empty(num_tensors, dtype=torch.float32, device=device)
            amax = torch.empty(num_tensors, dtype=torch.float32, device=device)
        else:
            raise ValueError(f"Unsupported quantizer for GroupedTensor: {quantizer}")

        # Construct wrapper vs storage based on quantizer.internal.
        # If quantizer is None (high precision path), default to wrapper class.
        # TODO(ksivaman): Properly handle high precision path.
        internal = False if quantizer is None else quantizer.internal
        if internal:
            grouped_tensor_class = GroupedTensorStorage
        else:
            from ..grouped_tensor import GroupedTensor

            grouped_tensor_class = GroupedTensor

        grouped_tensor = grouped_tensor_class(
            logical_shape,
            dtype,
            num_tensors=num_tensors,
            shapes=shape,
            quantizer=quantizer,
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
            with_gemm_swizzled_scales=(
                quantizer.optimize_for_gemm if quantizer is not None else False
            ),
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

        This API is NOT graph safe, but can be used for testing & debugging.

        TODO(ksivaman): Block cases where any dims are varying. This is needed only
        to expose the weights as separate parameters.
        """

        result = []

        no_quantization = self.quantizer is None

        # if self.tensor_shapes is None, then trigger D2H copy and get the shape (not graph safe)
        if self.tensor_shapes is None:
            first_dims_list = (
                [self.logical_shape[0]] * self.num_tensors
                if self.first_dims is None
                else self.first_dims.tolist()
            )
            last_dims_list = (
                [self.logical_shape[1]] * self.num_tensors
                if self.last_dims is None
                else self.last_dims.tolist()
            )
            shape_list = []
            for i in range(self.num_tensors):
                shape_list.append((first_dims_list[i], last_dims_list[i]))
            self.tensor_shapes = shape_list

        # edge case: handle the case where tensor_offsets is given but offsets is not set
        if self.offsets is None and self.tensor_offsets is not None:
            self.offsets = self.tensor_offsets.tolist()

        # Case 1: No quantization - return regular torch tensors
        if no_quantization:
            for i in range(self.num_tensors):
                # Get tensor shape
                tensor_shape = self.tensor_shapes[i]

                # Get tensor data slice
                if self.offsets is not None:
                    start_offset = self.offsets[i]
                    numel = math.prod(tensor_shape)
                    end_offset = start_offset + numel

                    if self.has_data():
                        tensor_data = self.rowwise_data[start_offset:end_offset].view(tensor_shape)
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
                    numel = math.prod(tensor_shape)
                    start_offset = i * numel
                    end_offset = start_offset + numel

                    if self.has_data():
                        tensor_data = self.rowwise_data[start_offset:end_offset].view(tensor_shape)
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
        recipe = self.quantizer._get_compatible_recipe()

        # populate scale_inv_offsets from the tensor offsets
        if self.scale_inv is not None and self.scale_inv_offsets is None:
            if recipe.nvfp4() or recipe.mxfp8() or recipe.float8_block_scaling():
                cum = 0
                scale_inv_offsets = [0]
                for i in range(self.num_tensors):
                    tensor_shape = self.tensor_shapes[i]
                    scale_shape = self.quantizer.get_scale_shape(tensor_shape, False)
                    cum += math.prod(scale_shape)
                    scale_inv_offsets.append(cum)
                self.scale_inv_offsets = scale_inv_offsets
        if self.columnwise_scale_inv is not None and self.columnwise_scale_inv_offsets is None:
            if recipe.nvfp4() or recipe.mxfp8() or recipe.float8_block_scaling():
                cum = 0
                columnwise_scale_inv_offsets = [0]
                for i in range(self.num_tensors):
                    tensor_shape = self.tensor_shapes[i]
                    scale_shape = self.quantizer.get_scale_shape(tensor_shape, True)
                    cum += math.prod(scale_shape)
                    columnwise_scale_inv_offsets.append(cum)
                self.columnwise_scale_inv_offsets = columnwise_scale_inv_offsets

        for i in range(self.num_tensors):
            quantizer = self.quantizer
            # Get tensor shape
            tensor_shape = self.tensor_shapes[i]
            numel = math.prod(tensor_shape)

            # Get data offsets
            if self.offsets is not None:
                data_start = self.offsets[i]
                data_end = data_start + numel
            else:
                # All same shape
                data_start = i * numel
                data_end = data_start + numel

            # Special shape handling for NVFP4.
            nvfp4 = quantizer._get_compatible_recipe().nvfp4()
            if nvfp4:
                data_start = data_start // 2
                data_end = data_end // 2

            # Extract rowwise and columnwise data
            rowwise_data = None
            columnwise_data = None

            if self.has_data():
                if nvfp4:
                    rowwise_tensor_shape = quantizer.convert_shape_for_fp4(tensor_shape)
                else:
                    rowwise_tensor_shape = tensor_shape
                rowwise_data = self.rowwise_data[data_start:data_end].view(rowwise_tensor_shape)

            if self.has_columnwise_data():
                columnwise_tensor_shape = quantizer.get_columnwise_shape(tensor_shape)
                if nvfp4:
                    columnwise_tensor_shape = quantizer.convert_shape_for_fp4(
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
                    # for paged stashing, scale_inv should depend on the split offsets
                    scale_end = self.scale_inv_offsets[i + 1]

                    # Calculate expected scale shape for MXFP8
                    scale_shape = quantizer.get_scale_shape(tensor_shape, False)
                    rowwise_scale_inv = self.scale_inv[scale_start:scale_end].view(scale_shape)

                if (
                    self.columnwise_scale_inv is not None
                    and self.columnwise_scale_inv_offsets is not None
                ):
                    cscale_start = self.columnwise_scale_inv_offsets[i]
                    # for paged stashing, columnwise_scale_inv should depend on the split offsets
                    cscale_end = self.columnwise_scale_inv_offsets[i + 1]

                    cscale_shape = quantizer.get_scale_shape(tensor_shape, True)
                    columnwise_scale_inv = self.columnwise_scale_inv[cscale_start:cscale_end].view(
                        cscale_shape
                    )

                if quantizer.internal:
                    mxfp8_tensor_class = MXFP8TensorStorage
                else:
                    mxfp8_tensor_class = MXFP8Tensor
                tensor = mxfp8_tensor_class(
                    shape=tensor_shape,
                    dtype=self.fake_dtype,
                    rowwise_data=rowwise_data,
                    rowwise_scale_inv=rowwise_scale_inv,
                    columnwise_data=columnwise_data,
                    columnwise_scale_inv=columnwise_scale_inv,
                    fp8_dtype=quantizer.dtype,
                    quantizer=quantizer,
                    with_gemm_swizzled_scales=quantizer.optimize_for_gemm,
                )
                result.append(tensor)

            # Delayed scaling or current scaling (both use Float8TensorStorage)
            elif recipe.delayed() or recipe.float8_current_scaling():
                # Scale inverse - one per tensor
                scale_inv = None
                if self.scale_inv is not None:
                    scale_inv = self.scale_inv[i : i + 1]

                if quantizer.internal:
                    float8_tensor_class = Float8TensorStorage
                else:
                    float8_tensor_class = Float8Tensor

                tensor = float8_tensor_class(
                    shape=tensor_shape,
                    dtype=self.fake_dtype,
                    data=rowwise_data,
                    fp8_scale_inv=scale_inv,
                    fp8_dtype=quantizer.dtype,
                    quantizer=quantizer,
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
                    # for paged stashing, scale_inv should depend on the split offsets
                    scale_end = self.scale_inv_offsets[i + 1]

                    # Get scale shape from quantizer
                    scale_shape = quantizer.get_scale_shape(tensor_shape, False)
                    rowwise_scale_inv = self.scale_inv[scale_start:scale_end].view(scale_shape)

                if (
                    self.columnwise_scale_inv is not None
                    and self.columnwise_scale_inv_offsets is not None
                ):
                    cscale_start = self.columnwise_scale_inv_offsets[i]
                    # for paged stashing, columnwise_scale_inv should depend on the split offsets
                    cscale_end = self.columnwise_scale_inv_offsets[i + 1]

                    # Get columnwise scale shape from quantizer
                    cscale_shape = quantizer.get_scale_shape(tensor_shape, True)
                    columnwise_scale_inv = self.columnwise_scale_inv[cscale_start:cscale_end].view(
                        cscale_shape
                    )

                # Compute is_2D_scaled and data_format from quantizer attributes
                is_2D_scaled = quantizer.block_scaling_dim == 2

                if quantizer.internal:
                    float8_blockwise_q_tensor_class = Float8BlockwiseQTensorStorage
                else:
                    float8_blockwise_q_tensor_class = Float8BlockwiseQTensor

                tensor = float8_blockwise_q_tensor_class(
                    shape=tensor_shape,
                    dtype=self.fake_dtype,
                    rowwise_data=rowwise_data,
                    rowwise_scale_inv=rowwise_scale_inv,
                    columnwise_data=columnwise_data,
                    columnwise_scale_inv=columnwise_scale_inv,
                    fp8_dtype=quantizer.dtype,
                    quantizer=quantizer,
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
                    # for paged stashing, scale_inv should depend on the split offsets
                    scale_end = self.scale_inv_offsets[i + 1]

                    # Get scale shape from quantizer
                    scale_shape = quantizer.get_scale_shape(tensor_shape, False)
                    rowwise_scale_inv = self.scale_inv[scale_start:scale_end].view(scale_shape)

                if (
                    self.columnwise_scale_inv is not None
                    and self.columnwise_scale_inv_offsets is not None
                ):
                    cscale_start = self.columnwise_scale_inv_offsets[i]
                    # for paged stashing, columnwise_scale_inv should depend on the split offsets
                    cscale_end = self.columnwise_scale_inv_offsets[i + 1]

                    # Get columnwise scale shape from quantizer
                    cscale_shape = quantizer.get_scale_shape(tensor_shape, True)
                    columnwise_scale_inv = self.columnwise_scale_inv[cscale_start:cscale_end].view(
                        cscale_shape
                    )

                # Extract amax - one per tensor
                if self.amax is not None:
                    amax_rowwise = self.amax[i : i + 1]

                if self.columnwise_amax is not None:
                    amax_columnwise = self.columnwise_amax[i : i + 1]

                if quantizer.internal:
                    nvfp4_tensor_class = NVFP4TensorStorage
                else:
                    nvfp4_tensor_class = NVFP4Tensor

                tensor = nvfp4_tensor_class(
                    shape=tensor_shape,
                    dtype=self.fake_dtype,
                    rowwise_data=rowwise_data,
                    rowwise_scale_inv=rowwise_scale_inv,
                    columnwise_data=columnwise_data,
                    columnwise_scale_inv=columnwise_scale_inv,
                    amax_rowwise=amax_rowwise,
                    amax_columnwise=amax_columnwise,
                    fp4_dtype=quantizer.dtype,
                    quantizer=quantizer,
                    with_gemm_swizzled_scales=quantizer.optimize_for_gemm,
                )
                result.append(tensor)

            else:
                raise ValueError(f"Unsupported quantization recipe: {recipe}")

        return result

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
            self.quantizer.update_quantized(tensors[i], quantized_tensors[i], noop_flag=noop_flag)
        return quantized_tensors
