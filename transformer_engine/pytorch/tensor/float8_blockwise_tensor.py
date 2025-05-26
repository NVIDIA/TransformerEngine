# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Tensor class with FP8 data quantized with NxN tiles"""
from __future__ import annotations
from typing import Optional, Tuple, Iterable, Union

import math
import torch
import transformer_engine_torch as tex
from transformer_engine_torch import DType as TE_DType

from transformer_engine.common.recipe import Float8BlockScaling, Recipe
from ._internal.float8_blockwise_tensor_base import Float8BlockwiseQTensorBase
from .quantized_tensor import QuantizedTensor, Quantizer, _IdentityFunc
from ..utils import devices_match, round_up_to_nearest_multiple

aten = torch.ops.aten


class Float8BlockQuantizer(Quantizer):
    """Builder class for tensors quantized with current scaling using
    NxN quantization tilings to choose scale.

    This class is typically used to convert a high-precision tensor
    (e.g. in FP32 or BF16) into a quantized tensor (e.g. in FP8).

    """

    dtype: TE_DType
    block_len: int
    amax_epsilon: float
    force_pow_2_scales: bool
    block_scaling_dim: int

    def __init__(
        self,
        fp8_dtype: TE_DType,
        *,
        rowwise: bool,
        columnwise: bool,
        amax_epsilon: float = 0.0,
        force_pow_2_scales: bool = True,
        block_scaling_dim: int = 2,
    ) -> None:
        super().__init__(rowwise=rowwise, columnwise=columnwise)
        self.dtype = fp8_dtype
        self.block_len = 128
        self.force_pow_2_scales = force_pow_2_scales
        self.amax_epsilon = amax_epsilon
        self.block_scaling_dim = block_scaling_dim

    def update_quantized(
        self,
        src: torch.Tensor,
        dst: QuantizedTensor,
        *,
        noop_flag: Optional[torch.Tensor] = None,
    ) -> QuantizedTensor:
        """Update the quantized tensor with data from the source tensor.

        This method quantizes the input tensor and stores the result in the destination tensor.

        Parameters
        ----------
        src : torch.Tensor
            Source tensor containing the data to be quantized
        dst : QuantizedTensor
            Destination tensor where the quantized data will be stored
        noop_flag : Optional[torch.Tensor]
            Optional flag tensor indicating whether to skip the quantization operation

        Returns
        -------
        QuantizedTensor
            The destination tensor containing the quantized data

        Raises
        ------
        AssertionError
            If the destination tensor is not a Float8BlockwiseQTensor
        """
        assert isinstance(
            dst, Float8BlockwiseQTensor
        ), f"Cannot store quantized blockwise tensor in {type(dst)} type."
        # Make sure input is in expected format
        if not devices_match(src.device, dst.device):
            src = src.to(device=dst.device)
        if not src.is_contiguous():
            src = src.contiguous()

        # Launch cast kernel
        tex.quantize(src, self, dst, noop_flag)

        dst._fp8_dtype = self.dtype
        return dst

    def get_scale_shape(self, shape: Iterable[int], columnwise: bool) -> Tuple[int, int]:
        """Calculate the shape of the scaling tensor for blockwise quantization.

        This method determines the shape of the scaling tensor needed for blockwise quantization,
        taking into account the input tensor shape and whether columnwise scaling is used.
        The scales are padded to multiples of 4 on the inner dimension for compatibility with GEMM.

        Parameters
        ----------
        shape : Iterable[int]
            Shape of the input tensor to be quantized
        columnwise : bool
            Whether to use columnwise scaling (True) or rowwise scaling (False)

        Returns
        -------
        Tuple[int, int]
            Shape of the scaling tensor as (outer_dim, inner_dim)
            For 2D tensors:
            - If columnwise: (roundup(K/blocksize), round_to_multiple(roundup(M/blocksize), 4))
            - If rowwise: (roundup(M/blocksize), round_to_multiple(roundup(K/blocksize), 4))
            For 1D tensors:
            - If columnwise: (roundup(M/blocksize), round_to_multiple(K, 4))
            - If rowwise: (roundup(K/blocksize), round_to_multiple(M, 4))
        """
        M, K = 1, 1
        for i in range(len(shape) - 1):
            M *= shape[i]
        if len(shape) > 0:
            K = shape[-1]
        if self.block_scaling_dim == 2:
            if columnwise:
                outer = math.ceil(K / self.block_len)
                inner = round_up_to_nearest_multiple(math.ceil(M / self.block_len), 4)
                return (outer, inner)
            outer = math.ceil(M / self.block_len)
            inner = round_up_to_nearest_multiple(math.ceil(K / self.block_len), 4)
            return (outer, inner)
        assert self.block_scaling_dim == 1, "Only 1D or 2D blocks supported"
        if columnwise:
            outer = math.ceil(M / self.block_len)
            inner = round_up_to_nearest_multiple(K, 4)
            return (outer, inner)
        outer = math.ceil(K / self.block_len)
        inner = round_up_to_nearest_multiple(M, 4)
        return (outer, inner)

    def get_columnwise_shape(self, shape: Iterable[int]) -> Tuple[int, ...]:
        """Calculate the shape of a tensor after columnwise permutation.

        This method rearranges the dimensions of a tensor to be columnwise,
        moving the last dimension to the front and keeping the order of other dimensions.

        Parameters
        ----------
        shape : Iterable[int]
            Original shape of the tensor

        Returns
        -------
        Tuple[int, ...]
            New shape with dimensions rearranged for columnwise layout.
            For a shape (d1, d2, ..., dn), returns (dn, d1, d2, ..., dn-1).
            Returns empty tuple for empty input shape.
        """
        if len(shape) == 0:
            return tuple()
        colwise_shape = [shape[-1]]
        for i in range(len(shape) - 1):
            colwise_shape.append(shape[i])
        return tuple(colwise_shape)

    # TODO(kwyss): With FP8 gather support, we need to implement a
    # shape/layout/swizzle check to know whether FP8 gather works
    # cleanly by stacking data without aliasing tiles and whether
    # the scales also stack on the proper dimensions.

    def make_empty(
        self,
        shape: Iterable[int],
        *,
        dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None,
        requires_grad: bool = False,
    ) -> Float8BlockwiseQTensor:
        """Construct quantized tensor with uninitialized data"""
        if device is None:
            device = torch.device("cuda")

        # Allocate FP8 data
        data = None
        scale_inv = None
        if self.rowwise_usage:
            data = torch.empty(shape, dtype=torch.uint8, device=device)
            scale_shape = self.get_scale_shape(shape, columnwise=False)
            scale_inv = torch.empty(
                scale_shape,
                dtype=torch.float32,
                device=device,
            )

        # Allocate FP8 data transpose if needed
        columnwise_data = None
        columnwise_scale_inv = None
        if self.columnwise_usage:
            columnwise_data = torch.empty(
                self.get_columnwise_shape(shape), dtype=torch.uint8, device=device
            )
            columnwise_scale_shape = self.get_scale_shape(shape, columnwise=True)
            columnwise_scale_inv = torch.empty(
                columnwise_scale_shape,
                dtype=torch.float32,
                device=device,
            )

        # Construct FP8 tensor
        return Float8BlockwiseQTensor(
            shape=shape,
            dtype=dtype,
            fp8_dtype=self.dtype,
            rowwise_data=data,
            rowwise_scale_inv=scale_inv,
            columnwise_data=columnwise_data,
            columnwise_scale_inv=columnwise_scale_inv,
            quantizer=self,
            is_2D_scaled=self.block_scaling_dim == 2,
            requires_grad=requires_grad,
        )

    def calibrate(self, tensor: torch.Tensor) -> None:
        # NOTE: This interface is specific to requirements like delayed scaling
        # where state from an estimator influences distribution parameters.
        pass

    def _get_compatible_recipe(self) -> Union[type[Recipe], None]:
        return Float8BlockScaling


class Float8BlockwiseQTensor(Float8BlockwiseQTensorBase, QuantizedTensor):
    """Tensor class with FP8 data quantized via NxN blocks or 1xN blocks.

    The tensor presents as having a standard, higher-precision dtype,
    but the data itself is (scaled) FP8. For most tensor operations,
    the data will be cast to the nominal dtype before performing the
    operation.

    Parameters
    ----------
    rowwise_data: torch.Tensor
          FP8 data in a uint8 tensor matching shape of dequantized tensor.
    rowwise_scale_inv: torch.Tensor
          FP32 dequantization scales in GEMM format for dequantizing rowwise_data.
    columnwise_data: Optional[torch.Tensor]
          FP8 data in a uint8 tensor matching shape of dequantized tensor transpose.
    columnwise_scale_inv: Optional[torch.Tensor]
          FP32 dequantization scales in GEMM format for dequantizing columnwise_data.

    fp8_dtype: transformer_engine_torch.DType, default = kFloat8E4M3
               FP8 format.
    quantizer: Quantizer - the Float8BlockQuantizer that quantized this tensor and
               holds configuration about quantization and dequantization modes.
    """

    def __repr__(self, *, tensor_contents=None):
        return (
            f"Float8BlockwiseQTensor(fp8_dtype={self._fp8_dtype},"
            f" is_2D_scaled={self._is_2D_scaled},"
            f" data={self.dequantize(dtype=self.dtype)})"
        )

    def _get_quantizer(self) -> Quantizer:
        """Get builder for quantized tensor

        Quantizer can be used for in-place operations.

        """
        assert self._quantizer is not None
        return self._quantizer

    def quantize_(
        self,
        tensor: torch.Tensor,
        *,
        noop_flag: Optional[torch.Tensor] = None,
    ) -> Float8BlockwiseQTensor:
        """Update FP8 data

        Parameters
        ----------
        tensor: torch.Tensor
            Tensor to copy from
        noop_flag: torch.Tensor, optional
            float32 flag indicating whether to avoid performing update

        """
        if isinstance(tensor, QuantizedTensor):
            return self.quantize_(tensor.dequantize())
        self._get_quantizer().update_quantized(tensor, self, noop_flag=noop_flag)
        return self

    def dequantize(self, *, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        """
        Construct plain PyTorch tensor from Float8BlockwiseQTensor

        By default the resulting tensor's dtype is the
        Float8BlockwiseQTensor's pre-quantized dtype.
        """
        if dtype is not None:
            dequant_dtype = dtype
        else:
            dequant_dtype = self.dtype
        return super().dequantize(dtype=dequant_dtype)

    def detach(self) -> Float8BlockwiseQTensor:
        # pylint: disable=missing-function-docstring
        return Float8BlockwiseQTensor.make_like(self)

    def clone(self) -> Float8BlockwiseQTensor:
        # pylint: disable=missing-function-docstring
        rowwise_data = None
        if self._rowwise_data is not None:
            rowwise_data = self._rowwise_data.detach().clone()
        columnwise_data = None
        if self._columnwise_data is not None:
            columnwise_data = self._columnwise_data.detach().clone()
        return _IdentityFunc.apply(
            self,
            {
                "rowwise_data": rowwise_data,
                "columnwise_data": columnwise_data,
            },
        )

    def view(self, *shape: Tuple[int]) -> Float8BlockwiseQTensor:
        # pylint: disable=missing-function-docstring
        return _ViewFunc.apply(self, shape)

    def reshape(self, *shape: Tuple[int]) -> Float8BlockwiseQTensor:
        # pylint: disable=missing-function-docstring
        return _ReshapeFunc.apply(self, shape)

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs=None):

        # View op
        if func == aten.view.default:
            tensor = args[0]
            data = tensor._rowwise_data
            if data is None:
                # Columnwise data only.
                super().__torch_dispatch__(func, types, args, kwargs)
            orig_size = data.size()
            out_data = data.__torch_dispatch__(
                func,
                types,
                [data] + list(args[1:]),
                kwargs,
            )
            if orig_size != out_data.size():
                raise NotImplementedError(
                    "Changing shape with view not implemented "
                    " (scales and columnwise data untouched)."
                )
            return Float8BlockwiseQTensor.make_like(tensor)

        # Default case
        return super().__torch_dispatch__(func, types, args, kwargs)

    def contiguous(
        self,
        memory_format: torch.memory_format = torch.contiguous_format,
    ) -> Float8BlockwiseQTensor:
        """Returns tensor with data in provided memory format

        Returns `self` if data is already in correct memory format.

        """
        if (
            self._rowwise_data is not None
            and self._rowwise_data.is_contiguous(memory_format=memory_format)
            and (
                (self._columnwise_data is None)
                or (self._columnwise_data.is_contiguous(memory_format=memory_format))
            )
        ):
            return self
        raise ValueError("Float8BlockwiseQTensor does not support different memory formats!")

    @classmethod
    def _make_in_reduce_ex(
        cls,
        shape: torch.Size,
        rowwise_data: torch.Tensor,
        rowwise_scale_inv: torch.Tensor,
        columnwise_data: torch.Tensor,
        columnwise_scale_inv: torch.Tensor,
        fp8_dtype: TE_DType,
        dtype: torch.dtype,
        quantizer: Quantizer,
        is_2D_scaled: bool,
    ) -> Float8BlockwiseQTensor:
        """Build Float8BlockwiseQTensor, for use in __reduce__

        __reduce_ex__ assumes object constructor has positional
        arguments.

        """
        return Float8BlockwiseQTensor(
            shape=shape,
            rowwise_data=rowwise_data,
            rowwise_scale_inv=rowwise_scale_inv,
            fp8_dtype=fp8_dtype,
            columnwise_data=columnwise_data,
            columnwise_scale_inv=columnwise_scale_inv,
            dtype=dtype,
            quantizer=quantizer,
            is_2D_scaled=is_2D_scaled,
        )

    def __reduce_ex__(self, protocol: int) -> tuple:
        """Custom pickling to remove references to FP8 metadata objects"""
        return (
            Float8BlockwiseQTensor._make_in_reduce_ex,
            (
                self.shape,
                self._rowwise_data,
                self._rowwise_scale_inv,
                self._columnwise_data,
                self._columnwise_scale_inv,
                self._fp8_dtype,
                self.dtype,
                self._quantizer,
                self._is_2D_scaled,
            ),
        )

    def _get_data(self) -> Float8BlockwiseQTensor:
        """Get tensor data property"""
        return self

    @torch.no_grad()
    def _set_data(self, tensor: torch.Tensor) -> None:
        """Set tensor data property

        Just takes FP8 data if setting from a Float8BlockwiseQTensor. Otherwise
        casts to FP8.

        """
        # Tensor device
        new_device = tensor.device if tensor.is_cuda else self.device

        def _set_from_tensor(dst: Float8BlockwiseQTensor, src: Float8BlockwiseQTensor):
            dst._rowwise_data = src._rowwise_data
            dst._columnwise_data = src._columnwise_data
            dst._quantizer = src._quantizer
            dst._fp8_dtype = src._fp8_dtype
            dst._rowwise_scale_inv = src._rowwise_scale_inv
            dst._columnwise_scale_inv = src._columnwise_scale_inv

        # Check that tensor dimensions match
        if (
            self.size() != tensor.size()
            or self.stride() != tensor.stride()
            or self.layout != tensor.layout
        ):
            raise ValueError("Invalid tensor for updating Float8BlockwiseQTensor data")

        # Just copy FP8 data if other tensor is Float8BlockwiseQTensor
        if (
            isinstance(tensor, Float8BlockwiseQTensor)
            and self.storage_offset() == tensor.storage_offset()
            and devices_match(self.device, new_device)
        ):
            _set_from_tensor(self, tensor)
            return

        if isinstance(tensor, Float8BlockwiseQTensor):
            assert tensor._quantizer is not None, "Can't quantize without a quantizer"
            quantizer = tensor._quantizer
        else:
            assert self._quantizer is not None, "Can't quantize without a quantizer"
            quantizer = self._quantizer

        # Quantize to FP8
        quantizer.update_quantized(tensor, self)

    # Cast to FP8 when setting Float8BlockwiseQTensor.data
    data = property(_get_data, _set_data)


class _ViewFunc(torch.autograd.Function):
    """View function

    View the Float8BlockwiseQTensor using the provided shape.

    """

    @staticmethod
    def forward(
        ctx,
        tensor: Float8BlockwiseQTensor,
        shape: Optional[list[int]] = None,
    ) -> Float8BlockwiseQTensor:
        # pylint: disable=missing-function-docstring

        # Return input tensor if shape is not provided
        ctx.shape = tensor.shape
        if shape is None:
            return tensor

        # Canonicalize shape
        if not isinstance(shape, Iterable):
            shape = [shape]
        elif len(shape) == 1 and isinstance(shape[0], Iterable):
            shape = shape[0]
        if -1 in shape:
            shape = list(shape)
            d_inferred = -math.prod(ctx.shape) // math.prod(shape)
            for i, d in enumerate(shape):
                if d == -1:
                    shape[i] = d_inferred
                    break

        if tensor._is_2D_scaled:
            # For the case of 2D scaled tensor, the last 2 dimensions should not change
            if shape[-1] != ctx.shape[-1] or shape[-2] != ctx.shape[-2]:
                raise RuntimeError(
                    "2D scaled Float8BlockwiseQTensor does not support view "
                    "the last 2 dimensions "
                    f"(attempted to view dims={tuple(tensor.shape)} to {tuple(shape)})"
                )
        else:
            # For the case of 1D scaled tensor, the last dimension should not change
            if shape[-1] != ctx.shape[-1]:
                raise RuntimeError(
                    "1D scaled Float8BlockwiseQTensor does not support view "
                    "the last dimension "
                    f"(attempted to view dims={tuple(tensor.shape)} to {tuple(shape)})"
                )

        if list(shape) == list(tensor.shape):
            return tensor

        # Construct new tensor if shape is provided
        new_rowwise_data = None
        new_columnwise_data = None
        if tensor._rowwise_data is not None:
            new_rowwise_data = tensor._rowwise_data.view(*shape)
        if tensor._columnwise_data is not None:
            columnwise_shape = [shape[-1]] + list(shape[:-1])
            new_columnwise_data = tensor._columnwise_data.view(columnwise_shape)

        return Float8BlockwiseQTensor(
            shape=shape,
            dtype=tensor.dtype,
            fp8_dtype=tensor._fp8_dtype,
            rowwise_data=new_rowwise_data,
            rowwise_scale_inv=tensor._rowwise_scale_inv,
            columnwise_data=new_columnwise_data,
            columnwise_scale_inv=tensor._columnwise_scale_inv,
            quantizer=tensor._quantizer,
            is_2D_scaled=tensor._is_2D_scaled,
            requires_grad=tensor.requires_grad,
        )

    @staticmethod
    def backward(
        ctx,
        grad: torch.Tensor,
    ) -> Tuple[Optional[torch.Tensor], ...]:
        # pylint: disable=missing-function-docstring

        if isinstance(grad, Float8BlockwiseQTensor):
            new_data = (
                grad._rowwise_data.view(*ctx.shape) if grad._rowwise_data is not None else None
            )
            if grad._columnwise_data is not None:
                columnwise_shape = [ctx.shape[-1]] + list(ctx.shape[:-1])
                new_columnwise_data = grad._columnwise_data.view(columnwise_shape)
            else:
                new_columnwise_data = None
            dgrad = Float8BlockwiseQTensor(
                shape=ctx.shape,
                dtype=grad.dtype,
                rowwise_data=new_data,
                rowwise_scale_inv=grad._rowwise_scale_inv,
                columnwise_data=new_columnwise_data,
                columnwise_scale_inv=grad._columnwise_scale_inv,
                fp8_dtype=grad._fp8_dtype,
                quantizer=grad._quantizer,
                is_2D_scaled=grad._is_2D_scaled,
                requires_grad=grad.requires_grad,
            )
            return dgrad, None
        return grad.view(ctx.shape), None


class _ReshapeFunc(torch.autograd.Function):
    """Reshape function

    Reshape the Float8BlockwiseQTensor using the provided shape.

    """

    @staticmethod
    def forward(
        ctx,
        tensor: Float8BlockwiseQTensor,
        shape: Optional[list[int]] = None,
    ) -> Float8BlockwiseQTensor:
        # pylint: disable=missing-function-docstring

        # Return input tensor if shape is not provided
        ctx.shape = tensor.shape
        if shape is None:
            return tensor

        # Canonicalize shape
        if not isinstance(shape, Iterable):
            shape = [shape]
        elif len(shape) == 1 and isinstance(shape[0], Iterable):
            shape = shape[0]
        if -1 in shape:
            shape = list(shape)
            d_inferred = -math.prod(tensor.shape) // math.prod(shape)
            for i, d in enumerate(shape):
                if d == -1:
                    shape[i] = d_inferred
                    break

        if tensor._is_2D_scaled:
            # For the case of 2D scaled tensor, the last 2 dimensions should not change
            if shape[-1] != ctx.shape[-1] or shape[-2] != ctx.shape[-2]:
                raise RuntimeError(
                    "2D scaled Float8BlockwiseQTensor does not support reshaping "
                    "the last 2 dimensions "
                    f"(attempted to reshape dims={tuple(tensor.shape)} to {tuple(shape)})"
                )
        else:
            # For the case of 1D scaled tensor, the last dimension should not change
            if shape[-1] != ctx.shape[-1]:
                raise RuntimeError(
                    "1D scaled Float8BlockwiseQTensor does not support reshaping "
                    "the last dimension "
                    f"(attempted to reshape dims={tuple(tensor.shape)} to {tuple(shape)})"
                )
        if list(shape) == list(tensor.shape):
            return tensor

        # Construct new tensor if shape is provided
        new_rowwise_data = None
        new_columnwise_data = None
        if tensor._rowwise_data is not None:
            new_rowwise_data = tensor._rowwise_data.reshape(*shape)
        if tensor._columnwise_data is not None:
            columnwise_shape = [shape[-1]] + list(shape[:-1])
            new_columnwise_data = tensor._columnwise_data.view(columnwise_shape)

        return Float8BlockwiseQTensor(
            shape=shape,
            dtype=tensor.dtype,
            fp8_dtype=tensor._fp8_dtype,
            rowwise_data=new_rowwise_data,
            rowwise_scale_inv=tensor._rowwise_scale_inv,
            columnwise_data=new_columnwise_data,
            columnwise_scale_inv=tensor._columnwise_scale_inv,
            quantizer=tensor._quantizer,
            is_2D_scaled=tensor._is_2D_scaled,
            requires_grad=tensor.requires_grad,
        )

    @staticmethod
    def backward(
        ctx,
        grad: torch.Tensor,
    ) -> Tuple[Optional[torch.Tensor], ...]:
        # pylint: disable=missing-function-docstring

        if isinstance(grad, Float8BlockwiseQTensor):
            new_rowwise_data = None
            new_columnwise_data = None
            if grad._rowwise_data is not None:
                new_rowwise_data = grad._rowwise_data.view(*ctx.shape)
            if grad._columnwise_data is not None:
                columnwise_shape = [ctx.shape[-1]] + list(ctx.shape[:-1])
                new_columnwise_data = grad._columnwise_data.view(columnwise_shape)
            dgrad = Float8BlockwiseQTensor(
                shape=ctx.shape,
                dtype=grad.dtype,
                rowwise_data=new_rowwise_data,
                rowwise_scale_inv=grad._rowwise_scale_inv,
                columnwise_data=new_columnwise_data,
                columnwise_scale_inv=grad._columnwise_scale_inv,
                fp8_dtype=grad._fp8_dtype,
                quantizer=grad._quantizer,
                is_2D_scaled=grad._is_2D_scaled,
                requires_grad=grad.requires_grad,
            )
            return dgrad, None
        return grad.view(ctx.shape), None
