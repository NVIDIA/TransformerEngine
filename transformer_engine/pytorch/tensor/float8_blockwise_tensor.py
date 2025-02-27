# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Tensor class with FP8 data quantized with NxN tiles"""
from __future__ import annotations
from typing import Optional, Tuple, Iterable
import warnings

import math
import torch
import transformer_engine_torch as tex

from transformer_engine_torch import DType as TE_DType
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
        force_pow_2_scales: bool = False,
        block_scaling_dim: int = 2,
    ) -> None:
        super().__init__(rowwise=rowwise, columnwise=columnwise)
        assert rowwise
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
        # cuBLAS kernel format (for NxN by NxN and 1xN by NxN GEMMs)
        # The scales for 2D block quantized tensors must have scales padded
        # to multiples of 4 on the inner dimension. TODO: Verify whether outer
        # dimension also to be padded for either GEMM.
        if self.block_scaling_dim == 2:
            logical_scale_shape = [1, 1]
            for i in range(len(shape) - 1):
                logical_scale_shape[-2] *= shape[i]
            if len(shape) > 0:
                logical_scale_shape[-1] = math.ceil(shape[-1] / self.block_len)
            logical_scale_shape[-2] = math.ceil(logical_scale_shape[-2] / self.block_len)
            if columnwise:
                tmp = logical_scale_shape[-1]
                logical_scale_shape[-1] = logical_scale_shape[-2]
                logical_scale_shape[-2] = tmp
            logical_scale_shape[-1] = round_up_to_nearest_multiple(logical_scale_shape[-1], 4)
            return tuple(logical_scale_shape)
        else:
            assert self.block_scaling_dim == 1, "Only 1D or 2D blocks supported"

            logical_scale_shape = [1, 1]
            for i in range(len(shape) - 1):
                logical_scale_shape[-1] *= shape[i]
            if len(shape) > 0:
                logical_scale_shape[-2] = shape[-1]
            if not columnwise:
                logical_scale_shape[-2] = math.ceil(logical_scale_shape[-2] / self.block_len)
                return tuple(logical_scale_shape)
            else:
                logical_scale_shape[-1] = math.ceil(logical_scale_shape[-1] / self.block_len)
                return (logical_scale_shape[1], logical_scale_shape[0])

    def get_columnwise_shape(self, shape: Iterable[int]) -> Tuple[int, ...]:
        if len(shape) == 0:
            return tuple()
        colwise_shape = [shape[-1]]
        for i in range(len(shape) - 1):
            colwise_shape.append(shape[i])
        return tuple(colwise_shape)

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
            requires_grad=requires_grad,
        )

    def calibrate(self, tensor: torch.Tensor) -> None:
        # NOTE: This interface is specific to requirements like delayed scaling
        # where state from an estimator influences distribution parameters.
        pass


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

    def update_usage(self, rowwise_usage=True, columnwise_usage=True):
        """
        update_usage can be used to clear out one of two possible copies of the data.
        """

        assert (
            columnwise_usage or rowwise_usage
        ), "Must retain some data either columnwise or rowwise"

        if columnwise_usage and rowwise_usage:
            assert (
                self._rowwise_data is not None
                and self._rowwise_scale_inv is not None
                and self._columnwise_data is not None
                and self._columnwise_scale_inv is not None
            ), "Cannot update to rowwise and columnwise usage."
            return

        if rowwise_usage:
            assert (
                self._rowwise_data is not None and self._rowwise_scale_inv is not None
            ), "Cannot update to rowwise usage."
            self._columnwise_data = None
            self._columnwise_scale_inv = None
            return
        if columnwise_usage:
            assert (
                self._columnwise_data is not None and self._columnwise_scale_inv is not None
            ), "Cannot update to columnwise usage."
            self._rowwise_data = None
            self._rowwise_scale_inv = None
            return

        return

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

    def clear(self):
        """Deallocate this tensor's memory. Typically not needed and must be used carefully."""
        self._rowwise_data = torch.Tensor() if self._rowwise_data is not None else None
        self._columnwise_data = torch.Tensor() if self._columnwise_data is not None else None

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs=None):

        # View op
        if func == aten.view.default:
            tensor = args[0]
            data = tensor._rowwise_data
            out_data = data.__torch_dispatch__(
                func,
                types,
                [data] + list(args[1:]),
                kwargs,
            )
            out_shape = out_data.size()
            return Float8BlockwiseQTensor(
                shape=out_shape,
                dtype=tensor.dtype,
                rowwise_data=out_data,
                rowwise_scale_inv=tensor._rowwise_scale_inv,
                columnwise_data=tensor._columnwise_data,
                columnwise_scale_inv=tensor._columnwise_scale_inv,
                quantizer=tensor._quantizer,
                requires_grad=False,
                fp8_dtype=tensor._fp8_dtype,
            )

        # Default case
        return super().__torch_dispatch__(func, types, args, kwargs)

    @classmethod
    def _make_in_reduce_ex(
        cls,
        rowwise_data: torch.Tensor,
        rowwise_scale_inv: torch.Tensor,
        columnwise_data: torch.Tensor,
        columnwise_scale_inv: torch.Tensor,
        fp8_dtype: TE_DType,
        dtype: torch.dtype,
    ) -> Float8BlockwiseQTensor:
        """Build Float8BlockwiseQTensor, for use in __reduce__

        __reduce_ex__ assumes object constructor has positional
        arguments.

        """
        return Float8BlockwiseQTensor(
            rowwise_data=rowwise_data,
            rowwise_scale_inv=rowwise_scale_inv,
            fp8_dtype=fp8_dtype,
            columnwise_data=columnwise_data,
            columnwise_scale_inv=columnwise_scale_inv,
            dtype=dtype,
        )

    def __reduce_ex__(self, protocol: int) -> tuple:
        """Custom pickling to remove references to FP8 metadata objects"""
        return (
            Float8BlockwiseQTensor._make_in_reduce_ex,
            (
                self._rowwise_data,
                self._rowwise_scale_inv,
                self._columnwise_data,
                self._columnwise_scale_inv,
                self._fp8_dtype,
                self.dtype,
            ),
        )

    def _get_data(self) -> Float8BlockwiseQTensor:
        """Get tensor data property"""
        return super().data

    @torch.no_grad()
    def _set_data(self, tensor: torch.Tensor) -> None:
        """Set tensor data property

        Just takes FP8 data if setting from a Float8BlockwiseQTensor. Otherwise
        casts to FP8.

        """

        # Tensor device
        new_device = tensor.device if tensor.is_cuda else self.device

        # Just copy FP8 data if other tensor is Float8BlockwiseQTensor
        if isinstance(tensor, Float8BlockwiseQTensor):
            if (  # pylint: disable=too-many-boolean-expressions
                self.size() != tensor.size()
                or self.stride() != tensor.stride()
                or self.storage_offset() != tensor.storage_offset()
                or self.dtype != tensor.dtype
                or self.layout != tensor.layout
                or not devices_match(self.device, new_device)
            ):
                dummy_tensor = torch.Tensor._make_wrapper_subclass(
                    Float8BlockwiseQTensor,
                    tensor.size(),
                    strides=tensor.stride(),
                    storage_offset=tensor.storage_offset(),
                    dtype=tensor.dtype,
                    layout=tensor.layout,
                    requires_grad=tensor.requires_grad,
                    device=new_device,
                )
                # pylint: disable=unnecessary-dunder-call
                super(Float8BlockwiseQTensor, type(self)).data.__set__(self, dummy_tensor)
            self._rowwise_data = tensor._rowwise_data
            self._columnwise_data = tensor._columnwise_data
            self._quantizer = tensor._quantizer
            self._fp8_dtype = tensor._fp8_dtype
            self._rowwise_scale_inv = tensor._rowwise_scale_inv
            self._columnwise_scale_inv = tensor._columnwise_scale_inv
            return

        # Quantize to FP8
        assert self._quantizer is not None, "Can't quantize without a quantizer"
        self.data = self._quantizer.quantize(tensor)
        if self.requires_grad != tensor.requires_grad:
            self.requires_grad_(requires_grad=tensor.requires_grad)

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

        if shape != ctx.shape:
            raise NotImplementedError("View not implemented.")
        else:
            return tensor

    @staticmethod
    def backward(
        ctx,
        grad: torch.Tensor,
    ) -> Tuple[Optional[torch.Tensor], ...]:
        # pylint: disable=missing-function-docstring

        if isinstance(grad, Float8BlockwiseQTensor):
            raise NotImplementedError("View bwd not implemented")
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
            d_inferred = -math.prod(ctx.shape) // math.prod(shape)
            for i, d in enumerate(shape):
                if d == -1:
                    shape[i] = d_inferred
                    break
        if shape != ctx.shape:
            raise NotImplementedError("Reshape not implemented yet.")

    @staticmethod
    def backward(
        ctx,
        grad: torch.Tensor,
    ) -> Tuple[Optional[torch.Tensor], ...]:
        # pylint: disable=missing-function-docstring

        if isinstance(grad, Float8BlockwiseQTensor):
            raise NotImplementedError("Reshape bwd not implemented yet.")
        return grad.view(ctx.shape), None
