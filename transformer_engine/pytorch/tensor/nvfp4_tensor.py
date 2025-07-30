# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Tensor class with NVFP4 data"""
from __future__ import annotations
from collections.abc import Iterable
import math
from typing import Optional, Tuple, Union

import torch
import transformer_engine_torch as tex

from transformer_engine.common.recipe import NVFP4BlockScaling, Recipe
from ..constants import NVFP4_BLOCK_SCALING_SIZE
from ..utils import devices_match, round_up_to_nearest_multiple

from ._internal.nvfp4_tensor_base import NVFP4TensorBase, _FromNVFP4Func
from .quantized_tensor import QuantizedTensor, Quantizer, _IdentityFunc

aten = torch.ops.aten


class NVFP4Quantizer(Quantizer):
    """Builder class for NVFP4 tensors with NV block scaling"""

    def __init__(
        self,
        *,
        rowwise: bool = True,
        columnwise: bool = True,
    ) -> None:
        super().__init__(rowwise=rowwise, columnwise=columnwise)

    def update_quantized(
        self,
        src: torch.Tensor,
        dst: QuantizedTensor,
        *,
        noop_flag: Optional[torch.Tensor] = None,
    ) -> QuantizedTensor:

        assert isinstance(dst, NVFP4Tensor), f"Cannot store quantized NVFP4 in {type(dst)} type."

        # Make sure input is in expected format
        if not devices_match(src.device, dst.device):
            src = src.to(device=dst.device)
        if not src.is_contiguous():
            src = src.contiguous()

        # Launch cast kernel
        tex.quantize(src, self, dst, noop_flag)

        return dst

    def is_quantizable(self, inp: torch.Tensor) -> bool:
        """Returns whether or not given inp can be quantized"""
        if inp.ndim < 2:
            return False
        if inp.shape[-1] % NVFP4_BLOCK_SCALING_SIZE != 0:
            return False
        if math.prod(inp.shape[:-1]) % NVFP4_BLOCK_SCALING_SIZE != 0:
            return False
        return True

    def make_empty(
        self,
        shape: Iterable[int],
        *,
        dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None,
        requires_grad: bool = False,
    ) -> NVFP4Tensor:

        # Canonicalize tensor attributes
        if device is None:
            device = torch.device("cuda")

        assert shape[-1] % NVFP4_BLOCK_SCALING_SIZE == 0, (
            f"Incorrect shape {shape} for NVFP4. Tensor dims must be divisible by"
            f" {NVFP4_BLOCK_SCALING_SIZE}"
        )

        assert math.prod(shape[:-1]) % NVFP4_BLOCK_SCALING_SIZE == 0, (
            f"Incorrect shape {shape} for NVFP4. Tensor dims must be divisible by"
            f" {NVFP4_BLOCK_SCALING_SIZE}"
        )

        # Allocate FP4 data
        data = torch.empty(shape, dtype=torch.uint4, device=device)

        # Allocate block scale inverse. E4M3 format.
        scale_inv = torch.zeros(
            round_up_to_nearest_multiple(math.prod(shape[:-1]), 128),
            round_up_to_nearest_multiple(shape[-1] // NVFP4_BLOCK_SCALING_SIZE, 4),
            dtype=torch.uint4,
            device=device,
        )

        # Allocate per tensor scale inverse. FP32 format.
        per_tensor_scale_inv = torch.ones(1, dtype=torch.float32, device=device)

        # Allocate FP8 data transpose if needed
        columnwise_data = None
        columnwise_scale_inv = None
        if self.columnwise_usage:
            columnwise_data = torch.empty_like(data)
            columnwise_scale_inv = torch.zeros(
                round_up_to_nearest_multiple(math.prod(shape[:-1]) // NVFP4_BLOCK_SCALING_SIZE, 4),
                round_up_to_nearest_multiple(shape[-1], 128),
                dtype=torch.uint4,
                device=device,
            )

        # Construct FP8 tensor
        return NVFP4Tensor(
            shape=shape,
            dtype=dtype,
            rowwise_data=data,
            rowwise_scale_inv=scale_inv,
            columnwise_data=columnwise_data,
            columnwise_scale_inv=columnwise_scale_inv,
            per_tensor_rowwise_scale_inv=per_tensor_scale_inv,
            quantizer=self,
            requires_grad=requires_grad,
        )

    def calibrate(self, tensor: torch.Tensor) -> None:
        # TODO(ksivamani): No calibration?
        pass

    def _get_compatible_recipe(self) -> Union[type[Recipe], None]:
        return NVFP4BlockScaling


class NVFP4Tensor(NVFP4TensorBase, QuantizedTensor):
    """Experimental tensor class with Hybrid FP8 data

    The tensor presents as having a standard, higher-precision dtype,
    but the data itself is (scaled) FP8. For most tensor operations,
    the data will be cast to the nominal dtype before performing the
    operation.

    Parameters
    ----------
    data: torch.Tensor
          Raw FP8 data in a uint4 tensor
    fp8_scale_inv: torch.Tensor
                   Reciprocal of the scaling factor applied when
                   casting to FP8, i.e. the scaling factor that must
                   be applied when casting from FP8 to higher
                   precision.
    dtype: torch.dtype, default = torch.float32
           Nominal tensor datatype.
    """

    def __repr__(self, *, tensor_contents=None):
        return f"NVFP4Tensor, data={self.dequantize(dtype=self.dtype)})"

    def dequantize(self, *, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        """
        Construct plain PyTorch tensor from NVFP4Tensor

        By default the resulting tensor's dtype is the
        NVFP4Tensor's nominal dtype.
        """
        # Convert PyTorch dtype to TE dtype
        if dtype is None:
            dtype = self.dtype

        if torch.is_grad_enabled():
            return _FromNVFP4Func.apply(self, dtype)
        return _FromNVFP4Func.forward(None, self, dtype)

    def _get_quantizer(self) -> Quantizer:
        """Get builder for quantized tensor

        Quantizer can be used for in-place operations.

        """
        if self._quantizer is not None:
            return self._quantizer
        return NVFP4Quantizer()

    def quantize_(
        self,
        tensor: torch.Tensor,
        *,
        noop_flag: Optional[torch.Tensor] = None,
    ) -> NVFP4Tensor:
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

    def detach(self) -> NVFP4Tensor:
        # pylint: disable=missing-function-docstring
        # TODO(ksivamani): Fix the detach bug
        return NVFP4Tensor.make_like(self)

    def clone(self) -> NVFP4Tensor:
        # pylint: disable=missing-function-docstring
        assert self._rowwise_data is not None
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

    def view(self, *shape: Tuple[int]) -> NVFP4Tensor:
        # pylint: disable=missing-function-docstring
        return _ViewFunc.apply(self, shape)

    def reshape(self, *shape: Tuple[int]) -> NVFP4Tensor:
        # pylint: disable=missing-function-docstring
        return _ReshapeFunc.apply(self, shape)

    def contiguous(
        self,
        memory_format: torch.memory_format = torch.contiguous_format,
    ) -> NVFP4Tensor:
        """Returns tensor with data in provided memory format

        Returns `self` if data is already in correct memory format.

        """
        if self._rowwise_data is not None and self._rowwise_data.is_contiguous(
            memory_format=memory_format
        ):
            return self
        if self._columnwise_data is not None and self._columnwise_data.is_contiguous(
            memory_format=memory_format
        ):
            return self
        raise ValueError("NVFP4Tensor does not support different memory formats!")

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
            return NVFP4Tensor(
                shape=out_shape,
                dtype=tensor.dtype,
                rowwise_data=out_data,
                rowwise_scale_inv=tensor._rowwise_scale_inv,
                columnwise_data=tensor._columnwise_data,
                columnwise_scale_inv=tensor._columnwise_scale_inv,
                per_tensor_rowwise_scale_inv=tensor._per_tensor_rowwise_scale_inv,
                quantizer=tensor._quantizer,
                requires_grad=False,
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
        per_tensor_rowwise_scale_inv: torch.Tensor,
        dtype: torch.dtype,
        shape: torch.shape,
    ) -> NVFP4Tensor:
        """Build NVFP4Tensor, for use in __reduce__

        __reduce_ex__ assumes object constructor has positional
        arguments.

        """
        return NVFP4Tensor(
            rowwise_data=rowwise_data,
            rowwise_scale_inv=rowwise_scale_inv,
            columnwise_data=columnwise_data,
            columnwise_scale_inv=columnwise_scale_inv,
            per_tensor_rowwise_scale_inv=per_tensor_rowwise_scale_inv,
            dtype=dtype,
            shape=shape,
        )

    def __reduce_ex__(self, protocol: int) -> tuple:
        """Custom pickling"""
        return (
            NVFP4Tensor._make_in_reduce_ex,
            (
                self._rowwise_data,
                self._rowwise_scale_inv,
                self._columnwise_data,
                self._columnwise_scale_inv,
                self._per_tensor_rowwise_scale_inv,
                self.dtype,
                self.shape,
            ),
        )

    def _get_data(self) -> NVFP4Tensor:
        """Get tensor data property"""
        return super().data

    @torch.no_grad()
    def _set_data(self, tensor: torch.Tensor) -> None:
        """Set tensor data property

        Just takes FP8 data if setting from a NVFP4Tensor. Otherwise
        casts to FP8.

        """

        # Tensor device
        new_device = tensor.device if tensor.is_cuda else self.device
        if not devices_match(new_device, tensor.device):
            tensor = tensor.to(device=new_device)

        # Just copy FP8 data if other tensor is NVFP4Tensor
        if isinstance(tensor, NVFP4Tensor):
            if (  # pylint: disable=too-many-boolean-expressions
                self.size() != tensor.size()
                or self.stride() != tensor.stride()
                or self.storage_offset() != tensor.storage_offset()
                or self.dtype != tensor.dtype
                or self.layout != tensor.layout
                or not devices_match(self.device, new_device)
            ):
                dummy_tensor = torch.Tensor._make_wrapper_subclass(
                    NVFP4Tensor,
                    tensor.size(),
                    strides=tensor.stride(),
                    storage_offset=tensor.storage_offset(),
                    dtype=tensor.dtype,
                    layout=tensor.layout,
                    requires_grad=tensor.requires_grad,
                    device=new_device,
                )
                # pylint: disable=unnecessary-dunder-call
                super(NVFP4Tensor, type(self)).data.__set__(self, dummy_tensor)
            self._rowwise_data = tensor._rowwise_data
            self._columnwise_data = tensor._columnwise_data
            self._quantizer = tensor._quantizer
            self._rowwise_scale_inv = tensor._rowwise_scale_inv
            self._columnwise_scale_inv = tensor._columnwise_scale_inv
            self._per_tensor_rowwise_scale_inv = tensor._per_tensor_rowwise_scale_inv
            return

        # Quantize to FP8
        assert self._quantizer is not None, "Can't quantize without a quantizer"
        self.data = self._quantizer.quantize(tensor)
        if self.requires_grad != tensor.requires_grad:
            self.requires_grad_(requires_grad=tensor.requires_grad)

    # Cast to FP8 when setting NVFP4Tensor.data
    data = property(_get_data, _set_data)


class _ViewFunc(torch.autograd.Function):
    """View function

    View the NVFP4Tensor using the provided shape.

    """

    @staticmethod
    def forward(
        ctx,
        tensor: NVFP4Tensor,
        shape: Optional[list[int]] = None,
    ) -> NVFP4Tensor:
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
        if shape[-1] != ctx.shape[-1]:
            raise RuntimeError(
                "NVFP4Tensor does not support reshaping inner dimension "
                f"(attempted to reshape dims={tuple(tensor.shape)} to {tuple(shape)})"
            )

        # Construct new tensor if shape is provided
        new_rowwise_data = None
        new_columnwise_data = None
        if tensor._rowwise_data is not None:
            new_rowwise_data = tensor._rowwise_data.view(*shape)
        if tensor._columnwise_data is not None:
            columnwise_shape = [shape[-1]] + list(shape[:-1])
            new_columnwise_data = tensor._columnwise_data.view(columnwise_shape)
        return NVFP4Tensor(
            shape,
            tensor.dtype,
            rowwise_data=new_rowwise_data,
            rowwise_scale_inv=tensor._rowwise_scale_inv,
            columnwise_data=new_columnwise_data,
            columnwise_scale_inv=tensor._columnwise_scale_inv,
            per_tensor_rowwise_scale_inv=tensor._per_tensor_rowwise_scale_inv,
            quantizer=tensor._quantizer,
        )

    @staticmethod
    def backward(
        ctx,
        grad: torch.Tensor,
    ) -> Tuple[Optional[torch.Tensor], ...]:
        # pylint: disable=missing-function-docstring

        if isinstance(grad, NVFP4Tensor):
            new_data = (
                grad._rowwise_data.view(*ctx.shape) if grad._rowwise_data is not None else None
            )
            if grad._columnwise_data is not None:
                new_columnwise_data = grad._columnwise_data.view(ctx.shape[-1], -1)
            else:
                new_columnwise_data = None
            dgrad = NVFP4Tensor(
                ctx.shape,
                grad.dtype,
                rowwise_data=new_data,
                rowwise_scale_inv=grad._rowwise_scale_inv,
                columnwise_data=new_columnwise_data,
                columnwise_scale_inv=grad._columnwise_scale_inv,
                per_tensor_rowwise_scale_inv=grad._per_tensor_rowwise_scale_inv,
                quantizer=grad._quantizer,
            )
            return dgrad, None
        return grad.view(ctx.shape), None


class _ReshapeFunc(torch.autograd.Function):
    """Reshape function

    Reshape the NVFP4Tensor using the provided shape.

    """

    @staticmethod
    def forward(
        ctx,
        tensor: NVFP4Tensor,
        shape: Optional[list[int]] = None,
    ) -> NVFP4Tensor:
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
        if shape[-1] != ctx.shape[-1]:
            raise RuntimeError(
                "NVFP4Tensor does not support reshaping inner dimension "
                f"(attempted to reshape dims={tuple(tensor.shape)} to {tuple(shape)})"
            )

        # Construct new tensor if shape is provided
        new_rowwise_data = None
        new_columnwise_data = None
        if tensor._rowwise_data is not None:
            new_rowwise_data = tensor._rowwise_data.reshape(*shape)
        if tensor._columnwise_data is not None:
            columnwise_shape = [shape[-1]] + list(shape[:-1])
            new_columnwise_data = tensor._columnwise_data.view(columnwise_shape)

        return NVFP4Tensor(
            shape,
            tensor.dtype,
            rowwise_data=new_rowwise_data,
            rowwise_scale_inv=tensor._rowwise_scale_inv,
            columnwise_data=new_columnwise_data,
            columnwise_scale_inv=tensor._columnwise_scale_inv,
            per_tensor_rowwise_scale_inv=tensor._per_tensor_rowwise_scale_inv,
            quantizer=tensor._quantizer,
        )

    @staticmethod
    def backward(
        ctx,
        grad: torch.Tensor,
    ) -> Tuple[Optional[torch.Tensor], ...]:
        # pylint: disable=missing-function-docstring

        if isinstance(grad, NVFP4Tensor):
            new_rowwise_data = None
            new_columnwise_data = None
            if grad._rowwise_data is not None:
                new_rowwise_data = grad._rowwise_data.view(*ctx.shape)
            if grad._columnwise_data is not None:
                columnwise_shape = [ctx.shape[-1]] + list(ctx.shape[:-1])
                new_columnwise_data = grad._columnwise_data.view(columnwise_shape)
            dgrad = NVFP4Tensor(
                ctx.shape,
                grad.dtype,
                rowwise_data=new_rowwise_data,
                rowwise_scale_inv=grad._rowwise_scale_inv,
                columnwise_data=new_columnwise_data,
                columnwise_scale_inv=grad._columnwise_scale_inv,
                per_tensor_rowwise_scale_inv=grad._per_tensor_rowwise_scale_inv,
                quantizer=grad._quantizer,
            )
            return dgrad, None
        return grad.view(ctx.shape), None
