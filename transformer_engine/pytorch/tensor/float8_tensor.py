# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Tensor class with FP8 data"""
from __future__ import annotations
from typing import Optional, Tuple, Iterable
import warnings

import torch
import transformer_engine_torch as tex

from transformer_engine_torch import DType as TE_DType
from ..utils import devices_match, non_tn_fp8_gemm_supported
from ._internal.float8_tensor_base import Float8TensorBase, _FromFloat8Func
from .quantized_tensor import QuantizedTensor, Quantizer, _IdentityFunc

aten = torch.ops.aten

_ops_to_preserve_subclass_in_fsdp2 = {
    torch.ops.aten.empty_like.default,
    torch.ops.aten.new_zeros.default,
    torch.ops.aten.slice.Tensor,
    torch.ops.aten.copy_.default,
    torch.ops.aten.view.default,
    torch.ops.aten.as_strided.default,
    torch.ops.aten._to_copy.default,
    torch.ops.aten._pin_memory.default,
    torch.ops.aten.split.Tensor,
    torch.ops.aten.clone.default,
}


class Float8Quantizer(Quantizer):
    """Builder class for FP8 tensors with per-tensor delayed scaling

    High-precision tensors (e.g. in FP32 or BF16) are quantized by
    multiplying with a scaling factor and casting to FP8. The max-abs
    value ("amax") in the tensor is also computed, which can be used
    for updating the scaling factor (handled externally by
    DelayedScalingRecipeState and FP8GlobalStateManager).

    """

    """Scaling factor to multiply when quantizing to FP8"""
    scale: torch.Tensor
    """Max-abs value from last FP8 cast"""
    amax: torch.Tensor
    """FP8 datatype"""
    dtype: TE_DType

    def __init__(
        self,
        scale: torch.Tensor,
        amax: torch.Tensor,
        fp8_dtype: TE_DType,
        *,
        rowwise: bool = True,
        columnwise: bool = True,
    ) -> None:
        super().__init__(rowwise=rowwise, columnwise=columnwise)
        self.scale = scale
        self.amax = amax
        self.dtype = fp8_dtype

    def update_quantized(
        self,
        src: torch.Tensor,
        dst: QuantizedTensor,
        *,
        noop_flag: Optional[torch.Tensor] = None,
    ) -> QuantizedTensor:
        if not isinstance(dst, Float8Tensor):
            raise ValueError("Float8Quantizer can only update Float8Tensor")

        # Make sure input is in expected format
        if not devices_match(src.device, dst.device):
            src = src.to(device=dst.device)
        if not src.is_contiguous():
            src = src.contiguous()

        # Launch cast kernel
        tex.quantize(src, self, dst, noop_flag)

        # Update FP8 dtype
        dst._fp8_dtype = self.dtype

        return dst

    def make_empty(
        self,
        shape: Iterable[int],
        *,
        dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None,
        requires_grad: bool = False,
    ) -> Float8Tensor:

        # Canonicalize tensor attributes
        if device is None:
            device = torch.device("cuda")

        # Allocate FP8 data
        data = torch.empty(shape, dtype=torch.uint8, device=device)

        # Allocate FP8 data transpose if needed
        data_transpose = None
        if self.columnwise_usage:
            inner_dim = data.size(-1)
            data_transpose = torch.empty(
                inner_dim,
                data.numel() // inner_dim,
                dtype=torch.uint8,
                device=device,
            )

        # Construct FP8 tensor
        return Float8Tensor(
            shape=shape,
            dtype=dtype,
            data=data,
            fp8_scale_inv=torch.empty(1, dtype=torch.float32, device=device),
            fp8_dtype=self.dtype,
            requires_grad=requires_grad,
            data_transpose=data_transpose,
            quantizer=self,
        )

    def calibrate(self, tensor: torch.Tensor) -> None:
        amin, amax = tensor.aminmax()
        self.amax.copy_(torch.max(-amin, amax))

    def create_tensor_from_data(
        self,
        data: torch.Tensor,
        fake_dtype=torch.float32,
        requires_grad: bool = False,
        internal: bool = False,
    ):
        """Create Float8Tensor from raw uint8 data"""
        assert data.dtype in [
            torch.uint8,
            torch.float8_e4m3fn,
            torch.float8_e4m3fnuz,
            torch.float8_e5m2,
            torch.float8_e5m2fnuz,
        ]
        if internal:
            return Float8TensorBase(
                data=data,
                fp8_scale_inv=1 / self.scale,
                fp8_dtype=self.dtype,
                requires_grad=requires_grad,
                data_transpose=None,
                quantizer=self,
            )
        return Float8Tensor(
            shape=data.shape,
            dtype=fake_dtype,
            data=data,
            fp8_scale_inv=1 / self.scale,
            fp8_dtype=self.dtype,
            requires_grad=requires_grad,
            data_transpose=None,
            quantizer=self,
        )


class Float8Tensor(Float8TensorBase, QuantizedTensor):
    """Experimental tensor class with FP8 data

    The tensor presents as having a standard, higher-precision dtype,
    but the data itself is (scaled) FP8. For most tensor operations,
    the data will be cast to the nominal dtype before performing the
    operation.

    Parameters
    ----------
    shape: int or iterable of int
        Tensor dimensions.
    dtype: torch.dtype
        Nominal tensor datatype.
    requires_grad: bool, optional = False
        Whether to compute gradients for this tensor.
    data: torch.Tensor
        Raw FP8 data in a uint8 tensor
    fp8_scale_inv: torch.Tensor
        Reciprocal of the scaling factor applied when casting to FP8,
        i.e. the scaling factor that must be applied when casting from
        FP8 to higher precision.
    fp8_dtype: transformer_engine_torch.DType
        FP8 format.
    data_transpose: torch.Tensor, optional
        FP8 transpose data in a uint8 tensor
    quantizer: Float8Quantizer, optional
        Builder class for FP8 tensors

    """

    def __repr__(self, *, tensor_contents=None):
        return (
            "Float8Tensor("
            f"fp8_dtype={self._fp8_dtype}, "
            f"scale_inv={self._scale_inv.item()}, "
            f"data={self.dequantize(dtype=self.dtype)}"
            ")"
        )

    def dequantize(self, *, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        """
        Construct plain PyTorch tensor from Float8Tensor

        By default the resulting tensor's dtype is the
        Float8Tensor's nominal dtype.
        """
        # Convert PyTorch dtype to TE dtype
        if dtype is None:
            dtype = self.dtype

        if torch.is_grad_enabled():
            return _FromFloat8Func.apply(self, dtype)
        return _FromFloat8Func.forward(None, self, dtype)

    def _get_quantizer(self) -> Quantizer:
        """Get builder for quantized tensor

        Quantizer can be used for in-place operations.

        """
        if self._quantizer is not None:
            return self._quantizer
        return Float8Quantizer(
            scale=torch.reciprocal(self._scale_inv),
            amax=torch.empty(1, dtype=torch.float32, device=self.device),
            fp8_dtype=self._fp8_dtype,
        )

    def quantize_(
        self,
        tensor: torch.Tensor,
        *,
        noop_flag: Optional[torch.Tensor] = None,
    ) -> Float8Tensor:
        """Update FP8 data

        Parameters
        ----------
        tensor: torch.Tensor
            Tensor to copy from
        noop_flag: torch.Tensor, optional
            float32 flag indicating whether to avoid performing update

        """
        if isinstance(tensor, QuantizedTensor):
            return self.quantize_(tensor.dequantize(), noop_flag=noop_flag)
        self._get_quantizer().update_quantized(tensor, self, noop_flag=noop_flag)
        return self

    def detach(self) -> Float8Tensor:
        # pylint: disable=missing-function-docstring
        return Float8Tensor.make_like(self)

    def _create_transpose(self):
        data = self._data
        if not data.is_contiguous():
            data = data.contiguous()
        self._transpose = tex.fp8_transpose(data, self._fp8_dtype, out=self._transpose)
        self._transpose_invalid = False

    def update_usage(self, rowwise_usage=True, columnwise_usage=True):
        assert rowwise_usage or columnwise_usage, "Could not disable all usages of the tensor"
        if rowwise_usage:
            assert self._data is not None, "Rowwise usage of the tensor was already disabled"
        else:
            if not non_tn_fp8_gemm_supported():
                if self._transpose is None or self._transpose_invalid:
                    self._create_transpose()
                self._data = None
        if columnwise_usage:
            if self._transpose is None or self._transpose_invalid:
                assert self._data is not None, "The tensor does not hold any data anymore"
                if not non_tn_fp8_gemm_supported():
                    self._create_transpose()
        else:
            self._transpose = None
            self._transpose_invalid = True

    def clone(self) -> Float8Tensor:
        # pylint: disable=missing-function-docstring
        assert self._data is not None
        data = self._data.detach().clone()
        data_transpose = None
        if self._transpose is not None:
            data_transpose = self._transpose.detach().clone()
        return _IdentityFunc.apply(
            self,
            {
                "data": data,
                "data_transpose": data_transpose,
            },
        )

    def view(self, *shape: Tuple[int]) -> Float8Tensor:
        # pylint: disable=missing-function-docstring
        return _ViewFunc.apply(self, shape)

    def reshape(self, *shape: Tuple[int]) -> Float8Tensor:
        # pylint: disable=missing-function-docstring
        return _ReshapeFunc.apply(self, shape)

    def contiguous(
        self,
        memory_format: torch.memory_format = torch.contiguous_format,
    ) -> Float8Tensor:
        """Returns tensor with data in provided memory format

        Returns `self` if data is already in correct memory format.

        """
        if self._data is not None and self._data.is_contiguous(memory_format=memory_format):
            return self
        if self._transpose is not None and self._transpose.is_contiguous(
            memory_format=memory_format
        ):
            return self
        return Float8Tensor.make_like(tensor=self, data=self._data.contiguous())

        # raise ValueError("Float8Tensor does not support different memory formats!")

    def _reset_caches(self) -> None:
        """
        Set transpose cache as invalid.
        Should be called after any in-place operation.
        """
        self._transpose_invalid = True

    def remove_caches(self) -> None:
        """
        Remove transpose cache and mark it as invalid.
        """
        self._transpose_invalid = True
        del self._transpose  # explicitly deletes the data for safety
        self._transpose = None

    def clear(self):
        """Deallocate this tensor's memory. Typically not needed and must be used carefully."""
        self._data = torch.Tensor() if self._data is not None else None
        self._transpose = torch.Tensor() if self._transpose is not None else None
        self._transpose_invalid = True

    def prepare_for_saving(self) -> Tuple[list[Optional[torch.Tensor]], Float8TensorBase]:
        """Prepare the tensor base for saving for backward

        After calling this, the tensor instance does not hold any
        data.

        """
        return [self], None

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs=None):

        # View op
        if func == aten.view.default:
            tensor = args[0]
            data = tensor._data
            out_data = data.__torch_dispatch__(
                func,
                types,
                [data] + list(args[1:]),
                kwargs,
            )
            out_shape = out_data.size()
            out_transpose = None if tensor._transpose_invalid else tensor._transpose
            if out_transpose is not None:
                out_transpose_shape = out_transpose.size()
                if (
                    out_transpose_shape[0] != out_shape[-1]
                    or out_transpose_shape[1:] != out_shape[:-1]
                ):
                    out_transpose = None
            return Float8Tensor(
                shape=out_shape,
                dtype=tensor.dtype,
                requires_grad=False,
                data=out_data,
                fp8_scale_inv=tensor._scale_inv,
                fp8_dtype=tensor._fp8_dtype,
                data_transpose=out_transpose,
                quantizer=tensor._quantizer,
            )

        if func in [aten.slice.Tensor, aten.select.int]:
            tensor = args[0]
            data = tensor._data
            data_slice = data.__torch_dispatch__(
                func,
                types,
                [data] + list(args[1:]),
                kwargs,
            )
            return Float8Tensor.make_like(tensor, data=data_slice, shape=data_slice.shape)

        # Related to FSDP2
        if func == aten.split.Tensor:
            tensor = args[0]
            data = tensor._data
            func_out = data.__torch_dispatch__(
                func,
                types,
                [data] + list(args[1:]),
                kwargs,
            )
            return [
                Float8Tensor.make_like(tensor, data=split_tensor, shape=split_tensor.shape)
                for split_tensor in func_out
            ]
        if func == aten.new_zeros.default:
            tensor = args[0]
            data = tensor._data
            func_out = data.__torch_dispatch__(
                func,
                types,
                [data] + list(args[1:]),
                kwargs,
            )
            return Float8Tensor.make_like(tensor, data=func_out, shape=func_out.shape)
        if func == torch.ops.aten.as_strided.default:
            tensor = args[0]
            data = tensor._data
            func_out = data.__torch_dispatch__(
                func,
                types,
                [data] + list(args[1:]),
                kwargs,
            )
            return Float8Tensor.make_like(tensor, data=func_out, shape=func_out.shape)
        if func == torch.ops.aten.detach.default:
            return cls.detach(args[0])
        if func == torch.ops.aten.clone.default:
            return cls.clone(args[0])
        if func == torch.ops.aten.copy_.default:
            dst, src = args[0], args[1]
            # Just copy FP8 attrs if copying between Float8Tensors
            if isinstance(src, Float8Tensor) and isinstance(dst, Float8Tensor):
                dst._data.copy_(src._data.detach())
                dst._scale_inv.copy_(src._scale_inv.view(dst._scale_inv.size()))
                if src._transpose is not None or dst._transpose is not None:
                    dst._create_transpose()
                return dst
        elif func in _ops_to_preserve_subclass_in_fsdp2:
            # Ops in the _ops_to_preserve_subclass_in_fsdp2 are recommened to return the same class instance to work fine with the torch fsdp2
            warnings.warn(
                f"A function call({func}) in {cls} may not return {cls} tensor as an output. It"
                " might cause an error in torch FSDP2!"
            )
        else:
            pass

        return super().__torch_dispatch__(func, types, args, kwargs)

    @classmethod
    def _make_in_reduce_ex(
        cls,
        data: torch.Tensor,
        fp8_dtype: TE_DType,
        fp8_scale_inv: torch.Tensor,
        dtype: torch.dtype,
        shape: torch.shape,
    ) -> Float8Tensor:
        """Build Float8Tensor, for use in __reduce__

        __reduce_ex__ assumes object constructor has positional
        arguments.

        """
        return Float8Tensor(
            data=data,
            fp8_dtype=fp8_dtype,
            fp8_scale_inv=fp8_scale_inv,
            dtype=dtype,
            shape=shape,
        )

    def __reduce_ex__(self, protocol: int) -> tuple:
        """Custom pickling to remove references to FP8 metadata objects"""
        return (
            Float8Tensor._make_in_reduce_ex,
            (self._data, self._fp8_dtype, self._scale_inv, self.dtype, self.shape),
        )

    def _get_data(self) -> Float8Tensor:
        """Get tensor data property"""
        return super().data

    @torch.no_grad()
    def _set_data(self, tensor: torch.Tensor) -> None:
        """Set tensor data property

        Just takes FP8 data if setting from a Float8Tensor. Otherwise
        casts to FP8.

        """

        # Tensor device
        new_device = tensor.device if tensor.is_cuda else self.device
        if not devices_match(new_device, tensor.device):
            tensor = tensor.to(device=new_device)

        # Just copy FP8 data if other tensor is Float8Tensor
        if isinstance(tensor, Float8Tensor):

            # PyTorch tensor attributes
            if (  # pylint: disable=too-many-boolean-expressions
                self.size() != tensor.size()
                or self.stride() != tensor.stride()
                or self.storage_offset() != tensor.storage_offset()
                or self.dtype != tensor.dtype
                or self.layout != tensor.layout
                or not devices_match(self.device, new_device)
            ):
                dummy_tensor = torch.Tensor._make_wrapper_subclass(
                    Float8Tensor,
                    tensor.size(),
                    strides=tensor.stride(),
                    storage_offset=tensor.storage_offset(),
                    dtype=tensor.dtype,
                    layout=tensor.layout,
                    requires_grad=tensor.requires_grad,
                    device=new_device,
                )
                # pylint: disable=unnecessary-dunder-call
                super(Float8Tensor, type(self)).data.__set__(self, dummy_tensor)

            # Float8Tensor attributes
            self._data = tensor._data
            self._quantizer = tensor._quantizer
            self._fp8_dtype = tensor._fp8_dtype
            self._scale_inv = tensor._scale_inv
            self._transpose = tensor._transpose
            self._transpose_invalid = tensor._transpose_invalid
            return

        # Quantize to FP8
        assert self._quantizer is not None, "Can't quantize without a quantizer"
        self._quantizer.internal = False
        self.data = self._quantizer.quantize(tensor)
        if self.requires_grad != tensor.requires_grad:
            self.requires_grad_(requires_grad=tensor.requires_grad)

    # Cast to FP8 when setting Float8Tensor.data
    data = property(_get_data, _set_data)


class _ViewFunc(torch.autograd.Function):
    """View function

    View the Float8Tensor using the provided shape.

    """

    @staticmethod
    def forward(
        ctx,
        tensor: Float8Tensor,
        shape: Optional[list[int]] = None,
    ) -> Float8Tensor:
        # pylint: disable=missing-function-docstring
        ctx.shape = tensor.shape
        if shape is None:
            return tensor.detach()
        out_data = tensor._data.view(*shape)
        out_shape = out_data.size()
        out_transpose = None if tensor._transpose_invalid else tensor._transpose
        if out_transpose is not None:
            out_transpose_shape = out_transpose.size()
            if out_transpose_shape[0] != out_shape[-1] or out_transpose_shape[1:] != out_shape[:-1]:
                out_transpose = None
        return Float8Tensor(
            shape=out_shape,
            dtype=tensor.dtype,
            requires_grad=tensor.requires_grad,
            data=out_data,
            fp8_scale_inv=tensor._scale_inv,
            fp8_dtype=tensor._fp8_dtype,
            data_transpose=out_transpose,
            quantizer=tensor._quantizer,
        )

    @staticmethod
    def backward(
        ctx,
        grad: torch.Tensor,
    ) -> Tuple[Optional[torch.Tensor], ...]:
        # pylint: disable=missing-function-docstring
        return grad.reshape(ctx.shape), None


class _ReshapeFunc(torch.autograd.Function):
    """Reshape function

    Reshape the Float8Tensor using the provided shape.

    """

    @staticmethod
    def forward(
        ctx,
        tensor: Float8Tensor,
        shape: Tuple[int],
    ) -> Float8Tensor:
        # pylint: disable=missing-function-docstring
        ctx.shape = tensor.shape
        if shape is None:
            return tensor.detach()
        out_data = tensor._data.reshape(*shape)
        out_shape = out_data.size()
        out_transpose = None if tensor._transpose_invalid else tensor._transpose
        if out_transpose is not None:
            out_transpose_shape = out_transpose.size()
            if out_transpose_shape[0] != out_shape[-1] or out_transpose_shape[1:] != out_shape[:-1]:
                out_transpose = None
        return Float8Tensor(
            shape=out_shape,
            dtype=tensor.dtype,
            requires_grad=tensor.requires_grad,
            data=out_data,
            fp8_scale_inv=tensor._scale_inv,
            fp8_dtype=tensor._fp8_dtype,
            data_transpose=out_transpose,
            quantizer=tensor._quantizer,
        )

    @staticmethod
    def backward(
        ctx,
        grad: torch.Tensor,
    ) -> Tuple[Optional[torch.Tensor], ...]:
        # pylint: disable=missing-function-docstring
        return grad.reshape(ctx.shape), None
