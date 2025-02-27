# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Tensor with quantized data"""

from __future__ import annotations
from typing import Optional, Tuple, Iterable, Any, Dict, Union
import abc
import copy

import torch
from torch.utils._pytree import tree_map

import transformer_engine_torch as tex


def prepare_for_saving(
    *tensors,
) -> Tuple[list[Optional[Union[torch.Tensor, torch.nn.Parameter]]], Optional[Any]]:
    """Prepare tensors for saving. Needed because save_for_backward accepts only
    torch.Tensor/torch.nn.Parameter types, while we want to be able to save
    the internal TensorBase types too."""
    # pylint: disable=unidiomatic-typecheck  # Using type instead of isinstance to check exact type
    tensor_list, tensor_objects_list = [], []
    for tensor in tensors:
        if tensor is None:
            tensor_list.append(None)
            tensor_objects_list.append(None)
        elif type(tensor) in (torch.Tensor, torch.nn.Parameter):
            tensor_list.append(tensor)
            tensor_objects_list.append(None)
        else:
            t, t_obj = tensor.prepare_for_saving()
            tensor_list.extend(t)
            tensor_objects_list.append(t_obj)
    return tensor_list, tensor_objects_list


def restore_from_saved(
    tensors: list[Optional[Any]],
    saved_tensors: list[Optional[Union[torch.Tensor, torch.nn.Parameter]]],
) -> list[Optional[Any]]:
    """Recombine the tensor data and metadata during backward pass."""
    tensor_objects = []
    for tensor in tensors:
        if tensor is None:
            tensor_objects.append(saved_tensors[0])
            saved_tensors = saved_tensors[1:]
        else:
            saved_tensors = tensor.restore_from_saved(saved_tensors)
            tensor_objects.append(tensor)
    return tensor_objects


class Quantizer(abc.ABC):
    """Builder class for quantized tensors.

    This class is typically used to convert a high-precision tensor
    (e.g. in FP32 or BF16) into a quantized tensor (e.g. in FP8).

    """

    """Whether to construct quantized tensors with "row-wise usage"

    Hand-wave explanation: Consider the matrix multiplication C = A *
    B^T (used in linear forward). Tensor Cores prefer "TN GEMMs" (in
    Fortran-style column-major order), so A and B should be in
    row-major order.

    """
    rowwise_usage: bool

    """Whether to construct quantized tensors with "column-wise usage"

    Hand-wave explanation: Consider the matrix multiplication C = A^T
    * B (used in linear backward wgrad). Tensor Cores prefer "TN
    GEMMs" (in Fortran-style column-major order), so A and B should be
    in column-major order.

    """
    columnwise_usage: bool

    """Whether to instantiates tensor for purely internal usage

    Internal tensors are storage classes with minimal logic. They have
    less overhead than PyTorch tensor sub-classes, but are not
    compatible with PyTorch's autograd infrastructure nor PyTorch
    operations.

    """
    internal: bool

    def __init__(self, *, rowwise: bool, columnwise: bool) -> None:
        self.rowwise_usage = rowwise
        self.columnwise_usage = columnwise
        self.internal = False

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"rowwise_usage={self.rowwise_usage}, "
            f"columnwise_usage={self.columnwise_usage}, "
            f"internal={self.internal}, "
            ")"
        )

    @abc.abstractmethod
    def update_quantized(
        self,
        src: torch.Tensor,
        dst: QuantizedTensor,
        *,
        noop_flag: Optional[torch.Tensor] = None,
    ) -> QuantizedTensor:
        """Quantize tensor in-place"""

    def quantize(
        self, tensor: torch.Tensor, *, out: Optional[QuantizedTensor] = None
    ) -> QuantizedTensor:
        """Quantize tensor"""
        if out is not None:
            return self.update_quantized(tensor, out)
        if (not self.internal) and torch.is_grad_enabled():
            return _QuantizeFunc.apply(tensor, self)
        return _QuantizeFunc.forward(None, tensor, self)

    def multi_quantize(self, list_of_tensors):
        """Quantize multiple tensors"""
        list_of_output_tensors = []
        for tensor in list_of_tensors:
            list_of_output_tensors.append(self.quantize(tensor))
        return list_of_output_tensors

    def __call__(self, tensor: torch.Tensor) -> QuantizedTensor:
        """Quantize tensor"""
        return self.quantize(tensor)

    @abc.abstractmethod
    def make_empty(
        self,
        shape: Iterable[int],
        *,
        dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None,
    ) -> QuantizedTensor:
        """Construct quantized tensor with uninitialized data"""

    @abc.abstractmethod
    def calibrate(self, tensor: torch.Tensor) -> None:
        """Calibrate quantizer state

        Updates quantization state as if quantizing a tensor, but
        without actually performing the quantization.

        """

    def set_usage(
        self, *, rowwise: Optional[bool] = None, columnwise: Optional[bool] = None
    ) -> None:
        """Set how the quantized tensor is expected to be used

        See documentation for `rowwise_usage` and `columnwise_usage`
        variables.

        """
        if rowwise is not None:
            self.rowwise_usage = rowwise
        if columnwise is not None:
            self.columnwise_usage = columnwise

    def copy(self) -> Quantizer:
        """Create shallow copy"""
        return copy.copy(self)


class _QuantizeFunc(torch.autograd.Function):
    """Cast to FP8 from other dtype"""

    @staticmethod
    def forward(
        _ctx: Optional[torch.autograd.function.FunctionCtx],  # unused
        tensor: torch.Tensor,
        quantizer: Quantizer,
    ) -> QuantizedTensor:
        # pylint: disable=missing-function-docstring
        return tex.quantize(tensor, quantizer)

    @staticmethod
    def backward(
        _ctx: torch.autograd.function.FunctionCtx, grad: torch.Tensor  # unused
    ) -> Tuple[Optional[torch.Tensor], ...]:
        # pylint: disable=missing-function-docstring
        # Assume that we want gradients in full precision
        return grad, None


class _IdentityFunc(torch.autograd.Function):
    """Identity function

    If constructor keyword-arguments are provided, then construct a
    new Float8Tensor using the provided tensor's attributes.

    """

    @staticmethod
    def forward(
        ctx, tensor: QuantizedTensor, init_kwargs: Optional[Dict[str, Any]] = None
    ) -> QuantizedTensor:
        # pylint: disable=missing-function-docstring

        # Return input tensor if constructor kwargs are not provided
        if init_kwargs is None:
            return tensor.detach()

        # Construct new tensor if constructor kwargs are provided
        ctx.input_dtype = tensor.dtype
        kwargs = tensor.get_metadata()
        for key, val in init_kwargs.items():
            kwargs[key] = val
        return type(tensor)(tensor.shape, tensor.dtype, **kwargs)

    @staticmethod
    def backward(ctx, grad_output):
        # pylint: disable=missing-function-docstring
        grad_input = grad_output
        if grad_input.dtype == ctx.input_dtype:
            grad_input = grad_input.detach()
        else:
            grad_input = grad_input.to(ctx.input_dtype)
        return grad_input, None


def _stride_from_shape(shape: list[int]):
    if len(shape) == 0:
        return []
    rstride = [1]
    for d in reversed(shape[1:]):
        rstride.append(rstride[-1] * d)
    return list(reversed(rstride))


class QuantizedTensor(torch.Tensor):
    """Abstract base class for tensor with quantized data

    This is a proxy class with the interface of a standard PyTorch
    tensor, but with data that has been encoded with some quantization
    scheme. Derived classes should implement the quantization scheme
    by overriding the `quantize_` and `dequantize` functions.

    """

    def __new__(cls, shape: Iterable[int], dtype: torch.dtype, *, requires_grad: bool = False):
        # We are assuming only contiguous tensors
        stride = _stride_from_shape(shape)
        instance = torch.Tensor._make_wrapper_subclass(
            cls,
            shape,
            strides=stride,
            storage_offset=0,
            dtype=dtype,
            layout=torch.strided,
            requires_grad=requires_grad,
            device=torch.cuda.current_device(),
        )

        return instance

    def dequantize(self, *, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        """Convert quantized data to standard PyTorch tensor"""
        raise NotImplementedError(
            f"{self.__class__.__name__} class does not implement dequantize function"
        )

    def quantize_(self, tensor: torch.Tensor) -> QuantizedTensor:
        """Update quantized data in-place"""
        raise NotImplementedError(
            f"{self.__class__.__name__} class does not implement quantize_ function"
        )

    def detach(self) -> QuantizedTensor:
        """Create new quantized tensor with same data

        Output tensor must be detached from the current autograd
        graph.

        """
        raise NotImplementedError(
            f"{self.__class__.__name__} class does not implement detach function"
        )

    def update_usage(self, rowwise_usage=True, columnwise_usage=True):
        """Indicate to the tensor how it is going to be used

        This enables optimizations to memory usage in some cases
        where forward and backward passes use the tensor in
        different directions.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} class does not implement update_usage function"
        )

    def clear(self):
        """Deallocate this tensor's memory. Typically not needed and must be used carefully"""

    def __repr__(self, *, tensor_contents=None) -> str:
        return f"{self.__class__.__name__}(data={self.dequantize(dtype=self.dtype)})"

    def float(self) -> torch.Tensor:
        # pylint: disable=missing-function-docstring
        return self.dequantize(dtype=torch.float32)

    def bfloat16(self) -> torch.Tensor:
        # pylint: disable=missing-function-docstring
        return self.dequantize(dtype=torch.bfloat16)

    def half(self) -> torch.Tensor:
        # pylint: disable=missing-function-docstring
        return self.dequantize(dtype=torch.float16)

    def cpu(self, memory_format=torch.preserve_format) -> torch.Tensor:
        # pylint: disable=missing-function-docstring
        return self.dequantize().cpu(memory_format=memory_format)

    def expand_as(self, other: torch.Tensor) -> torch.Tensor:
        # pylint: disable=missing-function-docstring
        if other is self:
            # Note: expand_as is hackily used to create dummy autograd nodes
            # and access the backward graph (see
            # https://github.com/pytorch/pytorch/blob/238fb660851268f44ff88127887041fea352fe48/torch/nn/parallel/distributed.py#L1026).
            # We hackily add a dummy function to handle this case.
            return _IdentityFunc.apply(self)
        return super().expand_as(other)

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs=None):

        # Detach op
        if func == torch.ops.aten.detach.default:
            return args[0].detach()

        # In-place copy op
        if func == torch.ops.aten.copy_.default:
            dst = args[0]
            src = args[1]
            if isinstance(dst, QuantizedTensor):
                dst.quantize_(src)
            else:
                if isinstance(src, QuantizedTensor):
                    src = src.dequantize()
                dst.copy_(src)
            return None

        # View op
        if func == torch.ops.aten.view.default:
            raise NotImplementedError("{cls.__name__} class does not support tensor views")

        def maybe_unwrap(arg):
            if isinstance(arg, QuantizedTensor):
                return arg.dequantize(dtype=arg.dtype)
            return arg

        def maybe_update_inplace(arg, new_arg, schema_arg):
            if (
                isinstance(arg, QuantizedTensor)
                and isinstance(new_arg, torch.Tensor)
                and hasattr(schema_arg, "alias_info")
                and hasattr(schema_arg.alias_info, "is_write")
                and schema_arg.alias_info.is_write
            ):
                arg.quantize_(new_arg)

        # In-place op: dequantize, perform op, and quantize
        if func._schema.is_mutable:
            new_args = tree_map(maybe_unwrap, args)
            new_kwargs = tree_map(maybe_unwrap, kwargs)
            schema_args = func._schema.arguments
            args_len = len(args)
            super().__torch_dispatch__(func, types, new_args, new_kwargs)
            for arg, new_arg, schema_arg in zip(args, new_args, schema_args):
                maybe_update_inplace(arg, new_arg, schema_arg)
            for kwarg, new_kwarg, schema_arg in zip(kwargs, new_kwargs, schema_args[args_len:]):
                assert kwarg == new_kwarg == schema_arg.name, "name of the kw argument should match"
                maybe_update_inplace(kwargs[kwarg], new_kwargs[new_kwarg], schema_arg)
            return None

        # Default op: dequantize and perform op
        args = tree_map(maybe_unwrap, args)
        if kwargs is not None:
            kwargs = tree_map(maybe_unwrap, kwargs)
        out = super().__torch_dispatch__(func, types, args, kwargs)
        return out

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        # Do not force the QuantizedTensor type on the returned tensor
        return torch._C._disabled_torch_function_impl(func, types, args, kwargs)

    def contiguous(
        self, memory_format: torch.memory_format = torch.contiguous_format
    ) -> QuantizedTensor:
        # pylint: disable=missing-function-docstring
        raise NotImplementedError(
            f"{self.__class__.__name__} class does not implement contiguous function"
        )

    def get_metadata(self) -> Dict[str, Any]:
        """Get keyword arguments for quantized tensor constructor

        Contains metadata so that the new quantized tensor has the
        same underlying quantized data.

        """
        raise NotImplementedError(
            f"{self.__class__.__name__} class does not implement get_metadata function"
        )

    @classmethod
    def make_like(
        cls,
        tensor: QuantizedTensor,
        *,
        shape: Optional[Iterable[int]] = None,
        dtype: Optional[torch.dtype] = None,
        requires_grad: bool = False,
        data: Optional[torch.Tensor] = None,
    ) -> QuantizedTensor:
        """Create new quantized tensor

        By default, new tensor has the same attributes and underlying
        data.

        """
        if shape is None:
            shape = data.shape if data is not None else tensor.shape
        dtype = dtype if dtype is not None else tensor.dtype
        kwargs = tensor.get_metadata()
        if data is not None:
            kwargs["data"] = data
        return cls(shape=shape, dtype=dtype, requires_grad=requires_grad, **kwargs)

    def to_dtype(self, dtype: torch.dtype) -> QuantizedTensor:
        """Create `QuantizedTensor` with given nominal dtype

        The new tensor has the same underlying data.

        """
        return self.__class__.make_like(self, dtype=dtype)
