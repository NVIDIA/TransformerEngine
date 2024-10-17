# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Tensor class with FP8 data"""
from __future__ import annotations
from typing import Any, Dict, Optional, Tuple
import warnings

import torch
import transformer_engine_torch as tex

from transformer_engine_torch import DType as TE_DType
from ..constants import TE_DType as torch_to_transformer_engine_dtype
from ..cpp_extensions import (
    cast_from_fp8,
    cast_to_fp8,
    fp8_cast_transpose_fused,
)
from ..fp8 import FP8GlobalStateManager
from ..utils import devices_match
from .quantized_tensor import QuantizedTensor

aten = torch.ops.aten
updated_fp8_params = {}


def _make_fp8_attr_property_funcs(name: str) -> Any:
    """Make accessors for an FP8 attribute

    We store FP8 attributes in a dictionary so we can share them
    between tensors with the same data, e.g. detached tensors. For
    convenience, we also expose them as property attributes. This
    function creates the accessors for property attributes.

    Parameters
    ----------
    name: str
          Key in dictionary of FP8 attributes

    """

    def get_func(self) -> Any:
        return self._fp8_attrs[name]

    def set_func(self, value: Any) -> None:
        self._fp8_attrs[name] = value

    def del_func(self) -> None:
        del self._fp8_attrs[name]

    return {"fget": get_func, "fset": set_func, "fdel": del_func}


class _FromFloat8Func(torch.autograd.Function):
    """Cast from FP8 to other dtype"""

    @staticmethod
    def forward(
        _ctx: torch.autograd.function.FunctionCtx,  # unused
        tensor: Float8Tensor,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        # pylint: disable=missing-function-docstring
        return tensor.dequantize(dtype=dtype)

    @staticmethod
    def backward(
        _ctx: torch.autograd.function.FunctionCtx,  # unused
        grad: torch.Tensor,
    ) -> Tuple[Optional[torch.Tensor], ...]:
        # pylint: disable=missing-function-docstring
        # Assume that we want gradients in full precision
        return grad, None


def post_optimizer_step_fwd_amax_reduction(param: Float8Tensor) -> None:
    """Amax scale and update when there is at least 1 trainable FP8 parameter."""
    param_id = id(param._data)

    if param_id not in FP8GlobalStateManager.fp8_param_to_autocast:
        return

    autocast_key = FP8GlobalStateManager.fp8_param_to_autocast[param_id]

    if autocast_key not in FP8GlobalStateManager.autocast_to_fp8_params:
        return

    if autocast_key in updated_fp8_params:
        updated_fp8_params[autocast_key].add(param_id)
    else:
        updated_fp8_params[autocast_key] = {param_id}

    current_fp8_params_set = FP8GlobalStateManager.autocast_to_fp8_params[autocast_key]
    # All FP8 trainable parameters have been updated.
    if updated_fp8_params[autocast_key] == current_fp8_params_set:
        FP8GlobalStateManager.reduce_and_update_fp8_tensors(forward=True, fp8_weights=True)
        del updated_fp8_params[autocast_key]


class _ToFloat8Func(torch.autograd.Function):
    """Cast to FP8 from other dtype"""

    @staticmethod
    def forward(
        _ctx: torch.autograd.function.FunctionCtx,  # unused
        tensor: torch.Tensor,
        fp8_meta: Optional[Dict[str, Any]] = None,
        fp8_meta_forward: bool = True,
        fp8_meta_index: Optional[int] = None,
        fp8_dtype: TE_DType = TE_DType.kFloat8E4M3,
        scale: Optional[torch.Tensor] = None,
        amax: Optional[torch.Tensor] = None,
        scale_inv: Optional[torch.Tensor] = None,
        with_transpose_cache: bool = False,
    ) -> Float8Tensor:
        # pylint: disable=missing-function-docstring

        # Tensor attributes
        dtype = tensor.dtype
        if dtype not in (torch.float32, torch.bfloat16, torch.float16):
            dtype = torch.float32
        device = tensor.device
        if device.type != "cuda":
            device = torch.device("cuda")

        # FP8 data buffer
        data = torch.empty(tensor.size(), dtype=torch.uint8, device=device)

        # Check scale
        if scale is None and fp8_meta is None:
            scale = torch.full([1], 1, dtype=torch.float32, device=device)
        if scale is not None:
            scale = scale.to(device=device, dtype=torch.float32)

        # Check scale-inverse
        if scale_inv is None:
            scale_inv = torch.empty([1], dtype=torch.float32, device=device)
        elif not devices_match(scale_inv.device, device) or scale_inv.dtype != dtype:
            scale_inv = scale_inv.to(device=device, dtype=torch.float32)

        # Transpose cache
        data_transpose = None
        if with_transpose_cache:
            data_transpose = torch.empty(
                (data.size(-1), data.numel() // data.size(-1)),
                dtype=torch.uint8,
                device=tensor.device,
            )

        # Construct FP8 tensor
        out = Float8Tensor(
            data=data,
            fp8_meta=fp8_meta,
            fp8_meta_forward=fp8_meta_forward,
            fp8_meta_index=fp8_meta_index,
            fp8_dtype=fp8_dtype,
            fp8_scale_inv=scale_inv,
            dtype=dtype,
            data_transpose=data_transpose,
        )

        # Cast to FP8 tensor
        out.quantize_(tensor, scale=scale, amax=amax)

        return out

    @staticmethod
    def backward(
        _ctx: torch.autograd.function.FunctionCtx,  # unused
        grad: torch.Tensor,
    ) -> Tuple[Optional[torch.Tensor], ...]:
        # pylint: disable=missing-function-docstring
        # Assume that we want gradients in full precision
        return grad, None, None, None, None, None, None, None


class _IdentityFunc(torch.autograd.Function):
    """Identity function

    If constructor keyword-arguments are provided, then construct a
    new Float8Tensor using the provided tensor's attributes.

    """

    @staticmethod
    def forward(
        ctx,
        tensor: Float8Tensor,
        init_kwargs: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        # pylint: disable=missing-function-docstring

        # Return input tensor if constructor kwargs are not provided
        ctx.input_dtype = tensor.dtype
        if init_kwargs is None:
            return tensor

        # Construct new tensor if constructor kwargs are provided
        default_kwargs = {
            "data": tensor._data,
            "fp8_meta": tensor._fp8_meta,
            "fp8_meta_forward": tensor._fp8_meta_forward,
            "fp8_meta_index": tensor._fp8_meta_index,
            "fp8_dtype": tensor._fp8_dtype,
            "fp8_scale_inv": tensor._scale_inv,
            "dtype": tensor.dtype,
        }
        for key, val in default_kwargs.items():
            if key not in init_kwargs:
                init_kwargs[key] = val
        return Float8Tensor(**init_kwargs)

    @staticmethod
    def backward(ctx, grad):
        # pylint: disable=missing-function-docstring
        return grad.to(ctx.input_dtype), None


class _ViewFunc(torch.autograd.Function):
    """View function

    View the Float8Tensor using the provided shape.

    """

    @staticmethod
    def forward(
        ctx,
        tensor: torch.Tensor,
        shape: Tuple[int] = None,
    ) -> torch.Tensor:
        # pylint: disable=missing-function-docstring

        # Return input tensor if shape is not provided
        ctx.shape = tensor.shape
        if shape is None:
            return tensor

        # Construct new tensor if shape is provided
        if isinstance(tensor, Float8Tensor):
            return Float8Tensor.make_like(
                tensor,
                data=tensor._data.view(*shape),
            )
        return tensor.view(*shape)

    @staticmethod
    def backward(
        ctx,
        grad: torch.Tensor,
    ) -> Tuple[Optional[torch.Tensor], ...]:
        # pylint: disable=missing-function-docstring

        if isinstance(grad, Float8Tensor):
            dgrad = Float8Tensor.make_like(
                grad,
                data=grad._data.view(ctx.shape),
            )
            return dgrad, None
        return grad.view(ctx.shape), None


class _ReshapeFunc(torch.autograd.Function):
    """Reshape function

    Reshape the Float8Tensor using the provided shape.

    """

    @staticmethod
    def forward(
        ctx,
        tensor: torch.Tensor,
        shape: Tuple[int] = None,
    ) -> torch.Tensor:
        # pylint: disable=missing-function-docstring

        # Return input tensor if shape is not provided
        ctx.shape = tensor.shape
        if shape is None:
            return tensor

        # Construct new tensor if shape is provided
        if isinstance(tensor, Float8Tensor):
            return Float8Tensor.make_like(
                tensor,
                data=tensor._data.reshape(*shape),
            )
        return tensor.reshape(*shape)

    @staticmethod
    def backward(
        ctx,
        grad: torch.Tensor,
    ) -> Tuple[Optional[torch.Tensor], ...]:
        # pylint: disable=missing-function-docstring

        if isinstance(grad, Float8Tensor):
            dgrad = Float8Tensor.make_like(
                grad,
                data=grad._data.reshape(ctx.shape),
            )
            return dgrad, None
        return grad.reshape(ctx.shape), None


class Float8Tensor(QuantizedTensor):
    """Experimental tensor class with FP8 data

    The tensor presents as having a standard, higher-precision dtype,
    but the data itself is (scaled) FP8. For most tensor operations,
    the data will be cast to the nominal dtype before performing the
    operation.

    Parameters
    ----------
    data: torch.Tensor
          Raw FP8 data in a uint8 tensor
    fp8_attrs: dict, optional
               FP8 metadata, primarily managed by Float8Tensor. If
               provided, all other FP8 configuration is ignored.
    fp8_meta: dict, optional
              FP8 metadata object, primarily managed by TE modules.
    fp8_meta_forward: bool, default = `True`
                      Whether to access the FP8 metadata for the
                      forward pass. Ignored if fp8_meta is not
                      provided.
    fp8_meta_index: int, optional
                    Index to access in FP8 meta tensors. Required if
                    fp8_meta is provided and otherwise ignored.
    fp8_dtype: transformer_engine_torch.DType, default = kFloat8E4M3
               FP8 format.
    fp8_scale_inv: torch.Tensor
                   Reciprocal of the scaling factor applied when
                   casting to FP8, i.e. the scaling factor that must
                   be applied when casting from FP8 to higher
                   precision. Can be inferred from fp8_meta if
                   provided.
    dtype: torch.dtype, default = torch.float32
           Nominal tensor datatype.

    """

    _data: torch.Tensor
    _fp8_attrs: Dict[str, Any]
    _fp8_meta: Optional[Dict[str, Any]]
    _fp8_meta_forward: bool
    _fp8_meta_index: Optional[int]
    _fp8_dtype: TE_DType
    _scale_inv: torch.Tensor

    # FP8 transpose cache
    _transpose: Optional[torch.Tensor]
    _transpose_invalid: bool

    def __new__(
        cls,
        *,
        data: torch.Tensor,
        fp8_attrs: Optional[Dict[str, Any]] = None,
        fp8_meta: Optional[Dict[str, Any]] = None,
        fp8_meta_forward: bool = True,
        fp8_meta_index: Optional[int] = None,
        fp8_dtype: TE_DType = TE_DType.kFloat8E4M3,
        fp8_scale_inv: Optional[torch.Tensor] = None,
        dtype: torch.dtype = torch.float32,
        requires_grad: bool = False,
        data_transpose: Optional[torch.Tensor] = None,
    ):

        # Check that data buffer is valid
        if data.element_size() != 1:
            raise ValueError(
                f"Float8Tensor requires data buffer with 8-bit dtype (got dtype={data.dtype})"
            )
        if data.requires_grad:
            raise ValueError("Float8Tensor requires non-differentiable data buffer")
        if not data.is_cuda:
            data = data.cuda()

        # Initialize tensor object
        self = torch.Tensor._make_wrapper_subclass(
            cls,
            data.size(),
            strides=data.stride(),
            storage_offset=data.storage_offset(),
            dtype=dtype,
            layout=data.layout,
            requires_grad=requires_grad,
            device=data.device,
        )
        self._data = data

        # Initialize dict of class attributes
        # Note: We store FP8 attributes in a dictionary so we can
        # share them between tensors with the same data, e.g. detached
        # tensors.
        if fp8_attrs is None:
            self._fp8_attrs = {}
        else:
            self._fp8_attrs = fp8_attrs
            return self

        # FP8 meta tensors
        if fp8_meta is not None and fp8_meta_index is None:
            raise ValueError(
                "To initialize Float8Tensor with FP8 meta tensors, "
                "the FP8 meta tensor index must also be provided"
            )
        self._fp8_meta = fp8_meta
        self._fp8_meta_forward = fp8_meta_forward
        self._fp8_meta_index = fp8_meta_index

        # FP8 dtype
        assert fp8_dtype in (
            TE_DType.kFloat8E4M3,
            TE_DType.kFloat8E5M2,
        ), f"Unsupported fp8_dtype {fp8_dtype}."
        self._fp8_dtype = fp8_dtype

        # FP8 scale-inverse
        if fp8_scale_inv is None and self._fp8_meta is not None:
            fp8_meta_key = FP8GlobalStateManager.get_meta_tensor_key(
                forward=self._fp8_meta_forward,
            )
            fp8_scale_inv = self._fp8_meta[fp8_meta_key].scale_inv[self._fp8_meta_index]
            fp8_scale_inv = fp8_scale_inv.detach().reshape(1).clone()
        if fp8_scale_inv is None:
            raise ValueError(
                "Attempted to initialize Float8Tensor without specifying scale-inverse"
            )
        if fp8_scale_inv.numel() != 1:
            raise ValueError(
                "Attempted to initialize Float8Tensor with invalid scale-inverse tensor"
            )
        if fp8_scale_inv.dim() != 1:
            fp8_scale_inv = fp8_scale_inv.reshape(1)
        if (
            not devices_match(fp8_scale_inv.device, self._data.device)
            or fp8_scale_inv.dtype != torch.float32
        ):
            fp8_scale_inv = fp8_scale_inv.to(
                device=self._data.device,
                dtype=torch.float32,
            )
        self._scale_inv = fp8_scale_inv

        # FP8 transpose cache
        self._transpose = data_transpose
        self._transpose_invalid = self._transpose is None

        return self

    @classmethod
    def make_like(
        cls,
        tensor: Float8Tensor,
        *,
        data: torch.Tensor,
        fp8_attrs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Float8Tensor:
        """Use attributes of a Float8Tensor to create another Float8Tensor

        See constructor for list of keyword arguments.

        """
        default_kwargs = {
            "fp8_meta": tensor._fp8_meta,
            "fp8_meta_forward": tensor._fp8_meta_forward,
            "fp8_meta_index": tensor._fp8_meta_index,
            "fp8_dtype": tensor._fp8_dtype,
            "fp8_scale_inv": tensor._scale_inv,
            "dtype": tensor.dtype,
        }
        for key, val in default_kwargs.items():
            if key not in kwargs:
                kwargs[key] = val
        return Float8Tensor(data=data, fp8_attrs=fp8_attrs, **kwargs)

    def __repr__(self):
        return (
            "Float8Tensor("
            f"fp8_dtype={self._fp8_dtype}, "
            f"scale_inv={self._scale_inv.item()}, "
            f"data={self.from_float8(dtype=self.dtype)}"
            ")"
        )

    def dequantize(self, *, dtype: Optional[torch.dtype] = None) -> torch.Tensor:

        # Convert PyTorch dtype to TE dtype
        if dtype is None:
            dtype = self.dtype
        dtype = torch_to_transformer_engine_dtype[dtype]

        # Make sure FP8 data is in expected format
        data = self._data
        if data.device.type != "cuda":
            data = data.cuda()
        if not data.is_contiguous():
            data = data.contiguous()
        if data.dim() != 2:
            data = data.reshape(1, -1)

        # Cast from FP8
        out = cast_from_fp8(
            data.reshape(1, -1),
            None,  # fp8_meta_tensor
            None,  # fp8_tensor
            self._fp8_dtype,
            dtype,
            scale_inv=self._scale_inv,
        )

        # Make sure output is in expected format
        if out.size() != self.size():
            out = out.reshape(self.size())
        return out

    def from_float8(self, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        """
        Construct plain PyTorch tensor from Float8Tensor

        By default the resulting tensor's dtype is the
        Float8Tensor's nominal dtype.
        """
        return _FromFloat8Func.apply(self, dtype)

    def quantize_(
        self,
        tensor: torch.Tensor,
        *,
        scale: Optional[torch.Tensor] = None,
        amax: Optional[torch.Tensor] = None,
        noop_flag: Optional[torch.Tensor] = None,
    ) -> Float8Tensor:
        """Update FP8 data

        Parameters
        ----------
        tensor: torch.Tensor
            Tensor to copy from
        scale: torch.Tensor, optional
            Scaling factor to use for FP8 quantization
        amax: torch.Tensor, optional
            History of maximum absolute values. The first entry will
            be updated with the absmax of `tensor`.
        noop_flag: torch.Tensor, optional
            float32 flag indicating whether to avoid performing update

        """
        src = tensor
        dst = self

        # In-place operations invalidate transpose cache
        self._reset_caches()

        # Special logic if other tensor is Float8Tensor
        if isinstance(src, Float8Tensor):

            # Cast to plain tensor if FP8 dtypes don't match
            if dst._fp8_dtype != src._fp8_dtype:
                return dst.quantize_(src.dequantize())

            # Directly copy FP8 data
            dst._data.copy_(src._data.detach())
            dst._scale_inv.copy_(src._scale_inv.detach())
            if amax is not None or dst._fp8_meta is not None:
                src_amax: torch.Tensor
                if src._fp8_meta is None:
                    src_min, src_max = src.dequantize().aminmax()
                    src_amax = torch.maximum(-src_min, src_max)
                else:
                    fp8_meta_key = FP8GlobalStateManager.get_meta_tensor_key(
                        forward=src._fp8_meta_forward,
                    )
                    fp8_meta_index = src._fp8_meta_index
                    src_amax = src._fp8_meta[fp8_meta_key].amax_history[0, fp8_meta_index]
                dst_amax: torch.Tensor
                if amax is None:
                    fp8_meta_key = FP8GlobalStateManager.get_meta_tensor_key(
                        forward=dst._fp8_meta_forward,
                    )
                    fp8_meta_index = dst._fp8_meta_index
                    dst_amax = dst._fp8_meta[fp8_meta_key].amax_history[0, fp8_meta_index]
                else:
                    dst_amax = amax
                    if dst_amax.dim() > 0:
                        dst_amax = dst_amax[tuple([0] * dst_amax.dim())]
                torch.maximum(src_amax, dst_amax, out=dst_amax)
            if dst._transpose is not None:
                if src._transpose is None:
                    dst.transpose_2d(force_compute=True, fill_cache=True)
                else:
                    dst._transpose.copy_(src._transpose)
                dst._transpose_invalid = False
            return self

        # Convert QuantizedTensor to plain tensor
        if isinstance(src, QuantizedTensor):
            return dst.quantize_(src.dequantize())

        # Make sure input is in expected format
        if src.size() != dst.size():
            src = src.expand(dst.size())
        if not devices_match(src.device, dst.device):
            src = src.to(device=dst.device)
        if src.dtype not in (torch.float32, torch.bfloat16, torch.float16):
            src = src.float()
        if not src.is_contiguous():
            src = src.contiguous()

        # Make sure FP8 scaling factors are in expected format
        if scale is not None:
            if not devices_match(scale.device, dst.device) or scale.dtype != torch.float32:
                scale = scale.to(device=dst.device, dtype=torch.float32)
        if amax is not None:
            while amax.dim() < 2:
                amax = amax.unsqueeze(0)
            if not devices_match(amax.device, dst.device):
                raise ValueError(
                    f"Invalid device for amax (expected {dst.device}, found {amax.device})"
                )
            if amax.dtype != torch.float32:
                raise ValueError(f"Invalid dtype for amax (expected float32, found {amax.type})")

        # Default FP8 scaling factors
        fp8_meta = None
        if dst._fp8_meta is None:
            if scale is None:
                scale = dst._scale_inv.reciprocal()
            if amax is None:
                amax = torch.empty((1, 1), dtype=torch.float32, device=dst.device)
        else:
            fp8_meta_key = FP8GlobalStateManager.get_meta_tensor_key(
                forward=dst._fp8_meta_forward,
            )
            fp8_meta = dst._fp8_meta[fp8_meta_key]

        # Check local data
        if not dst._data.is_contiguous():
            raise RuntimeError("Transformer Engine cast kernels require contiguous data")

        # Perform FP8 cast
        if dst._transpose is None:
            dst_data = dst._data
            if src.dim() != 2:
                src = src.reshape(1, -1)
                dst_data = dst_data.reshape(1, -1)
            cast_to_fp8(
                src,
                fp8_meta,
                dst._fp8_meta_index,
                dst._fp8_dtype,
                out=dst_data,
                scale=scale,
                amax=amax,
                scale_inv=dst._scale_inv,
            )
        else:
            fp8_cast_transpose_fused(
                src.reshape(-1, src.size(-1)),
                fp8_meta,
                dst._fp8_meta_index,
                dst._fp8_dtype,
                cast_out=dst._data,
                transpose_out=dst._transpose,
                scale=scale,
                amax=amax,
                scale_inv=dst._scale_inv,
                noop_flag=noop_flag,
            )
            dst._transpose_invalid = False

        # Callback hook to perform amax reduction after optimizer step
        post_optimizer_step_fwd_amax_reduction(self)

        return self

    @classmethod
    def to_float8(
        cls,
        tensor: torch.Tensor,
        *,
        fp8_meta: Optional[Dict[str, Any]] = None,
        fp8_meta_forward: bool = True,
        fp8_meta_index: Optional[int] = None,
        fp8_dtype: TE_DType = TE_DType.kFloat8E4M3,
        scale: Optional[torch.Tensor] = None,
        amax: Optional[torch.Tensor] = None,
        scale_inv: Optional[torch.Tensor] = None,
        with_transpose_cache: bool = False,
    ):
        """Construct Float8Tensor from plain PyTorch tensor"""
        return _ToFloat8Func.apply(
            tensor,
            fp8_meta,
            fp8_meta_forward,
            fp8_meta_index,
            fp8_dtype,
            scale,
            amax,
            scale_inv,
            with_transpose_cache,
        )

    def detach(self) -> Float8Tensor:
        # pylint: disable=missing-function-docstring
        return Float8Tensor.make_like(
            self,
            data=self._data,
            fp8_attrs=self._fp8_attrs,
        )

    def clone(self) -> Float8Tensor:
        # pylint: disable=missing-function-docstring
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
        *,
        memory_format: torch.memory_format = torch.contiguous_format,
    ) -> Float8Tensor:
        """Returns tensor with data in provided memory format

        Returns `self` if data is already in correct memory format.

        """
        if self._data.is_contiguous(memory_format=memory_format):
            return self
        return _IdentityFunc.apply(
            self,
            {"data": self._data.detach().contiguous(memory_format=memory_format)},
        )

    def transpose_2d(
        self,
        *,
        force_compute: bool = False,
        fill_cache: bool = False,
        noop_flag: Optional[torch.Tensor] = None,
        cache: Optional[bool] = None,
    ) -> torch.Tensor:
        """
        2D transpose with caching support.

        Parameters
        ----------
        force_compute: bool, default = `False`
                       Force computation of transpose. Otherwise use
                       cached values, if possible.
        fill_cache: bool, default = `False`
                    Cache output tensor for future function calls.
        noop_flag: torch.Tensor, optional
                   float32 flag indicating whether to avoid updating
                   cached values, if possible.
        cache: bool, deprecated

        """

        # Handle deprecated cache kwarg
        if cache is not None:
            msg = (
                "cache kwarg for Float8Tensor.transpose_2d is deprecated, "
                "please use force_compute and fill_cache instead"
            )
            warnings.warn(msg, DeprecationWarning)
            if cache:
                force_compute = False
                fill_cache = True
            else:
                force_compute = True
                fill_cache = False

        # Need to compute transpose if cache is invalid
        need_compute = (
            force_compute
            or (self._transpose is None)
            or self._transpose_invalid
            or (noop_flag is not None)
        )

        # Return cached transpose if possible
        if not need_compute:
            assert self._transpose is not None
            return self._transpose

        # Allocate output if needed
        data = self._data.contiguous().reshape(-1, self.size(-1))
        out: Optional[torch.Tensor] = self._transpose
        if out is None:
            out = torch.empty(
                (data.size(1), data.size(0)),
                dtype=torch.uint8,
                device=data.device,
            )
            noop_flag = None
        else:
            self._transpose_invalid = False

        # Apply transpose kernel
        fp8_dtype = self._fp8_dtype
        if noop_flag is None:
            tex.fp8_transpose_noalloc(data, out, fp8_dtype)
        else:
            noop_flag = noop_flag.to(dtype=torch.float32, device=data.device)
            tex.fp8_transpose_noalloc_noop(data, out, noop_flag, fp8_dtype)

        # Fill cache if needed
        if fill_cache:
            self._transpose = out
            self._transpose_invalid = False

        return out

    @torch.no_grad()
    def cast_transpose_(
        self,
        tensor: torch.Tensor,
        noop_flag: Optional[torch.Tensor] = None,
    ) -> None:
        """Cast from tensor and populate transpose cache

        Tensor is reshaped as a 2D matrix.

        Parameters
        ----------
        tensor: torch.Tensor
                Tensor to copy from. Must have same dimensions as
                destination tensor.
        noop_flag: torch.Tensor, optional
                   float32 flag indicating whether to avoid updating
                   destination tensor.

        """
        if self._transpose is None:
            self._transpose = torch.empty(
                (self.size(-1), self.numel() // self.size(-1)),
                dtype=torch.uint8,
                device=self.device,
            )
        self.quantize_(tensor, noop_flag=noop_flag)

    @torch.no_grad()
    def reset_fp8_meta_scale_inv(self) -> None:
        """Replace FP8 meta tensor scale-inverse with cached value

        The FP8 meta tensor scale_inv entry corresponding to this
        tensor is replaced with the scale_inv value used to construct
        the tensor.

        """
        assert self._fp8_meta is not None, "FP8 meta tensors not found."
        fp8_meta_key = FP8GlobalStateManager.get_meta_tensor_key(
            forward=self._fp8_meta_forward,
        )
        self._fp8_meta[fp8_meta_key].scale_inv[self._fp8_meta_index].copy_(self._scale_inv[0])

    def to_dtype(self, dtype: torch.dtype) -> Float8Tensor:
        """Create `Float8Tensor` with given nominal dtype

        The new tensor has the same underlying FP8 data.

        """
        return Float8Tensor.make_like(
            self,
            data=self._data,
            fp8_attrs=self._fp8_attrs,
            dtype=dtype,
        )

    def _reset_caches(self) -> None:
        """
        Set transpose cache as invalid.
        Should be called after any in-place operation.
        """
        self._transpose_invalid = True

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs=None):

        # Slice op
        if func == aten.slice.Tensor:
            tensor = args[0]
            data = tensor._data
            data_slice = data.__torch_dispatch__(
                func,
                types,
                [data] + list(args[1:]),
                kwargs,
            )
            return Float8Tensor.make_like(tensor, data=data_slice)

        # View op
        if func == aten.view.default:
            tensor = args[0]
            data = tensor._data
            data_view = data.__torch_dispatch__(
                func,
                types,
                [data] + list(args[1:]),
                kwargs,
            )
            return Float8Tensor.make_like(tensor, data=data_view)

        # Default case
        return super().__torch_dispatch__(func, types, args, kwargs)

    @classmethod
    def _make_in_reduce_ex(
        cls,
        data: torch.Tensor,
        fp8_dtype: TE_DType,
        fp8_scale_inv: torch.Tensor,
        dtype: torch.dtype,
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
        )

    def __reduce_ex__(self, protocol: int) -> tuple:
        """Custom pickling to remove references to FP8 metadata objects"""
        return (
            Float8Tensor._make_in_reduce_ex,
            (self._data, self._fp8_dtype, self._scale_inv, self.dtype),
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

        # Check whether grad is required
        if self.requires_grad != tensor.requires_grad:
            self.requires_grad_(requires_grad=tensor.requires_grad)

        # Just copy FP8 data if other tensor is Float8Tensor
        if isinstance(tensor, Float8Tensor):
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
            self._data = tensor._data
            self._fp8_attrs = tensor._fp8_attrs
            return

        # Reallocate FP8 data if needed
        if (
            self.size() != tensor.size()
            or self.stride() != tensor.stride()
            or self.dtype != tensor.dtype
            or self.layout != tensor.layout
            or not devices_match(self.device, new_device)
        ):
            self._data = torch.empty_like(
                tensor,
                dtype=torch.uint8,
                device=new_device,
            )
            dummy_tensor = torch.Tensor._make_wrapper_subclass(
                Float8Tensor,
                self._data.size(),
                strides=self._data.stride(),
                storage_offset=self._data.storage_offset(),
                dtype=tensor.dtype,
                layout=self._data.layout,
                requires_grad=tensor.requires_grad,
                device=self._data.device,
            )
            # pylint: disable=unnecessary-dunder-call
            super(Float8Tensor, type(self)).data.__set__(self, dummy_tensor)
            if self._transpose is not None:
                self._transpose = torch.empty(
                    (self._data.size(-1), self._data.numel() // self._data.size(-1)),
                    dtype=torch.uint8,
                    device=self.device,
                )
            self._transpose_invalid = True

        # Copy values from other tensor
        self.quantize_(tensor)

    # Cast to FP8 when setting Float8Tensor.data
    data = property(_get_data, _set_data)

    # Accessors for objects in self._fp8_attrs
    # Note: We store FP8 attributes in a dictionary so we can share
    # them between tensors with the same data, e.g. detached tensors.
    # For convenience, we also expose them as property attributes.
    _fp8_meta = property(**_make_fp8_attr_property_funcs("fp8_meta"))
    _fp8_meta_forward = property(**_make_fp8_attr_property_funcs("fp8_meta_forward"))
    _fp8_meta_index = property(**_make_fp8_attr_property_funcs("fp8_meta_index"))
    _fp8_dtype = property(**_make_fp8_attr_property_funcs("dtype"))
    _transpose = property(**_make_fp8_attr_property_funcs("transpose"))
    _transpose_invalid = property(**_make_fp8_attr_property_funcs("transpose_invalid"))
    _scale_inv = property(**_make_fp8_attr_property_funcs("scale_inv"))
