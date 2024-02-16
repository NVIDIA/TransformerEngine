# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Tensor class with FP8 data"""
from __future__ import annotations
from typing import Any, Dict, Optional

import torch
from torch.utils._pytree import tree_map
import transformer_engine_extensions as tex

from .constants import TE_DType
from .fp8 import FP8GlobalStateManager


aten = torch.ops.aten
c10d = torch.ops.c10d


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
    return dict(fget=get_func, fset=set_func, fdel=del_func)


class _FromFloat8Func(torch.autograd.Function):
    """Cast from FP8 to other dtype"""
    @staticmethod
    def forward(
        ctx,
        tensor: Float8Tensor,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        if dtype is None:
            dtype = tensor.dtype
        data = tensor._data.contiguous().view(1,-1).detach()
        out = tex.cast_from_fp8(
            data,
            tensor._scale_inv,
            tensor._fp8_dtype,
            TE_DType[dtype],
        )
        out = out.view(tensor.size())
        return out

    @staticmethod
    def backward(ctx, grad):
        # Assume that we want gradients in full precision
        return grad, None


class _ToFloat8Func(torch.autograd.Function):
    """Cast to FP8 from other dtype"""
    @staticmethod
    def forward(
        ctx,
        tensor: torch.Tensor,
        fp8_meta: Optional[Dict[str, Any]] = None,
        fp8_meta_forward: bool = True,
        fp8_meta_index: Optional[int] = None,
        fp8_dtype: tex.DType = tex.DType.kFloat8E4M3,
        scale: Optional[torch.Tensor] = None,
        amax: Optional[torch.Tensor] = None,
        scale_inv: Optional[torch.Tensor] = None,
    ):

        # Manually compute scale-inverse if needed
        if scale is not None and scale_inv is None:
            if isinstance(scale, torch.Tensor):
                scale_inv = scale.reciprocal()
            else:
                scale_inv = 1 / scale

        # Extract data from FP8 meta tensors if provided
        if fp8_meta is not None:
            fp8_meta_key = FP8GlobalStateManager.get_meta_tensor_key(
                forward=fp8_meta_forward,
            )
            if fp8_meta_index is None:
                raise ValueError(
                    "To initialize Float8Tensor with FP8 meta tensors, "
                    "the FP8 meta tensor index must also be provided"
                )
            if scale is None:
                scale = fp8_meta[fp8_meta_key].scale[fp8_meta_index]
            if amax is None:
                amax = fp8_meta[fp8_meta_key].amax_history[0][fp8_meta_index]
            if scale_inv is None:
                scale_inv = fp8_meta[fp8_meta_key].scale_inv[fp8_meta_index]
                scale_inv = scale_inv.detach().view(1).clone()

        # Check input tensor
        tensor = tensor.contiguous().cuda().detach()
        if tensor.dtype not in (torch.float32, torch.bfloat16, torch.float16):
            tensor = tensor.float()

        # Check scale
        if not isinstance(scale, torch.Tensor):
            if scale is None:
                scale = 1
            scale = torch.full(
                [1],
                scale,
                dtype=torch.float32,
                device=tensor.device,
            )
        if scale.numel() != 1:
            raise ValueError(
                "Attempted to initialize Float8Tensor with invalid scale tensor"
            )
        scale = scale.to(device=tensor.device, dtype=torch.float32)

        # Check scale-inverse
        if scale_inv is None:
            scale_inv = scale.reciprocal()
        scale_inv = scale_inv.to(device=tensor.device, dtype=torch.float32)

        # Check amax
        if amax is None:
            amax = torch.empty_like(scale)
        if not (amax.numel() == 1 and amax.is_cuda and amax.dtype == torch.float32):
            raise ValueError(
                "Attempted to initialize Float8Tensor with invalid amax tensor"
            )

        # Cast data to FP8
        data = tex.cast_to_fp8(
            tensor.view(1,-1),
            scale,
            amax,
            scale_inv,
            fp8_dtype,
        )
        data = data.view(tensor.size())

        # Construct FP8 tensor
        return Float8Tensor(
            data=data,
            fp8_meta=fp8_meta,
            fp8_meta_forward=fp8_meta_forward,
            fp8_meta_index=fp8_meta_index,
            fp8_dtype=fp8_dtype,
            fp8_scale_inv=scale_inv,
            dtype=tensor.dtype,
        )

    @staticmethod
    def backward(ctx, grad):
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

        # Return input tensor if constructor kwargs are not provided
        ctx.input_dtype = tensor.dtype
        if init_kwargs is None:
            return tensor

        # Construct new tensor if constructor kwargs are provided
        default_kwargs = dict(
            data=tensor._data,
            fp8_meta=tensor._fp8_meta,
            fp8_meta_forward=tensor._fp8_meta_forward,
            fp8_meta_index=tensor._fp8_meta_index,
            fp8_dtype=tensor._fp8_dtype,
            fp8_scale_inv=tensor._scale_inv,
            dtype=tensor.dtype,
        )
        for key, val in default_kwargs.items():
            if key not in init_kwargs:
                init_kwargs[key] = val
        return Float8Tensor(**init_kwargs)

    @staticmethod
    def backward(ctx, grad):
        return grad.to(ctx.input_dtype), None


class Float8Tensor(torch.Tensor):
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
    fp8_dtype: transformer_engine_extensions.DType, tex.DType.kFloat8E4M3
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

    def __new__(
        cls,
        *,
        data: torch.Tensor,
        fp8_attrs: Optional[Dict[str, Any]] = None,
        fp8_meta: Optional[Dict[str, Any]] = None,
        fp8_meta_forward: bool = True,
        fp8_meta_index: Optional[int] = None,
        fp8_dtype: tex.DType = tex.DType.kFloat8E4M3,
        fp8_scale_inv: Optional[torch.Tensor] = None,
        dtype: torch.dtype = torch.float32,
    ):

        # Check that data buffer is valid
        if data.element_size() != 1:
            raise ValueError(
                "Float8Tensor requires data buffer with 8-bit dtype "
                f"(got dtype={data.dtype})"
            )
        if data.requires_grad:
            raise ValueError(
                "Float8Tensor requires non-differentiable data buffer"
            )
        data = data.cuda()

        # Initialize tensor object
        self = torch.Tensor._make_wrapper_subclass(
            cls,
            data.size(),
            strides=data.stride(),
            storage_offset=data.storage_offset(),
            dtype=dtype,
            layout=data.layout,
            requires_grad=data.requires_grad,
            device=data.device,
        )
        self._data: torch.Tensor = data

        # Initialize dict of class attributes
        # Note: We store FP8 attributes in a dictionary so we can
        # share them between tensors with the same data, e.g. detached
        # tensors.
        self._fp8_attrs: dict = {}
        if fp8_attrs is not None:
            self._fp8_attrs = fp8_attrs
            return self

        # FP8 meta tensors
        if fp8_meta is not None and fp8_meta_index is None:
            raise ValueError(
                "To initialize Float8Tensor with FP8 meta tensors, "
                "the FP8 meta tensor index must also be provided"
            )
        self._fp8_meta: Optional[Dict[str, Any]] = fp8_meta
        self._fp8_meta_forward: bool = fp8_meta_forward
        self._fp8_meta_index: Optional[int] = fp8_meta_index

        # FP8 dtype
        assert (
            fp8_dtype in (tex.DType.kFloat8E4M3, tex.DType.kFloat8E5M2)
        ), f"Unsupported fp8_dtype {fp8_dtype}."
        self._fp8_dtype: tex.DType = fp8_dtype

        # Cached transpose
        self._transpose: Optional[Float8Tensor] = None

        # FP8 scale-inverse
        self._scale_inv: Optional[torch.Tensor] = fp8_scale_inv
        if self._scale_inv is None and self._fp8_meta is not None:
            fp8_meta_key = FP8GlobalStateManager.get_meta_tensor_key(
                forward=self._fp8_meta_forward,
            )
            scale_inv = self._fp8_meta[fp8_meta_key].scale_inv[self._fp8_meta_index]
            self._scale_inv = scale_inv.detach().view(1).clone()
        if self._scale_inv is None:
            raise ValueError(
                "Attempted to initialize Float8Tensor without specifying scale-inverse"
            )
        if not isinstance(self._scale_inv, torch.Tensor):
            self._scale_inv = torch.full(
                [1],
                self._scale_inv,
                dtype=torch.float32,
                device=self._data.device,
            )
        if self._scale_inv.numel() != 1:
            raise ValueError(
                "Attempted to initialize Float8Tensor with invalid scale-inverse tensor"
            )
        self._scale_inv = self._scale_inv.to(
            device=self._data.device,
            dtype=torch.float32,
        )

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
        default_kwargs = dict(
            fp8_meta=tensor._fp8_meta,
            fp8_meta_forward=tensor._fp8_meta_forward,
            fp8_meta_index=tensor._fp8_meta_index,
            fp8_dtype=tensor._fp8_dtype,
            fp8_scale_inv=tensor._scale_inv,
            dtype=tensor.dtype,
        )
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

    def from_float8(self, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        """
        Construct plain PyTorch tensor from Float8Tensor

        By default the resulting tensor's dtype is the
        Float8Tensor's nominal dtype.
        """
        return _FromFloat8Func.apply(self, dtype)

    @classmethod
    def to_float8(
        cls,
        tensor: torch.Tensor,
        *,
        fp8_meta: Optional[Dict[str, Any]] = None,
        fp8_meta_forward: bool = True,
        fp8_meta_index: Optional[int] = None,
        fp8_dtype: tex.DType = tex.DType.kFloat8E4M3,
        scale: Optional[torch.Tensor] = None,
        amax: Optional[torch.Tensor] = None,
        scale_inv: Optional[torch.Tensor] = None,
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
        )

    def float(self) -> torch.Tensor:
        return self.from_float8(dtype=torch.float32)

    def bfloat16(self) -> torch.Tensor:
        return self.from_float8(dtype=torch.bfloat16)

    def half(self) -> torch.Tensor:
        return self.from_float8(dtype=torch.float16)

    def cpu(self) -> torch.Tensor:
        return self.from_float8().cpu()

    def clone(self) -> Float8Tensor:
        return _IdentityFunc.apply(self, {"data": self._data.detach().clone()})

    def expand_as(self, other: torch.Tensor):
        if other is self:
            # Note: expand_as is hackily used to create dummy autograd nodes
            # and access the backward graph (see
            # https://github.com/pytorch/pytorch/blob/238fb660851268f44ff88127887041fea352fe48/torch/nn/parallel/distributed.py#L1026).
            # We equally hackily add a dummy function to handle this
            # case.
            return _IdentityFunc.apply(self)
        return super().expand_as(other)

    def transpose(
        self,
        dim0: int = 0,
        dim1: int = 1,
        *,
        update_cache: str | bool = "reuse_only",
    ) -> torch.Tensor:
        """
        Swap tensor dimensions

        For basic 2D matrix transposes, an optimized transpose kernel
        is applied and a Float8Tensor is returned.

        Parameters
        ----------
        dim0: int, default = 0
              The first dimension to be transposed
        dim1: int, default = 1
              The second dimension to be transposed
        update_cache: str or bool, default = "reuse_only"
                      Memoization behavior. Options are
                      "reuse_only"/`False` (reuse cached value if
                      available, otherwise calculate transpose without
                      caching), "force"/`True` (calculate transpose
                      and cache), "lazy" (reuse cached value if
                      available, otherwise calculate transpose and
                      cache if possible). Caching is only supported
                      for basic 2D transposes and the cache is reset
                      after any in-place operations.

        """

        # Check caching mode
        if not isinstance(update_cache, str):
            update_cache = "force" if update_cache else "reuse_only"
        if update_cache not in ("force", "reuse_only", "lazy"):
            raise ValueError(
                "Supported values for update_cache are "
                '"force" (True), "reuse_only" (False), "lazy" '
                f"(got {update_cache})"
            )

        # Handle non-2D transposes
        if -self.dim() <= dim0 < 0:
            dim0 += self.dim()
        if -self.dim() <= dim1 < 0:
            dim1 += self.dim()
        if self.dim() != 2 or dim0 == dim1:
            if update_cache == "force":
                raise ValueError(
                    "Transpose caching is only supported for basic 2D transposes "
                    f"(ndims={self.dim()}, dim0={dim0}, dim1={dim1})"
                )
            return super().transpose(dim0, dim1)

        # Clear cache if needed
        if update_cache == "force":
            self._transpose = None

        # Compute transpose if needed
        out = self._transpose
        if out is None:
            out = Float8Tensor.make_like(
                self,
                data=tex.fp8_transpose(
                    self._data.contiguous(),
                    self._fp8_dtype,
                ),
            )

        # Update cache if needed
        if update_cache in ("force", "lazy"):
            self._transpose = out
        return out

    @torch.no_grad()
    def reset_fp8_meta_scale_inv(self) -> None:
        """Replace FP8 meta tensor scale-inverse with cached value

        The FP8 meta tensor scale_inv entry corresponding to this
        tensor is replaced with the scale_inv value used to construct
        the tensor.

        """
        if self._fp8_meta is None:
            return
        fp8_meta_key = FP8GlobalStateManager.get_meta_tensor_key(
            forward=self._fp8_meta_forward,
        )
        scale_inv = self._fp8_meta[fp8_meta_key].scale_inv[self._fp8_meta_index]
        scale_inv.view(1).copy_(self._scale_inv.view(1))

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
        """Reset cached values

        Should be called after any in-place operation.

        """
        self._transpose = None

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs=None):

        # In-place copy op
        if func == aten.copy_.default:

            # Check tensors
            dst = args[0]
            src = args[1]
            if not isinstance(dst, torch.Tensor):
                raise RuntimeError(
                    "Attempted to copy into something that isn't a PyTorch tensor"
                )
            if not isinstance(src, torch.Tensor):
                raise RuntimeError(
                    "Attempted to copy from something that isn't a PyTorch tensor"
                )

            # Special handling based on which tensors are FP8
            dst_is_fp8 = isinstance(dst, Float8Tensor)
            src_is_fp8 = isinstance(src, Float8Tensor)
            if dst_is_fp8 and src_is_fp8:

                # Directly copy FP8 data if possible
                if dst._fp8_dtype == src._fp8_dtype:
                    dst._data.copy_(src._data)
                    dst._scale_inv = src._scale_inv.clone()
                    if dst._fp8_meta is not None:
                        if src._fp8_meta is None:
                            src_min, src_max = src.from_float8().aminmax()
                            src_amax = torch.maximum(-src_min, src_max)
                        else:
                            fp8_meta_key = FP8GlobalStateManager.get_meta_tensor_key(
                                forward=src._fp8_meta_forward,
                            )
                            fp8_meta_index = src._fp8_meta_index
                            src_amax = src._fp8_meta[fp8_meta_key].amax_history[0][fp8_meta_index]
                        fp8_meta_key = FP8GlobalStateManager.get_meta_tensor_key(
                            forward=dst._fp8_meta_forward,
                        )
                        fp8_meta_index = dst._fp8_meta_index
                        dst_amax = dst._fp8_meta[fp8_meta_key].amax_history[0][fp8_meta_index]
                        torch.maximum(src_amax, dst_amax, out=dst_amax)
                else:
                    dst.copy_(src.from_float8())

            elif not dst_is_fp8 and src_is_fp8:

                # Cast source tensor to higher precision
                dst.copy_(src.from_float8())

            elif dst_is_fp8 and not src_is_fp8:

                # Make sure input is in expected format
                src = src.expand(dst.size())
                src = src.to(
                    device=dst.device,
                    memory_format=torch.contiguous_format,
                )

                # Update scaling factor if FP8 meta tensors are available
                if dst._fp8_meta is None:
                    scale = dst._scale_inv.reciprocal()
                    amax = torch.empty_like(scale)
                else:
                    fp8_meta_key = FP8GlobalStateManager.get_meta_tensor_key(
                        forward=dst._fp8_meta_forward,
                    )
                    fp8_meta_index = dst._fp8_meta_index
                    scale = dst._fp8_meta[fp8_meta_key].scale[fp8_meta_index]
                    amax = dst._fp8_meta[fp8_meta_key].amax_history[0][fp8_meta_index]
                    dst._scale_inv = scale.detach().view(1).reciprocal()

                # Cast to FP8
                if not dst._data.is_contiguous():
                    raise RuntimeError("Transformer Engine cast kernels require contiguous data")
                tex.cast_to_fp8_noalloc(
                    src.view(1,-1),
                    scale,
                    dst._data.view(1,-1),
                    amax,
                    dst._scale_inv,
                    dst._fp8_dtype,
                )

            else:

                # Invalid case
                raise RuntimeError("Using Float8Tensor copy logic, but no Float8Tensor found")

            # Nothing to return for in-place ops
            if dst_is_fp8:
                dst._reset_caches()
            return None

        # Slice op
        # TODO Consider additional bookkeeping so we invalidate caches # pylint: disable=fixme
        # if these slices are modified in-place
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

        # Detach op
        if func == aten.detach.default:
            # Simply return a new Float8Tensor with the same attrs
            return Float8Tensor.make_like(
                args[0],
                data=args[0]._data,
                fp8_attrs=args[0]._fp8_attrs,
            )

        def maybe_unwrap(t):
            if isinstance(t, Float8Tensor):
                return t.from_float8()
            return t

        def maybe_update_inplace(arg, new_arg, schema_arg):
            """Update values of FP8 tensors

            Keep the same FP8 scaling factors.

            """
            if(
                isinstance(arg, Float8Tensor) and
                isinstance(new_arg, torch.Tensor) and
                hasattr(schema_arg, 'alias_info') and
                hasattr(schema_arg.alias_info, 'is_write') and
                schema_arg.alias_info.is_write
            ):
                arg.copy_(new_arg)
                arg._reset_caches()

        # In-place op
        if func._schema.is_mutable:
            # Cast to higher precision, perform op, and cast values
            # back to original FP8 buffers
            new_args = tree_map(maybe_unwrap, args)
            new_kwargs = tree_map(maybe_unwrap, kwargs)
            schema_args = func._schema.arguments
            args_len = len(args)
            out = super().__torch_dispatch__(func, types, new_args, new_kwargs)
            for arg, new_arg, schema_arg in zip(args, new_args, schema_args):
                maybe_update_inplace(arg, new_arg, schema_arg)
            for kwarg, new_kwarg, schema_arg in zip(kwargs, new_kwargs, schema_args[args_len:]):
                assert kwarg == new_kwarg == schema_arg.name, "name of the kw argument should match"
                maybe_update_inplace(kwargs[kwarg], new_kwargs[new_kwarg], schema_arg)
            return None

        # Default op
        # Note: cast to higher precision and perform op
        args = tree_map(maybe_unwrap, args)
        if kwargs is not None:
            kwargs = tree_map(maybe_unwrap, kwargs)
        out = super().__torch_dispatch__(func, types, args, kwargs)
        return out

    @classmethod
    def _make_in_reduce_ex(
        cls,
        data: torch.Tensor,
        fp8_dtype: tex.DType,
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

    def _set_data(self, tensor: torch.Tensor) -> None:
        """Set tensor data property

        Cast tensor to FP8 and store in FP8 buffer.

        """
        with torch.no_grad():
            self.copy_(tensor)

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
    _scale_inv = property(**_make_fp8_attr_property_funcs("scale_inv"))

    # Do not force the Float8Tensor type on the returned tensor
    __torch_function__ = torch._C._disabled_torch_function_impl
