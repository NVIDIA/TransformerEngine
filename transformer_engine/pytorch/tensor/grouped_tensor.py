# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Grouped tensor class for handling collections of tensors with different shapes"""
from __future__ import annotations

from typing import List, Optional, Tuple

import torch
from torch.utils._pytree import tree_map

from ..quantized_tensor import QuantizedTensorStorage, Quantizer
from .storage.grouped_tensor_storage import GroupedTensorStorage


def _stride_from_shape(shape: Tuple[int, ...]) -> Tuple[int, ...]:
    """Calculate contiguous stride from shape."""
    if len(shape) == 0:
        return ()
    stride = [1] * len(shape)
    for i in range(len(shape) - 2, -1, -1):
        stride[i] = stride[i + 1] * shape[i + 1]
    return tuple(stride)


class _GroupedIdentityFunc(torch.autograd.Function):
    """Identity autograd function used to create a dummy grad_fn node."""

    @staticmethod
    def forward(ctx, tensor: "GroupedTensor") -> "GroupedTensor":
        # pylint: disable=missing-function-docstring
        ctx.input_dtype = tensor.dtype
        return tensor.detach()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        # pylint: disable=missing-function-docstring
        grad_input = grad_output
        if grad_input.dtype != ctx.input_dtype:
            grad_input = grad_input.to(ctx.input_dtype)
        return grad_input


# For now, conservatively ban 'most' shape manipulating ops.
BANNED_SHAPE_OPS = {
    torch.ops.aten.reshape.default,
    torch.ops.aten._reshape_alias.default,
    torch.ops.aten.flatten.using_ints,
    torch.ops.aten.unflatten.int,
    torch.ops.aten.squeeze.dim,
    torch.ops.aten.squeeze.dims,
    torch.ops.aten.unsqueeze.default,
    torch.ops.aten.transpose.int,
    torch.ops.aten.permute.default,
    torch.ops.aten.movedim.int,
    torch.ops.aten.t.default,
    torch.ops.aten.slice.Tensor,
    torch.ops.aten.narrow.default,
    torch.ops.aten.select.int,
    torch.ops.aten.split.Tensor,
    torch.ops.aten.chunk.default,
    torch.ops.aten.cat.default,
    torch.ops.aten.stack.default,
}


class GroupedTensor(GroupedTensorStorage, torch.Tensor):
    """Tensor wrapper class for grouped tensor storage."""

    def __new__(
        cls,
        shape: Tuple[int, int],
        dtype: torch.dtype,
        *,
        num_tensors: int,
        shapes: Optional[List[Tuple[int, int]]] = None,
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
        if (
            shapes is not None
            and len(shapes) == num_tensors
            and num_tensors > 0
            and all(shapes[0] == s for s in shapes)
        ):
            wrapper_shape = (num_tensors, shapes[0][0], shapes[0][1])
        else:
            wrapper_shape = shape

        device = None
        for maybe_tensor in (
            data,
            columnwise_data,
            scale_inv,
            columnwise_scale_inv,
            amax,
            columnwise_amax,
            scale,
            first_dims,
            last_dims,
            tensor_offsets,
        ):
            if maybe_tensor is not None:
                device = maybe_tensor.device
                break
        if device is None:
            device = torch.device("cuda")

        # Match QuantizedTensor __new__: accept externally-computed stride to
        # avoid Python-side stride computation overhead for C++ construction.
        strides = _stride_from_shape(tuple(wrapper_shape)) if stride is None else tuple(stride)
        instance = torch.Tensor._make_wrapper_subclass(
            cls,
            wrapper_shape,
            strides=strides,
            storage_offset=0,
            dtype=dtype,
            layout=torch.strided,
            requires_grad=requires_grad,
            device=device,
        )
        GroupedTensorStorage._initialize_storage_fields(
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
            with_gemm_swizzled_scales=with_gemm_swizzled_scales,
        )
        return instance

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs=None):
        """Dispatch by dequantizing grouped members, then requantizing writes."""
        if kwargs is None:
            kwargs = {}

        def copy_grouped_storage_metadata(dst: GroupedTensor, src: GroupedTensor) -> None:
            """Shallow-copy grouped-storage metadata onto wrapper outputs."""
            dst.num_tensors = src.num_tensors
            dst.quantizer = src.quantizer
            dst.tensor_shapes = src.tensor_shapes
            dst.fake_dtype = src.fake_dtype
            dst.rowwise_data = src.rowwise_data
            dst.columnwise_data = src.columnwise_data
            dst.scale_inv = src.scale_inv
            dst.columnwise_scale_inv = src.columnwise_scale_inv
            dst.amax = src.amax
            dst.columnwise_amax = src.columnwise_amax
            dst.scale = src.scale
            dst.first_dims = src.first_dims
            dst.last_dims = src.last_dims
            dst.tensor_offsets = src.tensor_offsets
            dst.offsets = src.offsets
            dst.scale_inv_offsets = src.scale_inv_offsets
            dst.columnwise_scale_inv_offsets = src.columnwise_scale_inv_offsets
            dst.logical_shape = src.logical_shape
            dst.quantized_tensors = src.quantized_tensors

        def make_wrapper_like(src: GroupedTensor, requires_grad: bool) -> GroupedTensor:
            """Create a wrapper of the same type and tensor metadata as src."""
            out = torch.Tensor._make_wrapper_subclass(
                type(src),
                tuple(src.shape),
                strides=tuple(src.stride()),
                storage_offset=src.storage_offset(),
                dtype=src.dtype,
                layout=src.layout,
                requires_grad=requires_grad,
                device=src.device,
            )
            copy_grouped_storage_metadata(out, src)
            return out

        # Parameter construction calls detach()/alias-like paths.
        if func in (torch.ops.aten.detach.default, torch.ops.aten.alias.default):
            src = args[0]
            if not isinstance(src, GroupedTensor):
                raise TypeError(f"Expected GroupedTensor, got {type(src).__name__}")
            if func == torch.ops.aten.detach.default:
                return make_wrapper_like(src, requires_grad=False)
            return make_wrapper_like(src, requires_grad=src.requires_grad)

        # Parameter construction may invoke aten.expand on tensor subclasses.
        # Handle this explicitly so grouped parameters can be created safely.
        if func == torch.ops.aten.expand.default:
            src = args[0]
            if not isinstance(src, GroupedTensor):
                raise TypeError(f"Expected GroupedTensor, got {type(src).__name__}")
            expanded_shape = tuple(args[1])
            src_shape = tuple(src.shape)
            if len(expanded_shape) == len(src_shape):
                normalized_shape = tuple(
                    src_shape[i] if dim == -1 else dim for i, dim in enumerate(expanded_shape)
                )
                if normalized_shape == src_shape:
                    return make_wrapper_like(src, requires_grad=src.requires_grad)
            return super().__torch_dispatch__(func, types, args, kwargs)

        # DDP and mcore use expand_as(self) to build a dummy autograd node and
        # access gradient accumulators during parameter hook registration.
        if func == torch.ops.aten.expand_as.default:
            src = args[0]
            other = args[1]
            if not isinstance(src, GroupedTensor):
                raise TypeError(f"Expected GroupedTensor, got {type(src).__name__}")
            if other is src:
                return _GroupedIdentityFunc.apply(src)
            if tuple(other.shape) == tuple(src.shape):
                return make_wrapper_like(src, requires_grad=src.requires_grad)
            return super().__torch_dispatch__(func, types, args, kwargs)

        # Distributed optimizer flattens detached parameters via
        # model_param.detach().view(-1). Support this path explicitly by
        # returning a flat view of grouped backing storage.
        if func in (torch.ops.aten.view.default, torch.ops.aten._unsafe_view.default):
            src = args[0]
            if not isinstance(src, GroupedTensor):
                raise TypeError(f"Expected GroupedTensor, got {type(src).__name__}")
            target_shape = tuple(args[1])
            if target_shape in ((-1,), (src.numel(),)):
                if src.rowwise_data is not None:
                    return src.rowwise_data.view(-1)
                raise RuntimeError(
                    f"{cls.__name__} view(-1) requires rowwise_data to be initialized"
                )
            raise RuntimeError(
                f"{cls.__name__} only supports view(-1) for distributed optimizer flattening"
            )

        # Don't allow reshape/view etc.
        if func in BANNED_SHAPE_OPS:
            raise RuntimeError(f"{cls.__name__} forbids shape-manipulation op: {func} ")

        def grouped_to_stacked_tensor(grouped: GroupedTensor) -> torch.Tensor:
            if not grouped.all_same_shape():
                raise NotImplementedError(
                    "GroupedTensor __torch_dispatch__ currently supports only uniform member shapes"
                )
            grouped_members = grouped.quantized_tensors
            if grouped_members is None:
                grouped_members = grouped.split_into_quantized_tensors()
            dequantized_members = [
                (
                    member.dequantize(dtype=grouped.get_dtype())
                    if isinstance(member, QuantizedTensorStorage)
                    else member
                )
                for member in grouped_members
            ]
            return torch.stack(dequantized_members, dim=0)

        def maybe_unwrap(arg):
            if isinstance(arg, GroupedTensor):
                return grouped_to_stacked_tensor(arg)
            return arg

        def update_grouped_tensor_inplace(grouped: GroupedTensor, updated: torch.Tensor):
            if not grouped.all_same_shape():
                raise NotImplementedError(
                    "GroupedTensor __torch_dispatch__ currently supports only uniform member shapes"
                )
            updated_members = list(updated.unbind(dim=0))
            if grouped.quantizer is None:
                grouped_members = grouped.quantized_tensors
                if grouped_members is None:
                    grouped_members = grouped.split_into_quantized_tensors()
                for dst, src in zip(grouped_members, updated_members):
                    dst.copy_(src)
            else:
                grouped.quantize(updated_members)

        def maybe_update_inplace(arg, new_arg, schema_arg):
            if (
                isinstance(arg, GroupedTensor)
                and isinstance(new_arg, torch.Tensor)
                and hasattr(schema_arg, "alias_info")
                and hasattr(schema_arg.alias_info, "is_write")
                and schema_arg.alias_info.is_write
            ):
                update_grouped_tensor_inplace(arg, new_arg)
            elif isinstance(arg, list) and isinstance(new_arg, list):
                for a, na in zip(arg, new_arg):
                    maybe_update_inplace(a, na, schema_arg)

        # In-place op: dequantize members, perform op, write back into grouped storage.
        if func._schema.is_mutable:
            new_args = tree_map(maybe_unwrap, args)
            new_kwargs = tree_map(maybe_unwrap, kwargs)
            schema_args = func._schema.arguments
            args_len = len(args)
            super().__torch_dispatch__(func, types, new_args, new_kwargs)
            for arg, new_arg, schema_arg in zip(args, new_args, schema_args):
                maybe_update_inplace(arg, new_arg, schema_arg)
            for kwarg, new_kwarg, schema_arg in zip(kwargs, new_kwargs, schema_args[args_len:]):
                if kwarg != new_kwarg or kwarg != schema_arg.name:
                    raise RuntimeError(
                        f"Name of kwarg should match schema, got kwarg={kwarg!r},"
                        f" new_kwarg={new_kwarg!r}, schema_arg.name={schema_arg.name!r}"
                    )
                maybe_update_inplace(kwargs[kwarg], new_kwargs[new_kwarg], schema_arg)
            return None

        # Default op: operate on dequantized stacked tensors.
        new_args = tree_map(maybe_unwrap, args)
        new_kwargs = tree_map(maybe_unwrap, kwargs)
        return super().__torch_dispatch__(func, types, new_args, new_kwargs)

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        # Do not force GroupedTensor on outputs.
        return torch._C._disabled_torch_function_impl(func, types, args, kwargs)

    def expand_as(self, other: torch.Tensor) -> torch.Tensor:
        # pylint: disable=missing-function-docstring
        # Needed during parameter creation/hook registration paths.
        if other is self:
            return _GroupedIdentityFunc.apply(self)
        return super().expand_as(other)
