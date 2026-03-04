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


# For now, conservatively ban all shape manipulating ops.
BANNED_SHAPE_OPS = {
    torch.ops.aten.view.default,
    torch.ops.aten._unsafe_view.default,
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
    torch.ops.aten.expand.default,
    torch.ops.aten.expand_as.default,
    torch.ops.aten.cat.default,
    torch.ops.aten.stack.default,
}


class GroupedTensor(GroupedTensorStorage, torch.Tensor):
    """Tensor wrapper class for grouped tensor storage."""

    def __new__(
        cls,
        shape: Tuple[int, int],
        dtype: torch.dtype,
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
    ):
        del quantizer
        del offsets
        del scale_inv_offsets
        del columnwise_scale_inv_offsets

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

        strides = [1] * len(wrapper_shape)
        for i in range(len(wrapper_shape) - 2, -1, -1):
            strides[i] = strides[i + 1] * wrapper_shape[i + 1]
        return torch.Tensor._make_wrapper_subclass(
            cls,
            wrapper_shape,
            strides=tuple(strides),
            storage_offset=0,
            dtype=dtype,
            layout=torch.strided,
            requires_grad=False,
            device=device,
        )

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs=None):
        """Dispatch by dequantizing grouped members, then requantizing writes."""
        if kwargs is None:
            kwargs = {}

        # Parameter construction calls detach()/alias-like paths.
        if func in (torch.ops.aten.detach.default, torch.ops.aten.alias.default):
            return args[0]

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
                assert kwarg == new_kwarg == schema_arg.name, "name of kwarg should match schema"
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
