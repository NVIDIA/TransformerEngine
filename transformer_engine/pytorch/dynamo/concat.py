# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Deferred row concatenation for tensors consumed inside a custom op."""

from typing import List, Tuple, TypeVar, Union

import torch
from torch._prims_common import make_contiguous_strides_for

from .tensor_spec import TensorSpec
from ..quantized_tensor import (
    QuantizedTensor,
    QuantizedTensorStorage,
    restore_from_func_ctx as restore_tensor_ctx,
)

_TensorT = TypeVar("_TensorT")
ConcatInput = Union[_TensorT, List[torch.Tensor]]


def validate_concat(parts: List[torch.Tensor]) -> None:
    """Require a nonempty list of non-quantized tensors."""
    if not parts:
        raise ValueError("Attempted to concatenate 0 tensors")
    if any(
        not isinstance(part, torch.Tensor)
        or isinstance(part, (QuantizedTensor, QuantizedTensorStorage))
        for part in parts
    ):
        raise TypeError("Deferred concatenation only supports non-quantized tensors")


def concat_shape(value: ConcatInput[torch.Tensor]) -> Tuple[int, ...]:
    """Shape after row concatenation."""
    if isinstance(value, list):
        return (sum(t.shape[0] for t in value), *value[0].shape[1:])
    return value.shape


def concat_requires_grad(value: ConcatInput[torch.Tensor]) -> bool:
    """Whether any input requires gradients."""
    if isinstance(value, list):
        return any(t.requires_grad for t in value)
    return value.requires_grad


def concat_to_spec(parts: List[torch.Tensor]) -> TensorSpec:
    """Describe the concatenation without accessing its storage."""
    validate_concat(parts)
    first = parts[0]
    dtype = first.dtype
    for tensor in parts[1:]:
        dtype = torch.promote_types(dtype, tensor.dtype)
    return TensorSpec(
        shape=concat_shape(parts),
        dtype=dtype,
        device=first.device,
        requires_grad=concat_requires_grad(parts),
    )


def materialize_concat(value: ConcatInput[torch.Tensor]) -> torch.Tensor:
    """Build a checked view, or copy parts whose storage is not adjacent."""
    if not isinstance(value, list):
        return value
    validate_concat(value)
    if len(value) == 1:
        return value[0]
    first = value[0]
    storage = first.untyped_storage()
    offset = first.storage_offset()
    for tensor in value:
        if (
            tensor.dtype != first.dtype
            or tensor.device != first.device
            or tensor.is_neg()
            or tensor.is_conj()
        ):
            return torch.cat(value)
        if (
            tensor.shape[1:] != first.shape[1:]
            or not tensor.is_contiguous()
            or tensor.untyped_storage().data_ptr() != storage.data_ptr()
            or tensor.storage_offset() != offset
        ):
            return torch.cat(value)
        offset += tensor.numel()
    if offset * first.element_size() > storage.nbytes():
        return torch.cat(value)
    shape = concat_shape(value)
    return first.as_strided(shape, make_contiguous_strides_for(shape))


def restore_from_func_ctx(ctx):
    """Restore original split parameters saved by the custom-op framework."""
    tensors = restore_tensor_ctx(ctx)
    lengths = getattr(ctx, "concatenated_saved_lengths", None)
    if lengths is None:
        return tensors
    restored = []
    offset = 0
    for length in lengths:
        if length is None:
            restored.append(tensors[offset])
            offset += 1
        else:
            restored.append(list(tensors[offset : offset + length]))
            offset += length
    ctx.concatenated_saved_lengths = None
    return restored
