# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Deferred row concatenation for tensors consumed inside a custom op."""

import math
from typing import List, Sequence, Tuple, Union

import torch

from .tensor_spec import TensorSpec


def concat_shape(parts: Sequence[Union[torch.Tensor, TensorSpec]]) -> Tuple[int, ...]:
    """Shape after row concatenation."""
    return (sum(t.shape[0] for t in parts), *parts[0].shape[1:])


def concat_to_spec(parts: Sequence[Union[torch.Tensor, TensorSpec]]) -> TensorSpec:
    """Describe the concatenation without accessing its storage."""
    first = parts[0]
    dtype = first.dtype
    for tensor in parts[1:]:
        dtype = torch.promote_types(dtype, tensor.dtype)
    return TensorSpec(
        shape=concat_shape(parts),
        dtype=dtype,
        device=first.device,
        requires_grad=any(t.requires_grad for t in parts),
    )


def materialize_concat(value: List[torch.Tensor]) -> torch.Tensor:
    """Build a checked view, or copy parts whose storage is not adjacent."""
    if len(value) == 1:
        return value[0]
    first = value[0]
    if first.shape[0] == 0:
        return torch.cat(value)
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
    return first.as_strided(shape, (math.prod(shape[1:]), *first.stride()[1:]))
