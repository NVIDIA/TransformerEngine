# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Deferred row concatenation for tensors consumed inside a custom op."""

from dataclasses import dataclass
from typing import Tuple, TypeVar, Union

import torch
from torch._prims_common import make_contiguous_strides_for

from .tensor_spec import TensorSpec
from ..quantized_tensor import (
    QuantizedTensor,
    QuantizedTensorStorage,
    restore_from_func_ctx as restore_tensor_ctx,
)


@dataclass(frozen=True, slots=True)
class ParameterParts:
    """Keep every parameter visible to autograd until the opaque consumer runs."""

    parts: Tuple[torch.Tensor, ...]

    def __post_init__(self):
        object.__setattr__(self, "parts", tuple(self.parts))
        if not self.parts:
            raise ValueError("ParameterParts requires at least one tensor")
        if any(
            not isinstance(part, torch.Tensor)
            or isinstance(part, (QuantizedTensor, QuantizedTensorStorage))
            for part in self.parts
        ):
            raise TypeError("ParameterParts only supports non-quantized tensors")

    @property
    def shape(self) -> Tuple[int, ...]:
        """Shape after row concatenation."""
        return (sum(t.shape[0] for t in self.parts), *self.parts[0].shape[1:])

    @property
    def requires_grad(self) -> bool:
        """Whether any part requires gradients."""
        return any(t.requires_grad for t in self.parts)

    def to_spec(self) -> TensorSpec:
        """Describe the concatenation without accessing its storage."""
        first = self.parts[0]
        dtype = first.dtype
        for tensor in self.parts[1:]:
            dtype = torch.promote_types(dtype, tensor.dtype)
        return TensorSpec(
            shape=self.shape,
            dtype=dtype,
            device=first.device,
            requires_grad=self.requires_grad,
        )

    def materialize(self):
        """Build a checked view, or copy parts whose storage is not adjacent."""
        first = self.parts[0]
        storage = first.untyped_storage()
        offset = first.storage_offset()
        for tensor in self.parts:
            if (
                tensor.dtype != first.dtype
                or tensor.device != first.device
                or tensor.is_neg()
                or tensor.is_conj()
            ):
                return torch.cat(self.parts)
            if (
                tensor.shape[1:] != first.shape[1:]
                or not tensor.is_contiguous()
                or tensor.untyped_storage().data_ptr() != storage.data_ptr()
                or tensor.storage_offset() != offset
            ):
                return torch.cat(self.parts)
            offset += tensor.numel()
        if offset * first.element_size() > storage.nbytes():
            return torch.cat(self.parts)
        shape = self.shape
        return first.as_strided(shape, make_contiguous_strides_for(shape))


_TensorT = TypeVar("_TensorT")
ConcatInput = Union[_TensorT, ParameterParts]


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
            restored.append(ParameterParts(tuple(tensors[offset : offset + length])))
            offset += length
    ctx.concatenated_saved_lengths = None
    return restored
