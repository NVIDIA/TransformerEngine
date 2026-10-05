# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Deferred row concatenation for tensors consumed inside a custom op."""

from dataclasses import dataclass
import math
from typing import List

import torch
from torch._prims_common import make_contiguous_strides_for

from ..quantized_tensor import restore_from_func_ctx as restore_tensor_ctx


@dataclass
class ConcatenatedTensor:
    """Keep every parameter visible to autograd until the opaque consumer runs."""

    tensors: List[torch.Tensor]

    @property
    def shape(self):
        """Combined shape along dimension zero."""
        return (sum(t.shape[0] for t in self.tensors), *self.tensors[0].shape[1:])

    @property
    def dtype(self):
        """Parameter dtype."""
        dtype = self.tensors[0].dtype
        for tensor in self.tensors[1:]:
            dtype = torch.promote_types(dtype, tensor.dtype)
        return dtype

    @property
    def device(self):
        """Parameter device."""
        return self.tensors[0].device

    @property
    def requires_grad(self):
        """Whether any part requires a gradient."""
        return any(t.requires_grad for t in self.tensors)

    def numel(self):
        """Combined element count."""
        return math.prod(self.shape)

    def materialize(self):
        """Build a checked view, or copy parts whose storage is not adjacent."""
        first = self.tensors[0]
        storage = first.untyped_storage()
        offset = first.storage_offset()
        for tensor in self.tensors:
            if tensor.dtype != first.dtype or tensor.device != first.device:
                return torch.cat(self.tensors)
            if (
                tensor.shape[1:] != first.shape[1:]
                or not tensor.is_contiguous()
                or tensor.is_neg()
                or tensor.is_conj()
                or tensor.untyped_storage().data_ptr() != storage.data_ptr()
                or tensor.storage_offset() != offset
            ):
                return torch.cat(self.tensors)
            offset += tensor.numel()
        if offset * first.element_size() > storage.nbytes():
            return torch.cat(self.tensors)
        return first.as_strided(self.shape, make_contiguous_strides_for(self.shape))


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
            restored.append(ConcatenatedTensor(list(tensors[offset : offset + length])))
            offset += length
    ctx.concatenated_saved_lengths = None
    return restored
