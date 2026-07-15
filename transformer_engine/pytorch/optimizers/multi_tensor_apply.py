# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Multi-tensor apply entry."""
from torch.distributed._tensor import DTensor


def filter_empty_tensor_lists(tensor_lists):
    """Remove aligned zero-sized tensor slots and return whether any slots remain."""
    if any(len(tensors) != len(tensor_lists[0]) for tensors in tensor_lists):
        raise RuntimeError("Expected aligned multi-tensor lists.")

    keep_slot = [tensor.numel() > 0 for tensor in tensor_lists[0]]
    for i, tensors in enumerate(tensor_lists):
        tensor_lists[i] = [tensor for tensor, keep in zip(tensors, keep_slot) if keep]

    return bool(tensor_lists[0])


class MultiTensorApply:  # pylint: disable=too-few-public-methods
    """Multi-tensor apply entry."""

    def __init__(self, chunk_size):
        self.chunk_size = chunk_size

    def __call__(self, op, noop_flag_buffer, tensor_lists, *args):
        for i, ts in enumerate(tensor_lists):
            for j, t in enumerate(ts):
                if isinstance(t, DTensor):
                    tensor_lists[i][j] = t._local_tensor

        return op(self.chunk_size, noop_flag_buffer, tensor_lists, *args)


multi_tensor_applier = MultiTensorApply(2048 * 32)
