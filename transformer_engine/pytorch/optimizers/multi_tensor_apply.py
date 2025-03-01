# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Multi-tensor apply entry."""
from torch.distributed._tensor import DTensor


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
