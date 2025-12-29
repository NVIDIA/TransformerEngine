# Copyright (c) 2025, BAAI. All rights reserved.
#
# See LICENSE for license information.

import torch
from torch.distributed._tensor import DTensor
import flag_gems


def multi_tensor_l2_norm_fl(chunk_size, noop_flag, tensor_lists, per_tensor, *args):
    with flag_gems.use_gems():
        tensors = tensor_lists[0]

        if per_tensor:
            norms = [torch.norm(t.float(), p=2) for t in tensors]
            return norms, None
        else:
            total_norm_sq = sum(torch.sum(t.float() ** 2) for t in tensors)
            total_norm = torch.sqrt(total_norm_sq)
            return total_norm, None


def multi_tensor_scale_fl(chunk_size, noop_flag, tensor_lists, scale):
    with flag_gems.use_gems():
        for src, dst in zip(tensor_lists[0], tensor_lists[1]):
            dst.copy_(src * scale)
