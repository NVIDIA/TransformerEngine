# Copyright (c) 2025, BAAI. All rights reserved.
#
# See LICENSE for license information.

import torch
from torch.distributed._tensor import DTensor


def multi_tensor_l2_norm_fl(chunk_size, noop_flag, tensor_lists, per_tensor, *args):
    """
    Computes l2 norm for a list of contiguous tensors
    works as a drop-in replacement for amp_C.multi_tensor_l2norm
    """
    l2 = [[(torch.norm(tensor)) for tensor in tensor_list] for tensor_list in tensor_lists]
    l2_reduced = torch.norm(torch.tensor(l2))
    l2_cuda = torch.tensor([float(l2_reduced)], dtype=torch.float, device="cuda")
    return l2_cuda, None


def multi_tensor_scale_fl(chunk_size, noop_flag, tensor_lists, scale):
    """Works as a drop-in replacement for amp_C.multi_tensor_scale."""
    for src, dst in zip(tensor_lists[0], tensor_lists[1]):
        dst.copy_(src * scale)
