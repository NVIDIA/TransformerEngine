# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""
Mathematical functions used to tensor statistics computation.
"""

import math
import torch

MAX_FP8_VALUE_INT8 = 126


@torch.compile
def _compute_dynamic_range_top(tensor):
    """Computes the log2 of the amax of the tensor"""
    tensor_abs = tensor.abs()
    tensor_abs = tensor_abs[tensor_abs != 0]
    amax = tensor_abs.max().float()
    if not amax.all():
        amax = torch.tensor(1, device=tensor.device).to(torch.float)
    return torch.log2(amax)


def _compute_dynamic_range_bottom(tensor):
    """Computes the log2 of the amin of the tensor"""
    tensor_abs = tensor.abs()
    tensor_abs = tensor_abs[tensor_abs != 0]
    if tensor_abs.any():
        amin = tensor_abs.min().float()
    else:
        amin = torch.tensor(1, device=tensor.device).to(torch.float)
    return torch.log2(amin)


def compute_variance(variances, numels, sums):
    """Welford algorithm is used for numerically stable distributed variance computation."""
    mean = torch.sum(sums) / torch.sum(numels)
    means = sums / numels
    var = torch.sum(numels * (variances - torch.pow((means - mean), 2))) / torch.sum(numels)
    return var


def compute_std(variances, numels, sums):
    """Computates standard deviation."""
    return torch.sqrt(compute_variance(variances, numels, sums))


# buffers is tensor of shape [nr_buffers, nr_stats]
def _get(buffers, stat_name):
    stat_nr = stats_to_num[stat_name]
    return buffers[:, stat_nr]


stats_to_num = {
    "min": 0,
    "max": 1,
    "sum": 2,
    "mean": 3,
    "numel": 4,
    "l1_norm": 5,
    "l2_norm_square": 6,
    "l2_norm": 7,
    "variance": 8,
    "cur_amax": 9,
    "dynamic_range_top": 10,
    "dynamic_range_bottom": 11,
    "underflows_num": 12,
    "std": 13,
    "dynamic_range": 14,
    "underflows%": 15,
}

DEPENDENCIES = {
    "min": {"min"},
    "max": {"max"},
    "sum": {"sum"},
    "mean": {"sum", "numel"},
    "numel": {"numel"},
    "l1_norm": {"l1_norm"},
    "l2_norm_square": {"l2_norm_square", "numel"},
    "l2_norm": {"l2_norm_square"},
    "variance": {"variance", "numel", "sum"},
    "cur_amax": {"cur_amax"},
    "dynamic_range_top": {"dynamic_range_top"},
    "dynamic_range_bottom": {"dynamic_range_bottom"},
    "underflows_num": {"underflows_num"},
    "std": {"variance", "numel", "sum"},
    "dynamic_range": {"dynamic_range_top", "dynamic_range_bottom"},
    "underflows%": {"underflows_num", "numel"},
}

STATS = {
    "min": (torch.min, lambda buffers: min(_get(buffers, "min"))),
    "max": (torch.max, lambda buffers: max(_get(buffers, "max"))),
    "sum": (torch.sum, lambda buffers: sum(_get(buffers, "sum"))),
    "mean": (torch.mean, lambda buffers: sum(_get(buffers, "sum")) / sum(_get(buffers, "numel"))),
    "numel": (
        lambda x: x.numel() if hasattr(x, "numel") else x.get_data_tensors()[0].numel(),
        lambda buffers: sum(_get(buffers, "numel")),
    ),
    "l1_norm": (lambda x: torch.norm(x, p=1), lambda buffers: sum(_get(buffers, "l1_norm"))),
    "l2_norm_square": (
        lambda x: torch.sum(x**2),
        lambda buffers: sum(_get(buffers, "l2_norm_square")),
    ),
    "l2_norm": (
        lambda x: torch.norm(x, p=2),
        lambda buffers: math.sqrt(sum(_get(buffers, "l2_norm_square"))),
    ),
    "variance": (
        torch.var,
        lambda buffers: compute_variance(
            _get(buffers, "variance"), _get(buffers, "numel"), _get(buffers, "sum")
        ),
    ),
    "cur_amax": (lambda x: x.abs().max(), lambda buffers: max(_get(buffers, "cur_amax"))),
    "dynamic_range_top": (
        _compute_dynamic_range_top,
        lambda buffers: max(_get(buffers, "dynamic_range_top")),
    ),
    "dynamic_range_bottom": (
        _compute_dynamic_range_bottom,
        lambda buffers: min(_get(buffers, "dynamic_range_bottom")),
    ),
    "underflows_num": (
        lambda x: (x._data == 0).sum(),
        lambda buffers: sum(_get(buffers, "underflows_num")),
    ),
    "std": (
        torch.std,
        lambda buffers: compute_std(
            _get(buffers, "variance"), _get(buffers, "numel"), _get(buffers, "sum")
        ),
    ),
    "dynamic_range": (
        lambda x: _compute_dynamic_range_top(x) - _compute_dynamic_range_bottom(x),
        lambda buffers: max(_get(buffers, "dynamic_range_top"))
        - min(_get(buffers, "dynamic_range_bottom")),
    ),
    "underflows%": (
        lambda x: (x.get_data_tensors()[0] == 0).sum() / x.get_data_tensors()[0].numel() * 100,
        lambda buffers: 100 * sum(_get(buffers, "underflows_num")) / sum(_get(buffers, "numel")),
    ),
}
