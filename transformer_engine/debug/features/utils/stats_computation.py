# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""
Mathematical functions used to tensor statistics computation.
"""

import math
import torch
import torch.nn.functional as F
from transformer_engine.common.recipe import Format
import transformer_engine_torch as tex

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

def compute_fp8_delayed_scaling_overflows_num(tensor, quantized_tensor):
    """Computes the overflows of the tensor."""
    scale_inv = quantized_tensor._scale_inv
    dtype = quantized_tensor._fp8_dtype
    if dtype == tex.DType.kFloat8E4M3:
        fp8_max = Format.E4M3.value.max_fwd
    elif dtype == tex.DType.kFloat8E5M2:
        fp8_max = Format.E5M2.value.max_fwd
    fp8_min = - fp8_max
    overflows = (tensor > fp8_max * scale_inv) | (tensor < fp8_min * scale_inv)
    return overflows.sum()


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
    "std": 12,
    "dynamic_range": 13,
    "fp8_delayed_scaling_overflows_num": 14,
    "fp8_delayed_scaling_overflows%": 15,
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
    "std": {"variance", "numel", "sum"},
    "dynamic_range": {"dynamic_range_top", "dynamic_range_bottom"},
    "fp8_delayed_scaling_overflows_num": {"fp8_delayed_scaling_overflows_num"},
    "fp8_delayed_scaling_overflows%": {"fp8_delayed_scaling_overflows_num", "numel"},
}

STATS = {
    "min": (lambda x, aux_dict: torch.min(x), lambda buffers: min(_get(buffers, "min"))),
    "max": (lambda x, aux_dict: torch.max(x), lambda buffers: max(_get(buffers, "max"))),
    "sum": (lambda x, aux_dict: torch.sum(x), lambda buffers: sum(_get(buffers, "sum"))),
    "mean": (lambda x, aux_dict: torch.mean(x), lambda buffers: sum(_get(buffers, "sum")) / sum(_get(buffers, "numel"))),
    "numel": (
        lambda x, aux_dict: x.numel() if hasattr(x, "numel") else x.get_data_tensors()[0].numel(),
        lambda buffers: sum(_get(buffers, "numel")),
    ),
    "l1_norm": (lambda x, aux_dict: torch.norm(x, p=1), lambda buffers: sum(_get(buffers, "l1_norm"))),
    "l2_norm_square": (
        lambda x, aux_dict: torch.sum(x**2),
        lambda buffers: sum(_get(buffers, "l2_norm_square")),
    ),
    "l2_norm": (
        lambda x, aux_dict: torch.norm(x, p=2),
        lambda buffers: math.sqrt(sum(_get(buffers, "l2_norm_square"))),
    ),
    "variance": (
        lambda x, aux_dict: torch.var(x),
        lambda buffers: compute_variance(
            _get(buffers, "variance"), _get(buffers, "numel"), _get(buffers, "sum")
        ),
    ),
    "cur_amax": (lambda x: x.abs().max(), lambda buffers: max(_get(buffers, "cur_amax"))),
    "dynamic_range_top": (
        lambda x, aux_dict: _compute_dynamic_range_top(x),
        lambda buffers: max(_get(buffers, "dynamic_range_top")),
    ),
    "dynamic_range_bottom": (
        lambda x, aux_dict: _compute_dynamic_range_bottom(x),
        lambda buffers: min(_get(buffers, "dynamic_range_bottom")),
    ),
    "std": (
        lambda x, aux_dict: torch.std(x),
        lambda buffers: compute_std(
            _get(buffers, "variance"), _get(buffers, "numel"), _get(buffers, "sum")
        ),
    ),
    "dynamic_range": (
        lambda x, aux_dict: _compute_dynamic_range_top(x) - _compute_dynamic_range_bottom(x),
        lambda buffers: max(_get(buffers, "dynamic_range_top"))
        - min(_get(buffers, "dynamic_range_bottom")),
    ),
    "fp8_delayed_scaling_overflows_num": (
        lambda x, aux_dict: compute_fp8_delayed_scaling_overflows_num(x, aux_dict["fp8_delayed_scaling"]),
        lambda buffers: sum(_get(buffers, "fp8_delayed_scaling_overflows_num")),
    ),
    "fp8_delayed_scaling_overflows%": (
        lambda x, aux_dict: compute_fp8_delayed_scaling_overflows_num(x, aux_dict["fp8_delayed_scaling"]) / x.numel() * 100,
        lambda buffers: 100 * sum(_get(buffers, "fp8_delayed_scaling_overflows_num")) / sum(_get(buffers, "numel")),
    ),
}


def add_underflows_num_stat(recipe_name: str, columnwise: bool = False):
    """Adds the underflows_num stat to the stats dictionary."""
    columnwise_suffix = "_columnwise" if columnwise else ""
    data_tensor_idx = 1 if columnwise else 0

    stats_to_num[f"{recipe_name}_underflows_num" + columnwise_suffix] = len(stats_to_num)
    stats_to_num[f"{recipe_name}_underflows%" + columnwise_suffix] = len(stats_to_num)
    
    STATS[f"{recipe_name}_underflows_num" + columnwise_suffix] = (
        lambda x, aux_dict: (aux_dict[recipe_name].get_data_tensors()[data_tensor_idx] == 0).sum(),
        lambda buffers: sum(_get(buffers, f"{recipe_name}_underflows_num")),
    )
    STATS[f"{recipe_name}_underflows%" + columnwise_suffix] = (
        lambda x, aux_dict: (aux_dict[recipe_name].get_data_tensors()[data_tensor_idx] == 0).sum() / aux_dict[recipe_name].numel() * 100,
        lambda buffers: 100 * sum(_get(buffers, f"{recipe_name}_underflows_num")) / sum(_get(buffers, "numel")),
    )

    DEPENDENCIES[f"{recipe_name}_underflows_num" + columnwise_suffix] = {f"{recipe_name}_underflows_num"}
    DEPENDENCIES[f"{recipe_name}_underflows%" + columnwise_suffix] = {f"{recipe_name}_underflows_num", "numel"}

def add_scale_inv_min_stat(recipe_name: str, columnwise: bool = False):
    """Adds the scale_inv_min stat to the stats dictionary."""
    columnwise_suffix = "_columnwise" if columnwise else ""
    scale_inv_name = "_scale_inv"
    if recipe_name in ["mxfp8", "fp8_block_scaling"]:
        scale_inv_name = "_columnwise_scale_inv" if columnwise else "_rowwise_scale_inv"

    stats_to_num[f"{recipe_name}_scale_inv_min" + columnwise_suffix] = len(stats_to_num)

    STATS[f"{recipe_name}_scale_inv_min" + columnwise_suffix] = (
        lambda x, aux_dict: getattr(aux_dict[recipe_name], scale_inv_name).min(),
        lambda buffers: min(_get(buffers, f"{recipe_name}_scale_inv_min")),
    )
    STATS[f"{recipe_name}_scale_inv_max" + columnwise_suffix] = (
        lambda x, aux_dict: getattr(aux_dict[recipe_name], scale_inv_name).max(),
        lambda buffers: max(_get(buffers, f"{recipe_name}_scale_inv_max")),
    )

    DEPENDENCIES[f"{recipe_name}_scale_inv_min" + columnwise_suffix] = {f"{recipe_name}_scale_inv_min"}
    DEPENDENCIES[f"{recipe_name}_scale_inv_max" + columnwise_suffix] = {f"{recipe_name}_scale_inv_max"}

def add_scale_inv_max_stat(recipe_name: str, columnwise: bool = False):
    """Adds the scale_inv_max stat to the stats dictionary."""
    columnwise_suffix = "_columnwise" if columnwise else ""
    scale_inv_name = "_scale_inv"
    if recipe_name in ["mxfp8", "fp8_block_scaling"]:
        scale_inv_name = "_columnwise_scale_inv" if columnwise else "_rowwise_scale_inv"

    stats_to_num[f"{recipe_name}_scale_inv_max" + columnwise_suffix] = len(stats_to_num)
    
    STATS[f"{recipe_name}_scale_inv_max" + columnwise_suffix] = (
        lambda x, aux_dict: getattr(aux_dict[recipe_name], scale_inv_name).max(),
        lambda buffers: max(_get(buffers, f"{recipe_name}_scale_inv_max")),
    )
    
    DEPENDENCIES[f"{recipe_name}_scale_inv_max" + columnwise_suffix] = {f"{recipe_name}_scale_inv_max"}

def add_mse_stat(recipe_name: str, columnwise: bool = False):
    """Adds the mse stat to the stats dictionary."""
    columnwise_suffix = "_columnwise" if columnwise else ""

    stats_to_num[f"{recipe_name}_mse" + columnwise_suffix] = len(stats_to_num)
    stats_to_num[f"{recipe_name}_total_square_error" + columnwise_suffix] = len(stats_to_num)
    
    STATS[f"{recipe_name}_mse" + columnwise_suffix] = (
        lambda x, aux_dict: F.mse_loss(x, aux_dict[recipe_name].dequantize(), reduction="mean"),
        lambda buffers: torch.sum(_get(buffers, f"{recipe_name}_total_square_error" + columnwise_suffix)) / sum(_get(buffers, "numel")),
    )
    STATS[f"{recipe_name}_total_square_error" + columnwise_suffix] = (
        lambda x, aux_dict: F.mse_loss(x, aux_dict[recipe_name].dequantize(), reduction="sum"),
        lambda buffers: torch.sum(_get(buffers, f"{recipe_name}_total_square_error" + columnwise_suffix)),
    )
    
    DEPENDENCIES[f"{recipe_name}_total_square_error" + columnwise_suffix] = {f"{recipe_name}_total_square_error" + columnwise_suffix}
    DEPENDENCIES[f"{recipe_name}_mse" + columnwise_suffix] = {f"{recipe_name}_mse", f"{recipe_name}_total_square_error" + columnwise_suffix, "numel"}

for columnwise in [True, False]:
    for recipe_name in ["fp8_delayed_scaling", "mxfp8", "fp8_current_scaling", "fp8_block_scaling"]:
        add_underflows_num_stat(recipe_name, columnwise)
        add_scale_inv_min_stat(recipe_name, columnwise)
        add_scale_inv_max_stat(recipe_name, columnwise)
        add_mse_stat(recipe_name, columnwise)
