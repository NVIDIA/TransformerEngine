# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""
Mathematical functions used to tensor statistics computation.
"""

import math
import torch
import torch.nn.functional as F
import transformer_engine_torch as tex
from transformer_engine.common.recipe import Format


@torch.compile
def _compute_dynamic_range_top(tensor):
    """Computes the log2 of the amax of the tensor"""
    tensor_abs = tensor.abs()
    tensor_abs = tensor_abs[tensor_abs != 0]
    if tensor_abs.numel() == 0:
        return torch.inf
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

    # Map each supported FP8 dtype to its corresponding max forward value.
    dtype_to_max = {
        tex.DType.kFloat8E4M3: Format.E4M3.value.max_fwd,
        tex.DType.kFloat8E5M2: Format.E5M2.value.max_fwd,
    }

    if dtype not in dtype_to_max:
        raise ValueError(
            f"Unsupported FP8 dtype {dtype} passed to compute_fp8_delayed_scaling_overflows_num()."
        )

    fp8_max = dtype_to_max[dtype]
    fp8_min = -fp8_max

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
    "overflows_num": 16,
    "overflows%": 17,
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
    "overflows_num": {"overflows_num"},
    "overflows%": {"overflows_num", "numel"},
}

STATS = {
    "min": (lambda x, aux_dict: torch.min(x), lambda buffers: min(_get(buffers, "min"))),
    "max": (lambda x, aux_dict: torch.max(x), lambda buffers: max(_get(buffers, "max"))),
    "sum": (lambda x, aux_dict: torch.sum(x), lambda buffers: sum(_get(buffers, "sum"))),
    "mean": (
        lambda x, aux_dict: torch.mean(x),
        lambda buffers: sum(_get(buffers, "sum")) / sum(_get(buffers, "numel")),
    ),
    "numel": (
        lambda x, aux_dict: x.numel() if hasattr(x, "numel") else x.get_data_tensors()[0].numel(),
        lambda buffers: sum(_get(buffers, "numel")),
    ),
    "l1_norm": (
        lambda x, aux_dict: torch.norm(x, p=1),
        lambda buffers: sum(_get(buffers, "l1_norm")),
    ),
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
    "cur_amax": (lambda x, aux_dict: x.abs().max(), lambda buffers: max(_get(buffers, "cur_amax"))),
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
        lambda x, aux_dict: compute_fp8_delayed_scaling_overflows_num(
            x, aux_dict["fp8_delayed_scaling"]
        ),
        lambda buffers: sum(_get(buffers, "fp8_delayed_scaling_overflows_num")),
    ),
    "fp8_delayed_scaling_overflows%": (
        lambda x, aux_dict: compute_fp8_delayed_scaling_overflows_num(
            x, aux_dict["fp8_delayed_scaling"]
        )
        / x.numel()
        * 100,
        lambda buffers: 100
        * sum(_get(buffers, "fp8_delayed_scaling_overflows_num"))
        / sum(_get(buffers, "numel")),
    ),
    "overflows_num": (
        lambda x, aux_dict: compute_fp8_delayed_scaling_overflows_num(x, aux_dict[""]),
        lambda buffers: sum(_get(buffers, "overflows_num")),
    ),
    "overflows%": (
        lambda x, aux_dict: compute_fp8_delayed_scaling_overflows_num(x, aux_dict[""])
        / x.numel()
        * 100,
        lambda buffers: 100 * sum(_get(buffers, "overflows_num")) / sum(_get(buffers, "numel")),
    ),
}


def add_underflows_stats(recipe_name: str, columnwise: bool = False):
    """Register *both* underflow stats (num and %) for the given recipe."""
    columnwise_suffix = "_columnwise" if columnwise else ""

    # Stat names
    stat_num = f"{recipe_name}{'_' if recipe_name != '' else ''}underflows_num{columnwise_suffix}"
    stat_pct = f"{recipe_name}{'_' if recipe_name != '' else ''}underflows%{columnwise_suffix}"

    stats_to_num[stat_num] = len(stats_to_num)
    stats_to_num[stat_pct] = len(stats_to_num)

    STATS[stat_num] = (
        lambda x, aux_dict: (
            aux_dict[recipe_name].get_data_tensors(
                rowwise_data=not columnwise, columnwise_data=columnwise
            )
            == 0
        ).sum()
        - (x == 0).sum(),
        lambda buffers, _sn=stat_num: sum(_get(buffers, _sn)),
    )
    STATS[stat_pct] = (
        lambda x, aux_dict: (
            aux_dict[recipe_name].get_data_tensors(
                rowwise_data=not columnwise, columnwise_data=columnwise
            )
            == 0
        ).sum()
        / aux_dict[recipe_name].numel()
        * 100,
        lambda buffers, _sn_num=stat_num: 100
        * sum(_get(buffers, _sn_num))
        / sum(_get(buffers, "numel")),
    )

    DEPENDENCIES[stat_num] = {stat_num}
    DEPENDENCIES[stat_pct] = {stat_num, "numel"}


def add_scale_inv_stats(recipe_name: str, columnwise: bool = False):
    """Register *both* scale-inv min and max stats for a given recipe.

    This replaces the earlier separate helpers and avoids duplicated boilerplate.
    """
    # Determine which attribute holds the scale-inverse tensor.

    def get_scale_inv(quantized_tensor, columnwise):
        if hasattr(quantized_tensor, "_scale_inv"):
            return getattr(quantized_tensor, "_scale_inv")
        if columnwise:
            return getattr(quantized_tensor, "_columnwise_scale_inv")
        return getattr(quantized_tensor, "_rowwise_scale_inv")

    columnwise_suffix = "_columnwise" if columnwise else ""
    # Prepare stat names.
    stat_name_min = (
        f"{recipe_name}{'_' if recipe_name != '' else ''}scale_inv_min{columnwise_suffix}"
    )
    stat_name_max = (
        f"{recipe_name}{'_' if recipe_name != '' else ''}scale_inv_max{columnwise_suffix}"
    )

    # Assign indices in `stats_to_num` (order matters — keep insertion order deterministic).
    stats_to_num[stat_name_min] = len(stats_to_num)
    stats_to_num[stat_name_max] = len(stats_to_num)

    # Capture the attribute name inside lambdas via default args to avoid late binding.
    STATS[stat_name_min] = (
        lambda x, aux_dict, _col=columnwise: get_scale_inv(aux_dict[recipe_name], _col).min(),
        lambda buffers, _sn=stat_name_min: min(_get(buffers, _sn)),
    )
    STATS[stat_name_max] = (
        lambda x, aux_dict, _col=columnwise: get_scale_inv(aux_dict[recipe_name], _col).max(),
        lambda buffers, _sn=stat_name_max: max(_get(buffers, _sn)),
    )

    DEPENDENCIES[stat_name_min] = {stat_name_min}
    DEPENDENCIES[stat_name_max] = {stat_name_max}


def add_mse_stats(recipe_name: str, columnwise: bool = False):
    """Register mse and total_square_error stats for the recipe."""
    columnwise_suffix = "_columnwise" if columnwise else ""

    stat_mse = f"{recipe_name}{'_' if recipe_name != '' else ''}mse{columnwise_suffix}"
    stat_err = (
        f"{recipe_name}{'_' if recipe_name != '' else ''}total_square_error{columnwise_suffix}"
    )

    stats_to_num[stat_mse] = len(stats_to_num)
    stats_to_num[stat_err] = len(stats_to_num)

    STATS[stat_mse] = (
        lambda x, aux_dict: F.mse_loss(x, aux_dict[recipe_name].dequantize(), reduction="mean"),
        lambda buffers, _sn_err=stat_err: torch.sum(_get(buffers, _sn_err))
        / sum(_get(buffers, "numel")),
    )
    STATS[stat_err] = (
        lambda x, aux_dict: F.mse_loss(x, aux_dict[recipe_name].dequantize(), reduction="sum"),
        lambda buffers, _sn_err=stat_err: torch.sum(_get(buffers, _sn_err)),
    )

    DEPENDENCIES[stat_err] = {stat_err}
    DEPENDENCIES[stat_mse] = {stat_mse, stat_err, "numel"}


for _columnwise in [True, False]:
    for _recipe_name in [
        "",  # default recipe
        "fp8_delayed_scaling",
        "mxfp8",
        "fp8_current_scaling",
        "fp8_block_scaling",
    ]:
        add_underflows_stats(_recipe_name, _columnwise)
        add_scale_inv_stats(_recipe_name, _columnwise)
        add_mse_stats(_recipe_name, _columnwise)
