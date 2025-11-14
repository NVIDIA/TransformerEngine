# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""
Mathematical functions used to tensor statistics computation.
"""

import math
from collections import namedtuple

import torch
import torch.nn.functional as F
import transformer_engine_torch as tex
from transformer_engine.common.recipe import Format


class BlockwiseDynamicRangeStat(
    namedtuple("BlockwiseDynamicRangeStat", ["block_size", "dims", "max_over_orientations"])
):
    """Named tuple representing a blockwise dynamic range statistic configuration."""

    def __str__(self) -> str:
        """Convert to string representation for stat name. Used for logging."""
        suffix = "_max_over_orientations" if self.max_over_orientations else ""
        return f"max_blockwise_dynamic_range_block_size_{self.block_size}_dims_{self.dims}{suffix}"


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


@torch.compile
def _compute_dynamic_range_bottom(tensor):
    """Computes the log2 of the amin of the tensor"""
    tensor_abs = tensor.abs()
    tensor_abs = tensor_abs[tensor_abs != 0]
    if tensor_abs.any():
        amin = tensor_abs.min().float()
    else:
        amin = torch.tensor(1, device=tensor.device).to(torch.float)
    return torch.log2(amin)


def compute_max_blockwise_dynamic_range(tensor, stat_config):
    """
    Computes maximum blockwise dynamic range (log2 max/min_nonzero) within blocks.

    Flattens tensor to 2D and computes maximum dynamic range within blocks. If max_over_orientations
    is True, computes for both rowwise and columnwise orientations and returns the maximum,
    capturing the worst-case scenario regardless of how the tensor is used in GEMM operations.
    If False, computes only for rowwise orientation.

    Returns 0 if all blocks are zeros, otherwise computes dynamic range over non-zero blocks.

    Args:
        tensor: Input tensor (will be flattened to 2D)
        stat_config: BlockwiseDynamicRangeStat named tuple with:
            - block_size: Size of blocks (int)
            - dims: 1 for 1D blocks (consecutive elements), 2 for 2D blocks (tiles)
            - max_over_orientations: If True, compute max over rowwise and columnwise orientations
    """
    # Extract parameters from stat_config
    block_size = stat_config.block_size
    dims = stat_config.dims
    max_over_orientations = stat_config.max_over_orientations

    def _compute_for_one_orientation(tensor):
        total_numel = tensor.numel()
        assert dims in [1, 2], f"dims must be 1 or 2, got {dims}"

        # torch.compile friendly code - standard ** power does not work with jit
        total_block_size = block_size * block_size if dims == 2 else block_size
        assert (
            total_numel % total_block_size == 0
        ), f"Tensor numel ({total_numel}) is not divisible by block_size ({block_size})."

        tensor = tensor.abs().float()
        if dims == 1:
            tensor = tensor.reshape(-1, block_size)
            per_block_amax = tensor.amax(dim=1)
            per_block_amin = tensor.masked_fill(tensor == 0, float("inf")).amin(dim=1)
        else:
            # We want to have tensor of shape [nr_blocks, block_size, block_size],
            # where each block is a block_size x block_size tile of the original tensor.
            dim_y = tensor.shape[-1] // block_size
            tensor = (
                tensor.reshape(-1, block_size, dim_y, block_size)
                .permute(0, 2, 1, 3)
                .reshape(-1, block_size, block_size)
            )
            per_block_amax = tensor.amax(dim=(1, 2))
            per_block_amin = tensor.masked_fill(tensor == 0, float("inf")).amin(dim=(1, 2))

        # Identify blocks that contain any non-zero element
        nonzero_blocks = per_block_amax != 0
        dynamic_range_per_block = torch.where(
            nonzero_blocks,
            torch.log2(per_block_amax) - torch.log2(per_block_amin),
            torch.zeros_like(per_block_amax, dtype=torch.float32),
        )
        return dynamic_range_per_block.max()

    # Flatten to 2D
    tensor_2d = tensor.reshape(-1, tensor.shape[-1])
    if max_over_orientations:
        return max(
            _compute_for_one_orientation(tensor_2d),  # Rowwise orientation
            _compute_for_one_orientation(tensor_2d.transpose(-2, -1)),  # Columnwise orientation
        )
    return _compute_for_one_orientation(tensor_2d)


@torch.compile
def compute_variance(variances, numels, sums):
    """Welford algorithm is used for numerically stable distributed variance computation."""
    mean = torch.sum(sums) / torch.sum(numels)
    means = sums / numels
    var = torch.sum(numels * (variances - torch.pow((means - mean), 2))) / torch.sum(numels)
    return var


@torch.compile
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

FP8_NEGATIVE_ZERO = 128  # represnts -0.0 in fp8


def count_nonzero_fp8(fp8_data: torch.Tensor) -> torch.Tensor:
    """Count the number of non-zero elements in the fp8 data."""
    fp8_data = fp8_data.view(dtype=torch.uint8)
    zero_vals = torch.tensor([0, FP8_NEGATIVE_ZERO], device=fp8_data.device, dtype=torch.uint8)
    return fp8_data.numel() - torch.isin(fp8_data, zero_vals).sum()


def add_underflows_stats(recipe_name: str, columnwise: bool = False):
    """Register *both* underflow stats (num and %) for the given recipe."""
    columnwise_suffix = "_columnwise" if columnwise else ""

    # Stat names
    stat_num = f"{recipe_name}{'_' if recipe_name != '' else ''}underflows_num{columnwise_suffix}"
    stat_pct = f"{recipe_name}{'_' if recipe_name != '' else ''}underflows%{columnwise_suffix}"

    stats_to_num[stat_num] = len(stats_to_num)
    stats_to_num[stat_pct] = len(stats_to_num)

    STATS[stat_num] = (
        lambda x, aux_dict: x.count_nonzero()
        - count_nonzero_fp8(
            aux_dict[recipe_name].get_data_tensors(
                rowwise_data=not columnwise, columnwise_data=columnwise
            )
        ),
        lambda buffers, _sn=stat_num: sum(_get(buffers, _sn)),
    )
    STATS[stat_pct] = (
        lambda x, aux_dict: (
            x.count_nonzero()
            - count_nonzero_fp8(
                aux_dict[recipe_name].get_data_tensors(
                    rowwise_data=not columnwise, columnwise_data=columnwise
                )
            )
        )
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


def add_max_blockwise_dynamic_range_stats(
    block_size: int, dims: int, max_over_orientations: bool = False
):
    """Register max_blockwise_X_dynamic_range stats for the recipe.

    Args:
        block_size: Size of blocks for computing blockwise dynamic range
        dims: 1 for 1D blocks, 2 for 2D blocks
        max_over_orientations: Whether to compute max over rowwise and columnwise orientations

    Returns:
        BlockwiseDynamicRangeStat named tuple representing this stat (used as the stat key)
    """
    # Use named tuple directly as the stat key - this is cleaner than string keys
    stat_key = BlockwiseDynamicRangeStat(block_size, dims, max_over_orientations)

    if stat_key in stats_to_num:
        return stat_key  # already registered

    assert dims in [1, 2], f"dims must be 1 or 2, got {dims}"
    stats_to_num[stat_key] = len(stats_to_num)
    DEPENDENCIES[stat_key] = {stat_key}

    STATS[stat_key] = (
        lambda x, aux_dict, _stat_key=stat_key: compute_max_blockwise_dynamic_range(x, _stat_key),
        lambda buffers, _stat_key=stat_key: max(_get(buffers, _stat_key)),
    )

    return stat_key


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
