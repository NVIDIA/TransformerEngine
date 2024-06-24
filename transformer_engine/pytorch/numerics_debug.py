# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Utilities for debugging numerical issues with FP8"""
from typing import Tuple
import torch
from transformer_engine.common import recipe

_NUMERICS_DEBUG = False


def debug(enabled: bool = True) -> None:
    """Set FP8 debug mode"""
    global _NUMERICS_DEBUG
    _NUMERICS_DEBUG = enabled


def fp8_tensor_statistics(tensor: torch.Tensor, fp8_format: str = "E4M3") -> Tuple[int, ...]:
    """Print FP8 tensor stats"""
    fp8_format = fp8_format.upper()
    assert fp8_format in (
        "E4M3",
        "E5M2",
    ), "fp8_format must be 'E4M3' or 'E5M2' for amax"

    fmt = recipe.Format[fp8_format]
    FP8_MAX = fmt.value.max_fwd

    num_overflows = (tensor == FP8_MAX).sum().item()
    num_underflows = (tensor == 0).sum().item()
    return (num_underflows, num_overflows)
