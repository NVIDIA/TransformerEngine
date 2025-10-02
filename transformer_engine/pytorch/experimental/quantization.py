# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Quantization API for experimental middleware between Transformer Engine and Kitchen."""

from __future__ import annotations
import dataclasses
import enum

import torch


@enum.unique
class GEMMType(enum.Enum):
    """Type of GEMM operation being performed."""

    FPROP = "fprop"
    DGRAD = "dgrad"
    WGRAD = "wgrad"


@dataclasses.dataclass(frozen=True)
class MMParams:
    """Matrix multiplication parameters."""

    out_dtype: torch.dtype | None = None
    # Use split accumulator for more accurate FP8 GEMM
    use_split_accumulator: bool = True
