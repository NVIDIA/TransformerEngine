# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Utility functions for experimental middleware between Transformer Engine and Kitchen."""

import enum

import torch


HIGH_PRECISION_FLOAT_DTYPES = (
    torch.float,
    torch.float16,
    torch.bfloat16,
    torch.float32,
)


class Fp4Formats(enum.Enum):
    """FP4 data format"""

    E2M1 = "e2m1"


def roundup_div(x: int, y: int) -> int:
    """Round up division"""
    assert x >= 0
    assert y > 0
    return (x + y - 1) // y
