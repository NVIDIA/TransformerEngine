# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""PyTorch version utilities"""
from __future__ import annotations
import functools
import torch
from packaging.version import Version as PkgVersion


@functools.lru_cache(maxsize=None)
def torch_version() -> tuple[int, ...]:
    """Get PyTorch version"""
    return PkgVersion(str(torch.__version__)).release
