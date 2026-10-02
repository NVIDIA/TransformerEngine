# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Experimental sparse attention variants.

Each variant owns its tensor/index contract and preparation helpers. Optional
kernels load on execution. Add future variants as siblings of ``dsv4``.
"""

from . import dsv4
from .dsv4 import DSv4Attention
from .dsv4_hybrid_attention import DSv4HybridAttention

__all__ = ["dsv4", "DSv4Attention", "DSv4HybridAttention"]
