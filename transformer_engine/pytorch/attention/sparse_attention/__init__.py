# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Experimental sparse attention variants.

Each variant owns its tensor/index contract and preparation helpers. Optional
kernels load on execution. Add future variants as sibling modules.
"""

from . import dsa_cudnn_kernels, dsv4_attention
from .dsv4_attention import DSv4Attention, DSv4HybridAttention

__all__ = [
    "dsa_cudnn_kernels",
    "dsv4_attention",
    "DSv4Attention",
    "DSv4HybridAttention",
]
