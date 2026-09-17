# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Python interface for linear attention.

This module is **experimental** and subject to change.
"""

from .gdn import GatedDeltaNetAttention
from .gdn2 import GatedDeltaNet2Attention

__all__ = ["GatedDeltaNetAttention", "GatedDeltaNet2Attention"]
