# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""KV Cache Manager."""
from collections import OrderedDict
from typing import Dict, List

import torch

class KVCacheManager:
    """
    KV cache manager. The base class for custom cache managers.
    """
    def __init__(self, *args, **kwargs):
        """Initialize the cache manager."""
        self.cache = {}
        self.sequences = OrderedDict()

    def reset(self):
        """Empty tracked sequences"""
        self.sequences = OrderedDict()

    def allocate_memory(self, layer_number: int):
        """Allocate memory for the KV cache."""
        self.cache[layer_number] = (None, None)

    def pre_step(
        self,
        step_dict: Dict[List, List],
    ):
        """Prepare for operations in step(). Update sequences with step_dict."""
        return self.sequences

    def step(
        self,
        layer_number: int,
        new_k: torch.Tensor,
        new_v: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_kv: torch.Tensor,
        qkv_format: str,
    ):
        """Update the cache with new_k and new_v tokens"""
        return *self.cache[layer_number], None
