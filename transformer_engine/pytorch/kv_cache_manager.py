# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""KV Cache Manager"""
from collections import OrderedDict
import torch


class KVCacheManager:
    """Base KV cache manager"""

    def __init__(self, *args, **kwargs):
        """Initialize cache manager"""
        self.cache = {}
        self.sequences = OrderedDict()

    def reset(self):
        """Reset cache manager state"""
        self.sequences = OrderedDict()

    def allocate_memory(self, layer_number: int):
        """Allocate memory for the cache"""
        self.cache[layer_number] = (None, None)

    def pre_step(
        self,
        step_dict: OrderedDict,
    ):
        """Update tracked sequences and prepare for step()"""
        return self.sequences

    def step(
        self,
        layer_number: int,
        new_k: torch.Tensor,
        new_v: torch.Tensor,
        cu_new_seqlens: torch.Tensor,
        cu_cached_seqlens: torch.Tensor,
        qkv_format: str,
    ):
        """Copy the new tokens to KV cache"""
        return *self.cache[layer_number], None
