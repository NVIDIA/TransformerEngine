# Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""Transformer Engine bindings for JAX"""

from . import flax
from .fp8 import fp8_autocast, update_collections, update_fp8_metas, get_delayed_scaling
from .sharding import MajorShardingType, ShardingResource, ShardingType


__all__ = [
    'fp8_autocast', 'update_collections', 'update_fp8_metas', 'get_delayed_scaling',
    'MajorShardingType', 'ShardingResource', 'ShardingType', 'flax', 'praxis',
]
