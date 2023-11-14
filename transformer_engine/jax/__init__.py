# Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""Transformer Engine bindings for JAX"""

from . import flax
from .fp8 import fp8_autocast, update_collections, update_fp8_metas, get_delayed_scaling
from .sharding import MeshResource
from .sharding import MajorShardingType, ShardingResource, ShardingType

from ..common.utils import deprecate_wrapper
from ..common.utils import DeprecatedEnum

MajorShardingType = DeprecatedEnum(MajorShardingType,
                                   "MajorShardingType is deprecating in the near feature.")
ShardingType = DeprecatedEnum(ShardingType, "ShardingType is deprecating in the near feature.")
ShardingResource = deprecate_wrapper(
    ShardingResource,
    "ShardingResource is renaming to MeshResource, and will be removed in the near feature.")

__all__ = [
    'fp8_autocast',
    'update_collections',
    'update_fp8_metas',
    'get_delayed_scaling',
    'MeshResource',
    'MajorShardingType',
    'ShardingResource',
    'ShardingType',
    'flax',
    'praxis',
]
