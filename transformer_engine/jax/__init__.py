# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""Transformer Engine bindings for JAX"""

# pylint: disable=wrong-import-position,wrong-import-order

import ctypes

from transformer_engine.common import get_te_path
from transformer_engine.common import _get_sys_extension


def _load_library():
    """Load shared library with Transformer Engine C extensions"""
    extension = _get_sys_extension()
    try:
        so_dir = get_te_path() / "transformer_engine"
        so_path = next(so_dir.glob(f"transformer_engine_jax.*.{extension}"))
    except StopIteration:
        so_dir = get_te_path()
        so_path = next(so_dir.glob(f"transformer_engine_jax.*.{extension}"))

    return ctypes.CDLL(so_path, mode=ctypes.RTLD_GLOBAL)


_TE_JAX_LIB_CTYPES = _load_library()
from . import flax
from .fp8 import fp8_autocast, update_collections, get_delayed_scaling
from .fp8 import NVTE_FP8_COLLECTION_NAME
from .sharding import MeshResource
from .sharding import MajorShardingType, ShardingResource, ShardingType

from ..common.utils import deprecate_wrapper
from ..common.utils import DeprecatedEnum

MajorShardingType = DeprecatedEnum(
    MajorShardingType, "MajorShardingType is deprecating in the near feature."
)
ShardingType = DeprecatedEnum(ShardingType, "ShardingType is deprecating in the near feature.")
ShardingResource = deprecate_wrapper(
    ShardingResource,
    "ShardingResource is renamed to MeshResource, and will be removed in the near feature.",
)

__all__ = [
    "NVTE_FP8_COLLECTION_NAME",
    "fp8_autocast",
    "update_collections",
    "get_delayed_scaling",
    "MeshResource",
    "MajorShardingType",
    "ShardingResource",
    "ShardingType",
    "flax",
    "praxis",
]
