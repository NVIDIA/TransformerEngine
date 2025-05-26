# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""Transformer Engine bindings for JAX.

This module provides JAX bindings for NVIDIA's Transformer Engine, enabling
high-performance transformer operations with mixed precision and quantization
support. It includes implementations of key transformer components like attention,
linear layers, and layer normalization, optimized for NVIDIA GPUs.

The module exports various transformer operations and utilities:
- Attention mechanisms (self-attention, cross-attention)
- Linear transformations with optional quantization
- Layer normalization operations
- Activation functions
- Softmax operations
- Sharding utilities for distributed training

All operations are designed to work seamlessly with JAX's functional programming
model and support automatic differentiation.
"""

# pylint: disable=wrong-import-position

# This unused import is needed because the top level `transformer_engine/__init__.py`
# file catches an `ImportError` as a guard for cases where the given framework's
# extensions are not available.
import jax

from transformer_engine.common import load_framework_extension

load_framework_extension("jax")

from . import flax
from . import quantize

from .quantize import fp8_autocast, update_collections, get_delayed_scaling
from .quantize import NVTE_FP8_COLLECTION_NAME

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
    "quantize",
]
