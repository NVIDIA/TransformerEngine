# Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""Transformer Engine bindings for JAX"""
from .flax import DenseGeneral, LayerNorm
from .flax import LayerNormDenseGeneral, LayerNormMLP, TransformerEngineBase
from .flax import extend_logical_axis_rules
from .flax import MultiHeadAttention, RelativePositionBiases
from .flax import TransformerLayer, TransformerLayerType
from .fp8 import fp8_autocast, update_collections, update_fp8_metas, get_delayed_scaling
from .sharding import MajorShardingType, ShardingResource, ShardingType
