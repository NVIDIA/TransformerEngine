# Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""Transformer Engine bindings for JAX"""

from . import flax
from .fp8 import fp8_autocast, update_collections, update_fp8_metas, get_delayed_scaling
from .sharding import MajorShardingType, ShardingResource, ShardingType
from ..common.utils import deprecate_wrapper

extend_logical_axis_rules = deprecate_wrapper(
    flax.extend_logical_axis_rules,
    "extend_logical_axis_rules is moving to transformer_engine.jax.flax module"
    " and will be fully removed in the next release (v1.0.0).")
DenseGeneral = deprecate_wrapper(flax.DenseGeneral,
                                 "DenseGeneral is moving to transformer_engine.jax.flax module"
                                 " and will be fully removed in the next release (v1.0.0).")
LayerNorm = deprecate_wrapper(flax.LayerNorm,
                              "LayerNorm is moving to transformer_engine.jax.flax module"
                              " and will be fully removed in the next release (v1.0.0).")
LayerNormDenseGeneral = deprecate_wrapper(
    flax.LayerNormDenseGeneral,
    "LayerNormDenseGeneral is moving to transformer_engine.jax.flax module"
    " and will be fully removed in the next release (v1.0.0).")
LayerNormMLP = deprecate_wrapper(flax.LayerNormMLP,
                                 "LayerNormMLP is moving to transformer_engine.jax.flax module"
                                 " and will be fully removed in the next release (v1.0.0).")
TransformerEngineBase = deprecate_wrapper(
    flax.TransformerEngineBase,
    "TransformerEngineBase is moving to transformer_engine.jax.flax module"
    " and will be fully removed in the next release (v1.0.0).")
MultiHeadAttention = deprecate_wrapper(
    flax.MultiHeadAttention, "MultiHeadAttention is moving to transformer_engine.jax.flax module"
                             " and will be fully removed in the next release (v1.0.0).")
RelativePositionBiases = deprecate_wrapper(
    flax.RelativePositionBiases,
    "RelativePositionBiases is moving to transformer_engine.jax.flax module"
    " and will be fully removed in the next release (v1.0.0).")
TransformerLayer = deprecate_wrapper(
    flax.TransformerLayer, "TransformerLayer is moving to transformer_engine.jax.flax module"
                           " and will be fully removed in the next release (v1.0.0).")
TransformerLayerType = deprecate_wrapper(
    flax.TransformerLayerType,
    "TransformerLayerType is moving to transformer_engine.jax.flax module"
    " and will be fully removed in the next release (v1.0.0).")

__all__ = [
    'fp8_autocast', 'update_collections', 'update_fp8_metas', 'get_delayed_scaling',
    'MajorShardingType', 'ShardingResource', 'ShardingType', 'flax', 'praxis', 'DenseGeneral',
    'LayerNorm', 'LayerNormDenseGeneral', 'LayerNormMLP', 'TransformerEngineBase',
    'MultiHeadAttention', 'RelativePositionBiases', 'TransformerLayer', 'TransformerLayerType'
]
