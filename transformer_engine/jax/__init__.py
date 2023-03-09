# Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""Transformer Engine bindings for JAX"""
from .fp8 import fp8_autocast, update_collections, update_fp8_metas
from .module import DenseGeneral, LayerNormDenseGeneral, LayerNormMLP, TransformerEngineBase
from .transformer import extend_logical_axis_rules
from .transformer import RelativePositionBiases, MultiHeadAttention
from .transformer import TransformerLayer, TransformerLayerType
from .sharding import ShardingResource
