# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""Transformer Engine bindings for JAX"""
from .module import DenseGeneral, LayerNorm, LayerNormDenseGeneral, LayerNormMLP
from .transformer import (
    DotProductAttention,
    MultiHeadAttention,
    RelativePositionBiases,
    TransformerLayer,
    TransformerLayerType,
    extend_logical_axis_rules,
)

__all__ = [
    "DenseGeneral",
    "LayerNorm",
    "LayerNormDenseGeneral",
    "LayerNormMLP",
    "extend_logical_axis_rules",
    "DotProductAttention",
    "MultiHeadAttention",
    "RelativePositionBiases",
    "TransformerLayer",
    "TransformerLayerType",
]
