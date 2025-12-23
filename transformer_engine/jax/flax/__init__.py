# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""Transformer Engine bindings for JAX"""
from .module import DenseGeneral, LayerNorm
from .module import LayerNormDenseGeneral, LayerNormMLP
from .module import wrap_function_in_te_state_module, make_dot_general_cls
from .transformer import extend_logical_axis_rules
from .transformer import DotProductAttention, MultiHeadAttention, RelativePositionBiases
from .transformer import TransformerLayer, TransformerLayerType

__all__ = [
    "DenseGeneral",
    "LayerNorm",
    "LayerNormDenseGeneral",
    "LayerNormMLP",
    "wrap_function_in_te_state_module",
    "make_dot_general_cls",
    "extend_logical_axis_rules",
    "DotProductAttention",
    "MultiHeadAttention",
    "RelativePositionBiases",
    "TransformerLayer",
    "TransformerLayerType",
]
