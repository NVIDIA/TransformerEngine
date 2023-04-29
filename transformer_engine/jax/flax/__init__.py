# Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""Transformer Engine bindings for JAX"""
from .module import DenseGeneral, LayerNorm
from .module import LayerNormDenseGeneral, LayerNormMLP, TransformerEngineBase
from .transformer import extend_logical_axis_rules
from .transformer import MultiHeadAttention, RelativePositionBiases
from .transformer import TransformerLayer, TransformerLayerType
