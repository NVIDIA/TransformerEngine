# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""Transformer Engine bindings for Paddle"""

from .fp8 import fp8_autocast
from .layer import (
    Linear,
    LayerNorm,
    LayerNormLinear,
    LayerNormMLP,
    FusedScaleMaskSoftmax,
    DotProductAttention,
    MultiHeadAttention,
    TransformerLayer,
    RotaryPositionEmbedding,
)
from .recompute import recompute
