# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Transformer Engine bindings for Paddle"""

# pylint: disable=wrong-import-position,wrong-import-order


def _load_library():
    """Load shared library with Transformer Engine C extensions"""
    from transformer_engine import transformer_engine_paddle  # pylint: disable=unused-import


_load_library()
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
