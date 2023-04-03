# Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Transformer Engine bindings for Tensorflow"""
from transformer_engine.common.recipe import DelayedScaling
from transformer_engine.common.recipe import Format

from .constants import TE_DType
from .fp8 import fp8_autocast
from .module import Dense
from .module import LayerNorm
from .module import LayerNormDense
from .module import LayerNormMLP
from .module import get_stream_id

from .transformer import MultiHeadAttention
from .transformer import TransformerLayer
