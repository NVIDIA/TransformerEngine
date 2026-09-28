# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Python interface for attention"""

from .dot_product_attention import DotProductAttention
from .packed_sequence import register_cu_seqlens, attention_backend_workspace
from .linear_attention import (
    GatedDeltaNetAttention,
    GatedDeltaNet2Attention,
    GatedDeltaProductAttention,
)
from .fused_mla_q_uproj import FusedMLAQUpProjFunction, FusedMLAQUpProjRopeQuant
from .multi_head_attention import MultiheadAttention
from .inference import InferenceParams
from .rope import RotaryPositionEmbedding

__all__ = [
    "register_cu_seqlens",
    "attention_backend_workspace",
    "DotProductAttention",
    "GatedDeltaNetAttention",
    "GatedDeltaNet2Attention",
    "GatedDeltaProductAttention",
    "FusedMLAQUpProjFunction",
    "FusedMLAQUpProjRopeQuant",
    "MultiheadAttention",
    "InferenceParams",
    "RotaryPositionEmbedding",
]
