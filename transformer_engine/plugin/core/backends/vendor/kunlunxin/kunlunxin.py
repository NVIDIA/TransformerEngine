# Copyright (c) 2025, BAAI. All rights reserved.
#
# See LICENSE for license information.

import os
from typing import Any, Dict, List, Optional, Tuple, Union

import torch

from transformer_engine.plugin.core.ops import TEFLBackendBase, FP8TensorMeta, NVTE_Fused_Attn_Backend


class KunLunXinBackend(TEFLBackendBase):
    @staticmethod
    def check_available() -> bool:
        return True

    def is_available(self) -> bool:
        return True

    def get_flash_attention_class(self):
        from .flash_attention import FlashAttentionTorch
        return FlashAttentionTorch
