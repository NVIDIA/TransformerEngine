# Copyright (c) 2025, BAAI. All rights reserved.
#
# See LICENSE for license information.

import os
import subprocess
from typing import Any, Dict, List, Optional, Tuple, Union

import torch

from transformer_engine.plugin.core.ops import TEFLBackendBase, FP8TensorMeta, NVTE_Fused_Attn_Backend


def _check_kunlunxin_available() -> bool:
    """Check if xpu-smi command can be executed successfully."""
    try:
        result = subprocess.run(
            ["xpu-smi"],
            capture_output=True,
            timeout=5,
            text=True
        )
        
        if result.returncode == 0:
            return True
        else:
            return False
            
    except subprocess.TimeoutExpired:
        return False
    except FileNotFoundError:
        return False
    except OSError as e:
        return False
    except Exception as e:
        return False


class KunLunXinBackend(TEFLBackendBase):
    @staticmethod
    def check_available() -> bool:
        return _check_kunlunxin_available()

    def is_available(self) -> bool:
        return _check_kunlunxin_available()

    def get_flash_attention_class(self):
        from .flash_attention import FlashAttentionTorch
        return FlashAttentionTorch
