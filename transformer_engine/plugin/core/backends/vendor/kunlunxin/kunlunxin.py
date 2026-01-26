# Copyright (c) 2025, BAAI. All rights reserved.
#
# See LICENSE for license information.

import os
import subprocess
from typing import Any, Dict, List, Optional, Tuple, Union

import torch

from transformer_engine.plugin.core.ops import TEFLBackendBase, FP8TensorMeta, NVTE_Fused_Attn_Backend

_kunlunxin_available = False

def _ensure_kunlunxin_available():
    global _kunlunxin_available
    if not _kunlunxin_available:
        try:
            result = subprocess.run(
                ["xpu-smi"],
                capture_output=True,
                timeout=10,
                text=True
            )
            
            if result.returncode == 0:
                _kunlunxin_available = True
            else:
                _kunlunxin_available = False
                
        except subprocess.TimeoutExpired:
            _kunlunxin_available = False
        except FileNotFoundError:
            _kunlunxin_available = False
        except OSError as e:
            _kunlunxin_available = False
        except Exception as e:
            _kunlunxin_available = False
    
    return _kunlunxin_available


def _check_kunlunxin_available() -> bool:
    """Check if xpu-smi command can be executed successfully."""
    if _ensure_kunlunxin_available():
        return True
    else:
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
