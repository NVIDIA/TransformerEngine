# Copyright (c) 2025, BAAI. All rights reserved.
#
# See LICENSE for license information.

import os
import torch
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from .import_utils import safety_import
from .register import register_backend
from .logger import get_logger
logger = get_logger()


### GEMM
general_gemm_fl = safety_import('transformer_engine.plugins.cpp_extensions', 'general_gemm_fl')
### RMSNORM
apply_normalization_fl = safety_import('transformer_engine.plugins.module._common', 'apply_normalization_fl')
rmsnorm_bwd_fl = safety_import('transformer_engine.plugins.cpp_extensions', 'rmsnorm_bwd_fl')
rmsnorm_fwd_fl = safety_import('transformer_engine.plugins.cpp_extensions', 'rmsnorm_fwd_fl')
### AdamW
multi_tensor_adam_fl = safety_import('transformer_engine.plugins.cpp_extensions', 'multi_tensor_adam_fl')
### Flash-Attn
# Use lazy=True to avoid circular imports
FlashAttentionFL = safety_import(
    'transformer_engine.plugins.attention.dot_product_attention.backends',
    'FlashAttentionFL',
    lazy=True
)

def register_backend_fl():
    # Register TE-FL backend
    register_backend("te_fl", {
        "gemm": general_gemm_fl,
        "apply_normalization": apply_normalization_fl,
        "rmsnorm_fwd": rmsnorm_fwd_fl,
        "rmsnorm_bwd": rmsnorm_bwd_fl,
        "adam": multi_tensor_adam_fl,
        "flash_attention": FlashAttentionFL,
    })
