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
general_gemm_native = safety_import('transformer_engine.pytorch.cpp_extensions', 'general_gemm')
### RMSNORM
apply_normalization_native = safety_import('transformer_engine.pytorch.module._common', 'apply_normalization')
rmsnorm_bwd_native = safety_import('transformer_engine_torch', 'rmsnorm_bwd')
rmsnorm_fwd_native = safety_import('transformer_engine_torch', 'rmsnorm_fwd')
### AdamW
multi_tensor_adam_native = safety_import('transformer_engine_torch', 'multi_tensor_adam')
### Flash-Attn
# Use lazy=True to avoid circular imports
FlashAttentionNative = safety_import(
    'transformer_engine.pytorch.attention.dot_product_attention.backends',
    'FlashAttention',
    lazy=True
)

# Register native backend
def register_backend_native():
    # Note: native_rmsnorm_bwd doesn't take eps as the last argument, so we wrap it
    def rmsnorm_bwd_native_wrapper(*args, **kwargs):
        return rmsnorm_bwd_native(*args[:-1], **kwargs)
    register_backend("native", {
        "gemm": general_gemm_native,
        "apply_normalization": apply_normalization_native,
        "rmsnorm_fwd": rmsnorm_fwd_native,
        "rmsnorm_bwd": rmsnorm_bwd_native_wrapper,
        "adam": multi_tensor_adam_native,
        "flash_attention": FlashAttentionNative,
    })
