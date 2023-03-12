# Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Transformer Engine bindings for pyTorch"""
from .module import LayerNormLinear
from .module import Linear
from .module import LayerNormMLP
from .module import LayerNorm
from .module import cuDNN_FlashAttn 
from .transformer import DotProductAttention
from .transformer import TransformerLayer
from .fp8 import fp8_autocast
from .export import onnx_export
from .distributed import checkpoint
# Register custom op symbolic ONNX functions
from .te_onnx_extensions import (
    onnx_cast_to_fp8,
    onnx_cast_from_fp8,
    onnx_fp8_gelu,
    onnx_te_gemm,
    onnx_layernorm_fwd_fp8,
    onnx_layernorm_fwd,
)
