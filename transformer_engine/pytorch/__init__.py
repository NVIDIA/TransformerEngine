# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Transformer Engine bindings for pyTorch"""
from .module import LayerNormLinear
from .module import Linear
from .module import LayerNormMLP
from .module import LayerNorm
from .module import RMSNorm
from .attention import DotProductAttention
from .attention import InferenceParams
from .attention import MultiheadAttention
from .transformer import TransformerLayer
from .fp8 import fp8_autocast
from .fp8 import fp8_model_init
from .graph import make_graphed_callables
from .export import onnx_export
from .distributed import checkpoint
from .distributed import CudaRNGStatesTracker
from .cpu_offload import get_cpu_offload_context
# Register custom op symbolic ONNX functions
from .te_onnx_extensions import (
    onnx_cast_to_fp8,
    onnx_cast_to_fp8_noalloc,
    onnx_cast_from_fp8,
    onnx_fp8_gelu,
    onnx_fp8_relu,
    onnx_te_gemm,
    onnx_layernorm_fwd_fp8,
    onnx_layernorm_fwd,
    onnx_rmsnorm_fwd,
    onnx_rmsnorm_fwd_fp8
)
try:
    import torch
    torch._dynamo.config.error_on_nested_jit_trace = False
except: # pylint: disable=bare-except
    pass
