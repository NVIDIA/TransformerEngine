# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Transformer Engine bindings for pyTorch"""

# pylint: disable=wrong-import-position,wrong-import-order

import importlib
import sys
import torch

from transformer_engine.common import get_te_path
from transformer_engine.common import _get_sys_extension


def _load_library():
    """Load shared library with Transformer Engine C extensions"""
    extension = _get_sys_extension()
    try:
        so_dir = get_te_path() / "transformer_engine"
        so_path = next(so_dir.glob(f"transformer_engine_torch.*.{extension}"))
    except StopIteration:
        so_dir = get_te_path()
        so_path = next(so_dir.glob(f"transformer_engine_torch.*.{extension}"))

    module_name = "transformer_engine_torch"
    spec = importlib.util.spec_from_file_location(module_name, so_path)
    solib = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = solib
    spec.loader.exec_module(solib)


_load_library()
from transformer_engine.pytorch.module import LayerNormLinear
from transformer_engine.pytorch.module import Linear
from transformer_engine.pytorch.module import LayerNormMLP
from transformer_engine.pytorch.module import LayerNorm
from transformer_engine.pytorch.module import RMSNorm
from transformer_engine.pytorch.module import initialize_ub
from transformer_engine.pytorch.module import destroy_ub
from transformer_engine.pytorch.attention import DotProductAttention
from transformer_engine.pytorch.attention import InferenceParams
from transformer_engine.pytorch.attention import MultiheadAttention
from transformer_engine.pytorch.transformer import TransformerLayer
from transformer_engine.pytorch.fp8 import fp8_autocast
from transformer_engine.pytorch.fp8 import fp8_model_init
from transformer_engine.pytorch.graph import make_graphed_callables
from transformer_engine.pytorch.export import onnx_export
from transformer_engine.pytorch.distributed import checkpoint
from transformer_engine.pytorch.distributed import CudaRNGStatesTracker
from transformer_engine.pytorch.cpu_offload import get_cpu_offload_context
from transformer_engine.pytorch import optimizers

# Register custom op symbolic ONNX functions
from transformer_engine.pytorch.te_onnx_extensions import (
    onnx_cast_to_fp8,
    onnx_cast_to_fp8_noalloc,
    onnx_cast_from_fp8,
    onnx_fp8_gelu,
    onnx_fp8_relu,
    onnx_te_gemm,
    onnx_layernorm_fwd_fp8,
    onnx_layernorm_fwd,
    onnx_rmsnorm_fwd,
    onnx_rmsnorm_fwd_fp8,
)

try:
    torch._dynamo.config.error_on_nested_jit_trace = False
except AttributeError:
    pass  # error_on_nested_jit_trace was added in PyTorch 2.2.0
