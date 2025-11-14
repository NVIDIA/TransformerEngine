# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Transformer Engine bindings for pyTorch"""

# pylint: disable=wrong-import-position

import functools

import torch
from packaging.version import Version as PkgVersion

from transformer_engine.common import load_framework_extension


@functools.lru_cache(maxsize=None)
def torch_version() -> tuple[int, ...]:
    """Get PyTorch version"""
    return PkgVersion(str(torch.__version__)).release


assert torch_version() >= (2, 1), f"Minimum torch version 2.1 required. Found {torch_version()}."


load_framework_extension("torch")
from transformer_engine.pytorch import ops, optimizers
from transformer_engine.pytorch.attention import (
    DotProductAttention,
    InferenceParams,
    MultiheadAttention,
    RotaryPositionEmbedding,
)
from transformer_engine.pytorch.cpu_offload import get_cpu_offload_context
from transformer_engine.pytorch.cross_entropy import parallel_cross_entropy
from transformer_engine.pytorch.distributed import CudaRNGStatesTracker, checkpoint
from transformer_engine.pytorch.export import onnx_export
from transformer_engine.pytorch.graph import make_graphed_callables
from transformer_engine.pytorch.module import (
    Fp8Padding,
    Fp8Unpadding,
    GroupedLinear,
    LayerNorm,
    LayerNormLinear,
    LayerNormMLP,
    Linear,
    RMSNorm,
    UserBufferQuantizationMode,
    destroy_ub,
    initialize_ub,
)
from transformer_engine.pytorch.permutation import (
    moe_permute,
    moe_permute_with_probs,
    moe_sort_chunks_by_index,
    moe_sort_chunks_by_index_with_probs,
    moe_unpermute,
)
from transformer_engine.pytorch.quantization import (
    autocast,
    fp8_autocast,
    fp8_model_init,
    get_default_recipe,
    is_fp8_available,
    is_fp8_block_scaling_available,
    is_mxfp8_available,
    is_nvfp4_available,
    quantized_model_init,
)
from transformer_engine.pytorch.quantized_tensor import (
    QuantizedTensor,
    QuantizedTensorStorage,
    Quantizer,
    prepare_for_saving,
    restore_from_saved,
)
from transformer_engine.pytorch.tensor import (
    Float8BlockQuantizer,
    Float8BlockwiseQTensor,
    Float8BlockwiseQTensorStorage,
    Float8CurrentScalingQuantizer,
    Float8Quantizer,
    Float8Tensor,
    Float8TensorStorage,
    MXFP8Quantizer,
    MXFP8Tensor,
    MXFP8TensorStorage,
    NVFP4Quantizer,
    NVFP4Tensor,
    NVFP4TensorStorage,
)
from transformer_engine.pytorch.transformer import TransformerLayer
from transformer_engine.pytorch.utils import (
    get_cudnn_version,
    get_device_compute_capability,
    is_bf16_available,
)

try:
    torch._dynamo.config.error_on_nested_jit_trace = False
except AttributeError:
    pass  # error_on_nested_jit_trace was added in PyTorch 2.2.0
