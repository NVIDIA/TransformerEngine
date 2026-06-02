# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Transformer Engine bindings for pyTorch"""

# pylint: disable=wrong-import-position

import functools

import torch

from transformer_engine.common import load_framework_extension
from transformer_engine.pytorch.torch_version import torch_version

assert torch_version() >= (2, 1), f"Minimum torch version 2.1 required. Found {torch_version()}."

load_framework_extension("torch")
from transformer_engine.pytorch import constants
from transformer_engine.pytorch.constants import DType
from transformer_engine.pytorch.module import LayerNormLinear
from transformer_engine.pytorch.module import Linear
from transformer_engine.pytorch.module import LayerNormMLP
from transformer_engine.pytorch.module import LayerNorm
from transformer_engine.pytorch.module import RMSNorm
from transformer_engine.pytorch.module import GroupedLinear
from transformer_engine.pytorch.module import Fp8Padding, Fp8Unpadding
from transformer_engine.pytorch.module import initialize_ub
from transformer_engine.pytorch.module import destroy_ub
from transformer_engine.pytorch.module import UserBufferQuantizationMode
from transformer_engine.pytorch.attention import DotProductAttention
from transformer_engine.pytorch.attention import MultiheadAttention
from transformer_engine.pytorch.attention import InferenceParams
from transformer_engine.pytorch.attention import RotaryPositionEmbedding
from transformer_engine.pytorch.transformer import TransformerLayer
from transformer_engine.pytorch.permutation import (
    moe_permute,
    moe_permute_with_probs,
    moe_permute_and_pad_with_probs,
    moe_unpermute,
    moe_sort_chunks_by_index,
    moe_sort_chunks_by_index_with_probs,
)
from transformer_engine.pytorch.quantization import fp8_autocast
from transformer_engine.pytorch.quantization import fp8_model_init
from transformer_engine.pytorch.quantization import autocast
from transformer_engine.pytorch.quantization import quantized_model_init
from transformer_engine.pytorch.quantization import is_fp8_available
from transformer_engine.pytorch.quantization import is_mxfp8_available
from transformer_engine.pytorch.quantization import is_fp8_block_scaling_available
from transformer_engine.pytorch.quantization import is_nvfp4_available
from transformer_engine.pytorch.quantization import get_default_recipe
from transformer_engine.pytorch.quantization import QuantizerRole
from transformer_engine.pytorch.quantization import QuantizerRequest
from transformer_engine.pytorch.quantization import DelayedScalingRequest
from transformer_engine.pytorch.utils import get_cudnn_version
from transformer_engine.pytorch.utils import get_device_compute_capability
from transformer_engine.pytorch.utils import is_bf16_available
from transformer_engine.pytorch.graph import make_graphed_callables
from transformer_engine.pytorch.distributed import checkpoint
from transformer_engine.pytorch.distributed import CudaRNGStatesTracker
from transformer_engine.pytorch.cpu_offload import (
    get_cpu_offload_context,
    mark_not_offload,
    ManualOffloadSynchronizer,
)
from transformer_engine.pytorch import ops
from transformer_engine.pytorch import optimizers
from transformer_engine.pytorch.export import onnx_export
from transformer_engine.pytorch.cross_entropy import parallel_cross_entropy
from transformer_engine.pytorch.newton_schulz import (
    CusolverMpCtx,
    newton_schulz,
)
from transformer_engine.pytorch.quantized_tensor import QuantizedTensorStorage
from transformer_engine.pytorch.quantized_tensor import QuantizedTensor
from transformer_engine.pytorch.quantized_tensor import Quantizer
from transformer_engine.pytorch.quantized_tensor import prepare_for_saving
from transformer_engine.pytorch.quantized_tensor import restore_from_saved
from transformer_engine.pytorch.quantized_tensor import restore_from_func_ctx
from transformer_engine.pytorch.tensor import Float8Quantizer
from transformer_engine.pytorch.tensor import Float8CurrentScalingQuantizer
from transformer_engine.pytorch.tensor import MXFP8Quantizer
from transformer_engine.pytorch.tensor import Float8BlockQuantizer
from transformer_engine.pytorch.tensor import NVFP4Quantizer
from transformer_engine.pytorch.tensor import Float8TensorStorage
from transformer_engine.pytorch.tensor import MXFP8TensorStorage
from transformer_engine.pytorch.tensor import Float8BlockwiseQTensorStorage
from transformer_engine.pytorch.tensor import NVFP4TensorStorage
from transformer_engine.pytorch.tensor import Float8Tensor
from transformer_engine.pytorch.tensor import MXFP8Tensor
from transformer_engine.pytorch.tensor import Float8BlockwiseQTensor
from transformer_engine.pytorch.tensor import NVFP4Tensor
from transformer_engine.pytorch.tensor.float8_tensor import (
    _make_float8_tensor_in_reduce_ex,
)
from transformer_engine.pytorch.tensor.mxfp8_tensor import (
    _make_mxfp8_tensor_in_reduce_ex,
)
from transformer_engine.pytorch.tensor.nvfp4_tensor import (
    _make_nvfp4_tensor_in_reduce_ex,
)
from transformer_engine.pytorch.tensor.float8_blockwise_tensor import (
    _make_float8_blockwise_tensor_in_reduce_ex,
)

try:
    torch._dynamo.config.error_on_nested_jit_trace = False
except AttributeError:
    pass  # error_on_nested_jit_trace was added in PyTorch 2.2.0

# To allow for safe unpickling of QuantizedTensors when using DCP
# checkpointing with FSDP2. ``tex.DType`` (the pybind11 enum) has its
# ``__reduce_ex__`` / ``__reduce__`` overridden in the C++ binding (see
# ``transformer_engine/common/util/pybind_helper.h``) so its pickle
# stream encodes as ``(tex.DType, (int,))`` and only the class itself
# needs to be allow-listed below.
try:
    from torch.serialization import add_safe_globals
    import transformer_engine_torch as tex

    add_safe_globals(
        [
            # Storage mixins (used during pickling of internal-only tensors)
            QuantizedTensorStorage,
            Float8TensorStorage,
            MXFP8TensorStorage,
            NVFP4TensorStorage,
            Float8BlockwiseQTensorStorage,
            # Quantizer types embedded in metadata
            Quantizer,
            Float8Quantizer,
            Float8CurrentScalingQuantizer,
            MXFP8Quantizer,
            NVFP4Quantizer,
            Float8BlockQuantizer,
            # Python IntEnum used as Quantizer.dtype.
            DType,
            # pybind11 enum used as Quantizer.dtype.
            # Kept for backward compatibility.
            tex.DType,
            # __reduce_ex__ reconstructors (module-level functions).
            _make_float8_tensor_in_reduce_ex,
            _make_mxfp8_tensor_in_reduce_ex,
            _make_nvfp4_tensor_in_reduce_ex,
            _make_float8_blockwise_tensor_in_reduce_ex,
        ]
    )
except (ImportError, AttributeError):
    import warnings as _warnings

    _warnings.warn(
        "transformer_engine: torch.serialization.add_safe_globals is "
        "unavailable on this PyTorch version (added in 2.4). DCP "
        "checkpointing of QuantizedTensor weights with FSDP2 will not "
        "work; upgrade to PyTorch >= 2.4 to enable it.",
        RuntimeWarning,
        stacklevel=2,
    )
