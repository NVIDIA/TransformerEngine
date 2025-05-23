# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Transformer Engine bindings for pyTorch"""

# pylint: disable=wrong-import-position

import functools
from packaging.version import Version as PkgVersion

try:
    import torch

    from transformer_engine.common import load_framework_extension

    load_framework_extension("torch")
except RuntimeError as e:
    if "Could not find shared object file" in str(e):
        # If we got here, we could import `torch` but could not load the framework extension.
        # This can happen when a user wants to work only with `transformer_engine.jax` on a system that
        # also has a PyTorch installation. In order to enable that use case, we issue a warning here
        # about the missing PyTorch extension and then convert the RuntimeError into an ImportError
        # that will be caught in the top level `transformer_engine/__init__.py`.
        import warnings

        warnings.warn(
            "Detected a PyTorch installation but could not find the shared object file for the "
            "Transformer Engine PyTorch extension library. If this is not intentional, please "
            "reinstall Transformer Engine with `pip install transformer_engine[pytorch]` or "
            "build from source with `NVTE_FRAMEWORK=pytorch`.",
            category=RuntimeWarning,
        )
        raise ImportError("") from e

    # If we got here, the RuntimeError we caught is unrelated to the framework extension.
    raise e


@functools.lru_cache(maxsize=None)
def torch_version() -> tuple[int, ...]:
    """Get PyTorch version"""
    return PkgVersion(str(torch.__version__)).release


assert torch_version() >= (2, 1), f"Minimum torch version 2.1 required. Found {torch_version()}."

from transformer_engine.pytorch.module import LayerNormLinear
from transformer_engine.pytorch.module import Linear
from transformer_engine.pytorch.module import LayerNormMLP
from transformer_engine.pytorch.module import LayerNorm
from transformer_engine.pytorch.module import RMSNorm
from transformer_engine.pytorch.module import GroupedLinear
from transformer_engine.pytorch.module import Fp8Padding, Fp8Unpadding
from transformer_engine.pytorch.module import initialize_ub
from transformer_engine.pytorch.module import destroy_ub
from transformer_engine.pytorch.attention import DotProductAttention
from transformer_engine.pytorch.attention import MultiheadAttention
from transformer_engine.pytorch.attention import InferenceParams
from transformer_engine.pytorch.attention import RotaryPositionEmbedding
from transformer_engine.pytorch.transformer import TransformerLayer
from transformer_engine.pytorch.permutation import (
    moe_permute,
    moe_permute_with_probs,
    moe_unpermute,
    moe_sort_chunks_by_index,
    moe_sort_chunks_by_index_with_probs,
)
from transformer_engine.pytorch.fp8 import fp8_autocast
from transformer_engine.pytorch.fp8 import fp8_model_init
from transformer_engine.pytorch.graph import make_graphed_callables
from transformer_engine.pytorch.distributed import checkpoint
from transformer_engine.pytorch.distributed import CudaRNGStatesTracker
from transformer_engine.pytorch.cpu_offload import get_cpu_offload_context
from transformer_engine.pytorch import ops
from transformer_engine.pytorch import optimizers
from transformer_engine.pytorch.cross_entropy import parallel_cross_entropy

try:
    torch._dynamo.config.error_on_nested_jit_trace = False
except AttributeError:
    pass  # error_on_nested_jit_trace was added in PyTorch 2.2.0
