# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""Transformer Engine bindings for JAX.

This module provides JAX bindings for NVIDIA's Transformer Engine, enabling
high-performance transformer operations with mixed precision and quantization
support. It includes implementations of key transformer components like attention,
linear layers, and layer normalization, optimized for NVIDIA GPUs.

The module exports various transformer operations and utilities:
- Attention mechanisms (self-attention, cross-attention)
- Linear transformations with optional quantization
- Layer normalization operations
- Activation functions
- Softmax operations
- Sharding utilities for distributed training

All operations are designed to work seamlessly with JAX's functional programming
model and support automatic differentiation.
"""

# pylint: disable=wrong-import-position

import os
import warnings

# This unused import is needed because the top level `transformer_engine/__init__.py`
# file catches an `ImportError` as a guard for cases where the given framework's
# extensions are not available.
import jax

from transformer_engine.common import load_framework_extension

if os.getenv("NVTE_ENABLE_CUTEDSL_BACKEND", "0") != "0":
    warnings.warn(
        "TE/JAX is not yet compatible with CuTeDSL due to XLA, TE/JAX, and Python GC conflicts. "
        "We are working on a fix; in the meantime NVTE_ENABLE_CUTEDSL_BACKEND is forced off."
    )
os.environ["NVTE_ENABLE_CUTEDSL_BACKEND"] = "0"

load_framework_extension("jax")

from . import flax
from . import quantize

from .quantize import autocast, fp8_autocast, update_collections
from .quantize import NVTE_FP8_COLLECTION_NAME

from .sharding import MeshResource

from ..common.utils import deprecate_wrapper
from ..common.utils import DeprecatedEnum


__all__ = [
    "NVTE_FP8_COLLECTION_NAME",
    "autocast",
    "fp8_autocast",
    "update_collections",
    "MeshResource",
    "flax",
    "quantize",
]
