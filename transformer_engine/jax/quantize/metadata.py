# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""
Metadata classes for quantization in JAX.

This module provides classes for managing quantization metadata, including
scale factors and amax history for different tensor types.
"""
from dataclasses import dataclass
import jax.numpy as jnp

__all__ = ["QuantizeMeta", "QuantizeMetaSet"]


@dataclass
class QuantizeMeta:
    """Metadata for quantization parameters.

    Attributes:
        scale: The scaling factor for quantization
        amax_history: History of maximum absolute values
    """

    scale: jnp.ndarray
    amax_history: jnp.ndarray


@dataclass
class QuantizeMetaSet:
    """Set of quantization metadata for different tensor types.

    Attributes:
        x: Quantization metadata for input tensors
        kernel: Quantization metadata for kernel tensors
        grad: Quantization metadata for gradient tensors
    """

    x: QuantizeMeta
    kernel: QuantizeMeta
    grad: QuantizeMeta
