# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""
Metadata classes for quantization in JAX.

This module provides classes for managing quantization metadata, including
scale factors and amax history for different tensor types.
"""
from dataclasses import dataclass


__all__ = ["QuantizeMeta", "QuantizeMetaSet"]


class QuantizeMeta:
    """Metadata for quantization parameters.

    For Delayed Scaling recipe:
        scale: The scaling factor for quantization
        amax_history: History of maximum absolute values

    For NVFP4 recipe with Stochastic Rounding:
        sr_rng_state: The state of the stochastic rounding RNG

    """

    def __init__(self, **kwargs):
        self._kwargs = kwargs

    def get_kwargs_dictionary(self):
        """Get the metadata as a dictionary."""
        return self._kwargs


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
