# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""EXPERIMENTAL debugging utilities for Transformer Engine JAX.

This API is experimental and may change or be removed without deprecation in future releases.
"""

from .inspect import inspect_array, load_array_dump

__all__ = [
    "inspect_array",
    "load_array_dump",
]
