# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Tensor class with FP8 data"""

import warnings

from .tensor.float8_tensor import Float8Tensor

warnings.warn(
    "transformer_engine.pytorch.float8_tensor is deprecated and will be removed"
    " in a future release. Float8Tensor should be imported directly through "
    "`from transformer_engine.pytorch import Float8Tensor`",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["Float8Tensor"]
