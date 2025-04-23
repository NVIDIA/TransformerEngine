# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Python interface for dot product attention"""

from .dot_product_attention import DotProductAttention, _attention_backends

__all__ = ["DotProductAttention", "_attention_backends"]
