# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""
This module provides additional enum and utilities for quantizing tensors in JAX.
"""
from dataclasses import dataclass
from enum import Enum

from transformer_engine_jax import JAXX_Quantize_Layout

__all__ = [
    "QuantizeLayout",
]


@dataclass(frozen=True)
class QuantizeLayout(Enum):
    "Wrapper for JAXX_Quantize_Layout"

    ROWWISE = JAXX_Quantize_Layout.ROWWISE
    COLWISE = JAXX_Quantize_Layout.COLWISE
    ROWWISE_COLWISE = JAXX_Quantize_Layout.ROWWISE_COLWISE

    @property
    def has_rowwise(self) -> bool:
        """If the layout has the rowwise component"""
        return self.value in (JAXX_Quantize_Layout.ROWWISE, JAXX_Quantize_Layout.ROWWISE_COLWISE)

    @property
    def has_colwise(self) -> bool:
        """If the layout has the colwise component"""
        return self.value in (JAXX_Quantize_Layout.COLWISE, JAXX_Quantize_Layout.ROWWISE_COLWISE)

    @property
    def is_rowwise_colwise(self) -> bool:
        """If layout is both rowwise and colwise"""
        return self.value == JAXX_Quantize_Layout.ROWWISE_COLWISE

    @property
    def is_rowwise_only(self) -> bool:
        """If layout is rowwise only"""
        return self.value == JAXX_Quantize_Layout.ROWWISE

    @property
    def is_colwise_only(self) -> bool:
        """If layout is colwise only"""
        return self.value == JAXX_Quantize_Layout.COLWISE

    def __eq__(self, other):
        """Compare this quantize layout with another.

        Args:
            other: The other quantize layout to compare with

        Returns:
            True if the modes are equal, False otherwise
        """
        if not isinstance(other, QuantizeLayout):
            return False
        return self.value == other.value
