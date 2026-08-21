# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Backward-compatible imports for Newton-Schulz orthogonalization."""

from transformer_engine.pytorch.optimizers.newton_schulz import (
    CoeffIterMode,
    CoeffT,
    CusolverMpCtx,
    NSCoeffT,
    get_coefficient_iterator,
    get_coefficients,
    newton_schulz,
    newton_schulz_tp,
)


__all__ = [
    "CoeffIterMode",
    "CoeffT",
    "CusolverMpCtx",
    "NSCoeffT",
    "get_coefficient_iterator",
    "get_coefficients",
    "newton_schulz",
    "newton_schulz_tp",
]
