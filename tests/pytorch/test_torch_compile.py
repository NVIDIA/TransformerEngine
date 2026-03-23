# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Tests for torch.compile integration — quantizer value objects."""

import pytest
import torch
from transformer_engine_torch import DType as TE_DType

from transformer_engine.pytorch.tensor.float8_tensor import Float8CurrentScalingQuantizer
from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer
from transformer_engine.pytorch.tensor.float8_blockwise_tensor import Float8BlockQuantizer
from transformer_engine.pytorch.tensor.nvfp4_tensor import NVFP4Quantizer


def _make_quantizers():
    """Build (quantizer, different_quantizer) pairs for each non-delayed recipe."""
    pairs = [
        (
            MXFP8Quantizer(fp8_dtype=TE_DType.kFloat8E4M3),
            MXFP8Quantizer(fp8_dtype=TE_DType.kFloat8E5M2),
        ),
        (
            Float8CurrentScalingQuantizer(fp8_dtype=TE_DType.kFloat8E4M3, force_pow_2_scales=True),
            Float8CurrentScalingQuantizer(fp8_dtype=TE_DType.kFloat8E4M3, force_pow_2_scales=False),
        ),
        (
            Float8BlockQuantizer(
                fp8_dtype=TE_DType.kFloat8E4M3, rowwise=True, columnwise=True, block_scaling_dim=2
            ),
            Float8BlockQuantizer(
                fp8_dtype=TE_DType.kFloat8E4M3, rowwise=True, columnwise=True, block_scaling_dim=1
            ),
        ),
    ]
    if torch.cuda.is_available():
        pairs.append(
            (
                NVFP4Quantizer(with_rht=True, stochastic_rounding=True),
                NVFP4Quantizer(with_rht=False, stochastic_rounding=False),
            )
        )
    return pairs


@pytest.mark.parametrize(
    "quantizer,different",
    _make_quantizers(),
    ids=lambda q: type(q).__name__,
)
def test_quantizer_value_object(quantizer, different):
    repr_str, globals_dict = quantizer.__fx_repr__()
    reconstructed = eval(repr_str, globals_dict)  # pylint: disable=eval-used

    assert quantizer == reconstructed
    assert hash(quantizer) == hash(reconstructed)
    assert quantizer != different
