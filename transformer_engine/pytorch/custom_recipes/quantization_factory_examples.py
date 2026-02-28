# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""
Quantizer factory examples.

Demonstrates how to use the ``CustomRecipe`` + ``qfactory`` interface to apply
*different* quantization recipes to different module/tensor types/instances within the same model.

Usage::

    from transformer_engine.common.recipe import CustomRecipe
    from transformer_engine.pytorch.quantization import autocast
    from transformer_engine.pytorch.custom_recipes.quantization_factory_examples import (
        nvfp4_linear_mxfp8_grouped_linear_factory,
    )

    recipe = CustomRecipe(qfactory=nvfp4_linear_mxfp8_grouped_linear_factory)
    with autocast(recipe=recipe):
        output = model(input)
"""

from __future__ import annotations

from typing import Optional

import transformer_engine_torch as tex

from transformer_engine.pytorch.quantization import QuantizerRole


def nvfp4_linear_mxfp8_grouped_linear_factory(
    role: Optional[QuantizerRole],
):
    """Quantizer factory: NVFP4 for ``Linear``, MXFP8 for ``GroupedLinear``.

    Dispatch logic:
        * ``role.module_type == "grouped_linear"`` -> MXFP8 (E4M3, block-32)
        * everything else (``"linear"`` or unknown)  -> NVFP4 (E2M1)

    NVFP4 settings follow the built-in ``NVFP4BlockScaling`` defaults:
        * Weights: 2D quantization (16x16), no RHT, no stochastic rounding
        * Inputs:  1D quantization, RHT enabled, no stochastic rounding
        * Grads:   1D quantization, RHT enabled, stochastic rounding enabled
    """
    is_grouped_linear = role is not None and role.module_type == "grouped_linear"

    if is_grouped_linear:
        return _make_mxfp8_quantizer()

    return _make_nvfp4_quantizer(role)


def _make_mxfp8_quantizer():
    """Return an MXFP8 quantizer with default settings (E4M3, block-32, E8M0 scales)."""
    from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer

    return MXFP8Quantizer(
        fp8_dtype=tex.DType.kFloat8E4M3,
    )


def _make_nvfp4_quantizer(role: Optional[QuantizerRole]):
    """Return an NVFP4 quantizer configured per tensor role.

    Mirrors :class:`NVFP4BlockScaling` recipe defaults.
    """
    from transformer_engine.pytorch.tensor.nvfp4_tensor import NVFP4Quantizer

    is_linear = role is not None and role.module_type == "linear"
    is_weight = is_linear and role.tensor_type == "weight"
    is_grad = is_linear and role.tensor_type == "grad_output"

    if is_weight:
        return NVFP4Quantizer(
            fp4_dtype=tex.DType.kFloat4E2M1,
            with_rht=False,
            with_post_rht_amax=False,
            with_2d_quantization=True,
            stochastic_rounding=False,
            with_random_sign_mask=True,
        )

    if is_grad:
        return NVFP4Quantizer(
            fp4_dtype=tex.DType.kFloat4E2M1,
            rowwise=True,
            columnwise=True,
            with_rht=True,
            with_post_rht_amax=True,
            with_2d_quantization=False,
            stochastic_rounding=True,
            with_random_sign_mask=True,
        )

    return NVFP4Quantizer(
        fp4_dtype=tex.DType.kFloat4E2M1,
        rowwise=True,
        columnwise=True,
        with_rht=True,
        with_post_rht_amax=True,
        with_2d_quantization=False,
        stochastic_rounding=False,
        with_random_sign_mask=True,
    )
