# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""
Quantizer factory examples using real silicon quantizers.

Each factory below replicates the behaviour of built-in TE recipe but via the
``CustomRecipe`` + ``qfactory`` interface.  This is useful when you want to
start from a known-good recipe and then selectively override quantizer settings
for specific layers / tensor types.

Usage (any factory)::

    from transformer_engine.common.recipe import CustomRecipe
    from transformer_engine.pytorch.quantization import autocast
    from transformer_engine.pytorch.custom_recipes.quantization_recipes_base import (
        nvfp4_quantizer_factory,
    )

    recipe = CustomRecipe(qfactory=nvfp4_quantizer_factory)
    with autocast(recipe=recipe):
        output = model(input)
"""

from __future__ import annotations

from typing import Optional

import torch
import transformer_engine_torch as tex

from transformer_engine.pytorch.quantization import QuantizerRole


def current_scaling_quantizer_factory(
    role: Optional[QuantizerRole],
) -> "Float8CurrentScalingQuantizer":
    """Factory that mirrors :class:`Float8CurrentScaling` recipe defaults.

    * Forward tensors (input, weight) → E4M3
    * Backward tensors (grad_output) → E5M2
    """
    from transformer_engine.pytorch.tensor.float8_tensor import (
        Float8CurrentScalingQuantizer,
    )

    is_backward = role is not None and role.tensor_type == "grad_output"
    fp8_dtype = tex.DType.kFloat8E5M2 if is_backward else tex.DType.kFloat8E4M3

    return Float8CurrentScalingQuantizer(
        fp8_dtype=fp8_dtype,
        device=torch.device("cuda"),
        force_pow_2_scales=False,  # constrain scale to powers of 2
        amax_epsilon=0.0,  # clamp amax from below to avoid div-by-zero
    )


def mxfp8_quantizer_factory(
    role: Optional[QuantizerRole],
) -> "MXFP8Quantizer":
    """Factory that mirrors :class:`MXFP8BlockScaling` recipe defaults.

    * E4M3 by default for all tensors
    * Block size 32, power-of-2 (E8M0) scales
    """
    from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer

    return MXFP8Quantizer(
        fp8_dtype=tex.DType.kFloat8E4M3,
    )


def float8_block_scaling_quantizer_factory(
    role: Optional[QuantizerRole],
) -> "Float8BlockQuantizer":
    """Factory that mirrors :class:`Float8BlockScaling` recipe defaults.

    * E4M3 by default for all tensors
    * Weights use 2D block scaling, everything else uses 1D
    * Power-of-2 scales by default
    """
    from transformer_engine.pytorch.tensor.float8_blockwise_tensor import (
        Float8BlockQuantizer,
    )

    is_weight = (
        role is not None
        and role.module_type in ("linear", "grouped_linear")
        and role.tensor_type == "weight"
    )
    block_scaling_dim = 2 if is_weight else 1

    return Float8BlockQuantizer(
        fp8_dtype=tex.DType.kFloat8E4M3,
        rowwise=True,
        columnwise=True,
        amax_epsilon=0.0,  # clamp amax from below to avoid div-by-zero
        force_pow_2_scales=True,
        block_scaling_dim=block_scaling_dim,  # 1 = 1D (1×128), 2 = 2D (128×128)
    )


def nvfp4_quantizer_factory(
    role: Optional[QuantizerRole],
) -> "NVFP4Quantizer":
    """Factory that mirrors :class:`NVFP4BlockScaling` recipe defaults.

    * All tensors quantized to E2M1 (FP4)
    * Weights: 2D quantization (16x16 blocks), no RHT, no stochastic rounding
    * Inputs:  1D quantization, RHT enabled, no stochastic rounding
    * Grads:   1D quantization, RHT enabled, stochastic rounding enabled

    Quantizer knobs:
        fp4_dtype            - E2M1 (only supported format)
        with_rht             - randomized Hadamard transform (smooths outliers)
        with_post_rht_amax   - recompute amax after RHT (should match with_rht)
        with_2d_quantization - 16x16 2D blocks (vs 1x16 1D)
        stochastic_rounding  - probabilistic rounding to reduce quant bias
        with_random_sign_mask - random sign flip in the Hadamard matrix
    """
    from transformer_engine.pytorch.tensor.nvfp4_tensor import NVFP4Quantizer

    is_linear = role is not None and role.module_type in ("linear", "grouped_linear")
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

    # For input and unknown roles
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
