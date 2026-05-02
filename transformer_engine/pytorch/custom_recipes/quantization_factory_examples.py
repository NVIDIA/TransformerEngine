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
        nvfp4_linear_fp8_dpa_factory,
        nvfp4_linear_mxfp8_dpa_factory,
    )

    # Mixed module types: NVFP4 for Linear, MXFP8 for GroupedLinear
    recipe = CustomRecipe(qfactory=nvfp4_linear_mxfp8_grouped_linear_factory)
    with autocast(recipe=recipe):
        output = model(input)

    # NVFP4 for Linear, FP8 current-scaling + delayed-scaling for DPA
    recipe = CustomRecipe(qfactory=nvfp4_linear_fp8_dpa_factory, fp8_dpa=True)
    with autocast(recipe=recipe):
        output = model(input)

    # NVFP4 for Linear, MXFP8 for DPA
    recipe = CustomRecipe(qfactory=nvfp4_linear_mxfp8_dpa_factory, fp8_dpa=True)
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


def nvfp4_linear_fp8_dpa_factory(
    role: Optional[QuantizerRole],
):
    """Quantizer factory: NVFP4 for ``Linear``, mixed FP8 for ``DotProductAttention``.

    This factory demonstrates how to use ``CustomRecipe`` with ``fp8_dpa=True``
    to combine NVFP4 quantization for linear layers with FP8 attention.

    DPA tensor types (``role.module_type == "dpa"``):

    =========== ============================================================
    tensor_type Description
    =========== ============================================================
    ``"qkv"``  Query, Key, Value inputs to the first attention GEMM
    ``"s"``    Softmax output (S = softmax(Q·K^T)), fed into the second GEMM
    ``"o"``    Attention output (O = S·V)
    ``"do"``   Gradient of the attention output (dO), backward input
    ``"dp"``   Gradient of the softmax output (dP = dO·V^T), backward
    ``"dqkv"`` Gradient flowing back to Q, K, V
    =========== ============================================================

    Dispatch logic:
        * ``role.module_type == "dpa"`` with ``tensor_type in ("s", "dp")``
          -> FP8 delayed scaling (stateful amax tracking)
        * ``role.module_type == "dpa"`` (QKV, dO)
          -> FP8 current scaling (E4M3)
        * DPA boundary hints (``"dpa_output"`` / ``"dpa_grad_input"`` in ``role.name``)
          -> FP8 current scaling placeholder.  The fused attention kernel requires
          FP8-compatible quantizers in all DPA slots, even when the output is
          produced in BF16 (``fp8_mha=False``).  DPA emits these hint-only roles
          (with empty ``module_type`` and ``tensor_type``) when the downstream
          consumer is unknown.
        * everything else (``"linear"`` / ``"grouped_linear"`` / ``None``)
          -> NVFP4 (E2M1), configured per tensor role

    Usage::

        from transformer_engine.common.recipe import CustomRecipe
        from transformer_engine.pytorch.quantization import autocast
        from transformer_engine.pytorch.custom_recipes.quantization_factory_examples import (
            nvfp4_linear_fp8_dpa_factory,
        )

        recipe = CustomRecipe(
            qfactory=nvfp4_linear_fp8_dpa_factory,
            fp8_dpa=True,
        )
        with autocast(recipe=recipe):
            output = model(input)
    """
    from transformer_engine.pytorch.quantization import DelayedScalingRequest
    from transformer_engine.pytorch.tensor.float8_tensor import Float8CurrentScalingQuantizer

    is_dpa = role is not None and role.module_type == "dpa"
    is_softmax_or_dp = is_dpa and role.tensor_type in ("s", "dp")

    if is_softmax_or_dp:
        return DelayedScalingRequest()

    if is_dpa:
        return Float8CurrentScalingQuantizer(
            fp8_dtype=tex.DType.kFloat8E4M3,
            device="cuda",
        )

    # DPA boundary slots (O output / dQKV grad-input): the fused attention
    # kernel only supports FP8 quantizers here, regardless of the linear recipe.
    is_dpa_boundary = (
        role is not None
        and not role.module_type
        and ("dpa_output" in role.name or "dpa_grad_input" in role.name)
    )
    if is_dpa_boundary:
        return Float8CurrentScalingQuantizer(
            fp8_dtype=tex.DType.kFloat8E4M3,
            device="cuda",
        )

    return _make_nvfp4_quantizer(role)


def nvfp4_linear_mxfp8_dpa_factory(
    role: Optional[QuantizerRole],
):
    """Quantizer factory: NVFP4 for ``Linear``, MXFP8 for ``DotProductAttention``.

    Mirrors the documented "NVFP4 linear + MXFP8 attention" combo from
    :mod:`transformer_engine.pytorch.attention.dot_product_attention.dot_product_attention`
    (see the recipe-combination table at the top of that module). With
    ``CustomRecipe`` the per-tensor decision is made directly here, so the
    ``NVTE_DPA_FP8_RECIPE="MXFP8BlockScaling"`` env override that the
    built-in recipes would otherwise need is unnecessary.

    DPA tensor types (``role.module_type == "dpa"``):

    =========== ============================================================
    tensor_type Description
    =========== ============================================================
    ``"qkv"``  Query, Key, Value inputs to the first attention GEMM
    ``"s"``    Softmax output (S = softmax(Q·K^T)), fed into the second GEMM
    ``"o"``    Attention output (O = S·V)
    ``"do"``   Gradient of the attention output (dO), backward input
    ``"dp"``   Gradient of the softmax output (dP = dO·V^T), backward
    ``"dqkv"`` Gradient flowing back to Q, K, V
    =========== ============================================================

    Dispatch logic:
        * ``role.module_type == "dpa"`` -> MXFP8 (E4M3, block-32)
          The MXFP8 fused-attention kernel handles the S/dP slots
          internally, so any quantizer returned for those roles is later
          nulled out by ``get_attention_quantizers``.  Returning MXFP8 is
          the simplest valid choice.
        * DPA boundary hints (``"dpa_output"`` / ``"dpa_grad_input"`` in
          ``role.name``) -> MXFP8 placeholder.  The fused attention kernel
          requires FP8-compatible quantizers in all DPA slots.
        * everything else (``"linear"`` / ``"grouped_linear"`` / ``None``)
          -> NVFP4 (E2M1), configured per tensor role.

    Usage::

        from transformer_engine.common.recipe import CustomRecipe
        from transformer_engine.pytorch.quantization import autocast
        from transformer_engine.pytorch.custom_recipes.quantization_factory_examples import (
            nvfp4_linear_mxfp8_dpa_factory,
        )

        recipe = CustomRecipe(
            qfactory=nvfp4_linear_mxfp8_dpa_factory,
            fp8_dpa=True,
        )
        with autocast(recipe=recipe):
            output = model(input)
    """
    is_dpa = role is not None and role.module_type == "dpa"
    if is_dpa:
        return _make_mxfp8_quantizer()

    # DPA boundary slots (O output / dQKV grad-input): emitted by DPA with
    # empty `module_type` and a `name` like "<dpa>.dpa_output". The fused
    # attention kernel requires an FP8-compatible quantizer here even when
    # the downstream consumer is unknown.
    is_dpa_boundary = (
        role is not None
        and not role.module_type
        and ("dpa_output" in role.name or "dpa_grad_input" in role.name)
    )
    if is_dpa_boundary:
        return _make_mxfp8_quantizer()

    return _make_nvfp4_quantizer(role)
