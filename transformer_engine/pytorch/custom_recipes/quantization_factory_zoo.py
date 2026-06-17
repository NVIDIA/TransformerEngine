# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""
Quantizer factory zoo.

A collection of composed/mixed-recipe factories built on top of the single-format
building blocks.  They demonstrate how to use the ``CustomRecipe`` + ``qfactory`` 
interface to apply *different* quantization recipes to different
module/tensor types/instances within the same model.

.. warning::

    Use these with caution.  These are **not** official, supported recipes
    provided by Transformer Engine -- they are illustrative examples meant to
    inspire your own experiments, not drop-in production defaults.  While most
    of the factories here are grounded in some evidence or rationale (see the
    per-factory docstrings), they have not been broadly validated for accuracy,
    convergence, or performance across models and hardware.  Treat them as
    starting points: benchmark and verify on your own workload before relying on
    any of them.

Usage::

    from transformer_engine.common.recipe import CustomRecipe
    from transformer_engine.pytorch.quantization import autocast
    from transformer_engine.pytorch.custom_recipes.quantization_factory_zoo import (
        mxfp8_fwd_nvfp4_bwd_quantizer_factory,
        nvfp4_linear_mxfp8_dpa_factory,
    )

    # Linear-only recipe (no attention quantization): the qfactory is the only knob.
    recipe = CustomRecipe(qfactory=mxfp8_fwd_nvfp4_bwd_quantizer_factory)
    with autocast(recipe=recipe):
        output = model(input)

    # Recipe that also quantizes DotProductAttention: set ``fp8_dpa=True`` so the
    # attention GEMMs request quantizers from the factory (DPA roles) too.
    recipe = CustomRecipe(qfactory=nvfp4_linear_mxfp8_dpa_factory, fp8_dpa=True)
    with autocast(recipe=recipe):
        output = model(input)

    # The other factories in this module follow the same two patterns; see their
    # docstrings for the exact per-role dispatch.
"""

from __future__ import annotations

from typing import Optional

from transformer_engine.pytorch.quantization import QuantizerRole
from ..constants import DType
from .quantization_factory_base import mxfp8_quantizer_factory, nvfp4_quantizer_factory


def mxfp8_fwd_nvfp4_bwd_quantizer_factory(
    role: Optional[QuantizerRole],
):
    """Quantizer factory: MXFP8 forward, NVFP4 backward.

    Per-GEMM format consumption:
        * fprop: ``weight.row(MXFP8) x input.row(MXFP8)``
        * dgrad: ``weight.col(NVFP4) x grad_output.row(NVFP4)``
        * wgrad: ``input.col(NVFP4) x grad_output.col(NVFP4)``

    The NVFP4 side mirrors :func:`nvfp4_quantizer_factory` role semantics:
    weights use 2D scaling, activations/grads use the base 1D RHT/SR settings.
    ``HybridQuantizer`` then pins each sub-quantizer to the direction that is
    actually consumed.
    """
    from transformer_engine.pytorch.tensor.hybrid_tensor import HybridQuantizer

    is_linear = role is not None and role.module_type in ("linear", "grouped_linear")
    if is_linear and role.tensor_type in ("input", "weight", "output"):
        return HybridQuantizer(
            rowwise_quantizer=mxfp8_quantizer_factory(role),
            columnwise_quantizer=nvfp4_quantizer_factory(role),
        )
    if is_linear and role.tensor_type in ("grad_output", "grad_input"):
        return nvfp4_quantizer_factory(role)
    return mxfp8_quantizer_factory(role)


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
        return mxfp8_quantizer_factory(role)

    return nvfp4_quantizer_factory(role)


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
        from transformer_engine.pytorch.custom_recipes.quantization_factory_zoo import (
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
            fp8_dtype=DType.kFloat8E4M3,
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
            fp8_dtype=DType.kFloat8E4M3,
            device="cuda",
        )

    return nvfp4_quantizer_factory(role)


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
        from transformer_engine.pytorch.custom_recipes.quantization_factory_zoo import (
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
        return mxfp8_quantizer_factory(role)

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
        return mxfp8_quantizer_factory(role)

    return nvfp4_quantizer_factory(role)


def fwd_high_precision_bwd_mxfp8_factory(
    role: Optional[QuantizerRole],
):
    """Quantizer factory: high-precision forward, MXFP8 backward.

    Dispatch logic:
        * ``grad_output`` / ``grad_input`` -> MXFP8 (E4M3, block-32)
        * everything else -> ``Hybrid(rowwise=IdentityQuantizer, columnwise=MXFP8)``
    """
    from transformer_engine.pytorch.tensor.hybrid_tensor import HybridQuantizer
    from transformer_engine.pytorch.tensor.identity_tensor import IdentityQuantizer

    is_linear = role is not None and role.module_type in ("linear", "grouped_linear")
    if is_linear and role.tensor_type in ("grad_output", "grad_input"):
        return mxfp8_quantizer_factory(role)

    # fprop consumes rowwise high precision; dgrad / wgrad consume columnwise MXFP8.
    return HybridQuantizer(
        rowwise_quantizer=IdentityQuantizer(),
        columnwise_quantizer=mxfp8_quantizer_factory(role),
    )
