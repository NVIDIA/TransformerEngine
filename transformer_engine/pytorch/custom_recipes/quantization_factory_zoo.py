# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""
Quantizer factory zoo.

A collection of composed/mixed-recipe factories built on top of the single-format
building blocks.  They demonstrate how to use the ``CustomRecipe`` + ``qfactory``
interface to apply *different* quantization recipes to different
module/tensor types/instances within the same model, and how to use
``HybridQuantizer`` when rowwise and columnwise tensor directions should use
different formats or sources.

Organization:
    * Linear / grouped-linear recipes: from conservative to more aggressive quantization.
    * RL-oriented recipes: dequantized-backward and MoE-inspired recipes that
      favor more precision or explicit source control in backward GEMMs.
    * Linear + attention recipes: factories that also cover ``DotProductAttention``
      roles and require ``CustomRecipe(..., fp8_dpa=True)``.

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


# -----------------------------------------------------------------------------
# Linear / GroupedLinear Recipes
# -----------------------------------------------------------------------------


def high_precision_fwd_mxfp8_bwd_quantizer_factory(
    role: Optional[QuantizerRole],
):
    """Quantizer factory: high-precision forward, MXFP8 backward.

    Dispatch logic:
        * ``grad_output`` -> MXFP8 (E4M3, block-32)
        * everything else -> ``Hybrid(rowwise=IdentityQuantizer, columnwise=MXFP8)``
    """
    from transformer_engine.pytorch.tensor.hybrid_tensor import HybridQuantizer
    from transformer_engine.pytorch.tensor.identity_tensor import IdentityQuantizer

    is_linear = role is not None and role.module_type in ("linear", "grouped_linear")
    if is_linear and role.tensor_type == "grad_output":
        return mxfp8_quantizer_factory(role)

    # fprop consumes rowwise high precision; dgrad / wgrad consume columnwise MXFP8.
    return HybridQuantizer(
        rowwise_quantizer=IdentityQuantizer(),
        columnwise_quantizer=mxfp8_quantizer_factory(role),
    )


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
    if is_linear and role.tensor_type in ("input", "weight"):
        return HybridQuantizer(
            rowwise_quantizer=mxfp8_quantizer_factory(role),
            columnwise_quantizer=nvfp4_quantizer_factory(role),
        )
    if is_linear and role.tensor_type == "grad_output":
        return nvfp4_quantizer_factory(role)
    return mxfp8_quantizer_factory(role)


def _plain_nvfp4_quantizer(*, row_scaled_nvfp4: bool = False):
    """NVFP4 quantizer without RHT, stochastic rounding, or 2D scaling."""
    from transformer_engine.pytorch.tensor.nvfp4_tensor import NVFP4Quantizer

    return NVFP4Quantizer(
        fp4_dtype=DType.kFloat4E2M1,
        rowwise=True,
        columnwise=True,
        with_rht=False,
        with_post_rht_amax=False,
        with_2d_quantization=False,
        stochastic_rounding=False,
        row_scaled_nvfp4=row_scaled_nvfp4,
    )


def nvfp4_1d_double_quantized_weight_quantizer_factory(
    role: Optional[QuantizerRole],
):
    """Quantizer factory: NVFP4 recipe with 1D weight double quantization.

    Dispatch logic:
        * ``linear`` / ``grouped_linear`` ``weight`` ->
          ``Hybrid(rowwise=plain 1D NVFP4, columnwise=plain 1D NVFP4,
          columnwise_source="rowwise_dequantized")``
        * everything else -> :func:`nvfp4_quantizer_factory`

    ``W.T`` is quantized from the dequantized forward/rowwise ``W`` value
    instead of directly from the original high-precision weight. In
    ``HybridQuantizer`` terms, that source choice is expressed with
    ``columnwise_source="rowwise_dequantized"``.

    All non-weight roles keep the standard NVFP4 factory behavior, including RHT
    for inputs and stochastic rounding for gradients. The weight override uses
    plain 1D NVFP4 in both directions: no RHT, stochastic rounding, row-scaled
    activations, or 2D weight scaling.
    """
    from transformer_engine.pytorch.tensor.hybrid_tensor import HybridQuantizer

    is_linear = role is not None and role.module_type in ("linear", "grouped_linear")
    if is_linear and role.tensor_type == "weight":
        return HybridQuantizer(
            rowwise_quantizer=_plain_nvfp4_quantizer(),
            columnwise_quantizer=_plain_nvfp4_quantizer(),
            columnwise_source="rowwise_dequantized",
        )
    return nvfp4_quantizer_factory(role)


# -----------------------------------------------------------------------------
# RL-Oriented Recipes
# -----------------------------------------------------------------------------


def mxfp8_fwd_dequantized_bwd_quantizer_factory(
    role: Optional[QuantizerRole],
):
    """Quantizer factory: MXFP8 forward, high-precision dequantized backward.

    This expresses the linear/grouped-linear equivalent of
    ``backward_override="dequantized"`` through per-direction quantizers:

        * ``input`` / ``weight`` ->
          ``Hybrid(rowwise=MXFP8, columnwise=Identity, columnwise_source="rowwise_dequantized")``
        * ``grad_output`` -> ``IdentityQuantizer``
        * everything else -> MXFP8

    Backward GEMMs use high-precision operands dequantized from the MXFP8
    forward payload, avoiding gradient quantization.

    This recipe targets RL-style training use cases and is motivated by
    NVIDIA/TransformerEngine#2644, where ``backward_override="dequantized"``
    was introduced:
    https://github.com/NVIDIA/TransformerEngine/pull/2644
    """
    from transformer_engine.pytorch.tensor.hybrid_tensor import HybridQuantizer
    from transformer_engine.pytorch.tensor.identity_tensor import IdentityQuantizer

    is_linear = role is not None and role.module_type in ("linear", "grouped_linear")
    if is_linear and role.tensor_type in ("input", "weight"):
        return HybridQuantizer(
            rowwise_quantizer=mxfp8_quantizer_factory(role),
            columnwise_quantizer=IdentityQuantizer(),
            columnwise_source="rowwise_dequantized",
        )
    if is_linear and role.tensor_type == "grad_output":
        return IdentityQuantizer()
    return mxfp8_quantizer_factory(role)


def nvfp4_row_scaled_fwd_dequantized_bwd_quantizer_factory(
    role: Optional[QuantizerRole],
):
    """Quantizer factory: row-scaled NVFP4 forward, dequantized backward.

    This expresses the linear/grouped-linear equivalent of
    ``NVFP4BlockScaling(row_scaled_activation=True,
    backward_override="dequantized")`` through per-direction quantizers:

        * ``input`` ->
          ``Hybrid(rowwise=row-scaled NVFP4, columnwise=Identity,
          columnwise_source="rowwise_dequantized")``
        * ``weight`` ->
          ``Hybrid(rowwise=plain NVFP4, columnwise=Identity,
          columnwise_source="rowwise_dequantized")``
        * ``grad_output`` -> ``IdentityQuantizer``
        * everything else -> plain NVFP4

    Row-scaled NVFP4 is fprop-only, so the forward quantizers avoid RHT,
    stochastic rounding, and 2D scaling.

    This recipe targets RL-style training use cases and builds on
    NVIDIA/TransformerEngine#2931, which introduced row-scaled NVFP4:
    https://github.com/NVIDIA/TransformerEngine/pull/2931
    """
    from transformer_engine.pytorch.tensor.hybrid_tensor import HybridQuantizer
    from transformer_engine.pytorch.tensor.identity_tensor import IdentityQuantizer

    is_linear = role is not None and role.module_type in ("linear", "grouped_linear")
    if is_linear and role.tensor_type == "input":
        return HybridQuantizer(
            rowwise_quantizer=_plain_nvfp4_quantizer(row_scaled_nvfp4=True),
            columnwise_quantizer=IdentityQuantizer(),
            columnwise_source="rowwise_dequantized",
        )
    if is_linear and role.tensor_type == "weight":
        return HybridQuantizer(
            rowwise_quantizer=_plain_nvfp4_quantizer(),
            columnwise_quantizer=IdentityQuantizer(),
            columnwise_source="rowwise_dequantized",
        )
    if is_linear and role.tensor_type == "grad_output":
        return IdentityQuantizer()
    return _plain_nvfp4_quantizer()


def nvfp4_row_scaled_fwd_mxfp8_bwd_quantizer_factory(
    role: Optional[QuantizerRole],
):
    """Quantizer factory: row-scaled NVFP4 forward, MXFP8 backward.

    This RL-related recipe is inspired by the Composer 2 MoE grouped-GEMM
    recipe described in arXiv:2603.24477.

    Derived from the report: Composer 2 describes row-scaled NVFP4 for the MoE
    forward pass and standard MXFP8 for the MoE backward pass. This factory maps
    that format split onto ``GroupedLinear`` roles.

    Assumed here: regular non-MoE ``Linear`` layers use the MXFP8 fallback. The
    public report does not specify the precision used for non-MoE linears.

    Dispatch logic:

        * ``GroupedLinear`` ``input`` ->
          ``Hybrid(rowwise=row-scaled NVFP4, columnwise=MXFP8)``
        * ``GroupedLinear`` ``weight`` ->
          ``Hybrid(rowwise=plain NVFP4, columnwise=MXFP8)``
        * regular ``Linear`` -> MXFP8
        * ``grad_output`` -> MXFP8
        * everything else -> MXFP8

    Row-scaled NVFP4 is fprop-only, so the forward NVFP4 quantizers avoid
    RHT, stochastic rounding, and 2D scaling. By default, ``HybridQuantizer``
    leaves ``columnwise_source="original"``, so MXFP8 backward operands are
    quantized from the original high-precision tensor. The paper does not
    specify how Composer materializes MXFP8 backward operands from saved
    training tensors.

    Users who want MXFP8 backward operands to include the forward NVFP4
    quantization source can construct the same ``HybridQuantizer`` with
    ``columnwise_source="rowwise_dequantized"``. That makes columnwise MXFP8
    quantization consume the dequantized rowwise NVFP4 value instead of the
    original high-precision tensor, which can be useful when the backward path
    should reflect the forward quantization error (might be helpful for convergence).

    Composer 2 Technical Report:
    https://arxiv.org/abs/2603.24477
    """
    from transformer_engine.pytorch.tensor.hybrid_tensor import HybridQuantizer

    is_grouped_linear = role is not None and role.module_type == "grouped_linear"
    is_linear = role is not None and role.module_type == "linear"
    if is_grouped_linear and role.tensor_type == "input":
        return HybridQuantizer(
            rowwise_quantizer=_plain_nvfp4_quantizer(row_scaled_nvfp4=True),
            columnwise_quantizer=mxfp8_quantizer_factory(role),
        )
    if is_grouped_linear and role.tensor_type == "weight":
        return HybridQuantizer(
            rowwise_quantizer=_plain_nvfp4_quantizer(),
            columnwise_quantizer=mxfp8_quantizer_factory(role),
        )
    if is_grouped_linear and role.tensor_type == "grad_output":
        return mxfp8_quantizer_factory(role)
    if is_linear:
        return mxfp8_quantizer_factory(role)
    return mxfp8_quantizer_factory(role)


# -----------------------------------------------------------------------------
# Linear + Attention Recipes
# -----------------------------------------------------------------------------


def nvfp4_linear_mixed_fp8_dpa_factory(
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
            nvfp4_linear_mixed_fp8_dpa_factory,
        )

        recipe = CustomRecipe(
            qfactory=nvfp4_linear_mixed_fp8_dpa_factory,
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
