# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""
Example quantizer factories for custom and mixed quantization recipes.

A collection of composed/mixed-recipe factories. They demonstrate how to use
the ``CustomRecipe`` + ``qfactory`` interface to apply *different* quantization
recipes to different module/tensor types/instances within the same model.
Factories may return native Transformer Engine quantizers, custom quantizers,
or ``HybridQuantizer`` instances when tensor directions should use different
representations or sources.

Within the Linear/GroupedLinear and RL-oriented families, examples are roughly
ordered by increasingly aggressive forward quantization. This is an
organizational convention, not an expected accuracy or performance ranking.
These factories demonstrate what can be composed; they are not necessarily
tuned for end-to-end performance.

When forward operands (inputs and weights) are quantized and use a different
representation in backward, consider setting
``columnwise_source="rowwise_dequantized"`` on their hybrid quantizers. This
applies whether backward uses another low-precision format or high precision.
It constructs backward operands from the value obtained during forward
quantization, improving forward/backward representation consistency. This
source choice should not be applied to gradient tensors.
Note: dequantization does not recover information discarded during forward quantization.

Organization:
    * Pre-training-oriented recipes: Favor more precision on the forward pass.
    * RL-oriented recipes: Favor more precision in backward GEMMs.
    * Linear + attention recipes: factories that also cover ``DotProductAttention``
      roles and require ``CustomRecipe(..., fp8_dpa=True)``.

.. warning::

    Use these with caution.  These are **not** official, supported recipes
    provided by Transformer Engine -- they are illustrative examples meant to
    inspire your own experiments, not drop-in production defaults. Most include
    a motivating rationale in their per-factory docstrings, but they have not
    been broadly validated for accuracy, convergence, or performance across
    models and hardware. Treat them as starting points: benchmark and verify on
    your own workload before relying on any of them.

Usage::

    from transformer_engine.common.recipe import CustomRecipe
    from transformer_engine.pytorch.quantization import autocast
    from transformer_engine.pytorch.custom_recipes.quantizer_factory_zoo import (
        mxfp8_fwd_nvfp4_bwd_factory,
        nvfp4_linear_fp8_dpa_factory,
    )

    # Linear-only recipe (no attention quantization): the qfactory is the only knob.
    recipe = CustomRecipe(qfactory=mxfp8_fwd_nvfp4_bwd_factory)
    with autocast(recipe=recipe):
        output = model(input)

    # Recipe that also quantizes DotProductAttention: set ``fp8_dpa=True`` so the
    # attention GEMMs request quantizers from the factory (DPA roles) too.
    recipe = CustomRecipe(qfactory=nvfp4_linear_fp8_dpa_factory, fp8_dpa=True)
    with autocast(recipe=recipe):
        output = model(input)

    # The other factories in this module follow the same two patterns; see their
    # docstrings for the exact per-role dispatch.
"""

from __future__ import annotations

from typing import Optional

from transformer_engine.pytorch.quantization import QuantizerRole
from ..constants import DType
from .quantizer_factories import mxfp8_factory, nvfp4_factory

# -----------------------------------------------------------------------------
# Pre-training-Oriented Recipes
# -----------------------------------------------------------------------------


def high_precision_fwd_mxfp8_bwd_factory(
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
        return mxfp8_factory(role)

    # fprop consumes rowwise high precision; dgrad / wgrad consume columnwise MXFP8.
    return HybridQuantizer(
        rowwise_quantizer=IdentityQuantizer(),
        columnwise_quantizer=mxfp8_factory(role),
    )


def _plain_nvfp4_quantizer(*, row_scaled_nvfp4: bool = False):
    """NVFP4 quantizer without RHT, stochastic rounding, or 2D scaling."""
    from transformer_engine.pytorch.tensor.nvfp4_tensor import NVFP4Quantizer

    return NVFP4Quantizer(
        fp4_dtype=DType.kFloat4E2M1,
        with_rht=False,
        with_post_rht_amax=False,
        with_2d_quantization=False,
        stochastic_rounding=False,
        row_scaled_nvfp4=row_scaled_nvfp4,
    )


def mxfp8_fwd_nvfp4_bwd_factory(
    role: Optional[QuantizerRole],
):
    """Quantizer factory: MXFP8 forward, NVFP4 backward.

    Per-GEMM format consumption:
        * fprop: ``weight.row(MXFP8) x input.row(MXFP8)``
        * dgrad: ``weight.col(NVFP4) x grad_output.row(NVFP4)``
        * wgrad: ``input.col(NVFP4) x grad_output.col(NVFP4)``

    Every backward operand uses 1D NVFP4 scaling. Inputs and gradients mirror
    :func:`nvfp4_factory` semantics: RHT is applied only to the columnwise
    representations consumed by wgrad, and gradients use stochastic rounding.
    The dgrad weight uses plain 1D NVFP4 without RHT or stochastic rounding.

    The backward weight representation consumed by dgrad is quantized to
    NVFP4 from the dequantized MXFP8 forward weight. In ``HybridQuantizer``
    terms, the weight uses ``columnwise_source="rowwise_dequantized"``. The
    backward input representation consumed by wgrad remains quantized directly
    from the original high-precision input with ``columnwise_source="original"``.
    """
    from transformer_engine.pytorch.tensor.hybrid_tensor import HybridQuantizer

    is_linear = role is not None and role.module_type in ("linear", "grouped_linear")
    if is_linear and role.tensor_type == "input":
        return HybridQuantizer(
            rowwise_quantizer=mxfp8_factory(role),
            columnwise_quantizer=nvfp4_factory(role),
            columnwise_source="original",
        )
    if is_linear and role.tensor_type == "weight":
        return HybridQuantizer(
            rowwise_quantizer=mxfp8_factory(role),
            columnwise_quantizer=_plain_nvfp4_quantizer(),
            columnwise_source="rowwise_dequantized",
        )
    if is_linear and role.tensor_type == "grad_output":
        return nvfp4_factory(role)
    return mxfp8_factory(role)


def nvfp4_1d_weight_factory(
    role: Optional[QuantizerRole],
):
    """Quantizer factory: NVFP4 recipe with 1D weight scaling.

    Dispatch logic:
        * ``linear`` / ``grouped_linear`` ``weight`` ->
          ``Hybrid(rowwise=plain 1D NVFP4, columnwise=plain 1D NVFP4,
          columnwise_source="rowwise_dequantized")``
        * everything else -> :func:`nvfp4_factory`

    The backward weight representation (``W.T``) is quantized to NVFP4 from
    the dequantized NVFP4 forward weight rather than directly from the original
    high-precision weight. In ``HybridQuantizer`` terms, this source choice is
    expressed with ``columnwise_source="rowwise_dequantized"``.

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
    return nvfp4_factory(role)


# -----------------------------------------------------------------------------
# RL-Oriented Recipes
# -----------------------------------------------------------------------------


def mxfp8_fwd_high_precision_bwd_factory(
    role: Optional[QuantizerRole],
):
    """Quantizer factory: MXFP8 forward, high-precision backward.

    This expresses the linear/grouped-linear equivalent of
    ``backward_override="dequantized"`` through per-direction quantizers:

        * ``input`` / ``weight`` ->
          ``Hybrid(rowwise=MXFP8, columnwise=Identity, columnwise_source="rowwise_dequantized")``
        * ``grad_output`` -> ``IdentityQuantizer``
        * everything else -> MXFP8

    The backward input and weight representations are high-precision values
    dequantized from the MXFP8 forward representations rather than the original
    high-precision tensors. In ``HybridQuantizer`` terms, this source choice is
    expressed with ``columnwise_source="rowwise_dequantized"``. The gradient
    output independently remains in high precision.

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
            rowwise_quantizer=mxfp8_factory(role),
            columnwise_quantizer=IdentityQuantizer(),
            columnwise_source="rowwise_dequantized",
        )
    if is_linear and role.tensor_type == "grad_output":
        return IdentityQuantizer()
    return mxfp8_factory(role)


def nvfp4_row_scaled_fwd_mxfp8_bwd_factory(
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
          ``Hybrid(rowwise=row-scaled NVFP4, columnwise=MXFP8,
          columnwise_source="rowwise_dequantized")``
        * ``GroupedLinear`` ``weight`` ->
          ``Hybrid(rowwise=plain NVFP4, columnwise=MXFP8,
          columnwise_source="rowwise_dequantized")``
        * regular ``Linear`` -> MXFP8
        * ``grad_output`` -> MXFP8
        * everything else -> MXFP8

    Row-scaled NVFP4 is fprop-only, so the forward NVFP4 quantizers avoid RHT,
    stochastic rounding, and 2D scaling. The backward input and weight
    representations are quantized to MXFP8 from the dequantized NVFP4 forward
    representations rather than directly from the original high-precision
    tensors. In ``HybridQuantizer`` terms, this source choice is expressed with
    ``columnwise_source="rowwise_dequantized"``. To use the original tensors
    instead, use ``columnwise_source="original"``.

    Composer 2 Technical Report:
    https://arxiv.org/abs/2603.24477
    """
    from transformer_engine.pytorch.tensor.hybrid_tensor import HybridQuantizer

    is_grouped_linear = role is not None and role.module_type == "grouped_linear"
    is_linear = role is not None and role.module_type == "linear"
    if is_grouped_linear and role.tensor_type == "input":
        return HybridQuantizer(
            rowwise_quantizer=_plain_nvfp4_quantizer(row_scaled_nvfp4=True),
            columnwise_quantizer=mxfp8_factory(role),
            columnwise_source="rowwise_dequantized",
        )
    if is_grouped_linear and role.tensor_type == "weight":
        return HybridQuantizer(
            rowwise_quantizer=_plain_nvfp4_quantizer(),
            columnwise_quantizer=mxfp8_factory(role),
            columnwise_source="rowwise_dequantized",
        )
    if is_grouped_linear and role.tensor_type == "grad_output":
        return mxfp8_factory(role)
    if is_linear:
        return mxfp8_factory(role)
    return mxfp8_factory(role)


def nvfp4_row_scaled_fwd_high_precision_bwd_factory(
    role: Optional[QuantizerRole],
):
    """Quantizer factory: row-scaled NVFP4 forward, high-precision backward.

    This expresses a linear/grouped-linear variant of
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

    The backward input and weight representations are high-precision values
    dequantized from the NVFP4 forward representations rather than the original
    high-precision tensors. In ``HybridQuantizer`` terms, this source choice is
    expressed with ``columnwise_source="rowwise_dequantized"``. The gradient
    output independently remains in high precision.

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


# -----------------------------------------------------------------------------
# Linear + Attention Recipes
# -----------------------------------------------------------------------------


def nvfp4_linear_fp8_dpa_factory(
    role: Optional[QuantizerRole],
):
    """Quantizer factory: NVFP4 for ``Linear``, FP8 for ``DotProductAttention``.

    This factory demonstrates how to use ``CustomRecipe`` with ``fp8_dpa=True``
    to combine NVFP4 quantization for linear layers with FP8 attention.

    DPA-owned tensor types (``role.module_type == "dpa"``):

    =========== ============================================================
    tensor_type Description
    =========== ============================================================
    ``"qkv"``  Query, Key, Value inputs to the first attention GEMM
    ``"s"``    Softmax output (S = softmax(Q·K^T)), fed into the second GEMM
    ``"do"``   Gradient of the attention output (dO), backward input
    ``"dp"``   Gradient of the softmax output (dP = dO·V^T), backward
    =========== ============================================================

    Dispatch logic:
        * ``role.module_type == "dpa"`` with ``tensor_type in ("s", "dp")``
          -> FP8 delayed scaling (``Format.HYBRID``, most_recent, history length 1)
        * other DPA roles
          -> FP8 current scaling (``Format.HYBRID``: E4M3 fwd, E5M2 bwd)
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
        from transformer_engine.pytorch.custom_recipes.quantizer_factory_zoo import (
            nvfp4_linear_fp8_dpa_factory,
        )

        recipe = CustomRecipe(
            qfactory=nvfp4_linear_fp8_dpa_factory,
            fp8_dpa=True,
        )
        with autocast(recipe=recipe):
            output = model(input)
    """
    from transformer_engine.common.recipe import Format
    from transformer_engine.pytorch.quantization import DelayedScalingRequest
    from transformer_engine.pytorch.tensor.float8_tensor import Float8CurrentScalingQuantizer

    is_dpa = role is not None and role.module_type == "dpa"
    is_dpa_boundary = (
        role is not None
        and not role.module_type
        and ("dpa_output" in role.name or "dpa_grad_input" in role.name)
    )

    # Native NVFP4 + FP8 attention uses delayed scaling for S/dP.
    if is_dpa and role.tensor_type in ("s", "dp"):
        return DelayedScalingRequest(
            fp8_format=Format.HYBRID,
            amax_history_len=1,
            amax_compute_algo="most_recent",
            reduce_amax=True,
        )

    if is_dpa or is_dpa_boundary:
        is_bwd_role = (is_dpa and role.tensor_type in ("do", "dp", "dqkv")) or (
            is_dpa_boundary and "dpa_grad_input" in role.name
        )
        fp8_dtype = DType.kFloat8E5M2 if is_bwd_role else DType.kFloat8E4M3
        return Float8CurrentScalingQuantizer(fp8_dtype=fp8_dtype, device="cuda")

    return nvfp4_factory(role)
