# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Shared utility functions for FSDP2 distributed tests."""

import transformer_engine.common.recipe
from transformer_engine.pytorch import HybridQuantizer, QuantizedTensor
from transformer_engine.pytorch.custom_recipes.quantization_recipes_base import (
    current_scaling_quantizer_factory,
    float8_block_scaling_quantizer_factory,
    mxfp8_quantizer_factory,
)


def get_recipe_from_string(recipe):
    return getattr(transformer_engine.common.recipe, recipe)()


# ── Hybrid qfactories ─────────────────────────────────────────────────
#
# Module-level (picklable) qfactories used by ``get_hybrid_recipe_from_string``.
# Each factory composes one or two role-aware base factories from
# ``quantization_recipes_base`` per direction. Per-role behavior is delegated
# to the base factory — the hybrid layer only decides direction pairing.
#
# DCP serializes ``CustomRecipe`` via ``pickle``; closure-based qfactories
# (lambdas, inner functions referencing captured state) are not picklable,
# so the qfactory must live at module scope. See
# ``run_fsdp2_fused_adam.py::test_hybrid_dcp_output_parity``.


def _hybrid_fp8_current_qfactory(role):
    """FP8 current-scaling rowwise + FP8 current-scaling columnwise."""
    is_linear = role is not None and role.module_type in ("linear", "grouped_linear")
    if is_linear and role.tensor_type in ("input", "weight", "output"):
        return HybridQuantizer(
            rowwise_quantizer=current_scaling_quantizer_factory(role),
            columnwise_quantizer=current_scaling_quantizer_factory(role),
        )
    return current_scaling_quantizer_factory(role)


def _hybrid_mxfp8_qfactory(role):
    """MXFP8 rowwise + MXFP8 columnwise."""
    is_linear = role is not None and role.module_type in ("linear", "grouped_linear")
    if is_linear and role.tensor_type in ("input", "weight", "output"):
        return HybridQuantizer(
            rowwise_quantizer=mxfp8_quantizer_factory(role),
            columnwise_quantizer=mxfp8_quantizer_factory(role),
        )
    return mxfp8_quantizer_factory(role)


def _hybrid_float8_block_qfactory(role):
    """Float8 block-scaling rowwise + Float8 block-scaling columnwise."""
    is_linear = role is not None and role.module_type in ("linear", "grouped_linear")
    if is_linear and role.tensor_type in ("input", "weight", "output"):
        return HybridQuantizer(
            rowwise_quantizer=float8_block_scaling_quantizer_factory(role),
            columnwise_quantizer=float8_block_scaling_quantizer_factory(role),
        )
    return float8_block_scaling_quantizer_factory(role)


def _hybrid_mixed_mxfp8_fp8_qfactory(role):
    """MXFP8 rowwise + FP8 current columnwise (cross-format hybrid)."""
    is_linear = role is not None and role.module_type in ("linear", "grouped_linear")
    if is_linear and role.tensor_type in ("input", "weight", "output"):
        return HybridQuantizer(
            rowwise_quantizer=mxfp8_quantizer_factory(role),
            columnwise_quantizer=current_scaling_quantizer_factory(role),
        )
    return current_scaling_quantizer_factory(role)


_HYBRID_QFACTORIES = {
    "HybridFP8CurrentScaling": _hybrid_fp8_current_qfactory,
    "HybridMXFP8": _hybrid_mxfp8_qfactory,
    "HybridFloat8BlockScaling": _hybrid_float8_block_qfactory,
    "HybridMixed_MXFP8_FP8": _hybrid_mixed_mxfp8_fp8_qfactory,
}


def get_hybrid_recipe_from_string(recipe):
    """Build a CustomRecipe wrapping a module-level (picklable) hybrid qfactory.

    Supported values:
        "HybridFP8CurrentScaling" — FP8 current for both directions
        "HybridMXFP8"             — MXFP8 for both directions
        "HybridFloat8BlockScaling" — Float8 block scaling for both directions
        "HybridMixed_MXFP8_FP8"   — MXFP8 rowwise + FP8 current columnwise
    """
    if recipe not in _HYBRID_QFACTORIES:
        raise ValueError(
            f"Unknown hybrid recipe '{recipe}'. Supported: {sorted(_HYBRID_QFACTORIES.keys())}"
        )
    return transformer_engine.common.recipe.CustomRecipe(qfactory=_HYBRID_QFACTORIES[recipe])


def save_custom_attrs(module):
    custom_attrs = {}
    for name, param in module.named_parameters():
        if isinstance(param, QuantizedTensor):
            ignore_keys = [key for key in param.__dict__.keys() if key.startswith("_")]
        else:
            ignore_keys = []
        attrs = vars(param)
        custom_attrs[name] = {k: v for k, v in attrs.items() if k not in ignore_keys}
    return custom_attrs


def restore_custom_attrs(module, custom_attrs):
    for name, param in module.named_parameters():
        if name in custom_attrs:
            for attr_name, attr_value in custom_attrs[name].items():
                setattr(param, attr_name, attr_value)
