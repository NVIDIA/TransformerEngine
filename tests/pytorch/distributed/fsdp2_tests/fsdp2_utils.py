# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Shared utility functions for FSDP2 distributed tests."""

import transformer_engine.common.recipe
from transformer_engine.pytorch import QuantizedTensor

from hybrid_quantization_utils import (
    hybrid_float8_block_qfactory,
    hybrid_fp8_current_identity_qfactory,
    hybrid_fp8_current_qfactory,
    hybrid_mixed_mxfp8_fp8_qfactory,
    hybrid_mxfp8_qfactory,
    identity_qfactory,
)


def get_recipe_from_string(recipe):
    return getattr(transformer_engine.common.recipe, recipe)()


# CustomRecipe has dynamic TE extra-state handling. Once FP8 state is
# initialized, TE's get_extra_state() pickles the recipe on save, so
# checkpoint-test qfactories must be module-level and picklable. On load,
# payloads without delayed-scaling state are identified and ignored without
# unpickling. See ``run_fsdp2_fused_adam.py::test_hybrid_dcp_output_parity``.
_HYBRID_QFACTORIES = {
    "HybridFP8CurrentScaling": hybrid_fp8_current_qfactory,
    "HybridMXFP8": hybrid_mxfp8_qfactory,
    "HybridFloat8BlockScaling": hybrid_float8_block_qfactory,
    "HybridMixed_MXFP8_FP8": hybrid_mixed_mxfp8_fp8_qfactory,
    "HybridFP8CurrentScalingIdentity": hybrid_fp8_current_identity_qfactory,
    "Identity": identity_qfactory,
}


def get_hybrid_recipe_from_string(recipe):
    """Build a CustomRecipe wrapping a module-level (picklable) hybrid qfactory.

    Each hybrid qfactory composes one or two role-aware base factories from
    ``quantization_factory_base`` per direction; per-role behavior is delegated
    to the base factory and the hybrid layer only decides the direction pairing.

    Supported values:
        "HybridFP8CurrentScaling" — FP8 current for both directions
        "HybridMXFP8"             — MXFP8 for both directions
        "HybridFloat8BlockScaling" — Float8 block scaling for both directions
        "HybridMixed_MXFP8_FP8"   — MXFP8 rowwise + FP8 current columnwise
        "HybridFP8CurrentScalingIdentity" — FP8 current forward + Identity backward
        "Identity" — high-precision passthrough for every slot
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
