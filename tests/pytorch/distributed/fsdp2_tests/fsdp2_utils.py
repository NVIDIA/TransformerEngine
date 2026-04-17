# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Shared utility functions for FSDP2 distributed tests."""

import transformer_engine.common.recipe
from transformer_engine.pytorch import QuantizedTensor


def get_recipe_from_string(recipe):
    return getattr(transformer_engine.common.recipe, recipe)()


def get_hybrid_recipe_from_string(recipe):
    """Build a CustomRecipe that uses HybridQuantizer with the given base format.

    Supported values:
        "HybridFP8CurrentScaling" — FP8 current for both directions
        "HybridMXFP8"            — MXFP8 for both directions
        "HybridMixed_MXFP8_FP8"  — MXFP8 rowwise + FP8 current columnwise
    """
    import transformer_engine_torch as tex
    from transformer_engine.pytorch import (
        Float8CurrentScalingQuantizer,
        Float8BlockQuantizer,
        MXFP8Quantizer,
        HybridQuantizer,
    )

    _BUILDERS = {
        "HybridFP8CurrentScaling": lambda: dict(
            row=lambda: Float8CurrentScalingQuantizer(tex.DType.kFloat8E4M3, device="cuda"),
            col=lambda: Float8CurrentScalingQuantizer(tex.DType.kFloat8E4M3, device="cuda"),
            grad=lambda: Float8CurrentScalingQuantizer(tex.DType.kFloat8E5M2, device="cuda"),
        ),
        "HybridMXFP8": lambda: dict(
            row=lambda: MXFP8Quantizer(fp8_dtype=tex.DType.kFloat8E4M3),
            col=lambda: MXFP8Quantizer(fp8_dtype=tex.DType.kFloat8E4M3),
            grad=lambda: MXFP8Quantizer(fp8_dtype=tex.DType.kFloat8E5M2),
        ),
        "HybridFloat8BlockScaling": lambda: dict(
            row=lambda: Float8BlockQuantizer(fp8_dtype=tex.DType.kFloat8E4M3, rowwise=True, columnwise=True),
            col=lambda: Float8BlockQuantizer(fp8_dtype=tex.DType.kFloat8E4M3, rowwise=True, columnwise=True),
            grad=lambda: Float8BlockQuantizer(fp8_dtype=tex.DType.kFloat8E5M2, rowwise=True, columnwise=True),
        ),
        "HybridMixed_MXFP8_FP8": lambda: dict(
            row=lambda: MXFP8Quantizer(fp8_dtype=tex.DType.kFloat8E4M3),
            col=lambda: Float8CurrentScalingQuantizer(tex.DType.kFloat8E4M3, device="cuda"),
            grad=lambda: Float8CurrentScalingQuantizer(tex.DType.kFloat8E5M2, device="cuda"),
        ),
    }

    if recipe not in _BUILDERS:
        raise ValueError(
            f"Unknown hybrid recipe '{recipe}'. Supported: {sorted(_BUILDERS.keys())}"
        )

    builders = _BUILDERS[recipe]()
    row_fn, col_fn, grad_fn = builders["row"], builders["col"], builders["grad"]

    def qfactory(role):
        if role in ("linear_input", "linear_weight", "linear_output"):
            return HybridQuantizer(
                rowwise_quantizer=row_fn(),
                columnwise_quantizer=col_fn(),
            )
        if role in ("linear_grad_output", "linear_grad_input"):
            return grad_fn()
        return row_fn()

    return transformer_engine.common.recipe.CustomRecipe(qfactory=qfactory)


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
