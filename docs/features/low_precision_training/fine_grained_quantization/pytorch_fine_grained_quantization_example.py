#!/usr/bin/env python3
# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Runnable fine-grained quantization recipe example.

The factory assigns one precision to each ``demo.fc1`` Linear GEMM:

* fprop: ``weight.row(MXFP8) x input.row(MXFP8)``
* dgrad: ``weight.col(NVFP4) x grad_output.row(NVFP4)``
* wgrad: ``input.col(MXFP8-dequantized BF16) x grad_output.col(original BF16)``

``demo.fc2`` runs every GEMM in high precision. ``demo.output`` is not
special-cased and therefore exercises the MXFP8 base-factory fallback.

Run from the Transformer Engine repository root::

    python docs/features/low_precision_training/fine_grained_quantization/\
        pytorch_fine_grained_quantization_example.py
"""

# START_FINE_GRAINED_QUANTIZATION_EXAMPLE

from __future__ import annotations

from typing import Optional

import torch

import transformer_engine.pytorch as te
from transformer_engine.common.recipe import CustomRecipe
from transformer_engine.pytorch.custom_recipes.quantizer_factories import (
    mxfp8_factory,
    nvfp4_factory,
)


THREE_FORMAT_MODULE = "demo.fc1"
HIGH_PRECISION_MODULE = "demo.fc2"
BASE_FACTORY = mxfp8_factory


def quantizer_factory(role: Optional[te.QuantizerRole]):
    """Return a fresh quantizer for every role, including ``None``.

    ``BASE_FACTORY`` makes the factory total: unknown roles, future role values,
    and untargeted modules all retain valid MXFP8 behavior.
    """

    if role is not None and role.name == THREE_FORMAT_MODULE:
        # Constructing fresh child quantizers for every call is recommended.
        if role.tensor_type == "input":
            # Wgrad retains the original BF16 input.
            return te.HybridQuantizer(
                rowwise_quantizer=mxfp8_factory(role),
                columnwise_quantizer=te.IdentityQuantizer(),
                columnwise_source="original",
            )
        if role.tensor_type == "weight":
            # Dgrad uses NVFP4 quantized from the dequantized MXFP8 fprop weight.
            return te.HybridQuantizer(
                rowwise_quantizer=mxfp8_factory(role),
                columnwise_quantizer=nvfp4_factory(role),
                columnwise_source="rowwise_dequantized",
            )
        if role.tensor_type == "grad_output":
            # Dgrad uses NVFP4 while wgrad retains the original BF16 gradient.
            return te.HybridQuantizer(
                rowwise_quantizer=nvfp4_factory(role),
                columnwise_quantizer=te.IdentityQuantizer(),
                columnwise_source="original",
            )

    if role is not None and role.name == HIGH_PRECISION_MODULE:
        return te.IdentityQuantizer()

    return BASE_FACTORY(role)


def require_supported_hardware() -> None:
    """Fail early with TE's reason when either required format is unavailable."""

    if not torch.cuda.is_available():
        raise SystemExit("This example requires a CUDA-capable NVIDIA GPU.")

    failures = []
    for name, check in (
        ("MXFP8", te.is_mxfp8_available),
        ("NVFP4", te.is_nvfp4_available),
    ):
        available, reason = check(return_reason=True)
        if not available:
            failures.append(f"{name}: {reason}")
    if failures:
        raise SystemExit("Required formats are unavailable: " + "; ".join(failures))


def build_model() -> torch.nn.Module:
    """Build aligned TE Linear layers with stable semantic names."""

    common = {
        "bias": False,
        "params_dtype": torch.bfloat16,
        "device": "cuda",
    }
    return torch.nn.Sequential(
        te.Linear(128, 256, name=THREE_FORMAT_MODULE, **common),
        torch.nn.GELU(),
        te.Linear(256, 256, name=HIGH_PRECISION_MODULE, **common),
        torch.nn.GELU(),
        te.Linear(256, 128, name="demo.output", **common),
    )


def main() -> None:
    """Run one training step through the custom recipe."""

    require_supported_hardware()
    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)

    model = build_model()
    recipe = CustomRecipe(qfactory=quantizer_factory)
    inputs = torch.randn(
        64,
        128,
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )

    with te.autocast(enabled=True, recipe=recipe):
        outputs = model(inputs)

    loss = outputs.float().square().mean()
    # Backward uses the quantizers selected and saved during the forward pass.
    loss.backward()

    gradients = [inputs.grad, *(parameter.grad for parameter in model.parameters())]
    assert all(gradient is not None for gradient in gradients)
    assert all(torch.isfinite(gradient).all() for gradient in gradients)

    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"TE Linear names: {[model[index].name for index in (0, 2, 4)]}")
    print(f"loss: {loss.item():.6f}; forward and backward completed")


if __name__ == "__main__":
    main()

# END_FINE_GRAINED_QUANTIZATION_EXAMPLE
