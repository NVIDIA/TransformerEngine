# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

# START_MIXED_PRECISION_EXAMPLE
import torch
import transformer_engine.pytorch as te
from transformer_engine.common import recipe
from transformer_engine.pytorch import Float8CurrentScalingQuantizer, MXFP8Quantizer
import transformer_engine_torch as tex


def mixed_recipe_factory(role: str):
    """
    Mixed precision factory: FP8 current scaling for forward, MXFP8 for backward.

    Note: Both Float8CurrentScalingQuantizer and MXFP8Quantizer return standard
    TE tensor types (Float8Tensor, MXFP8Tensor), not custom tensors. This means
    the optimized C++ GEMM kernels will be used - custom_gemm is NOT invoked.

    This approach allows mixing different standard TE quantization strategies
    without performance penalty.
    """
    # Forward pass: Use FP8 current scaling (E4M3)
    if role in ("linear_input", "linear_weight", "linear_output"):
        return Float8CurrentScalingQuantizer(tex.DType.kFloat8E4M3)

    # Backward pass: Use MXFP8 block scaling
    if role in ("linear_grad_output", "linear_grad_input"):
        return MXFP8Quantizer()

    return Float8CurrentScalingQuantizer(tex.DType.kFloat8E4M3)


custom_recipe = recipe.CustomRecipe(qfactory=mixed_recipe_factory)

# Example usage:
model = te.Linear(64, 64).cuda()
x = torch.randn(32, 64, device="cuda").requires_grad_(True)

with te.autocast(enabled=True, recipe=custom_recipe):
    y = model(x)

loss = y.sum()
loss.backward()
# END_MIXED_PRECISION_EXAMPLE
