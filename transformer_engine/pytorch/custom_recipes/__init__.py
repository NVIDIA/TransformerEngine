# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Experimental features and APIs."""

# Per-token NVFP4: per-row outer + 1x16 e4m3 inner SF; cuBLAS LT NVFP4 GEMM
# with operand amaxes pinned to 1.0 and a trailing row-amax post-scale.
# See quantization_nvfp4_per_token.py / gemm_nvfp4_per_token.py for the math.
from transformer_engine.pytorch.custom_recipes.quantization_nvfp4_per_token import (
    NVFP4QuantizerPerTokenRef,
    RefNVFP4TensorPerToken,
    nvfp4_per_token_amax,
    nvfp4_per_token_encode,
    nvfp4_per_token_quantize,
)
from transformer_engine.pytorch.custom_recipes.quantization_nvfp4_per_token_group import (
    nvfp4_per_token_group_quantize,
)
from transformer_engine.pytorch.custom_recipes.gemm_nvfp4_per_token import (
    dequantize_nvfp4_per_token,
    nvfp4_per_token_gemm,
    nvfp4_per_token_gemm_dequant,
)


__all__ = [
    "NVFP4QuantizerPerTokenRef",
    "RefNVFP4TensorPerToken",
    "nvfp4_per_token_quantize",
    "nvfp4_per_token_group_quantize",
    "nvfp4_per_token_amax",
    "nvfp4_per_token_encode",
    "dequantize_nvfp4_per_token",
    "nvfp4_per_token_gemm",
    "nvfp4_per_token_gemm_dequant",
]
