# Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Python interface for c++ extensions"""
from transformer_engine_extensions import *

from .fused_attn import *
from .gemm import *
from .transpose import *
from .activation import *
from .normalization import *
from .cast import *

__all__ = [
    # Activations
    "gelu",
    "relu",
    "geglu",
    "reglu",
    "swiglu",
    # Casts
    "cast_from_fp8",
    "cast_to_fp8",
    # Gemm
    "gemm",
    "fp8_gemm",
    # LayerNorm
    "layernorm_fwd_fp8",
    "layernorm_fwd_fp8_inf",
    "layernorm_fwd_inf",
    # Fused attention
    "fused_attn_fwd_qkvpacked",
    "fused_attn_bwd_qkvpacked",
    "fused_attn_fwd_kvpacked",
    "fused_attn_bwd_kvpacked",
    # Other fused kernels
    "fp8_cast_transpose_fused",
    "fp8_cast_transpose_bgrad_fused",
    "fp8_transpose_bgrad_fused",
    "fp8_cast_transpose_bgrad_dgelu_fused",
]
