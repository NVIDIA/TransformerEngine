# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Compatibility imports for the Python cuDNN attention implementation."""

from enum import IntEnum

import torch
from transformer_engine_torch import NVTE_QKV_Format

from ..constants import DType, FP8BwdTensorIdx, FP8FwdTensorIdx

__all__ = ["fused_attn_fwd", "fused_attn_bwd"]


# Retained for rotary-position and custom-op compatibility. These operations
# still use TE's format enum even though attention execution itself does not.
TORCH_DType = {
    DType.kFloat8E4M3: torch.uint8,
    DType.kFloat8E5M2: torch.uint8,
    DType.kFloat16: torch.half,
    DType.kBFloat16: torch.bfloat16,
    DType.kFloat32: torch.float32,
    DType.kInt32: torch.int32,
}

QKVFormat = {
    None: NVTE_QKV_Format.NVTE_QKV_Format_NOT_SET,
    "bshd": NVTE_QKV_Format.NVTE_BSHD,
    "sbhd": NVTE_QKV_Format.NVTE_SBHD,
    "thd": NVTE_QKV_Format.NVTE_THD,
    "sbhd_2bshd": NVTE_QKV_Format.NVTE_SBHD_2BSHD,
    "bshd_2sbhd": NVTE_QKV_Format.NVTE_BSHD_2SBHD,
    "thd_2bshd": NVTE_QKV_Format.NVTE_THD_2BSHD,
    "thd_2sbhd": NVTE_QKV_Format.NVTE_THD_2SBHD,
    "bhsd": NVTE_QKV_Format.NVTE_BHSD,
}


class FusedAttnBackend(IntEnum):
    """Legacy import-path mirror of the Python cuDNN attention backend enum."""

    No_Backend = -1
    F16_arbitrary_seqlen = 1
    FP8 = 2

    @classmethod
    def cast(cls, backend):
        """Convert an integer or another compatible enum to this enum."""
        if isinstance(backend, cls):
            return backend
        return cls(int(backend))


META_QKV = FP8FwdTensorIdx.GEMM1_OUTPUT
META_DQKV = FP8BwdTensorIdx.GRAD_OUTPUT1
META_O = FP8FwdTensorIdx.GEMM2_INPUT
META_DO = FP8BwdTensorIdx.GRAD_INPUT2
META_S = FP8FwdTensorIdx.GEMM3_OUTPUT
META_DP = FP8BwdTensorIdx.GRAD_INPUT3


# Resolve lazily because cpp_extensions is imported while transformer_engine.pytorch
# itself is still initializing.
def fused_attn_fwd(*args, **kwargs):
    """Run fused attention forward through the cuDNN frontend Python API."""
    from transformer_engine.pytorch.attention.dot_product_attention.cudnn_attention import (
        fused_attn_fwd as implementation,
    )

    return implementation(*args, **kwargs)


def fused_attn_bwd(*args, **kwargs):
    """Run fused attention backward through the cuDNN frontend Python API."""
    from transformer_engine.pytorch.attention.dot_product_attention.cudnn_attention import (
        fused_attn_bwd as implementation,
    )

    return implementation(*args, **kwargs)
