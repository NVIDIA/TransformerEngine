# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Enums for e2e transformer"""

import enum
from types import SimpleNamespace
import torch
import torch.distributed
import transformer_engine_torch as tex
from transformer_engine.common.dtype import DType


class CPLoadBalancingStrategy(enum.Enum):
    """Token partition strategy for context-parallel attention.

    ``NO_LOAD_BALANCE`` is experimental.
    """

    DUAL_CHUNK_SWAP = "dual_chunk_swap"
    NO_LOAD_BALANCE = "no_load_balance"


# One-to-one mapping ``torch.dtype -> DType`` (mirrors the enum order in
# ``transformer_engine.h``). Use the bracket syntax ``TE_DType[torch_dtype]``
# to resolve a ``torch.dtype`` to its matching ``DType`` member.
# Used for passing dtypes into cuda extension.
TE_DType = {
    torch.uint8: DType.kByte,
    torch.float8_e4m3fn: DType.kFloat8E4M3,
    torch.float8_e5m2: DType.kFloat8E5M2,
    torch.int16: DType.kInt16,
    torch.int32: DType.kInt32,
    torch.int64: DType.kInt64,
    torch.float32: DType.kFloat32,
    torch.half: DType.kFloat16,
    torch.bfloat16: DType.kBFloat16,
}


# Map ``DType -> torch.dtype`` for resolving cuda extension types to
# torch.
TE_DType_To_Torch = {value: key for key, value in TE_DType.items()}

# Cache enum -> int conversions to avoid repeated PyObject lookups.
FP8FwdTensorIdx = SimpleNamespace(
    GEMM1_INPUT=int(tex.FP8FwdTensors.GEMM1_INPUT),
    GEMM1_WEIGHT=int(tex.FP8FwdTensors.GEMM1_WEIGHT),
    GEMM1_OUTPUT=int(tex.FP8FwdTensors.GEMM1_OUTPUT),
    GEMM2_INPUT=int(tex.FP8FwdTensors.GEMM2_INPUT),
    GEMM2_WEIGHT=int(tex.FP8FwdTensors.GEMM2_WEIGHT),
    GEMM2_OUTPUT=int(tex.FP8FwdTensors.GEMM2_OUTPUT),
    GEMM3_OUTPUT=int(tex.FP8FwdTensors.GEMM3_OUTPUT),
)
FP8BwdTensorIdx = SimpleNamespace(
    GRAD_INPUT1=int(tex.FP8BwdTensors.GRAD_INPUT1),
    GRAD_INPUT2=int(tex.FP8BwdTensors.GRAD_INPUT2),
    GRAD_INPUT3=int(tex.FP8BwdTensors.GRAD_INPUT3),
    GRAD_OUTPUT1=int(tex.FP8BwdTensors.GRAD_OUTPUT1),
    GRAD_OUTPUT2=int(tex.FP8BwdTensors.GRAD_OUTPUT2),
    GRAD_OUTPUT3=int(tex.FP8BwdTensors.GRAD_OUTPUT3),
)

AttnMaskTypes = (
    "no_mask",
    "padding",
    "causal",
    "padding_causal",
    "causal_bottom_right",
    "padding_causal_bottom_right",
    "arbitrary",
)

AttnTypes = ("self", "cross")

AttnBiasTypes = ("pre_scale_bias", "post_scale_bias", "no_bias", "alibi")

QKVLayouts = (
    "sb3hd",
    "sbh3d",
    "sbhd_sb2hd",
    "sbhd_sbh2d",
    "sbhd_sbhd_sbhd",
    "bs3hd",
    "bsh3d",
    "bshd_bs2hd",
    "bshd_bsh2d",
    "bshd_bshd_bshd",
    "t3hd",
    "th3d",
    "thd_t2hd",
    "thd_th2d",
    "thd_thd_thd",
    "sbhd_bshd_bshd",
    "bshd_sbhd_sbhd",
    "thd_bshd_bshd",
    "thd_sbhd_sbhd",
    "paged_kv_bshd_bshd_bshd",
    "paged_kv_bshd_sbhd_sbhd",
    "paged_kv_sbhd_bshd_bshd",
    "paged_kv_sbhd_sbhd_sbhd",
    "paged_kv_thd_bshd_bshd",
    "paged_kv_thd_sbhd_sbhd",
)

LayerTypes = ("encoder", "decoder")

GemmParallelModes = ("row", "column", None)

dist_group_type = torch.distributed.ProcessGroup

MXFP8_BLOCK_SCALING_SIZE = 32

NVFP4_BLOCK_SCALING_SIZE = 16
