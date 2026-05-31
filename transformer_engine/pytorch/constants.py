# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Enums for e2e transformer"""
import enum
from types import SimpleNamespace
from typing import Union
import torch
import torch.distributed
import transformer_engine_torch as tex


class DType(enum.IntEnum):
    """Python mirror of ``transformer_engine_torch.DType`` (pybind11 enum).

    Members are constructed manually from the underlying pybind enum so
    that this class is the single source of truth for dtype tags used
    across ``transformer_engine.pytorch``. Using a Python ``IntEnum``
    avoids the per-access cost of looking up attributes on the pybind11
    enum class (which traverses C++ ``tp_getattro``) and reduces
    comparisons to plain ``int.__eq__``.
    """

    kByte = int(tex.DType.kByte)
    kInt32 = int(tex.DType.kInt32)
    kFloat32 = int(tex.DType.kFloat32)
    kFloat16 = int(tex.DType.kFloat16)
    kBFloat16 = int(tex.DType.kBFloat16)
    kFloat8E4M3 = int(tex.DType.kFloat8E4M3)
    kFloat8E5M2 = int(tex.DType.kFloat8E5M2)
    kFloat4E2M1 = int(tex.DType.kFloat4E2M1)

    @classmethod
    def cast(cls, dtype: "DTypeLike") -> "DType":
        """Cast a dtype tag to the canonical ``DType`` ``IntEnum``.

        ``DType`` is the dtype tag used internally throughout
        ``transformer_engine.pytorch``. For backward compatibility, the public
        ``Quantizer`` / ``QuantizedTensor`` constructors also accept the pybind
        ``transformer_engine_torch.DType`` enum that external callers used
        historically; this converts it (or an existing ``DType``) to the
        matching ``DType`` member so stored attributes are always ``DType``.
        """
        if isinstance(dtype, cls):
            return dtype
        return cls(int(dtype))


# Fail fast at import time if a new enumerator is added
# on the C++ side without being mirrored above.
assert {m.name for m in DType} == set(tex.DType.__members__), (
    "DType is out of sync with transformer_engine_torch.DType; "
    "add the new pybind enumerator to DType in constants.py."
)


# Anything accepted by ``DType.cast`` as a stand-in for a ``DType`` member.
# ``DType`` is canonical internally; the pybind ``tex.DType`` enum is accepted
# at API boundaries for backward compatibility.
DTypeLike = Union[DType, tex.DType]


# One-to-one mapping ``torch.dtype -> DType`` (mirrors the enum order in
# ``transformer_engine.h``). Use the bracket syntax ``TE_DType[torch_dtype]``
# to resolve a ``torch.dtype`` to its matching ``DType`` member.
# Used for passing dtypes into cuda extension. 
TE_DType = {
    torch.uint8: DType.kByte,
    torch.float8_e4m3fn: DType.kFloat8E4M3,
    torch.float8_e5m2: DType.kFloat8E5M2,
    torch.int32: DType.kInt32,
    torch.float32: DType.kFloat32,
    torch.half: DType.kFloat16,
    torch.bfloat16: DType.kBFloat16,
}


# Map ``DType -> torch.dtype`` for resolving cuda extension types to
# torch.
TE_DType_To_Torch = {
    DType.kByte: torch.uint8,
    DType.kFloat8E4M3: torch.float8_e4m3fn,
    DType.kFloat8E5M2: torch.float8_e5m2,
    DType.kInt32: torch.int32,
    DType.kFloat32: torch.float32,
    DType.kFloat16: torch.half,
    DType.kBFloat16: torch.bfloat16,
}

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
