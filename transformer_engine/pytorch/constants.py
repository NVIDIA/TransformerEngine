# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Enums for e2e transformer"""
import enum
from types import SimpleNamespace
import torch
import torch.distributed
import transformer_engine_torch as tex


class _TE_DTypeMeta(enum.EnumMeta):
    """Metaclass that extends ``cls[key]`` / ``key in cls`` on ``TE_DType``.

    - ``TE_DType[torch.dtype]`` returns the matching ``TE_DType`` member
      (replaces the old ``TORCH_DTYPE_TO_TE_DTYPE[dtype]`` pattern).
    - ``torch.dtype in TE_DType`` reports whether a mapping exists.
    - Anything that is not a ``torch.dtype`` (most notably member-name
      strings) falls through to the standard ``EnumMeta`` behavior, so
      ``TE_DType["kFloat32"]`` and ``TE_DType.kFloat32 in TE_DType``
      keep working exactly as before.
    """

    def __getitem__(cls, key):
        if isinstance(key, torch.dtype):
            return _TORCH_DTYPE_TO_TE_DTYPE[key]
        return super().__getitem__(key)

    def __contains__(cls, key):
        if isinstance(key, torch.dtype):
            return key in _TORCH_DTYPE_TO_TE_DTYPE
        return super().__contains__(key)


class TE_DType(enum.IntEnum, metaclass=_TE_DTypeMeta):
    """Python mirror of ``transformer_engine_torch.DType`` (pybind11 enum).

    Members are constructed manually from the underlying pybind enum so
    that this class is the single source of truth for dtype tags used
    across ``transformer_engine.pytorch``. Using a Python ``IntEnum``
    avoids the per-access cost of looking up attributes on the pybind11
    enum class (which traverses C++ ``tp_getattro``) and reduces
    comparisons to plain ``int.__eq__``.

    The custom metaclass adds dict-like lookup by ``torch.dtype``:
    ``TE_DType[torch.float32] is TE_DType.kFloat32``.
    """

    kByte = int(tex.DType.kByte)
    kInt32 = int(tex.DType.kInt32)
    kFloat32 = int(tex.DType.kFloat32)
    kFloat16 = int(tex.DType.kFloat16)
    kBFloat16 = int(tex.DType.kBFloat16)
    kFloat8E4M3 = int(tex.DType.kFloat8E4M3)
    kFloat8E5M2 = int(tex.DType.kFloat8E5M2)
    kFloat4E2M1 = int(tex.DType.kFloat4E2M1)


# Fail fast at import time if a new enumerator is added
# on the C++ side without being mirrored above.
assert {m.name for m in TE_DType} == set(tex.DType.__members__), (
    "TE_DType is out of sync with transformer_engine_torch.DType; "
    "add the new pybind enumerator to TE_DType in constants.py."
)


# Private one-to-one mapping ``torch.dtype -> TE_DType`` (mirrors the
# enum order in ``transformer_engine.h``). The metaclass above forwards
# ``TE_DType[torch_dtype]`` to this dict, so callers should use the
# bracket syntax on ``TE_DType`` rather than importing this directly.
_TORCH_DTYPE_TO_TE_DTYPE = {
    torch.uint8: TE_DType.kByte,
    torch.float8_e4m3fn: TE_DType.kFloat8E4M3,
    torch.float8_e5m2: TE_DType.kFloat8E5M2,
    torch.int32: TE_DType.kInt32,
    torch.float32: TE_DType.kFloat32,
    torch.half: TE_DType.kFloat16,
    torch.bfloat16: TE_DType.kBFloat16,
}


# Map ``TE_DType -> torch.dtype`` for resolving cuda extension types to
# torch. One-to-one with the enum in ``transformer_engine.h``.
#
# C++ sites that stamp dtype tags onto Python tensors (e.g. ``_fp8_dtype``,
# ``_fp4_dtype``) route through the ``MakeTEDType`` helper in
# ``transformer_engine/pytorch/csrc/common.{h,cpp}``, so every key we
# look up here is guaranteed to be a ``TE_DType`` member. Keep this dict
# keyed by ``TE_DType`` (not ``int``) so accidental mixing with the
# pybind11 ``tex.DType`` enum surfaces as a ``KeyError`` instead of
# silently succeeding.
TE_DType_To_Torch = {
    TE_DType.kByte: torch.uint8,
    TE_DType.kFloat8E4M3: torch.float8_e4m3fn,
    TE_DType.kFloat8E5M2: torch.float8_e5m2,
    TE_DType.kInt32: torch.int32,
    TE_DType.kFloat32: torch.float32,
    TE_DType.kFloat16: torch.half,
    TE_DType.kBFloat16: torch.bfloat16,
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
