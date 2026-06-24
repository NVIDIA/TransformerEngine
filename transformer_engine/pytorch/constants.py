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
    """Transformer Engine data types used to tag tensors passed to the
    Transformer Engine backend.
    This is the canonical dtype enum for ``transformer_engine.pytorch`` and
    is used throughout the library (for example, to specify the precision of
    quantized tensors and quantizers). Each member corresponds to a data type
    supported by the Transformer Engine backend:

    * ``kByte`` -- 8-bit unsigned integer (``torch.uint8``).
    * ``kInt32`` -- 32-bit signed integer (``torch.int32``).
    * ``kFloat32`` -- 32-bit floating point (``torch.float32``).
    * ``kFloat16`` -- 16-bit floating point (``torch.float16``).
    * ``kBFloat16`` -- 16-bit brain floating point (``torch.bfloat16``).
    * ``kFloat8E4M3`` -- 8-bit floating point with 4 exponent and 3 mantissa
      bits (``torch.float8_e4m3fn``).
    * ``kFloat8E5M2`` -- 8-bit floating point with 5 exponent and 2 mantissa
      bits (``torch.float8_e5m2``).
    * ``kFloat4E2M1`` -- 4-bit floating point with 2 exponent and 1 mantissa
      bits.

    The enum mirrors the backend ``transformer_engine_torch.DType`` (pybind11)
    enum value-for-value, and instances of the two enums compare equal when
    they share the same integer value.
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
    def cast(cls, dtype: "Union[DType, tex.DType]") -> "DType":
        """Normalize a supported dtype value to the canonical ``DType`` ``IntEnum``.
        ``DType`` is the canonical dtype tag used internally throughout
        ``transformer_engine.pytorch``, and is what this function always outputs.
        The pybind ``transformer_engine_torch.DType`` enum is an additional type
        accepted as input (for backward compatibility), which this function maps
        to the matching ``DType`` member so stored attributes are always ``DType``.
        """
        if isinstance(dtype, cls):
            return dtype
        return cls(int(dtype))

    def __eq__(self, other: object) -> bool:
        # ``DType`` is an ``IntEnum`` while ``tex.DType`` is a pybind11 enum.
        # ``int.__eq__`` returns ``NotImplemented`` for a pybind enum, so without
        # this override a comparison such as ``quantizer.dtype == tex.DType.kX``
        # would silently be ``False``. Compare by integer value so the two enums
        # stay equivalent (the pybind ``DType.__eq__`` handles the reverse order).
        if isinstance(other, tex.DType):
            return int(self) == int(other)
        return int.__eq__(self, other)

    def __ne__(self, other: object) -> bool:
        result = self.__eq__(other)
        if result is NotImplemented:
            return result
        return not result

    def __hash__(self) -> int:
        return int.__hash__(self)


# Fail fast at import time if a new enumerator is added
# on the C++ side without being mirrored above.
assert {m.name for m in DType} == set(tex.DType.__members__), (
    "DType in python is out of sync with transformer_engine_torch.DType; "
    "defined in C++ side. Please make sure TE C++ and python are in sync."
)


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
