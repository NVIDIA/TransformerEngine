# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Dynamo-friendly mirror of ``transformer_engine_torch.DType``.

The C++-binded ``transformer_engine_torch.DType`` enum is opaque to
TorchDynamo (see ``UserDefinedObjectVariable(DType)`` graph-break under
``fullgraph=True``): Dynamo cannot proxy a pybind11 enum value as a
constant in the FX graph it builds for tensor-subclass constructors
(e.g. :class:`Float8Tensor`).

:class:`FP8DType` is a Python :class:`enum.IntEnum` that mirrors
``tex.DType`` 1:1 by integer value. Because :class:`IntEnum` derives
from :class:`int`, Dynamo recognises it as a ``ConstantVariable`` and
captures it as a static constant on subclass-constructor calls inside
a compiled region. Conversion to/from the C++ enum is one
``int(...)`` call.
"""
from __future__ import annotations
from enum import IntEnum

import transformer_engine_torch as tex


class FP8DType(IntEnum):
    """Python mirror of :class:`transformer_engine_torch.DType` (int values).

    Values match :class:`tex.DType` 1:1 so that ``int(FP8DType.x) ==
    int(tex.DType.x)`` for every member. Use :func:`to_tex` to bridge
    back to the C++ enum at pybind boundaries.
    """

    kByte = int(tex.DType.kByte)
    kInt32 = int(tex.DType.kInt32)
    kFloat32 = int(tex.DType.kFloat32)
    kFloat16 = int(tex.DType.kFloat16)
    kBFloat16 = int(tex.DType.kBFloat16)
    kFloat8E4M3 = int(tex.DType.kFloat8E4M3)
    kFloat8E5M2 = int(tex.DType.kFloat8E5M2)
    kFloat4E2M1 = int(tex.DType.kFloat4E2M1)


# Precomputed at module load so Dynamo doesn't have to trace
# ``IntEnum.__new__`` / ``tex.DType.__int__`` inside compiled regions
# (both recurse through Python's internal inspect machinery and exhaust
# Dynamo's frame stack).
_TEX_TO_FP8DTYPE = {member.value: member for member in FP8DType}
_TEX_TO_FP8DTYPE_BY_TEX = {tex.DType(v): m for v, m in _TEX_TO_FP8DTYPE.items()}


def to_tex(d) -> tex.DType:
    """Coerce ``d`` (``FP8DType`` / ``tex.DType`` / int) to ``tex.DType``."""
    if isinstance(d, tex.DType):
        return d
    return tex.DType(int(d))


def from_tex(d: tex.DType) -> FP8DType:
    """Coerce a ``tex.DType`` (or int matching one of its enum values) to
    :class:`FP8DType` via a precomputed lookup table.
    """
    if isinstance(d, FP8DType):
        return d
    if isinstance(d, tex.DType):
        return _TEX_TO_FP8DTYPE_BY_TEX[d]
    return _TEX_TO_FP8DTYPE[int(d)]


# Register ``tex.DType`` as a torch.compile value-opaque type so it
# can flow through Dynamo as a constant inside ``__tensor_flatten__``
# meta dicts and other traced metadata payloads. Without this,
# Dynamo trips on ``UserDefinedObjectVariable(DType)`` because the
# pybind11 enum carries a custom ``__hash__``. ``__fx_repr__`` is
# injected once here so the FX codegen can serialize literal values
# as ``TE_DType(<int>)``. Gated by a try/except so importing this
# module remains safe on older PyTorch versions that lack the
# private ``opaque_object`` API.
try:
    from torch._library.opaque_object import (
        is_opaque_value_type as _is_opaque_value_type,
        register_opaque_type as _register_opaque_type,
    )

    if not hasattr(tex.DType, "__fx_repr__"):
        tex.DType.__fx_repr__ = lambda self: (
            f"TE_DType({int(self)})",
            {"TE_DType": tex.DType},
        )
    if not _is_opaque_value_type(tex.DType):
        _register_opaque_type(tex.DType, typ="value", members={})
except Exception:  # pragma: no cover - older torch / partial init
    pass
