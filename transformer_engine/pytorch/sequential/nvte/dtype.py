from __future__ import annotations
from .. import cpp_extensions as _nvte
from ..cpp_extensions import te_to_torch_dtype, torch_to_te_dtype, dtype_name, bit_width


def is_fp8(t: _nvte.Tensor | _nvte.DType):
    if isinstance(t, _nvte.DType):
        dtype = t
    else:
        dtype = t.dtype
    return dtype is _nvte.DType.Float8E4M3 or dtype is _nvte.DType.Float8E5M2


__all__ = [
    "is_fp8",
    "te_to_torch_dtype",
    "torch_to_te_dtype",
    "dtype_name",
    "bit_width",
]
