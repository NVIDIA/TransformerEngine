from __future__ import annotations

from ..utils import reinterpret_cast
from .. import cpp_extensions as _nvte
from ._common import torch_op

from .dtype import is_fp8
from .empty import empty, multi_empty_share_metadata


@torch_op
def _fp8_quantize(t: _nvte.Tensor, out_dtype: _nvte.DType) -> _nvte.Tensor:
    output = empty(t.shape, out_dtype)
    _nvte.fp8_quantize(t, output)
    return output


@torch_op
def _fp8_dequantize(t: _nvte.Tensor, out_dtype: _nvte.DType) -> _nvte.Tensor:
    output = empty(t.shape, out_dtype)
    _nvte.fp8_dequantize(t, output)
    return output


def cast(t: _nvte.Tensor, out_dtype: _nvte.DType):
    assert t.dtype != out_dtype
    if is_fp8(t):
        assert not is_fp8(out_dtype)

    if is_fp8(out_dtype):
        return _fp8_quantize(t, out_dtype)
    elif is_fp8(t):
        return _fp8_dequantize(t, out_dtype)
    else:
        output = empty(t.shape, out_dtype)
        output.data.copy_(t.data)
        return output


def cast_checked(t: _nvte.Tensor, out_dtype: _nvte.DType | None):
    if out_dtype is None or t.dtype == out_dtype:
        return t
    else:
        return cast(t, out_dtype)


@torch_op
def transpose(t: _nvte.Tensor) -> _nvte.Tensor:
    output = empty(t.shape[::-1], t.dtype)
    _nvte.transpose(t, output)
    return output


@torch_op
def cast_transpose(
    t: _nvte.Tensor, out_dtype: _nvte.DType
) -> tuple[_nvte.Tensor, _nvte.Tensor]:
    assert t.dtype != out_dtype
    if is_fp8(t):
        assert not is_fp8(out_dtype)

    out_cast, out_transpose = multi_empty_share_metadata(
        (t.shape, out_dtype), (t.shape[::-1], out_dtype)
    )

    _nvte.cast_transpose(t, out_cast, out_transpose)
    return out_cast, out_transpose


def cast_transpose_checked(t: _nvte.Tensor, out_dtype: _nvte.DType | None):
    if out_dtype is None or t.dtype == out_dtype:
        return t, transpose(t)
    else:
        return cast_transpose(t, out_dtype)


def multi_cast_transpose(
    *desc: tuple[_nvte.Tensor, _nvte.DType]
) -> list[tuple[_nvte.Tensor, _nvte.Tensor]]:
    outs = [
        multi_empty_share_metadata((t.shape, dtype), (t.shape[::-1], dtype))
        for t, dtype in desc
    ]
    out_cast_list, out_transpose_list = zip(*outs)
    input_list, _ = zip(*desc)
    input_list = reinterpret_cast(input_list, tuple[_nvte.Tensor, ...])
    _nvte.multi_cast_transpose(input_list, out_cast_list, out_transpose_list)
    return outs


def multi_cast_transpose_checked(*desc: tuple[_nvte.Tensor, _nvte.DType | None]):
    transpose_results = list[tuple[_nvte.Tensor, _nvte.Tensor] | None]()
    to_cast_transpose = list[tuple[_nvte.Tensor, _nvte.DType]]()
    for t, dtype in desc:
        if dtype is None or t.dtype == dtype:
            transpose_results.append((t, transpose(t)))
        else:
            to_cast_transpose.append((t, dtype))
            transpose_results.append(None)
    cast_transpose_results = (
        multi_cast_transpose(*to_cast_transpose) if to_cast_transpose else []
    )
    results = list[tuple[_nvte.Tensor, _nvte.Tensor]]()
    i = 0
    for result in transpose_results:
        if result is None:
            results.append(cast_transpose_results[i])
            i += 1
        else:
            results.append(result)
    return results
