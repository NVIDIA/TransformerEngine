from . import _nvte
from .dtype import is_fp8
from .empty import empty, multi_empty_share_metadata


def cast(t: _nvte.Tensor, dtype: _nvte.DType):
    assert t.dtype != dtype
    if is_fp8(t):
        assert not is_fp8(dtype)

    output = empty(t.shape, dtype)
    if is_fp8(dtype):
        _nvte.fp8_quantize(t, output)
    elif is_fp8(t):
        _nvte.fp8_dequantize(t, output)
    else:
        output.data.copy_(t.data)

    return output


def cast_checked(t: _nvte.Tensor, dtype: _nvte.DType | None):
    if dtype is None or t.dtype == dtype:
        return t
    else:
        return cast(t, dtype)


def transpose(t: _nvte.Tensor):
    output = empty(t.shape[::-1], t.dtype)
    _nvte.transpose(t, output)
    return output


def cast_transpose(t: _nvte.Tensor, dtype: _nvte.DType):
    assert t.dtype != dtype
    if is_fp8(t):
        assert not is_fp8(dtype)

    out_cast, out_transpose = multi_empty_share_metadata(
        (t.shape, dtype), (t.shape[::-1], dtype)
    )

    _nvte.cast_transpose(t, out_cast, out_transpose)
    return out_cast, out_transpose


def cast_transpose_checked(t: _nvte.Tensor, dtype: _nvte.DType | None):
    if dtype is None or t.dtype == dtype:
        return t, transpose(t)
    else:
        return cast_transpose(t, dtype)


def multi_cast_transpose(*desc: tuple[_nvte.Tensor, _nvte.DType]):
    outs = [
        multi_empty_share_metadata((t.shape, dtype), (t.shape[::-1], dtype))
        for t, dtype in desc
    ]
    out_cast_list, out_transpose_list = zip(*outs)
    input_list, _ = zip(*desc)
    _nvte.multi_cast_transpose(input_list, out_cast_list, out_transpose_list)  # type: ignore
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
    cast_transpose_results = multi_cast_transpose(*to_cast_transpose)
    results = list[tuple[_nvte.Tensor, _nvte.Tensor]]()
    i = 0
    for result in transpose_results:
        if result is None:
            results.append(cast_transpose_results[i])
            i += 1
        else:
            results.append(result)
    return results
