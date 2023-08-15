from .dtype import is_fp8
from . import _nvte
from .cast_transpose import cast_transpose_checked
from .empty import multi_empty_share_metadata, empty, empty_like
from .add import dbias


def cast_transpose_dbias_checked(
    grad: _nvte.Tensor, cast_dtype: _nvte.DType | None, dbias_dtype: _nvte.DType
):
    if (
        dbias_dtype == grad.dtype
        and cast_dtype is not None
        and cast_dtype != grad.dtype
    ):
        out_cast, out_transpose = multi_empty_share_metadata(
            (grad.shape, cast_dtype), (grad.shape[::-1], cast_dtype)
        )
        out_dbias = empty((grad.shape[1],), dbias_dtype)
        workspace = empty()
        for _ in range(2):
            _nvte.cast_transpose_dbias(
                grad, out_cast, out_transpose, out_dbias, workspace
            )
            workspace = empty_like(workspace)
        return out_cast, out_transpose, out_dbias
    elif is_fp8(grad.dtype) and cast_dtype is None or cast_dtype == grad.dtype:
        out_transpose = empty(grad.shape[::-1], grad.dtype)
        out_dbias = empty((grad.shape[1],), dbias_dtype)
        workspace = empty()
        for _ in range(2):
            _nvte.fp8_transpose_dbias(grad, out_transpose, out_dbias, workspace)
            workspace = empty_like(workspace)
        return grad, out_transpose, out_dbias
    else:
        out_cast, out_transpose = cast_transpose_checked(grad, cast_dtype)
        out_dbias = dbias(grad, dbias_dtype)
        return out_cast, out_transpose, out_dbias
