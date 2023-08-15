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
        grad_cast, grad_transpose = multi_empty_share_metadata(
            (grad.shape, cast_dtype), (grad.shape[::-1], cast_dtype)
        )
        out_dbias = empty((grad.shape[1],), dbias_dtype)
        workspace = empty()
        for _ in range(2):
            _nvte.cast_transpose_dbias(
                grad, grad_cast, grad_transpose, out_dbias, workspace
            )
            workspace = empty_like(workspace)
        return grad_cast, grad_transpose, out_dbias
    elif is_fp8(grad) and (cast_dtype is None or cast_dtype == grad.dtype):
        grad_transpose = empty(grad.shape[::-1], grad.dtype)
        out_dbias = empty((grad.shape[1],), dbias_dtype)
        workspace = empty()
        for _ in range(2):
            _nvte.fp8_transpose_dbias(grad, grad_transpose, out_dbias, workspace)
            workspace = empty_like(workspace)
        return grad, grad_transpose, out_dbias
    else:
        grad_cast, grad_transpose = cast_transpose_checked(grad, cast_dtype)
        out_dbias = dbias(grad, dbias_dtype)
        return grad_cast, grad_transpose, out_dbias


def cast_transpose_dbias_dgelu_checked(
    grad: _nvte.Tensor,
    pre_gelu: _nvte.Tensor,
    cast_dtype: _nvte.DType | None,
    dbias_dtype: _nvte.DType,
):
    if (
        dbias_dtype == grad.dtype
        and cast_dtype is not None
        and cast_dtype != grad.dtype
        and grad.dtype == pre_gelu.dtype
    ):
        dgelu_cast, dgelu_transpose = multi_empty_share_metadata(
            (grad.shape, cast_dtype), (grad.shape[::-1], cast_dtype)
        )
        out_dbias = empty((grad.shape[1],), dbias_dtype)
        workspace = empty()
        for _ in range(2):
            _nvte.cast_transpose_dbias_dgelu(
                grad, pre_gelu, dgelu_cast, dgelu_transpose, out_dbias, workspace
            )
            workspace = empty_like(workspace)
        return dgelu_cast, dgelu_transpose, out_dbias
    else:
        dgelu = empty(grad.shape, cast_dtype or grad.dtype)
        _nvte.dgelu(grad, pre_gelu, dgelu)
        return cast_transpose_dbias_checked(dgelu, cast_dtype, dbias_dtype)


def cast_transpose_dgeglu_checked(
    grad: _nvte.Tensor, pre_geglu: _nvte.Tensor, cast_dtype: _nvte.DType | None
):
    if (
        grad.dtype == pre_geglu.dtype
        and cast_dtype is not None
        and cast_dtype != grad.dtype
    ):
        dgeglu_cast, dgeglu_transpose = multi_empty_share_metadata(
            (grad.shape, cast_dtype), (grad.shape[::-1], cast_dtype)
        )
        _nvte.dgeglu_cast_transpose(grad, pre_geglu, dgeglu_cast, dgeglu_transpose)
        return dgeglu_cast, dgeglu_transpose
    else:
        dgeglu = empty(grad.shape, cast_dtype or grad.dtype)
        _nvte.dgeglu(grad, pre_geglu, dgeglu)
        return cast_transpose_checked(dgeglu, cast_dtype)
