from __future__ import annotations
from . import cpp_extensions as _nvte
from ._common import torch_op
from .dtype import is_fp8
from .cast_transpose import cast_transpose_checked
from .empty import multi_empty_share_metadata, empty, empty_like
from .add import dbias
from .activation import dgeglu, dgelu


@torch_op
def _cast_transpose_dbias(
    grad: _nvte.Tensor, cast_dtype: _nvte.DType, dbias_dtype: _nvte.DType
) -> tuple[_nvte.Tensor, _nvte.Tensor, _nvte.Tensor]:
    grad_cast, grad_transpose = multi_empty_share_metadata(
        (grad.shape, cast_dtype), (grad.shape[::-1], cast_dtype)
    )
    out_dbias = empty((grad.shape[1],), dbias_dtype)
    workspace = empty()
    for _ in range(2):
        _nvte.cast_transpose_dbias(
            grad, grad_cast, grad_transpose, out_dbias, workspace
        )
        workspace = empty_like(workspace.query_shape_dtype())
    return grad_cast, grad_transpose, out_dbias


@torch_op
def _fp8_transpose_dbias(
    grad: _nvte.Tensor, dbias_dtype: _nvte.DType
) -> tuple[_nvte.Tensor, _nvte.Tensor, _nvte.Tensor]:
    grad_transpose = empty(grad.shape[::-1], grad.dtype)
    out_dbias = empty((grad.shape[1],), dbias_dtype)
    workspace = empty()
    for _ in range(2):
        _nvte.fp8_transpose_dbias(grad, grad_transpose, out_dbias, workspace)
        workspace = empty_like(workspace.query_shape_dtype())
    return grad, grad_transpose, out_dbias


def cast_transpose_dbias_checked(
    grad: _nvte.Tensor, cast_dtype: _nvte.DType | None, dbias_dtype: _nvte.DType
):
    if (
        dbias_dtype == grad.dtype
        and cast_dtype is not None
        and cast_dtype != grad.dtype
    ):
        return _cast_transpose_dbias(grad, cast_dtype, dbias_dtype)
    elif is_fp8(grad) and (cast_dtype is None or cast_dtype == grad.dtype):
        return _fp8_transpose_dbias(grad, dbias_dtype)
    else:
        grad_cast, grad_transpose = cast_transpose_checked(grad, cast_dtype)
        out_dbias = dbias(grad, dbias_dtype)
        return grad_cast, grad_transpose, out_dbias


@torch_op
def _cast_transpose_dbias_dgelu(
    grad: _nvte.Tensor,
    pre_gelu: _nvte.Tensor,
    cast_dtype: _nvte.DType,
    dbias_dtype: _nvte.DType,
) -> tuple[_nvte.Tensor, _nvte.Tensor, _nvte.Tensor]:
    dgelu_cast, dgelu_transpose = multi_empty_share_metadata(
        (grad.shape, cast_dtype), (grad.shape[::-1], cast_dtype)
    )
    out_dbias = empty((grad.shape[1],), dbias_dtype)
    workspace = empty()
    for _ in range(2):
        _nvte.cast_transpose_dbias_dgelu(
            grad, pre_gelu, dgelu_cast, dgelu_transpose, out_dbias, workspace
        )
        workspace = empty_like(workspace.query_shape_dtype())
    return dgelu_cast, dgelu_transpose, out_dbias


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
        return _cast_transpose_dbias_dgelu(grad, pre_gelu, cast_dtype, dbias_dtype)
    else:
        dgelu_ = dgelu(grad, pre_gelu, cast_dtype or grad.dtype)
        return cast_transpose_dbias_checked(dgelu_, cast_dtype, dbias_dtype)


@torch_op
def _cast_transpose_dgeglu(
    grad: _nvte.Tensor, pre_geglu: _nvte.Tensor, cast_dtype: _nvte.DType
) -> tuple[_nvte.Tensor, _nvte.Tensor]:
    dgeglu_cast, dgeglu_transpose = multi_empty_share_metadata(
        (grad.shape, cast_dtype), (grad.shape[::-1], cast_dtype)
    )
    _nvte.dgeglu_cast_transpose(grad, pre_geglu, dgeglu_cast, dgeglu_transpose)
    return dgeglu_cast, dgeglu_transpose


def cast_transpose_dgeglu_checked(
    grad: _nvte.Tensor, pre_geglu: _nvte.Tensor, cast_dtype: _nvte.DType | None
):
    if (
        grad.dtype == pre_geglu.dtype
        and cast_dtype is not None
        and cast_dtype != grad.dtype
    ):
        return _cast_transpose_dgeglu(grad, pre_geglu, cast_dtype)
    else:
        dgeglu_ = dgeglu(grad, pre_geglu, cast_dtype or grad.dtype)
        return cast_transpose_checked(dgeglu_, cast_dtype)
