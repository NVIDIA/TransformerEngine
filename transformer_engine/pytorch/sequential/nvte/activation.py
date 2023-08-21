from . import _nvte
from .tensor import Tensor
from .empty import empty


def relu(x: Tensor, out_dtype: _nvte.DType):
    output = empty(x.shape, out_dtype)
    _nvte.relu(x, output)
    return output


def drelu(grad: Tensor, x: Tensor, out_dtype: _nvte.DType):
    output = empty(x.shape, out_dtype)
    _nvte.drelu(grad, x, output)
    return output


def gelu(x: Tensor, out_dtype: _nvte.DType):
    output = empty(x.shape, out_dtype)
    _nvte.gelu(x, output)
    return output


def dgelu(grad: Tensor, x: Tensor, out_dtype: _nvte.DType):
    output = empty(x.shape, out_dtype)
    _nvte.dgelu(grad, x, output)
    return output


def reglu(x: Tensor, out_dtype: _nvte.DType):
    output = empty((x.shape[0], x.shape[1] // 2), out_dtype)
    _nvte.reglu(x, output)
    return output


def dreglu(grad: Tensor, x: Tensor, out_dtype: _nvte.DType):
    output = empty(x.shape, out_dtype)
    _nvte.dreglu(grad, x, output)
    return output


def geglu(x: Tensor, out_dtype: _nvte.DType):
    output = empty((x.shape[0], x.shape[1] // 2), out_dtype)
    _nvte.geglu(x, output)
    return output


def dgeglu(grad: Tensor, x: Tensor, out_dtype: _nvte.DType):
    output = empty(x.shape, out_dtype)
    _nvte.dgeglu(grad, x, output)
    return output


def swiglu(x: Tensor, out_dtype: _nvte.DType):
    output = empty((x.shape[0], x.shape[1] // 2), out_dtype)
    _nvte.swiglu(x, output)
    return output


def dswiglu(grad: Tensor, x: Tensor, out_dtype: _nvte.DType):
    output = empty(x.shape, out_dtype)
    _nvte.dswiglu(grad, x, output)
    return output
