from . import _nvte
from .empty import empty


def gelu(x: _nvte.Tensor, out_dtype: _nvte.DType):
    output = empty(x.shape, out_dtype)
    _nvte.gelu(x, output)
    return output


def dgelu(grad: _nvte.Tensor, x: _nvte.Tensor, out_dtype: _nvte.DType):
    output = empty(x.shape, out_dtype)
    _nvte.dgelu(grad, x, output)
    return output


def relu(x: _nvte.Tensor, out_dtype: _nvte.DType):
    output = empty(x.shape, out_dtype)
    _nvte.relu(x, output)
    return output


def drelu(grad: _nvte.Tensor, x: _nvte.Tensor, out_dtype: _nvte.DType):
    output = empty(x.shape, out_dtype)
    _nvte.drelu(grad, x, output)
    return output


def reglu(x: _nvte.Tensor, out_dtype: _nvte.DType):
    output = empty(x.shape, out_dtype)
    _nvte.reglu(x, output)
    return output


def dreglu(grad: _nvte.Tensor, x: _nvte.Tensor, out_dtype: _nvte.DType):
    output = empty(x.shape, out_dtype)
    _nvte.dreglu(grad, x, output)
    return output


def swiglu(x: _nvte.Tensor, out_dtype: _nvte.DType):
    output = empty(x.shape, out_dtype)
    _nvte.swiglu(x, output)
    return output


def dswiglu(grad: _nvte.Tensor, x: _nvte.Tensor, out_dtype: _nvte.DType):
    output = empty(x.shape, out_dtype)
    _nvte.dswiglu(grad, x, output)
    return output
