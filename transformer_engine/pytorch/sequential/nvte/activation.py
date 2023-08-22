from .. import cpp_extensions as _nvte
from .empty import empty


def relu(x: _nvte.Tensor, out_dtype: _nvte.DType):
    output = empty(x.shape, out_dtype)
    _nvte.relu(x, output)
    return output


def drelu(grad: _nvte.Tensor, x: _nvte.Tensor, out_dtype: _nvte.DType):
    output = empty(x.shape, out_dtype)
    _nvte.drelu(grad, x, output)
    return output


def gelu(x: _nvte.Tensor, out_dtype: _nvte.DType):
    output = empty(x.shape, out_dtype)
    _nvte.gelu(x, output)
    return output


def dgelu(grad: _nvte.Tensor, x: _nvte.Tensor, out_dtype: _nvte.DType):
    output = empty(x.shape, out_dtype)
    _nvte.dgelu(grad, x, output)
    return output


def reglu(x: _nvte.Tensor, out_dtype: _nvte.DType):
    output = empty((x.shape[0], x.shape[1] // 2), out_dtype)
    _nvte.reglu(x, output)
    return output


def dreglu(grad: _nvte.Tensor, x: _nvte.Tensor, out_dtype: _nvte.DType):
    output = empty(x.shape, out_dtype)
    _nvte.dreglu(grad, x, output)
    return output


def geglu(x: _nvte.Tensor, out_dtype: _nvte.DType):
    output = empty((x.shape[0], x.shape[1] // 2), out_dtype)
    _nvte.geglu(x, output)
    return output


def dgeglu(grad: _nvte.Tensor, x: _nvte.Tensor, out_dtype: _nvte.DType):
    output = empty(x.shape, out_dtype)
    _nvte.dgeglu(grad, x, output)
    return output


def swiglu(x: _nvte.Tensor, out_dtype: _nvte.DType):
    output = empty((x.shape[0], x.shape[1] // 2), out_dtype)
    _nvte.swiglu(x, output)
    return output


def dswiglu(grad: _nvte.Tensor, x: _nvte.Tensor, out_dtype: _nvte.DType):
    output = empty(x.shape, out_dtype)
    _nvte.dswiglu(grad, x, output)
    return output
