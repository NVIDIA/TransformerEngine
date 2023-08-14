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
