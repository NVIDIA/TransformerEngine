import torch
from . import _nvte
from .tensor import Tensor
from ._common import make_nvte_tensor
from .dtype import is_fp8, te_to_torch_dtype


def add(A: Tensor, B: Tensor, out_dtype: _nvte.DType):
    if is_fp8(A) or is_fp8(B):
        raise NotImplementedError()
    else:
        output = torch.empty(A.shape, dtype=te_to_torch_dtype(out_dtype), device="cuda")
        torch.add(A.data, B.data, out=output)
        return make_nvte_tensor(output)


def dbias(grad: Tensor, out_dtype: _nvte.DType):
    if is_fp8(grad):
        raise NotImplementedError()
    else:
        output = torch.sum(grad.data, dtype=te_to_torch_dtype(out_dtype), dim=0)
        return make_nvte_tensor(output)
