from __future__ import annotations
import torch
from . import cpp_extensions as _nvte

from ._common import make_nvte_tensor
from .dtype import is_fp8, te_to_torch_dtype


def add(A: _nvte.Tensor, B: _nvte.Tensor, out_dtype: _nvte.DType):
    if is_fp8(A) or is_fp8(B):
        raise NotImplementedError()  # TODO
    else:
        output = torch.empty(A.shape, dtype=te_to_torch_dtype(out_dtype), device="cuda")
        torch.add(A.data, B.data, out=output)
        return make_nvte_tensor(output)


def dbias(grad: _nvte.Tensor, out_dtype: _nvte.DType):
    if is_fp8(grad):
        raise NotImplementedError()  # TODO
    else:
        output = torch.sum(grad.data, dtype=te_to_torch_dtype(out_dtype), dim=0)
        return make_nvte_tensor(output)
