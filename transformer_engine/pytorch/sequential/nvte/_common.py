import torch
from . import _nvte
from .dtype import torch_to_te_dtype


def make_nvte_tensor(t: torch.Tensor):
    return _nvte.Tensor(
        torch_to_te_dtype(t.dtype),
        t.data,
        torch.Tensor(),
        torch.Tensor(),
        torch.Tensor(),
    )
