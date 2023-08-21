import torch
from .tensor import Tensor
from .dtype import torch_to_te_dtype


def make_nvte_tensor(t: torch.Tensor):
    return Tensor(
        torch_to_te_dtype(t.dtype),
        t.data,
        torch.Tensor(),
        torch.Tensor(),
        torch.Tensor(),
    )
