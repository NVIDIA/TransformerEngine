from __future__ import annotations
import torch
from .. import cpp_extensions as _nvte
from ..utils import torch_op


@torch_op
def make_nvte_tensor(t: torch.Tensor) -> _nvte.Tensor:
    return _nvte.Tensor(
        t.data,
        torch.Tensor().cuda(),
        torch.Tensor().cuda(),
        torch.Tensor().cuda(),
    )
