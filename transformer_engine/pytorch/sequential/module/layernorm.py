import torch
from torch import nn
from .base import BaseModule
from .. import ops
from nvte_utils import make_nvte_tensor


class LayerNorm(BaseModule):
    def __init__(
        self,
        features: int,
        eps: float = 1e-5,
        zero_centered_gamma: bool = False,
        param_dtype: torch.dtype = torch.get_default_dtype(),
    ):
        nn.Module.__init__(self)  # type: ignore

        self.weight = nn.Parameter(
            torch.zeros(features, dtype=param_dtype, device="cuda")
            if zero_centered_gamma
            else torch.ones(features, dtype=param_dtype, device="cuda")
        )
        self.bias = nn.Parameter(
            torch.zeros(features, dtype=param_dtype, device="cuda")
        )

        super().__init__(
            ops.LayerNorm(
                eps,
                zero_centered_gamma,
                make_nvte_tensor(self.weight),
                make_nvte_tensor(self.bias),
            )
        )
