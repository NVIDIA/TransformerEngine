from abc import ABC
import torch
from torch import nn
from .base import BaseModule
from ..compute_pipeline import ops
from ..nvte import make_nvte_tensor


class Normalization(BaseModule, ABC):
    def __init__(
        self,
        features: int,
        eps: float = 1e-5,
        zero_centered_gamma: bool = False,
        param_dtype: torch.dtype = torch.get_default_dtype(),
    ):
        super().__init__()

        self.features = features
        self.eps = eps
        self.zero_centered_gamma = zero_centered_gamma

        self.weight = nn.Parameter(
            torch.zeros(features, dtype=param_dtype, device="cuda")
            if zero_centered_gamma
            else torch.ones(features, dtype=param_dtype, device="cuda")
        )
        self.bias = (
            nn.Parameter(torch.zeros(features, dtype=param_dtype, device="cuda"))
            if type(self)._bias
            else None
        )

    def _ops(self) -> list[ops.Op | None]:
        return [
            type(self)._op_type(
                *(
                    (
                        self.eps,
                        self.zero_centered_gamma,
                        make_nvte_tensor(self.weight),
                    )
                    + ((make_nvte_tensor(self.bias),) if self.bias is not None else ())
                )
            ),
        ]

    def extra_repr(self):
        return f"features={self.features}, eps={self.eps}, zero_centered_gamma={self.zero_centered_gamma}"

    _bias: bool
    _op_type: type[ops.Op]


class LayerNorm(Normalization):
    _bias = True
    _op_type = ops.LayerNorm


class RMSNorm(Normalization):
    _bias = False
    _op_type = ops.RMSNorm
