from __future__ import annotations
from math import sqrt
import torch
from torch import nn
from ..compute_pipeline import ops
from ..nvte import make_nvte_tensor
from ._common import ParameterInitMethod
from .base import BaseModule


def _default_weight_init_method(weight: torch.Tensor):
    in_features = weight.shape[0]
    k = 1 / sqrt(in_features)
    return nn.init.uniform_(weight, -k, k)


def _default_bias_init_method(bias: torch.Tensor):
    out_features = bias.shape[0]
    k = 1 / sqrt(out_features)
    return nn.init.uniform_(bias, -k, k)


class Linear(BaseModule):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        param_dtype: torch.dtype = torch.get_default_dtype(),
        weight_init_method: ParameterInitMethod = _default_weight_init_method,
        bias_init_method: ParameterInitMethod = _default_bias_init_method,
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(
            weight_init_method(
                torch.empty(out_features, in_features, dtype=param_dtype, device="cuda")
            )
        )
        self.bias = (
            nn.Parameter(
                bias_init_method(
                    torch.empty(out_features, dtype=param_dtype, device="cuda")
                )
            )
            if bias
            else None
        )

    def _ops(self) -> list[ops.Op | None]:
        return [
            ops.MMT(make_nvte_tensor(self.weight)),
            ops.Add(make_nvte_tensor(self.bias)) if self.bias is not None else None,
        ]

    def extra_repr(self):
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}"
