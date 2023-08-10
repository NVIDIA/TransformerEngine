from typing import Callable
from math import sqrt
import torch
from torch import nn
from .base import BaseModule
from ..ops import MMT, Add
from ..nvte_utils import make_nvte_tensor


def _default_weight_init_method(weight: torch.Tensor):
    in_features = weight.shape[0]
    k = 1 / sqrt(in_features)
    torch.nn.init.uniform_(weight, -k, k)


def _default_bias_init_method(bias: torch.Tensor):
    out_features = bias.shape[0]
    k = 1 / sqrt(out_features)
    torch.nn.init.uniform_(bias, -k, k)


class Linear(BaseModule):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        param_dtype: torch.dtype = torch.get_default_dtype(),
        weight_init_method: Callable[
            [torch.Tensor], None
        ] = _default_weight_init_method,
        bias_init_method: Callable[[torch.Tensor], None] = _default_bias_init_method,
    ):
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, dtype=param_dtype)
        )
        weight_init_method(self.weight)
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, dtype=param_dtype))
            bias_init_method(self.bias)

        super().__init__(
            MMT(make_nvte_tensor(self.weight)),
            Add(make_nvte_tensor(self.bias)) if bias else None,
        )
