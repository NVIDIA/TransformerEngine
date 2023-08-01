from typing_extensions import deprecated
from torch import nn
from ..base_modules.sequential_module_base import SequentialModuleBase
from ..atomic_modules import LayerNorm, Linear

Activation = nn.ReLU | nn.GELU


@deprecated
class LayerNormMLP(SequentialModuleBase):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        eps: float = 1e-5,
        zero_centered_gamma: bool = False,
        bias: bool = True,
        activation: Activation = nn.GELU(),
    ):
        super().__init__(
            LayerNorm(in_features, eps, zero_centered_gamma),
            Linear(in_features, out_features, bias=bias),
            activation,
            Linear(in_features, out_features, bias=bias),
        )
