from typing_extensions import deprecated
from ..base_modules.sequential_module_base import SequentialModuleBase
from ..atomic_modules import LayerNorm, Linear


@deprecated
class LayerNormLinear(SequentialModuleBase):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        eps: float = 1e-5,
        zero_centered_gamma: bool = False,
        bias: bool = True,
    ):
        super().__init__(
            LayerNorm(in_features, eps, zero_centered_gamma),
            Linear(in_features, out_features, bias=bias),
        )
