import torch
import torch.nn as nn

from ....common import ops, DType

from ..base_modules.compute_pipeline_module_base import (
    ComputePipelineModuleBase,
)


class LayerNorm(ComputePipelineModuleBase):
    def __init__(
        self,
        features: int,
        eps: float = 1e-5,
        zero_centered_gamma: bool = False,
    ):
        self.weight = nn.Parameter(torch.Tensor(features))
        self.bias = nn.Parameter(torch.Tensor(features))
        super().__init__(
            ops.LayerNorm(
                "layernorm",
                DType.Infer,
                DType.Infer,
                eps,
                zero_centered_gamma,
                NVTE_Tensor.from_pytorch(self.weight),
                NVTE_Tensor.from_pytorch(self.bias),
            )
        )
