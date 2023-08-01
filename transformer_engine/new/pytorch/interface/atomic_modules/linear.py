import torch
import torch.nn as nn

from ....common_back import ops, DType

from ..base_modules.compute_pipeline_module_base import (
    ComputePipelineModuleBase,
)


class Linear(ComputePipelineModuleBase):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
    ):
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features)) if bias else None
        super().__init__(
            ops.Gemm(
                "gemm",
                DType.Infer,
                DType.LowestPrec,
                NVTE_Tensor.from_pytorch(self.weight),
            ),
            ops.Bias(
                "bias",
                DType.LowestPrec,
                DType.Infer,
                NVTE_Tensor.from_pytorch(self.bias),
            )
            if bias
            else None,
        )
