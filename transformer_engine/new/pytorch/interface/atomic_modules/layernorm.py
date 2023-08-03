import torch
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
        param_dtype: torch.dtype | DType = torch.get_default_dtype(),
    ):
        if isinstance(param_dtype, torch.dtype):
            param_dtype = DType.from_torch_dtype(param_dtype)
        super().__init__(
            ops.LayerNorm(
                "layernorm",
                DType.Infer,
                DType.Infer,
                param_dtype,
                features,
                eps,
                zero_centered_gamma,
            )
        )
