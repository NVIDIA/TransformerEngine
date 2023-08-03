from math import sqrt
from typing import Callable
import torch
from ....common import ops, DType
from ....pytorch_back.tensor import init_wrapper
from ..base_modules.compute_pipeline_module_base import (
    ComputePipelineModuleBase,
)


class Linear(ComputePipelineModuleBase):
    @staticmethod
    def default_weight_init_method(weight: torch.Tensor) -> None:
        """
        The default way to initialize the weight parameter.
        Mimics the behavior of nn.Linear
        """
        in_features = weight.shape[0]
        k = 1 / sqrt(in_features)
        return torch.nn.init.uniform_(weight, -k, k)  # type: ignore

    @staticmethod
    def default_bias_init_method(bias: torch.Tensor) -> None:
        """
        The default way to initialize the bias parameter.
        Mimics the behavior of nn.Linear
        """
        out_features = bias.shape[0]
        k = 1 / sqrt(out_features)
        return torch.nn.init.uniform_(bias, -k, k)  # type: ignore

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        param_dtype: torch.dtype | DType = torch.get_default_dtype(),
        weight_init_method: Callable[[torch.Tensor], None] = default_weight_init_method,
        bias_init_method: Callable[[torch.Tensor], None] = default_bias_init_method,
    ):
        if isinstance(param_dtype, torch.dtype):
            param_dtype = DType.from_torch_dtype(param_dtype)
        super().__init__(
            ops.Gemm(
                "gemm",
                DType.Infer,
                DType.LowestPrec,
                param_dtype,
                in_features,
                out_features,
                init_wrapper(weight_init_method),
            ),
            ops.Bias(
                "bias",
                DType.LowestPrec,
                DType.Infer,
                param_dtype,
                out_features,
                init_wrapper(bias_init_method),
            )
            if bias
            else None,
        )
