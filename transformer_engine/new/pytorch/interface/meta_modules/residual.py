from typing import OrderedDict, overload
import torch

from transformer_engine.new.common import ops
from .sequential import Sequential
from ..base_modules.compute_pipeline_module_base import ComputePipelineModuleBase


class Residual(ComputePipelineModuleBase):
    @overload
    def __init__(
        self,
        *modules: ComputePipelineModuleBase,
        out_dtype: torch.dtype = torch.get_default_dtype(),
    ) -> None:
        ...

    @overload
    def __init__(
        self,
        module_dict: OrderedDict[str, ComputePipelineModuleBase],
        /,
        *,
        out_dtype: torch.dtype = torch.get_default_dtype(),
    ) -> None:
        ...

    def __init__(
        self,
        *args: ComputePipelineModuleBase | OrderedDict[str, ComputePipelineModuleBase],
        out_dtype: torch.dtype = torch.get_default_dtype(),
    ):
        begin = ops.ResidualBegin("residual_begin", None)
        end = ops.ResidualEnd("residual_end", begin)
        begin.end = end
        super().__init__(
            begin,
            *Sequential(args, out_dtype=out_dtype).ops,  # type: ignore[arg-type]
            end,
            output_type=out_dtype,
        )


__all__ = ["Residual"]
