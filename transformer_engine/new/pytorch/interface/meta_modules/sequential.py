from __future__ import annotations
from typing import OrderedDict, overload
import torch
from ..base_modules.compute_pipeline_module_base import ComputePipelineModuleBase


class Sequential(ComputePipelineModuleBase):
    # from nn.Module
    _modules: dict[str, ComputePipelineModuleBase]  # type: ignore[assignment]

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
        modules: list[tuple[str, ComputePipelineModuleBase]]
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            modules = list(args[0].items())
        else:
            args1: tuple[ComputePipelineModuleBase, ...] = args  # type: ignore
            modules = list(map(lambda p: (f"{p[0]}", p[1]), enumerate(args1)))

        for name, module in modules:
            submodules: list[tuple[str, ComputePipelineModuleBase]]
            if isinstance(module, Sequential):
                submodules = [(k, v) for k, v in Sequential._modules.items()]
                for i, (submodule_name, submodule) in enumerate(submodules):
                    submodules[i] = (f"{name}[{submodule_name}]", submodule)
            else:
                submodules = [(name, module)]

            for submodule_name, submodule in submodules:
                self.add_module(submodule_name, submodule)

        super().__init__(
            *[op for _, module in modules for op in module.ops],
            output_type=out_dtype,
        )

    def __len__(self):
        return len(self._modules)

    def __add__(self, other: Sequential) -> Sequential:
        return Sequential(
            self,
            other,
            out_dtype=other.output_type,
        )

    def __mul__(self, other: int):
        if other <= 0:
            raise ValueError("Repetition factor must be >= 1")
        else:
            return Sequential(
                *(self for _ in range(other)),
                out_dtype=self.output_type,
            )

    def __rmul__(self, other: int):
        return self * other


__all__ = ["Sequential"]
