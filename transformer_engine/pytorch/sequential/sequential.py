from typing import Any, OrderedDict, overload
import torch.nn as nn
from .compute_pipeline import ComputePipeline

from numpy import ndarray


class Sequential(nn.Module):
    # from nn.Module
    _modules: dict[str, nn.Module]  # type: ignore[assignment]

    _op_cache: ComputePipeline | None

    @overload
    def __init__(self, *modules: nn.Module) -> None:
        ...

    @overload
    def __init__(self, module_dict: OrderedDict[str, nn.Module], /) -> None:
        ...

    def __init__(self, *args: nn.Module | OrderedDict[str, nn.Module]):
        super().__init__()  # type: ignore

        self._subsequence_count = 0

        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for name, module in args[0].items():
                self.append(module, name=name)
        else:
            args1: tuple[nn.Module, ...] = args  # type: ignore
            for module in args1:
                self.append(module)

    def append(self, module: nn.Module, *, name: str | None = None):
        if name is None:
            name = str(len(self._modules))
        if isinstance(module, Sequential):
            for submodule_name, submodule in module._modules.items():
                self.append(submodule, name=f"{name}_{submodule_name}")
        else:
            self.add_module(name, module)
            setattr(module, "_compute_pipeline_name", name)
        self._op_cache = None

    def __len__(self):
        return len(self._modules)

    def __add__(self, other: "Sequential") -> "Sequential":
        return Sequential(self, other)

    def __mul__(self, other: int):
        if other <= 0:
            raise ValueError("Repetition factor must be >= 1")
        else:
            return Sequential(*(self for _ in range(other)))

    def __rmul__(self, other: int):
        return self * other

    def forward(self, x: Any):
        if self._op_cache is None:
            self._op_cache = ComputePipeline(*self._modules.values())
        return self._op_cache(x)


__all__ = ["Sequential"]
