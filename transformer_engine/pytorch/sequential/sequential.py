from typing import Any, OrderedDict, overload
import torch.nn as nn
from .simple_compute_pipeline import ComputePipeline


class Sequential(nn.Module):
    # from nn.Module
    _modules: dict[str, nn.Module]  # type: ignore[assignment]

    _is_cache_valid: bool
    _op_cache: ComputePipeline
    _had_run: bool

    @overload
    def __init__(self, *modules: nn.Module) -> None:
        ...

    @overload
    def __init__(self, module_dict: OrderedDict[str, nn.Module], /) -> None:
        ...

    def __init__(self, *args: nn.Module | OrderedDict[str, nn.Module]):
        super().__init__()  # type: ignore

        self._had_run = False

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
        if isinstance(module, Sequential) or isinstance(module, nn.Sequential):
            for submodule_name, submodule in module._modules.items():
                self.append(submodule, name=f"{name}_{submodule_name}")
        else:
            self.add_module(name, module)
            setattr(module, "_compute_pipeline_name", name)
        self._is_cache_valid = False

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
        if not self._is_cache_valid:
            if self._had_run:
                raise RuntimeError(
                    "Sequential is being run again,"
                    "but the module list has changed since the previous run."
                    "This would invalidate the compute pipeline and delete all current module data."
                )
            self._op_cache = ComputePipeline(*self._modules.values())
            self._is_cache_valid = True
        self._had_run = True
        return self._op_cache(x)


__all__ = ["Sequential"]
