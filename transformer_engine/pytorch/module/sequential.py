from typing import Any, OrderedDict, overload
import torch.nn as nn


class ComputePipeline:
    def __init__(self, *modules: nn.Module) -> None:
        ...

    def __call__(self, x: Any) -> Any:
        ...


class Sequential(nn.Module):
    # from nn.Module
    _modules: dict[str, nn.Module]  # type: ignore[assignment]

    _is_cache_valid: bool
    _op_cache: ComputePipeline

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
                self.append(submodule, name=f"{name}.{submodule_name}")
        else:
            self.add_module(name, module)
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
            self._op_cache = ComputePipeline(*self._modules.values())
            self._is_cache_valid = True
        return self._op_cache(x)
