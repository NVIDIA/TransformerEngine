from __future__ import annotations
from typing import OrderedDict, overload
from torch import nn
from .base import BaseModule


class Sequential(BaseModule):
    _modules: dict[str, BaseModule]  # type: ignore[assignment]

    @overload
    def __init__(
        self,
        *modules: BaseModule,
    ) -> None:
        ...

    @overload
    def __init__(
        self,
        module_dict: OrderedDict[str, BaseModule],
        /,
    ) -> None:
        ...

    def __init__(
        self,
        *args: BaseModule | OrderedDict[str, BaseModule],
    ):
        nn.Module.__init__(self)  # type: ignore
        modules: list[tuple[str, BaseModule]]
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            modules = list(args[0].items())
        else:
            args1: tuple[BaseModule, ...] = args  # type: ignore
            modules = list(map(lambda p: (f"{p[0]}", p[1]), enumerate(args1)))

        for name, module in modules:
            submodules: list[tuple[str, BaseModule]]
            if isinstance(module, Sequential):
                submodules = [(k, v) for k, v in module._modules.items()]
                for i, (submodule_name, submodule) in enumerate(submodules):
                    submodules[i] = (f"{name}[{submodule_name}]", submodule)
            else:
                submodules = [(name, module)]

            for submodule_name, submodule in submodules:
                self.add_module(submodule_name, submodule)

        super().__init__(*[op for _, module in modules for op in module.ops])

    def __len__(self):
        return len(self._modules)

    def __add__(self, other: Sequential) -> Sequential:
        return Sequential(
            self,
            other,
        )

    def __mul__(self, other: int):
        if other <= 0:
            raise ValueError("Repetition factor must be >= 1")
        else:
            return Sequential(
                *(self for _ in range(other)),
            )

    def __rmul__(self, other: int):
        return self * other
