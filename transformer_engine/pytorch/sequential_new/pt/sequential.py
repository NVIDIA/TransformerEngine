from typing import Any, Callable, Iterable, OrderedDict, overload

import torch
import torch.nn as nn

from .expand_for_sequential import expand
from ..common.compute_pipeline import ComputePipeline
from ..common.compile_env import CompileEnv
from ..common.ops import Op
from .pytorch_interface import PytorchInterface
from ..common.model_parallel_transform import model_parallel_transform


class Sequential(nn.Module):
    # from nn.Module
    _modules: dict[str, nn.Module]  # type: ignore[assignment]

    _had_run: bool
    _args: tuple[nn.Module | OrderedDict[str, nn.Module], ...]
    _model_parallel: bool
    _compile_env: CompileEnv
    _args_during_compilation: tuple[nn.Module | OrderedDict[str, nn.Module], ...]
    _compiled_op_list: list[Op]
    _pipeline: ComputePipeline

    @overload
    def __init__(self, *modules: nn.Module, model_parallel: bool = False) -> None:
        ...

    @overload
    def __init__(
        self,
        module_dict: OrderedDict[str, nn.Module],
        /,
        *,
        model_parallel: bool = False,
    ) -> None:
        ...

    def __init__(
        self,
        *args: nn.Module | OrderedDict[str, nn.Module],
        model_parallel: bool = False,
    ):
        super().__init__()  # type: ignore

        self._had_run = False
        self._args = args
        self._model_parallel = model_parallel

    def __len__(self):
        return len(self._modules)

    def __add__(self, other: "Sequential") -> "Sequential":
        return Sequential(
            self, other, model_parallel=self._model_parallel and other._model_parallel
        )

    def __mul__(self, other: int):
        if other <= 0:
            raise ValueError("Repetition factor must be >= 1")
        else:
            return Sequential(
                *(self for _ in range(other)), model_parallel=self._model_parallel
            )

    def __rmul__(self, other: int):
        return self * other

    def forward(self, x: Any) -> Any:
        self._compile_checked(CompileEnv.current())
        return self._pipeline(x)

    def expand_for_sequential(self, compile_env: CompileEnv):
        self._compile_checked(compile_env)
        return self._compiled_op_list

    def _compile_checked(self, compile_env: CompileEnv):
        if not self._had_run:
            self._had_run = True
            self._compile_env = compile_env
            self._args_during_compilation = self._args
            modules = Sequential._flatten(self._args)
            self._compiled_op_list = [
                op.named(name)
                for name, module in modules
                for op in expand(module, compile_env)
            ]

            additional_transforms: list[Callable[[list[Op]], list[Op]]] = []
            if self._model_parallel:
                additional_transforms.append(model_parallel_transform)

            self._pipeline = ComputePipeline(
                PytorchInterface(), self._compiled_op_list, additional_transforms
            )
        else:
            assert self._compile_env == compile_env
            assert self._args_during_compilation == self._args

    @staticmethod
    def _flatten(
        args: tuple[nn.Module | OrderedDict[str, nn.Module], ...]
    ) -> list[tuple[str, nn.Module]]:
        modules: list[tuple[str, nn.Module]]
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            modules = list(args[0].items())
        else:
            args1: tuple[nn.Module, ...] = args  # type: ignore
            modules = list(map(lambda p: (f"{p[0]}", p[1]), enumerate(args1)))

        flattened: list[tuple[str, nn.Module]] = []
        for name, module in modules:
            submodules: list[tuple[str, nn.Module]]
            if isinstance(module, Sequential):
                submodules = Sequential._flatten(module._args)
                for i, (submodule_name, submodule) in enumerate(submodules):
                    submodules[i] = (f"{name}[{submodule_name}]", submodule)
            else:
                submodules = [(name, module)]
            flattened.extend(submodules)

        return flattened


__all__ = ["Sequential"]
