from typing import Any, Iterable, OrderedDict, overload

import torch
import torch.nn as nn

from .expand_for_sequential import expand
from .compute_pipeline import ComputePipeline
from .compile_env import CompileEnv
from .ops import Op
from .pytorch_interface import PytorchInterface


class Sequential(nn.Module):
    # from nn.Module
    _modules: dict[str, nn.Module]  # type: ignore[assignment]

    _had_run: bool
    _args: tuple[nn.Module | OrderedDict[str, nn.Module], ...]
    _compile_env: CompileEnv
    _args_during_compilation: tuple[nn.Module | OrderedDict[str, nn.Module], ...]
    _compiled_op_list: list[Op]
    _pipeline: ComputePipeline[torch.Tensor]

    @overload
    def __init__(self, *modules: nn.Module) -> None:
        ...

    @overload
    def __init__(self, module_dict: OrderedDict[str, nn.Module], /) -> None:
        ...

    def __init__(self, *args: nn.Module | OrderedDict[str, nn.Module]):
        super().__init__()  # type: ignore

        self._had_run = False
        self._args = args

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
            self._compiled_op_list = Sequential._compile(self._args, compile_env)
            self._pipeline = ComputePipeline(PytorchInterface(), self._compiled_op_list)
        else:
            assert self._compile_env == compile_env
            assert self._args_during_compilation == self._args

    @staticmethod
    def _compile(
        args: tuple[nn.Module | OrderedDict[str, nn.Module], ...],
        compile_env: CompileEnv,
    ):
        modules: Iterable[tuple[str, nn.Module]]
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            modules = args[0].items()
        else:
            args1: tuple[nn.Module, ...] = args  # type: ignore
            modules = map(lambda p: (f"seq[{p[0]}]", p[1]), enumerate(args1))

        return [
            op.named(name)
            for name, module in modules
            for op in expand(module, compile_env)
        ]


__all__ = ["Sequential"]
