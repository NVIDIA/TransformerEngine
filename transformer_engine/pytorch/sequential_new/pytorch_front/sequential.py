from typing import Any, OrderedDict, overload
import torch

import torch.nn as nn

from ...fp8 import is_fp8_enabled

from .expand_for_sequential import expand
from ..common_back.compute_pipeline import ComputePipeline
from ..pytorch_back.environment import PytorchExecutionEnv, PytorchDistributedGroup
from ..common_back.generic_environment import ExecutionEnv
from ...distributed import dist_group_type
from ..common_back.ops import Op


class Sequential(nn.Module):
    # from nn.Module
    _modules: dict[str, nn.Module]  # type: ignore[assignment]

    _had_run: bool
    _args: tuple[nn.Module | OrderedDict[str, nn.Module], ...]
    _distributed_group: dist_group_type | None
    _env: ExecutionEnv
    _args_during_compilation: tuple[nn.Module | OrderedDict[str, nn.Module], ...]
    _compiled_op_list: list[Op]
    _pipeline: ComputePipeline

    @overload
    def __init__(
        self, *modules: nn.Module, distributed_group: dist_group_type | None = None
    ) -> None:
        ...

    @overload
    def __init__(
        self,
        module_dict: OrderedDict[str, nn.Module],
        /,
        *,
        distributed_group: dist_group_type | None = None,
    ) -> None:
        ...

    def __init__(
        self,
        *args: nn.Module | OrderedDict[str, nn.Module],
        distributed_group: dist_group_type | None = None,
    ):
        super().__init__()  # type: ignore

        self._had_run = False
        self._args = args
        self._distributed_group = distributed_group

    def __len__(self):
        return len(self._modules)

    def __add__(self, other: "Sequential") -> "Sequential":
        if self._distributed_group is not other._distributed_group:
            raise ValueError(
                "Cannot add two sequentials with different distributed groups"
            )
        return Sequential(self, other, distributed_group=self._distributed_group)

    def __mul__(self, other: int):
        if other <= 0:
            raise ValueError("Repetition factor must be >= 1")
        else:
            return Sequential(
                *(self for _ in range(other)), distributed_group=self._distributed_group
            )

    def __rmul__(self, other: int):
        return self * other

    def forward(self, x: torch.Tensor) -> Any:
        self._compile_checked(x.shape, self.current_execution_env())
        return self._pipeline(x)

    def expand_for_sequential(self, compile_env: ExecutionEnv):
        return Sequential._create_op_list(self._args, compile_env)

    def current_execution_env(self):
        return PytorchExecutionEnv(
            is_fp8_enabled(),
            torch.is_grad_enabled(),
            PytorchDistributedGroup(self._distributed_group)
            if self._distributed_group is not None
            else None,
        )

    def _compile_checked(self, input_shape: tuple[int, ...], compile_env: ExecutionEnv):
        if not self._had_run:
            self._had_run = True
            self._env = compile_env
            self._args_during_compilation = self._args
            self._compiled_op_list = Sequential._create_op_list(self._args, compile_env)

            self._pipeline = ComputePipeline(
                self._compiled_op_list, input_shape, self._env
            )
        else:
            assert self._env == compile_env
            assert self._args_during_compilation == self._args

    @staticmethod
    def _create_op_list(
        args: tuple[nn.Module | OrderedDict[str, nn.Module], ...],
        compile_env: ExecutionEnv,
    ):
        modules = Sequential._flatten(args)
        return [
            op.set_parent_name(name).set_environment(compile_env)
            for name, module in modules
            for op in expand(module, compile_env)
        ]

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
