from typing import Any, OrderedDict, overload
import torch
import torch.nn as nn
from .expand_for_sequential import expand
from ..common_back.compute_pipeline import ComputePipeline
from ..pytorch_back.environment import PytorchExecutionEnv, PytorchDistributedGroup
from ..pytorch_back.pipeline_function import PipelineFunction
from ..common_back.generic_environment import ExecutionEnv
from ..common_back.ops import Op, ParallelismClass
from ..common_back.enums import DType
from ...fp8 import is_fp8_enabled
from ...distributed import dist_group_type


class Sequential(nn.Module):
    # from nn.Module
    _modules: dict[str, nn.Module]  # type: ignore[assignment]

    _had_run: bool
    _args: tuple[nn.Module | OrderedDict[str, nn.Module], ...]
    _out_dtype: torch.dtype
    _distributed_group: dist_group_type | None
    _env: ExecutionEnv
    _args_during_compilation: tuple[nn.Module | OrderedDict[str, nn.Module], ...]
    _pipeline: ComputePipeline

    @overload
    def __init__(
        self,
        *modules: nn.Module,
        out_dtype: torch.dtype = DType.default.torch_dtype(),
        distributed_group: dist_group_type | None = None,
    ) -> None:
        ...

    @overload
    def __init__(
        self,
        module_dict: OrderedDict[str, nn.Module],
        /,
        *,
        out_dtype: torch.dtype = DType.default.torch_dtype(),
        distributed_group: dist_group_type | None = None,
    ) -> None:
        ...

    def __init__(
        self,
        *args: nn.Module | OrderedDict[str, nn.Module],
        out_dtype: torch.dtype = DType.default.torch_dtype(),
        distributed_group: dist_group_type | None = None,
    ):
        super().__init__()  # type: ignore

        self._had_run = False
        self._args = args
        self._out_dtype = out_dtype
        self._distributed_group = distributed_group

    def __len__(self):
        return len(self._modules)

    def __add__(self, other: "Sequential") -> "Sequential":
        if self._distributed_group is not other._distributed_group:
            raise ValueError(
                "Cannot add two sequentials with different distributed groups"
            )
        return Sequential(
            self,
            other,
            out_dtype=other._out_dtype,
            distributed_group=self._distributed_group,
        )

    def __mul__(self, other: int):
        if other <= 0:
            raise ValueError("Repetition factor must be >= 1")
        else:
            return Sequential(
                *(self for _ in range(other)),
                out_dtype=self._out_dtype,
                distributed_group=self._distributed_group,
            )

    def __rmul__(self, other: int):
        return self * other

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._compile_checked(x, self._out_dtype, self.current_execution_env())
        if self._env.training:
            return PipelineFunction.apply(x, self._pipeline)  # type: ignore
        else:
            return PipelineFunction.forward(None, x, self._pipeline)

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

    def _compile_checked(
        self,
        input_like: torch.Tensor,
        out_dtype: torch.dtype,
        compile_env: ExecutionEnv,
    ):
        if not self._had_run:
            self._had_run = True
            self._env = compile_env
            self._args_during_compilation = self._args
            compiled_op_list = Sequential._create_op_list(self._args, compile_env)
            self._pipeline = Sequential._create_pipeline(
                compiled_op_list, input_like, out_dtype, compile_env
            )
        else:
            assert self._env == compile_env
            assert self._args_during_compilation == self._args

    @staticmethod
    def _create_pipeline(
        ops: list[Op],
        input_like: torch.Tensor,
        out_dtype: torch.dtype,
        compile_env: ExecutionEnv,
    ):
        pipeline = (
            ComputePipeline(ops)
            .set_parent_name("Sequential")
            .set_environment(compile_env)
            .set_types_inferred(
                DType.from_torch_dtype(input_like.dtype),
                DType.from_torch_dtype(out_dtype),
            )
            .set_parallelism(ParallelismClass.NORMAL)
            .set_input_shape(input_like.shape)
        )
        params = pipeline.describe_params()
        assert params == {}

        pipeline.describe_activation_shape()

        tensors = pipeline.describe_supplementary_tensors_training()
        assert tensors == {}
        pipeline.set_tensors_allocated()

        return pipeline

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
