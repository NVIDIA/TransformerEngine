import torch
import torch.nn as nn

from ....pytorch_back.environment import PytorchExecutionEnv, PytorchDistributedGroup

from ....common_back import ComputePipeline, Op
from ...implementation.compute_pipeline_function import apply
from ...implementation.environment import Environment, get_current_environment


def _convert_env(env: Environment) -> PytorchExecutionEnv:
    return PytorchExecutionEnv(
        fp8=env.fp8,
        training=env.training,
        distributed_group=PytorchDistributedGroup(env.distributed_group)
        if env.distributed_group is not None
        else None,
    )


class ComputePipelineModuleBase(nn.Module):
    pipeline: ComputePipeline
    output_type: torch.dtype
    prev_compile_args: tuple[torch.dtype, torch.dtype, PytorchExecutionEnv] | None

    def __init__(
        self, *ops: Op | None, output_type: torch.dtype = torch.get_default_dtype()
    ):
        super().__init__()  # type: ignore
        clean_ops = [op for op in ops if op is not None]
        self.pipeline = ComputePipeline(clean_ops)
        self.output_type = output_type
        self.prev_compile_args = None

    def forward(self, x: torch.Tensor):
        self._compile_checked(
            x.dtype, self.output_type, _convert_env(get_current_environment())
        )
        return apply(x, self.pipeline)

    def _compile_checked(
        self,
        input_type: torch.dtype,
        output_type: torch.dtype,
        env: PytorchExecutionEnv,
    ):
        compile_args = (input_type, output_type, env)
        if self.prev_compile_args is None or compile_args != self.prev_compile_args:
            self.prev_compile_args = compile_args
            self._compile(input_type, output_type, env)

    def _compile(
        self,
        input_type: torch.dtype,
        output_type: torch.dtype,
        env: PytorchExecutionEnv,
    ) -> None:
        ...  # TODO
