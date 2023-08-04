from __future__ import annotations
import torch
import torch.nn as nn
from ....pytorch_back.environment import PytorchExecutionEnv, PytorchDistributedGroup
from ....pytorch_back.tensor import PytorchTensor
from ....common import ComputePipeline, Op, DType
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
    op_owner: ComputePipelineModuleBase
    ops: list[Op]

    output_type: torch.dtype
    distributed_world_size: int | None
    pipeline: ComputePipeline | None
    prev_compile_args: tuple[torch.dtype, torch.dtype, PytorchExecutionEnv] | None

    def __init__(
        self, *ops: Op | None, output_type: torch.dtype = torch.get_default_dtype()
    ):
        super().__init__()  # type: ignore
        clean_ops = [op for op in ops if op is not None]
        for op in clean_ops:
            if op.original_source is None:
                op.original_source = self
        self.op_owner = self
        self.ops = list(clean_ops)
        self.output_type = output_type
        self.pipeline = None
        self.prev_compile_args = None

    def parameters(self, recurse: bool = True):
        self._find_owner()._pre_compile()
        return super().parameters(recurse)

    def named_parameters(
        self, prefix: str = "", recurse: bool = True, remove_duplicate: bool = True
    ):
        self._find_owner()._pre_compile()
        return super().named_parameters(prefix, recurse, remove_duplicate)

    def buffers(self, recurse: bool = True):
        self._find_owner()._pre_compile()
        return super().buffers(recurse)

    def named_buffers(
        self, prefix: str = "", recurse: bool = True, remove_duplicate: bool = True
    ):
        self._find_owner()._pre_compile()
        return super().named_buffers(prefix, recurse, remove_duplicate)

    def _find_owner(self) -> ComputePipelineModuleBase:
        while self.op_owner != self.op_owner.op_owner:
            self.op_owner = self.op_owner.op_owner
        return self.op_owner

    def _pre_compile(self):
        cur_size = _convert_env(get_current_environment()).world_size()
        assert self.distributed_world_size in [None, cur_size]
        if self.distributed_world_size is None:
            self.distributed_world_size = cur_size
            self.pipeline = ComputePipeline(self.ops, cur_size)
            for param_tensor, param_name, parent_op in self.pipeline.parameters():
                assert isinstance(param_tensor, PytorchTensor)
                assert isinstance(parent_op.original_source, nn.Module)
                parent_op.original_source.register_parameter(
                    param_name, nn.Parameter(param_tensor.tensor)
                )
            for buffer_tensor, buffer_name, parent_op in self.pipeline.buffers():
                assert isinstance(buffer_tensor, PytorchTensor)
                assert isinstance(parent_op.original_source, nn.Module)
                parent_op.original_source.register_buffer(
                    buffer_name, buffer_tensor.tensor
                )

    def forward(self, x: torch.Tensor):
        assert self.op_owner is self

        if self.pipeline is None:
            self._pre_compile()
            assert self.pipeline is not None
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
            self._compile(input_type, output_type)

    def _compile(self, input_type: torch.dtype, output_type: torch.dtype) -> None:
        assert self.pipeline is not None
        self.pipeline.compile(
            DType.from_torch_dtype(input_type), DType.from_torch_dtype(output_type)
        )
