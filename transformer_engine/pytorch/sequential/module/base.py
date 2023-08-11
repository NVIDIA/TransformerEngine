import torch
from torch import nn
from ...distributed import get_distributed_world_size
from ...fp8 import is_fp8_enabled
from ..ops import Op
from ..environment import Environment
from ..compute_pipeline import ComputePipeline
from ..compute_pipeline_function import apply


class BaseModule(nn.Module):
    ops: list[Op]
    pipeline: ComputePipeline | None
    compile_env: Environment | None

    def __init__(self, *ops: Op | None):
        "Note: nn.Module.__init__ must be called by the derived class"
        ops_clean = [op for op in ops if op is not None]
        self.ops = ops_clean
        self.pipeline = None
        self.compile_env = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        env = self._current_env()
        if self.pipeline is None or env != self.compile_env:
            self.pipeline = ComputePipeline(self.ops, env)
            self.compile_env = env
        return apply(x, self.pipeline, self.training)

    def _current_env(self) -> Environment:
        return Environment(is_fp8_enabled(), get_distributed_world_size())
