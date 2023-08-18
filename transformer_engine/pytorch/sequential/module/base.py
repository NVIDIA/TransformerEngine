import torch
from torch import nn
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

    def forward(
        self, x: torch.Tensor, seq_lens: torch.Tensor | None = None
    ) -> torch.Tensor:
        if seq_lens is None:
            if x.dim() == 2:
                seq_lens = torch.tensor([x.shape[0]], dtype=torch.int32, device="cuda")
            elif x.dim() == 3:
                seq_lens = torch.tensor(
                    [x.shape[1]] * x.shape[0], dtype=torch.int32, device="cuda"
                )
                x = x.view(x.shape[1] * x.shape[0], x.shape[2])
            else:
                raise ValueError(f"Unsupported input shape: {x.shape}")
        else:
            assert x.dim() == 2
            assert x.shape[0] == seq_lens.sum().item()
        assert x.is_cuda
        assert seq_lens.is_cuda
        assert x.is_contiguous()
        assert seq_lens.is_contiguous()

        env = self._current_env()
        if self.pipeline is None or env != self.compile_env:
            self.pipeline = ComputePipeline(self.ops, env)
            self.compile_env = env
        return apply(x, self.pipeline, self.training)

    def _current_env(self) -> Environment:
        return Environment.current()
