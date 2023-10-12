from __future__ import annotations
from abc import ABC, abstractmethod
import torch
from torch import nn
from ..compute_pipeline.ops import Op
from ..recipe import Recipe
from ..compute_pipeline.compute_pipeline import ComputePipeline
from ..compute_pipeline_function import apply


class BaseModule(nn.Module, ABC):
    pipeline: ComputePipeline | None
    compile_env: Recipe | None

    @abstractmethod
    def _ops(self) -> list[Op | None]:
        ...

    def __init__(self):
        super().__init__()  # type: ignore
        self.pipeline = None
        self.compile_env = None

    def forward(
        self, x: torch.Tensor, seq_lens: torch.Tensor | None = None
    ) -> torch.Tensor:
        self._precompiled_for(x, seq_lens)
        return self._run(x)

    def _precompiled_for(self, x: torch.Tensor, seq_lens: torch.Tensor | None = None):
        with torch.no_grad():
            assert x.is_cuda
            assert x.is_contiguous()
            if seq_lens is None:
                seq_lens = BaseModule._create_seq_lens_tensor(x)
            assert seq_lens.is_cuda
            assert seq_lens.is_contiguous()

            self._setup_pipeline(x, seq_lens)

        return self._run

    def _run(self, x: torch.Tensor):
        assert self.pipeline is not None
        return apply(x, self.pipeline, self.training)

    @staticmethod
    def _create_seq_lens_tensor(x: torch.Tensor):
        if x.dim() == 2:
            seq_lens = torch.tensor([x.shape[0]], dtype=torch.int32, device="cuda")
        elif x.dim() == 3:
            seq_lens = torch.tensor(
                [x.shape[1]] * x.shape[0], dtype=torch.int32, device="cuda"
            )
            x = x.view(x.shape[1] * x.shape[0], x.shape[2])
        else:
            raise ValueError(f"Unsupported input shape: {x.shape}")
        return seq_lens

    def _setup_pipeline(self, x: torch.Tensor, seq_lens: torch.Tensor):
        del x, seq_lens  # TODO: take x's type into account, save seq_lens
        env = self._current_env()
        if self.pipeline is None or env != self.compile_env:
            self.pipeline = ComputePipeline(
                [op for op in self._ops() if op is not None], env
            )
            self.compile_env = env

    def _current_env(self) -> Recipe:
        return Recipe.current()
