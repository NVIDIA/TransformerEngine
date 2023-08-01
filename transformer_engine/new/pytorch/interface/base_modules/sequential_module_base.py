import torch
from torch import nn
from ..meta_modules.sequential import Sequential


class SequentialModuleBase(nn.Module):
    def __init__(self, *modules: nn.Module | None):
        super().__init__()  # type: ignore
        clean_modules = [module for module in modules if module is not None]
        self.seq = Sequential(*clean_modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.seq(x)
