from __future__ import annotations
import torch
from torch import nn
from .framework_interface import FrameworkInterface


class PytorchInterface(FrameworkInterface[torch.Tensor], nn.Module):
    @staticmethod
    def fi_empty(shape: tuple[int, ...]):
        return torch.empty(shape)

    @staticmethod
    def fi_zeros(shape: tuple[int, ...]):
        return torch.zeros(shape)

    @staticmethod
    def fi_ones(shape: tuple[int, ...]):
        return torch.ones(shape)

    @staticmethod
    def fi_normal(mean: float, std: float, shape: tuple[int, ...]):
        return torch.normal(mean, std, shape)

    @staticmethod
    def fi_uniform(min: float, max: float, shape: tuple[int, ...]):
        return torch.rand(shape) * (max - min) + min

    def fi_register_buffer(self, name: str, tensor: torch.Tensor):
        name = name.replace(".", "_")
        self.register_buffer(name, tensor, False)
