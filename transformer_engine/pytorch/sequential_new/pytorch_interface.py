from __future__ import annotations
import torch
from torch import nn
from .framework_interface import FrameworkInterface


class PytorchInterface(nn.Module, FrameworkInterface[torch.Tensor]):
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

    def __getattr__(self, name: str) -> torch.Tensor | PytorchInterface:
        attr = super().__getattr__(name)
        if isinstance(attr, nn.Module):
            return PytorchInterface(attr)
        else:
            return attr

    def __setattr__(self, name: str, value: object) -> None:
        super().__setattr__(name, value)  # type: ignore
