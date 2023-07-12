import torch
from torch import nn
from .framework_interface import FrameworkInterface


class PytorchInterface(FrameworkInterface[torch.Tensor], nn.Module):
    @staticmethod
    def fi_empty(shape: tuple[int, ...]):
        return torch.empty(shape)

    def fi_register_buffer(self, name: str, tensor: torch.Tensor):
        name = name.replace(".", "_")
        self.register_buffer(name, tensor, False)
