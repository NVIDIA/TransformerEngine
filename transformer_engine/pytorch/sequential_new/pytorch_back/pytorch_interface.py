from __future__ import annotations
import torch
from torch import nn
from ..common_back.framework_interface import FrameworkInterface
from ..common_back.enums import DType

_types = {
    DType.FP8E4M3: torch.uint8,
    DType.FP8E5M2: torch.uint8,
    DType.FP16: torch.float16,
    DType.BF16: torch.bfloat16,
    DType.FP32: torch.float32,
}


class PytorchInterface(FrameworkInterface[torch.Tensor], nn.Module):
    Tensor: type[torch.Tensor] = torch.Tensor

    @staticmethod
    def fi_empty(shape: tuple[int, ...], dtype: DType):
        return torch.empty(shape, dtype=_types[dtype])

    @staticmethod
    def fi_zeros(
        shape: tuple[int, ...] | None,
        dtype: DType | None,
        out: torch.Tensor | None,
    ):
        if out is None:
            assert shape is not None and dtype is not None
            return torch.zeros(shape, dtype=_types[dtype])
        else:
            out.zero_()

    @staticmethod
    def fi_ones(
        shape: tuple[int, ...] | None, dtype: DType | None, out: torch.Tensor | None
    ):
        if out is None:
            assert shape is not None and dtype is not None
            return torch.ones(shape, dtype=_types[dtype])
        else:
            out.fill_(1.0)

    @staticmethod
    def fi_normal(
        mean: float,
        std: float,
        shape: tuple[int, ...] | None,
        dtype: DType | None,
        out: torch.Tensor | None,
    ):
        if out is None:
            assert shape is not None and dtype is not None
            return torch.normal(mean, std, shape, dtype=_types[dtype])
        else:
            out.normal_(mean, std)

    @staticmethod
    def fi_uniform(
        min: float,
        max: float,
        shape: tuple[int, ...] | None,
        dtype: DType | None,
        out: torch.Tensor | None,
    ):
        if out is None:
            assert shape is not None and dtype is not None
            return torch.rand(shape, dtype=_types[dtype]) * (max - min) + min
        else:
            out.uniform_(min, max)

    def fi_register_buffer(self, name: str, tensor: torch.Tensor):
        name = name.replace(".", "_")
        self.register_buffer(name, tensor, False)
