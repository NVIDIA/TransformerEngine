from __future__ import annotations
from abc import ABC, abstractmethod
import transformer_engine_cuda as _nvte  # pylint: disable=import-error

Context = dict[str, _nvte.Tensor]
Grads = list[_nvte.Tensor]

class Op(ABC):
    @abstractmethod
    def inference(self, x: _nvte.Tensor) -> _nvte.Tensor:
        ...

    @abstractmethod
    def forward(self, x: _nvte.Tensor) -> tuple[_nvte.Tensor, Context]:
        ...

    @abstractmethod
    def backward(self, ctx: Context, dy: _nvte.Tensor) -> tuple[_nvte.Tensor, Grads]:
        ...

    @abstractmethod
    def args(self) -> list[_nvte.Tensor]:
        ...

    def __repr__(self):
        return self.__class__.__name__


__all__ = ["Op", "Context", "Grads"]
