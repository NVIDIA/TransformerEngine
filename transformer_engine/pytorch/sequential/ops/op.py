from __future__ import annotations
from abc import ABC, abstractmethod
from .. import nvte

Context = dict[str, nvte.Tensor]
Grads = list[nvte.Tensor]


class Op(ABC):
    @abstractmethod
    def inference(self, x: nvte.Tensor) -> nvte.Tensor:
        ...

    @abstractmethod
    def forward(self, x: nvte.Tensor) -> tuple[nvte.Tensor, Context]:
        ...

    @abstractmethod
    def backward(self, ctx: Context, dy: nvte.Tensor) -> tuple[nvte.Tensor, Grads]:
        ...

    @abstractmethod
    def require_grad(self) -> list[nvte.Tensor]:
        ...

    def __repr__(self):
        return self.__class__.__name__


__all__ = ["Op", "Context", "Grads"]
