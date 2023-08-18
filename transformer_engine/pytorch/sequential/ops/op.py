from __future__ import annotations
from abc import ABC, abstractmethod
from .. import nvte

Context = dict[str, nvte.Tensor]
Grads = list[nvte.Tensor]


class Op(ABC):
    @abstractmethod
    def __init__(
        self,
        *,
        x_dtype: nvte.DType | None = None,
        y_dtype: nvte.DType | None = None,
        dy_dtype: nvte.DType | None = None,
        dx_dtype: nvte.DType | None = None,
    ):
        ...

    def inference(self, x: nvte.Tensor, /):
        return self.forward(x)[0]

    @abstractmethod
    def forward(self, x: nvte.Tensor, /) -> tuple[nvte.Tensor, Context]:
        ...

    @abstractmethod
    def backward(self, ctx: Context, dy: nvte.Tensor, /) -> tuple[nvte.Tensor, Grads]:
        ...

    @abstractmethod
    def require_grad(self) -> list[nvte.Tensor]:
        ...

    def __repr__(self):
        return self.__class__.__name__

    @property
    def x_dtype(self):
        return self._x_dtype

    @property
    def y_dtype(self):
        return self._y_dtype or self.x_dtype

    @property
    def dy_dtype(self):
        return self._dy_dtype

    @property
    def dx_dtype(self):
        return self._dx_dtype or self._dy_dtype

    _x_dtype: nvte.DType | None
    _y_dtype: nvte.DType | None
    _dy_dtype: nvte.DType | None
    _dx_dtype: nvte.DType | None


__all__ = ["Op", "Context", "Grads"]
