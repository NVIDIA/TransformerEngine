from __future__ import annotations
from typing import Callable
from abc import ABC
from .. import nvte
from .op import Grads, Op, Context


class Activation(Op, ABC):
    def __init__(
        self,
        x_dtype: nvte.DType | None = None,
        dy_dtype: nvte.DType | None = nvte.DType.Float8E5M2,
        y_dtype: nvte.DType = nvte.DType.Float8E4M3,
        dx_dtype: nvte.DType = nvte.DType.BFloat16,
    ):
        self.x_dtype = x_dtype
        self.dy_dtype = dy_dtype
        self.y_dtype = y_dtype
        self.dx_dtype = dx_dtype

    def inference(self, x: nvte.Tensor):
        return self.forward(x)[0]

    def forward(self, x: nvte.Tensor):
        x = nvte.cast_checked(x, self.x_dtype)

        y = type(self)._forward(x, self.y_dtype)

        return y, {"x": x}

    def backward(self, ctx: Context, dy: nvte.Tensor):
        x = ctx["x"]
        dy = nvte.cast_checked(dy, self.dy_dtype)

        dx = type(self)._backward(dy, x, self.dx_dtype)

        return dx, Grads()

    def require_grad(self):
        return list[nvte.Tensor]()

    _forward: Callable[[nvte.Tensor, nvte.DType], nvte.Tensor]
    _backward: Callable[[nvte.Tensor, nvte.Tensor, nvte.DType], nvte.Tensor]


class ReLU(Activation):
    _forward = nvte.relu
    _backward = nvte.drelu


class GELU(Activation):
    _forward = nvte.gelu
    _backward = nvte.dgelu


class ReGLU(Activation):
    _forward = nvte.reglu
    _backward = nvte.dreglu


class GeGLU(Activation):
    _forward = nvte.geglu
    _backward = nvte.dgeglu


class SwiGLU(Activation):
    _forward = nvte.swiglu
    _backward = nvte.dswiglu


__all__ = [
    "Activation",
    "ReLU",
    "GELU",
    "ReGLU",
    "GeGLU",
    "SwiGLU",
]
