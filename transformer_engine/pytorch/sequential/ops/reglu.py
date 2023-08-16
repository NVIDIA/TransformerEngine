from __future__ import annotations
from .. import nvte
from .op import Grads, Op, Context


class ReGLU(Op):
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

        y = nvte.reglu(x, self.y_dtype)

        return y, {"x": x}

    def backward(self, ctx: Context, dy: nvte.Tensor):
        x = ctx["x"]
        dy = nvte.cast_checked(dy, self.dy_dtype)

        dx = nvte.dreglu(dy, x, self.dx_dtype)

        return dx, Grads()

    def require_grad(self):
        return list[nvte.Tensor]()


__all__ = ["ReGLU"]
