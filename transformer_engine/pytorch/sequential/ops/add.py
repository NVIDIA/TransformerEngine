from __future__ import annotations
from .. import nvte
from .op import Op, Context


class Add(Op):
    def __init__(
        self,
        bias: nvte.Tensor,
        *,
        x_dtype: nvte.DType | None = None,
        bias_dtype: nvte.DType | None = nvte.DType.Float8E4M3,
        dy_dtype: nvte.DType | None = None,
        y_dtype: nvte.DType | None = nvte.DType.Float8E4M3,
        dx_dtype: nvte.DType | None = nvte.DType.BFloat16,
        dbias_dtype: nvte.DType | None = nvte.DType.BFloat16,
    ):
        self.bias = bias
        self._x_dtype = x_dtype
        self.bias_dtype = bias_dtype
        self._dy_dtype = dy_dtype
        self._y_dtype = y_dtype
        self._dx_dtype = dx_dtype
        self.dbias_dtype = dbias_dtype

    def forward(self, x: nvte.Tensor):
        x = nvte.cast_checked(x, self.x_dtype)
        bias = nvte.cast_checked(self.bias, self.bias_dtype)

        y = nvte.add(x, bias, self.y_dtype or x.dtype)

        return y, Context()

    def backward(self, ctx: Context, dy: nvte.Tensor):
        del ctx
        dy = nvte.cast_checked(dy, self.dy_dtype)

        dx = nvte.cast_checked(dy, self.dx_dtype)
        dbias = nvte.dbias(dy, self.dbias_dtype or self.bias.dtype)

        return dx, [dbias]

    def require_grad(self):
        return [self.bias]


__all__ = ["Add"]
