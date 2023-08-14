from __future__ import annotations
import transformer_engine_cuda as _nvte  # pylint: disable=import-error
from .. import nvte
from .op import Op, Context


class Add(Op):
    def __init__(
        self,
        bias: _nvte.Tensor,
        x_dtype: _nvte.DType | None = None,
        bias_dtype: _nvte.DType | None = _nvte.DType.Float8E4M3,
        dy_dtype: _nvte.DType | None = _nvte.DType.Float8E5M2,
        y_dtype: _nvte.DType = _nvte.DType.Float8E4M3,
        dx_dtype: _nvte.DType = _nvte.DType.BFloat16,
        dbias_dtype: _nvte.DType = _nvte.DType.BFloat16,
    ):
        self.bias = bias
        self.x_dtype = x_dtype
        self.bias_dtype = bias_dtype
        self.dy_dtype = dy_dtype
        self.y_dtype = y_dtype
        self.dx_dtype = dx_dtype
        self.dbias_dtype = dbias_dtype

    def inference(self, x: _nvte.Tensor):
        return self.forward(x)[0]

    def forward(self, x: _nvte.Tensor):
        x = nvte.cast_checked(x, self.x_dtype)
        bias = nvte.cast_checked(self.bias, self.bias_dtype)

        y = nvte.add(x, bias, self.y_dtype)

        return y, Context()

    def backward(self, ctx: Context, dy: _nvte.Tensor):
        del ctx
        dy = nvte.cast_checked(dy, self.dy_dtype)

        dx = nvte.cast_checked(dy, self.dx_dtype)
        dbias = nvte.dbias(dy, self.dbias_dtype)

        return dx, [dbias]

    def args(self):
        return [self.bias]


__all__ = ["Add"]
