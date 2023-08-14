from __future__ import annotations
from .. import nvte
from .op import Op, Context


class MMT(Op):
    def __init__(
        self,
        weight: nvte.Tensor,
        x_dtype: nvte.DType | None = nvte.DType.Float8E4M3,
        weight_dtype: nvte.DType | None = nvte.DType.Float8E4M3,
        dy_dtype: nvte.DType | None = nvte.DType.Float8E5M2,
        y_dtype: nvte.DType = nvte.DType.Float8E4M3,
        dx_dtype: nvte.DType = nvte.DType.BFloat16,
        dweight_dtype: nvte.DType = nvte.DType.BFloat16,
    ):
        self.weight = weight
        self.x_dtype = x_dtype
        self.weight_dtype = weight_dtype
        self.dy_dtype = dy_dtype
        self.y_dtype = y_dtype
        self.dx_dtype = dx_dtype
        self.dweight_dtype = dweight_dtype

    def inference(self, x: nvte.Tensor):
        x = nvte.cast_checked(x, self.x_dtype)
        weight = nvte.cast_checked(self.weight, self.weight_dtype)

        y = nvte.matmul_transpose(x, weight, self.y_dtype)

        return y

    def forward(self, x: nvte.Tensor):
        (x, x_t), (weight, weight_t) = nvte.multi_cast_transpose_checked(
            (x, self.x_dtype), (self.weight, self.weight_dtype)
        )

        y = nvte.matmul_transpose(x, weight, self.y_dtype)

        return y, {"x_t": x_t, "weight_t": weight_t}

    def backward(self, ctx: Context, dy: nvte.Tensor):
        x_t, weight_t = ctx["x_t"], ctx["weight_t"]
        dy, dy_t = nvte.cast_transpose_checked(dy, self.dy_dtype)

        dx = nvte.matmul_transpose(dy, weight_t, self.dx_dtype)
        dweight = nvte.matmul_transpose(x_t, dy_t, self.dweight_dtype)

        return dx, [dweight]

    def args(self):
        return [self.weight]


__all__ = ["MMT"]
