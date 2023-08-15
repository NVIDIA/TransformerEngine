from __future__ import annotations
from .. import nvte
from .op import Op, Context


class RMSNorm(Op):
    def __init__(
        self,
        eps: float,
        zero_centered_gamma: bool,
        weight: nvte.Tensor,
        x_dtype: nvte.DType | None = nvte.DType.BFloat16,
        weight_dtype: nvte.DType | None = nvte.DType.Float8E4M3,
        dy_dtype: nvte.DType | None = None,
        y_dtype: nvte.DType = nvte.DType.Float8E4M3,
        dx_dtype: nvte.DType = nvte.DType.BFloat16,
        dweight_dtype: nvte.DType = nvte.DType.BFloat16,
    ):
        self.eps = eps
        self.zero_centered_gamma = zero_centered_gamma
        self.weight = weight
        self.x_dtype = x_dtype
        self.weight_dtype = weight_dtype
        self.dy_dtype = dy_dtype
        self.y_dtype = y_dtype
        self.dx_dtype = dx_dtype
        self.dweight_dtype = dweight_dtype

    def inference(self, x: nvte.Tensor):
        return self.forward(x)[0]

    def forward(self, x: nvte.Tensor):
        x = nvte.cast_checked(x, self.x_dtype)
        weight = nvte.cast_checked(self.weight, self.weight_dtype)

        y, rsigma = nvte.rmsnorm(
            x, self.eps, self.zero_centered_gamma, weight, self.y_dtype
        )

        return y, {"x": x, "weight": weight, "rsigma": rsigma}

    def backward(self, ctx: Context, dy: nvte.Tensor):
        x, weight, rsigma = ctx["x"], ctx["weight"], ctx["rsigma"]
        dy = nvte.cast_checked(dy, self.dy_dtype)

        dx, dweight = nvte.drmsnorm(
            dy,
            self.zero_centered_gamma,
            x,
            weight,
            rsigma,
            self.dx_dtype,
            self.dweight_dtype,
        )

        return dx, [dweight]

    def args(self):
        return [self.weight]


__all__ = ["RMSNorm"]
