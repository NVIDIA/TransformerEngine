from __future__ import annotations
from .. import nvte
from .op import Op, Context


class LayerNorm(Op):
    def __init__(
        self,
        eps: float,
        zero_centered_gamma: bool,
        weight: nvte.Tensor,
        bias: nvte.Tensor,
        *,
        x_dtype: nvte.DType | None = nvte.DType.BFloat16,
        weight_dtype: nvte.DType | None = nvte.DType.Float8E4M3,
        bias_dtype: nvte.DType | None = nvte.DType.Float8E4M3,
        dy_dtype: nvte.DType | None = None,
        y_dtype: nvte.DType | None = nvte.DType.Float8E4M3,
        dx_dtype: nvte.DType | None = nvte.DType.BFloat16,
        dweight_dtype: nvte.DType | None = nvte.DType.BFloat16,
        dbias_dtype: nvte.DType | None = nvte.DType.BFloat16,
    ):
        self.eps = eps
        self.zero_centered_gamma = zero_centered_gamma
        self.weight = weight
        self.bias = bias
        self._x_dtype = x_dtype
        self.weight_dtype = weight_dtype
        self.bias_dtype = bias_dtype
        self._dy_dtype = dy_dtype
        self._y_dtype = y_dtype
        self._dx_dtype = dx_dtype
        self.dweight_dtype = dweight_dtype
        self.dbias_dtype = dbias_dtype

    def forward(self, x: nvte.Tensor):
        x = nvte.cast_checked(x, self.x_dtype)
        weight = nvte.cast_checked(self.weight, self.weight_dtype)
        bias = nvte.cast_checked(self.bias, self.bias_dtype)

        y, mu, rsigma = nvte.layernorm(
            x,
            self.eps,
            self.zero_centered_gamma,
            weight,
            bias,
            self.y_dtype or x.dtype,
        )

        return y, {"x": x, "weight": weight, "mu": mu, "rsigma": rsigma}

    def backward(self, ctx: Context, dy: nvte.Tensor):
        x, weight, mu, rsigma = ctx["x"], ctx["weight"], ctx["mu"], ctx["rsigma"]
        dy = nvte.cast_checked(dy, self.dy_dtype)

        dx, dweight, dbias = nvte.dlayernorm(
            dy,
            self.zero_centered_gamma,
            x,
            weight,
            mu,
            rsigma,
            self.dx_dtype or dy.dtype,
            self.dweight_dtype or self.weight.dtype,
            self.dbias_dtype or self.bias.dtype,
        )

        return dx, [dweight, dbias]

    def require_grad(self):
        return [self.weight, self.bias]


__all__ = ["LayerNorm"]
