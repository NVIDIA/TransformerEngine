from __future__ import annotations

from transformer_engine.pytorch.sequential import nvte

from . import Op, Grads, Context
from . import Add
from ... import nvte


class ResidualBegin(Op):
    end: ResidualEnd
    residual_backward: nvte.Tensor

    def __init__(
        self,
        *,
        x_dtype: nvte.DType | None = nvte.DType.BFloat16,
        dy_dtype: nvte.DType | None = nvte.DType.BFloat16,
        y_dtype: nvte.DType | None = nvte.DType.BFloat16,
        dx_dtype: nvte.DType | None = nvte.DType.BFloat16,
    ):
        self._x_dtype = x_dtype
        self._dy_dtype = dy_dtype
        self._y_dtype = y_dtype
        self._dx_dtype = dx_dtype

    def forward(self, x: nvte.Tensor) -> tuple[nvte.Tensor, Context]:
        x = nvte.cast_checked(x, self.x_dtype)
        self.end.residual_forward = x
        y = nvte.cast_checked(x, self.y_dtype)
        return y, {}

    def backward(self, ctx: Context, dy: nvte.Tensor) -> tuple[nvte.Tensor, Grads]:
        del ctx
        dy = nvte.cast_checked(dy, self.dy_dtype)
        dx = nvte.add(dy, self.residual_backward, self.dx_dtype or dy.dtype)
        del self.residual_backward
        return dx, []

    def require_grad(self) -> list[nvte.Tensor]:
        return []


class ResidualEnd(Op):
    begin: ResidualBegin
    residual_forward: nvte.Tensor

    def __init__(
        self,
        *,
        x_dtype: nvte.DType | None = nvte.DType.BFloat16,
        dy_dtype: nvte.DType | None = nvte.DType.BFloat16,
        y_dtype: nvte.DType | None = nvte.DType.BFloat16,
        dx_dtype: nvte.DType | None = nvte.DType.BFloat16,
    ):
        self._x_dtype = x_dtype
        self._dy_dtype = dy_dtype
        self._y_dtype = y_dtype
        self._dx_dtype = dx_dtype

    def forward(self, x: nvte.Tensor) -> tuple[nvte.Tensor, Context]:
        x = nvte.cast_checked(x, self.x_dtype)
        y = nvte.add(x, self.residual_forward, self.y_dtype or x.dtype)
        del self.residual_forward
        return y, {}

    def backward(self, ctx: Context, dy: nvte.Tensor) -> tuple[nvte.Tensor, Grads]:
        del ctx
        dy = nvte.cast_checked(dy, self.dy_dtype)
        self.begin.residual_backward = dy
        dx = nvte.cast_checked(dy, self.dx_dtype)
        return dx, []

    def require_grad(self) -> list[nvte.Tensor]:
        return []

    @property
    def bias(self):
        return self.residual_forward

    @property
    def fusion_type(self):
        return super().fusion_type | {
            "forward": Add,
        }
