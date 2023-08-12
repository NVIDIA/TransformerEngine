from __future__ import annotations
from abc import ABC, abstractmethod
import ast
import typing
from typing import Any, Callable, final
from typing_extensions import Unpack, TypeVarTuple
import transformer_engine_cuda as nvte
from . import nvte_utils

Context = dict[str, nvte.Tensor]
Grads = list[nvte.Tensor]

Forward = Callable[[nvte.Tensor], tuple[nvte.Tensor, Context]]
ForwardFused = Callable[[nvte.Tensor], tuple[nvte.Tensor, tuple[Context, ...]]]
Backward = Callable[[Context, nvte.Tensor], tuple[nvte.Tensor, Grads]]
BackwardFused = Callable[
    [Unpack[tuple[Context, ...]], nvte.Tensor], tuple[nvte.Tensor, tuple[Grads, ...]]
]
Inference = Callable[[nvte.Tensor], nvte.Tensor]

FUSIONS_INF: dict[tuple[type, ...], Callable[..., Any]] = {}
FUSIONS_FWD: dict[tuple[type, ...], Callable[..., Any]] = {}
FUSIONS_BWD: dict[tuple[type, ...], Callable[..., Any]] = {}

Ops = TypeVarTuple("Ops")
OpsAndCtxs = TypeVarTuple("OpsAndCtxs")


def _get_arg_types(f: Callable[..., Any]):
    annotations = typing.get_type_hints(f)
    annotations.pop("return", None)
    arg_type_annotations: tuple[str | type] = tuple(annotations.values())
    assert all(isinstance(val, (str, type)) for val in arg_type_annotations)
    arg_types: tuple[type] = tuple(
        ast.literal_eval(val) if isinstance(val, str) else val
        for val in arg_type_annotations
    )
    return arg_types


def register_fusion_inference(f: Callable[[Unpack[Ops], nvte.Tensor], nvte.Tensor]):
    fused_modules = _get_arg_types(f)[:-1]
    FUSIONS_INF[fused_modules] = f
    return f


def register_fusion_forward(
    f: Callable[
        [Unpack[Ops], nvte.Tensor],
        tuple[nvte.Tensor, tuple[Context, ...]],
    ]
):
    fused_modules = _get_arg_types(f)[:-1]
    FUSIONS_FWD[fused_modules] = f
    return f


def register_fusion_backward(
    f: Callable[
        [Unpack[OpsAndCtxs], nvte.Tensor],
        tuple[nvte.Tensor, tuple[Grads, ...]],
    ]
):
    arg_types = _get_arg_types(f)
    module_count = (len(arg_types) - 1) // 2
    fused_modules = arg_types[:module_count]
    FUSIONS_BWD[fused_modules] = f
    return f


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
    def args(self) -> list[nvte.Tensor]:
        ...


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
        x = nvte_utils.cast_checked(x, self.x_dtype)
        weight = nvte_utils.cast_checked(self.weight, self.weight_dtype)

        y = nvte_utils.matmul_transpose(x, weight, self.y_dtype)

        return y

    def forward(self, x: nvte.Tensor):
        (x, x_t), (weight, weight_t) = nvte_utils.multi_cast_transpose_checked(
            (x, self.x_dtype), (self.weight, self.weight_dtype)
        )

        y = nvte_utils.matmul_transpose(x, weight, self.y_dtype)

        return y, {"x_t": x_t, "weight_t": weight_t}

    def backward(self, ctx: Context, dy: nvte.Tensor):
        x_t, weight_t = ctx["x_t"], ctx["weight_t"]
        dy, dy_t = nvte_utils.cast_transpose_checked(dy, self.dy_dtype)

        dx = nvte_utils.matmul_transpose(dy, weight_t, self.dx_dtype)
        dweight = nvte_utils.matmul_transpose(x_t, dy_t, self.dweight_dtype)

        return dx, [dweight]

    def args(self):
        return [self.weight]


class Add(Op):
    def __init__(
        self,
        bias: nvte.Tensor,
        x_dtype: nvte.DType | None = None,
        bias_dtype: nvte.DType | None = nvte.DType.Float8E4M3,
        dy_dtype: nvte.DType | None = nvte.DType.Float8E5M2,
        y_dtype: nvte.DType = nvte.DType.Float8E4M3,
        dx_dtype: nvte.DType = nvte.DType.BFloat16,
        dbias_dtype: nvte.DType = nvte.DType.BFloat16,
    ):
        self.bias = bias
        self.x_dtype = x_dtype
        self.bias_dtype = bias_dtype
        self.dy_dtype = dy_dtype
        self.y_dtype = y_dtype
        self.dx_dtype = dx_dtype
        self.dbias_dtype = dbias_dtype

    def inference(self, x: nvte.Tensor):
        return self.forward(x)[0]

    def forward(self, x: nvte.Tensor):
        x = nvte_utils.cast_checked(x, self.x_dtype)
        bias = nvte_utils.cast_checked(self.bias, self.bias_dtype)

        y = nvte_utils.add(x, bias, self.y_dtype)

        return y, Context()

    def backward(self, ctx: Context, dy: nvte.Tensor):
        del ctx
        dy = nvte_utils.cast_checked(dy, self.dy_dtype)

        dx = nvte_utils.cast_checked(dy, self.dx_dtype)
        dbias = nvte_utils.dbias(dy, self.dbias_dtype)

        return dx, [dbias]

    def args(self):
        return [self.bias]


class LayerNorm(Op):
    def __init__(
        self,
        eps: float,
        zero_centered_gamma: bool,
        weight: nvte.Tensor,
        bias: nvte.Tensor,
        x_dtype: nvte.DType | None = None,
        weight_dtype: nvte.DType | None = nvte.DType.Float8E4M3,
        bias_dtype: nvte.DType | None = nvte.DType.Float8E4M3,
        dy_dtype: nvte.DType | None = None,
        y_dtype: nvte.DType = nvte.DType.Float8E4M3,
        dx_dtype: nvte.DType = nvte.DType.BFloat16,
        dweight_dtype: nvte.DType = nvte.DType.BFloat16,
        dbias_dtype: nvte.DType = nvte.DType.BFloat16,
    ):
        self.eps = eps
        self.zero_centered_gamma = zero_centered_gamma
        self.weight = weight
        self.bias = bias
        self.x_dtype = x_dtype
        self.weight_dtype = weight_dtype
        self.bias_dtype = bias_dtype
        self.dy_dtype = dy_dtype
        self.y_dtype = y_dtype
        self.dx_dtype = dx_dtype
        self.dweight_dtype = dweight_dtype
        self.dbias_dtype = dbias_dtype

    def inference(self, x: nvte.Tensor) -> nvte.Tensor:
        return self.forward(x)[0]

    def forward(self, x: nvte.Tensor):
        x = nvte_utils.cast_checked(x, self.x_dtype)
        weight = nvte_utils.cast_checked(self.weight, self.weight_dtype)
        bias = nvte_utils.cast_checked(self.bias, self.bias_dtype)

        y, mu, rsigma = nvte_utils.layernorm(
            x, self.eps, self.zero_centered_gamma, weight, bias, self.y_dtype
        )

        return y, {"x": x, "weight": weight, "mu": mu, "rsigma": rsigma}

    def backward(self, ctx: Context, dy: nvte.Tensor):
        x, weight, mu, rsigma = ctx["x"], ctx["weight"], ctx["mu"], ctx["rsigma"]
        dy = nvte_utils.cast_checked(dy, self.dy_dtype)

        dx, dweight, dbias = nvte_utils.dlayernorm(
            dy,
            self.zero_centered_gamma,
            x,
            weight,
            mu,
            rsigma,
            self.dx_dtype,
            self.dweight_dtype,
            self.dbias_dtype,
        )

        return dx, [dweight, dbias]

    def args(self):
        return [self.weight, self.bias]


@register_fusion_inference
def mmt_add_inf_fused(mmt: MMT, add: Add, x: nvte.Tensor):
    x = nvte_utils.cast_checked(x, mmt.x_dtype)
    weight = nvte_utils.cast_checked(mmt.weight, mmt.weight_dtype)
    bias = nvte_utils.cast_checked(add.bias, add.bias_dtype)

    y = nvte_utils.matmul_transpose_add(x, weight, bias, add.y_dtype)

    return y


@register_fusion_forward
def mmt_add_fwd_fused(mmt: MMT, add: Add, x: nvte.Tensor):
    (x, x_t), (weight, weight_t) = nvte_utils.multi_cast_transpose_checked(
        (x, mmt.x_dtype), (mmt.weight, mmt.weight_dtype)
    )
    bias = nvte_utils.cast_checked(add.bias, add.bias_dtype)

    y = nvte_utils.matmul_transpose_add(x, weight, bias, add.y_dtype)

    return y, ({"x_t": x_t, "weight_t": weight_t}, Context())


@register_fusion_backward
def mmt_add_bwd_fused(
    mmt: MMT,
    add: Add,
    mmt_ctx: Context,
    add_ctx: Context,
    dy: nvte.Tensor,
):
    del add_ctx
    x_t, weight_t = mmt_ctx["x_t"], mmt_ctx["weight_t"]
    dy, dy_t, dbias = nvte_utils.cast_transpose_dbias_checked(
        dy, mmt.dy_dtype, add.dbias_dtype
    )

    dx = nvte_utils.matmul_transpose(dy, weight_t, mmt.dx_dtype)
    dweight = nvte_utils.matmul_transpose(x_t, dy_t, mmt.dweight_dtype)

    return dx, ([dweight], [dbias])
