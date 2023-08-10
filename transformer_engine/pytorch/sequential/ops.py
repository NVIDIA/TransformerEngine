from __future__ import annotations
from abc import ABC, abstractmethod
import ast
import typing
from typing import Any, Callable
from typing_extensions import Unpack, TypeVarTuple
import transformer_engine_cuda as nvte
from . import nvte_utils

TensorProvider = Callable[[], nvte.Tensor]
TensorRecipient = Callable[[nvte.Tensor], None]
Context = dict[str, nvte.Tensor]
Grads = tuple[nvte.Tensor | None, ...]

FUSIONS_INF: dict[tuple[type, ...], Callable[..., Any]] = {}
FUSIONS_FWD: dict[tuple[type, ...], Callable[..., Any]] = {}
FUSIONS_BWD: dict[tuple[type, ...], Callable[..., Any]] = {}


def get_parameters(*param: nvte.Tensor | TensorProvider):
    return tuple(p if isinstance(p, nvte.Tensor) else p() for p in param)


def return_grads(*grad: tuple[nvte.Tensor, TensorRecipient | None]):
    return tuple(t if rec is None else rec(t) for t, rec in grad)


Ops = TypeVarTuple("Ops")
OpsAndCtxs = TypeVarTuple("OpsAndCtxs")


def _get_arg_types(f: Callable[..., Any]):
    annotations = typing.get_type_hints(f)
    annotations.pop("return", None)
    arg_type_names: tuple[str] = tuple(annotations.values())
    assert all(
        isinstance(val, str) for val in arg_type_names
    )  # True due to __future__.annotations
    arg_types: tuple[type] = tuple(ast.literal_eval(val) for val in arg_type_names)
    return arg_types


def register_fusion_inference(f: Callable[[Unpack[Ops], nvte.Tensor], nvte.Tensor]):
    fused_modules = _get_arg_types(f)[:-1]
    FUSIONS_INF[fused_modules] = f
    return f


def register_fusion_forward(
    f: Callable[
        [Unpack[Ops], nvte.Tensor],
        tuple[nvte.Tensor, Unpack[tuple[Context, ...]]],
    ]
):
    fused_modules = _get_arg_types(f)[:-1]
    FUSIONS_FWD[fused_modules] = f
    return f


def register_fusion_backward(
    f: Callable[
        [Unpack[OpsAndCtxs], nvte.Tensor],
        tuple[nvte.Tensor, Unpack[tuple[Grads, ...]]],
    ]
):
    arg_types = _get_arg_types(f)
    module_count = (len(arg_types) - 1) / 2
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
        weight: nvte.Tensor | TensorProvider,
        dweight_r: TensorRecipient | None = None,
        x_dtype: nvte.DType | None = None,
        weight_dtype: nvte.DType | None = nvte.DType.Float8E4M3,
        dy_dtype: nvte.DType | None = nvte.DType.Float8E5M2,
        y_dtype: nvte.DType = nvte.DType.Float8E4M3,
        dx_dtype: nvte.DType = nvte.DType.BFloat16,
        dweight_dtype: nvte.DType = nvte.DType.BFloat16,
    ):
        self.weight = weight
        self.dweight_r = dweight_r
        self.x_dtype = x_dtype
        self.weight_dtype = weight_dtype
        self.dy_dtype = dy_dtype
        self.y_dtype = y_dtype
        self.dx_dtype = dx_dtype
        self.dweight_dtype = dweight_dtype

    def inference(self, x: nvte.Tensor):
        (weight,) = get_parameters(self.weight)
        x = nvte_utils.cast_checked(x, self.x_dtype)
        weight = nvte_utils.cast_checked(weight, self.weight_dtype)

        y = nvte_utils.matmul_transpose(x, weight, self.y_dtype)

        return y

    def forward(self, x: nvte.Tensor):
        (weight,) = get_parameters(self.weight)
        (x, x_t), (weight, weight_t) = nvte_utils.multi_cast_transpose_checked(
            (x, self.x_dtype), (weight, self.weight_dtype)
        )

        y = nvte_utils.matmul_transpose(x, weight, self.y_dtype)

        return y, {"x_t": x_t, "weight_t": weight_t}

    def backward(self, ctx: Context, dy: nvte.Tensor):
        x_t, weight_t = ctx["x_t"], ctx["weight_t"]
        dy, dy_t = nvte_utils.cast_transpose_checked(dy, self.dy_dtype)

        dx = nvte_utils.matmul_transpose(dy, weight_t, self.dx_dtype)
        dweight = nvte_utils.matmul_transpose(x_t, dy_t, self.dweight_dtype)

        return dx, return_grads((dweight, self.dweight_r))

    def args(self):
        return [*get_parameters(self.weight)]


class Add(Op):
    def __init__(
        self,
        bias: nvte.Tensor | TensorProvider,
        dbias_r: TensorRecipient | None = None,
        x_dtype: nvte.DType | None = None,
        bias_dtype: nvte.DType | None = nvte.DType.Float8E4M3,
        dy_dtype: nvte.DType | None = nvte.DType.Float8E5M2,
        y_dtype: nvte.DType = nvte.DType.Float8E4M3,
        dx_dtype: nvte.DType = nvte.DType.BFloat16,
        dbias_dtype: nvte.DType = nvte.DType.BFloat16,
    ):
        self.bias = bias
        self.dbias_r = dbias_r
        self.x_dtype = x_dtype
        self.bias_dtype = bias_dtype
        self.dy_dtype = dy_dtype
        self.y_dtype = y_dtype
        self.dx_dtype = dx_dtype
        self.dbias_dtype = dbias_dtype

    def inference(self, x: nvte.Tensor):
        return self.forward(x)[0]

    def forward(self, x: nvte.Tensor):
        (bias,) = get_parameters(self.bias)
        x = nvte_utils.cast_checked(x, self.x_dtype)
        bias = nvte_utils.cast_checked(bias, self.bias_dtype)

        y = nvte_utils.add(x, bias, self.y_dtype)

        return y, Context()

    def backward(self, ctx: dict[str, nvte.Tensor], dy: nvte.Tensor):
        del ctx
        dy = nvte_utils.cast_checked(dy, self.dy_dtype)

        dx = nvte_utils.cast_checked(dy, self.dx_dtype)
        dbias = nvte_utils.dbias(dy, self.dbias_dtype)

        return dx, return_grads((dbias, self.dbias_r))

    def args(self):
        return [*get_parameters(self.bias)]


@register_fusion_inference
def _(mmt: MMT, add: Add, x: nvte.Tensor):
    (weight, bias) = get_parameters(mmt.weight, add.bias)
    x = nvte_utils.cast_checked(x, mmt.x_dtype)
    weight = nvte_utils.cast_checked(weight, mmt.weight_dtype)
    bias = nvte_utils.cast_checked(bias, add.bias_dtype)

    y = nvte_utils.matmul_transpose_add(x, weight, bias, add.y_dtype)

    return y


@register_fusion_forward
def _(mmt: MMT, add: Add, x: nvte.Tensor):
    (weight, bias) = get_parameters(mmt.weight, add.bias)
    (x, x_t), (weight, weight_t) = nvte_utils.multi_cast_transpose_checked(
        (x, mmt.x_dtype), (weight, mmt.weight_dtype)
    )
    bias = nvte_utils.cast_checked(bias, add.bias_dtype)

    y = nvte_utils.matmul_transpose_add(x, weight, bias, add.y_dtype)

    return y, {"x_t": x_t, "weight_t": weight_t}, Context()


@register_fusion_backward
def _(
    mmt: MMT,
    add: Add,
    mmt_ctx: dict[str, nvte.Tensor],
    add_ctx: dict[str, nvte.Tensor],
    dy: nvte.Tensor,
):
    del add_ctx
    x_t, weight_t = mmt_ctx["x_t"], mmt_ctx["weight_t"]
    dy, dy_t, dbias = nvte_utils.cast_transpose_dbias_checked(
        dy, mmt.dy_dtype, add.dbias_dtype
    )

    dx = nvte_utils.matmul_transpose(dy, weight_t, mmt.dx_dtype)
    dweight = nvte_utils.matmul_transpose(x_t, dy_t, mmt.dweight_dtype)

    return (
        dx,
        return_grads((dweight, mmt.dweight_r)),
        return_grads((dbias, add.dbias_r)),
    )
