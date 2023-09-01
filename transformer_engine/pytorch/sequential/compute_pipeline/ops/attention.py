from __future__ import annotations
from ...utils import prevent_import

prevent_import("torch")
from typing import Callable
from abc import ABC
from ... import nvte
from .op import Grads, Op, Context


class DotProductAttention(Op, ABC):
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

    def forward(self, qkv_packed: nvte.Tensor):
        ...  # TODO
