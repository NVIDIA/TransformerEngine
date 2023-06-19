from __future__ import annotations
from typing import TYPE_CHECKING
from .base import Op

if TYPE_CHECKING:
    from .op_graph import OpGraph


class OpFLayerNormCore(Op):
    def __init__(self, a: Op, eps: float):
        super().__init__()
        self.a = a
        self.eps = eps

    def backward_a(self, graph: OpGraph, grad: Op):
        df = graph.df_layernorm_core_(self.a, self.eps)
        return graph.mul_(df, grad)


class OpDFLayerNormCore(Op):
    def __init__(self, a: Op, eps: float):
        super().__init__()
        self.a = a
        self.eps = eps
