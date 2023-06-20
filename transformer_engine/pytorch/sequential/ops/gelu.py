from __future__ import annotations
from typing import TYPE_CHECKING
from .base import Op

if TYPE_CHECKING:
    from .op_graph import OpGraph


class OpFGelu(Op):
    def __init__(self, a: Op):
        super().__init__()
        self.a = a

    def backward_a(self, graph: OpGraph, grad: Op):
        df = graph.df_gelu_(self.a)
        return graph.mul_(df, grad)


class OpDFGelu(Op):
    def __init__(self, a: Op):
        super().__init__()
        self.a = a
