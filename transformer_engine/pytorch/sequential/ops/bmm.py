from __future__ import annotations
from typing import TYPE_CHECKING
from .base import Op

if TYPE_CHECKING:
    from .op_graph import OpGraph


class OpBMM(Op):
    def __init__(self, a: Op, b: Op):
        super().__init__()
        self.a = a
        self.b = b

    def backward_a(self, graph: OpGraph, grad: Op):
        bT = graph.t_(self.b)
        return graph.bmm_(bT, grad)

    def backward_b(self, graph: OpGraph, grad: Op):
        aT = graph.t_(self.a)
        return graph.bmm_(aT, grad)
