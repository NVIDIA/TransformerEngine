from __future__ import annotations
from typing import TYPE_CHECKING
from .base import Op

if TYPE_CHECKING:
    from .op_graph import OpGraph


class OpAdd(Op):
    def __init__(self, a: Op, b: Op):
        super().__init__()
        self.a = a
        self.b = b

    def backward_a(self, graph: OpGraph, grad: Op):
        del graph  # unused
        return grad

    def backward_b(self, graph: OpGraph, grad: Op):
        del graph  # unused
        return grad
