from __future__ import annotations
from typing import TYPE_CHECKING
from .base import Op

if TYPE_CHECKING:
    from .op_graph import OpGraph


class OpScale(Op):
    def __init__(self, a: Op, scale: float):
        super().__init__()
        self.a = a
        self.scale = scale

    def backward_a(self, graph: OpGraph, grad: Op):
        return graph.scale_(self.scale, grad)
