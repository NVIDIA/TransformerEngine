from __future__ import annotations
from typing import TYPE_CHECKING, ContextManager
from .base import Op

if TYPE_CHECKING:
    from .op_graph import OpGraph


class OpFDropout(Op):
    def __init__(self, a: Op, p: float, rng_ctx: ContextManager[None]):
        super().__init__()
        self.a = a
        self.p = p
        self.rng_ctx = rng_ctx

    def backward_a(self, graph: OpGraph, grad: Op):
        del graph  # unused
        return grad
