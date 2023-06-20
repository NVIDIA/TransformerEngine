from __future__ import annotations
from types import EllipsisType
from typing import TYPE_CHECKING
from .base import Op

if TYPE_CHECKING:
    from .op_graph import OpGraph


class OpView(Op):
    def __init__(self, a: Op, shape: list[int | EllipsisType]):
        super().__init__()
        self.a = a
        self.shape = shape
        self.reverse_shape: list[int | EllipsisType]

        if shape[0] == (...):
            reverse_shape = [0] * (len(shape) - 1)
            for i in range(1, len(shape)):
                reverse_shape[shape[i]] = i  # type: ignore
            self.reverse_shape = [...] + reverse_shape
        elif shape[-1] == (...):
            reverse_shape = [0] * (len(shape) - 1)
            for i in range(len(shape) - 1):
                reverse_shape[shape[i]] = i  # type: ignore
            self.reverse_shape = reverse_shape + [...]
        else:
            reverse_shape = [0] * len(shape)
            for i in range(len(shape) - 1):
                reverse_shape[shape[i]] = i  # type: ignore
            self.reverse_shape = reverse_shape  # type: ignore (checker bug)

    def backward_a(self, graph: OpGraph, grad: Op):
        return graph.view_(grad, self.reverse_shape)
