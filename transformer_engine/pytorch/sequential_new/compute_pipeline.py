from typing import Any
from .ops import Op


class ComputePipeline:
    def __init__(self, ops: list[Op]):
        self._ops = ops
        self.compile()

    def __call__(self, *args: Any, **kwargs: Any):
        ...

    def compile(self):
        ...


__all__ = ["ComputePipeline"]
