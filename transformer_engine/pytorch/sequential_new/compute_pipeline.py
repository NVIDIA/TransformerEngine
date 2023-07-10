from typing import Any
from .ops import Op, DType


class ComputePipeline:
    def __init__(self, ops: list[Op]):
        self._ops = ops
        self.compile()

    def __call__(self, *args: Any, **kwargs: Any):
        ...

    def compile(self):
        self.infer_types()
        ...

    def infer_types(self):
        if len(self._ops) >= 2:
            if self._ops[0].output_type is DType.infer:
                assert self._ops[1].input_type is not DType.infer
                self._ops[0].output_type = self._ops[1].input_type
            if self._ops[-1].input_type is DType.infer:
                assert self._ops[-2].output_type is not DType.infer
                self._ops[-1].input_type = self._ops[-2].output_type

        for i, op in enumerate(self._ops[1:-1]):
            prev = self._ops[i - 1]
            next = self._ops[i + 1]

            if op.input_type is DType.infer:
                op.input_type = prev.output_type
            if op.output_type is DType.infer:
                assert next.input_type is not DType.infer
                op.output_type = next.input_type


__all__ = ["ComputePipeline"]
