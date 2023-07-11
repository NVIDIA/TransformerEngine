from typing import Any
from .ops import Op, DType


class ComputePipeline:
    def __init__(self, ops: list[Op]):
        self._ops = ops
        self.compile()

    def __call__(self, *args: Any, **kwargs: Any):
        ...

    def compile(self):
        self.transform_op_list()
        self.allocate_parameters()

    def transform_op_list(self):
        # region infer_types
        if len(self._ops) >= 2:
            if self._ops[0].output_type is DType.infer:
                assert self._ops[1].input_type is not DType.infer
                self._ops[0].output_type = self._ops[1].input_type
            if self._ops[-1].input_type is DType.infer:
                assert self._ops[-2].output_type is not DType.infer
                self._ops[-1].input_type = self._ops[-2].output_type
        if self._ops[-1].output_type is DType.infer:
            self._ops[-1].output_type = DType.default

        for i, op in enumerate(self._ops[1:-1]):
            prev = self._ops[i - 1]
            next = self._ops[i + 1]

            if op.input_type is DType.infer:
                op.input_type = prev.output_type
            if op.output_type is DType.infer:
                assert next.input_type is not DType.infer
                op.output_type = next.input_type
        # endregion
        # TODO region auto_parallel
        # region insert_casts
        assert not any(isinstance(m, Cast) for m in self._ops)
        for i, op in enumerate(self._ops[:-1]):
            next = self._ops[i + 1]
            if op.output_type is not next.input_type:
                name = f"Cast({op.name}, {next.name})"
                self._ops.insert(i + 1, Cast(name, op.output_type, next.input_type))
        # endregion

    def allocate_parameters(self):
        for op in self._ops:
            params = op.describe_params()
            for name, shape in params.items():
                self.allocate(op.name + "." + name, shape)

    def allocate(self, name: str, shape: tuple[int, ...]):
        ...


class Cast(Op):
    pass


__all__ = ["ComputePipeline"]
