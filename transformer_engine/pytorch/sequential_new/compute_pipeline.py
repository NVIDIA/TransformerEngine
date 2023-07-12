from typing import Any, Callable, Generic
from .ops import Op, DType, PassthroughOp
from .framework_interface import FrameworkInterface, TensorType


class ComputePipeline(Generic[TensorType]):
    def __init__(
        self,
        framework_interface: FrameworkInterface[TensorType],
        ops: list[Op],
        extra_transforations: list[Callable[[list[Op]], list[Op]]] = [],
    ):
        self._framework_interface = framework_interface
        self._framework = type(framework_interface)
        self._ops = ops
        self._extra_transforations = extra_transforations
        self.compile()

    def __call__(self, *args: Any, **kwargs: Any):
        ...

    def compile(self):
        self.transform_op_list()
        self.allocate_parameters()

    def transform_op_list(self):
        for transform in self._extra_transforations:
            self.infer_types()
            self._ops = transform(self._ops)
        self.infer_types()
        self.insert_casts()

    def infer_types(self):
        for i, op in enumerate(self._ops):
            prev = self._ops[i - 1] if i > 0 else None
            next = self._ops[i + 1] if i < len(self._ops) - 1 else None

            if op.input_type is DType.infer:
                if prev is None:
                    pass
                elif prev.output_type is not DType.infer:
                    op.input_type = prev.output_type
                elif op.output_type is not DType.infer:
                    op.input_type = op.output_type
                elif next is not None and next.input_type is not DType.infer:
                    op.input_type = next.input_type
                else:
                    raise RuntimeError("Cannot infer input type")
            if op.output_type is DType.infer:
                if next is None:
                    pass
                elif next.input_type is not DType.infer:
                    op.output_type = next.input_type
                elif op.input_type is not DType.infer:
                    op.output_type = op.input_type
                elif prev is not None and prev.output_type is not DType.infer:
                    op.output_type = prev.output_type
                else:
                    for next in self._ops[i + 2 :]:
                        if next.input_type is not DType.infer:
                            op.output_type = next.input_type
                            break
                        elif next.output_type is not DType.infer:
                            op.output_type = next.output_type
                            break
                    if op.output_type is DType.infer:
                        raise RuntimeError("Cannot infer output type")

    def insert_casts(self):
        assert not any(isinstance(m, Cast) for m in self._ops)
        i = 0
        while i < len(self._ops) - 1:
            op = self._ops[i]
            next = self._ops[i + 1]
            if op.output_type is not next.input_type:
                name = f"Cast({op.name}, {next.name})"
                self._ops.insert(i + 1, Cast(name, op.output_type, next.input_type))
                i += 1  # skip cast
            i += 1

    def allocate_parameters(self):
        for op in self._ops:
            op_params = op.describe_params()
            for name, desc in op_params.items():
                self.allocate(op.name + "." + name, desc.shape)

    def allocate(self, name: str, shape: tuple[int, ...]):
        tensor = self._framework.fi_empty(shape)
        self._framework_interface.fi_register_buffer(name, tensor)


__all__ = ["ComputePipeline"]


class Cast(PassthroughOp):
    pass
