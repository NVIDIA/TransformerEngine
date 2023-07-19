from typing import Any, Callable
from .ops import Op, PassthroughOp
from .enums import DType
from .framework_interface import FrameworkInterface, TensorType


class ComputePipeline:
    def __init__(
        self,
        framework_interface: FrameworkInterface[TensorType],
        ops: list[Op],
        extra_transformations: list[Callable[[list[Op]], list[Op]]] = [],
    ):
        self._framework_interface = framework_interface
        self._framework = type(framework_interface)
        self._fwd = ComputePipeline.compile(ops, extra_transformations)
        self._bwd = ComputePipeline.compile(
            [op.bwd() for op in ops[::-1]], extra_transformations
        )
        # TODO: construct parameters and activations

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError()

    @staticmethod
    def compile(_ops: list[Op], _transforms: list[Callable[[list[Op]], list[Op]]]):
        return ComputePipeline.transform_op_list(_ops, _transforms)

    @staticmethod
    def transform_op_list(
        _ops: list[Op], _transforms: list[Callable[[list[Op]], list[Op]]]
    ):
        for transform in _transforms:
            _ops = ComputePipeline.infer_types(_ops)
            _ops = transform(_ops)
        _ops = ComputePipeline.infer_types(_ops)
        _ops = ComputePipeline.insert_casts(_ops)
        return _ops

    @staticmethod
    def infer_types(_ops: list[Op]):
        for i, op in enumerate(_ops):
            prev = _ops[i - 1] if i > 0 else None
            next = _ops[i + 1] if i < len(_ops) - 1 else None

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
                    for next in _ops[i + 2 :]:
                        if next.input_type is not DType.infer:
                            op.output_type = next.input_type
                            break
                        elif next.output_type is not DType.infer:
                            op.output_type = next.output_type
                            break
                    if op.output_type is DType.infer:
                        raise RuntimeError("Cannot infer output type")
        return _ops

    @staticmethod
    def insert_casts(_ops: list[Op]):
        assert not any(isinstance(m, Cast) for m in _ops)
        i = 0
        while i < len(_ops) - 1:
            op = _ops[i]
            next = _ops[i + 1]
            if op.output_type is not next.input_type:
                name = f"Cast({op.name}, {next.name})"
                _ops.insert(i + 1, Cast(name, op.output_type, next.input_type))
                i += 1  # skip cast
            i += 1
        return _ops


__all__ = ["ComputePipeline"]


class Cast(PassthroughOp):
    def bwd(self):
        raise ValueError("This operation is meant for internal use only")
