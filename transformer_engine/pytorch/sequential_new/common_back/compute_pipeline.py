from typing import Any, Callable, final

from torch import Generator

from .tensor_operations import OpMan, TensorHandle
from .ops import (
    OpBase,
    PassthroughOp,
    PointwiseOp,
    ShapePreserveOp,
    returning,
)
from .enums import DType, PType
from .framework_interface import FrameworkInterface, TensorType, TensorDescriptor
from .tensor_manager import TensorManager


class ComputePipeline:
    def __init__(
        self,
        framework_interface: FrameworkInterface[TensorType],
        ops: list[OpBase],
        input_shape: tuple[int, ...],
        training: bool,
        extra_transformations: list[Callable[[list[OpBase]], list[OpBase]]] = [],
    ):
        self._framework_interface = framework_interface
        self._framework = type(framework_interface)
        self._training = training
        self._input_shape = input_shape
        self._fwd = ComputePipeline.compile(ops, extra_transformations)
        self._tensor_manager = TensorManager(self._framework_interface)
        self.allocate_tensors()

    def allocate_tensors(self):
        input_shape = self._input_shape
        for op in self._fwd:
            params = op.describe_params()
            for name, param in params.items():
                self._tensor_manager.register_tensor(op.name + "." + name, param)

            op.input_shape = input_shape
            act_shape = op.describe_activation_shape()
            input_shape = act_shape

            call = (
                op.describe_supplementary_tensors_training
                if self._training
                else op.describe_supplementary_tensors_inference
            )

            for (
                name,
                tensor,
            ) in call().items():
                self._tensor_manager.register_tensor(
                    op.name + "." + name,
                    TensorDescriptor(tensor.shape, None, tensor.dtype),
                )

        self._tensor_manager.allocate_storage()

        for op in self._fwd:
            params = op.describe_params()
            for name, param in params.items():
                setattr(
                    op, name, self._tensor_manager.retrieve_tensor(op.name + "." + name)
                )

            call = (
                op.describe_supplementary_tensors_training
                if self._training
                else op.describe_supplementary_tensors_inference
            )

            for (
                name,
                tensor,
            ) in call().items():
                setattr(
                    op,
                    name,
                    self._tensor_manager.retrieve_tensor(op.name + "." + name),
                )

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError()

    @staticmethod
    def compile(
        _ops: list[OpBase], _transforms: list[Callable[[list[OpBase]], list[OpBase]]]
    ):
        return ComputePipeline.transform_op_list(_ops, _transforms)

    @staticmethod
    def transform_op_list(
        _ops: list[OpBase], _transforms: list[Callable[[list[OpBase]], list[OpBase]]]
    ):
        for transform in _transforms:
            _ops = ComputePipeline.infer_types(_ops)
            _ops = transform(_ops)
        _ops = ComputePipeline.infer_types(_ops)
        _ops = ComputePipeline.insert_casts(_ops)
        return _ops

    @staticmethod
    def infer_types(_ops: list[OpBase]):
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
    def insert_casts(_ops: list[OpBase]):
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


@final
class Cast(PointwiseOp, ShapePreserveOp, PassthroughOp):
    describe_supplementary_tensors_inference = returning(dict[str, TensorDescriptor]())
    describe_supplementary_tensors_training = returning(dict[str, TensorDescriptor]())

    def training(
        self,
        typing: tuple[DType, DType],
        parallel: tuple[PType, PType],
        f: OpMan,
        x: TensorHandle,
        x_copy: TensorHandle,
    ) -> Generator:
        raise RuntimeError("Cast is not supposed to be invoked directly")

    def inference(
        self,
        typing: tuple[DType, DType],
        parallel: tuple[PType, PType],
        f: OpMan,
        x: TensorHandle,
    ) -> TensorHandle:
        raise RuntimeError("Cast is not supposed to be invoked directly")
