from typing import Generator, final

from .ops import (
    _normal,
    NonParallelOp,
    Op,
    Cast,
    NoTensorOp,
)
from .enums import DType, DTypeInfer
from .generic_tensor import GenericTensor
from .tensor_manager import TensorManager
from .model_parallel_transform import model_parallel_transform


@final
class ComputePipeline(NonParallelOp, NoTensorOp):
    _ops: list[Op]
    _original_types: list[tuple[DType | DTypeInfer, DType | DTypeInfer]]
    _tensor_manager: TensorManager
    _parameters: list[tuple[GenericTensor, str, Op]]
    _buffers: list[tuple[GenericTensor, str, Op]]
    _compiled: list[Op] | None

    def __init__(self, ops: list[Op], world_size: int = 1):
        ops = [op.set_world_size(world_size) for op in ops]
        ops = (
            [_normal(op)[0] for op in ops]
            if world_size == 1
            else model_parallel_transform(ops)
        )
        (
            self._ops,
            self._tensor_manager,
            self._parameters,
            self._buffers,
        ) = ComputePipeline._allocate_tensors(ops)
        self._original_types = [
            (op.raw_input_type, op.raw_output_type) for op in self._ops
        ]
        self._compiled = None
        super().__init__(
            "ComputePipeline", ops[0].raw_input_type, ops[-1].raw_output_type
        )

    def parameters(self):
        return self._parameters

    def buffers(self):
        return self._buffers

    def compile(self, input_type: DType, output_type: DType):
        for op, orig_types in zip(self._ops, self._original_types):
            op.raw_input_type, op.raw_output_type = orig_types
        self._compiled = ComputePipeline._compile(self._ops, input_type, output_type)

    def _pre_describe_tensors(self) -> None:
        self.compile(self.input_type, self.output_type)

    def training(
        self, x: GenericTensor
    ) -> Generator[GenericTensor, GenericTensor, None]:
        assert self._compiled is not None

        backwards = list[Generator[GenericTensor, GenericTensor, None]]()
        for op in self._compiled:
            gen = op.training(x)
            x = next(gen)
            backwards.append(gen)
        backwards.reverse()

        grad = yield x

        for bwd in backwards:
            grad = bwd.send(grad)

        yield grad

    def inference(self, x: GenericTensor):
        assert self._compiled is not None

        for op in self._compiled:
            x = op.inference(x)
        return x

    @staticmethod
    def _compile(_ops: list[Op], _input_type: DType, _output_type: DType):
        _ops = ComputePipeline._infer_types(_ops)
        _ops = ComputePipeline._insert_casts(_ops)
        _ops = ComputePipeline._set_types(_ops)
        if _ops[0].input_type is not _input_type:
            _ops = [Cast("CastInput", _input_type, _ops[0].input_type)] + _ops
        if _ops[-1].output_type is not _output_type:
            _ops = _ops + [Cast("CastOutput", _ops[-1].output_type, _output_type)]
        return _ops

    @staticmethod
    def _allocate_tensors(ops: list[Op]):
        tensor_manager = TensorManager()

        for op in ops:
            for tensor_name, tensor_desc in op.describe_tensors().items():
                qual_name = op.name + tensor_name
                tensor_manager.register_tensor(qual_name, tensor_desc)

        tensor_manager.allocate_storage()

        parameters = list[tuple[GenericTensor, str, Op]]()
        buffers = list[tuple[GenericTensor, str, Op]]()
        for op in ops:
            tensors = dict[str, GenericTensor]()
            for tensor_name, tensor_desc in op.describe_tensors().items():
                qual_name = op.name + tensor_name
                tensor = tensor_manager.retrieve_tensor(qual_name)
                tensors[qual_name] = tensor
                if tensor_desc.is_parameter:
                    parameters.append((tensor, tensor_name, op))
                else:
                    buffers.append((tensor, tensor_name, op))
            op.set_tensors_allocated(**tensors)

        return (ops, tensor_manager, parameters, buffers)

    @staticmethod
    def _infer_types(_ops: list[Op]):
        for i, op in enumerate(_ops):
            prev = _ops[i - 1] if i > 0 else None
            nxt_ = _ops[i + 1] if i < len(_ops) - 1 else None

            if isinstance(op.raw_input_type, DTypeInfer):
                if prev is None:
                    pass
                elif not isinstance(prev.raw_output_type, DTypeInfer):
                    op.raw_input_type = prev.raw_output_type
                elif not isinstance(op.raw_output_type, DTypeInfer):
                    op.raw_input_type = op.raw_output_type
                elif nxt_ is not None and not isinstance(
                    nxt_.raw_input_type, DTypeInfer
                ):
                    op.raw_input_type = nxt_.raw_input_type
                else:
                    raise RuntimeError("Cannot infer input type")
            if isinstance(op.raw_output_type, DTypeInfer):
                if nxt_ is None:
                    pass
                elif not isinstance(nxt_.raw_input_type, DTypeInfer):
                    op.raw_output_type = nxt_.raw_input_type
                elif not isinstance(op.raw_input_type, DTypeInfer):
                    op.raw_output_type = op.raw_input_type
                elif prev is not None and not isinstance(
                    prev.raw_output_type, DTypeInfer
                ):
                    op.raw_output_type = prev.raw_output_type
                else:
                    for nxt_ in _ops[i + 2 :]:
                        if not isinstance(nxt_.raw_input_type, DTypeInfer):
                            op.raw_output_type = nxt_.raw_input_type
                            break
                        elif not isinstance(nxt_.raw_output_type, DTypeInfer):
                            op.raw_output_type = nxt_.raw_output_type
                            break
                    if isinstance(op.raw_output_type, DTypeInfer):
                        raise RuntimeError("Cannot infer output type")
        return _ops

    @staticmethod
    def _insert_casts(_ops: list[Op]):
        assert not any(isinstance(m, Cast) for m in _ops)
        i = 0
        while i < len(_ops) - 1:
            op = _ops[i]
            nxt_ = _ops[i + 1]
            assert op.raw_output_type is not DTypeInfer
            assert nxt_.raw_input_type is not DTypeInfer
            if op.raw_output_type is not nxt_.raw_input_type:
                name = f"Cast({op.name}, {nxt_.name})"
                _ops.insert(i + 1, Cast(name, op.output_type, nxt_.input_type))
                i += 1  # skip cast
            i += 1
        return _ops

    @staticmethod
    def _set_types(_ops: list[Op]):
        for op in _ops:
            assert not isinstance(op.raw_input_type, DTypeInfer)
            assert not isinstance(op.raw_output_type, DTypeInfer)
            op.set_types_inferred(op.raw_input_type, op.raw_output_type)
        return _ops


__all__ = ["ComputePipeline"]
