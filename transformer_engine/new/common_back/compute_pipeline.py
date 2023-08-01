from typing import Generator, final

from transformer_engine.pytorch.sequential_new.common_back.generic_tensor import (
    GenericTensor,
)


from .ops import (
    NonParallelOp,
    Op,
    Cast,
    ParallelismClass,
    ParameterFreeOp,
    NoSupplementaryTensorsOp,
)
from .enums import DType, DTypeInfer

from .generic_tensor import GenericTensor, TensorDescriptor
from .tensor_manager import TensorManager
from .model_parallel_transform import model_parallel_transform


@final
class ComputePipeline(NonParallelOp, ParameterFreeOp, NoSupplementaryTensorsOp):
    def __init__(self, ops: list[Op]):
        self._ops = ops
        self._tensor_manager = TensorManager()
        self._compiled = None
        self._act_shape = None
        super().__init__(
            "ComputePipeline", ops[0].raw_input_type, ops[-1].raw_output_type
        )

    def _pre_describe_tensors(self) -> None:
        assert self.parallellism == ParallelismClass.NORMAL
        self._compiled = ComputePipeline.compile(
            self._ops,
            self.environment.distributed_group is not None,
            self.input_type,
            self.output_type,
        )
        self._act_shape = self.allocate_tensors()

    def describe_activation_shape(self):
        assert self._act_shape is not None
        return self._act_shape

    def allocate_tensors(self):
        assert self._compiled is not None
        input_shape = self.input_shape
        params = list[dict[str, TensorDescriptor]]()
        for op in self._compiled:
            op.set_input_shape(input_shape)
            params.append(op.describe_params())
            for name, param in params[-1].items():
                self._tensor_manager.register_tensor(op.name + "." + name, param)

            act_shape = op.describe_activation_shape()
            input_shape = act_shape

        self._tensor_manager.allocate_storage()

        for op, p_tensors in zip(self._compiled, params):
            tensors = dict[str, GenericTensor]()
            for name, _ in p_tensors.items():
                tensors[name] = self._tensor_manager.retrieve_tensor(
                    op.name + "." + name
                )
            op.set_tensors_allocated(**tensors)

        return input_shape

    def training(
        self, x: GenericTensor
    ) -> Generator[GenericTensor, GenericTensor, None]:
        assert self._compiled is not None
        assert self.environment.training

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
        assert not self.environment.training

        for op in self._compiled:
            x = op.inference(x)
        return x

    @staticmethod
    def compile(
        _ops: list[Op], _model_parallel: bool, _input_type: DType, _output_type: DType
    ):
        if _model_parallel:
            _ops = ComputePipeline.infer_types(_ops)
            _ops = model_parallel_transform(_ops)
        else:
            _ops = ComputePipeline.infer_types(_ops)
            for op in _ops:
                op.set_parallelism(ParallelismClass.NORMAL)
        _ops = ComputePipeline.infer_types(_ops)
        _ops = ComputePipeline.insert_casts(_ops)
        _ops = ComputePipeline.set_types(_ops)
        if _ops[0].input_type is not _input_type:
            _ops = [Cast("CastInput", _input_type, _ops[0].input_type)] + _ops
        if _ops[-1].output_type is not _output_type:
            _ops = _ops + [Cast("CastOutput", _ops[-1].output_type, _output_type)]
        return _ops

    @staticmethod
    def infer_types(_ops: list[Op]):
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
    def insert_casts(_ops: list[Op]):
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
    def set_types(_ops: list[Op]):
        for op in _ops:
            assert not isinstance(op.raw_input_type, DTypeInfer)
            assert not isinstance(op.raw_output_type, DTypeInfer)
            op.set_types_inferred(op.raw_input_type, op.raw_output_type)
        return _ops


__all__ = ["ComputePipeline"]
