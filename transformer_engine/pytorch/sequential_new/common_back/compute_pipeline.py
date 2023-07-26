from typing import Any
from .ops import Op, Cast
from .enums import DTypeInfer
from .generic_tensor import GenericTensor, TensorDescriptor
from .tensor_manager import TensorManager
from .model_parallel_transform import model_parallel_transform


class ComputePipeline:
    def __init__(
        self,
        ops: list[Op],
        input_shape: tuple[int, ...],
        training: bool,
        model_parallel: bool,
    ):
        self._training = training
        self._input_shape = input_shape
        self._model_parallel = model_parallel
        self._fwd = ComputePipeline.compile(ops, model_parallel)
        self._tensor_manager = TensorManager()
        self.allocate_tensors()

    def allocate_tensors(self):
        input_shape = self._input_shape
        params = list[dict[str, TensorDescriptor]]()
        supplementary_tensors = list[dict[str, TensorDescriptor]]()
        for op in self._fwd:
            op.set_input_shape(input_shape)
            params.append(op.describe_params())
            for name, param in params[-1].items():
                self._tensor_manager.register_tensor(op.name + "." + name, param)

            act_shape = op.describe_activation_shape()
            input_shape = act_shape

            supplementary_tensors.append(
                op.describe_supplementary_tensors_training()
                if self._training
                else op.describe_supplementary_tensors_inference()
            )

            for (
                name,
                tensor,
            ) in supplementary_tensors[-1].items():
                self._tensor_manager.register_tensor(
                    op.name + "." + name,
                    TensorDescriptor(tensor.shape, None, tensor.dtype),
                )

        self._tensor_manager.allocate_storage()

        for op, p_tensors, s_tensors in zip(self._fwd, params, supplementary_tensors):
            tensors = dict[str, GenericTensor]()
            for name, _ in (p_tensors | s_tensors).items():
                tensors[name] = self._tensor_manager.retrieve_tensor(
                    op.name + "." + name
                )
            op.set_tensors_allocated(**tensors)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError()

    @staticmethod
    def compile(_ops: list[Op], _model_parallel: bool):
        if _model_parallel:
            _ops = ComputePipeline.infer_types(_ops)
            _ops = model_parallel_transform(_ops)
        _ops = ComputePipeline.infer_types(_ops)
        _ops = ComputePipeline.insert_casts(_ops)
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
            for op in _ops:
                assert not isinstance(op.raw_input_type, DTypeInfer)
                assert not isinstance(op.raw_output_type, DTypeInfer)
                op.set_types_inferred(op.raw_input_type, op.raw_output_type)
        return _ops

    @staticmethod
    def insert_casts(_ops: list[Op]):
        assert not any(isinstance(m, Cast) for m in _ops)
        i = 0
        while i < len(_ops) - 1:
            op = _ops[i]
            nxt_ = _ops[i + 1]
            if op.output_type is not nxt_.input_type:
                name = f"Cast({op.name}, {nxt_.name})"
                _ops.insert(i + 1, Cast(name, op.output_type, nxt_.input_type))
                i += 1  # skip cast
            i += 1
        return _ops


__all__ = ["ComputePipeline"]
