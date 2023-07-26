from __future__ import annotations
from abc import ABC, abstractmethod
from copy import deepcopy
from enum import Enum
import enum
from typing import Generator, final

from transformer_engine.pytorch.sequential_new.common_back.enums import DType, PType
from transformer_engine.pytorch.sequential_new.common_back.generic_tensor import (
    GenericTensor,
    TensorDescriptor,
    ParamInitializer,
)
from .enums import DType, DTypeInfer, PType
from .generic_tensor import GenericTensor
from . import generic_tensor as f

ExecutionFlow = list["Op"]
Parallelism = tuple[PType, PType]


# Op Protocol
class _OpState(Enum):
    POST_INIT = enum.auto()
    POST_SET_TYPES_INFERRED = enum.auto()
    POST_SET_PARALLELISM = enum.auto()
    POST_DESCRIBE_PARALLELISM = enum.auto()
    POST_SET_INPUT_SHAPE = enum.auto()
    POST_DESCRIBE_PARAMS = enum.auto()
    POST_DESCRIBE_ACTIVATION_SHAPE = enum.auto()
    POST_DESCRIBE_SUPPLEMENTARY_TENSORS = enum.auto()
    POST_SET_TENSORS_ALLOCATED = enum.auto()

    def __ge__(self, other: _OpState):
        return self.value >= other.value

    def __gt__(self, other: _OpState):
        return self.value > other.value

    def __le__(self, other: _OpState):
        return self.value <= other.value

    def __lt__(self, other: _OpState):
        return self.value < other.value


class Op(ABC):
    """
    An Op represents a transformation applied to the main data tensor.
    It is akin to an nn.Module + autograd.Function pair in PyTorch.

    Inside ComputePipeline, the following methods are called in the order
    in which they are declared below.
    """

    name: str
    __input_type: DType | DTypeInfer
    __output_type: DType | DTypeInfer
    __parallellism: Parallelism | None
    __input_shape: tuple[int, ...] | None
    __state: _OpState

    @property
    def raw_input_type(self):
        assert self.__state < _OpState.POST_SET_TYPES_INFERRED
        return self.__input_type

    @raw_input_type.setter
    def raw_input_type(self, value: DType):
        assert self.__state < _OpState.POST_SET_TYPES_INFERRED
        self.__input_type = value

    @property
    def raw_output_type(self):
        assert self.__state < _OpState.POST_SET_TYPES_INFERRED
        return self.__output_type

    @raw_output_type.setter
    def raw_output_type(self, value: DType):
        assert self.__state < _OpState.POST_SET_TYPES_INFERRED
        self.__output_type = value

    @property
    def input_type(self):
        assert self.__state >= _OpState.POST_SET_TYPES_INFERRED
        assert isinstance(self.__input_type, DType)
        return self.__input_type

    @property
    def output_type(self):
        assert self.__state >= _OpState.POST_SET_TYPES_INFERRED
        assert isinstance(self.__output_type, DType)
        return self.__output_type

    @property
    def parallellism(self):
        assert self.__state >= _OpState.POST_SET_PARALLELISM
        assert self.__parallellism is not None
        return self.__parallellism

    @property
    def input_shape(self):
        assert self.__state >= _OpState.POST_SET_INPUT_SHAPE
        assert self.__input_shape is not None
        return self.__input_shape

    def __init__(
        self,
        relative_name: str,
        input_type: DType | DTypeInfer,
        output_type: DType | DTypeInfer,
    ):
        self.name = relative_name
        self.__input_type = input_type
        self.__output_type = output_type
        self.__parallellism = None
        self.__input_shape = None
        self.__state = _OpState.POST_INIT

    def named(self, parent_module_name: str):
        self.name = parent_module_name + "." + self.name
        return self

    def set_types_inferred(
        self, inferred_input_type: DType, inferred_output_type: DType
    ):
        assert self.__state == _OpState.POST_INIT
        self.__input_type = inferred_input_type
        self.__output_type = inferred_output_type
        self.__state = _OpState.POST_SET_TYPES_INFERRED

    def set_parallelism(self, chosen_parallelism: Parallelism):
        assert self.__state == _OpState.POST_SET_TYPES_INFERRED
        self.__parallellism = chosen_parallelism
        self.__state = _OpState.POST_SET_PARALLELISM

    def _pre_describe_parallellism_hook(self):
        assert self.__state == _OpState.POST_SET_TYPES_INFERRED
        self.__state = _OpState.POST_DESCRIBE_PARALLELISM

    @abstractmethod
    def describe_parallellism(self) -> list[ExecutionFlow]:
        raise NotImplementedError()

    def set_input_shape(self, input_shape: tuple[int, ...]):
        assert self.__state == _OpState.POST_SET_PARALLELISM
        self.__input_shape = input_shape
        self.__state = _OpState.POST_SET_INPUT_SHAPE

    def _pre_describe_params_hook(self):
        assert self.__state == _OpState.POST_SET_INPUT_SHAPE
        self.__state = _OpState.POST_DESCRIBE_PARAMS

    @abstractmethod
    def describe_params(self) -> dict[str, TensorDescriptor]:
        raise NotImplementedError()

    def _pre_describe_activation_shape_hook(self):
        assert self.__state == _OpState.POST_DESCRIBE_PARAMS
        self.__state = _OpState.POST_DESCRIBE_ACTIVATION_SHAPE

    @abstractmethod
    def describe_activation_shape(self) -> tuple[int, ...]:
        raise NotImplementedError()

    def _pre_describe_supplementary_tensors_hook(self):
        assert self.__state == _OpState.POST_DESCRIBE_ACTIVATION_SHAPE
        self.__state = _OpState.POST_DESCRIBE_SUPPLEMENTARY_TENSORS

    @abstractmethod
    def describe_supplementary_tensors_training(self) -> dict[str, TensorDescriptor]:
        raise NotImplementedError()

    @abstractmethod
    def describe_supplementary_tensors_inference(self) -> dict[str, TensorDescriptor]:
        raise NotImplementedError()

    def set_tensors_allocated(self, **tensors: GenericTensor):
        assert self.__state == _OpState.POST_DESCRIBE_SUPPLEMENTARY_TENSORS
        for name, tensor in tensors.items():
            setattr(self, name, tensor)
        self.__state = _OpState.POST_SET_TENSORS_ALLOCATED

    @abstractmethod
    def training(
        self,
        x: GenericTensor,
    ) -> Generator[GenericTensor, GenericTensor, GenericTensor]:
        raise NotImplementedError()

    @abstractmethod
    def inference(
        self,
        x: GenericTensor,
    ) -> GenericTensor:
        raise NotImplementedError()


# Base classes
class ParameterFreeOp(Op):
    """
    Base class for ops that do not have any trainable parameters.
    """

    def describe_params(self) -> dict[str, TensorDescriptor]:
        self._pre_describe_params_hook()
        return {}


class ShapePreserveOp(Op):
    """
    Base class for ops, whose activation's shape is the same as the input's.
    """

    def describe_activation_shape(self):
        self._pre_describe_activation_shape_hook()
        return self.input_shape


class NoSupplementaryTensorsOp(Op):
    """
    Base class for ops that do not need any additional tensors.
    """

    def describe_supplementary_tensors_training(self) -> dict[str, TensorDescriptor]:
        self._pre_describe_supplementary_tensors_hook()
        return {}

    def describe_supplementary_tensors_inference(self) -> dict[str, TensorDescriptor]:
        self._pre_describe_supplementary_tensors_hook()
        return self.describe_supplementary_tensors_training()


# Parallelism base classes

NORMAL = (PType.NA, PType.NA)
ROW_PARALLEL = (PType.NRS, PType.NRS)
COLUMN_PARALLEL = (PType.NCS, PType.NCS)


def _normal(op: Op):
    op = deepcopy(op)
    op.set_parallelism(NORMAL)
    return [op]


def _row_parallel(op: Op):
    op = deepcopy(op)
    op.set_parallelism(ROW_PARALLEL)
    return [op]


def _column_parallel(op: Op):
    op = deepcopy(op)
    op.set_parallelism(COLUMN_PARALLEL)
    return [op]


class PointwiseOp(Op):
    def describe_parallellism(self):
        self._pre_describe_parallellism_hook()
        return [_normal(self), _row_parallel(self), _column_parallel(self)]


class RowwiseOp(Op):
    def describe_parallellism(self):
        self._pre_describe_parallellism_hook()
        return [_normal(self), _row_parallel(self)]


class ColumnwiseOp(Op):
    def describe_parallellism(self):
        self._pre_describe_parallellism_hook()
        return [_normal(self), _column_parallel(self)]


class NonParallelOp(Op):
    def describe_parallellism(self):
        self._pre_describe_parallellism_hook()
        return [_normal(self)]


# Identity
@final
class Identity(PointwiseOp, ParameterFreeOp, ShapePreserveOp, NoSupplementaryTensorsOp):
    def training(
        self,
        x: GenericTensor,
    ) -> Generator[GenericTensor, GenericTensor, GenericTensor]:
        grad = yield x
        return grad

    def inference(
        self,
        x: GenericTensor,
    ) -> GenericTensor:
        return x


# Transpose
@final
class Transpose(NonParallelOp, ParameterFreeOp):
    act: GenericTensor

    def _act_shape(self):
        return self.input_shape[:-2] + (self.input_shape[-1], self.input_shape[-2])

    def describe_activation_shape(self) -> tuple[int, ...]:
        self._pre_describe_activation_shape_hook()
        return self._act_shape()

    def describe_supplementary_tensors_training(self) -> dict[str, TensorDescriptor]:
        self._pre_describe_supplementary_tensors_hook()
        if self.input_shape != self._act_shape():
            return {"act": TensorDescriptor(self._act_shape(), None, self.output_type)}
        else:
            return {}

    def describe_supplementary_tensors_inference(self) -> dict[str, TensorDescriptor]:
        return self.describe_supplementary_tensors_training()

    def training(
        self, x: GenericTensor
    ) -> Generator[GenericTensor, GenericTensor, GenericTensor]:
        if self.input_shape != self._act_shape():
            f.transpose(x, out=self.act)
            grad = yield self.act
            f.transpose(grad, out=x)
            return x
        else:
            f.transpose(x, out=x)
            grad = yield x
            f.transpose(grad, out=grad)
            return grad

    def inference(self, x: GenericTensor) -> GenericTensor:
        if self.input_shape != self._act_shape():
            f.transpose(x, out=self.act)
            return self.act
        else:
            f.transpose(x, out=x)
            return x


# Normalization
@final
class LayerNorm(RowwiseOp, ShapePreserveOp):
    weight: GenericTensor
    weight_grad: GenericTensor
    bias: GenericTensor
    bias_grad: GenericTensor
    act: GenericTensor

    def __init__(
        self,
        name: str,
        input_type: DType | DTypeInfer,
        output_type: DType | DTypeInfer,
        features: int,
        eps: float,
        zero_centered_gamma: bool,
    ):
        super().__init__(name, input_type, output_type)
        self.features = features
        self.eps = eps
        self.zero_centered_gamma = zero_centered_gamma

    def describe_params(
        self,
    ) -> dict[str, TensorDescriptor]:  # TODO: take self.parallelism into account
        self._pre_describe_params_hook()
        return {
            "weight": TensorDescriptor(
                (self.features,),
                f.zeros if self.zero_centered_gamma else f.ones,
                self.output_type,
            ),
            "bias": TensorDescriptor((self.features,), f.zeros, self.output_type),
        }

    def describe_supplementary_tensors_training(
        self,
    ) -> dict[str, TensorDescriptor]:  # TODO: take self.parallelism into account
        self._pre_describe_supplementary_tensors_hook()
        return {"act": TensorDescriptor(self.input_shape, None, self.output_type)}

    def describe_supplementary_tensors_inference(
        self,
    ) -> dict[str, TensorDescriptor]:  # TODO: take self.parallelism into account
        self._pre_describe_supplementary_tensors_hook()
        if self.output_type != self.input_type:
            return {"act": TensorDescriptor(self.input_shape, None, self.output_type)}
        else:
            return {}

    def training(
        self,
        x: GenericTensor,
    ) -> Generator[GenericTensor, GenericTensor, GenericTensor]:
        f.layer_norm(x, self.weight, self.bias, self.eps, out=self.act)
        grad = yield self.act
        f.dlayer_norm(
            grad,
            x,
            self.weight,
            self.eps,
            out_dgrad=grad,
            out_wgrad=self.weight_grad,
            out_bgrad=self.bias_grad,
        )
        return grad

    def inference(
        self,
        x: GenericTensor,
    ) -> GenericTensor:
        if self.output_type != self.input_type:
            f.layer_norm(x, self.weight, self.bias, self.eps, out=self.act)
            return self.act
        else:
            f.layer_norm(x, self.weight, self.bias, self.eps, out=x)
            return x


# Linear
CGEMM = (PType.NA, PType.NCS)
RGEMM = (PType.NA, PType.PA)
RGEMM_SPLIT = (PType.NCS, PType.PA)
RGEMM_RS = (PType.NCS, PType.NRS)
AG_CGEMM = (PType.NRS, PType.NCS)


def _cgemm(op: Gemm) -> list[Op]:
    op = deepcopy(op)
    op.set_parallelism(CGEMM)
    return [op]


def _rgemm(op: Gemm) -> list[Op]:
    op = deepcopy(op)
    op.set_parallelism(RGEMM)
    return [op]


def _rgemm_split(op: Gemm) -> list[Op]:
    op = deepcopy(op)
    op.set_parallelism(RGEMM_SPLIT)
    return [op]


def _rgemm_rs(op: Gemm) -> list[Op]:
    op = deepcopy(op)
    op.set_parallelism(RGEMM_RS)
    rs = ReduceScatter()
    return [op, rs]


def _ag_cgemm(op: Gemm) -> list[Op]:
    op = deepcopy(op)
    op.set_parallelism(AG_CGEMM)
    ag = AllGather()
    return [ag, op]


@final
class Gemm(Op):  # TODO: take self.parallelism into account
    weight: GenericTensor
    weight_grad: GenericTensor
    weight_t: GenericTensor
    x_t: GenericTensor
    act: GenericTensor

    def __init__(
        self,
        name: str,
        input_type: DType | DTypeInfer,
        output_type: DType | DTypeInfer,
        param_type: DType,
        in_features: int,
        out_features: int,
        init_method: ParamInitializer,
    ):
        super().__init__(name, input_type, output_type)
        self.in_features = in_features
        self.out_features = out_features
        self.param_dtype = param_type
        self.init_method = init_method

    def describe_parallellism(self):
        self._pre_describe_parallellism_hook()
        return [
            _normal(self),
            _cgemm(self),
            _rgemm(self),
            _rgemm_split(self),
            _rgemm_rs(self),
            _ag_cgemm(self),
        ]

    def describe_params(self) -> dict[str, TensorDescriptor]:
        self._pre_describe_params_hook()
        return {
            "weight": TensorDescriptor(
                (self.in_features, self.out_features),
                self.init_method,
                self.param_dtype,
            ),
        }

    def _act_shape(self):
        return self.input_shape[:-1] + (self.out_features,)

    def describe_activation_shape(self) -> tuple[int, ...]:
        self._pre_describe_activation_shape_hook()
        return self._act_shape()

    def describe_supplementary_tensors_training(self) -> dict[str, TensorDescriptor]:
        self._pre_describe_supplementary_tensors_hook()
        return {
            "act": TensorDescriptor(self._act_shape(), None, self.output_type),
            "weight_t": TensorDescriptor(
                (self.out_features, self.in_features), None, self.param_dtype
            ),
            "x_t": TensorDescriptor(
                (self.input_shape[-1], self.input_shape[-2]), None, self.input_type
            ),
        }

    def describe_supplementary_tensors_inference(self) -> dict[str, TensorDescriptor]:
        self._pre_describe_supplementary_tensors_hook()
        if self.output_type != self.input_type or self.input_shape != self._act_shape():
            return {"act": TensorDescriptor(self._act_shape(), None, self.output_type)}
        else:
            return {}

    def training(
        self, x: GenericTensor
    ) -> Generator[GenericTensor, GenericTensor, GenericTensor]:
        f.transpose(self.weight, out=self.weight_t)
        f.gemm(x, self.weight, out=self.act)
        grad = yield self.act
        f.gemm(grad, self.weight_t, out=x)
        f.gemm(grad, self.x_t, out=self.weight_grad)
        return x

    def inference(self, x: GenericTensor) -> GenericTensor:
        if self.output_type != self.input_type or self.input_shape != self._act_shape():
            f.gemm(x, self.weight, out=self.act)
            return self.act
        else:
            f.gemm(x, self.weight, out=x)
            return x


@final
class Bias(PointwiseOp, ShapePreserveOp, NoSupplementaryTensorsOp):
    bias: GenericTensor
    bias_grad: GenericTensor

    def __init__(
        self,
        name: str,
        input_type: DType | DTypeInfer,
        output_type: DType | DTypeInfer,
        param_type: DType,
        features: int,
        init_method: ParamInitializer,
    ):
        super().__init__(name, input_type, output_type)
        self.param_dtype = param_type
        self.features = features
        self.init_method = init_method

    def describe_params(self) -> dict[str, TensorDescriptor]:
        self._pre_describe_params_hook()
        return {
            "bias": TensorDescriptor(
                (self.features,), self.init_method, self.param_dtype
            ),
        }

    def training(
        self, x: GenericTensor
    ) -> Generator[GenericTensor, GenericTensor, GenericTensor]:
        f.add(x, self.bias, out=x)
        grad = yield x
        f.copy(grad, out=self.bias_grad)
        return grad

    def inference(self, x: GenericTensor) -> GenericTensor:
        f.add(x, self.bias, out=x)
        return x


# Attention
HEAD_PARALLEL_SCATTERING = (PType.NA, PType.NCS)
HEAD_PARALLEL_GATHERING = (PType.NCS, PType.NA)


@final
class DotProductAttention(ParameterFreeOp, ShapePreserveOp):  # TODO
    features_per_head: int

    def __init__(
        self,
        name: str,
        input_type: DType | DTypeInfer,
        output_type: DType | DTypeInfer,
        features_per_head: int,
    ):
        super().__init__(
            name,
            input_type,
            output_type,
        )
        self.features_per_head = features_per_head

    def describe_parallellism(self):
        return [
            _normal(self),
            _column_parallel(self),
            HEAD_PARALLEL_SCATTERING,
            HEAD_PARALLEL_GATHERING,
        ]


# Residual
@final
class ResidualBegin(PointwiseOp, ParameterFreeOp, ShapePreserveOp):
    end: ResidualEnd | None = None
    fwd_residue: GenericTensor

    def __init__(self, name: str, end: ResidualEnd | None = None):
        super().__init__(name, DType.default, DType.default)
        self.end = end

    def training(
        self, x: GenericTensor
    ) -> Generator[GenericTensor, GenericTensor, GenericTensor]:
        assert self.end is not None
        f.copy(x, out=self.fwd_residue)
        grad = yield x
        f.add(self.end.bwd_residue, grad, out=self.end.bwd_residue)
        return self.end.bwd_residue

    def inference(self, x: GenericTensor) -> GenericTensor:
        f.copy(x, out=self.fwd_residue)
        return x

    def describe_supplementary_tensors_training(self) -> dict[str, TensorDescriptor]:
        self._pre_describe_supplementary_tensors_hook()
        return {
            "fwd_residue": TensorDescriptor(self.input_shape, None, self.input_type)
        }

    def describe_supplementary_tensors_inference(self) -> dict[str, TensorDescriptor]:
        return self.describe_supplementary_tensors_training()


@final
class ResidualEnd(PointwiseOp, ParameterFreeOp, ShapePreserveOp):
    begin: ResidualBegin
    bwd_residue: GenericTensor

    def __init__(self, name: str, begin: ResidualBegin):
        super().__init__(name, DType.default, DType.default)
        self.begin = begin

    def training(
        self, x: GenericTensor
    ) -> Generator[GenericTensor, GenericTensor, GenericTensor]:
        f.add(self.begin.fwd_residue, x, out=self.begin.fwd_residue)
        grad = yield self.begin.fwd_residue
        f.copy(grad, out=self.bwd_residue)
        return grad

    def inference(self, x: GenericTensor) -> GenericTensor:
        f.add(self.begin.fwd_residue, x, out=self.begin.fwd_residue)
        return self.begin.fwd_residue

    def describe_supplementary_tensors_training(self) -> dict[str, TensorDescriptor]:
        self._pre_describe_supplementary_tensors_hook()
        return {
            "bwd_residue": TensorDescriptor(self.input_shape, None, self.input_type)
        }

    def describe_supplementary_tensors_inference(self) -> dict[str, TensorDescriptor]:
        self._pre_describe_supplementary_tensors_hook()
        return {}


# Dropout
@final
class Dropout(PointwiseOp, ParameterFreeOp, ShapePreserveOp, NoSupplementaryTensorsOp):
    p: float

    def __init__(self, name: str, p: float):
        super().__init__(name, DTypeInfer(), DTypeInfer())
        self.p = p

    def training(
        self, x: GenericTensor
    ) -> Generator[GenericTensor, GenericTensor, GenericTensor]:
        f.dropout(x, self.p, out=x)
        grad = yield x
        f.dropout(grad, self.p, out=grad)
        return grad

    def inference(self, x: GenericTensor) -> GenericTensor:
        f.dropout(x, self.p, out=x)
        return x


# Activation
@final
class Gelu(PointwiseOp, ParameterFreeOp, ShapePreserveOp):
    act: GenericTensor

    def training(
        self,
        x: GenericTensor,
    ) -> Generator[GenericTensor, GenericTensor, GenericTensor]:
        f.gelu(x, out=self.act)
        grad = yield self.act
        f.dgelu(grad, x, out_dgrad=grad)
        return grad

    def inference(
        self,
        x: GenericTensor,
    ) -> GenericTensor:
        if self.output_type != self.input_type:
            f.gelu(x, out=self.act)
            return self.act
        else:
            f.gelu(x, out=x)
            return x

    def describe_supplementary_tensors_training(self) -> dict[str, TensorDescriptor]:
        self._pre_describe_supplementary_tensors_hook()
        return {"act": TensorDescriptor(self.input_shape, None, self.input_type)}

    def describe_supplementary_tensors_inference(self) -> dict[str, TensorDescriptor]:
        self._pre_describe_supplementary_tensors_hook()
        if self.output_type != self.input_type:
            return {"act": TensorDescriptor(self.input_shape, None, self.output_type)}
        else:
            return {}


@final
class Relu(PointwiseOp, ParameterFreeOp, ShapePreserveOp):
    act: GenericTensor

    def training(
        self,
        x: GenericTensor,
    ) -> Generator[GenericTensor, GenericTensor, GenericTensor]:
        f.relu(x, out=self.act)
        grad = yield self.act
        f.drelu(grad, x, out_dgrad=grad)
        return grad

    def inference(
        self,
        x: GenericTensor,
    ) -> GenericTensor:
        if self.output_type != self.input_type:
            f.relu(x, out=self.act)
            return self.act
        else:
            f.relu(x, out=x)
            return x

    def describe_supplementary_tensors_training(self) -> dict[str, TensorDescriptor]:
        self._pre_describe_supplementary_tensors_hook()
        return {"act": TensorDescriptor(self.input_shape, None, self.input_type)}

    def describe_supplementary_tensors_inference(self) -> dict[str, TensorDescriptor]:
        self._pre_describe_supplementary_tensors_hook()
        if self.output_type != self.input_type:
            return {"act": TensorDescriptor(self.input_shape, None, self.output_type)}
        else:
            return {}


# Cast
@final
class Cast(PointwiseOp, ShapePreserveOp, ParameterFreeOp):
    act: GenericTensor

    def training(
        self,
        x: GenericTensor,
    ) -> Generator[GenericTensor, GenericTensor, GenericTensor]:
        f.cast(x, out=self.act)
        grad = yield self.act
        f.cast(grad, out=x)
        return x

    def inference(
        self,
        x: GenericTensor,
    ) -> GenericTensor:
        f.cast(x, out=self.act)
        return self.act

    def describe_supplementary_tensors_training(self) -> dict[str, TensorDescriptor]:
        return {"act": TensorDescriptor(self.input_shape, None, self.output_type)}

    def describe_supplementary_tensors_inference(self) -> dict[str, TensorDescriptor]:
        return self.describe_supplementary_tensors_training()


# Communication
@final
class ReduceScatter(Op):
    ...


@final
class AllGather(Op):
    ...


@final
class AllReduce(Op):
    ...


@final
class Scatter(Op):
    ...
