from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Generator, Never, NewType, TypeVar, TypedDict, final

from transformer_engine.pytorch.sequential_new.common_back.enums import DType, PType
from . import framework_interface as fi
from .framework_interface import Activation, Gradient, ParamConstructor, TensorType
from .enums import DType, PType
from .tensor_operations import TensorHandle, OpMan

T = TypeVar("T")


def returning(x: T) -> Callable[..., T]:
    def func(*args: Any, **kwargs: Any):
        del args, kwargs
        return x

    return func


# Op Protocol
@dataclass
class ParamDescriptor:
    _shape: tuple[int, ...]
    _constructor: ParamConstructor
    _dtype: DType


@dataclass
class TensorDescriptor:
    _shape: tuple[int, ...]
    _dtype: DType


class AnyKwargs(TypedDict, total=False):
    pass


class Op(ABC):
    @abstractmethod
    def describe_parallellism(self) -> list[tuple[PType, PType]]:
        raise NotImplementedError()

    @abstractmethod
    def describe_params(
        self, typing: tuple[DType, DType], parallel: tuple[PType, PType]
    ) -> dict[str, ParamDescriptor]:
        raise NotImplementedError()

    @abstractmethod
    def describe_activation_shape(
        self,
        typing: tuple[DType, DType],
        parallel: tuple[PType, PType],
        input_shape: tuple[int, ...],
    ) -> tuple[int, ...]:
        raise NotImplementedError()

    @abstractmethod
    def describe_supplementary_tensors_training(
        self,
        typing: tuple[DType, DType],
        parallel: tuple[PType, PType],
        input_shape: tuple[int, ...],
    ) -> dict[str, TensorDescriptor]:
        raise NotImplementedError()

    @abstractmethod
    def describe_supplementary_tensors_inference(
        self,
        typing: tuple[DType, DType],
        parallel: tuple[PType, PType],
        input_shape: tuple[int, ...],
    ) -> dict[str, TensorDescriptor]:
        raise NotImplementedError()

    @abstractmethod
    def training(
        self,
        typing: tuple[DType, DType],
        parallel: tuple[PType, PType],
        f: OpMan,
        x: TensorHandle,
    ) -> Generator[TensorHandle, TensorHandle, TensorHandle]:
        raise NotImplementedError()

    @abstractmethod
    def inference(
        self,
        typing: tuple[DType, DType],
        parallel: tuple[PType, PType],
        f: OpMan,
        x: TensorHandle,
    ) -> TensorHandle:
        raise NotImplementedError()


# Base classes
class OpBase(Op):
    name: str
    input_type: DType
    output_type: DType

    def __init__(self, relative_name: str, input_type: DType, output_type: DType):
        self.name = relative_name
        self.input_type = input_type
        self.output_type = output_type

    def named(self, parent_module_name: str):
        self.name = parent_module_name + "." + self.name
        return self


class PassthroughOp(OpBase):
    describe_params = returning(dict[str, ParamDescriptor]())


NORMAL = (PType.NA, PType.NA)
ROW_PARALLEL = (PType.NRS, PType.NRS)
COLUMN_PARALLEL = (PType.NCS, PType.NCS)


class PointwiseOp(OpBase):
    describe_parallellism = returning([NORMAL, ROW_PARALLEL, COLUMN_PARALLEL])


class RowwiseOp(OpBase):
    describe_parallellism = returning([NORMAL, ROW_PARALLEL])


class ShapePreserveOp(OpBase):
    def describe_activation_shape(
        self,
        typing: tuple[DType, DType],
        parallel: tuple[PType, PType],
        input_shape: tuple[int, ...],
    ) -> tuple[int, ...]:
        return input_shape


# Normalization
@final
class LayerNorm(RowwiseOp, ShapePreserveOp, OpBase):
    features: int
    eps: float
    zero_centered_gamma: bool

    def __init__(
        self,
        name: str,
        input_type: DType,
        output_type: DType,
        features: int,
        eps: float,
        zero_centered_gamma: bool,
    ):
        super().__init__(
            name,
            input_type,
            output_type,
            weight=ParamDescriptor(
                (features,), fi.zeros if zero_centered_gamma else fi.ones
            ),
            bias=ParamDescriptor((features,), fi.zeros),
        )
        self.features = features
        self.eps = eps
        self.zero_centered_gamma = zero_centered_gamma


# Linear
CGEMM = (PType.NA, PType.NCS)
RGEMM = (PType.NA, PType.PA)
RGEMM_SPLIT = (PType.NCS, PType.PA)
RGEMM_RS = (PType.NCS, PType.NRS)
AG_CGEMM = (PType.NRS, PType.NCS)


@final
class Gemm(OpBase):
    in_features: int
    out_features: int
    init_method: ParamConstructor

    def __init__(
        self,
        name: str,
        input_type: DType,
        output_type: DType,
        in_features: int,
        out_features: int,
        init_method: ParamConstructor,
    ):
        super().__init__(
            name,
            input_type,
            output_type,
            weight=ParamDescriptor((out_features, in_features), init_method),
        )
        self.in_features = in_features
        self.out_features = out_features
        self.init_method = init_method

    def describe_parallellism(self) -> list[tuple[PType, PType]]:
        return [
            NORMAL,
            CGEMM,
            RGEMM,
            RGEMM_SPLIT,
            RGEMM_RS,
            AG_CGEMM,
        ]


@final
class Bias(PointwiseOp, ShapePreserveOp, OpBase):
    features: int
    init_method: ParamConstructor

    def __init__(
        self,
        name: str,
        input_type: DType,
        output_type: DType,
        features: int,
        init_method: ParamConstructor,
    ):
        super().__init__(
            name,
            input_type,
            output_type,
            bias=ParamDescriptor((features,), init_method),
        )
        self.features = features
        self.init_method = init_method


# Transpose
@final
class Transpose(PassthroughOp):
    def describe_parallellism(self) -> list[tuple[PType, PType]]:
        return [NORMAL]


# Attention
HEAD_PARALLEL_SCATTERING = (PType.NA, PType.NCS)
HEAD_PARALLEL_GATHERING = (PType.NCS, PType.NA)


@final
class DotProductAttention(PassthroughOp, ShapePreserveOp):
    features_per_head: int

    def __init__(
        self,
        name: str,
        input_type: DType,
        output_type: DType,
        features_per_head: int,
    ):
        super().__init__(
            name,
            input_type,
            output_type,
        )
        self.features_per_head = features_per_head

    def describe_parallellism(self) -> list[tuple[PType, PType]]:
        return [
            NORMAL,
            COLUMN_PARALLEL,
            HEAD_PARALLEL_SCATTERING,
            HEAD_PARALLEL_GATHERING,
        ]


# Residual
@final
class ResidualBegin(PointwiseOp, PassthroughOp, ShapePreserveOp):
    end: ResidualEnd | None = None

    def __init__(self, name: str, end: ResidualEnd | None = None):
        super().__init__(name, DType.default, DType.default)
        self.end = end

    def describe_supplementary_tensors(
        self,
        typing: tuple[DType, DType],
        parallel: tuple[PType, PType],
        input_shape: tuple[int, ...],
    ) -> dict[str, TensorDescriptor]:
        return {"residue": TensorDescriptor(input_shape, typing[0])}


@final
class ResidualEnd(PointwiseOp, PassthroughOp, ShapePreserveOp):
    begin: ResidualBegin

    def __init__(self, name: str, begin: ResidualBegin):
        super().__init__(name, DType.default, DType.default)
        self.begin = begin


# Dropout
@final
class Dropout(PointwiseOp, PassthroughOp, ShapePreserveOp):
    p: float

    def __init__(self, name: str, p: float):
        super().__init__(name, DType.infer, DType.infer)
        self.p = p


# Activation
@final
class Gelu(PointwiseOp, PassthroughOp, ShapePreserveOp):
    def training(
        self,
        typing: tuple[DType, DType],
        parallel: tuple[PType, PType],
        f: OpMan,
        x: TensorHandle,
        x_copy: TensorHandle,
    ) -> Generator[TensorHandle, TensorHandle, TensorHandle]:
        f.gelu(x, out=x_copy)
        grad = yield x_copy
        f.dgelu_(grad, x)
        return grad

    def inference(
        self,
        typing: tuple[DType, DType],
        parallel: tuple[PType, PType],
        f: OpMan,
        x: TensorHandle,
    ) -> TensorHandle:
        f.gelu_(x)
        return x

    def describe_supplementary_tensors_training(
        self,
        typing: tuple[DType, DType],
        parallel: tuple[PType, PType],
        input_shape: tuple[int, ...],
    ) -> dict[str, TensorDescriptor]:
        return {"x_copy": TensorDescriptor(input_shape, typing[0])}

    describe_supplementary_tensors_inference = returning(dict[str, TensorDescriptor]())


@final
class Relu(PointwiseOp, PassthroughOp, ShapePreserveOp):
    def training(
        self,
        typing: tuple[DType, DType],
        parallel: tuple[PType, PType],
        f: OpMan,
        x: TensorHandle,
        x_copy: TensorHandle,
    ) -> Generator[TensorHandle, TensorHandle, TensorHandle]:
        f.relu(x, out=x_copy)
        grad = yield x_copy
        f.drelu_(grad, x)
        return grad

    def inference(
        self,
        typing: tuple[DType, DType],
        parallel: tuple[PType, PType],
        f: OpMan,
        x: TensorHandle,
    ) -> TensorHandle:
        f.relu_(x)
        return x

    def describe_supplementary_tensors_training(
        self,
        typing: tuple[DType, DType],
        parallel: tuple[PType, PType],
        input_shape: tuple[int, ...],
    ) -> dict[str, TensorDescriptor]:
        return {"x_copy": TensorDescriptor(input_shape, typing[0])}

    describe_supplementary_tensors_inference = returning(dict[str, TensorDescriptor]())
