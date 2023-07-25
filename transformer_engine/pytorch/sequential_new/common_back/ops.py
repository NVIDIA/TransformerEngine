from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Callable, Generator, TypeVar, final

from transformer_engine.pytorch.sequential_new.common_back.enums import DType, PType
from transformer_engine.pytorch.sequential_new.common_back.generic_tensor import (
    GenericTensor,
    TensorDescriptor,
    ParamInitializer,
)
from .enums import DType, PType
from .generic_tensor import GenericTensor
from . import generic_tensor as f

T = TypeVar("T")


def returning(x: T) -> Callable[..., T]:
    def func(*args: Any, **kwargs: Any):
        del args, kwargs
        return x

    return func


# Op Protocol
class Op(ABC):
    input_type: DType
    output_type: DType
    parallellism: tuple[PType, PType]
    input_shape: tuple[int, ...]

    @abstractmethod
    def describe_parallellism(self) -> list[tuple[PType, PType]]:
        raise NotImplementedError()

    @abstractmethod
    def describe_params(self) -> dict[str, TensorDescriptor]:
        raise NotImplementedError()

    @abstractmethod
    def describe_activation_shape(self) -> tuple[int, ...]:
        raise NotImplementedError()

    @abstractmethod
    def describe_supplementary_tensors_training(self) -> dict[str, TensorDescriptor]:
        raise NotImplementedError()

    @abstractmethod
    def describe_supplementary_tensors_inference(self) -> dict[str, TensorDescriptor]:
        raise NotImplementedError()

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
    describe_params = returning(dict[str, TensorDescriptor]())


NORMAL = (PType.NA, PType.NA)
ROW_PARALLEL = (PType.NRS, PType.NRS)
COLUMN_PARALLEL = (PType.NCS, PType.NCS)


class PointwiseOp(OpBase):
    describe_parallellism = returning([NORMAL, ROW_PARALLEL, COLUMN_PARALLEL])


class RowwiseOp(OpBase):
    describe_parallellism = returning([NORMAL, ROW_PARALLEL])


class ShapePreserveOp(OpBase):
    def describe_activation_shape(self) -> tuple[int, ...]:
        return self.input_shape


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
    init_method: ParamInitializer

    def __init__(
        self,
        name: str,
        input_type: DType,
        output_type: DType,
        in_features: int,
        out_features: int,
        init_method: ParamInitializer,
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
    init_method: ParamInitializer

    def __init__(
        self,
        name: str,
        input_type: DType,
        output_type: DType,
        features: int,
        init_method: ParamInitializer,
    ):
        super().__init__(
            name,
            input_type,
            output_type,
            bias=ParamDescriptor((features,), init_method),
        )
        self.features = features
        self.init_method = init_method


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
        return {
            "fwd_residue": TensorDescriptor(self.input_shape, None, self.input_type)
        }

    def describe_supplementary_tensors_inference(self) -> dict[str, TensorDescriptor]:
        return self.describe_supplementary_tensors_training()


@final
class ResidualEnd(PointwiseOp, PassthroughOp, ShapePreserveOp):
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
        return {
            "bwd_residue": TensorDescriptor(self.input_shape, None, self.input_type)
        }

    describe_supplementary_tensors_inference = returning(dict[str, TensorDescriptor]())


# Dropout
@final
class Dropout(PointwiseOp, PassthroughOp, ShapePreserveOp):
    p: float

    def __init__(self, name: str, p: float):
        super().__init__(name, DType.infer, DType.infer)
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

    describe_supplementary_tensors_training = returning(dict[str, TensorDescriptor]())
    describe_supplementary_tensors_inference = returning(dict[str, TensorDescriptor]())


# Activation
@final
class Gelu(PointwiseOp, PassthroughOp, ShapePreserveOp):
    x_copy: GenericTensor

    def training(
        self,
        x: GenericTensor,
    ) -> Generator[GenericTensor, GenericTensor, GenericTensor]:
        f.gelu(x, out=self.x_copy)
        grad = yield self.x_copy
        f.dgelu(grad, x, out=grad)
        return grad

    def inference(
        self,
        x: GenericTensor,
    ) -> GenericTensor:
        f.gelu(x, out=x)
        return x

    def describe_supplementary_tensors_training(self) -> dict[str, TensorDescriptor]:
        return {"x_copy": TensorDescriptor(self.input_shape, None, self.input_type)}

    describe_supplementary_tensors_inference = returning(dict[str, TensorDescriptor]())


@final
class Relu(PointwiseOp, PassthroughOp, ShapePreserveOp):
    x_copy: GenericTensor

    def training(
        self,
        x: GenericTensor,
    ) -> Generator[GenericTensor, GenericTensor, GenericTensor]:
        f.relu(x, out=self.x_copy)
        grad = yield self.x_copy
        f.drelu(grad, x, out=grad)
        return grad

    def inference(
        self,
        x: GenericTensor,
    ) -> GenericTensor:
        f.relu(x, out=x)
        return x

    def describe_supplementary_tensors_training(self) -> dict[str, TensorDescriptor]:
        return {"x_copy": TensorDescriptor(self.input_shape, None, self.input_type)}

    describe_supplementary_tensors_inference = returning(dict[str, TensorDescriptor]())


@final
class Cast(PointwiseOp, ShapePreserveOp, PassthroughOp):
    cast: GenericTensor

    def training(
        self,
        x: GenericTensor,
    ) -> Generator[GenericTensor, GenericTensor, GenericTensor]:
        f.cast(x, out=self.cast)
        grad = yield self.cast
        f.cast(grad, out=x)
        return x

    def inference(
        self,
        x: GenericTensor,
    ) -> GenericTensor:
        f.cast(x, out=self.cast)
        return self.cast

    def describe_supplementary_tensors_training(self) -> dict[str, TensorDescriptor]:
        return {"cast": TensorDescriptor(self.input_shape, None, self.output_type)}

    def describe_supplementary_tensors_inference(self) -> dict[str, TensorDescriptor]:
        return self.describe_supplementary_tensors_training()
