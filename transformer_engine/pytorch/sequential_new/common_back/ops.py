from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import NoReturn
from . import framework_interface as fi
from .framework_interface import ParamConstructor
from .enums import DType


@dataclass
class ParamDescriptor:
    _shape: tuple[int, ...]
    _constructor: ParamConstructor
    _dtype: DType = DType.FP32  # TODO: is this correct?


# Base classes
class Op(ABC):
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

    @abstractmethod
    def describe_params(self) -> dict[str, ParamDescriptor]:
        raise NotImplementedError()

    @abstractmethod
    def bwd(self) -> Op:
        raise NotImplementedError()


class PassthroughOp(Op):
    def __init__(
        self,
        name: str,
        input_type: DType = DType.infer,
        output_type: DType = DType.infer,
    ):
        super().__init__(name, input_type, output_type)

    def describe_params(self) -> dict[str, ParamDescriptor]:
        return {}


class ParamOp(Op):
    _params: dict[str, ParamDescriptor]

    def __init__(
        self,
        name: str,
        input_type: DType,
        output_type: DType,
        **params: ParamDescriptor,
    ):
        super().__init__(name, input_type, output_type)
        self._params = params

    def describe_params(self) -> dict[str, ParamDescriptor]:
        return self._params


class Grad(PassthroughOp):
    def __init__(self, orig: Op):
        self.orig = orig
        super().__init__(orig.name + "_grad", *self.io_types())

    @abstractmethod
    def io_types(self) -> tuple[DType, DType]:
        raise NotImplementedError()

    def bwd(self) -> NoReturn:
        raise NotImplementedError("Second order gradient not supported")


# Normalization
class LayerNorm(ParamOp):
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

    def bwd(self):
        return LayerNormGrad(self)


class LayerNormGrad(Grad):
    def io_types(self):
        return (self.orig.input_type, self.orig.input_type)


# Linear
class Gemm(ParamOp):
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

    def bwd(self):
        return GemmGrad(self)


class GemmGrad(Grad):
    def io_types(self):
        return (self.orig.input_type, DType.default)


class Bias(ParamOp):
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

    def bwd(self):
        return BiasGrad(self)


class BiasGrad(Grad):
    def io_types(self):
        return (DType.infer, DType.default)


# Transpose
class Transpose(PassthroughOp):
    def bwd(self):
        return self


# Attention
class DotProductAttention(PassthroughOp):
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

    def bwd(self):
        return DotProductAttentionGrad(self)


class DotProductAttentionGrad(Grad):
    def io_types(self):
        return (self.orig.output_type, self.orig.input_type)


# Residual
class ResidualBegin(PassthroughOp):
    end: ResidualEnd | None = None

    def __init__(self, name: str, end: ResidualEnd | None = None):
        super().__init__(name, DType.default, DType.default)
        self.end = end

    def bwd(self):
        assert self.end is not None
        return self.end


class ResidualEnd(PassthroughOp):
    begin: ResidualBegin

    def __init__(self, name: str, begin: ResidualBegin):
        super().__init__(name, DType.default, DType.default)
        self.begin = begin

    def bwd(self):
        return self.begin


# Dropout
class Dropout(PassthroughOp):
    p: float

    def __init__(self, name: str, p: float):
        super().__init__(name, DType.infer, DType.infer)
        self.p = p

    def bwd(self):
        return DropoutGrad(self)


class DropoutGrad(Grad):
    def io_types(self):
        return (DType.infer, DType.infer)


# Activation
class Gelu(PassthroughOp):
    def bwd(self):
        return GeluGrad(self)


class GeluGrad(Grad):
    def io_types(self):
        return (DType.infer, DType.infer)


class Relu(PassthroughOp):
    def bwd(self):
        return ReluGrad(self)


class ReluGrad(Grad):
    def io_types(self):
        return (DType.infer, DType.infer)
