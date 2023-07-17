from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import NoReturn

from . import framework_interface as fi
from .framework_interface import FrameworkInterface, TensorType
from .enums import DType


@dataclass
class ParamDescriptor:
    _shape: tuple[int, ...]
    _constructor: ...  # Callable[[FrameworkInterface[TensorType]], TensorType]

    def construct(self, framework: type[FrameworkInterface[TensorType]]) -> TensorType:
        return self._constructor(framework, self._shape)


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
        ...

    @abstractmethod
    def bwd(self) -> Op:
        ...


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


class Identity(PassthroughOp):
    def bwd(self):
        return Identity(self.name + "_grad", self.input_type, self.output_type)


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


# Linear
class Gemm(ParamOp):
    in_features: int
    out_features: int
    init_method: ...

    def __init__(
        self,
        name: str,
        input_type: DType,
        output_type: DType,
        in_features: int,
        out_features: int,
        init_method: ...,
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


class Bias(ParamOp):
    features: int
    init_method: ...

    def __init__(
        self,
        name: str,
        input_type: DType,
        output_type: DType,
        features: int,
        init_method: ...,
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
        return Identity(self.name + "_grad", self.input_type, self.output_type)

    def bwd_bias(self):
        return Identity(self.name + "_grad_bias", self.input_type, self.output_type)


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
        return DotProductAttentionGrad(
            self.name + "_grad",
            self.input_type,
            self.output_type,
            self.features_per_head,
        )


class DotProductAttentionGrad(PassthroughOp):
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

    def bwd(self) -> NoReturn:
        raise NotImplementedError("Second order gradient not supported")


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


# Activation
class Gelu(PassthroughOp):
    def bwd(self):
        return GeluGrad(self.name + "_grad", self.input_type, self.output_type)


class GeluGrad(PassthroughOp):
    def bwd(self) -> NoReturn:
        raise NotImplementedError("Second order gradient not supported")


class Relu(PassthroughOp):
    def bwd(self):
        return ReluGrad(self.name + "_grad", self.input_type, self.output_type)


class ReluGrad(PassthroughOp):
    def bwd(self) -> NoReturn:
        raise NotImplementedError("Second order gradient not supported")
