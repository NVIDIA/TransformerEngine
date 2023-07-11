from __future__ import annotations
from abc import abstractmethod
from enum import Enum


class DType(Enum):
    FP8 = "FP8"
    FP16 = "FP16"
    BF16 = "BF16"
    FP32 = "FP32"
    infer = "INFER"
    default = BF16


class Op:
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
    def describe_params(self) -> dict[str, tuple[int, ...]]:
        ...


class PassthroughOp(Op):
    def __init__(self, name: str):
        super().__init__(name, DType.infer, DType.infer)

    def describe_params(self) -> dict[str, tuple[int, ...]]:
        return {}


class ParamOp(Op):
    _params: dict[str, tuple[int, ...]]

    def __init__(
        self,
        name: str,
        input_type: DType,
        output_type: DType,
        **params: tuple[int, ...],
    ):
        super().__init__(name, input_type, output_type)
        self._params = params

    def describe_params(self) -> dict[str, tuple[int, ...]]:
        return self._params


class Gemm(ParamOp):
    in_features: int
    out_features: int

    def __init__(
        self,
        name: str,
        input_type: DType,
        output_type: DType,
        in_features: int,
        out_features: int,
    ):
        super().__init__(
            name, input_type, output_type, weight=(out_features, in_features)
        )
        self.in_features = in_features
        self.out_features = out_features


class Add(ParamOp):
    features: int

    def __init__(self, name: str, input_type: DType, output_type: DType, features: int):
        super().__init__(name, input_type, output_type, bias=(features,))
        self.features = features


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
            name, input_type, output_type, weight=(features,), bias=(features,)
        )
        self.features = features
        self.eps = eps
        self.zero_centered_gamma = zero_centered_gamma


class Gelu(PassthroughOp):
    pass


class Relu(PassthroughOp):
    pass


class ResidualBegin(PassthroughOp):
    end: ResidualEnd | None = None

    def __init__(self, name: str, end: ResidualEnd | None = None):
        super().__init__(name)
        self.end = end


class ResidualEnd(PassthroughOp):
    begin: ResidualBegin

    def __init__(self, name: str, begin: ResidualBegin):
        super().__init__(name)
        self.begin = begin
