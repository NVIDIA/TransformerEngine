from __future__ import annotations
from enum import Enum


class DType(Enum):
    FP8 = "FP8"
    FP16 = "FP16"
    BF16 = "BF16"
    FP32 = "FP32"
    infer = "INFER"
    default = BF16


class Op:
    input_type: DType
    output_type: DType

    def __init__(self, input_type: DType, output_type: DType):
        self.input_type = input_type
        self.output_type = output_type


class PassthroughOp(Op):
    def __init__(self):
        super().__init__(DType.infer, DType.infer)


class ParamOp(Op):
    pass


class Gemm(ParamOp):
    in_features: int
    out_features: int

    def __init__(
        self, input_type: DType, output_type: DType, in_features: int, out_features: int
    ):
        super().__init__(input_type, output_type)
        self.in_features = in_features
        self.out_features = out_features


class Add(ParamOp):
    features: int

    def __init__(self, input_type: DType, output_type: DType, features: int):
        super().__init__(input_type, output_type)
        self.features = features


class LayerNorm(ParamOp):
    features: int
    eps: float
    zero_centered_gamma: bool

    def __init__(
        self,
        input_type: DType,
        output_type: DType,
        features: int,
        eps: float,
        zero_centered_gamma: bool,
    ):
        super().__init__(input_type, output_type)
        self.features = features
        self.eps = eps
        self.zero_centered_gamma = zero_centered_gamma


class Gelu(PassthroughOp):
    pass


class Relu(PassthroughOp):
    pass


class ResidualBegin(PassthroughOp):
    end: ResidualEnd | None = None

    def __init__(self, end: ResidualEnd | None = None):
        super().__init__()
        self.end = end


class ResidualEnd(PassthroughOp):
    begin: ResidualBegin

    def __init__(self, begin: ResidualBegin):
        super().__init__()
        self.begin = begin
