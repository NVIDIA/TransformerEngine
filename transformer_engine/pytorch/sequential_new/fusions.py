from typing import Generic
from typing_extensions import TypeVarTuple, Unpack

from .ops import (
    Bias,
    BiasGrad,
    Gelu,
    GeluGrad,
    Gemm,
    Op,
    ResidualBegin,
    ResidualEnd,
    Dropout,
    DropoutGrad,
)


# Fused op base class
Ops = TypeVarTuple("Ops", default=Unpack[tuple[Op]])


class FusedOp(Op, Generic[Unpack[Ops]]):
    pass


# Auto fuser
class AutoFuse(FusedOp[Unpack[Ops]]):
    pass


# Manual fusions
class GemmBias(FusedOp[Gemm, Bias]):
    pass


class GemmBiasGelu(FusedOp[Gemm, Bias, Gelu]):
    pass


# Fuser
FusedOpTypes = TypeVarTuple("FusedOpTypes", default=Unpack[tuple[FusedOp]])


class Fuser(Generic[Unpack[FusedOpTypes]]):
    pass


TE_FUSER = Fuser[
    GemmBias,
    GemmBiasGelu,
    AutoFuse[GeluGrad, BiasGrad],
    AutoFuse[Bias, Dropout, ResidualEnd],
    AutoFuse[ResidualBegin, BiasGrad, DropoutGrad],
    AutoFuse[Bias, Dropout, ResidualEnd, ResidualBegin],
    AutoFuse[ResidualEnd, ResidualBegin, BiasGrad, DropoutGrad],
]()
